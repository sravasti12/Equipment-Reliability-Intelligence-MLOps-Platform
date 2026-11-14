"""Delayed-label RUL performance monitoring with machine-level bootstrap CIs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from reliability_platform.modeling.metrics import grouped_metrics, regression_metrics

_METRIC_SUM_COLUMNS = (
    "rows",
    "squared_error",
    "absolute_error",
    "asymmetric_score",
)
_BOOTSTRAP_METRICS = ("rmse", "mae", "asymmetric_score_per_row")
_REQUIRED_COLUMNS = {
    "source",
    "machine_id",
    "actual_rul",
    "predicted_rul",
}


def _validate_predictions(
    frame: pd.DataFrame,
    *,
    windowed: bool,
    additional_prediction_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    required = set(_REQUIRED_COLUMNS)
    if windowed:
        required.add("window_index")
    required.update(additional_prediction_columns)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "delayed-label data is missing columns: " + ", ".join(sorted(missing))
        )
    if frame.empty:
        raise ValueError("delayed-label data must contain at least one row")

    data = frame.copy()
    if data[["source", "machine_id"]].isna().any().any():
        raise ValueError("source and machine_id values cannot be missing")
    data["source"] = data["source"].astype("string")
    data["machine_id"] = data["machine_id"].astype("string")
    if data[["source", "machine_id"]].eq("").any().any():
        raise ValueError("source and machine_id values cannot be empty")

    numeric_columns = ("actual_rul", "predicted_rul", *additional_prediction_columns)
    if windowed:
        numeric_columns = ("window_index", *numeric_columns)
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    numeric_values = data.loc[:, list(numeric_columns)].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("delayed-label values must be finite numeric values")
    if data["actual_rul"].lt(0).any():
        raise ValueError("actual_rul values cannot be negative")
    if windowed and (
        data["window_index"].lt(0).any()
        or not np.equal(
            data["window_index"].to_numpy(dtype=float),
            np.floor(data["window_index"].to_numpy(dtype=float)),
        ).all()
    ):
        raise ValueError("window_index must contain non-negative integers")
    identity = ["source", "machine_id"]
    if windowed:
        data["window_index"] = data["window_index"].astype("int64")
        identity.insert(1, "window_index")
    if data.duplicated(identity).any():
        raise ValueError("each source/machine/period must have one delayed label")
    return data


def _metric_sums(
    frame: pd.DataFrame,
    prediction_column: str,
) -> tuple[list[str], npt.NDArray[np.float64]]:
    actual = frame["actual_rul"].to_numpy(dtype=np.float64)
    predicted = np.maximum(
        frame[prediction_column].to_numpy(dtype=np.float64),
        0.0,
    )
    errors = predicted - actual
    exponents = np.where(errors < 0, -errors / 13.0, errors / 10.0)
    if (exponents > np.log(np.finfo(float).max)).any():
        raise ValueError("asymmetric score overflows for supplied predictions")
    rows = pd.DataFrame(
        {
            "machine_id": frame["machine_id"].to_numpy(),
            "rows": np.ones(len(frame), dtype=np.float64),
            "squared_error": np.square(errors),
            "absolute_error": np.abs(errors),
            "asymmetric_score": np.expm1(exponents),
        }
    )
    grouped = rows.groupby("machine_id", sort=True)[list(_METRIC_SUM_COLUMNS)].sum()
    return list(grouped.index), grouped.to_numpy(dtype=np.float64)


def _metrics_from_sums(
    values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    row_count = values[..., 0]
    return np.stack(
        (
            np.sqrt(values[..., 1] / row_count),
            values[..., 2] / row_count,
            values[..., 3] / row_count,
        ),
        axis=-1,
    )


def _independent_bootstrap_deltas(
    reference_sums: npt.NDArray[np.float64],
    current_sums: npt.NDArray[np.float64],
    *,
    resamples: int,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    reference_indices = rng.integers(
        0,
        len(reference_sums),
        size=(resamples, len(reference_sums)),
    )
    current_indices = rng.integers(
        0,
        len(current_sums),
        size=(resamples, len(current_sums)),
    )
    reference_bootstrap = _metrics_from_sums(reference_sums[reference_indices].sum(1))
    current_bootstrap = _metrics_from_sums(current_sums[current_indices].sum(1))
    reference_point = _metrics_from_sums(reference_sums.sum(0))
    current_point = _metrics_from_sums(current_sums.sum(0))
    return {
        metric: {
            "estimate": float(current_point[index] - reference_point[index]),
            "ci_lower": float(
                np.quantile(
                    current_bootstrap[:, index] - reference_bootstrap[:, index],
                    0.025,
                )
            ),
            "ci_upper": float(
                np.quantile(
                    current_bootstrap[:, index] - reference_bootstrap[:, index],
                    0.975,
                )
            ),
        }
        for index, metric in enumerate(_BOOTSTRAP_METRICS)
    }


def _paired_bootstrap_deltas(
    current_sums: npt.NDArray[np.float64],
    baseline_sums: npt.NDArray[np.float64],
    *,
    resamples: int,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    if current_sums.shape != baseline_sums.shape:
        raise ValueError("baseline and model metrics must cover the same machines")
    indices = rng.integers(
        0,
        len(current_sums),
        size=(resamples, len(current_sums)),
    )
    current_bootstrap = _metrics_from_sums(current_sums[indices].sum(1))
    baseline_bootstrap = _metrics_from_sums(baseline_sums[indices].sum(1))
    current_point = _metrics_from_sums(current_sums.sum(0))
    baseline_point = _metrics_from_sums(baseline_sums.sum(0))
    return {
        metric: {
            "estimate": float(current_point[index] - baseline_point[index]),
            "ci_lower": float(
                np.quantile(
                    current_bootstrap[:, index] - baseline_bootstrap[:, index],
                    0.025,
                )
            ),
            "ci_upper": float(
                np.quantile(
                    current_bootstrap[:, index] - baseline_bootstrap[:, index],
                    0.975,
                )
            ),
        }
        for index, metric in enumerate(_BOOTSTRAP_METRICS)
    }


def _model_metrics(
    frame: pd.DataFrame,
    prediction_column: str,
) -> dict[str, float | int]:
    return regression_metrics(
        frame["actual_rul"].astype(float),
        frame[prediction_column].to_numpy(dtype=np.float64),
    )


def _performance_status(
    deltas: Mapping[str, Mapping[str, float]],
    *,
    reference_machines: int,
    current_machines: int,
    minimum_machines: int,
) -> tuple[str, str]:
    if min(reference_machines, current_machines) < minimum_machines:
        return (
            "inconclusive_insufficient_machines",
            (
                "Machine-level evidence is weak; confidence intervals are "
                "descriptive and do not support a degradation conclusion."
            ),
        )
    rmse_interval = deltas["rmse"]
    if rmse_interval["ci_lower"] > 0:
        return (
            "degradation_supported",
            "The 95% RMSE-difference interval is above zero.",
        )
    if rmse_interval["ci_upper"] <= 0:
        return (
            "no_degradation_detected",
            "The 95% RMSE-difference interval is not above zero.",
        )
    return (
        "inconclusive_confidence_interval",
        "The 95% RMSE-difference interval includes zero.",
    )


def build_performance_report(
    reference: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    baseline_prediction_columns: Mapping[str, str] | None = None,
    bootstrap_resamples: int = 1000,
    random_seed: int = 42,
    minimum_machines_for_conclusion: int = 10,
) -> dict[str, object]:
    """Compare delayed-label performance by source/window using machine bootstrap."""
    if bootstrap_resamples < 2:
        raise ValueError("bootstrap_resamples must be at least 2")
    if minimum_machines_for_conclusion < 2:
        raise ValueError("minimum_machines_for_conclusion must be at least 2")
    baselines = dict(baseline_prediction_columns or {})
    if any(not name or not column for name, column in baselines.items()):
        raise ValueError("baseline model names and prediction columns cannot be empty")
    baseline_columns = tuple(baselines.values())
    if len(baseline_columns) != len(set(baseline_columns)):
        raise ValueError("baseline prediction columns must be unique")

    reference_data = _validate_predictions(reference, windowed=False)
    current_data = _validate_predictions(
        windows,
        windowed=True,
        additional_prediction_columns=baseline_columns,
    )
    reference_sources = set(reference_data["source"].astype(str))
    current_sources = set(current_data["source"].astype(str))
    if not current_sources.issubset(reference_sources):
        raise ValueError("every monitored source must have a reference period")

    window_reports: list[dict[str, object]] = []
    rng = np.random.default_rng(random_seed)
    for source, source_windows in current_data.groupby("source", sort=True):
        reference_source = reference_data.loc[reference_data["source"] == source]
        reference_metrics = _model_metrics(reference_source, "predicted_rul")
        reference_machine_ids, reference_sums = _metric_sums(
            reference_source,
            "predicted_rul",
        )
        for window_index, current in source_windows.groupby("window_index", sort=True):
            current_metrics = _model_metrics(current, "predicted_rul")
            current_machine_ids, current_sums = _metric_sums(
                current,
                "predicted_rul",
            )
            metric_deltas = _independent_bootstrap_deltas(
                reference_sums,
                current_sums,
                resamples=bootstrap_resamples,
                rng=rng,
            )
            status, evidence_note = _performance_status(
                metric_deltas,
                reference_machines=len(reference_machine_ids),
                current_machines=len(current_machine_ids),
                minimum_machines=minimum_machines_for_conclusion,
            )
            metadata_columns = [
                column
                for column in (
                    "machine_type",
                    "operating_condition",
                    "setting_1",
                    "setting_2",
                    "setting_3",
                )
                if column in current.columns
            ]
            grouped = grouped_metrics(
                current["actual_rul"].astype(float),
                current["predicted_rul"].to_numpy(dtype=np.float64),
                current.loc[:, metadata_columns],
                training_metadata=reference_source,
            )

            baseline_reports: list[dict[str, object]] = []
            for baseline_name, prediction_column in sorted(baselines.items()):
                baseline_metrics = _model_metrics(current, prediction_column)
                baseline_machine_ids, baseline_sums = _metric_sums(
                    current,
                    prediction_column,
                )
                if current_machine_ids != baseline_machine_ids:
                    raise RuntimeError(
                        f"baseline {baseline_name!r} does not align by machine"
                    )
                baseline_reports.append(
                    {
                        "model": baseline_name,
                        "metrics": baseline_metrics,
                        "metric_deltas": _paired_bootstrap_deltas(
                            current_sums,
                            baseline_sums,
                            resamples=bootstrap_resamples,
                            rng=rng,
                        ),
                    }
                )

            window_reports.append(
                {
                    "source": str(source),
                    "window_index": int(cast(int, window_index)),
                    "machines": len(current_machine_ids),
                    "reference_machines": len(reference_machine_ids),
                    "metrics": current_metrics,
                    "reference_metrics": reference_metrics,
                    "metric_deltas": metric_deltas,
                    "degradation_status": status,
                    "evidence_note": evidence_note,
                    "by_group": grouped,
                    "baseline_comparisons": baseline_reports,
                }
            )

    return {
        "schema_version": 1,
        "random_seed": random_seed,
        "bootstrap_resamples": bootstrap_resamples,
        "minimum_machines_for_conclusion": minimum_machines_for_conclusion,
        "confidence_level": 0.95,
        "degradation_metric": (
            "RMSE difference: monitored window minus reference period"
        ),
        "windows": window_reports,
    }
