"""Delayed-label conformal coverage and interval-width monitoring."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from reliability_platform.modeling.conformal import interval_metrics

_REQUIRED_COLUMNS = {
    "source",
    "window_index",
    "machine_id",
    "cycle",
    "actual_rul",
    "lower_bound",
    "upper_bound",
}


def _validate_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    missing = _REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            "calibration data is missing columns: " + ", ".join(sorted(missing))
        )
    if frame.empty:
        raise ValueError("calibration data must contain at least one row")

    data = frame.copy()
    if data[["source", "machine_id"]].isna().any().any():
        raise ValueError("source and machine_id values cannot be missing")
    data["source"] = data["source"].astype("string")
    data["machine_id"] = data["machine_id"].astype("string")
    if data[["source", "machine_id"]].eq("").any().any():
        raise ValueError("source and machine_id values cannot be empty")

    for column in ("window_index", "cycle", "actual_rul", "lower_bound", "upper_bound"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    numeric_values = data.loc[
        :, ["window_index", "cycle", "actual_rul", "lower_bound", "upper_bound"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("calibration values must be finite numeric values")
    for column in ("window_index", "cycle"):
        values = data[column].to_numpy(dtype=float)
        if (values < 0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{column} must contain non-negative integers")
    if data["cycle"].eq(0).any():
        raise ValueError("cycle must contain positive integers")
    if (
        data["actual_rul"].lt(0).any()
        or data["lower_bound"].lt(0).any()
        or data["upper_bound"].lt(data["lower_bound"]).any()
    ):
        raise ValueError("calibration values contain invalid RUL intervals")
    if data.duplicated(["source", "window_index", "machine_id", "cycle"]).any():
        raise ValueError("source/window/machine/cycle rows must be unique")
    data["window_index"] = data["window_index"].astype("int64")
    data["cycle"] = data["cycle"].astype("int64")
    return data


def _bootstrap_intervals(
    machine_summaries: npt.NDArray[np.float64],
    *,
    resamples: int,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    indices = rng.integers(
        0,
        len(machine_summaries),
        size=(resamples, len(machine_summaries)),
    )
    totals = machine_summaries[indices].sum(axis=1)
    row_count = totals[:, 0]
    machine_count = len(machine_summaries)
    values = {
        "row_coverage": totals[:, 1] / row_count,
        "trajectory_coverage": totals[:, 2] / machine_count,
        "average_interval_width_cycles": totals[:, 3] / row_count,
    }
    return {
        name: {
            "ci_lower": float(np.quantile(samples, 0.025)),
            "ci_upper": float(np.quantile(samples, 0.975)),
        }
        for name, samples in values.items()
    }


def _coverage_status(
    confidence_interval: dict[str, float],
    *,
    target_coverage: float,
    machines: int,
    minimum_machines: int,
) -> tuple[str, str]:
    if machines < minimum_machines:
        return (
            "inconclusive_insufficient_machines",
            (
                "Machine-level evidence is weak; confidence intervals are "
                "descriptive and do not support a coverage conclusion."
            ),
        )
    if confidence_interval["ci_upper"] < target_coverage:
        return (
            "undercoverage_supported",
            "The 95% trajectory-coverage interval is below target coverage.",
        )
    if confidence_interval["ci_lower"] >= target_coverage:
        return (
            "coverage_at_or_above_target",
            "The 95% trajectory-coverage interval is at or above target coverage.",
        )
    return (
        "inconclusive_confidence_interval",
        "The 95% trajectory-coverage interval overlaps target coverage.",
    )


def build_calibration_report(
    intervals: pd.DataFrame,
    *,
    target_coverage: float,
    bootstrap_resamples: int = 1000,
    random_seed: int = 42,
    minimum_machines_for_conclusion: int = 10,
) -> dict[str, object]:
    """Report observed coverage and width with machine-cluster confidence intervals."""
    if not 0 < target_coverage < 1:
        raise ValueError("target_coverage must be between 0 and 1")
    if bootstrap_resamples < 2:
        raise ValueError("bootstrap_resamples must be at least 2")
    if minimum_machines_for_conclusion < 2:
        raise ValueError("minimum_machines_for_conclusion must be at least 2")

    data = _validate_intervals(intervals)
    rng = np.random.default_rng(random_seed)
    windows: list[dict[str, object]] = []
    for (source, window_index), group in data.groupby(
        ["source", "window_index"], sort=True
    ):
        metadata_columns = [
            column
            for column in (
                "source",
                "machine_id",
                "cycle",
                "machine_type",
                "operating_condition",
            )
            if column in group.columns
        ]
        observed = interval_metrics(
            group["actual_rul"].astype(float),
            group["lower_bound"].to_numpy(dtype=np.float64),
            group["upper_bound"].to_numpy(dtype=np.float64),
            group.loc[:, metadata_columns],
            target_coverage=target_coverage,
        )
        machine_data = group.assign(
            covered=group["actual_rul"].between(
                group["lower_bound"],
                group["upper_bound"],
                inclusive="both",
            ),
            interval_width=group["upper_bound"] - group["lower_bound"],
        )
        machine_summaries = (
            machine_data.groupby("machine_id", sort=True)
            .agg(
                rows=("covered", "size"),
                covered_rows=("covered", "sum"),
                trajectory_covered=("covered", "all"),
                interval_width_sum=("interval_width", "sum"),
            )
            .to_numpy(dtype=np.float64)
        )
        confidence_intervals = _bootstrap_intervals(
            machine_summaries,
            resamples=bootstrap_resamples,
            rng=rng,
        )
        status, evidence_note = _coverage_status(
            confidence_intervals["trajectory_coverage"],
            target_coverage=target_coverage,
            machines=len(machine_summaries),
            minimum_machines=minimum_machines_for_conclusion,
        )
        windows.append(
            {
                "source": str(source),
                "window_index": int(window_index),
                "machines": int(len(machine_summaries)),
                "metrics": observed,
                "confidence_intervals_95": confidence_intervals,
                "coverage_status": status,
                "evidence_note": evidence_note,
            }
        )

    return {
        "schema_version": 1,
        "target_coverage": target_coverage,
        "random_seed": random_seed,
        "bootstrap_resamples": bootstrap_resamples,
        "minimum_machines_for_conclusion": minimum_machines_for_conclusion,
        "windows": windows,
    }
