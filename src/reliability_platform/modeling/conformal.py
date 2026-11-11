"""Machine-level split-conformal intervals for RUL trajectories."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil

import numpy as np
import numpy.typing as npt
import pandas as pd


def _machine_columns(metadata: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in ("source", "dataset_split", "machine_id")
        if column in metadata
    ]
    if "machine_id" not in columns:
        raise ValueError("metadata must include machine_id")
    return columns


def calibrate_trajectory_error(
    actual: pd.Series[float],
    predicted: npt.NDArray[np.float64],
    metadata: pd.DataFrame,
    *,
    alpha: float,
) -> tuple[float, int]:
    """Calibrate a simultaneous trajectory error bound from held-out machines.

    The nonconformity score is each calibration machine's maximum absolute
    cycle-level residual. This treats machines, rather than correlated cycle
    rows, as the exchangeable calibration units.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    values = actual.to_numpy(dtype=np.float64)
    estimates = np.asarray(predicted, dtype=np.float64)
    if (
        len(values) == 0
        or values.shape != estimates.shape
        or len(metadata) != len(values)
        or not np.isfinite(values).all()
        or not np.isfinite(estimates).all()
    ):
        raise ValueError("calibration inputs must be aligned finite vectors")
    keys = _machine_columns(metadata)
    residuals = metadata.loc[:, keys].copy()
    residuals["error"] = np.abs(values - estimates)
    scores = (
        residuals.groupby(keys, sort=False, dropna=False)["error"]
        .max()
        .to_numpy(dtype=np.float64)
    )
    rank = ceil((len(scores) + 1) * (1 - alpha))
    if rank > len(scores):
        raise ValueError(
            "not enough calibration machines for the requested conformal coverage"
        )
    return float(np.sort(scores)[rank - 1]), len(scores)


def prediction_intervals(
    predicted: npt.NDArray[np.float64],
    *,
    error_bound: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return non-negative RUL intervals clipped at the physical lower bound."""
    estimates = np.asarray(predicted, dtype=np.float64)
    if not np.isfinite(estimates).all() or error_bound < 0:
        raise ValueError("predictions must be finite and error_bound non-negative")
    lower = np.maximum(0.0, estimates - error_bound)
    upper = estimates + error_bound
    return lower, upper


def interval_metrics(
    actual: pd.Series[float],
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
    metadata: pd.DataFrame,
    *,
    target_coverage: float,
    training_metadata: pd.DataFrame | None = None,
    operating_contexts: Sequence[str] = ("operating_condition",),
) -> dict[str, object]:
    """Summarize row coverage, simultaneous trajectory coverage, and width."""
    values = actual.to_numpy(dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if (
        values.shape != low.shape
        or values.shape != high.shape
        or len(metadata) != len(values)
        or (low > high).any()
    ):
        raise ValueError("intervals, targets, and metadata must be aligned")
    covered = (values >= low) & (values <= high)
    widths = high - low
    keys = _machine_columns(metadata)
    machine_table = metadata.loc[:, keys].copy()
    machine_table["covered"] = covered
    machine_coverage = machine_table.groupby(keys, dropna=False, sort=False)[
        "covered"
    ].all()
    result: dict[str, object] = {
        "target_coverage": target_coverage,
        "observed_row_coverage": float(covered.mean()),
        "observed_trajectory_coverage": float(machine_coverage.mean()),
        "calibration_unit": "machine trajectory; maximum absolute cycle residual",
        "average_interval_width_cycles": float(widths.mean()),
        "rows": int(len(values)),
        "machines": int(len(machine_coverage)),
    }

    if "machine_type" in metadata:
        result["machine_type"] = _machine_group_metrics(
            covered, widths, metadata, "machine_type", keys
        )
    contexts: dict[str, object] = {}
    for column in operating_contexts:
        if column in metadata:
            contexts[column] = _row_group_metrics(covered, widths, metadata, column)
    if training_metadata is not None:
        for column in ("setting_1", "setting_2", "setting_3"):
            if column not in metadata or column not in training_metadata:
                continue
            training_values = pd.to_numeric(
                training_metadata[column], errors="coerce"
            ).dropna()
            if training_values.empty:
                continue
            edges = np.unique(
                np.quantile(training_values.to_numpy(dtype=float), [0.25, 0.5, 0.75])
            )
            if edges.size == 0:
                continue
            bins: list[float] = [
                -np.inf,
                *(float(edge) for edge in edges),
                np.inf,
            ]
            labels = pd.cut(
                pd.to_numeric(metadata[column], errors="coerce"),
                bins=bins,
                include_lowest=True,
            ).astype("string")
            setting_data = metadata.copy()
            setting_data[column] = labels
            contexts[column] = {
                "training_bin_edges": [float(edge) for edge in edges],
                "metrics": _row_group_metrics(covered, widths, setting_data, column),
            }
    if contexts:
        result["operating_conditions"] = contexts
    return result


def _machine_group_metrics(
    covered: npt.NDArray[np.bool_],
    widths: npt.NDArray[np.float64],
    metadata: pd.DataFrame,
    group_column: str,
    machine_columns: list[str],
) -> list[dict[str, object]]:
    data = metadata.loc[:, [*machine_columns, group_column]].copy()
    data["covered"] = covered
    data["width"] = widths
    machine_rows = data.groupby(machine_columns, sort=False, dropna=False).agg(
        group=(group_column, "first"),
        trajectory_covered=("covered", "all"),
        average_width=("width", "mean"),
    )
    output: list[dict[str, object]] = []
    for group, values in machine_rows.groupby("group", dropna=False, sort=False):
        output.append(
            {
                "group": str(group),
                "machines": int(len(values)),
                "trajectory_coverage": float(values["trajectory_covered"].mean()),
                "average_interval_width_cycles": float(values["average_width"].mean()),
            }
        )
    return output


def _row_group_metrics(
    covered: npt.NDArray[np.bool_],
    widths: npt.NDArray[np.float64],
    metadata: pd.DataFrame,
    group_column: str,
) -> list[dict[str, object]]:
    data = pd.DataFrame(
        {
            "group": metadata[group_column].astype("string"),
            "covered": covered,
            "width": widths,
        }
    )
    output: list[dict[str, object]] = []
    for group, values in data.groupby("group", dropna=True, sort=False):
        output.append(
            {
                "group": str(group),
                "rows": int(len(values)),
                "row_coverage": float(values["covered"].mean()),
                "average_interval_width_cycles": float(values["width"].mean()),
            }
        )
    return output
