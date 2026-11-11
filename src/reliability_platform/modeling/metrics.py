"""RUL prediction metrics and operating-context summaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd


def regression_metrics(
    y_true: pd.Series[float],
    y_pred: npt.NDArray[np.float64],
) -> dict[str, float | int]:
    """Calculate RMSE, MAE, and the asymmetric NASA C-MAPSS score."""
    actual = y_true.to_numpy(dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    if (
        actual.ndim != 1
        or predicted.ndim != 1
        or actual.size == 0
        or actual.shape != predicted.shape
        or not np.isfinite(actual).all()
        or not np.isfinite(predicted).all()
        or (actual < 0).any()
    ):
        raise ValueError("RUL targets and predictions must be aligned finite vectors")
    predicted = np.maximum(predicted, 0.0)
    errors = predicted - actual
    exponents = np.where(errors < 0, -errors / 13.0, errors / 10.0)
    if (exponents > np.log(np.finfo(float).max)).any():
        raise ValueError("asymmetric score overflows for the supplied predictions")
    score = float(np.expm1(exponents).sum())
    return {
        "samples": int(actual.size),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
        "asymmetric_score": score,
    }


def grouped_metrics(
    y_true: pd.Series[float],
    y_pred: npt.NDArray[np.float64],
    metadata: pd.DataFrame,
    *,
    training_metadata: pd.DataFrame,
) -> dict[str, object]:
    """Summarize test errors by machine group and train-defined conditions."""
    result: dict[str, object] = {}
    if "machine_type" in metadata.columns:
        result["machine_type"] = _metrics_by_group(
            y_true, y_pred, metadata["machine_type"].astype("string")
        )
    conditions: dict[str, object] = {}
    if "operating_condition" in metadata.columns:
        conditions["operating_condition"] = _metrics_by_group(
            y_true, y_pred, metadata["operating_condition"].astype("string")
        )
    for column in ("setting_1", "setting_2", "setting_3"):
        if column not in metadata or column not in training_metadata:
            continue
        training_series = pd.to_numeric(
            training_metadata[column], errors="coerce"
        ).dropna()
        training_values = training_series.to_numpy(dtype=np.float64)
        if training_values.size == 0 or np.unique(training_values).size < 2:
            continue
        quantiles = np.quantile(training_values, [0.25, 0.5, 0.75])
        interior_edges = np.unique(quantiles)
        if interior_edges.size == 0:
            continue
        bins: list[float] = [
            -np.inf,
            *(float(value) for value in interior_edges),
            np.inf,
        ]
        if len(bins) < 3:
            continue
        labels = pd.cut(
            pd.to_numeric(metadata[column], errors="coerce"),
            bins=bins,
            include_lowest=True,
        ).astype("string")
        conditions[column] = {
            "training_bin_edges": [float(value) for value in interior_edges],
            "metrics": _metrics_by_group(y_true, y_pred, labels),
        }
    result["operating_conditions"] = conditions
    return result


def _metrics_by_group(
    y_true: pd.Series[float],
    y_pred: npt.NDArray[np.float64],
    groups: pd.Series[Any],
) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for group in groups.dropna().unique():
        mask = groups.eq(group).to_numpy()
        metrics = regression_metrics(y_true.iloc[np.flatnonzero(mask)], y_pred[mask])
        rows.append({"group": str(group), **metrics})
    return rows
