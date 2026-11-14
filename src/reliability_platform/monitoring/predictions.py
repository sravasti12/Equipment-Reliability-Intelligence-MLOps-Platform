"""Summaries of saved, windowed machine-level model predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_REQUIRED_COLUMNS = {
    "source",
    "window_index",
    "machine_id",
    "predicted_rul",
    "risk_score",
    "lower_bound",
    "upper_bound",
}


def _distribution(values: pd.Series[float]) -> dict[str, float | int]:
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p10": float(values.quantile(0.1)),
        "p90": float(values.quantile(0.9)),
    }


def build_prediction_report(predictions: pd.DataFrame) -> dict[str, object]:
    """Summarize API/model predictions without fitting or changing the model."""
    missing_columns = _REQUIRED_COLUMNS.difference(predictions.columns)
    if missing_columns:
        raise ValueError(
            "prediction data is missing columns: " + ", ".join(sorted(missing_columns))
        )
    if predictions.empty:
        raise ValueError("prediction data must contain at least one row")

    data = predictions.loc[:, sorted(_REQUIRED_COLUMNS)].copy()
    if data[["source", "machine_id"]].isna().any().any():
        raise ValueError("source and machine_id values cannot be missing")
    data["source"] = data["source"].astype("string")
    data["machine_id"] = data["machine_id"].astype("string")
    if data[["source", "machine_id"]].eq("").any().any():
        raise ValueError("source and machine_id values cannot be empty")

    numeric_columns = (
        "window_index",
        "predicted_rul",
        "risk_score",
        "lower_bound",
        "upper_bound",
    )
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    numeric_values = data.loc[:, list(numeric_columns)].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("prediction values must be finite numeric values")
    if (
        data["window_index"].lt(0).any()
        or not np.equal(
            data["window_index"].to_numpy(dtype=float),
            np.floor(data["window_index"].to_numpy(dtype=float)),
        ).all()
    ):
        raise ValueError("window_index must contain non-negative integers")
    if (
        data["predicted_rul"].lt(0).any()
        or data["risk_score"].lt(0).any()
        or data["risk_score"].gt(1).any()
        or data["lower_bound"].lt(0).any()
        or data["lower_bound"].gt(data["predicted_rul"]).any()
        or data["predicted_rul"].gt(data["upper_bound"]).any()
    ):
        raise ValueError("predictions contain invalid RUL, risk, or interval values")
    if data.duplicated(["source", "window_index", "machine_id"]).any():
        raise ValueError("each source/window/machine combination must be unique")

    data["window_index"] = data["window_index"].astype("int64")
    data["interval_width_cycles"] = data["upper_bound"] - data["lower_bound"]
    window_summaries: list[dict[str, object]] = []
    for (source, window_index), group in data.groupby(
        ["source", "window_index"], sort=True
    ):
        window_summaries.append(
            {
                "source": str(source),
                "window_index": int(window_index),
                "machines": int(group["machine_id"].nunique()),
                "predicted_rul_cycles": _distribution(group["predicted_rul"]),
                "risk_score": _distribution(group["risk_score"]),
                "interval_width_cycles": _distribution(group["interval_width_cycles"]),
            }
        )

    trend_columns = {
        "predicted_rul_cycles": "predicted_rul",
        "risk_score": "risk_score",
        "interval_width_cycles": "interval_width_cycles",
    }
    machine_trends: list[dict[str, object]] = []
    ordered = data.sort_values(
        ["source", "machine_id", "window_index"],
        kind="stable",
    )
    for (source, machine_id), group in ordered.groupby(
        ["source", "machine_id"], sort=True
    ):
        previous: pd.Series[Any] | None = None
        for _, row in group.iterrows():
            deltas = {
                f"{column}_change": (
                    None
                    if previous is None
                    else float(row[source_column] - previous[source_column])
                )
                for column, source_column in trend_columns.items()
            }
            machine_trends.append(
                {
                    "source": str(source),
                    "machine_id": str(machine_id),
                    "window_index": int(row["window_index"]),
                    "previous_window_index": (
                        None if previous is None else int(previous["window_index"])
                    ),
                    "predicted_rul_cycles": float(row["predicted_rul"]),
                    "risk_score": float(row["risk_score"]),
                    "interval_width_cycles": float(row["interval_width_cycles"]),
                    **deltas,
                }
            )
            previous = row

    return {
        "schema_version": 1,
        "sources": sorted(str(source) for source in data["source"].unique()),
        "windows": window_summaries,
        "machine_trends": machine_trends,
    }
