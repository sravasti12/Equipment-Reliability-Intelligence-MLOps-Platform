"""Deterministic, capacity-constrained maintenance prioritization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def maintenance_risk(
    predicted_rul: float,
    conformal_lower_bound: float,
    *,
    urgency_horizon_cycles: int,
) -> float:
    """Map a conservative RUL estimate to a bounded, monotone urgency score."""
    if urgency_horizon_cycles < 1:
        raise ValueError("urgency_horizon_cycles must be positive")
    if (
        not np.isfinite(predicted_rul)
        or not np.isfinite(conformal_lower_bound)
        or predicted_rul < 0
        or conformal_lower_bound < 0
        or conformal_lower_bound > predicted_rul
    ):
        raise ValueError("RUL estimates must be finite and satisfy 0 <= lower <= RUL")
    return float(
        np.clip(1.0 - conformal_lower_bound / urgency_horizon_cycles, 0.0, 1.0)
    )


def schedule_machines(
    predictions: pd.DataFrame,
    *,
    crew_capacity_per_shift: int,
    risk_threshold: float,
) -> pd.DataFrame:
    """Rank current machine predictions and flag the eligible top-capacity set.

    Machine identity columns are included in deterministic tie breaking. Only
    rows at or above the risk threshold can be scheduled.
    """
    if crew_capacity_per_shift < 1:
        raise ValueError("crew_capacity_per_shift must be positive")
    if not np.isfinite(risk_threshold) or not 0 <= risk_threshold <= 1:
        raise ValueError("risk_threshold must be between 0 and 1")
    identity = [
        column
        for column in ("source", "dataset_split", "machine_id")
        if column in predictions.columns
    ]
    required = {
        "machine_id",
        "predicted_rul",
        "conformal_lower_bound",
        "risk_score",
    }
    if missing := required.difference(predictions.columns):
        raise ValueError(
            f"schedule data is missing columns: {', '.join(sorted(missing))}"
        )
    if predictions.empty:
        raise ValueError("schedule data must contain at least one machine")
    if predictions[identity].isna().any().any():
        raise ValueError("machine identity columns cannot be missing")
    if predictions.duplicated(identity).any():
        raise ValueError("schedule data must contain one row per machine")

    result = predictions.copy()
    numeric_columns = ("predicted_rul", "conformal_lower_bound", "risk_score")
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    numeric_values = result.loc[:, list(numeric_columns)].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("schedule predictions must contain finite values")
    if (
        result["predicted_rul"].lt(0).any()
        or result["conformal_lower_bound"].lt(0).any()
        or result["conformal_lower_bound"].gt(result["predicted_rul"]).any()
        or result["risk_score"].lt(0).any()
        or result["risk_score"].gt(1).any()
    ):
        raise ValueError("schedule predictions contain out-of-range values")

    sort_columns = ["risk_score", "conformal_lower_bound"]
    result = result.assign(
        **{
            f"__identity_{index}": result[column].astype("string")
            for index, column in enumerate(identity)
        }
    )
    identity_sort_columns = [f"__identity_{index}" for index in range(len(identity))]
    result = result.sort_values(
        [*sort_columns, *identity_sort_columns],
        ascending=[False, True, *([True] * len(identity_sort_columns))],
        kind="stable",
    ).reset_index(drop=True)
    result["priority_rank"] = np.arange(1, len(result) + 1, dtype=np.int64)
    eligible = result["risk_score"].ge(risk_threshold)
    eligible_rank = eligible.cumsum()
    result["scheduled"] = eligible & eligible_rank.le(crew_capacity_per_shift)
    result["deferred"] = eligible & ~result["scheduled"]
    result["below_risk_threshold"] = ~eligible
    return result.drop(columns=identity_sort_columns)
