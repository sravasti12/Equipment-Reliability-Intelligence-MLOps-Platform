"""Reliability metrics with explicit event and exposure requirements."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _validate_grouping(frame: pd.DataFrame, group_by: Sequence[str]) -> None:
    missing_columns = set(group_by).difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"unknown grouping columns: {', '.join(sorted(missing_columns))}"
        )


def _validate_cycle_events(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"machine_id", "cycle", "failure_event"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "failure metrics require columns: " + ", ".join(sorted(missing))
        )
    if frame.empty:
        raise ValueError("dataset must contain at least one row")
    if frame[["machine_id", "cycle", "failure_event"]].isna().any().any():
        raise ValueError("machine_id, cycle, and failure_event cannot be missing")
    cycles = pd.to_numeric(frame["cycle"], errors="coerce")
    if (
        cycles.isna().any()
        or not np.isfinite(cycles.to_numpy(dtype=float)).all()
        or cycles.le(0).any()
        or not np.equal(
            cycles.to_numpy(dtype=float), np.floor(cycles.to_numpy(dtype=float))
        ).all()
    ):
        raise ValueError("cycle must contain positive finite integer values")
    if frame.duplicated(["machine_id", "cycle"]).any():
        raise ValueError("duplicate machine_id/cycle rows found")
    if not frame["failure_event"].isin([True, False, 0, 1]).all():
        raise ValueError("failure_event must contain only boolean values")
    data = frame.assign(cycle=cycles, failure_event=frame["failure_event"].astype(bool))
    cycle_changes = data.groupby("machine_id", sort=False)["cycle"].diff()
    if cycle_changes.le(0).any():
        raise ValueError("cycles must be strictly increasing within each machine")
    return data


def machine_lifetimes(frame: pd.DataFrame) -> pd.DataFrame:
    """Return observed exposure and event counts for each machine.

    Exposure is the number of observed cycle records, not an extrapolation to
    unobserved lifetime. Machines without an observed failure are censored.
    """
    required = {"machine_id", "cycle"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"dataset is missing columns: {', '.join(sorted(missing))}")
    if frame.empty or frame["machine_id"].isna().any():
        raise ValueError("dataset must contain non-empty machine identifiers")

    data = frame.copy()
    if "failure_event" in data.columns:
        if not data["failure_event"].isin([True, False, 0, 1]).all():
            raise ValueError("failure_event must contain only boolean values")
        data["failure_event"] = data["failure_event"].astype(bool)
    else:
        data["failure_event"] = False

    if "maintenance_event" in data.columns:
        if not data["maintenance_event"].isin([True, False, 0, 1]).all():
            raise ValueError("maintenance_event must contain only boolean values")
        data["maintenance_event"] = data["maintenance_event"].astype(bool)
    else:
        data["maintenance_event"] = False

    data = _validate_cycle_events(data)

    grouping_columns: list[str] = []
    for column in ("source", "dataset_split", "machine_type"):
        if column in data.columns:
            consistent = data.groupby("machine_id", sort=False)[column].nunique(
                dropna=False
            )
            if consistent.gt(1).any():
                raise ValueError(f"{column} must be constant within each machine")
            grouping_columns.append(column)

    summary = data.groupby("machine_id", sort=True, as_index=False).agg(
        exposure_cycles=("cycle", "size"),
        last_observed_cycle=("cycle", "max"),
        failure_events=("failure_event", "sum"),
        maintenance_events=("maintenance_event", "sum"),
    )
    summary["failure_observed"] = summary["failure_events"].gt(0)
    for column in grouping_columns:
        first_value = data.groupby("machine_id", sort=True)[column].first()
        summary[column] = summary["machine_id"].map(first_value)
    return summary


def summarize_lifetimes(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed cycles for run-to-failure and censored machines."""
    summary = machine_lifetimes(frame)
    grouping = [
        column
        for column in ("source", "dataset_split", "machine_type")
        if column in summary.columns
    ]
    if grouping:
        result = summary.groupby(grouping, dropna=False)["exposure_cycles"].agg(
            machine_count="count",
            mean_cycles="mean",
            median_cycles="median",
            minimum_cycles="min",
            maximum_cycles="max",
        )
        event_counts = (
            summary.groupby(grouping, dropna=False)["failure_observed"]
            .agg(failures="sum")
            .astype("int64")
        )
        result = result.join(event_counts)
        result["censored_machines"] = result["machine_count"] - result["failures"]
        return result.reset_index()

    return pd.DataFrame(
        [
            {
                "machine_count": len(summary),
                "mean_cycles": summary["exposure_cycles"].mean(),
                "median_cycles": summary["exposure_cycles"].median(),
                "minimum_cycles": summary["exposure_cycles"].min(),
                "maximum_cycles": summary["exposure_cycles"].max(),
                "failures": int(summary["failure_observed"].sum()),
                "censored_machines": int((~summary["failure_observed"]).sum()),
            }
        ]
    )


def calculate_failure_rates(
    frame: pd.DataFrame,
    *,
    group_by: Sequence[str] = (),
) -> pd.DataFrame:
    """Calculate observed failure events per 1,000 observed machine cycles.

    The rate is an exposure-based descriptive estimate. Censored observations
    contribute their observed cycles to exposure and do not count as failures.
    """
    data = _validate_cycle_events(frame)
    _validate_grouping(data, group_by)
    grouping = list(group_by)
    data["_exposure_cycles"] = 1
    data["_failure_count"] = data["failure_event"].astype("int64")

    if grouping:
        result = data.groupby(grouping, dropna=False, as_index=False).agg(
            failures=("_failure_count", "sum"),
            exposure_cycles=("_exposure_cycles", "sum"),
            machines=("machine_id", "nunique"),
        )
    else:
        result = pd.DataFrame(
            [
                {
                    "failures": int(data["_failure_count"].sum()),
                    "exposure_cycles": len(data),
                    "machines": data["machine_id"].nunique(),
                }
            ]
        )
    result["failure_rate_per_1000_cycles"] = (
        result["failures"] / result["exposure_cycles"] * 1000.0
    )
    return result


def calculate_maintenance_frequency(
    frame: pd.DataFrame,
    *,
    group_by: Sequence[str] = (),
) -> pd.DataFrame:
    """Calculate observed maintenance events per 1,000 observed machine cycles."""
    if "maintenance_event" not in frame.columns:
        raise ValueError("maintenance frequency requires maintenance_event")
    data = _validate_cycle_events(
        frame.assign(failure_event=frame.get("failure_event", False))
    )
    if not frame["maintenance_event"].isin([True, False, 0, 1]).all():
        raise ValueError("maintenance_event must contain only boolean values")
    _validate_grouping(data, group_by)
    data["_exposure_cycles"] = 1
    data["_maintenance_count"] = frame["maintenance_event"].astype("int64")
    grouping = list(group_by)

    if grouping:
        result = data.groupby(grouping, dropna=False, as_index=False).agg(
            maintenance_events=("_maintenance_count", "sum"),
            exposure_cycles=("_exposure_cycles", "sum"),
            machines=("machine_id", "nunique"),
        )
    else:
        result = pd.DataFrame(
            [
                {
                    "maintenance_events": int(data["_maintenance_count"].sum()),
                    "exposure_cycles": len(data),
                    "machines": data["machine_id"].nunique(),
                }
            ]
        )
    result["events_per_1000_cycles"] = (
        result["maintenance_events"] / result["exposure_cycles"] * 1000.0
    )
    return result


def calculate_mtbf(frame: pd.DataFrame) -> dict[str, float | int | None | str]:
    """Estimate MTBF only when at least one machine has recurrent failures.

    The returned interval gaps are computed from cycle differences between
    observed failure events on the same machine. A single terminal failure per
    run-to-failure trajectory is insufficient to estimate this quantity.
    """
    data = _validate_cycle_events(frame)
    events = data.loc[data["failure_event"], ["machine_id", "cycle"]].sort_values(
        ["machine_id", "cycle"]
    )
    intervals = events.groupby("machine_id", sort=False)["cycle"].diff().dropna()
    if intervals.empty:
        return {
            "available": False,
            "mtbf_cycles": None,
            "failure_intervals": 0,
            "reason": "No machine has two or more observed failure events.",
        }
    return {
        "available": True,
        "mtbf_cycles": float(intervals.mean()),
        "failure_intervals": int(intervals.size),
        "reason": "Mean observed cycle gap between recurrent failures.",
    }


def calculate_downtime(frame: pd.DataFrame) -> dict[str, float | int | None | str]:
    """Summarize explicit downtime durations or report that they are unavailable."""
    if "downtime_hours" not in frame.columns:
        return {
            "available": False,
            "downtime_hours": None,
            "event_count": 0,
            "reason": "Dataset does not contain measured downtime_hours.",
        }
    durations = pd.to_numeric(frame["downtime_hours"], errors="coerce")
    if (
        durations.isna().any()
        or not np.isfinite(durations.to_numpy(dtype=float)).all()
        or durations.lt(0).any()
    ):
        raise ValueError("downtime_hours must contain finite non-negative values")
    return {
        "available": True,
        "downtime_hours": float(durations.sum()),
        "event_count": int(durations.gt(0).sum()),
        "reason": "Summed explicit downtime_hours observations.",
    }
