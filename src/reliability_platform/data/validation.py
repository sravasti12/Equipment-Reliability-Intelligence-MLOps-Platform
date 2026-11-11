"""Schema and data-quality validation for machine-cycle datasets."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationReport:
    """Summary of validated rows, machines, sensors, and missing values."""

    row_count: int
    machine_count: int
    sensor_columns: tuple[str, ...]
    missing_by_column: dict[str, int]
    sensor_missing_fraction: float


def validate_fleet(
    frame: pd.DataFrame,
    *,
    max_sensor_missing_fraction: float = 1.0,
    sensor_ranges: Mapping[str, tuple[float, float]] | None = None,
) -> ValidationReport:
    """Validate schema, cycle ordering, uniqueness, ranges, and missingness.

    Missing sensor readings are preserved and summarized. Missing identifiers,
    invalid numeric values, duplicate machine cycles, and unordered cycles fail
    validation rather than being silently corrected.
    """
    if frame.empty:
        raise ValueError("dataset must contain at least one row")
    if not 0 <= max_sensor_missing_fraction <= 1:
        raise ValueError("max_sensor_missing_fraction must be in [0, 1]")

    required_columns = {"machine_id", "cycle"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"dataset is missing required columns: {', '.join(sorted(missing_columns))}"
        )

    if frame["machine_id"].isna().any():
        raise ValueError("machine_id cannot contain missing values")
    machine_ids = pd.to_numeric(frame["machine_id"], errors="coerce")
    if (
        machine_ids.isna().any()
        or not np.isfinite(machine_ids.to_numpy(dtype=float)).all()
        or machine_ids.le(0).any()
        or not np.equal(
            machine_ids.to_numpy(dtype=float),
            np.floor(machine_ids.to_numpy(dtype=float)),
        ).all()
    ):
        raise ValueError("machine_id must contain positive integer values")

    cycles = pd.to_numeric(frame["cycle"], errors="coerce")
    if cycles.isna().any() or not np.isfinite(cycles.to_numpy(dtype=float)).all():
        raise ValueError("cycle must contain only finite numeric values")
    if (cycles < 1).any() or not np.equal(cycles, np.floor(cycles)).all():
        raise ValueError("cycle values must be positive integers")

    if frame.duplicated(subset=["machine_id", "cycle"]).any():
        raise ValueError("duplicate machine_id/cycle rows found")

    ordered_cycles = pd.Series(cycles.to_numpy(), index=frame.index)
    cycle_differences = ordered_cycles.groupby(frame["machine_id"], sort=False).diff()
    if cycle_differences.le(0).any():
        raise ValueError("cycles must be strictly increasing within each machine")

    sensor_columns = tuple(
        column for column in frame.columns if column.startswith("sensor_")
    )
    if not sensor_columns:
        raise ValueError("dataset must contain at least one sensor_* column")

    numeric_context_columns = [
        column
        for column in frame.columns
        if column.startswith("setting_") or column == "operating_condition"
    ]
    for column in numeric_context_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain only finite numeric values")

    missing_by_column: dict[str, int] = {}
    sensor_missing_count = 0
    for column in sensor_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid_values = frame[column].notna() & values.isna()
        if invalid_values.any():
            raise ValueError(f"{column} contains non-numeric sensor values")
        non_missing = values.dropna().to_numpy(dtype=float)
        if not np.isfinite(non_missing).all():
            raise ValueError(f"{column} contains infinite values")

        missing_count = int(values.isna().sum())
        missing_by_column[column] = missing_count
        sensor_missing_count += missing_count

    sensor_value_count = len(frame) * len(sensor_columns)
    sensor_missing_fraction = sensor_missing_count / sensor_value_count
    if sensor_missing_fraction > max_sensor_missing_fraction:
        raise ValueError(
            "sensor missing fraction "
            f"{sensor_missing_fraction:.4f} exceeds configured maximum "
            f"{max_sensor_missing_fraction:.4f}"
        )

    for column, bounds in (sensor_ranges or {}).items():
        if column not in sensor_columns:
            raise ValueError(f"range configured for unknown sensor column: {column}")
        minimum, maximum = bounds
        if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum > maximum:
            raise ValueError(f"invalid finite range configured for {column}")
        values = pd.to_numeric(frame[column], errors="coerce")
        out_of_range = values.notna() & ((values < minimum) | (values > maximum))
        if out_of_range.any():
            raise ValueError(f"{column} contains values outside [{minimum}, {maximum}]")

    if "rul" in frame.columns:
        rul = pd.to_numeric(frame["rul"], errors="coerce")
        if rul.isna().any() or not np.isfinite(rul.to_numpy(dtype=float)).all():
            raise ValueError("rul must contain only finite numeric values")
        if (rul < 0).any():
            raise ValueError("rul cannot contain negative values")

    for event_column in ("maintenance_event", "failure_event"):
        if event_column in frame.columns and (
            frame[event_column].isna().any()
            or not frame[event_column].isin([True, False, 0, 1]).all()
        ):
            raise ValueError(f"{event_column} must contain only boolean values")

    if "failure_event" in frame.columns:
        failure_events = frame["failure_event"].astype(bool)
        events_per_machine = failure_events.groupby(
            frame["machine_id"], sort=False
        ).sum()
        if events_per_machine.gt(1).any():
            raise ValueError("each machine can have at most one failure_event")
        if failure_events.any():
            final_cycles = cycles.groupby(frame["machine_id"], sort=False).transform(
                "max"
            )
            if (failure_events & cycles.ne(final_cycles)).any():
                raise ValueError("failure_event must occur on the final machine cycle")

    for column in ("source", "dataset_split"):
        if column in frame.columns:
            if frame[column].isna().any():
                raise ValueError(f"{column} cannot contain missing values")
            values_per_machine = frame.groupby("machine_id", sort=False)[
                column
            ].nunique(dropna=False)
            if values_per_machine.gt(1).any():
                raise ValueError(f"{column} must be consistent within each machine")

    return ValidationReport(
        row_count=len(frame),
        machine_count=int(frame["machine_id"].nunique()),
        sensor_columns=sensor_columns,
        missing_by_column=missing_by_column,
        sensor_missing_fraction=sensor_missing_fraction,
    )
