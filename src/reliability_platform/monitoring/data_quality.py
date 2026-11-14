"""Windowed data-quality reports for incoming machine telemetry."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def _numeric(values: pd.Series[Any]) -> pd.Series[float]:
    return pd.to_numeric(values, errors="coerce").astype("float64")


def build_data_quality_report(
    reference: pd.DataFrame,
    incoming: pd.DataFrame,
    *,
    window_column: str = "window_index",
    machine_column: str = "machine_id",
    timestamp_column: str = "cycle",
    sensor_columns: Sequence[str] | None = None,
    max_missing_fraction: float = 0.1,
    stuck_value_fraction: float = 0.95,
    minimum_stuck_observations: int = 5,
) -> dict[str, object]:
    """Measure schema, range, missingness, stuck-sensor, and duplicate issues.

    Sensor range checks use observed reference minima and maxima, not assumed
    physical operating limits.
    """
    for frame_name, frame in (("reference", reference), ("incoming", incoming)):
        if frame.empty:
            raise ValueError(f"{frame_name} data must not be empty")
        if not all(isinstance(column, str) for column in frame.columns):
            raise ValueError("data column names must be strings")

    if not math.isfinite(max_missing_fraction) or not 0 <= max_missing_fraction <= 1:
        raise ValueError("max_missing_fraction must be between 0 and 1")
    if not math.isfinite(stuck_value_fraction) or not 0 < stuck_value_fraction <= 1:
        raise ValueError("stuck_value_fraction must be in (0, 1]")
    if (
        isinstance(minimum_stuck_observations, bool)
        or not isinstance(minimum_stuck_observations, int)
        or minimum_stuck_observations < 2
    ):
        raise ValueError("minimum_stuck_observations must be an integer of at least 2")
    if window_column not in incoming.columns:
        raise ValueError(f"incoming data must contain {window_column!r}")
    for column in (machine_column, timestamp_column):
        if column not in reference.columns or column not in incoming.columns:
            raise ValueError(f"reference and incoming data must contain {column!r}")

    selected_sensors = (
        list(sensor_columns)
        if sensor_columns is not None
        else [column for column in reference.columns if column.startswith("sensor_")]
    )
    if not selected_sensors:
        raise ValueError("at least one reference sensor column is required")
    if len(selected_sensors) != len(set(selected_sensors)):
        raise ValueError("sensor columns must be unique")
    unknown_sensors = sorted(set(selected_sensors).difference(reference.columns))
    if unknown_sensors:
        raise ValueError(
            "sensor columns must exist in reference data: " + ", ".join(unknown_sensors)
        )

    window_values = _numeric(incoming[window_column])
    window_numbers = window_values.to_numpy(dtype=float)
    if (
        not np.isfinite(window_numbers).all()
        or (window_numbers < 0).any()
        or not np.equal(window_numbers, np.floor(window_numbers)).all()
    ):
        raise ValueError("window indices must be finite non-negative integers")

    expected_columns = set(reference.columns).difference({window_column})
    incoming_columns = set(incoming.columns).difference({window_column})
    missing_columns = sorted(expected_columns.difference(incoming_columns))
    unexpected_columns = sorted(incoming_columns.difference(expected_columns))

    reference_ranges: dict[str, tuple[float, float] | None] = {}
    for sensor in selected_sensors:
        reference_values = _numeric(reference[sensor])
        finite_values = reference_values[np.isfinite(reference_values)]
        reference_ranges[sensor] = (
            (float(finite_values.min()), float(finite_values.max()))
            if not finite_values.empty
            else None
        )

    incoming = incoming.copy()
    incoming[window_column] = window_values.astype("int64")
    window_results: list[dict[str, object]] = []
    for window_index, window in incoming.groupby(window_column, sort=True):
        window_index_int = int(window_index)  # type: ignore[arg-type]
        sensor_missingness: list[dict[str, object]] = []
        out_of_range: list[dict[str, object]] = []
        non_numeric_sensors: list[dict[str, object]] = []
        for sensor in selected_sensors:
            if sensor not in window.columns:
                values = pd.Series(float("nan"), index=window.index, dtype="float64")
                missing_count = len(window)
            else:
                raw_values = window[sensor]
                converted = _numeric(raw_values)
                invalid = raw_values.notna() & (
                    converted.isna() | ~np.isfinite(converted)
                )
                invalid_count = int(invalid.sum())
                if invalid_count:
                    non_numeric_sensors.append(
                        {"column": sensor, "invalid_values": invalid_count}
                    )
                values = converted.replace([np.inf, -np.inf], np.nan)
                missing_count = int(raw_values.isna().sum())

            missing_fraction = float(missing_count / len(window))
            sensor_missingness.append(
                {
                    "column": sensor,
                    "missing_values": missing_count,
                    "missing_fraction": missing_fraction,
                    "threshold": max_missing_fraction,
                    "threshold_exceeded": missing_fraction > max_missing_fraction,
                }
            )
            bounds = reference_ranges[sensor]
            if bounds is not None:
                minimum, maximum = bounds
                outside = values.notna() & ((values < minimum) | (values > maximum))
                outside_count = int(outside.sum())
                if outside_count:
                    out_of_range.append(
                        {
                            "column": sensor,
                            "values": outside_count,
                            "reference_minimum": minimum,
                            "reference_maximum": maximum,
                        }
                    )

        stuck_sensors: list[dict[str, object]] = []
        for machine_id, machine_rows in window.groupby(
            machine_column, sort=True, dropna=False
        ):
            for sensor in selected_sensors:
                if sensor not in machine_rows.columns:
                    continue
                readings = _numeric(machine_rows[sensor])
                readings = readings[np.isfinite(readings)]
                if len(readings) < minimum_stuck_observations:
                    continue
                repeated_fraction = float(readings.value_counts().max() / len(readings))
                if repeated_fraction >= stuck_value_fraction:
                    stuck_sensors.append(
                        {
                            "machine_id": (
                                None if pd.isna(machine_id) else str(machine_id)
                            ),
                            "column": sensor,
                            "observations": int(len(readings)),
                            "repeated_value_fraction": repeated_fraction,
                        }
                    )

        valid_timestamps = window.loc[
            window[machine_column].notna() & window[timestamp_column].notna(),
            [machine_column, timestamp_column],
        ]
        duplicate_timestamp_rows = int(
            valid_timestamps.duplicated(subset=[machine_column, timestamp_column]).sum()
        )
        missing_identity_rows = int(
            (window[machine_column].isna() | window[timestamp_column].isna()).sum()
        )
        schema_violations = {
            "missing_columns": missing_columns,
            "unexpected_columns": unexpected_columns,
            "non_numeric_sensor_values": non_numeric_sensors,
            "missing_machine_or_timestamp_rows": missing_identity_rows,
        }
        issues_detected = bool(
            missing_columns
            or unexpected_columns
            or non_numeric_sensors
            or missing_identity_rows
            or any(item["threshold_exceeded"] for item in sensor_missingness)
            or out_of_range
            or stuck_sensors
            or duplicate_timestamp_rows
        )
        sources = (
            sorted(str(value) for value in window["source"].dropna().unique())
            if "source" in window.columns
            else ["unspecified"]
        )
        window_results.append(
            {
                "window_index": window_index_int,
                "rows": int(len(window)),
                "sources": sources or ["unspecified"],
                "schema_violations": schema_violations,
                "missingness": sensor_missingness,
                "out_of_range_values": out_of_range,
                "out_of_range_basis": "observed_reference_min_max",
                "stuck_sensors": stuck_sensors,
                "duplicate_timestamp_rows": duplicate_timestamp_rows,
                "quality_status": (
                    "issues_detected" if issues_detected else "no_thresholds_exceeded"
                ),
            }
        )

    reference_sources = (
        sorted(str(value) for value in reference["source"].dropna().unique())
        if "source" in reference.columns
        else ["unspecified"]
    )
    return {
        "schema_version": 1,
        "reference_sources": reference_sources or ["unspecified"],
        "windows": window_results,
        "thresholds": {
            "max_missing_fraction": max_missing_fraction,
            "stuck_value_fraction": stuck_value_fraction,
            "minimum_stuck_observations": minimum_stuck_observations,
        },
    }
