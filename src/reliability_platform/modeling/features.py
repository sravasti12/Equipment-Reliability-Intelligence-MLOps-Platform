"""Causal machine-cycle feature engineering for remaining useful life."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass(frozen=True)
class FeatureDataset:
    """Model inputs, targets, and non-feature metadata aligned by row."""

    features: pd.DataFrame
    target: pd.Series[float]
    metadata: pd.DataFrame


def _trailing_slope(
    cycles: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    window: int,
) -> npt.NDArray[np.float64]:
    slopes = np.full(len(values), np.nan, dtype=float)
    for end in range(len(values)):
        start = max(0, end - window + 1)
        x = cycles[start : end + 1].astype(float)
        y = values[start : end + 1]
        valid = np.isfinite(y)
        if valid.sum() < 2:
            continue
        x_valid = x[valid]
        y_valid = y[valid]
        centered_x = x_valid - x_valid.mean()
        denominator = float(np.dot(centered_x, centered_x))
        if denominator > 0:
            slopes[end] = float(
                np.dot(centered_x, y_valid - y_valid.mean()) / denominator
            )
    return slopes


def engineer_features(
    frame: pd.DataFrame,
    *,
    window_size: int = 5,
) -> FeatureDataset:
    """Create past/current-only features and keep labels out of the model inputs.

    Rolling windows use the current and preceding observations. Sensor variation
    is measured from the first recorded cycle, never from a later non-missing
    reading. Machine ID, RUL, events, and dataset labels are returned only as
    metadata or targets, not as predictors.
    """
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    required = {"machine_id", "cycle", "rul"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {', '.join(sorted(missing))}")
    if frame.empty or frame["machine_id"].isna().any():
        raise ValueError("dataset must contain non-empty machine identifiers")

    sensor_columns = sorted(
        (column for column in frame.columns if column.startswith("sensor_")),
        key=lambda column: (
            0,
            int(column.removeprefix("sensor_")),
        )
        if column.removeprefix("sensor_").isdigit()
        else (1, column),
    )
    if not sensor_columns:
        raise ValueError("dataset must contain at least one sensor_* column")

    data = frame.copy()
    for column in ("cycle", "rul", *sensor_columns):
        numeric = pd.to_numeric(data[column], errors="coerce")
        finite = numeric.dropna().to_numpy(dtype=float)
        if column in {"cycle", "rul"} and numeric.isna().any():
            raise ValueError(f"{column} must contain finite numeric values")
        if not np.isfinite(finite).all():
            raise ValueError(f"{column} must contain finite numeric values or NaN")
        data[column] = numeric
    if (
        data["cycle"].le(0).any()
        or not np.equal(
            data["cycle"].to_numpy(dtype=float),
            np.floor(data["cycle"].to_numpy(dtype=float)),
        ).all()
    ):
        raise ValueError("cycle must contain positive integer values")
    if data["rul"].lt(0).any():
        raise ValueError("rul cannot contain negative values")

    identity_columns = [
        column for column in ("source", "dataset_split", "machine_id") if column in data
    ]
    if data[identity_columns].isna().any().any():
        raise ValueError("machine identity columns cannot contain missing values")
    if data.duplicated([*identity_columns, "cycle"]).any():
        raise ValueError("duplicate machine identity/cycle rows found")

    data = data.sort_values([*identity_columns, "cycle"], kind="stable").reset_index(
        drop=True
    )
    metadata_columns = [
        column
        for column in (
            "machine_id",
            "cycle",
            "machine_type",
            "operating_condition",
            "setting_1",
            "setting_2",
            "setting_3",
            "source",
            "dataset_split",
        )
        if column in data.columns
    ]
    feature_columns = [
        column
        for column in (*sensor_columns, "setting_1", "setting_2", "setting_3")
        if column in data.columns
    ]
    if "operating_condition" in data.columns:
        feature_columns.append("operating_condition")

    feature_frames: list[pd.DataFrame] = []
    for _, machine in data.groupby(identity_columns, sort=False, dropna=False):
        machine_features = machine.loc[:, feature_columns].copy()
        machine_features["cycle_number"] = machine["cycle"].to_numpy(dtype=float)
        cycles = machine["cycle"].to_numpy(dtype=np.float64)
        relative_variations: list[npt.NDArray[np.float64]] = []

        for sensor in sensor_columns:
            values = machine[sensor].to_numpy(dtype=np.float64)
            series = pd.Series(values, index=machine.index)
            machine_features[f"{sensor}_rolling_mean_{window_size}"] = (
                series.rolling(window_size, min_periods=1).mean().to_numpy()
            )
            machine_features[f"{sensor}_rolling_std_{window_size}"] = (
                series.rolling(window_size, min_periods=2).std().to_numpy()
            )
            machine_features[f"{sensor}_trend_{window_size}"] = _trailing_slope(
                cycles, values, window_size
            )

            first_value = values[0]
            if np.isfinite(first_value):
                variation = np.abs(values - first_value)
                relative_variations.append(variation / max(abs(first_value), 1.0))
            else:
                variation = np.full(len(values), np.nan, dtype=float)
                relative_variations.append(variation)
            machine_features[f"{sensor}_variation"] = variation

        variation_matrix = np.vstack(relative_variations)
        finite_counts = np.isfinite(variation_matrix).sum(axis=0)
        machine_features["sensor_health_index"] = np.divide(
            np.nansum(variation_matrix, axis=0),
            finite_counts,
            out=np.full(len(machine), np.nan, dtype=float),
            where=finite_counts > 0,
        )
        feature_frames.append(machine_features)

    features = pd.concat(feature_frames, ignore_index=True)
    if "operating_condition" in features.columns:
        features["operating_condition"] = features["operating_condition"].astype(
            "string"
        )
    target = data["rul"].astype(float).reset_index(drop=True)
    metadata = data.loc[:, metadata_columns].reset_index(drop=True)
    return FeatureDataset(features=features, target=target, metadata=metadata)


def machine_identity_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns that jointly identify trajectories within a combined table."""
    columns = [
        column
        for column in ("source", "dataset_split", "machine_id")
        if column in frame
    ]
    if "machine_id" not in columns:
        raise ValueError("dataset must contain machine_id")
    return columns
