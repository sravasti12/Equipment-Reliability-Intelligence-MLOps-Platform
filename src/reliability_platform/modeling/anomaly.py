"""Isolation Forest evaluation on deterministic injected sensor anomalies."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _anomaly_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    columns = [column for column in features.columns if isinstance(column, str)]
    if len(columns) != len(features.columns):
        raise ValueError("feature column names must be strings")
    categorical: list[str] = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(features[column])
    ]
    numeric = [column for column in columns if column not in categorical]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median", keep_empty_features=True),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def fit_isolation_forest(
    train_features: pd.DataFrame,
    *,
    random_seed: int,
) -> tuple[IsolationForest, ColumnTransformer]:
    """Fit the detector and all preprocessing exclusively on training rows."""
    preprocessor = _anomaly_preprocessor(train_features)
    transformed = preprocessor.fit_transform(train_features)
    detector = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=random_seed,
        n_jobs=1,
    )
    detector.fit(transformed)
    return detector, preprocessor


def evaluate_injected_sensor_shift(
    detector: IsolationForest,
    preprocessor: ColumnTransformer,
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    training_features: pd.DataFrame,
    *,
    random_seed: int,
    shift_sigma: float = 6.0,
) -> tuple[dict[str, object], npt.NDArray[np.float64]]:
    """Score paired normal and shifted holdout readings, not failure risk.

    One observed feature vector per machine is paired with a copy whose first
    available sensor is increased by ``shift_sigma`` training standard
    deviations. This is a controlled injection benchmark, not a fault label.
    """
    identities = [
        column
        for column in ("source", "dataset_split", "machine_id")
        if column in metadata
    ]
    if "machine_id" not in identities:
        raise ValueError("metadata must include machine_id")
    sensor_columns = [
        column for column in features.columns if column.startswith("sensor_")
    ]
    if not sensor_columns:
        raise ValueError("anomaly evaluation requires sensor features")
    sensor = sensor_columns[0]
    machine_groups = metadata.groupby(identities, sort=False, dropna=False).indices
    rng = np.random.default_rng(random_seed)
    selected = np.asarray(
        [
            rng.choice(np.asarray(indices, dtype=np.intp))
            for indices in machine_groups.values()
        ],
        dtype=np.intp,
    )
    normal = features.iloc[selected].copy()
    injected = normal.copy()
    train_sensor = pd.to_numeric(training_features[sensor], errors="coerce")
    scale = float(train_sensor.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        scale = max(abs(float(train_sensor.mean())) * 0.1, 1.0)
    injected[sensor] = (
        pd.to_numeric(injected[sensor], errors="coerce") + shift_sigma * scale
    )
    paired_features = pd.concat([normal, injected], ignore_index=True)
    transformed = preprocessor.transform(paired_features)
    scores = -detector.decision_function(transformed)
    labels = np.concatenate(
        (np.zeros(len(normal), dtype=np.int8), np.ones(len(injected), dtype=np.int8))
    )
    top_count = len(injected)
    top_indices = np.argsort(scores)[-top_count:]
    return (
        {
            "machines": int(len(normal)),
            "injected_anomalies": int(len(injected)),
            "sensor": sensor,
            "shift_sigma": shift_sigma,
            "roc_auc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
            "precision_at_anomaly_count": float(labels[top_indices].mean()),
            "normal_mean_anomaly_score": float(scores[: len(normal)].mean()),
            "injected_mean_anomaly_score": float(scores[len(normal) :].mean()),
            "interpretation": (
                "Controlled injected-shift detection only; anomaly scores are not "
                "calibrated failure probabilities."
            ),
        },
        scores,
    )
