"""Reproducible baseline RUL training with MLflow experiment tracking."""

import json
import re
from functools import partial
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from reliability_platform.data.validation import validate_fleet
from reliability_platform.modeling.features import FeatureDataset, engineer_features
from reliability_platform.modeling.metrics import grouped_metrics, regression_metrics
from reliability_platform.modeling.splitting import (
    MachinePartitions,
    split_by_machine,
)


def _preprocessor(features: pd.DataFrame, *, scale_numeric: bool) -> ColumnTransformer:
    categorical = [
        column
        for column in features.columns
        if not pd.api.types.is_numeric_dtype(features[column])
    ]
    numeric = [column for column in features.columns if column not in categorical]
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
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


def _baseline_models(
    features: pd.DataFrame,
    *,
    random_seed: int,
) -> dict[str, Pipeline]:
    estimators: dict[str, tuple[Any, bool]] = {
        "constant_mean": (DummyRegressor(strategy="mean"), False),
        "linear_regression": (LinearRegression(), True),
        "random_forest": (
            RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=2,
                random_state=random_seed,
                n_jobs=1,
            ),
            False,
        ),
    }
    return {
        name: Pipeline(
            [
                ("preprocessor", _preprocessor(features, scale_numeric=scale)),
                (
                    "regressor",
                    TransformedTargetRegressor(
                        regressor=estimator,
                        func=np.asarray,
                        inverse_func=partial(np.clip, a_min=0.0, a_max=None),
                        check_inverse=False,
                    ),
                ),
            ]
        )
        for name, (estimator, scale) in estimators.items()
    }


def _split_counts(partitions: MachinePartitions) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for name in ("train", "validation", "test"):
        frame = getattr(partitions, name)
        identity = [
            column
            for column in ("source", "dataset_split", "machine_id")
            if column in frame
        ]
        result[name] = {
            "rows": int(len(frame)),
            "machines": int(frame[identity].drop_duplicates().shape[0]),
        }
    return result


def _log_group_metrics(
    split_name: str,
    groups: dict[str, object],
) -> None:
    def log_rows(prefix: str, rows: object) -> None:
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            group = re.sub(r"[^A-Za-z0-9_-]+", "_", str(row["group"])).strip("_")
            for metric in ("rmse", "mae", "asymmetric_score"):
                mlflow.log_metric(
                    f"{split_name}_{prefix}_{group}_{metric}",
                    float(row[metric]),
                )

    machine_groups = groups.get("machine_type")
    log_rows("machine_type", machine_groups)
    conditions = groups.get("operating_conditions")
    if not isinstance(conditions, dict):
        return
    for condition, summary in conditions.items():
        if isinstance(summary, list):
            log_rows(str(condition), summary)
        elif isinstance(summary, dict):
            log_rows(str(condition), summary.get("metrics"))


def train_baselines(
    frame: pd.DataFrame,
    *,
    tracking_uri: str = "file:./mlruns",
    experiment_name: str = "reliability-rul-baselines",
    random_seed: int = 42,
    window_size: int = 5,
    dataset_name: str = "unspecified",
) -> dict[str, object]:
    """Train three baselines using machine-disjoint splits and log each run."""
    validate_fleet(frame)
    partitions = split_by_machine(frame, random_seed=random_seed)
    feature_sets: dict[str, FeatureDataset] = {
        name: engineer_features(getattr(partitions, name), window_size=window_size)
        for name in ("train", "validation", "test")
    }
    training = feature_sets["train"]
    models = _baseline_models(training.features, random_seed=random_seed)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    split_counts = _split_counts(partitions)
    results: dict[str, object] = {
        "dataset": dataset_name,
        "random_seed": random_seed,
        "window_size": window_size,
        "split_fractions": {"train": 0.7, "validation": 0.15, "test": 0.15},
        "splits": split_counts,
        "feature_columns": list(training.features.columns),
        "metric_definitions": {
            "rmse": "Root mean squared error in RUL cycles.",
            "mae": "Mean absolute error in RUL cycles.",
            "asymmetric_score": (
                "Sum exp(error/10)-1 for overprediction and "
                "exp(-error/13)-1 for underprediction."
            ),
            "prediction_constraint": "Negative predicted RUL is clipped to zero.",
        },
        "models": {},
    }
    models_result = results["models"]
    if not isinstance(models_result, dict):
        raise RuntimeError("internal model result structure is invalid")

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.log_params(
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "random_seed": random_seed,
                    "window_size": window_size,
                    "training_rows": len(training.target),
                    "training_machines": split_counts["train"]["machines"],
                }
            )
            model.fit(training.features, training.target)
            split_results: dict[str, dict[str, object]] = {}
            for split_name, feature_set in feature_sets.items():
                predictions = model.predict(feature_set.features)
                metrics = regression_metrics(feature_set.target, predictions)
                split_results[split_name] = {"overall": metrics}
                if split_name in {"validation", "test"}:
                    grouped = grouped_metrics(
                        feature_set.target,
                        predictions,
                        feature_set.metadata,
                        training_metadata=training.metadata,
                    )
                    split_results[split_name]["by_group"] = grouped
                    _log_group_metrics(split_name, grouped)
                mlflow.log_metrics(
                    {
                        f"{split_name}_{metric_name}": float(value)
                        for metric_name, value in metrics.items()
                    }
                )
            input_example = training.features.head(5)
            signature = mlflow.models.infer_signature(
                input_example,
                model.predict(input_example),
            )
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example,
                signature=signature,
            )
            models_result[model_name] = {
                "run_id": run.info.run_id,
                "metrics": split_results,
            }
    return results


def save_report(report: dict[str, object], output: Path) -> Path:
    """Write a JSON-safe model report, creating its parent directory."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
