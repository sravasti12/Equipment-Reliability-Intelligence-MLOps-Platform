"""Fit and track a calibrated random-forest model for API serving."""

from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.pipeline import Pipeline

from reliability_platform.data.validation import validate_fleet
from reliability_platform.modeling.baselines import _baseline_models
from reliability_platform.modeling.conformal import calibrate_trajectory_error
from reliability_platform.modeling.features import engineer_features
from reliability_platform.modeling.metrics import regression_metrics
from reliability_platform.modeling.splitting import split_by_machine


class ServingManifest(BaseModel):
    """Model artifact and calibration metadata required by the API."""

    model_config = ConfigDict(extra="forbid")

    model_run_id: str = Field(min_length=1)
    feature_columns: list[str] = Field(min_length=1)
    conformal_error_bound_cycles: float = Field(ge=0, allow_inf_nan=False)
    target_coverage: float = Field(gt=0, lt=1)
    window_size: int = Field(ge=2)

    @field_validator("feature_columns")
    @classmethod
    def feature_columns_are_unique(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)) or any(not value for value in values):
            raise ValueError("feature_columns must contain unique non-empty names")
        return values


def train_serving_model(
    frame: pd.DataFrame,
    *,
    output_path: Path,
    tracking_uri: str = "file:./mlruns",
    experiment_name: str = "reliability-platform-serving",
    random_seed: int = 42,
    window_size: int = 5,
    target_coverage: float = 0.9,
) -> ServingManifest:
    """Fit on training machines and calibrate a serving model on validation ones."""
    if not 0 < target_coverage < 1:
        raise ValueError("target_coverage must be between 0 and 1")
    validate_fleet(frame)
    partitions = split_by_machine(frame, random_seed=random_seed)
    training = engineer_features(partitions.train, window_size=window_size)
    calibration = engineer_features(partitions.validation, window_size=window_size)
    model: Pipeline = _baseline_models(
        training.features,
        random_seed=random_seed,
    )["random_forest"]
    model.fit(training.features, training.target)

    calibration_prediction = np.asarray(
        model.predict(calibration.features),
        dtype=np.float64,
    )
    error_bound, calibration_machines = calibrate_trajectory_error(
        calibration.target,
        calibration_prediction,
        calibration.metadata,
        alpha=1 - target_coverage,
    )
    validation_metrics = regression_metrics(
        calibration.target,
        calibration_prediction,
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="calibrated_random_forest_api") as run:
        mlflow.log_params(
            {
                "model": "random_forest",
                "random_seed": random_seed,
                "window_size": window_size,
                "target_coverage": target_coverage,
                "training_machines": int(partitions.train["machine_id"].nunique()),
                "calibration_machines": calibration_machines,
            }
        )
        mlflow.log_metrics(
            {
                "validation_rmse": float(validation_metrics["rmse"]),
                "validation_mae": float(validation_metrics["mae"]),
                "conformal_error_bound_cycles": error_bound,
            }
        )
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
        )
        manifest = ServingManifest(
            model_run_id=run.info.run_id,
            feature_columns=list(training.features.columns),
            conformal_error_bound_cycles=error_bound,
            target_coverage=target_coverage,
            window_size=window_size,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
