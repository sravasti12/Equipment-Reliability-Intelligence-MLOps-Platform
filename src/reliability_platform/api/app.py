"""FastAPI application for calibrated RUL prediction and maintenance priorities."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pydantic import ValidationError
from sklearn.pipeline import Pipeline

from reliability_platform.api.schemas import (
    ApiSettings,
    ExplanationContribution,
    ExplanationResponse,
    HealthResponse,
    MachineTelemetry,
    PredictionInterval,
    PredictionResponse,
    ScheduledMachine,
    ScheduleRequest,
    ScheduleResponse,
    serving_manifest_from_json,
)
from reliability_platform.modeling.conformal import prediction_intervals
from reliability_platform.modeling.features import engineer_features
from reliability_platform.modeling.scheduling import maintenance_risk, schedule_machines
from reliability_platform.modeling.serving import ServingManifest

logger = logging.getLogger(__name__)


def settings_from_environment() -> ApiSettings:
    """Parse runtime settings without implicit defaults for model artifacts."""
    return ApiSettings.model_validate(
        {
            "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
            "serving_manifest_path": os.getenv(
                "SERVING_MANIFEST_PATH",
                "reports/generated/serving_model.json",
            ),
            "crew_capacity_per_shift": os.getenv("CREW_CAPACITY_PER_SHIFT", "5"),
            "risk_threshold": os.getenv("RISK_THRESHOLD", "0.5"),
            "urgency_horizon_cycles": os.getenv("URGENCY_HORIZON_CYCLES", "30"),
        }
    )


class ServingRuntime:
    """Owns the loaded artifact and exposes an explicit unavailable state."""

    def __init__(
        self,
        settings: ApiSettings,
        *,
        model: Pipeline | None = None,
        manifest: ServingManifest | None = None,
    ) -> None:
        self.settings = settings
        self.model = model
        self.manifest = manifest
        self.load_error: str | None = None
        if (model is None) != (manifest is None):
            raise ValueError("model and manifest must be provided together")
        if model is None:
            self._load_configured_model()

    @property
    def available(self) -> bool:
        return self.model is not None and self.manifest is not None

    @property
    def model_version(self) -> str | None:
        return self.manifest.model_run_id if self.manifest is not None else None

    def _load_configured_model(self) -> None:
        try:
            manifest_path = Path(self.settings.serving_manifest_path)
            manifest = serving_manifest_from_json(
                manifest_path.read_text(encoding="utf-8")
            )
            mlflow.set_tracking_uri(self.settings.tracking_uri)
            model_path = self._local_model_artifact_path(manifest.model_run_id)
            model_uri = (
                str(model_path)
                if model_path is not None
                else f"runs:/{manifest.model_run_id}/model"
            )
            model = mlflow.sklearn.load_model(model_uri)
            model_features = list(model.feature_names_in_)
            if model_features != manifest.feature_columns:
                raise ValueError(
                    "serving manifest feature columns do not match the model artifact"
                )
            self.manifest = manifest
            self.model = model
        except (
            OSError,
            AttributeError,
            ValidationError,
            MlflowException,
            ValueError,
        ) as error:
            self.load_error = str(error)
            logger.error("Configured serving model is unavailable: %s", error)

    def _local_model_artifact_path(self, run_id: str) -> Path | None:
        parsed_uri = urlparse(self.settings.tracking_uri)
        if parsed_uri.scheme != "file":
            return None
        root = unquote(parsed_uri.path)
        if os.name == "nt" and len(root) > 2 and root[0] == "/" and root[2] == ":":
            root = root[1:]
        tracking_path = Path(root)
        if not tracking_path.is_absolute():
            tracking_path = Path.cwd() / tracking_path
        run = MlflowClient(tracking_uri=self.settings.tracking_uri).get_run(run_id)
        model_path = (
            tracking_path / run.info.experiment_id / run_id / "artifacts" / "model"
        )
        return model_path if model_path.is_dir() else None

    def require_model(self) -> tuple[Pipeline, ServingManifest]:
        if self.model is None or self.manifest is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model is unavailable; check the serving manifest and "
                    "MLflow artifact."
                ),
            )
        return self.model, self.manifest

    def features_for(self, machine: MachineTelemetry) -> pd.DataFrame:
        model, manifest = self.require_model()
        rows: list[dict[str, Any]] = []
        for observation in machine.observations:
            row: dict[str, Any] = {
                "machine_id": machine.machine_id,
                "cycle": observation.cycle,
                # The feature builder needs a target column; only features are served.
                "rul": 0.0,
                **observation.sensor_values,
            }
            for name in ("setting_1", "setting_2", "setting_3"):
                value = getattr(observation, name)
                if value is not None:
                    row[name] = value
            if observation.operating_condition is not None:
                row["operating_condition"] = observation.operating_condition
            rows.append(row)

        try:
            features = engineer_features(
                pd.DataFrame(rows),
                window_size=manifest.window_size,
            ).features
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        missing = [
            column for column in manifest.feature_columns if column not in features
        ]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Telemetry is missing required model features: {missing}",
            )
        selected = features.loc[:, manifest.feature_columns]
        if list(model.feature_names_in_) != manifest.feature_columns:
            raise HTTPException(
                status_code=503,
                detail="Model feature schema does not match the serving manifest.",
            )
        return selected.iloc[[-1]]

    def predict_machine(
        self, machine: MachineTelemetry
    ) -> tuple[float, float, float, float]:
        model, manifest = self.require_model()
        features = self.features_for(machine)
        predicted = float(np.asarray(model.predict(features), dtype=float)[0])
        lower, upper = prediction_intervals(
            np.asarray([predicted], dtype=float),
            error_bound=manifest.conformal_error_bound_cycles,
        )
        risk = maintenance_risk(
            predicted,
            float(lower[0]),
            urgency_horizon_cycles=self.settings.urgency_horizon_cycles,
        )
        return predicted, float(lower[0]), float(upper[0]), risk


def create_app(
    settings: ApiSettings | None = None,
    *,
    runtime: ServingRuntime | None = None,
) -> FastAPI:
    """Build the API app, loading a configured MLflow run unless injected."""
    resolved_settings = settings or settings_from_environment()
    serving_runtime = runtime or ServingRuntime(resolved_settings)
    application = FastAPI(
        title="Equipment Reliability Prediction API",
        version="1.0.0",
        description=(
            "Serves RUL estimates, calibrated prediction intervals, SHAP "
            "explanations, and deterministic maintenance priorities."
        ),
    )
    application.state.serving_runtime = serving_runtime

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        available = serving_runtime.available
        return HealthResponse(
            status="ok" if available else "degraded",
            model_available=available,
            model_version=serving_runtime.model_version,
        )

    @application.post("/predict", response_model=PredictionResponse)
    def predict(machine: MachineTelemetry) -> PredictionResponse:
        predicted, lower, upper, risk = serving_runtime.predict_machine(machine)
        version = serving_runtime.model_version
        if version is None:
            raise HTTPException(status_code=503, detail="Model version is unavailable.")
        return PredictionResponse(
            machine_id=machine.machine_id,
            predicted_rul=predicted,
            risk_score=risk,
            prediction_interval=PredictionInterval(lower=lower, upper=upper),
            model_version=version,
        )

    @application.post("/explain", response_model=ExplanationResponse)
    def explain(machine: MachineTelemetry) -> ExplanationResponse:
        model, _ = serving_runtime.require_model()
        features = serving_runtime.features_for(machine)
        if not hasattr(model, "named_steps"):
            raise HTTPException(
                status_code=501,
                detail="SHAP explanations require a fitted sklearn pipeline.",
            )
        try:
            preprocessor = model.named_steps["preprocessor"]
            wrapped_regressor = model.named_steps["regressor"]
            estimator = wrapped_regressor.regressor_
        except (AttributeError, KeyError) as error:
            raise HTTPException(
                status_code=501,
                detail=(
                    "The configured model does not expose a supported tree pipeline."
                ),
            ) from error
        if not hasattr(estimator, "estimators_"):
            raise HTTPException(
                status_code=501,
                detail="SHAP explanations are available for tree-based models only.",
            )

        import shap

        transformed = preprocessor.transform(features)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed_array = np.asarray(transformed, dtype=float)
        explainer = shap.TreeExplainer(
            estimator,
            feature_perturbation="tree_path_dependent",
        )
        values = np.asarray(explainer.shap_values(transformed_array), dtype=float)
        if values.ndim == 3:
            values = values[..., 0]
        contributions = values.reshape(-1)
        base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        prediction = float(np.asarray(estimator.predict(transformed_array))[0])
        residual = abs(base_value + float(contributions.sum()) - prediction)
        if not np.isclose(
            base_value + float(contributions.sum()),
            prediction,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise HTTPException(
                status_code=500,
                detail="SHAP contributions failed the model additivity check.",
            )
        names = [str(name) for name in preprocessor.get_feature_names_out()]
        top_indices = np.argsort(np.abs(contributions))[::-1][:15]
        version = serving_runtime.model_version
        if version is None:
            raise HTTPException(status_code=503, detail="Model version is unavailable.")
        return ExplanationResponse(
            machine_id=machine.machine_id,
            model_version=version,
            predicted_rul=serving_runtime.predict_machine(machine)[0],
            base_value=base_value,
            max_abs_additivity_residual=residual,
            method="TreeExplainer tree-path-dependent",
            top_contributions=[
                ExplanationContribution(
                    feature=names[index],
                    shap_value=float(contributions[index]),
                )
                for index in top_indices
            ],
        )

    @application.post("/schedule", response_model=ScheduleResponse)
    def schedule(request: ScheduleRequest) -> ScheduleResponse:
        serving_runtime.require_model()
        prediction_rows: list[dict[str, Any]] = []
        for machine in request.machines:
            predicted, lower, upper, risk = serving_runtime.predict_machine(machine)
            prediction_rows.append(
                {
                    "machine_id": machine.machine_id,
                    "predicted_rul": predicted,
                    "conformal_lower_bound": lower,
                    "conformal_upper_bound": upper,
                    "risk_score": risk,
                }
            )
        priorities = schedule_machines(
            pd.DataFrame(prediction_rows),
            crew_capacity_per_shift=serving_runtime.settings.crew_capacity_per_shift,
            risk_threshold=serving_runtime.settings.risk_threshold,
        )
        version = serving_runtime.model_version
        if version is None:
            raise HTTPException(status_code=503, detail="Model version is unavailable.")
        priority_rows = priorities.to_dict(orient="records")
        return ScheduleResponse(
            model_version=version,
            priorities=[
                ScheduledMachine(
                    machine_id=str(row["machine_id"]),
                    predicted_rul=float(row["predicted_rul"]),
                    risk_score=float(row["risk_score"]),
                    prediction_interval=PredictionInterval(
                        lower=float(row["conformal_lower_bound"]),
                        upper=float(row["conformal_upper_bound"]),
                    ),
                    model_version=version,
                    priority_rank=int(row["priority_rank"]),
                    scheduled=bool(row["scheduled"]),
                    deferred=bool(row["deferred"]),
                    below_risk_threshold=bool(row["below_risk_threshold"]),
                )
                for row in priority_rows
            ],
        )

    return application


app = create_app()
