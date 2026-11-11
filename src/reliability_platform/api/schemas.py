"""Validated API request and response models."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from reliability_platform.modeling.serving import ServingManifest


class TelemetryObservation(BaseModel):
    """One cycle of causal machine telemetry."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    cycle: int = Field(gt=0)
    sensor_values: dict[str, float | None] = Field(min_length=1)
    setting_1: float | None = None
    setting_2: float | None = None
    setting_3: float | None = None
    operating_condition: str | int | float | None = None

    @field_validator("sensor_values")
    @classmethod
    def validate_sensor_names(
        cls, values: dict[str, float | None]
    ) -> dict[str, float | None]:
        if any(re.fullmatch(r"sensor_[1-9]\d*", name) is None for name in values):
            raise ValueError("sensor_values keys must use names such as sensor_1")
        return values


class MachineTelemetry(BaseModel):
    """Current and historical observations for one machine."""

    model_config = ConfigDict(extra="forbid")

    machine_id: str = Field(min_length=1, max_length=128)
    observations: list[TelemetryObservation] = Field(min_length=1, max_length=5000)

    @field_validator("machine_id")
    @classmethod
    def machine_id_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("machine_id must not be blank")
        return value

    @model_validator(mode="after")
    def observations_have_unique_cycles(self) -> MachineTelemetry:
        cycles = [observation.cycle for observation in self.observations]
        if len(cycles) != len(set(cycles)):
            raise ValueError("observations must not contain duplicate cycles")
        return self


class ScheduleRequest(BaseModel):
    """Fleet-level input for the maintenance-priority endpoint."""

    machines: list[MachineTelemetry] = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def machines_are_unique(self) -> ScheduleRequest:
        machine_ids = [machine.machine_id for machine in self.machines]
        if len(machine_ids) != len(set(machine_ids)):
            raise ValueError("machines must have unique machine_id values")
        return self


class PredictionInterval(BaseModel):
    lower: float = Field(ge=0)
    upper: float = Field(ge=0)


class PredictionResponse(BaseModel):
    machine_id: str
    predicted_rul: float = Field(ge=0)
    risk_score: float = Field(ge=0, le=1)
    prediction_interval: PredictionInterval
    model_version: str


class ScheduledMachine(PredictionResponse):
    priority_rank: int = Field(ge=1)
    scheduled: bool
    deferred: bool
    below_risk_threshold: bool


class ScheduleResponse(BaseModel):
    model_version: str
    priorities: list[ScheduledMachine]


class ExplanationContribution(BaseModel):
    feature: str
    shap_value: float


class ExplanationResponse(BaseModel):
    machine_id: str
    model_version: str
    predicted_rul: float = Field(ge=0)
    base_value: float
    max_abs_additivity_residual: float = Field(ge=0)
    method: Literal["TreeExplainer tree-path-dependent"]
    top_contributions: list[ExplanationContribution]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service_available: Literal[True] = True
    model_available: bool
    model_version: str | None


class ApiSettings(BaseModel):
    """Validated runtime configuration read from environment variables."""

    tracking_uri: str = Field(min_length=1)
    serving_manifest_path: str = Field(min_length=1)
    crew_capacity_per_shift: int = Field(ge=1, le=500)
    risk_threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    urgency_horizon_cycles: int = Field(ge=1)


def serving_manifest_from_json(raw_json: str) -> ServingManifest:
    """Validate persisted model identity and calibration metadata."""
    return ServingManifest.model_validate_json(raw_json)
