"""Read saved dashboard artifacts and call the configured prediction API."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from reliability_platform.data.storage import read_parquet


class DashboardDataError(RuntimeError):
    """Raised when a saved dashboard artifact is absent or malformed."""


class ApiUnavailable(RuntimeError):
    """Raised when the prediction API cannot serve a dashboard request."""


@dataclass(frozen=True)
class DatasetSpec:
    """A saved fleet dataset and its human-readable source label."""

    dataset_id: str
    label: str
    relative_path: Path


@dataclass(frozen=True)
class DashboardPaths:
    """Root paths for immutable processed data and generated reports."""

    project_root: Path

    @classmethod
    def from_environment(cls) -> DashboardPaths:
        configured_root = os.getenv("RELIABILITY_PLATFORM_ROOT")
        root = (
            Path(configured_root)
            if configured_root
            else Path(__file__).resolve().parents[3]
        )
        return cls(project_root=root)

    def dataset_path(self, dataset_id: str) -> Path:
        spec = DATASETS.get(dataset_id)
        if spec is None:
            raise DashboardDataError(f"Unknown dashboard dataset: {dataset_id}")
        return self.project_root / spec.relative_path

    def report_path(self, report_id: str) -> Path:
        relative_path = REPORTS.get(report_id)
        if relative_path is None:
            raise DashboardDataError(f"Unknown dashboard report: {report_id}")
        return self.project_root / relative_path


DATASETS = {
    "synthetic_fleet": DatasetSpec(
        dataset_id="synthetic_fleet",
        label="Synthetic industrial fleet (seed 42)",
        relative_path=Path("data/processed/synthetic_fleet.parquet"),
    ),
    "cmapss_fd001_train": DatasetSpec(
        dataset_id="cmapss_fd001_train",
        label="NASA C-MAPSS FD001 training set",
        relative_path=Path("data/processed/cmapss_FD001_train.parquet"),
    ),
    "cmapss_fd002_train": DatasetSpec(
        dataset_id="cmapss_fd002_train",
        label="NASA C-MAPSS FD002 training set",
        relative_path=Path("data/processed/cmapss_FD002_train.parquet"),
    ),
}

REPORTS = {
    "synthetic_reliability": Path("reports/generated/synthetic_reliability.json"),
    "fd001_reliability": Path("reports/generated/fd001_reliability.json"),
    "synthetic_baselines": Path("reports/generated/synthetic_baselines.json"),
    "fd002_baselines": Path("reports/generated/fd002_baselines.json"),
    "synthetic_advanced": Path("reports/generated/synthetic_advanced.json"),
    "fd002_advanced": Path("reports/generated/fd002_advanced.json"),
    "synthetic_maintenance": Path(
        "reports/generated/synthetic_maintenance_backtest.json"
    ),
    "serving_manifest": Path("reports/generated/serving_model.json"),
}


def available_datasets(paths: DashboardPaths) -> list[DatasetSpec]:
    """Return only dataset choices present in the processed-data directory."""
    return [
        spec
        for spec in DATASETS.values()
        if (paths.project_root / spec.relative_path).is_file()
    ]


def load_fleet(
    dataset_id: str,
    *,
    paths: DashboardPaths | None = None,
) -> pd.DataFrame:
    """Read one explicitly identified saved fleet dataset."""
    resolved_paths = paths or DashboardPaths.from_environment()
    spec = DATASETS.get(dataset_id)
    if spec is None:
        raise DashboardDataError(f"Unknown dashboard dataset: {dataset_id}")
    path = resolved_paths.project_root / spec.relative_path
    try:
        frame = read_parquet(path)
    except (OSError, ValueError) as error:
        raise DashboardDataError(f"Could not read {spec.label}: {error}") from error
    required = {"machine_id", "cycle"}
    if missing := required.difference(frame.columns):
        raise DashboardDataError(
            f"{spec.label} is missing required columns: {', '.join(sorted(missing))}"
        )
    if frame.empty:
        raise DashboardDataError(f"{spec.label} contains no machine-cycle rows")
    return frame


def load_report(
    report_id: str,
    *,
    paths: DashboardPaths | None = None,
) -> dict[str, Any]:
    """Load one known JSON report without accepting arbitrary file paths."""
    resolved_paths = paths or DashboardPaths.from_environment()
    path = resolved_paths.report_path(report_id)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DashboardDataError(
            f"Could not read saved report {report_id}: {error}"
        ) from error
    if not isinstance(report, dict):
        raise DashboardDataError(f"Saved report {report_id} must contain a JSON object")
    return report


def serving_window_size(*, paths: DashboardPaths | None = None) -> int:
    """Read the causal feature window configured for the saved serving model."""
    manifest = load_report("serving_manifest", paths=paths)
    window_size = manifest.get("window_size")
    if (
        isinstance(window_size, bool)
        or not isinstance(window_size, int)
        or window_size < 2
    ):
        raise DashboardDataError(
            "The serving manifest must define a window_size of at least 2."
        )
    return window_size


def machine_request(
    machine_rows: pd.DataFrame,
    *,
    window_size: int | None = None,
) -> dict[str, Any]:
    """Build causal API telemetry for one machine, excluding labels and events."""
    if machine_rows.empty:
        raise DashboardDataError("Machine history must not be empty")
    if window_size is not None and window_size < 2:
        raise DashboardDataError("window_size must be at least 2")
    required = {"machine_id", "cycle"}
    if missing := required.difference(machine_rows.columns):
        raise DashboardDataError(
            f"Machine history is missing required columns: {', '.join(sorted(missing))}"
        )
    machine_ids = machine_rows["machine_id"].drop_duplicates()
    if len(machine_ids) != 1:
        raise DashboardDataError("Machine history must contain exactly one machine")
    sensors = sorted(
        (column for column in machine_rows.columns if column.startswith("sensor_")),
        key=lambda name: (
            0,
            int(name.removeprefix("sensor_")),
        )
        if name.removeprefix("sensor_").isdigit()
        else (1, name),
    )
    if not sensors:
        raise DashboardDataError("Machine history must contain sensor columns")

    model_columns = [
        column
        for column in ("setting_1", "setting_2", "setting_3", "operating_condition")
        if column in machine_rows.columns
    ]
    observations: list[dict[str, Any]] = []
    ordered = machine_rows.sort_values("cycle", kind="stable")
    if window_size is not None and len(ordered) > window_size + 1:
        ordered = pd.concat(
            [ordered.iloc[[0]], ordered.iloc[-window_size:]]
        ).drop_duplicates("cycle", keep="first")
        ordered = ordered.sort_values("cycle", kind="stable")
    for row in ordered.to_dict(orient="records"):
        sensor_values = {sensor: _json_value(row[sensor]) for sensor in sensors}
        observation: dict[str, Any] = {
            "cycle": int(row["cycle"]),
            "sensor_values": sensor_values,
        }
        for column in model_columns:
            observation[column] = _json_value(row[column])
        observations.append(observation)
    return {
        "machine_id": str(machine_ids.iloc[0]),
        "observations": observations,
    }


def _json_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, int | float | str | bool):
        return value
    raise DashboardDataError(
        f"Unsupported telemetry value type: {type(value).__name__}"
    )


class PredictionApi:
    """Small standard-library HTTP client for the Phase 6 API."""

    def __init__(self, base_url: str | None = None, *, timeout: float = 5.0) -> None:
        resolved_url = base_url or os.getenv("API_BASE_URL") or "http://localhost:8000"
        self.base_url = resolved_url.rstrip("/")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        result = self._request("GET", "/health")
        if result.get("model_available") is not True:
            raise ApiUnavailable(
                "API is running, but its configured model is unavailable."
            )
        return result

    def predict(self, machine: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/predict", machine)

    def explain(self, machine: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/explain", machine)

    def schedule(self, machines: list[dict[str, Any]]) -> dict[str, Any]:
        return self._request("POST", "/schedule", {"machines": machines})

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise ApiUnavailable(
                f"Prediction API returned HTTP {error.code} for {path}"
            ) from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise ApiUnavailable(f"Prediction API is unavailable: {error}") from error
        if not isinstance(result, dict):
            raise ApiUnavailable(
                f"Prediction API returned an invalid response for {path}"
            )
        return result
