"""Small-fixture tests for dashboard artifact and API access helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.error import URLError

import pandas as pd
import pytest

import reliability_platform.dashboard.data_access as data_access
from reliability_platform.dashboard.data_access import (
    ApiUnavailable,
    DashboardDataError,
    DashboardPaths,
    PredictionApi,
    available_datasets,
    load_fleet,
    load_report,
    machine_request,
    serving_window_size,
)
from reliability_platform.data.storage import write_parquet
from reliability_platform.modeling.features import engineer_features


@pytest.fixture
def paths(tmp_path: Path) -> DashboardPaths:
    return DashboardPaths(project_root=tmp_path)


def _small_fleet() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "machine_id": 1,
                "cycle": cycle,
                "sensor_1": float(cycle),
                "sensor_2": float(cycle + 10) if cycle != 2 else None,
                "setting_1": 0.1,
                "operating_condition": 1,
                "rul": float(4 - cycle),
                "failure_event": cycle == 4,
                "maintenance_event": False,
            }
            for cycle in (3, 1, 2)
        ]
    )


def test_load_fleet_reads_saved_parquet_and_available_source(
    paths: DashboardPaths,
) -> None:
    source = paths.project_root / "data" / "processed" / "synthetic_fleet.parquet"
    write_parquet(_small_fleet(), source)

    frame = load_fleet("synthetic_fleet", paths=paths)

    assert len(frame) == 3
    assert [spec.dataset_id for spec in available_datasets(paths)] == [
        "synthetic_fleet"
    ]


def test_load_fleet_rejects_missing_or_unknown_datasets(
    paths: DashboardPaths,
) -> None:
    with pytest.raises(DashboardDataError, match="Unknown"):
        load_fleet("unknown", paths=paths)
    with pytest.raises(DashboardDataError, match="Could not read"):
        load_fleet("synthetic_fleet", paths=paths)


def test_load_report_accepts_only_registered_json_artifacts(
    paths: DashboardPaths,
) -> None:
    report_path = paths.report_path("synthetic_reliability")
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps({"machine_count": 1}), encoding="utf-8")

    assert load_report("synthetic_reliability", paths=paths) == {"machine_count": 1}
    with pytest.raises(DashboardDataError, match="Unknown"):
        load_report("../outside", paths=paths)
    with pytest.raises(DashboardDataError, match="Could not read"):
        load_report("synthetic_advanced", paths=paths)


def test_machine_request_sorts_history_and_excludes_labels() -> None:
    frame = _small_fleet()

    request = machine_request(frame)

    assert request["machine_id"] == "1"
    assert [row["cycle"] for row in request["observations"]] == [1, 2, 3]
    assert request["observations"][1]["sensor_values"]["sensor_2"] is None
    assert all(
        not {"rul", "failure_event", "maintenance_event"}.intersection(row)
        for row in request["observations"]
    )


def test_windowed_machine_request_preserves_latest_engineered_features() -> None:
    frame = pd.DataFrame(
        [
            {
                "machine_id": 1,
                "cycle": cycle,
                "sensor_1": float(cycle * cycle),
                "sensor_2": float(cycle + 10),
                "setting_1": 0.1 * cycle,
                "rul": float(10 - cycle),
            }
            for cycle in range(1, 9)
        ]
    )

    request = machine_request(frame, window_size=3)
    compact = pd.DataFrame(
        [
            {
                "machine_id": request["machine_id"],
                "rul": 0.0,
                "cycle": observation["cycle"],
                **observation["sensor_values"],
                **{
                    name: observation[name]
                    for name in ("setting_1", "setting_2", "setting_3")
                    if name in observation
                },
            }
            for observation in request["observations"]
        ]
    )
    full_features = engineer_features(frame, window_size=3).features.iloc[[-1]]
    compact_features = engineer_features(compact, window_size=3).features.iloc[[-1]]

    assert [row["cycle"] for row in request["observations"]] == [1, 6, 7, 8]
    pd.testing.assert_frame_equal(
        full_features.reset_index(drop=True),
        compact_features.reset_index(drop=True),
    )


def test_serving_window_size_reads_saved_manifest(paths: DashboardPaths) -> None:
    manifest_path = paths.report_path("serving_manifest")
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"window_size": 5}), encoding="utf-8")

    assert serving_window_size(paths=paths) == 5


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame(), "must not be empty"),
        (pd.DataFrame({"machine_id": [1], "cycle": [1]}), "sensor columns"),
        (
            pd.DataFrame(
                {
                    "machine_id": [1, 2],
                    "cycle": [1, 1],
                    "sensor_1": [2.0, 3.0],
                }
            ),
            "exactly one machine",
        ),
    ],
)
def test_machine_request_rejects_invalid_histories(
    frame: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(DashboardDataError, match=message):
        machine_request(frame)


class _FakeResponse:
    def __init__(self, value: dict[str, Any]) -> None:
        self.value = value

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.value).encode("utf-8")


def test_prediction_api_posts_json_and_checks_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bytes | None]] = []

    def fake_urlopen(request: Any, *, timeout: float) -> _FakeResponse:
        calls.append((request.full_url, request.data))
        return _FakeResponse({"model_available": True, "model_version": "run-1"})

    monkeypatch.setattr(data_access, "urlopen", fake_urlopen)
    client = PredictionApi("http://example.test")

    assert client.health()["model_version"] == "run-1"
    response = client.schedule([{"machine_id": "m-1"}])

    assert response["model_available"] is True
    assert calls[0][0] == "http://example.test/health"
    assert json.loads(calls[1][1] or b"{}") == {"machines": [{"machine_id": "m-1"}]}


def test_prediction_api_reports_unavailable_service_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_urlopen(*args: object, **kwargs: object) -> None:
        raise URLError("connection refused")

    monkeypatch.setattr(data_access, "urlopen", failed_urlopen)
    with pytest.raises(ApiUnavailable, match="unavailable"):
        PredictionApi("http://example.test").health()

    monkeypatch.setattr(
        data_access,
        "urlopen",
        lambda *args, **kwargs: _FakeResponse({"model_available": False}),
    )
    with pytest.raises(ApiUnavailable, match="model is unavailable"):
        PredictionApi("http://example.test").health()
