"""API model loading, endpoint contracts, and unavailable-model behavior."""

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from reliability_platform.api.app import ServingRuntime, create_app
from reliability_platform.api.schemas import ApiSettings
from reliability_platform.data.cli import main
from reliability_platform.data.storage import write_parquet


def _fleet() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for machine_id in range(1, 41):
        lifetime = 34 + machine_id % 7
        for cycle in range(1, lifetime + 1):
            degradation = cycle / lifetime
            rows.append(
                {
                    "machine_id": machine_id,
                    "cycle": cycle,
                    "sensor_1": 80.0 + machine_id + 12.0 * degradation,
                    "sensor_2": 40.0 + 0.5 * machine_id + 7.0 * degradation,
                    "sensor_3": 20.0 + 2.0 * degradation + machine_id % 4,
                    "setting_1": float(machine_id % 3),
                    "setting_2": float(machine_id % 4) * 0.2,
                    "setting_3": float(machine_id % 2),
                    "operating_condition": machine_id % 3,
                    "machine_type": f"type_{machine_id % 3}",
                    "source": "synthetic",
                    "dataset_split": "synthetic",
                    "rul": float(lifetime - cycle),
                    "failure_event": cycle == lifetime,
                    "maintenance_event": False,
                }
            )
    return pd.DataFrame(rows)


def _machine_payload(machine_id: str, offset: float = 0.0) -> dict[str, object]:
    return {
        "machine_id": machine_id,
        "observations": [
            {
                "cycle": cycle,
                "sensor_values": {
                    "sensor_1": 82.0 + offset + cycle * 0.3,
                    "sensor_2": 41.0 + offset + cycle * 0.2,
                    "sensor_3": 21.0 + offset + cycle * 0.1,
                },
                "setting_1": 0.1,
                "setting_2": 0.2,
                "setting_3": 0.0,
                "operating_condition": "1",
            }
            for cycle in range(1, 6)
        ],
    }


def test_api_serves_predictions_explanations_and_priorities(
    tmp_path: Path,
) -> None:
    input_path = write_parquet(_fleet(), tmp_path / "fleet.parquet")
    manifest_path = tmp_path / "serving_model.json"
    tracking_uri = (tmp_path / "mlruns").as_uri()
    main(
        [
            "prepare-serving-model",
            "--input",
            str(input_path),
            "--output",
            str(manifest_path),
            "--tracking-uri",
            tracking_uri,
            "--experiment-name",
            "phase6-api-integration",
            "--target-coverage",
            "0.8",
        ]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    settings = ApiSettings(
        tracking_uri=tracking_uri,
        serving_manifest_path=str(manifest_path),
        crew_capacity_per_shift=2,
        risk_threshold=0.0,
        urgency_horizon_cycles=30,
    )
    client = TestClient(create_app(settings))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "service_available": True,
        "model_available": True,
        "model_version": manifest["model_run_id"],
    }

    payload = _machine_payload("machine-api-1")
    prediction = client.post("/predict", json=payload)
    assert prediction.status_code == 200
    prediction_data = prediction.json()
    assert prediction_data["machine_id"] == payload["machine_id"]
    assert prediction_data["predicted_rul"] >= 0
    assert 0 <= prediction_data["risk_score"] <= 1
    assert (
        prediction_data["prediction_interval"]["lower"]
        <= prediction_data["predicted_rul"]
        <= prediction_data["prediction_interval"]["upper"]
    )
    assert prediction_data["model_version"] == manifest["model_run_id"]

    explanation = client.post("/explain", json=payload)
    assert explanation.status_code == 200
    assert explanation.json()["top_contributions"]
    assert explanation.json()["max_abs_additivity_residual"] < 1e-6

    schedule = client.post(
        "/schedule",
        json={
            "machines": [
                _machine_payload(
                    f"machine-api-{index}",
                    offset=0.0 if index == 1 else float(index),
                )
                for index in range(1, 5)
            ]
        },
    )
    assert schedule.status_code == 200
    schedule_data = schedule.json()
    assert schedule_data["model_version"] == manifest["model_run_id"]
    assert len(schedule_data["priorities"]) == 4
    assert sum(row["scheduled"] for row in schedule_data["priorities"]) == 2
    assert [row["priority_rank"] for row in schedule_data["priorities"]] == [
        1,
        2,
        3,
        4,
    ]
    first_scheduled = next(
        row
        for row in schedule_data["priorities"]
        if row["machine_id"] == prediction_data["machine_id"]
    )
    assert first_scheduled["predicted_rul"] == prediction_data["predicted_rul"]
    assert (
        first_scheduled["prediction_interval"] == prediction_data["prediction_interval"]
    )
    assert first_scheduled["risk_score"] == prediction_data["risk_score"]

    duplicate_cycles = _machine_payload("invalid-machine")
    duplicate_cycles["observations"].append(duplicate_cycles["observations"][0])
    assert client.post("/predict", json=duplicate_cycles).status_code == 422


def test_api_reports_missing_model_and_rejects_prediction(
    tmp_path: Path,
) -> None:
    settings = ApiSettings(
        tracking_uri=(tmp_path / "missing-mlruns").as_uri(),
        serving_manifest_path=str(tmp_path / "missing.json"),
        crew_capacity_per_shift=2,
        risk_threshold=0.5,
        urgency_horizon_cycles=30,
    )
    runtime = ServingRuntime(settings)
    client = TestClient(create_app(settings, runtime=runtime))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert health.json()["model_available"] is False
    assert (
        client.post("/predict", json=_machine_payload("machine-1")).status_code == 503
    )
