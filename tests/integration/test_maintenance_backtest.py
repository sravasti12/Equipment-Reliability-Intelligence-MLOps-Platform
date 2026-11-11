"""End-to-end Phase 5 CLI and MLflow backtest coverage."""

import json
from pathlib import Path

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from reliability_platform.data.cli import main
from reliability_platform.data.storage import write_parquet


def _fleet() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for machine_id in range(1, 31):
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


def test_maintenance_cli_replays_policies_and_logs_results(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = write_parquet(_fleet(), tmp_path / "fleet.parquet")
    config_path = tmp_path / "maintenance.yaml"
    config_path.write_text(
        "\n".join(
            (
                "costs:",
                "  planned_maintenance_cost: 2500",
                "  unplanned_failure_cost: 15000",
                "  downtime_cost_per_hour: 1000",
                "  downtime_hours_per_failure: 24",
                "  maintenance_intervention_cost: 500",
                "  false_alarm_cost: 1500",
                "backtest:",
                "  random_seed: 13",
                "  window_size: 3",
                "  target_coverage: 0.8",
                "  crew_capacity_per_shift: 2",
                "  risk_threshold: 0.1",
                "  urgency_horizon_cycles: 30",
                "  fixed_interval_cycles: 10",
                "  maintenance_effectiveness_cycles: 12",
            )
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "reports" / "maintenance.json"
    tracking_uri = (tmp_path / "mlruns").as_uri()
    experiment_name = "phase5-integration"

    main(
        [
            "backtest-maintenance",
            "--input",
            str(input_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--experiment-name",
            experiment_name,
            "--tracking-uri",
            tracking_uri,
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(report["strategies"]) == {
        "run_to_failure",
        "fixed_interval",
        "ml_informed",
    }
    assert (
        report["split_protocol"]["test_machines_used_for_training_or_calibration"]
        is False
    )
    assert report["risk_policy"]["calibration_machines"] == 5
    assert report["strategies"]["run_to_failure"]["planned_interventions"] == 0
    assert report["strategies"]["run_to_failure"]["unplanned_failures"] == 4
    fixed_actions = pd.DataFrame(report["strategies"]["fixed_interval"]["actions"])
    assert fixed_actions["cycle"].ge(10).all()
    assert fixed_actions.groupby("cycle").size().le(2).all()
    assert fixed_actions.groupby("machine_label")["cycle"].diff().dropna().ge(10).all()
    assert report["strategies"]["ml_informed"]["planned_interventions"] <= 4
    assert all(
        "failure_event" not in action and "rul" not in action
        for result in report["strategies"].values()
        for action in result["actions"]
    )
    assert Path(report["comparison_plot"]).is_file()
    assert report["limitations"]
    assert report["mlflow_run_id"]

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == "FINISHED"
    assert "ml_informed_total_cost" in runs[0].data.metrics
    assert "Wrote maintenance backtest report" in capsys.readouterr().out
