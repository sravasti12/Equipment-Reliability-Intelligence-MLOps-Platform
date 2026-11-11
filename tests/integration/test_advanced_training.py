"""End-to-end Phase 4 model, calibration, survival, anomaly, and SHAP test."""

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
        lifetime = 32 + machine_id % 9
        machine_type = f"type_{machine_id % 3}"
        for cycle in range(1, lifetime + 1):
            degradation = cycle / lifetime
            rows.append(
                {
                    "machine_id": machine_id,
                    "cycle": cycle,
                    "sensor_1": float(100 + machine_id + 15 * degradation),
                    "sensor_2": float(50 + 0.5 * machine_id + 8 * degradation),
                    "sensor_3": float(20 + 2 * degradation + (machine_id % 4)),
                    "sensor_4": float(200 - 10 * degradation + machine_id),
                    "setting_1": float(machine_id % 3),
                    "setting_2": float((machine_id % 4) * 0.2),
                    "setting_3": float(machine_id % 2),
                    "operating_condition": machine_id % 3,
                    "machine_type": machine_type,
                    "source": "synthetic",
                    "dataset_split": "synthetic",
                    "rul": float(lifetime - cycle),
                    "failure_event": cycle == lifetime,
                    "maintenance_event": False,
                }
            )
    return pd.DataFrame(rows)


def test_advanced_cli_runs_models_uncertainty_and_explanations(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = write_parquet(_fleet(), tmp_path / "fleet.parquet")
    output_path = tmp_path / "reports" / "advanced.json"
    tracking_uri = (tmp_path / "mlruns").as_uri()
    experiment_name = "phase4-integration"

    main(
        [
            "train-advanced",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--experiment-name",
            experiment_name,
            "--tracking-uri",
            tracking_uri,
            "--seed",
            "13",
            "--window-size",
            "3",
            "--sequence-length",
            "5",
            "--epochs",
            "2",
            "--target-coverage",
            "0.8",
            "--landmark-cycles",
            "5",
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(report["models"]) == {
        "constant_mean",
        "linear_regression",
        "random_forest",
        "lstm",
    }
    assert report["model_selection"]["selected_rul_model"] in report["models"]
    assert report["split_protocol"]["partitions"]["test"]["machines"] == 4
    assert report["conformal"]["target_coverage"] == 0.8
    assert report["conformal"]["calibration_machines"] == 4
    assert report["conformal"]["average_interval_width_cycles"] > 0
    assert "machine_type" in report["conformal"]
    assert report["survival"]["test"]["weibull_aft"]["integrated_brier_score"] >= 0
    assert (
        0
        <= report["survival"]["test"]["weibull_aft"]["failure_probability_by_horizon"][
            str(report["survival"]["horizons_cycles"][1])
        ]
        <= 1
    )
    assert report["anomaly_detection"]["test_injected_shift"]["roc_auc"] >= 0
    assert report["explainability"]["global_sensor_importance"]
    assert report["explainability"]["max_abs_additivity_residual"] < 1e-6
    assert Path(report["explainability"]["global_plot"]).is_file()
    assert Path(report["explainability"]["individual_plot"]).is_file()
    compared = {item["model"] for item in report["comparison"]}
    assert {
        "random_forest",
        "lstm",
        "weibull_aft",
        "isolation_forest",
    }.issubset(compared)

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) >= 10
    assert all(run.info.status == "FINISHED" for run in runs)
    assert any("test_rmse" in run.data.metrics for run in runs)
    assert any("observed_trajectory_coverage" in run.data.metrics for run in runs)
    assert any("test_roc_auc" in run.data.metrics for run in runs)
    assert "Wrote advanced model comparison" in capsys.readouterr().out
