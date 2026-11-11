"""End-to-end baseline training and MLflow logging tests."""

import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from reliability_platform.data.cli import main
from reliability_platform.data.storage import write_parquet
from reliability_platform.modeling.features import engineer_features
from reliability_platform.modeling.splitting import split_by_machine


def _fleet() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for machine_id in range(1, 13):
        lifetime = 12 + machine_id
        for cycle in range(1, lifetime + 1):
            rows.append(
                {
                    "machine_id": machine_id,
                    "cycle": cycle,
                    "sensor_1": float(cycle + machine_id),
                    "sensor_2": float(2 * cycle - machine_id),
                    "setting_1": float(machine_id % 4),
                    "operating_condition": machine_id % 3,
                    "machine_type": f"type_{machine_id % 2}",
                    "source": "synthetic",
                    "dataset_split": "synthetic",
                    "rul": float(lifetime - cycle),
                    "failure_event": cycle == lifetime,
                    "maintenance_event": cycle % 7 == 0,
                }
            )
    return pd.DataFrame(rows)


def test_train_baselines_cli_logs_metrics_and_models(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = write_parquet(_fleet(), tmp_path / "fleet.parquet")
    output_path = tmp_path / "reports" / "baseline_metrics.json"
    tracking_uri = (tmp_path / "mlruns").as_uri()
    experiment_name = "phase3-integration"

    main(
        [
            "train-baselines",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--experiment-name",
            experiment_name,
            "--tracking-uri",
            tracking_uri,
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(report["models"]) == {
        "constant_mean",
        "linear_regression",
        "random_forest",
    }
    assert report["splits"]["train"]["machines"] == 8
    assert report["splits"]["validation"]["machines"] == 2
    assert report["splits"]["test"]["machines"] == 2
    assert report["models"]["random_forest"]["metrics"]["test"]["overall"]["rmse"] >= 0
    assert report["models"]["linear_regression"]["metrics"]["test"]["by_group"][
        "machine_type"
    ]

    mlflow.set_tracking_uri(tracking_uri)
    experiment = MlflowClient(tracking_uri=tracking_uri).get_experiment_by_name(
        experiment_name
    )
    assert experiment is not None
    runs = MlflowClient(tracking_uri=tracking_uri).search_runs(
        [experiment.experiment_id]
    )
    assert len(runs) == 3
    for run in runs:
        assert run.data.metrics["test_rmse"] >= 0
        assert any(name.startswith("test_machine_type_") for name in run.data.metrics)
        assert any(
            name.startswith("test_operating_condition_") for name in run.data.metrics
        )
        assert MlflowClient(tracking_uri=tracking_uri).list_artifacts(
            run.info.run_id, "model"
        )
    test_features = engineer_features(
        split_by_machine(_fleet()).test, window_size=5
    ).features
    logged_model = mlflow.sklearn.load_model(
        f"runs:/{report['models']['random_forest']['run_id']}/model"
    )
    assert np.all(logged_model.predict(test_features) >= 0)
    assert "Wrote baseline evaluation report" in capsys.readouterr().out
