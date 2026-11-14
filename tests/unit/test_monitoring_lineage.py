import json
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from reliability_platform.monitoring.lineage import RunLineage


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["synthetic", "synthetic"],
            "machine_id": [1, 1],
            "cycle": [1, 2],
            "sensor_1": [0.1, 0.2],
        }
    )


def test_lineage_fingerprints_dataset_and_feature_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "reliability_platform.monitoring.lineage._code_revision",
        lambda: ("revision-123", "test"),
    )
    monkeypatch.setattr(
        "reliability_platform.monitoring.lineage._git_output",
        lambda *_arguments: "",
    )

    frame = _frame()
    original = RunLineage.from_frame(frame)
    changed = RunLineage.from_frame(frame.assign(sensor_1=[0.1, 0.3]))
    requested_identifier = RunLineage.from_frame(
        frame,
        dataset_identifier="synthetic-fleet-v2",
    )

    assert original.dataset_identifier == "synthetic"
    assert original.dataset_version != changed.dataset_version
    assert requested_identifier.dataset_identifier == "synthetic-fleet-v2"
    assert original.code_revision == "revision-123"
    assert original.working_tree_status == "clean"


def test_lineage_logs_tags_and_schema_artifact(tmp_path: Path) -> None:
    tracking_uri = (tmp_path / "mlruns").as_uri()
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment("test-lineage")
    lineage = RunLineage(
        dataset_identifier="synthetic",
        dataset_version="dataset-hash",
        code_revision="revision-123",
        code_revision_source="test",
        working_tree_status="clean",
    )

    try:
        with mlflow.start_run(experiment_id=experiment_id) as run:
            lineage.log(_frame().loc[:, ["sensor_1", "cycle"]])
        logged = MlflowClient(tracking_uri=tracking_uri).get_run(run.info.run_id)
        artifact_path = MlflowClient(tracking_uri=tracking_uri).download_artifacts(
            run.info.run_id,
            "metadata/lineage.json",
        )
        artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))

        assert logged.data.tags["lineage.dataset_identifier"] == "synthetic"
        assert logged.data.tags["lineage.dataset_version_sha256"] == "dataset-hash"
        assert logged.data.tags["lineage.code_revision"] == "revision-123"
        assert artifact["run_id"] == run.info.run_id
        assert artifact["feature_schema"]["columns"] == [
            {"name": "sensor_1", "dtype": "float64"},
            {"name": "cycle", "dtype": "int64"},
        ]
    finally:
        mlflow.set_tracking_uri(previous_uri)


def test_lineage_requires_an_active_mlflow_run() -> None:
    lineage = RunLineage(
        dataset_identifier="synthetic",
        dataset_version="dataset-hash",
        code_revision="revision-123",
        code_revision_source="test",
        working_tree_status="clean",
    )

    with pytest.raises(RuntimeError, match="active MLflow run"):
        lineage.log(_frame())
