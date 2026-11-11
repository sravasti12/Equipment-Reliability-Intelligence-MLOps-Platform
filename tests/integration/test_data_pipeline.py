"""Integration tests for the documented synthetic-data workflow."""

import json
from pathlib import Path

import pandas as pd
import pytest

from reliability_platform.data.cli import main
from reliability_platform.data.storage import read_parquet, write_parquet


def test_synthetic_command_generates_validated_parquet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "nested" / "fleet.parquet"

    main(
        [
            "generate-synthetic",
            "--config",
            str(project_root / "configs" / "data.yaml"),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    frame = read_parquet(output)
    assert result["validation"]["machine_count"] == 100
    assert frame["machine_id"].nunique() == 100
    assert frame["source"].eq("synthetic").all()


def test_parquet_round_trip_preserves_missing_sensor_values(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {"machine_id": [1, 1], "cycle": [1, 2], "sensor_1": [1.0, None]}
    )
    path = write_parquet(frame, tmp_path / "nested" / "data.parquet")

    pd.testing.assert_frame_equal(read_parquet(path), frame)
