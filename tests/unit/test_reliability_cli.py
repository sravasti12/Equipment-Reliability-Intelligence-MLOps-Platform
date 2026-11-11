"""Tests for reusable reliability report generation."""

import json
from pathlib import Path

import pandas as pd
import pytest

from reliability_platform.data.cli import main
from reliability_platform.data.storage import write_parquet
from reliability_platform.evaluation.cli import build_reliability_report


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "machine_id": [1, 1, 2],
            "cycle": [1, 2, 1],
            "sensor_1": [2.0, None, 4.0],
            "source": ["synthetic"] * 3,
            "dataset_split": ["synthetic"] * 3,
            "machine_type": ["A", "A", "B"],
            "maintenance_event": [False, True, False],
            "failure_event": [False, True, True],
        }
    )


def test_report_states_mtbf_and_downtime_limits() -> None:
    report = build_reliability_report(_frame())

    assert report["row_count"] == 3
    assert report["machine_count"] == 2
    assert report["sensor_missing_fraction"] == 1 / 3
    assert report["mtbf"]["available"] is False
    assert report["downtime"]["available"] is False
    assert len(report["failure_rates"]) == 2


def test_analyze_reliability_cli_writes_json_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = write_parquet(_frame(), tmp_path / "input.parquet")
    output_path = tmp_path / "reports" / "summary.json"

    main(
        [
            "analyze-reliability",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["machine_count"] == 2
    assert result["mtbf"]["available"] is False
    assert "Wrote reliability report" in capsys.readouterr().out
