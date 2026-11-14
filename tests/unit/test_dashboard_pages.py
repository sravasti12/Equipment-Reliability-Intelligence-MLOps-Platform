"""Streamlit AppTest smoke tests using only local fixture artifacts."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from reliability_platform.data.storage import write_parquet


@pytest.fixture
def dashboard_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("RELIABILITY_PLATFORM_ROOT", str(tmp_path))
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:1")
    fleet = pd.DataFrame(
        [
            {
                "machine_id": machine_id,
                "cycle": cycle,
                "sensor_1": float(machine_id + cycle),
                "sensor_2": float(machine_id * 2 + cycle),
                "setting_1": 0.1,
                "setting_2": 0.2,
                "setting_3": 0.3,
                "operating_condition": 1,
                "rul": float(5 - cycle),
            }
            for machine_id in (1, 2)
            for cycle in (1, 2, 3)
        ]
    )
    write_parquet(
        fleet,
        tmp_path / "data" / "processed" / "synthetic_fleet.parquet",
    )
    manifest_path = tmp_path / "reports" / "generated" / "serving_model.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text('{"window_size": 2}', encoding="utf-8")
    return tmp_path


def _run_page(filename: str) -> AppTest:
    page_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "reliability_platform"
        / "dashboard"
        / filename
    )
    return AppTest.from_file(str(page_path), default_timeout=10).run()


def test_fleet_overview_runs_with_fixture_artifacts_and_unavailable_api(
    dashboard_root: Path,
) -> None:
    app = _run_page("app.py")

    assert not app.exception
    assert any("Fleet overview" in title.value for title in app.title)
    assert any("Synthetic industrial fleet" in caption.value for caption in app.caption)
    assert any("API is unavailable" in warning.value for warning in app.warning)


def test_machine_analysis_runs_with_fixture_artifacts_and_unavailable_api(
    dashboard_root: Path,
) -> None:
    app = _run_page("pages/1_Machine_Analysis.py")

    assert not app.exception
    assert any("Individual machine analysis" in title.value for title in app.title)
    assert any("Saved RUL target" in heading.value for heading in app.subheader)
    assert any("API is unavailable" in warning.value for warning in app.warning)


def test_business_analytics_reads_saved_backtest_report(
    dashboard_root: Path,
) -> None:
    report_path = (
        dashboard_root / "reports" / "generated" / "synthetic_maintenance_backtest.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        """
        {
          "comparison": [
            {
              "strategy": "run_to_failure",
              "total_cost": 1000,
              "planned_interventions": 0,
              "false_alarm_count": 0,
              "unplanned_failures": 2,
              "prevented_failures_assumed": 0,
              "downtime_hours": 12
            },
            {
              "strategy": "fixed_interval",
              "total_cost": 800,
              "planned_interventions": 3,
              "false_alarm_count": 1,
              "unplanned_failures": 1,
              "prevented_failures_assumed": 1,
              "downtime_hours": 6
            }
          ],
          "cost_assumptions": {"failure_cost": 500},
          "maintenance_effectiveness_assumption": "fixture assumption",
          "limitations": ["Fixture limitation"],
          "mlflow_run_id": "fixture-run"
        }
        """,
        encoding="utf-8",
    )

    app = _run_page("pages/2_Business_Analytics.py")

    assert not app.exception
    assert any("Business analytics" in title.value for title in app.title)
    assert any("synthetic industrial fleet" in caption.value for caption in app.caption)


def test_model_analytics_reads_saved_evaluation_and_shows_phase8_placeholders(
    dashboard_root: Path,
) -> None:
    report_path = dashboard_root / "reports" / "generated" / "synthetic_advanced.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        """
        {
          "comparison": [
            {
              "model": "random_forest",
              "test_rmse": 12.5,
              "test_mae": 9.0,
              "test_asymmetric_score": 400
            }
          ],
          "models": {
            "random_forest": {"mlflow_run_id": "model-run"}
          },
          "conformal": {
            "target_coverage": 0.9,
            "observed_trajectory_coverage": 1.0,
            "average_interval_width_cycles": 20.0,
            "machines": 4,
            "mlflow_run_id": "conformal-run"
          },
          "model_selection": {
            "selected_rul_model": "random_forest",
            "selection_metric": "validation RMSE"
          }
        }
        """,
        encoding="utf-8",
    )

    app = _run_page("pages/3_Model_Analytics.py")

    assert not app.exception
    assert any("Model analytics" in title.value for title in app.title)
    assert any("Synthetic industrial fleet" in caption.value for caption in app.caption)
    assert any("Drift analysis: Phase 8 placeholder" in item.value for item in app.info)
