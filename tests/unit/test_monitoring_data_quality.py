import pandas as pd
import pytest

from reliability_platform.monitoring.data_quality import build_data_quality_report


def _reference() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "machine_id": [1, 1, 1, 1],
            "cycle": [1, 2, 3, 4],
            "sensor_1": [0.0, 1.0, 2.0, 3.0],
            "sensor_2": [10.0, 11.0, 12.0, 13.0],
            "source": ["synthetic"] * 4,
        }
    )


def test_data_quality_report_detects_windowed_quality_issues() -> None:
    reference = _reference()
    incoming = pd.DataFrame(
        {
            "window_index": [0, 0, 0, 0],
            "machine_id": [1, 1, 1, 1],
            "cycle": [5, 6, 7, 7],
            "sensor_1": [3.0, 3.0, 3.0, 3.0],
            "sensor_2": [11.0, 12.0, 13.0, 20.0],
            "source": ["synthetic"] * 4,
        }
    )

    report = build_data_quality_report(
        reference,
        incoming,
        stuck_value_fraction=0.75,
        minimum_stuck_observations=4,
    )

    window = report["windows"][0]
    assert window["quality_status"] == "issues_detected"
    assert window["duplicate_timestamp_rows"] == 1
    assert window["out_of_range_values"] == [
        {
            "column": "sensor_2",
            "values": 1,
            "reference_minimum": 10.0,
            "reference_maximum": 13.0,
        }
    ]
    assert window["stuck_sensors"] == [
        {
            "machine_id": "1",
            "column": "sensor_1",
            "observations": 4,
            "repeated_value_fraction": 1.0,
        }
    ]


def test_data_quality_report_records_schema_and_missingness() -> None:
    reference = _reference()
    incoming = pd.DataFrame(
        {
            "window_index": [0, 0, 0],
            "machine_id": [1, 1, 1],
            "cycle": [5, 6, 7],
            "sensor_1": [1.0, None, "invalid"],
            "sensor_3": [1.0, 2.0, 3.0],
            "source": ["synthetic"] * 3,
        }
    )

    report = build_data_quality_report(reference, incoming, max_missing_fraction=0.25)

    window = report["windows"][0]
    assert window["schema_violations"]["missing_columns"] == ["sensor_2"]
    assert window["schema_violations"]["unexpected_columns"] == ["sensor_3"]
    assert window["schema_violations"]["non_numeric_sensor_values"] == [
        {"column": "sensor_1", "invalid_values": 1}
    ]
    missingness = {item["column"]: item for item in window["missingness"]}
    assert missingness["sensor_1"]["missing_fraction"] == pytest.approx(1 / 3)
    assert missingness["sensor_1"]["threshold_exceeded"] is True
    assert missingness["sensor_2"]["missing_fraction"] == 1.0


def test_data_quality_report_rejects_invalid_windows_and_thresholds() -> None:
    reference = _reference()
    incoming = reference.assign(window_index=0)
    incoming.loc[0, "window_index"] = -1

    with pytest.raises(ValueError, match="non-negative integers"):
        build_data_quality_report(reference, incoming)
    with pytest.raises(ValueError, match="stuck_value_fraction"):
        build_data_quality_report(
            reference,
            reference.assign(window_index=0),
            stuck_value_fraction=0,
        )
