import pandas as pd
import pytest

from reliability_platform.monitoring.drift import build_drift_report


def test_drift_report_detects_feature_and_missingness_shifts() -> None:
    reference = pd.DataFrame(
        {
            "sensor_1": [float(value) for value in range(100)],
            "operating_condition": ["idle"] * 80 + ["load"] * 20,
            "source": ["synthetic"] * 100,
        }
    )
    current = pd.DataFrame(
        {
            "sensor_1": [float(value + 100) for value in range(100)],
            "operating_condition": ["load"] * 80 + ["idle"] * 20,
            "source": ["synthetic"] * 100,
        }
    )
    current.loc[:19, "sensor_1"] = float("nan")

    report = build_drift_report(
        reference,
        current,
        columns=["sensor_1", "operating_condition"],
    )

    assert report["overall_drift_detected"] is True
    assert report["reference_sources"] == ["synthetic"]
    assert report["current_rows"] == 100
    features = {row["column"]: row for row in report["features"]}
    assert features["sensor_1"]["drift_detected"] is True
    assert features["sensor_1"]["missingness_drift_detected"] is True
    assert features["operating_condition"]["drift_detected"] is True


def test_drift_report_returns_no_drift_for_identical_data() -> None:
    reference = pd.DataFrame(
        {
            "sensor_1": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            "operating_condition": ["idle", "load"] * 3,
        }
    )

    report = build_drift_report(reference, reference.copy(), columns=[])

    assert report["overall_drift_detected"] is False
    assert report["drifted_columns"] == []
    assert all(feature["value"] == pytest.approx(0.0) for feature in report["features"])


def test_drift_report_rejects_missing_features_and_small_windows() -> None:
    frame = pd.DataFrame({"sensor_1": [0.0, 1.0]})

    with pytest.raises(ValueError, match="exist in both"):
        build_drift_report(frame, frame, columns=["sensor_2"])
    with pytest.raises(ValueError, match="2\\+ rows"):
        build_drift_report(frame.iloc[:1], frame)
