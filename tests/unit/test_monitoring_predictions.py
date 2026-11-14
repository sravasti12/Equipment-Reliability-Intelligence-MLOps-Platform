import pandas as pd
import pytest

from reliability_platform.monitoring.predictions import build_prediction_report


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": "synthetic",
                "window_index": 0,
                "machine_id": "unit-1",
                "predicted_rul": 40.0,
                "risk_score": 0.3,
                "lower_bound": 30.0,
                "upper_bound": 50.0,
            },
            {
                "source": "synthetic",
                "window_index": 0,
                "machine_id": "unit-2",
                "predicted_rul": 20.0,
                "risk_score": 0.7,
                "lower_bound": 10.0,
                "upper_bound": 30.0,
            },
            {
                "source": "synthetic",
                "window_index": 1,
                "machine_id": "unit-1",
                "predicted_rul": 35.0,
                "risk_score": 0.4,
                "lower_bound": 25.0,
                "upper_bound": 45.0,
            },
        ]
    )


def test_prediction_report_summarizes_windows_and_machine_trends() -> None:
    report = build_prediction_report(_predictions())

    assert report["sources"] == ["synthetic"]
    assert len(report["windows"]) == 2
    first_window = report["windows"][0]
    assert first_window["machines"] == 2
    assert first_window["predicted_rul_cycles"]["mean"] == pytest.approx(30.0)
    assert first_window["risk_score"]["mean"] == pytest.approx(0.5)
    assert first_window["interval_width_cycles"]["mean"] == pytest.approx(20.0)
    unit_one_trend = [
        row for row in report["machine_trends"] if row["machine_id"] == "unit-1"
    ]
    assert unit_one_trend[1]["predicted_rul_cycles_change"] == pytest.approx(-5.0)
    assert unit_one_trend[1]["risk_score_change"] == pytest.approx(0.1)


def test_prediction_report_keeps_sources_separate() -> None:
    predictions = _predictions()
    second_source = predictions.iloc[[0]].copy()
    second_source["source"] = "nasa_cmapss"
    predictions = pd.concat([predictions, second_source], ignore_index=True)

    report = build_prediction_report(predictions)

    assert report["sources"] == ["nasa_cmapss", "synthetic"]
    assert {row["source"] for row in report["windows"]} == {
        "nasa_cmapss",
        "synthetic",
    }


@pytest.mark.parametrize(
    "change",
    [
        {"risk_score": 1.1},
        {"lower_bound": 45.0},
        {"window_index": -1},
    ],
)
def test_prediction_report_rejects_invalid_predictions(
    change: dict[str, object],
) -> None:
    predictions = _predictions()
    for column, value in change.items():
        predictions.loc[0, column] = value

    with pytest.raises(ValueError):
        build_prediction_report(predictions)
