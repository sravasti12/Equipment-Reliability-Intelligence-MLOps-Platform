from copy import deepcopy
from typing import Any

from reliability_platform.monitoring.recommendation import (
    build_retraining_recommendation,
)


def _reports() -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    drift = [
        {
            "current_sources": ["synthetic"],
            "overall_drift_detected": False,
            "drifted_columns": [],
            "features": [
                {
                    "column": "sensor_1",
                    "value": 0.02,
                    "threshold": 0.1,
                    "drift_detected": False,
                }
            ],
        }
    ]
    performance = {
        "windows": [
            {
                "source": "synthetic",
                "window_index": 0,
                "machines": 12,
                "metrics": {"rmse": 2.0},
                "metric_deltas": {
                    "rmse": {"estimate": 0.0, "ci_lower": -0.5, "ci_upper": 0.5}
                },
                "degradation_status": "no_degradation_detected",
                "evidence_note": "The interval is not above zero.",
            }
        ]
    }
    calibration = {
        "windows": [
            {
                "source": "synthetic",
                "window_index": 0,
                "machines": 12,
                "metrics": {"observed_trajectory_coverage": 0.95},
                "confidence_intervals_95": {
                    "trajectory_coverage": {"ci_lower": 0.92, "ci_upper": 0.98}
                },
                "coverage_status": "coverage_at_or_above_target",
                "evidence_note": "Coverage is at or above target.",
            }
        ]
    }
    data_quality = {
        "windows": [
            {
                "sources": ["synthetic"],
                "window_index": 0,
                "quality_status": "no_thresholds_exceeded",
                "schema_violations": {"missing_columns": []},
                "missingness": [],
                "out_of_range_values": [],
                "stuck_sensors": [],
                "duplicate_timestamp_rows": 0,
            }
        ]
    }
    return drift, performance, calibration, data_quality


def _recommend(
    reports: tuple[
        list[dict[str, Any]],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
) -> dict[str, Any]:
    drift, performance, calibration, data_quality = reports
    return build_retraining_recommendation(
        drift,
        performance,
        calibration,
        data_quality,
    )


def test_recommendation_is_no_action_when_all_evidence_is_clear() -> None:
    report = _recommend(_reports())

    assert report["recommendation"] == "no_action"
    assert report["advisory_only"] is True
    source_report = report["source_recommendations"][0]
    assert source_report["conditions"]["evidence_incomplete_or_inconclusive"] is False
    assert source_report["evidence"]["performance"][0]["rmse_difference"] == {
        "estimate": 0.0,
        "ci_lower": -0.5,
        "ci_upper": 0.5,
    }


def test_drift_alone_recommends_investigation_not_retraining() -> None:
    reports = _reports()
    reports[0][0]["overall_drift_detected"] = True
    reports[0][0]["drifted_columns"] = ["sensor_1"]

    report = _recommend(reports)

    assert report["recommendation"] == "investigate"
    assert "Drift alone is not proof" in report["summary"]


def test_supported_performance_degradation_recommends_retraining() -> None:
    reports = _reports()
    reports[1]["windows"][0]["degradation_status"] = "degradation_supported"
    reports[1]["windows"][0]["metric_deltas"]["rmse"] = {
        "estimate": 4.0,
        "ci_lower": 2.0,
        "ci_upper": 6.0,
    }

    report = _recommend(reports)

    assert report["recommendation"] == "retrain"
    assert (
        report["source_recommendations"][0]["conditions"]["delayed_label_degradation"]
        is True
    )


def test_inconclusive_evidence_recommends_investigation() -> None:
    reports = _reports()
    reports[1]["windows"][0]["degradation_status"] = (
        "inconclusive_insufficient_machines"
    )

    report = _recommend(reports)

    assert report["recommendation"] == "investigate"
    assert (
        report["source_recommendations"][0]["conditions"][
            "evidence_incomplete_or_inconclusive"
        ]
        is True
    )


def test_undercoverage_recommends_retraining_and_sources_stay_separate() -> None:
    reports = deepcopy(_reports())
    drift, performance, calibration, data_quality = reports
    drift.append(
        {
            "current_sources": ["nasa_cmapss"],
            "overall_drift_detected": False,
            "drifted_columns": [],
            "features": [],
        }
    )
    performance["windows"].append(
        {
            **performance["windows"][0],
            "source": "nasa_cmapss",
            "degradation_status": "no_degradation_detected",
        }
    )
    calibration["windows"].append(
        {
            **calibration["windows"][0],
            "source": "nasa_cmapss",
            "coverage_status": "undercoverage_supported",
        }
    )
    data_quality["windows"].append(
        {**data_quality["windows"][0], "sources": ["nasa_cmapss"]}
    )

    report = _recommend(reports)

    by_source = {
        item["source"]: item["recommendation"]
        for item in report["source_recommendations"]
    }
    assert by_source == {"nasa_cmapss": "retrain", "synthetic": "no_action"}
