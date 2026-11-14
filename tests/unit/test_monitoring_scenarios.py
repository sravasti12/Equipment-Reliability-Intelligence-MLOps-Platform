import pandas as pd

from reliability_platform.monitoring.scenarios import (
    evaluate_drift_degradation_scenarios,
)


def _reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    features: list[dict[str, object]] = []
    performance: list[dict[str, object]] = []
    for machine_index in range(12):
        machine_id = f"unit-{machine_index}"
        actual = 80.0 + machine_index
        features.append(
            {
                "source": "synthetic",
                "machine_id": machine_id,
                "sensor_1": float(machine_index),
                "machine_type": f"type-{machine_index % 2}",
            }
        )
        performance.append(
            {
                "source": "synthetic",
                "machine_id": machine_id,
                "actual_rul": actual,
                "predicted_rul": actual + 2.0,
            }
        )
    return pd.DataFrame(features), pd.DataFrame(performance)


def _scenario(
    features: pd.DataFrame,
    performance: pd.DataFrame,
    *,
    shift_machine_type: bool,
    shift_actual_rul: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current_features = features.copy()
    current_performance = performance.copy()
    if shift_machine_type:
        current_features["machine_type"] = "type-shifted"
    if shift_actual_rul:
        current_performance["actual_rul"] += 20.0
    current_performance["window_index"] = 0
    return current_features, current_performance


def test_scenario_report_measures_all_four_drift_degradation_cases() -> None:
    reference_features, reference_performance = _reference()
    scenarios = {
        "no_change": _scenario(
            reference_features,
            reference_performance,
            shift_machine_type=False,
            shift_actual_rul=False,
        ),
        "condition_mix_change": _scenario(
            reference_features,
            reference_performance,
            shift_machine_type=True,
            shift_actual_rul=False,
        ),
        "outcome_degradation": _scenario(
            reference_features,
            reference_performance,
            shift_machine_type=False,
            shift_actual_rul=True,
        ),
        "combined_change": _scenario(
            reference_features,
            reference_performance,
            shift_machine_type=True,
            shift_actual_rul=True,
        ),
    }

    report = evaluate_drift_degradation_scenarios(
        reference_features,
        reference_performance,
        scenarios,
        drift_columns=["sensor_1", "machine_type"],
        bootstrap_resamples=100,
        random_seed=42,
        minimum_machines_for_conclusion=10,
    )

    observed = {row["observed_case"]: row for row in report["scenarios"]}
    assert report["complete_four_case_coverage"] is True
    assert set(observed) == {
        "no_drift_no_degradation",
        "drift_without_degradation",
        "degradation_without_obvious_drift",
        "drift_with_degradation",
    }
    drift_without = observed["drift_without_degradation"]
    assert drift_without["drift_detected"] is True
    assert drift_without["performance_status"] == "no_degradation_detected"
    degraded_without_drift = observed["degradation_without_obvious_drift"]
    assert degraded_without_drift["drift_detected"] is False
    assert degraded_without_drift["performance_status"] == "degradation_supported"
    assert (
        degraded_without_drift["performance_metrics"]["metric_deltas"]["rmse"][
            "ci_lower"
        ]
        > 0
    )
    assert (
        degraded_without_drift["performance_metrics"]["metric_deltas"]["rmse"][
            "estimate"
        ]
        == 16.0
    )


def test_scenario_report_rejects_missing_measured_cases() -> None:
    reference_features, reference_performance = _reference()
    only_baseline = {
        "no_change": _scenario(
            reference_features,
            reference_performance,
            shift_machine_type=False,
            shift_actual_rul=False,
        )
    }

    try:
        evaluate_drift_degradation_scenarios(
            reference_features,
            reference_performance,
            only_baseline,
            drift_columns=["sensor_1", "machine_type"],
            bootstrap_resamples=100,
            minimum_machines_for_conclusion=10,
        )
    except ValueError as error:
        assert "four drift/degradation cases" in str(error)
    else:
        raise AssertionError("incomplete scenario results must be rejected")
