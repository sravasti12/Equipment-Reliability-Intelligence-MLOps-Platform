"""Measured drift-versus-degradation scenario analysis."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from reliability_platform.monitoring.drift import build_drift_report
from reliability_platform.monitoring.performance import build_performance_report

_CASE_BY_MEASUREMENTS = {
    (False, "no_degradation_detected"): "no_drift_no_degradation",
    (True, "no_degradation_detected"): "drift_without_degradation",
    (False, "degradation_supported"): "degradation_without_obvious_drift",
    (True, "degradation_supported"): "drift_with_degradation",
}


def evaluate_drift_degradation_scenarios(
    reference_features: pd.DataFrame,
    reference_performance: pd.DataFrame,
    scenarios: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    *,
    drift_columns: list[str] | None = None,
    psi_threshold: float = 0.1,
    missingness_threshold: float = 0.05,
    bootstrap_resamples: int = 1000,
    random_seed: int = 42,
    minimum_machines_for_conclusion: int = 10,
) -> dict[str, object]:
    """Classify four cases from separate measured drift and performance reports.

    The label for each case is derived from the observed drift result and the
    delayed-label performance status; drift alone cannot imply degradation.
    """
    if not scenarios:
        raise ValueError("at least one measured scenario is required")

    results: list[dict[str, object]] = []
    observed_cases: list[str] = []
    for offset, (scenario_id, (current_features, current_performance)) in enumerate(
        sorted(scenarios.items())
    ):
        if not scenario_id:
            raise ValueError("scenario IDs cannot be empty")
        drift = build_drift_report(
            reference_features,
            current_features,
            columns=drift_columns,
            psi_threshold=psi_threshold,
            missingness_threshold=missingness_threshold,
        )
        performance = build_performance_report(
            reference_performance,
            current_performance,
            bootstrap_resamples=bootstrap_resamples,
            random_seed=random_seed + offset,
            minimum_machines_for_conclusion=minimum_machines_for_conclusion,
        )
        windows = performance["windows"]
        if not isinstance(windows, list) or len(windows) != 1:
            raise ValueError(
                f"scenario {scenario_id!r} must contain exactly one measured window"
            )
        performance_window = windows[0]
        drift_detected = bool(drift["overall_drift_detected"])
        performance_status = str(performance_window["degradation_status"])
        measurements = (drift_detected, performance_status)
        observed_case = _CASE_BY_MEASUREMENTS.get(measurements)
        if observed_case is None:
            observed_case = "inconclusive"
        observed_cases.append(observed_case)
        results.append(
            {
                "scenario_id": scenario_id,
                "observed_case": observed_case,
                "drift_detected": drift["overall_drift_detected"],
                "drifted_columns": drift["drifted_columns"],
                "drift_metrics": drift,
                "performance_status": performance_status,
                "performance_metrics": performance_window,
            }
        )

    expected_cases = set(_CASE_BY_MEASUREMENTS.values())
    if set(observed_cases) != expected_cases or len(observed_cases) != len(
        expected_cases
    ):
        raise ValueError(
            "measured scenarios must demonstrate each of the four drift/degradation "
            f"cases; observed {sorted(observed_cases)}"
        )
    return {
        "schema_version": 1,
        "random_seed": random_seed,
        "bootstrap_resamples": bootstrap_resamples,
        "minimum_machines_for_conclusion": minimum_machines_for_conclusion,
        "classification_rule": (
            "Drift comes only from the feature-distribution report; degradation "
            "comes only from the delayed-label RMSE confidence interval."
        ),
        "complete_four_case_coverage": True,
        "scenarios": results,
    }
