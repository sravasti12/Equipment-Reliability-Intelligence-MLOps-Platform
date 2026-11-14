import pandas as pd
import pytest

from reliability_platform.monitoring.performance import build_performance_report


def _period(
    prediction_offset: float,
    *,
    machine_count: int = 12,
    window_index: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for machine_index in range(machine_count):
        actual = 100.0 + machine_index
        row: dict[str, object] = {
            "source": "synthetic",
            "machine_id": f"unit-{machine_index}",
            "actual_rul": actual,
            "predicted_rul": actual + prediction_offset,
            "baseline_prediction": actual + 5.0,
            "machine_type": f"type-{machine_index % 2}",
            "operating_condition": machine_index % 3,
        }
        if window_index is not None:
            row["window_index"] = window_index
        rows.append(row)
    return pd.DataFrame(rows)


def test_performance_report_detects_degradation_and_compares_baselines() -> None:
    report = build_performance_report(
        _period(2.0),
        _period(12.0, window_index=0),
        baseline_prediction_columns={"linear_regression": "baseline_prediction"},
        bootstrap_resamples=200,
        random_seed=42,
    )

    window = report["windows"][0]
    assert window["metrics"]["rmse"] == pytest.approx(12.0)
    assert window["reference_metrics"]["rmse"] == pytest.approx(2.0)
    assert window["metric_deltas"]["rmse"]["ci_lower"] > 0
    assert window["degradation_status"] == "degradation_supported"
    assert "operating_conditions" in window["by_group"]
    baseline = window["baseline_comparisons"][0]
    assert baseline["model"] == "linear_regression"
    assert baseline["metric_deltas"]["rmse"]["estimate"] == pytest.approx(7.0)


def test_performance_report_marks_small_machine_evidence_inconclusive() -> None:
    report = build_performance_report(
        _period(2.0, machine_count=3),
        _period(12.0, machine_count=3, window_index=0),
        bootstrap_resamples=100,
        random_seed=123,
        minimum_machines_for_conclusion=10,
    )

    window = report["windows"][0]
    assert window["metric_deltas"]["rmse"]["ci_lower"] > 0
    assert window["degradation_status"] == "inconclusive_insufficient_machines"
    assert "evidence is weak" in window["evidence_note"]


def test_performance_bootstrap_is_reproducible_and_requires_reference() -> None:
    reference = _period(2.0)
    current = _period(8.0, window_index=0)
    first = build_performance_report(
        reference,
        current,
        bootstrap_resamples=100,
        random_seed=17,
    )
    second = build_performance_report(
        reference,
        current,
        bootstrap_resamples=100,
        random_seed=17,
    )
    assert first["windows"] == second["windows"]

    with pytest.raises(ValueError, match="reference period"):
        build_performance_report(reference, current.assign(source="nasa_cmapss"))
