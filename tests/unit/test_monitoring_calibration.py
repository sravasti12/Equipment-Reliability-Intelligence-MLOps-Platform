import pandas as pd
import pytest

from reliability_platform.monitoring.calibration import build_calibration_report


def _intervals(
    machine_count: int,
    *,
    all_covered: bool,
    cycles_per_machine: int = 2,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for machine_index in range(machine_count):
        for cycle in range(1, cycles_per_machine + 1):
            actual = 100.0 + cycle
            covered = all_covered or machine_index % 2 == 0
            lower = actual - 10.0 if covered else actual + 5.0
            upper = actual + 10.0 if covered else actual + 25.0
            rows.append(
                {
                    "source": "synthetic",
                    "window_index": 0,
                    "machine_id": f"unit-{machine_index}",
                    "cycle": cycle,
                    "actual_rul": actual,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "machine_type": f"type-{machine_index % 2}",
                }
            )
    return pd.DataFrame(rows)


def test_calibration_report_detects_undercoverage_with_enough_machines() -> None:
    report = build_calibration_report(
        _intervals(20, all_covered=False),
        target_coverage=0.9,
        bootstrap_resamples=200,
        random_seed=42,
    )

    window = report["windows"][0]
    assert window["metrics"]["observed_trajectory_coverage"] == pytest.approx(0.5)
    assert window["metrics"]["observed_row_coverage"] == pytest.approx(0.5)
    assert window["confidence_intervals_95"]["trajectory_coverage"]["ci_upper"] < 0.9
    assert window["coverage_status"] == "undercoverage_supported"
    assert window["metrics"]["average_interval_width_cycles"] == pytest.approx(20.0)


def test_calibration_report_keeps_small_machine_evidence_inconclusive() -> None:
    report = build_calibration_report(
        _intervals(3, all_covered=False),
        target_coverage=0.9,
        bootstrap_resamples=100,
        random_seed=7,
        minimum_machines_for_conclusion=10,
    )

    window = report["windows"][0]
    assert "trajectory_coverage" in window["confidence_intervals_95"]
    assert window["coverage_status"] == "inconclusive_insufficient_machines"
    assert "evidence is weak" in window["evidence_note"]


def test_calibration_report_distinguishes_row_and_trajectory_coverage() -> None:
    frame = _intervals(2, all_covered=True)
    frame.loc[
        (frame["machine_id"] == "unit-0") & (frame["cycle"] == 2),
        "lower_bound",
    ] = (
        frame.loc[
            (frame["machine_id"] == "unit-0") & (frame["cycle"] == 2),
            "actual_rul",
        ]
        + 5.0
    )

    report = build_calibration_report(
        frame,
        target_coverage=0.5,
        bootstrap_resamples=50,
        minimum_machines_for_conclusion=2,
    )

    metrics = report["windows"][0]["metrics"]
    assert metrics["observed_row_coverage"] == pytest.approx(0.75)
    assert metrics["observed_trajectory_coverage"] == pytest.approx(0.5)
