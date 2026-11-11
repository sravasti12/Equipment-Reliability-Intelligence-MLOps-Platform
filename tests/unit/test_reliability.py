"""Tests for reusable reliability metrics and unsupported-metric handling."""

import pandas as pd
import pytest

from reliability_platform.evaluation.reliability import (
    calculate_downtime,
    calculate_failure_rates,
    calculate_maintenance_frequency,
    calculate_mtbf,
    machine_lifetimes,
    summarize_lifetimes,
)


def _fleet() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "machine_id": [1, 1, 1, 2, 2, 3],
            "cycle": [1, 2, 3, 1, 2, 1],
            "failure_event": [False, True, True, False, False, True],
            "maintenance_event": [False, True, False, False, True, False],
            "machine_type": ["A", "A", "A", "B", "B", "B"],
            "source": ["synthetic"] * 6,
            "dataset_split": ["synthetic"] * 6,
        }
    )


def test_machine_lifetime_summary_preserves_censoring_and_grouping() -> None:
    summary = machine_lifetimes(_fleet())
    lifetimes = summarize_lifetimes(_fleet())

    assert summary["exposure_cycles"].tolist() == [3, 2, 1]
    assert summary["failure_observed"].tolist() == [True, False, True]
    assert summary["maintenance_events"].tolist() == [1, 1, 0]
    type_b = lifetimes.loc[lifetimes["machine_type"].eq("B")].iloc[0]
    assert type_b["machine_count"] == 2
    assert type_b["failures"] == 1
    assert type_b["censored_machines"] == 1


def test_failure_rate_uses_events_and_exposure_by_group() -> None:
    rates = calculate_failure_rates(_fleet(), group_by=("machine_type",))
    type_a = rates.loc[rates["machine_type"].eq("A")].iloc[0]
    type_b = rates.loc[rates["machine_type"].eq("B")].iloc[0]

    assert type_a["failures"] == 2
    assert type_a["exposure_cycles"] == 3
    assert type_a["machines"] == 1
    assert type_a["failure_rate_per_1000_cycles"] == pytest.approx(2000 / 3)
    assert type_b["failures"] == 1
    assert type_b["exposure_cycles"] == 3


def test_maintenance_frequency_reports_events_per_exposure() -> None:
    result = calculate_maintenance_frequency(_fleet(), group_by=("machine_type",))
    type_a = result.loc[result["machine_type"].eq("A")].iloc[0]
    assert type_a["maintenance_events"] == 1
    assert type_a["events_per_1000_cycles"] == pytest.approx(1000 / 3)


def test_mtbf_is_unavailable_without_recurrent_failure_events() -> None:
    result = calculate_mtbf(_fleet().loc[lambda data: data["machine_id"].ne(1)])
    assert result["available"] is False
    assert result["mtbf_cycles"] is None


def test_mtbf_uses_cycle_gaps_for_recurrent_failure_events() -> None:
    result = calculate_mtbf(_fleet())
    assert result["available"] is True
    assert result["mtbf_cycles"] == 1.0
    assert result["failure_intervals"] == 1


def test_downtime_is_explicitly_unavailable_or_summed() -> None:
    unavailable = calculate_downtime(_fleet())
    measured = calculate_downtime(_fleet().assign(downtime_hours=[0, 2, 0, 0, 1, 0]))

    assert unavailable["available"] is False
    assert unavailable["downtime_hours"] is None
    assert measured["available"] is True
    assert measured["downtime_hours"] == 3.0
    assert measured["event_count"] == 2


def test_invalid_failure_cycle_table_is_rejected() -> None:
    duplicated = pd.concat([_fleet(), _fleet().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        calculate_failure_rates(duplicated)

    unordered = _fleet().copy()
    unordered.loc[1, "cycle"] = 0
    with pytest.raises(ValueError, match="positive finite integer"):
        calculate_failure_rates(unordered)


def test_machine_lifetimes_rejects_invalid_cycle_records() -> None:
    duplicated = pd.concat([_fleet(), _fleet().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        machine_lifetimes(duplicated)

    unordered = _fleet().iloc[[1, 0, 2, 3, 4, 5]]
    with pytest.raises(ValueError, match="strictly increasing"):
        machine_lifetimes(unordered)
