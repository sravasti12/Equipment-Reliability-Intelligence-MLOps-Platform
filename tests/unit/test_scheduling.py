"""Tests for deterministic risk ranking and maintenance-policy scoring."""

import pandas as pd
import pytest

from reliability_platform.modeling.backtesting import (
    MaintenanceBacktestConfig,
    MaintenanceCosts,
    _score_strategy,
    maintenance_config_from_mapping,
)
from reliability_platform.modeling.scheduling import (
    maintenance_risk,
    schedule_machines,
)


def test_maintenance_risk_uses_conformal_lower_bound() -> None:
    assert maintenance_risk(50, 0, urgency_horizon_cycles=30) == 1
    assert maintenance_risk(50, 15, urgency_horizon_cycles=30) == 0.5
    assert maintenance_risk(50, 45, urgency_horizon_cycles=30) == 0


def test_scheduler_is_deterministic_and_enforces_capacity() -> None:
    predictions = pd.DataFrame(
        {
            "source": ["synthetic"] * 3,
            "dataset_split": ["test"] * 3,
            "machine_id": [3, 10, 2],
            "predicted_rul": [20.0, 20.0, 20.0],
            "conformal_lower_bound": [6.0, 6.0, 6.0],
            "risk_score": [0.8, 0.8, 0.2],
        }
    )

    first = schedule_machines(
        predictions,
        crew_capacity_per_shift=1,
        risk_threshold=0.5,
    )
    second = schedule_machines(
        predictions.sample(frac=1, random_state=5),
        crew_capacity_per_shift=1,
        risk_threshold=0.5,
    )

    assert first.loc[first["scheduled"], "machine_id"].tolist() == [10]
    assert first.loc[first["deferred"], "machine_id"].tolist() == [3]
    assert first.loc[first["below_risk_threshold"], "machine_id"].tolist() == [2]
    assert first["machine_id"].tolist() == second["machine_id"].tolist()
    assert first["scheduled"].sum() <= 1


def test_scheduler_rejects_duplicate_machine_predictions() -> None:
    predictions = pd.DataFrame(
        {
            "machine_id": [1, 1],
            "predicted_rul": [10.0, 9.0],
            "conformal_lower_bound": [5.0, 4.0],
            "risk_score": [0.8, 0.9],
        }
    )

    with pytest.raises(ValueError, match="one row per machine"):
        schedule_machines(
            predictions,
            crew_capacity_per_shift=1,
            risk_threshold=0.5,
        )


def test_policy_scoring_separates_effective_false_and_unverified_actions() -> None:
    config = MaintenanceBacktestConfig(
        costs=MaintenanceCosts(),
        maintenance_effectiveness_cycles=40,
    )
    actions = [
        {"_machine_token": "effective", "cycle": 60},
        {"_machine_token": "false", "cycle": 10},
        {"_machine_token": "unverified", "cycle": 5},
    ]
    outcomes = {
        "effective": {"failure_cycle": 80, "last_observed_cycle": 80},
        "false": {"failure_cycle": 100, "last_observed_cycle": 100},
        "unverified": {"failure_cycle": None, "last_observed_cycle": 15},
    }

    result = _score_strategy(actions, outcomes, config=config)

    assert result["prevented_failures_assumed"] == 1
    assert result["unplanned_failures"] == 1
    assert result["false_alarm_count"] == 1
    assert result["unverified_intervention_count"] == 1
    assert result["average_lead_time_cycles"] == 20
    assert result["total_cost"] == 49500


def test_maintenance_config_requires_complete_explicit_costs() -> None:
    mapping = {
        "costs": {
            "planned_maintenance_cost": 10,
            "unplanned_failure_cost": 20,
            "downtime_cost_per_hour": 3,
            "downtime_hours_per_failure": 4,
            "maintenance_intervention_cost": 5,
            "false_alarm_cost": 6,
        },
        "backtest": {
            "random_seed": 1,
            "window_size": 5,
            "target_coverage": 0.8,
            "crew_capacity_per_shift": 2,
            "risk_threshold": 0.5,
            "urgency_horizon_cycles": 30,
            "fixed_interval_cycles": 50,
            "maintenance_effectiveness_cycles": 20,
        },
    }

    config = maintenance_config_from_mapping(mapping)

    assert config.costs.planned_maintenance_cost == 10
    assert config.fixed_interval_cycles == 50
    mapping["costs"].pop("false_alarm_cost")
    with pytest.raises(ValueError, match="costs must define exactly"):
        maintenance_config_from_mapping(mapping)
