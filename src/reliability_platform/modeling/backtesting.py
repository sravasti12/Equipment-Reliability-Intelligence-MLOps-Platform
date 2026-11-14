"""Chronological heuristic maintenance backtests with configurable costs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from reliability_platform.data.validation import validate_fleet
from reliability_platform.modeling.baselines import _baseline_models
from reliability_platform.modeling.conformal import calibrate_trajectory_error
from reliability_platform.modeling.features import (
    FeatureDataset,
    engineer_features,
    machine_identity_columns,
)
from reliability_platform.modeling.scheduling import (
    maintenance_risk,
    schedule_machines,
)
from reliability_platform.modeling.splitting import split_by_machine
from reliability_platform.monitoring.lineage import RunLineage


@dataclass(frozen=True)
class MaintenanceCosts:
    """Illustrative configurable unit costs; these are not industry estimates."""

    planned_maintenance_cost: float = 2500.0
    unplanned_failure_cost: float = 15000.0
    downtime_cost_per_hour: float = 1000.0
    downtime_hours_per_failure: float = 24.0
    maintenance_intervention_cost: float = 500.0
    false_alarm_cost: float = 1500.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{name} must be a non-negative number")
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class MaintenanceBacktestConfig:
    """Reproducible model, risk, crew, interval, and counterfactual settings."""

    costs: MaintenanceCosts = MaintenanceCosts()
    random_seed: int = 42
    window_size: int = 5
    target_coverage: float = 0.8
    crew_capacity_per_shift: int = 5
    risk_threshold: float = 0.5
    urgency_horizon_cycles: int = 30
    fixed_interval_cycles: int = 60
    maintenance_effectiveness_cycles: int = 40

    def __post_init__(self) -> None:
        for name in (
            "window_size",
            "crew_capacity_per_shift",
            "urgency_horizon_cycles",
            "fixed_interval_cycles",
            "maintenance_effectiveness_cycles",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
        ):
            raise ValueError("random_seed must be a non-negative integer")
        if not np.isfinite(self.target_coverage) or not 0 < self.target_coverage < 1:
            raise ValueError("target_coverage must be between 0 and 1")
        if not np.isfinite(self.risk_threshold) or not 0 <= self.risk_threshold <= 1:
            raise ValueError("risk_threshold must be between 0 and 1")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    return value


def _config_number(section: Mapping[str, object], key: str) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"configuration value {key} must be numeric")
    return float(value)


def _config_integer(section: Mapping[str, object], key: str) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"configuration value {key} must be an integer")
    return value


def _as_integer(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not np.isfinite(value)
        or int(value) != value
    ):
        raise ValueError(f"{name} must be a finite integer")
    return int(value)


def _as_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def maintenance_config_from_mapping(value: object) -> MaintenanceBacktestConfig:
    """Parse and validate the costs and backtest sections of a YAML mapping."""
    root = _mapping(value, "maintenance configuration")
    if set(root) != {"costs", "backtest"}:
        raise ValueError("maintenance configuration must contain costs and backtest")
    costs_data = _mapping(root["costs"], "costs")
    backtest_data = _mapping(root["backtest"], "backtest")
    cost_names = set(asdict(MaintenanceCosts()))
    backtest_names = {
        "random_seed",
        "window_size",
        "target_coverage",
        "crew_capacity_per_shift",
        "risk_threshold",
        "urgency_horizon_cycles",
        "fixed_interval_cycles",
        "maintenance_effectiveness_cycles",
    }
    if set(costs_data) != cost_names:
        raise ValueError(f"costs must define exactly: {', '.join(sorted(cost_names))}")
    if set(backtest_data) != backtest_names:
        raise ValueError(
            "backtest must define exactly: " + ", ".join(sorted(backtest_names))
        )
    costs = MaintenanceCosts(
        **{name: _config_number(costs_data, name) for name in sorted(cost_names)}
    )
    return MaintenanceBacktestConfig(
        costs=costs,
        random_seed=_config_integer(backtest_data, "random_seed"),
        window_size=_config_integer(backtest_data, "window_size"),
        target_coverage=_config_number(backtest_data, "target_coverage"),
        crew_capacity_per_shift=_config_integer(
            backtest_data, "crew_capacity_per_shift"
        ),
        risk_threshold=_config_number(backtest_data, "risk_threshold"),
        urgency_horizon_cycles=_config_integer(backtest_data, "urgency_horizon_cycles"),
        fixed_interval_cycles=_config_integer(backtest_data, "fixed_interval_cycles"),
        maintenance_effectiveness_cycles=_config_integer(
            backtest_data, "maintenance_effectiveness_cycles"
        ),
    )


def _predict_test_rows(
    model: Any,
    dataset: FeatureDataset,
    *,
    conformal_error_bound: float,
    urgency_horizon_cycles: int,
) -> pd.DataFrame:
    """Build policy inputs only; targets and observed event labels stay separate."""
    identity = machine_identity_columns(dataset.metadata)
    prediction = np.maximum(
        np.asarray(model.predict(dataset.features), dtype=np.float64),
        0.0,
    )
    lower = np.maximum(0.0, prediction - conformal_error_bound)
    result = dataset.metadata.loc[:, [*identity, "cycle"]].copy()
    result["predicted_rul"] = prediction
    result["conformal_lower_bound"] = lower
    result["risk_score"] = [
        maintenance_risk(
            float(estimate),
            float(bound),
            urgency_horizon_cycles=urgency_horizon_cycles,
        )
        for estimate, bound in zip(prediction, lower, strict=True)
    ]
    return result


def _machine_token(values: tuple[object, ...], columns: list[str]) -> str:
    return json.dumps(
        [[column, str(value)] for column, value in zip(columns, values, strict=True)],
        separators=(",", ":"),
    )


def _machine_label(values: tuple[object, ...], columns: list[str]) -> str:
    return " | ".join(
        f"{column}={value}" for column, value in zip(columns, values, strict=True)
    )


def _policy_inputs(
    predictions: pd.DataFrame,
    identity: list[str],
) -> pd.DataFrame:
    result = predictions.loc[
        :,
        [
            *identity,
            "cycle",
            "predicted_rul",
            "conformal_lower_bound",
            "risk_score",
        ],
    ].copy()
    result["_machine_token"] = [
        _machine_token(tuple(values), identity)
        for values in result.loc[:, identity].itertuples(index=False, name=None)
    ]
    return result


def _simulate_policy(
    policy_inputs: pd.DataFrame,
    *,
    identity: list[str],
    strategy: str,
    config: MaintenanceBacktestConfig,
) -> list[dict[str, object]]:
    """Replay decisions in cycle order without exposing evaluation outcomes."""
    actions: list[dict[str, object]] = []
    serviced: set[str] = set()
    last_service_cycle: dict[str, int] = {}
    for cycle_value, readings in policy_inputs.groupby("cycle", sort=True):
        cycle = _as_integer(cycle_value, "cycle")
        if strategy == "run_to_failure":
            continue
        if strategy == "fixed_interval":
            due_tokens = {
                str(token)
                for token in readings["_machine_token"]
                if cycle - last_service_cycle.get(str(token), 0)
                >= config.fixed_interval_cycles
            }
            candidates = readings.loc[
                readings["_machine_token"].isin(due_tokens)
            ].copy()
            if candidates.empty:
                continue
            candidates["risk_score"] = 1.0
            candidates["conformal_lower_bound"] = 0.0
            risk_threshold = 0.0
        else:
            candidates = readings.loc[~readings["_machine_token"].isin(serviced)].copy()
            if candidates.empty:
                continue
            risk_threshold = config.risk_threshold
        ranked = schedule_machines(
            candidates,
            crew_capacity_per_shift=config.crew_capacity_per_shift,
            risk_threshold=risk_threshold,
        )
        for _, row in ranked.loc[ranked["scheduled"]].iterrows():
            values = tuple(row[column] for column in identity)
            token = str(row["_machine_token"])
            actions.append(
                {
                    "machine_id": str(row["machine_id"]),
                    "machine_label": _machine_label(values, identity),
                    "cycle": cycle,
                    "predicted_rul": float(row["predicted_rul"]),
                    "conformal_lower_bound": float(
                        policy_inputs.loc[
                            policy_inputs["_machine_token"].eq(token)
                            & policy_inputs["cycle"].eq(cycle),
                            "conformal_lower_bound",
                        ].iloc[0]
                    ),
                    "risk_score": float(
                        policy_inputs.loc[
                            policy_inputs["_machine_token"].eq(token)
                            & policy_inputs["cycle"].eq(cycle),
                            "risk_score",
                        ].iloc[0]
                    ),
                    "priority_rank": int(row["priority_rank"]),
                    "_machine_token": token,
                }
            )
            if strategy == "ml_informed":
                serviced.add(token)
            elif strategy == "fixed_interval":
                last_service_cycle[token] = cycle
    return actions


def _observed_outcomes(
    test_frame: pd.DataFrame,
    identity: list[str],
) -> tuple[dict[str, dict[str, object]], int]:
    outcomes: dict[str, dict[str, object]] = {}
    for values, machine in test_frame.groupby(identity, sort=False, dropna=False):
        machine_values = values if isinstance(values, tuple) else (values,)
        token = _machine_token(machine_values, identity)
        machine_label = _machine_label(machine_values, identity)
        last_cycle = int(pd.to_numeric(machine["cycle"]).max())
        failure_rows = (
            machine.loc[machine["failure_event"].astype(bool)]
            if "failure_event" in machine
            else machine.iloc[0:0]
        )
        failure_cycle = (
            int(pd.to_numeric(failure_rows["cycle"]).min())
            if not failure_rows.empty
            else None
        )
        outcomes[token] = {
            "machine_label": machine_label,
            "last_observed_cycle": last_cycle,
            "failure_cycle": failure_cycle,
        }
    return outcomes, len(outcomes)


def _score_strategy(
    actions: list[dict[str, object]],
    outcomes: dict[str, dict[str, object]],
    *,
    config: MaintenanceBacktestConfig,
) -> dict[str, object]:
    false_alarms = 0
    late_interventions = 0
    unverified_interventions = 0
    redundant_interventions = 0
    prevented_failures = 0
    effective_lead_times: list[int] = []
    actions_by_machine: dict[str, list[dict[str, object]]] = {}
    for action in actions:
        token = str(action["_machine_token"])
        actions_by_machine.setdefault(token, []).append(action)
    for token, machine_actions in actions_by_machine.items():
        outcome = outcomes[token]
        failure_cycle = outcome["failure_cycle"]
        if isinstance(failure_cycle, int):
            eligible_actions = [
                action
                for action in machine_actions
                if 0
                < failure_cycle - _as_integer(action["cycle"], "action cycle")
                <= config.maintenance_effectiveness_cycles
            ]
            effective_action = (
                max(
                    eligible_actions,
                    key=lambda action: _as_integer(action["cycle"], "action cycle"),
                )
                if eligible_actions
                else None
            )
            if effective_action is not None:
                prevented_failures += 1
                effective_lead_times.append(
                    failure_cycle
                    - _as_integer(effective_action["cycle"], "action cycle")
                )
            for action in machine_actions:
                lead_time = failure_cycle - _as_integer(action["cycle"], "action cycle")
                if action is effective_action:
                    continue
                if 0 < lead_time <= config.maintenance_effectiveness_cycles:
                    redundant_interventions += 1
                elif lead_time > config.maintenance_effectiveness_cycles:
                    false_alarms += 1
                elif lead_time <= 0:
                    late_interventions += 1
        else:
            for action in machine_actions:
                follow_up = _as_integer(
                    outcome["last_observed_cycle"],
                    "last observed cycle",
                ) - _as_integer(action["cycle"], "action cycle")
                if follow_up >= config.maintenance_effectiveness_cycles:
                    false_alarms += 1
                else:
                    unverified_interventions += 1

    observed_failures = sum(
        isinstance(outcome["failure_cycle"], int) for outcome in outcomes.values()
    )
    censored_machines = len(outcomes) - observed_failures
    unplanned_failures = observed_failures - prevented_failures
    costs = config.costs
    planned_cost = len(actions) * (
        costs.planned_maintenance_cost + costs.maintenance_intervention_cost
    )
    false_alarm_cost = false_alarms * costs.false_alarm_cost
    failure_cost = unplanned_failures * costs.unplanned_failure_cost
    downtime_hours = unplanned_failures * costs.downtime_hours_per_failure
    downtime_cost = downtime_hours * costs.downtime_cost_per_hour
    return {
        "planned_interventions": len(actions),
        "false_alarm_count": false_alarms,
        "false_alarm_rate": (false_alarms / len(actions) if actions else 0.0),
        "late_intervention_count": late_interventions,
        "redundant_intervention_count": redundant_interventions,
        "unverified_intervention_count": unverified_interventions,
        "observed_failures": observed_failures,
        "prevented_failures_assumed": prevented_failures,
        "unplanned_failures": unplanned_failures,
        "censored_machines": censored_machines,
        "downtime_hours": downtime_hours,
        "average_lead_time_cycles": (
            float(np.mean(effective_lead_times)) if effective_lead_times else None
        ),
        "cost_breakdown": {
            "planned_maintenance_and_intervention": planned_cost,
            "false_alarms": false_alarm_cost,
            "unplanned_failures": failure_cost,
            "downtime": downtime_cost,
        },
        "total_cost": planned_cost + false_alarm_cost + failure_cost + downtime_cost,
        "actions": [
            {key: value for key, value in action.items() if key != "_machine_token"}
            for action in actions
        ],
    }


def _comparison_plot(
    strategies: dict[str, dict[str, object]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(strategies)
    labels = [name.replace("_", " ").title() for name in names]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, key, title, color in (
        (axes[0], "total_cost", "Simulated total cost", "#4472C4"),
        (axes[1], "unplanned_failures", "Unplanned failures", "#C0504D"),
        (axes[2], "planned_interventions", "Planned interventions", "#70AD47"),
    ):
        values = [_as_float(strategies[name][key], f"{name}_{key}") for name in names]
        axis.bar(labels, values, color=color)
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=20)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(figure)


def run_maintenance_backtest(
    frame: pd.DataFrame,
    *,
    config: MaintenanceBacktestConfig | None = None,
    tracking_uri: str = "file:./mlruns",
    experiment_name: str = "reliability-maintenance-backtest",
    output_prefix: Path = Path("reports") / "generated" / "maintenance_backtest",
) -> dict[str, object]:
    """Train on development machines and compare policies on held-out histories."""
    if config is None:
        config = MaintenanceBacktestConfig()
    validate_fleet(frame)
    if "rul" not in frame.columns:
        raise ValueError("maintenance backtesting requires RUL labels")
    partitions = split_by_machine(frame, random_seed=config.random_seed)
    identity = machine_identity_columns(frame)
    training = engineer_features(
        partitions.train,
        window_size=config.window_size,
    )
    validation = engineer_features(
        partitions.validation,
        window_size=config.window_size,
    )
    testing = engineer_features(
        partitions.test,
        window_size=config.window_size,
    )

    model = _baseline_models(
        training.features,
        random_seed=config.random_seed,
    )["random_forest"]
    model.fit(training.features, training.target)
    validation_prediction = np.maximum(
        np.asarray(model.predict(validation.features), dtype=np.float64),
        0.0,
    )
    conformal_error_bound, calibration_machines = calibrate_trajectory_error(
        validation.target,
        validation_prediction,
        validation.metadata,
        alpha=1 - config.target_coverage,
    )
    policy_inputs = _policy_inputs(
        _predict_test_rows(
            model,
            testing,
            conformal_error_bound=conformal_error_bound,
            urgency_horizon_cycles=config.urgency_horizon_cycles,
        ),
        identity,
    )
    outcomes, test_machines = _observed_outcomes(partitions.test, identity)

    strategies: dict[str, dict[str, object]] = {}
    for strategy in ("run_to_failure", "fixed_interval", "ml_informed"):
        actions = _simulate_policy(
            policy_inputs,
            identity=identity,
            strategy=strategy,
            config=config,
        )
        strategies[strategy] = _score_strategy(
            actions,
            outcomes,
            config=config,
        )

    comparison = [
        {
            "strategy": name,
            **{
                key: value
                for key, value in metrics.items()
                if key not in {"actions", "cost_breakdown"}
            },
        }
        for name, metrics in strategies.items()
    ]
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plot_path = output_prefix.with_name(f"{output_prefix.name}_comparison.png")
    _comparison_plot(strategies, plot_path)

    identity_counts = {
        name: int(partition.loc[:, identity].drop_duplicates().shape[0])
        for name, partition in (
            ("training", partitions.train),
            ("validation", partitions.validation),
            ("test", partitions.test),
        )
    }
    validation_rmse = float(
        np.sqrt(
            np.mean(
                np.square(
                    validation.target.to_numpy(dtype=np.float64) - validation_prediction
                )
            )
        )
    )
    lineage = RunLineage.from_frame(frame)
    report: dict[str, object] = {
        "dataset_rows": int(len(frame)),
        "test_machines": test_machines,
        "random_seed": config.random_seed,
        "model": "Phase 3 random-forest RUL pipeline",
        "validation_rmse_cycles": validation_rmse,
        "split_protocol": {
            "fractions": {"training": 0.7, "validation": 0.15, "test": 0.15},
            "machines": identity_counts,
            "test_machines_used_for_training_or_calibration": False,
            "decision_features": "current and preceding cycles only",
        },
        "risk_policy": {
            "definition": (
                "clip(1 - conformal lower RUL bound / urgency horizon, 0, 1)"
            ),
            "urgency_horizon_cycles": config.urgency_horizon_cycles,
            "risk_threshold": config.risk_threshold,
            "crew_capacity_per_shift": config.crew_capacity_per_shift,
            "target_coverage": config.target_coverage,
            "trajectory_error_bound_cycles": conformal_error_bound,
            "calibration_machines": calibration_machines,
            "fixed_interval_cycles": config.fixed_interval_cycles,
        },
        "cost_assumptions": asdict(config.costs),
        "maintenance_effectiveness_assumption": {
            "cycles": config.maintenance_effectiveness_cycles,
            "rule": (
                "An intervention is assumed to prevent an observed failure only "
                "when it is scheduled strictly before that failure and within this "
                "many cycles. This is a counterfactual scoring assumption, not a "
                "measured treatment effect."
            ),
        },
        "censoring_policy": (
            "Only explicit failure_event labels count as observed failures. "
            "Censored trajectories are not counted as failures; interventions "
            "without enough follow-up to assess the effectiveness window are "
            "reported as unverified, not false alarms."
        ),
        "policy_replay": {
            "fixed_interval": (
                "Schedules each machine when its interval is due; capacity-deferred "
                "machines remain eligible until serviced."
            ),
            "ml_informed": (
                "Schedules at most one intervention per machine because this "
                "backtest has no post-maintenance counterfactual features."
            ),
            "false_alarm_rate_denominator": (
                "All scheduled interventions; interventions with insufficient "
                "censored follow-up are reported separately as unverified."
            ),
        },
        "strategies": strategies,
        "comparison": comparison,
        "comparison_plot": str(plot_path),
        "limitations": [
            "Maintenance priority is a greedy risk ranking, not a global optimum.",
            (
                "Machines have no dependency or shared-resource constraints beyond "
                "crew capacity."
            ),
            "Crew capacity is fixed per simulated cycle/shift.",
            "Maintenance benefit and costs are configurable synthetic assumptions.",
            (
                "The backtest does not observe post-maintenance counterfactual "
                "telemetry."
            ),
            (
                "Results depend on RUL prediction quality and the selected "
                "operating policy."
            ),
        ],
    }

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="maintenance-policy-backtest") as run:
        lineage.log(training.features)
        mlflow.log_params(
            {
                "model": "Phase 3 random-forest RUL pipeline",
                "random_seed": config.random_seed,
                "window_size": config.window_size,
                "target_coverage": config.target_coverage,
                "crew_capacity_per_shift": config.crew_capacity_per_shift,
                "risk_threshold": config.risk_threshold,
                "fixed_interval_cycles": config.fixed_interval_cycles,
                "maintenance_effectiveness_cycles": (
                    config.maintenance_effectiveness_cycles
                ),
                **{f"cost_{key}": value for key, value in asdict(config.costs).items()},
            }
        )
        mlflow.log_metric("validation_rmse_cycles", validation_rmse)
        for name, result in strategies.items():
            for metric in (
                "total_cost",
                "unplanned_failures",
                "downtime_hours",
                "planned_interventions",
                "false_alarm_rate",
            ):
                mlflow.log_metric(
                    f"{name}_{metric}",
                    _as_float(result[metric], f"{name}_{metric}"),
                )
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(str(plot_path))
        mlflow.log_dict(report, "maintenance_backtest.json")
        report["mlflow_run_id"] = run.info.run_id
    return report
