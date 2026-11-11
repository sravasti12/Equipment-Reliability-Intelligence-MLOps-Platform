"""Reproducible synthetic industrial fleet generation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticFleetConfig:
    """Configuration for generating a synthetic machine fleet."""

    num_machines: int = 100
    minimum_lifetime_cycles: int = 80
    maximum_lifetime_cycles: int = 160
    sensor_count: int = 5
    sensor_noise_std: float = 1.0
    missing_probability: float = 0.02
    maintenance_probability: float = 0.65
    maximum_maintenance_events: int = 2
    random_seed: int = 42

    def __post_init__(self) -> None:
        """Reject invalid simulator settings before consuming random state."""
        integer_fields = (
            "num_machines",
            "minimum_lifetime_cycles",
            "maximum_lifetime_cycles",
            "sensor_count",
            "maximum_maintenance_events",
            "random_seed",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be an integer")
        if self.num_machines < 1:
            raise ValueError("num_machines must be at least 1")
        if self.random_seed < 0:
            raise ValueError("random_seed cannot be negative")
        if self.minimum_lifetime_cycles < 2:
            raise ValueError("minimum_lifetime_cycles must be at least 2")
        if self.maximum_lifetime_cycles < self.minimum_lifetime_cycles:
            raise ValueError(
                "maximum_lifetime_cycles must be greater than or equal to "
                "minimum_lifetime_cycles"
            )
        if self.sensor_count < 1:
            raise ValueError("sensor_count must be at least 1")
        if not np.isfinite(self.sensor_noise_std) or self.sensor_noise_std < 0:
            raise ValueError("sensor_noise_std must be finite and non-negative")
        if not 0 <= self.missing_probability < 1:
            raise ValueError("missing_probability must be in [0, 1)")
        if not 0 <= self.maintenance_probability <= 1:
            raise ValueError("maintenance_probability must be in [0, 1]")
        if self.maximum_maintenance_events < 0:
            raise ValueError("maximum_maintenance_events cannot be negative")


def generate_synthetic_fleet(config: SyntheticFleetConfig) -> pd.DataFrame:
    """Generate reproducible cycle-level telemetry for a synthetic fleet.

    Sensor missingness is retained as NaN and reported by data validation; it is
    not imputed. The generated telemetry is simulated and is not industrial data.
    """
    rng = np.random.default_rng(config.random_seed)
    rows: list[dict[str, object]] = []
    machine_types = ("type_a", "type_b", "type_c")

    for machine_id in range(1, config.num_machines + 1):
        lifetime = int(
            rng.integers(
                config.minimum_lifetime_cycles,
                config.maximum_lifetime_cycles + 1,
            )
        )
        cycles = np.arange(1, lifetime + 1)
        machine_type = str(rng.choice(machine_types))
        degradation_shape = float(rng.uniform(0.8, 1.5))
        degradation = np.power(cycles / lifetime, degradation_shape)

        operating_mode = rng.integers(0, 3, size=lifetime)
        setting_1 = operating_mode.astype(float) + rng.normal(0.0, 0.05, lifetime)
        ambient_offset = float(rng.normal(0.0, 0.2))
        setting_2 = ambient_offset + rng.normal(0.0, 0.05, lifetime)
        setting_3 = 0.8 + 0.1 * operating_mode + rng.normal(0.0, 0.03, lifetime)

        baselines = rng.normal(100.0, 20.0, size=config.sensor_count)
        sensitivities = rng.normal(12.0, 5.0, size=config.sensor_count)
        condition_effects = rng.normal(0.0, 2.0, size=(3, config.sensor_count))
        noise = rng.normal(
            0.0,
            config.sensor_noise_std,
            size=(lifetime, config.sensor_count),
        )
        sensor_values = (
            baselines
            + degradation[:, np.newaxis] * sensitivities
            + condition_effects[operating_mode]
            + noise
        )
        missing_mask = rng.random(sensor_values.shape) < config.missing_probability
        sensor_values[missing_mask] = np.nan

        maintenance_cycles: set[int] = set()
        if (
            config.maximum_maintenance_events > 0
            and lifetime > 2
            and rng.random() < config.maintenance_probability
        ):
            event_count = int(
                rng.integers(
                    1,
                    min(config.maximum_maintenance_events, lifetime - 1) + 1,
                )
            )
            selected = rng.choice(cycles[:-1], size=event_count, replace=False)
            maintenance_cycles = {int(cycle) for cycle in selected}

        for index, cycle in enumerate(cycles):
            row: dict[str, object] = {
                "machine_id": machine_id,
                "cycle": int(cycle),
                "setting_1": float(setting_1[index]),
                "setting_2": float(setting_2[index]),
                "setting_3": float(setting_3[index]),
                "operating_condition": int(operating_mode[index]),
                "machine_type": machine_type,
                "rul": int(lifetime - cycle),
                "maintenance_event": int(cycle) in maintenance_cycles,
                "failure_event": int(cycle) == lifetime,
                "source": "synthetic",
                "dataset_split": "synthetic",
            }
            row.update(
                {
                    f"sensor_{sensor_index + 1}": float(
                        sensor_values[index, sensor_index]
                    )
                    for sensor_index in range(config.sensor_count)
                }
            )
            rows.append(row)

    return pd.DataFrame.from_records(rows)
