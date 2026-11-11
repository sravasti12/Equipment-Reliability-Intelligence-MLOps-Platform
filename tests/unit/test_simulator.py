"""Tests for reproducible synthetic fleet generation."""

import pandas as pd
import pytest

from reliability_platform.data.simulator import (
    SyntheticFleetConfig,
    generate_synthetic_fleet,
)


def test_generator_is_reproducible_and_contains_failure_labels() -> None:
    config = SyntheticFleetConfig(
        num_machines=8,
        minimum_lifetime_cycles=8,
        maximum_lifetime_cycles=14,
        sensor_count=3,
        random_seed=123,
    )

    first = generate_synthetic_fleet(config)
    second = generate_synthetic_fleet(config)

    pd.testing.assert_frame_equal(first, second)
    assert first["machine_id"].nunique() == 8
    assert first.groupby("machine_id")["failure_event"].sum().eq(1).all()
    assert first.groupby("machine_id")["rul"].min().eq(0).all()
    assert first.groupby("machine_id")["cycle"].max().between(8, 14).all()


def test_generator_retains_configured_sensor_missingness() -> None:
    config = SyntheticFleetConfig(
        num_machines=20,
        minimum_lifetime_cycles=10,
        maximum_lifetime_cycles=10,
        sensor_count=4,
        missing_probability=0.25,
        random_seed=8,
    )

    frame = generate_synthetic_fleet(config)

    assert frame.filter(regex=r"^sensor_").isna().any().any()
    assert frame["maintenance_event"].any()


def test_invalid_generator_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing_probability"):
        SyntheticFleetConfig(missing_probability=1.0)
    with pytest.raises(ValueError, match="sensor_count must be an integer"):
        SyntheticFleetConfig(sensor_count=1.5)
