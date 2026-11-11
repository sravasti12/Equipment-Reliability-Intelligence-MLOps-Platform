"""Tests for machine-cycle data-quality validation."""

from collections.abc import Callable

import pandas as pd
import pytest

from reliability_platform.data.validation import validate_fleet


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "machine_id": [1, 1, 2],
            "cycle": [1, 2, 1],
            "sensor_1": [1.0, None, 3.0],
            "source": ["synthetic", "synthetic", "synthetic"],
            "dataset_split": ["synthetic", "synthetic", "synthetic"],
            "failure_event": [False, True, True],
        }
    )


def test_validation_reports_missing_sensor_values() -> None:
    report = validate_fleet(_valid_frame(), max_sensor_missing_fraction=0.5)

    assert report.row_count == 3
    assert report.machine_count == 2
    assert report.sensor_columns == ("sensor_1",)
    assert report.missing_by_column == {"sensor_1": 1}
    assert report.sensor_missing_fraction == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.assign(cycle=[1, 1, 1]), "duplicate"),
        (lambda frame: frame.assign(cycle=[2, 1, 1]), "strictly increasing"),
        (lambda frame: frame.assign(cycle=[0, 2, 1]), "positive integers"),
        (lambda frame: frame.assign(sensor_1=[1.0, float("inf"), 3.0]), "infinite"),
    ],
)
def test_invalid_data_is_rejected(
    mutate: Callable[[pd.DataFrame], pd.DataFrame], message: str
) -> None:
    frame = mutate(_valid_frame())

    with pytest.raises(ValueError, match=message):
        validate_fleet(frame)


def test_missingness_limit_and_sensor_ranges_are_enforced() -> None:
    frame = _valid_frame()

    with pytest.raises(ValueError, match="exceeds configured maximum"):
        validate_fleet(frame, max_sensor_missing_fraction=0.1)
    with pytest.raises(ValueError, match="outside"):
        validate_fleet(frame, sensor_ranges={"sensor_1": (0.0, 2.0)})
    with pytest.raises(ValueError, match="finite range"):
        validate_fleet(frame, sensor_ranges={"sensor_1": (float("nan"), 2.0)})


def test_invalid_machine_context_and_event_values_are_rejected() -> None:
    invalid_machine_id = _valid_frame().assign(machine_id=[1, 1, 0])
    with pytest.raises(ValueError, match="positive integer"):
        validate_fleet(invalid_machine_id)

    invalid_setting = _valid_frame().assign(setting_1=[0.1, 0.2, float("nan")])
    with pytest.raises(ValueError, match="setting_1"):
        validate_fleet(invalid_setting)

    invalid_event = _valid_frame().assign(failure_event=[False, None, True])
    with pytest.raises(ValueError, match="failure_event"):
        validate_fleet(invalid_event)
