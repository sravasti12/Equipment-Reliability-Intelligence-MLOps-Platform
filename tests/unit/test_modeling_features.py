"""Causality and target-isolation tests for RUL feature engineering."""

import pandas as pd
import pytest

from reliability_platform.modeling.features import engineer_features


def _machine_frame(machine_id: int = 1, cycles: int = 8) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "machine_id": machine_id,
            "cycle": range(1, cycles + 1),
            "sensor_1": [float(cycle) for cycle in range(1, cycles + 1)],
            "sensor_2": [float(cycle * 10) for cycle in range(1, cycles + 1)],
            "setting_1": 0.1,
            "operating_condition": [cycle % 2 for cycle in range(1, cycles + 1)],
            "machine_type": "type_a",
            "source": "synthetic",
            "dataset_split": "synthetic",
            "rul": [cycles - cycle for cycle in range(1, cycles + 1)],
            "failure_event": [cycle == cycles for cycle in range(1, cycles + 1)],
            "maintenance_event": False,
        }
    )
    return frame


def test_features_use_only_current_and_past_observations() -> None:
    frame = _machine_frame()
    full_features = engineer_features(frame, window_size=3)
    repeated_features = engineer_features(frame, window_size=3)
    truncated_features = engineer_features(frame.iloc[:5], window_size=3)

    pd.testing.assert_frame_equal(full_features.features, repeated_features.features)
    pd.testing.assert_series_equal(full_features.target, repeated_features.target)
    pd.testing.assert_frame_equal(
        full_features.features.iloc[:5],
        truncated_features.features,
        check_dtype=False,
    )
    assert "rul" not in full_features.features
    assert "failure_event" not in full_features.features
    assert "maintenance_event" not in full_features.features
    assert "machine_id" not in full_features.features
    assert pd.isna(full_features.features.loc[0, "sensor_1_trend_3"])
    assert full_features.features.loc[1, "sensor_1_trend_3"] == pytest.approx(1.0)


def test_future_sensor_and_label_changes_do_not_change_earlier_features() -> None:
    original = _machine_frame()
    changed = original.copy()
    changed.loc[changed["cycle"].eq(8), "sensor_1"] = 1_000_000.0
    changed.loc[changed["cycle"].eq(8), "rul"] = 10_000.0

    original_features = engineer_features(original, window_size=4).features
    changed_features = engineer_features(changed, window_size=4).features
    pd.testing.assert_frame_equal(
        original_features.iloc[:7],
        changed_features.iloc[:7],
        check_dtype=False,
    )


def test_sensor_missingness_at_start_does_not_use_a_future_baseline() -> None:
    frame = _machine_frame()
    frame.loc[0, "sensor_1"] = None

    features = engineer_features(frame, window_size=3).features

    assert (
        features.loc[0, "sensor_1_variation"] != features.loc[0, "sensor_1_variation"]
    )
    assert not pd.isna(features.loc[0, "sensor_health_index"])
    assert pd.isna(features.loc[0, "sensor_1_variation"])


@pytest.mark.parametrize("window_size", [0, 1])
def test_invalid_feature_window_is_rejected(window_size: int) -> None:
    with pytest.raises(ValueError, match="window_size"):
        engineer_features(_machine_frame(), window_size=window_size)
