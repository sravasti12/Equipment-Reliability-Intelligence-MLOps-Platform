"""Tests for causal LSTM windows and machine-level conformal calibration."""

import numpy as np
import pandas as pd
import pytest

from reliability_platform.modeling.conformal import (
    calibrate_trajectory_error,
    interval_metrics,
    prediction_intervals,
)
from reliability_platform.modeling.deep_learning import causal_sequences


def test_lstm_windows_include_only_current_and_past_rows() -> None:
    metadata = pd.DataFrame(
        {
            "source": ["synthetic"] * 5,
            "dataset_split": ["synthetic"] * 5,
            "machine_id": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
        }
    )
    values = np.array([[1], [2], [3], [10], [11]], dtype=np.float32)
    sequences = causal_sequences(values, metadata, window_size=3)

    assert sequences[0, :, 0].tolist() == [1, 1, 1]
    assert sequences[2, :, 0].tolist() == [1, 2, 3]
    assert sequences[3, :, 0].tolist() == [10, 10, 10]
    changed = values.copy()
    changed[2, 0] = 300
    changed_sequences = causal_sequences(changed, metadata, window_size=3)
    np.testing.assert_array_equal(sequences[:2], changed_sequences[:2])


def test_conformal_calibration_uses_machine_maximum_residual() -> None:
    metadata = pd.DataFrame(
        {
            "source": ["s"] * 8,
            "dataset_split": ["train"] * 8,
            "machine_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "machine_type": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "operating_condition": [0, 1] * 4,
        }
    )
    actual = pd.Series([1.0, 3.0, 2.0, 4.0, 3.0, 8.0, 7.0, 9.0])
    predicted = np.zeros(8, dtype=np.float64)

    bound, machine_count = calibrate_trajectory_error(
        actual,
        predicted,
        metadata,
        alpha=0.5,
    )
    lower, upper = prediction_intervals(predicted, error_bound=bound)
    metrics = interval_metrics(
        actual,
        lower,
        upper,
        metadata,
        target_coverage=0.5,
    )

    assert machine_count == 4
    assert bound == 8.0
    assert metrics["machines"] == 4
    assert metrics["average_interval_width_cycles"] == 8.0
    assert len(metrics["machine_type"]) == 2
    assert "operating_conditions" in metrics


def test_conformal_requires_enough_calibration_machines() -> None:
    with pytest.raises(ValueError, match="not enough calibration machines"):
        calibrate_trajectory_error(
            pd.Series([1.0, 2.0]),
            np.zeros(2, dtype=np.float64),
            pd.DataFrame({"machine_id": [1, 2]}),
            alpha=0.05,
        )
