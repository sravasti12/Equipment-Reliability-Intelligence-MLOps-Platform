"""Tests for machine-level partitions and RUL metric semantics."""

import numpy as np
import pandas as pd
import pytest

from reliability_platform.modeling.metrics import regression_metrics
from reliability_platform.modeling.splitting import split_by_machine


def _frame(machine_count: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": "synthetic",
            "dataset_split": "synthetic",
            "machine_id": [
                machine for machine in range(machine_count) for _ in range(2)
            ],
            "cycle": [1, 2] * machine_count,
        }
    )


def test_machine_split_is_disjoint_reproducible_and_complete() -> None:
    frame = _frame()
    first = split_by_machine(frame, random_seed=7)
    repeated = split_by_machine(frame, random_seed=7)
    partitions = (first.train, first.validation, first.test)

    machine_sets = [set(part["machine_id"]) for part in partitions]
    assert all(
        machine_sets[left].isdisjoint(machine_sets[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )
    assert set.union(*machine_sets) == set(frame["machine_id"])
    assert sum(map(len, partitions)) == len(frame)
    pd.testing.assert_frame_equal(first.train, repeated.train)
    pd.testing.assert_frame_equal(first.validation, repeated.validation)
    pd.testing.assert_frame_equal(first.test, repeated.test)


def test_machine_split_rejects_too_few_machines_and_invalid_fractions() -> None:
    with pytest.raises(ValueError, match="at least three"):
        split_by_machine(_frame(2))
    with pytest.raises(ValueError, match="summing to 1"):
        split_by_machine(_frame(), fractions=(0.8, 0.1, 0.2))
    with pytest.raises(ValueError, match="non-negative"):
        split_by_machine(_frame(), random_seed=-1)


def test_rul_metrics_include_asymmetric_penalty_and_nonnegative_predictions() -> None:
    metrics = regression_metrics(pd.Series([10.0, 10.0]), np.array([12.0, 7.0]))

    assert metrics["samples"] == 2
    assert metrics["rmse"] == pytest.approx(np.sqrt(6.5))
    assert metrics["mae"] == pytest.approx(2.5)
    assert metrics["asymmetric_score"] == pytest.approx(
        np.expm1(0.2) + np.expm1(3 / 13)
    )
    assert regression_metrics(pd.Series([10.0]), np.array([-5.0]))["mae"] == 10.0


def test_rul_metrics_reject_nonfinite_or_misaligned_values() -> None:
    with pytest.raises(ValueError, match="aligned finite"):
        regression_metrics(pd.Series([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="aligned finite"):
        regression_metrics(pd.Series([1.0]), np.array([float("nan")]))
