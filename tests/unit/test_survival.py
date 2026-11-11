"""Censoring and landmark feature tests for survival data preparation."""

import pandas as pd
import pytest

from reliability_platform.modeling.survival import build_landmark_dataset


def _survival_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for machine_id, lifetime in ((1, 8), (2, 9)):
        for cycle in range(1, lifetime + 1):
            rows.append(
                {
                    "machine_id": machine_id,
                    "cycle": cycle,
                    "sensor_1": float(cycle),
                    "sensor_2": float(machine_id * cycle),
                    "setting_1": 0.1,
                    "machine_type": "A",
                    "source": "synthetic",
                    "dataset_split": "synthetic",
                    "failure_event": machine_id == 1 and cycle == lifetime,
                }
            )
    return pd.DataFrame(rows)


def test_landmark_features_do_not_read_beyond_time_origin_and_preserve_censoring() -> (
    None
):
    frame = _survival_frame()
    changed = frame.copy()
    changed.loc[changed["cycle"].gt(4), "sensor_1"] = 1_000_000.0

    original = build_landmark_dataset(frame, landmark_cycles=4)
    altered = build_landmark_dataset(changed, landmark_cycles=4)

    pd.testing.assert_frame_equal(original.features, altered.features)
    assert original.duration.tolist() == [8.0, 9.0]
    assert original.event_observed.tolist() == [True, False]


def test_landmark_requires_full_prefix_for_each_machine() -> None:
    frame = _survival_frame()
    short = frame.loc[~((frame["machine_id"] == 2) & (frame["cycle"] > 2))]
    with pytest.raises(ValueError, match="fewer than"):
        build_landmark_dataset(short, landmark_cycles=4)
