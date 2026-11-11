"""Deterministic machine-disjoint train, validation, and test partitions."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from reliability_platform.modeling.features import machine_identity_columns


@dataclass(frozen=True)
class MachinePartitions:
    """Rows assigned to three disjoint machine-level partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _partition_sizes(
    machine_count: int, fractions: tuple[float, float, float]
) -> npt.NDArray[np.int64]:
    if machine_count < 3:
        raise ValueError("at least three machines are required for three partitions")
    values = np.asarray(fractions, dtype=float)
    if (
        values.shape != (3,)
        or not np.isfinite(values).all()
        or (values <= 0).any()
        or not np.isclose(values.sum(), 1.0)
    ):
        raise ValueError("fractions must be three positive values summing to 1")
    exact = values * machine_count
    sizes: npt.NDArray[np.int64] = np.floor(exact).astype(np.int64)
    for index in np.argsort(-(exact - sizes))[: machine_count - sizes.sum()]:
        sizes[index] += 1
    for empty_index in np.flatnonzero(sizes == 0):
        donor = int(np.argmax(sizes))
        if sizes[donor] <= 1:
            raise ValueError("each partition must contain at least one machine")
        sizes[donor] -= 1
        sizes[empty_index] += 1
    return sizes


def split_by_machine(
    frame: pd.DataFrame,
    *,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    random_seed: int = 42,
) -> MachinePartitions:
    """Split whole machine trajectories without sharing a machine across sets.

    When different sources or dataset splits are combined, their corresponding
    labels are part of the machine identity because their IDs may be reused.
    """
    identity = list(machine_identity_columns(frame))
    if frame.empty or frame[identity].isna().any().any():
        raise ValueError("dataset must contain non-empty machine identities")
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
        or random_seed < 0
    ):
        raise ValueError("random_seed must be a non-negative integer")
    machines = frame.loc[:, identity].drop_duplicates(ignore_index=True)
    sizes = _partition_sizes(len(machines), fractions)
    shuffled = np.random.default_rng(random_seed).permutation(len(machines))
    assignments = np.full(len(machines), "", dtype=object)
    names = ("train", "validation", "test")
    offset = 0
    for name, size_value in zip(names, sizes, strict=True):
        size = int(size_value)
        assignments[shuffled[offset : offset + size]] = name
        offset += size

    group_ids = frame.groupby(identity, sort=False, dropna=False).ngroup().to_numpy()
    row_assignments = assignments[group_ids]
    return MachinePartitions(
        train=frame.loc[row_assignments == "train"].copy(),
        validation=frame.loc[row_assignments == "validation"].copy(),
        test=frame.loc[row_assignments == "test"].copy(),
    )
