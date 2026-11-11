"""Load NASA C-MAPSS text files into a consistent machine-cycle table."""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

CMAPSS_SUBSETS = ("FD001", "FD002", "FD003", "FD004")
CMAPSS_SETTING_COLUMNS = ("setting_1", "setting_2", "setting_3")
CMAPSS_SENSOR_COLUMNS = tuple(f"sensor_{index}" for index in range(1, 22))
CMAPSS_COLUMNS = (
    "machine_id",
    "cycle",
    *CMAPSS_SETTING_COLUMNS,
    *CMAPSS_SENSOR_COLUMNS,
)


def load_cmapss(
    input_dir: Path,
    subset: str,
    split: Literal["train", "test"],
) -> pd.DataFrame:
    """Load one C-MAPSS subset and derive its cycle-level RUL targets.

    For test data, NASA's RUL file is joined by the sorted engine identifier;
    these labels are targets for evaluation and must not be used as input features.
    Test trajectories remain marked as not-observed failures.
    """
    normalized_subset = subset.upper()
    if normalized_subset not in CMAPSS_SUBSETS:
        raise ValueError(
            f"unsupported C-MAPSS subset {subset!r}; choose from "
            f"{', '.join(CMAPSS_SUBSETS)}"
        )
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")

    data_path = input_dir / f"{split}_{normalized_subset}.txt"
    if not data_path.is_file():
        raise FileNotFoundError(f"C-MAPSS data file not found: {data_path}")

    frame = pd.read_csv(data_path, sep=r"\s+", header=None, engine="python")
    if frame.shape[1] != len(CMAPSS_COLUMNS):
        raise ValueError(
            f"{data_path} has {frame.shape[1]} columns; expected {len(CMAPSS_COLUMNS)}"
        )
    frame = frame.set_axis(pd.Index(CMAPSS_COLUMNS), axis="columns")

    for column in ("machine_id", "cycle"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if (
            numeric.isna().any()
            or not np.equal(
                numeric.to_numpy(dtype=float), np.floor(numeric.to_numpy(dtype=float))
            ).all()
        ):
            raise ValueError(f"{data_path} contains invalid {column} values")
        frame[column] = numeric.astype("int64")

    frame["source"] = "nasa_cmapss"
    frame["dataset_split"] = split
    frame["maintenance_event"] = False
    frame["failure_event"] = False

    last_cycles = frame.groupby("machine_id", sort=True)["cycle"].max()
    if split == "train":
        frame["rul"] = frame["machine_id"].map(last_cycles) - frame["cycle"]
        frame["failure_event"] = frame["cycle"].eq(frame["machine_id"].map(last_cycles))
    else:
        rul_path = input_dir / f"RUL_{normalized_subset}.txt"
        if not rul_path.is_file():
            raise FileNotFoundError(f"C-MAPSS RUL label file not found: {rul_path}")
        labels = pd.read_csv(rul_path, sep=r"\s+", header=None, engine="python")
        if labels.shape[1] != 1:
            raise ValueError(f"{rul_path} must contain exactly one RUL value per row")
        test_rul = pd.to_numeric(labels.iloc[:, 0], errors="coerce")
        if (
            test_rul.isna().any()
            or not np.isfinite(test_rul.to_numpy(dtype=float)).all()
            or test_rul.lt(0).any()
        ):
            raise ValueError(f"{rul_path} contains invalid RUL labels")
        machine_ids = list(last_cycles.index)
        if len(machine_ids) != len(test_rul):
            raise ValueError(
                f"{rul_path} has {len(test_rul)} labels for {len(machine_ids)} "
                "test engines"
            )
        final_rul_by_machine = dict(zip(machine_ids, test_rul.tolist(), strict=True))
        frame["rul"] = (
            frame["machine_id"].map(last_cycles)
            - frame["cycle"]
            + frame["machine_id"].map(final_rul_by_machine)
        )

    return frame.loc[
        :,
        [
            *CMAPSS_COLUMNS,
            "rul",
            "source",
            "dataset_split",
            "maintenance_event",
            "failure_event",
        ],
    ].sort_values(["machine_id", "cycle"], ignore_index=True)
