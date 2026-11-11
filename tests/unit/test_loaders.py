"""Tests for NASA C-MAPSS text loading and target construction."""

from pathlib import Path

import pytest

from reliability_platform.data.loaders import load_cmapss


def _write_cmapss_file(path: Path, machine_cycles: dict[int, int]) -> None:
    rows = []
    for machine_id, final_cycle in machine_cycles.items():
        for cycle in range(1, final_cycle + 1):
            values = [
                machine_id,
                cycle,
                0.1,
                0.2,
                0.3,
                *[float(sensor) for sensor in range(1, 22)],
            ]
            rows.append(" ".join(map(str, values)))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_load_train_derives_rul_and_failure_event(tmp_path: Path) -> None:
    _write_cmapss_file(tmp_path / "train_FD001.txt", {1: 2, 2: 3})

    frame = load_cmapss(tmp_path, "FD001", "train")

    assert frame.loc[frame["machine_id"].eq(1), "rul"].tolist() == [1, 0]
    assert frame.loc[frame["machine_id"].eq(2), "rul"].tolist() == [2, 1, 0]
    assert frame.groupby("machine_id")["failure_event"].sum().eq(1).all()
    assert frame.filter(regex=r"^sensor_").shape[1] == 21


def test_load_test_uses_final_rul_labels_without_marking_failure(
    tmp_path: Path,
) -> None:
    _write_cmapss_file(tmp_path / "test_FD001.txt", {1: 2, 2: 3})
    (tmp_path / "RUL_FD001.txt").write_text("5\n7\n", encoding="utf-8")

    frame = load_cmapss(tmp_path, "FD001", "test")

    assert frame.loc[frame["machine_id"].eq(1), "rul"].tolist() == [6, 5]
    assert frame.loc[frame["machine_id"].eq(2), "rul"].tolist() == [9, 8, 7]
    assert not frame["failure_event"].any()


def test_loader_rejects_missing_or_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="data file"):
        load_cmapss(tmp_path, "FD001", "train")

    (tmp_path / "train_FD001.txt").write_text("1 2 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 26"):
        load_cmapss(tmp_path, "FD001", "train")
