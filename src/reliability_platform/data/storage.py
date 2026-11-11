"""Parquet persistence for validated machine-cycle datasets."""

from pathlib import Path

import pandas as pd


def write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    """Write a data frame to Parquet, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", index=False)
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet data frame from disk."""
    if not path.is_file():
        raise FileNotFoundError(f"Parquet dataset not found: {path}")
    return pd.read_parquet(path, engine="pyarrow")
