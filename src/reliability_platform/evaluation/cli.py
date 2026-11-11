"""Command-line report generation for reliability analytics."""

import argparse
import json
from pathlib import Path

import pandas as pd

from reliability_platform.data.storage import read_parquet
from reliability_platform.evaluation.reliability import (
    calculate_downtime,
    calculate_failure_rates,
    calculate_maintenance_frequency,
    calculate_mtbf,
    summarize_lifetimes,
)


def build_reliability_report(frame: pd.DataFrame) -> dict[str, object]:
    """Build JSON-ready descriptive reliability metrics for a fleet table."""
    group_by = [
        column
        for column in ("source", "dataset_split", "machine_type")
        if column in frame.columns
    ]
    report: dict[str, object] = {
        "row_count": int(len(frame)),
        "machine_count": int(frame["machine_id"].nunique()),
        "sensor_missing_fraction": float(
            frame.filter(regex=r"^sensor_").isna().to_numpy().mean()
        ),
        "lifetime_summary": json.loads(
            summarize_lifetimes(frame).to_json(orient="records")
        ),
        "failure_rates": json.loads(
            calculate_failure_rates(frame, group_by=group_by).to_json(orient="records")
        ),
        "mtbf": calculate_mtbf(frame),
        "downtime": calculate_downtime(frame),
    }
    if "maintenance_event" in frame.columns:
        report["maintenance_frequency"] = json.loads(
            calculate_maintenance_frequency(frame, group_by=group_by).to_json(
                orient="records"
            )
        )
    else:
        report["maintenance_frequency"] = {
            "available": False,
            "reason": "Dataset does not contain maintenance_event.",
        }
    return report


def main(argv: list[str] | None = None) -> None:
    """Generate a reliability report from a processed Parquet dataset."""
    parser = argparse.ArgumentParser(
        prog="reliability-platform analyze-reliability",
        description=(
            "Summarize observed failure, lifetime, maintenance, and data quality."
        ),
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/reliability_summary.json"),
    )
    arguments = parser.parse_args(argv)
    try:
        frame = read_parquet(arguments.input)
        report = build_reliability_report(frame)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Wrote reliability report to {arguments.output}")
