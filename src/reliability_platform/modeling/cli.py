"""Command-line entry point for training and evaluating RUL baselines."""

import argparse
from pathlib import Path

from reliability_platform.data.storage import read_parquet
from reliability_platform.modeling.baselines import save_report, train_baselines


def main(argv: list[str] | None = None) -> None:
    """Train baseline models, log MLflow runs, and write evaluation metrics."""
    parser = argparse.ArgumentParser(
        prog="reliability-platform train-baselines",
        description="Train machine-disjoint RUL baselines and log MLflow metrics.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/baseline_metrics.json"),
    )
    parser.add_argument("--experiment-name", default="reliability-rul-baselines")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=5)
    arguments = parser.parse_args(argv)
    try:
        frame = read_parquet(arguments.input)
        report = train_baselines(
            frame,
            tracking_uri=arguments.tracking_uri,
            experiment_name=arguments.experiment_name,
            random_seed=arguments.seed,
            window_size=arguments.window_size,
            dataset_name=arguments.input.name,
        )
        save_report(report, arguments.output)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(f"Wrote baseline evaluation report to {arguments.output}")
