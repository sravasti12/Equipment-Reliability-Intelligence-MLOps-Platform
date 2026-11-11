"""CLI for the Phase 5 maintenance-policy historical backtest."""

import argparse
from pathlib import Path

import yaml

from reliability_platform.data.cli import _read_config
from reliability_platform.data.storage import read_parquet
from reliability_platform.modeling.backtesting import (
    maintenance_config_from_mapping,
    run_maintenance_backtest,
)
from reliability_platform.modeling.baselines import save_report


def main(argv: list[str] | None = None) -> None:
    """Run the deterministic policy comparison and write its JSON report."""
    parser = argparse.ArgumentParser(
        prog="reliability-platform backtest-maintenance",
        description=(
            "Compare run-to-failure, fixed-interval, and ML-informed maintenance "
            "on machine-disjoint historical trajectories."
        ),
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs") / "maintenance.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports") / "generated" / "maintenance_backtest.json",
    )
    parser.add_argument(
        "--experiment-name",
        default="reliability-maintenance-backtest",
    )
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    arguments = parser.parse_args(argv)
    try:
        frame = read_parquet(arguments.input)
        config = maintenance_config_from_mapping(_read_config(arguments.config))
        report = run_maintenance_backtest(
            frame,
            config=config,
            tracking_uri=arguments.tracking_uri,
            experiment_name=arguments.experiment_name,
            output_prefix=arguments.output.with_suffix(""),
        )
        save_report(report, arguments.output)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    print(f"Wrote maintenance backtest report to {arguments.output}")
