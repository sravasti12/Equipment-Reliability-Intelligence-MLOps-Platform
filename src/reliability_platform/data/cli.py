"""Command-line entry points for the Phase 1 data pipeline."""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from reliability_platform.data.loaders import load_cmapss
from reliability_platform.data.simulator import (
    SyntheticFleetConfig,
    generate_synthetic_fleet,
)
from reliability_platform.data.storage import write_parquet
from reliability_platform.data.validation import ValidationReport, validate_fleet
from reliability_platform.evaluation.cli import main as reliability_report_main
from reliability_platform.evaluation.notebook import main as eda_notebook_main


def _read_config(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and report malformed configuration explicitly."""
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a YAML mapping: {path}")
    if not all(isinstance(key, str) for key in config):
        raise ValueError(f"configuration keys must be strings: {path}")
    return config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reliability-platform",
        description="Generate or process validated equipment reliability data.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser(
        "generate-synthetic",
        help="generate and validate a reproducible synthetic fleet",
    )
    generate.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    generate.add_argument("--output", type=Path)

    process = commands.add_parser(
        "process-cmapss",
        help="load, validate, and store one NASA C-MAPSS subset",
    )
    process.add_argument("--input-dir", type=Path, required=True)
    process.add_argument(
        "--subset",
        choices=("FD001", "FD002", "FD003", "FD004"),
        required=True,
    )
    process.add_argument("--split", choices=("train", "test"), required=True)
    process.add_argument("--output", type=Path)
    process.add_argument(
        "--max-sensor-missing-fraction",
        type=float,
        default=1.0,
    )
    reliability = commands.add_parser(
        "analyze-reliability",
        help="calculate reliability metrics from processed Parquet data",
    )
    reliability.add_argument("--input", type=Path, required=True)
    reliability.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/reliability_summary.json"),
    )
    eda = commands.add_parser(
        "run-eda",
        help="execute the exploratory analysis notebook and write its outputs",
    )
    eda.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/exploratory_analysis.executed.ipynb"),
    )
    train = commands.add_parser(
        "train-baselines",
        help="train and evaluate reproducible RUL baseline models",
    )
    train.add_argument("--input", type=Path, required=True)
    train.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/baseline_metrics.json"),
    )
    train.add_argument("--experiment-name", default="reliability-rul-baselines")
    train.add_argument("--tracking-uri", default="file:./mlruns")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--window-size", type=int, default=5)
    return parser


def _print_result(output: Path, report: ValidationReport) -> None:
    print(
        json.dumps(
            {"output": str(output), "validation": asdict(report)},
            indent=2,
            sort_keys=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run a data-generation or C-MAPSS processing command."""
    parser = _parser()
    arguments = parser.parse_args(argv)

    if arguments.command == "analyze-reliability":
        reliability_report_main(
            ["--input", str(arguments.input), "--output", str(arguments.output)]
        )
        return
    if arguments.command == "run-eda":
        eda_notebook_main(["--output", str(arguments.output)])
        return
    if arguments.command == "train-baselines":
        from reliability_platform.modeling.cli import (
            main as baseline_training_main,
        )

        baseline_training_main(
            [
                "--input",
                str(arguments.input),
                "--output",
                str(arguments.output),
                "--experiment-name",
                arguments.experiment_name,
                "--tracking-uri",
                arguments.tracking_uri,
                "--seed",
                str(arguments.seed),
                "--window-size",
                str(arguments.window_size),
            ]
        )
        return

    try:
        if arguments.command == "generate-synthetic":
            config = _read_config(arguments.config)
            synthetic_config = SyntheticFleetConfig(**config.get("synthetic", {}))
            validation_config = config.get("validation", {})
            if not isinstance(validation_config, dict):
                raise ValueError("validation configuration must be a YAML mapping")
            paths_config = config.get("paths", {})
            if not isinstance(paths_config, dict):
                raise ValueError("paths configuration must be a YAML mapping")
            configured_output = paths_config.get(
                "synthetic_output",
                "data/processed/synthetic_fleet.parquet",
            )
            if not isinstance(configured_output, str) or not configured_output:
                raise ValueError("paths.synthetic_output must be a non-empty path")
            output = arguments.output or Path(str(configured_output))
            frame = generate_synthetic_fleet(synthetic_config)
            report = validate_fleet(
                frame,
                max_sensor_missing_fraction=float(
                    validation_config.get("max_sensor_missing_fraction", 1.0)
                ),
            )
        else:
            output = arguments.output or Path(
                f"data/processed/cmapss_{arguments.subset}_{arguments.split}.parquet"
            )
            frame = load_cmapss(
                arguments.input_dir,
                arguments.subset,
                arguments.split,
            )
            report = validate_fleet(
                frame,
                max_sensor_missing_fraction=arguments.max_sensor_missing_fraction,
            )

        write_parquet(frame, output)
        _print_result(output, report)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
