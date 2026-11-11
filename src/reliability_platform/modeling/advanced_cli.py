"""CLI for the Phase 4 model comparison and uncertainty workflow."""

import argparse
from pathlib import Path

from reliability_platform.data.storage import read_parquet
from reliability_platform.modeling.advanced import train_advanced_models
from reliability_platform.modeling.baselines import save_report


def main(argv: list[str] | None = None) -> None:
    """Train and evaluate Phase 4 models, logging each experiment to MLflow."""
    parser = argparse.ArgumentParser(
        prog="reliability-platform train-advanced",
        description=(
            "Compare an LSTM, survival and anomaly models, conformal intervals, "
            "and SHAP explanations on machine-disjoint splits."
        ),
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/advanced_model_comparison.json"),
    )
    parser.add_argument("--experiment-name", default="reliability-phase4-advanced")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--target-coverage", type=float, default=0.9)
    parser.add_argument("--landmark-cycles", type=int, default=20)
    arguments = parser.parse_args(argv)
    try:
        frame = read_parquet(arguments.input)
        report = train_advanced_models(
            frame,
            tracking_uri=arguments.tracking_uri,
            experiment_name=arguments.experiment_name,
            random_seed=arguments.seed,
            window_size=arguments.window_size,
            sequence_length=arguments.sequence_length,
            epochs=arguments.epochs,
            target_coverage=arguments.target_coverage,
            landmark_cycles=arguments.landmark_cycles,
            output_prefix=arguments.output.with_suffix(""),
        )
        save_report(report, arguments.output)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(f"Wrote advanced model comparison to {arguments.output}")
