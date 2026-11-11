"""Command-line entry point for preparing an MLflow API model artifact."""

import argparse
from pathlib import Path

from reliability_platform.data.storage import read_parquet
from reliability_platform.modeling.serving import train_serving_model


def main(argv: list[str] | None = None) -> None:
    """Fit, calibrate, and write the API model manifest."""
    parser = argparse.ArgumentParser(
        prog="reliability-platform prepare-serving-model",
        description=(
            "Train an MLflow random-forest artifact and calibrate its RUL interval."
        ),
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/serving_model.json"),
    )
    parser.add_argument("--experiment-name", default="reliability-platform-serving")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--target-coverage", type=float, default=0.9)
    arguments = parser.parse_args(argv)
    try:
        manifest = train_serving_model(
            read_parquet(arguments.input),
            output_path=arguments.output,
            tracking_uri=arguments.tracking_uri,
            experiment_name=arguments.experiment_name,
            random_seed=arguments.seed,
            window_size=arguments.window_size,
            target_coverage=arguments.target_coverage,
        )
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(
        f"Wrote serving manifest to {arguments.output}; "
        f"MLflow run ID: {manifest.model_run_id}"
    )
