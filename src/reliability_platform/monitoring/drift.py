"""Reproducible feature-distribution drift reports using Evidently."""

from __future__ import annotations

import math
from collections.abc import Sequence
from importlib.metadata import version

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

_NON_FEATURE_COLUMNS = {
    "machine_id",
    "cycle",
    "rul",
    "failure_event",
    "maintenance_event",
    "source",
    "dataset_split",
}


def _source_labels(frame: pd.DataFrame) -> list[str]:
    if "source" not in frame.columns:
        return ["unspecified"]
    labels = sorted(str(value) for value in frame["source"].dropna().unique())
    return labels or ["unspecified"]


def _validate_threshold(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative value")


def build_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    psi_threshold: float = 0.1,
    missingness_threshold: float = 0.05,
) -> dict[str, object]:
    """Compare reference and current feature distributions using PSI.

    Evidently supplies per-column PSI; missingness changes are reported
    separately so null-rate shifts are not hidden by the distribution test.
    """
    _validate_threshold("psi_threshold", psi_threshold)
    _validate_threshold("missingness_threshold", missingness_threshold)
    if len(reference) < 2 or len(current) < 2:
        raise ValueError("reference and current datasets must each contain 2+ rows")

    selected_columns = (
        list(columns)
        if columns
        else [
            str(column)
            for column in reference.columns
            if column not in _NON_FEATURE_COLUMNS
        ]
    )
    if not selected_columns:
        raise ValueError("at least one feature column is required")
    if len(selected_columns) != len(set(selected_columns)):
        raise ValueError("feature columns must be unique")

    missing_columns = [
        column
        for column in selected_columns
        if column not in reference.columns or column not in current.columns
    ]
    if missing_columns:
        raise ValueError(
            "feature columns must exist in both datasets: "
            + ", ".join(sorted(missing_columns))
        )

    reference_features = reference.loc[:, selected_columns]
    current_features = current.loc[:, selected_columns]
    snapshot = Report(
        [
            DataDriftPreset(
                columns=selected_columns,
                method="psi",
                threshold=psi_threshold,
                include_tests=False,
            )
        ]
    ).run(current_features, reference_features)
    report_data = snapshot.dict()
    value_metrics: dict[str, float] = {}
    for metric in report_data["metrics"]:
        config = metric.get("config", {})
        if config.get("type") != "evidently:metric_v2:ValueDrift":
            continue
        column = config.get("column")
        value = metric.get("value")
        if isinstance(column, str) and isinstance(value, int | float):
            value_metrics[column] = float(value)

    if set(value_metrics) != set(selected_columns):
        raise RuntimeError(
            "Evidently did not return PSI values for every selected feature"
        )

    feature_results: list[dict[str, object]] = []
    for column in selected_columns:
        reference_missing_rate = float(reference_features[column].isna().mean())
        current_missing_rate = float(current_features[column].isna().mean())
        missing_rate_change = current_missing_rate - reference_missing_rate
        psi = value_metrics[column]
        feature_results.append(
            {
                "column": column,
                "method": "psi",
                "value": psi,
                "threshold": psi_threshold,
                "drift_detected": psi > psi_threshold,
                "reference_missing_rate": reference_missing_rate,
                "current_missing_rate": current_missing_rate,
                "missing_rate_change": missing_rate_change,
                "missingness_drift_detected": (
                    abs(missing_rate_change) >= missingness_threshold
                ),
            }
        )

    drifted_columns = [
        result["column"]
        for result in feature_results
        if result["drift_detected"] or result["missingness_drift_detected"]
    ]
    return {
        "schema_version": 1,
        "library": "evidently",
        "library_version": version("evidently"),
        "method": "psi",
        "psi_threshold": psi_threshold,
        "missingness_threshold": missingness_threshold,
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "reference_sources": _source_labels(reference),
        "current_sources": _source_labels(current),
        "features": feature_results,
        "drifted_columns": drifted_columns,
        "overall_drift_detected": bool(drifted_columns),
    }
