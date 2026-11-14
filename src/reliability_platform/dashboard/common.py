"""Shared display helpers for dashboard pages."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from reliability_platform.dashboard.data_access import (
    DATASETS,
    DashboardDataError,
    DashboardPaths,
    DatasetSpec,
    available_datasets,
    load_fleet,
)


def render_dataset_selector(
    *,
    key: str,
) -> tuple[DatasetSpec, pd.DataFrame, DashboardPaths] | None:
    """Render a source-labelled selector and load its saved fleet."""
    paths = DashboardPaths.from_environment()
    options = available_datasets(paths)
    if not options:
        st.error(
            "No processed fleet artifacts are available. Generate or mount the "
            "Phase 1 Parquet data before using this page."
        )
        return None

    selected_id = st.selectbox(
        "Fleet dataset",
        options=[spec.dataset_id for spec in options],
        format_func=lambda dataset_id: DATASETS[dataset_id].label,
        key=key,
    )
    spec = DATASETS[selected_id]
    st.caption(f"Data source: {spec.label}.")
    try:
        frame = load_fleet(spec.dataset_id, paths=paths)
    except DashboardDataError as error:
        st.error(str(error))
        return None
    return spec, frame, paths


def render_saved_artifact_error(error: DashboardDataError) -> None:
    """Show an explicit message when an expected saved report is missing."""
    st.info(f"Saved result is not available: {error}")


def metric_value(value: Any, *, digits: int = 2) -> str:
    """Format saved numeric values without inventing substitutes."""
    if value is None:
        return "Not reported"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)
