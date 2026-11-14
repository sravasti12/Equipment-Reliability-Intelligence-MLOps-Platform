"""Explore saved synthetic maintenance-backtest outcomes and assumptions."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from reliability_platform.dashboard.common import render_saved_artifact_error
from reliability_platform.dashboard.data_access import (
    DashboardDataError,
    DashboardPaths,
    load_report,
)

st.set_page_config(page_title="Business Analytics", layout="wide")
st.title("Business analytics")
st.write(
    "Compare the saved chronological maintenance-backtest strategies and their "
    "assumption-dependent simulated outcomes."
)
st.caption("Data source: synthetic industrial fleet (seed 42).")

try:
    report = load_report(
        "synthetic_maintenance", paths=DashboardPaths.from_environment()
    )
except DashboardDataError as error:
    render_saved_artifact_error(error)
else:
    st.warning(
        "These are simulated backtest outcomes under documented cost and "
        "maintenance-effectiveness assumptions; they are not observed savings or "
        "causal estimates."
    )
    comparison = pd.DataFrame(report.get("comparison", []))
    if comparison.empty:
        st.info("The saved backtest report contains no strategy comparison rows.")
    else:
        displayed_columns = [
            column
            for column in (
                "strategy",
                "total_cost",
                "planned_interventions",
                "false_alarm_count",
                "unplanned_failures",
                "prevented_failures_assumed",
                "downtime_hours",
            )
            if column in comparison.columns
        ]
        st.subheader("Strategy comparison")
        st.dataframe(
            comparison.loc[:, displayed_columns],
            use_container_width=True,
            hide_index=True,
        )
        if {"strategy", "total_cost"}.issubset(comparison.columns):
            cost_chart = comparison.set_index("strategy")[["total_cost"]]
            st.subheader("Simulated total cost by strategy")
            st.bar_chart(cost_chart)
            st.caption(
                "Costs are the values recorded in the saved backtest report and "
                "depend on its configured assumptions."
            )

    st.subheader("Saved model and replay context")
    model = report.get("model", {})
    if isinstance(model, dict) and model:
        st.json(model)
    if report.get("mlflow_run_id"):
        st.caption(f"Backtest MLflow run ID: {report['mlflow_run_id']}")

    with st.expander("Cost assumptions and limitations"):
        assumptions = report.get("cost_assumptions")
        if assumptions:
            st.json(assumptions)
        if report.get("maintenance_effectiveness_assumption"):
            st.write(
                "Maintenance-effectiveness assumption: "
                f"{report['maintenance_effectiveness_assumption']}"
            )
        limitations = report.get("limitations")
        if isinstance(limitations, list):
            for limitation in limitations:
                st.markdown(f"- {limitation}")
        elif limitations:
            st.write(limitations)
