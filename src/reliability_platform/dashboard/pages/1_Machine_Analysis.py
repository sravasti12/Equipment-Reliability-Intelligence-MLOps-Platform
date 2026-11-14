"""Inspect a single machine's saved history and, when available, API output."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from reliability_platform.dashboard.common import render_dataset_selector
from reliability_platform.dashboard.data_access import (
    ApiUnavailable,
    DashboardDataError,
    PredictionApi,
    machine_request,
    serving_window_size,
)

st.set_page_config(page_title="Machine Analysis", layout="wide")
st.title("Individual machine analysis")
st.write(
    "Inspect one machine's saved history. The charts always use the selected "
    "artifact; API predictions are available only for the synthetic fleet."
)

selection = render_dataset_selector(key="machine-analysis-dataset")
if selection is not None:
    spec, frame, paths = selection
    machine_ids = sorted(frame["machine_id"].astype(str).unique())
    selected_machine_id = st.selectbox(
        "Machine",
        options=machine_ids,
        key="machine-analysis-machine",
    )
    machine_rows = frame.loc[
        frame["machine_id"].astype(str) == selected_machine_id
    ].sort_values("cycle")

    sensor_columns = sorted(
        (column for column in machine_rows.columns if column.startswith("sensor_")),
        key=lambda name: int(name.removeprefix("sensor_"))
        if name.removeprefix("sensor_").isdigit()
        else name,
    )
    if sensor_columns:
        selected_sensor = st.selectbox(
            "Sensor",
            options=sensor_columns,
            key="machine-analysis-sensor",
        )
        sensor_history = machine_rows.set_index("cycle")[[selected_sensor]]
        st.subheader(f"{selected_sensor} history")
        st.line_chart(sensor_history)
    else:
        st.info("This saved dataset contains no sensor columns to chart.")

    if "rul" in machine_rows.columns:
        st.subheader("Saved RUL target")
        st.line_chart(machine_rows.set_index("cycle")[["rul"]])
        st.caption(
            "RUL is a saved training/evaluation target, not a prediction or "
            "observable live measurement."
        )
    else:
        st.info("This dataset has no saved RUL target for this trajectory.")

    if spec.dataset_id != "synthetic_fleet":
        st.info(
            "The configured Phase 6 serving model is trained on synthetic fleet "
            "data. No NASA C-MAPSS serving model is configured; no NASA rows were "
            "sent to the API."
        )
    else:
        st.subheader("Serving API result")
        try:
            api = PredictionApi(timeout=30.0)
            api.health()
            payload = machine_request(
                machine_rows,
                window_size=serving_window_size(paths=paths),
            )
            prediction = api.predict(payload)
            explanation = api.explain(payload)
        except ApiUnavailable as error:
            st.warning(
                "The prediction API is unavailable for this machine. Saved "
                f"telemetry and charts remain available. Details: {error}"
            )
        except DashboardDataError as error:
            st.warning(
                "Could not prepare telemetry for the prediction API. Saved "
                f"telemetry and charts remain available. Details: {error}"
            )
        else:
            left, middle, right = st.columns(3)
            left.metric("Predicted RUL (cycles)", f"{prediction['predicted_rul']:.2f}")
            interval = prediction["prediction_interval"]
            middle.metric(
                "Saved-model interval",
                f"[{interval['lower']:.2f}, {interval['upper']:.2f}]",
            )
            right.metric("API risk score", f"{prediction['risk_score']:.3f}")
            st.caption(
                f"Serving model run ID: {prediction['model_version']}. "
                "The risk score is model-derived, not an observed failure probability."
            )
            contributions = pd.DataFrame(explanation.get("top_contributions", []))
            if not contributions.empty:
                st.subheader("Top explanation contributions")
                st.dataframe(contributions, use_container_width=True, hide_index=True)
                st.caption(
                    "SHAP contributions explain the current API prediction; they "
                    "are not causal effects."
                )
