"""Fleet overview dashboard for saved equipment reliability outputs."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from reliability_platform.dashboard.common import (
    render_dataset_selector,
    render_saved_artifact_error,
)
from reliability_platform.dashboard.data_access import (
    ApiUnavailable,
    DashboardDataError,
    PredictionApi,
    load_report,
    machine_request,
    serving_window_size,
)

st.set_page_config(page_title="Fleet Reliability Overview", layout="wide")
st.title("Fleet overview")
st.write(
    "Explore saved telemetry and reliability outputs. Live maintenance priorities "
    "are requested from the Phase 6 API only for the synthetic fleet."
)

selection = render_dataset_selector(key="fleet-overview-dataset")
if selection is not None:
    spec, frame, paths = selection
    machine_count = frame["machine_id"].nunique()
    cycle_count = frame["cycle"].nunique()
    sensor_columns = sorted(
        (column for column in frame.columns if column.startswith("sensor_")),
        key=lambda name: int(name.removeprefix("sensor_"))
        if name.removeprefix("sensor_").isdigit()
        else name,
    )
    left, middle, right = st.columns(3)
    left.metric("Machines", f"{machine_count:,}")
    middle.metric("Machine-cycle rows", f"{len(frame):,}")
    right.metric("Sensors", f"{len(sensor_columns):,}")

    if sensor_columns:
        selected_sensor = st.selectbox(
            "Fleet trend sensor",
            options=sensor_columns,
            key="fleet-overview-sensor",
        )
        trend = (
            frame.groupby("cycle", as_index=True)[selected_sensor]
            .mean()
            .rename("Mean sensor value")
            .to_frame()
        )
        st.subheader(f"{selected_sensor} by cycle")
        st.line_chart(trend)
        st.caption(
            f"Mean across machines in {spec.label}; this is a descriptive "
            "aggregation, not a forecast."
        )
    else:
        st.info("This saved dataset contains no sensor columns to chart.")

    report_ids = {
        "synthetic_fleet": "synthetic_reliability",
        "cmapss_fd001_train": "fd001_reliability",
    }
    report_id = report_ids.get(spec.dataset_id)
    if report_id is not None:
        st.subheader("Saved reliability summary")
        try:
            report = load_report(report_id, paths=paths)
        except DashboardDataError as error:
            render_saved_artifact_error(error)
        else:
            rates = pd.DataFrame(report.get("failure_rates", []))
            if not rates.empty:
                st.dataframe(rates, use_container_width=True, hide_index=True)
            st.caption(
                "Values are read from the saved reliability report; unsupported "
                "metrics remain unavailable in that report."
            )

    st.subheader("Live maintenance priorities")
    if spec.dataset_id != "synthetic_fleet":
        st.info(
            "The configured Phase 6 serving model was trained on the synthetic "
            "fleet. NASA C-MAPSS telemetry is not sent to that model."
        )
    else:
        api = PredictionApi(timeout=30.0)
        try:
            api.health()
            window_size = serving_window_size(paths=paths)
            machines = [
                machine_request(rows, window_size=window_size)
                for _, rows in frame.groupby("machine_id", sort=True)
            ]
            result = api.schedule(machines)
        except ApiUnavailable as error:
            st.warning(
                "The prediction API is unavailable. Saved telemetry, charts, and "
                f"reliability reports remain available. Details: {error}"
            )
        except DashboardDataError as error:
            st.warning(
                "Could not prepare fleet telemetry for the prediction API. Saved "
                "telemetry, charts, and reliability reports remain available. "
                f"Details: {error}"
            )
        else:
            priorities = pd.DataFrame(result.get("priorities", []))
            if priorities.empty:
                st.info("The API returned no maintenance priorities.")
            else:
                st.dataframe(priorities, use_container_width=True, hide_index=True)
                st.caption(
                    "These are model-derived API priorities, not observed failures "
                    "or confirmed maintenance outcomes."
                )
