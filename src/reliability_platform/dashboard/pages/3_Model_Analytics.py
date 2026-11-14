"""Display saved model evaluations without generating monitoring data."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from reliability_platform.dashboard.common import (
    metric_value,
    render_saved_artifact_error,
)
from reliability_platform.dashboard.data_access import (
    DashboardDataError,
    DashboardPaths,
    load_report,
)

st.set_page_config(page_title="Model Analytics", layout="wide")
st.title("Model analytics")
st.write(
    "Review saved test metrics, interval results, and MLflow run identifiers. "
    "This page does not train models or produce monitoring data."
)

paths = DashboardPaths.from_environment()
report_options = [
    (report_id, label)
    for report_id, label in (
        ("synthetic_advanced", "Synthetic industrial fleet (seed 42)"),
        ("fd002_advanced", "NASA C-MAPSS FD002 training set"),
    )
    if paths.report_path(report_id).is_file()
]

if not report_options:
    st.caption("Data source: saved Phase 4 model reports (not available).")
    st.info("No saved Phase 4 model comparison report is available.")
else:
    selected_report_id = st.selectbox(
        "Model evaluation dataset",
        options=[report_id for report_id, _ in report_options],
        format_func=lambda report_id: dict(report_options)[report_id],
    )
    source_label = dict(report_options)[selected_report_id]
    st.caption(f"Data source: {source_label}.")
    try:
        report = load_report(selected_report_id, paths=paths)
    except DashboardDataError as error:
        render_saved_artifact_error(error)
    else:
        conformal = report.get("conformal", {})
        model_selection = report.get("model_selection", {})
        if isinstance(conformal, dict) and conformal:
            target = conformal.get("target_coverage")
            observed = conformal.get("observed_trajectory_coverage")
            width = conformal.get("average_interval_width_cycles")
            st.subheader("Saved test interval results")
            left, middle, right = st.columns(3)
            left.metric(
                "Target coverage",
                "Not reported" if target is None else f"{float(target):.1%}",
            )
            middle.metric(
                "Observed trajectory coverage",
                "Not reported" if observed is None else f"{float(observed):.1%}",
            )
            right.metric("Average interval width (cycles)", metric_value(width))
            st.caption(
                f"Coverage and width come from the selected saved test report. "
                "Test trajectories: "
                f"{metric_value(conformal.get('machines'), digits=0)}."
            )

        model_run_ids: dict[str, str] = {}
        models = report.get("models", {})
        if isinstance(models, dict):
            for name, result in models.items():
                if isinstance(result, dict) and result.get("mlflow_run_id"):
                    model_run_ids[name] = str(result["mlflow_run_id"])
        if isinstance(conformal, dict) and conformal.get("mlflow_run_id"):
            model_run_ids["conformal_calibration"] = str(conformal["mlflow_run_id"])
        for component, comparison_model in (
            ("anomaly_detection", "isolation_forest"),
            ("feature_ablation", "random_forest_raw_feature_ablation"),
            ("explainability", "explainability"),
        ):
            result = report.get(component, {})
            if isinstance(result, dict) and result.get("mlflow_run_id"):
                model_run_ids[comparison_model] = str(result["mlflow_run_id"])

        if model_selection.get("selected_rul_model"):
            selection_metric = model_selection.get(
                "selection_metric", "saved selection metric"
            )
            st.caption(
                "Validation-selected RUL model: "
                f"{model_selection['selected_rul_model']} "
                f"({selection_metric})."
            )
        if model_run_ids:
            st.subheader("MLflow run IDs")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"component": component, "mlflow_run_id": run_id}
                        for component, run_id in model_run_ids.items()
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

        comparison = pd.DataFrame(report.get("comparison", []))
        if comparison.empty:
            st.info("The saved report contains no model comparison rows.")
        else:
            comparison["mlflow_run_id"] = comparison["model"].map(model_run_ids)
            st.subheader("Test model comparison")
            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True,
            )

        if selected_report_id == "synthetic_advanced":
            try:
                serving = load_report("serving_manifest", paths=paths)
            except DashboardDataError as error:
                st.caption(f"Synthetic API serving-model manifest unavailable: {error}")
            else:
                if serving.get("model_run_id"):
                    st.caption(
                        "Synthetic-fleet API serving model run ID: "
                        f"{serving['model_run_id']}."
                    )

st.subheader("Phase 8 monitoring panels")
st.info("Drift analysis: Phase 8 placeholder; no drift results have been generated.")
st.info(
    "Calibration over time: Phase 8 placeholder; only the saved one-shot test "
    "coverage and interval width are available."
)
st.info(
    "Data quality monitoring: Phase 8 placeholder; no monitoring data is "
    "available to display."
)
