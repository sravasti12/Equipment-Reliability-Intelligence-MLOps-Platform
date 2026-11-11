"""SHAP explanations for the fitted random-forest RUL regressor."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def explain_random_forest(
    model: Pipeline,
    test_features: pd.DataFrame,
    test_target: pd.Series[float],
    output_prefix: Path,
    *,
    random_seed: int,
) -> dict[str, object]:
    """Write global, individual, and sensor-aggregated SHAP explanations."""
    preprocessor = model.named_steps["preprocessor"]
    target_model = model.named_steps["regressor"].regressor_
    testing_transformed = preprocessor.transform(test_features)
    if hasattr(testing_transformed, "toarray"):
        testing_transformed = testing_transformed.toarray()
    testing_array = np.asarray(testing_transformed, dtype=float)

    names = [str(name) for name in preprocessor.get_feature_names_out()]
    sample_count = min(32, len(testing_array))
    rng = np.random.default_rng(random_seed)
    sample_indices = np.sort(
        rng.choice(len(testing_array), size=sample_count, replace=False)
    )
    explainer = shap.TreeExplainer(
        target_model,
        feature_perturbation="tree_path_dependent",
    )
    shap_values = np.asarray(explainer.shap_values(testing_array[sample_indices]))
    if shap_values.ndim == 3:
        shap_values = shap_values[..., 0]
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    model_outputs = np.asarray(
        target_model.predict(testing_array[sample_indices]), dtype=float
    )
    additivity_residual = np.abs(base_value + shap_values.sum(axis=-1) - model_outputs)
    max_additivity_residual = float(additivity_residual.max())
    if not np.allclose(
        base_value + shap_values.sum(axis=-1),
        model_outputs,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise RuntimeError(
            "SHAP contributions do not add up to the random-forest predictions"
        )
    mean_absolute = np.mean(np.abs(shap_values), axis=0)
    ordered = np.argsort(mean_absolute)[::-1]
    global_importance = [
        {"feature": names[index], "mean_absolute_shap": float(mean_absolute[index])}
        for index in ordered[:20]
    ]

    sensor_totals: dict[str, float] = {}
    for name, importance in zip(names, mean_absolute, strict=True):
        match = re.search(r"sensor_(\d+)", name)
        if match:
            sensor_name = f"sensor_{match.group(1)}"
            sensor_totals[sensor_name] = sensor_totals.get(sensor_name, 0.0) + float(
                importance
            )
    sensor_importance = [
        {"sensor": sensor, "mean_absolute_shap": value}
        for sensor, value in sorted(
            sensor_totals.items(), key=lambda item: item[1], reverse=True
        )
    ]

    global_plot = output_prefix.with_name(f"{output_prefix.stem}_shap_global.png")
    local_plot = output_prefix.with_name(f"{output_prefix.stem}_shap_individual.png")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        testing_array[sample_indices],
        feature_names=names,
        plot_type="bar",
        max_display=15,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(global_plot, dpi=140, bbox_inches="tight")
    plt.close()

    first = 0
    explanation = shap.Explanation(
        values=shap_values[first],
        base_values=base_value,
        data=testing_array[sample_indices[first]],
        feature_names=names,
    )
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(local_plot, dpi=140, bbox_inches="tight")
    plt.close()

    local_indices = np.argsort(np.abs(shap_values[first]))[::-1][:15]
    actual_index = int(sample_indices[first])
    local_explanation = [
        {
            "feature": names[index],
            "shap_value": float(shap_values[first, index]),
        }
        for index in local_indices
    ]
    return {
        "method": "SHAP TreeExplainer; tree-path-dependent feature perturbation",
        "max_abs_additivity_residual": max_additivity_residual,
        "explained_test_rows": int(sample_count),
        "global_feature_importance": global_importance,
        "global_sensor_importance": sensor_importance,
        "individual_example": {
            "test_row_index": actual_index,
            "actual_rul": float(test_target.iloc[actual_index]),
            "predicted_rul": float(
                model.predict(test_features.iloc[[actual_index]])[0]
            ),
            "base_value": base_value,
            "top_contributions": local_explanation,
        },
        "limitations": [
            (
                "SHAP values explain this fitted model and selected feature "
                "background, not causal effects."
            ),
            "Correlated sensors can share or redistribute feature attribution.",
            (
                "Local explanations describe model output, not failure probability "
                "or engineering mechanism."
            ),
        ],
        "global_plot": str(global_plot),
        "individual_plot": str(local_plot),
    }
