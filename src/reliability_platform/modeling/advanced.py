"""Phase 4 model training, calibration, anomaly scoring, and comparison."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import numpy.typing as npt
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.pipeline import Pipeline

from reliability_platform.data.validation import validate_fleet
from reliability_platform.modeling.anomaly import (
    evaluate_injected_sensor_shift,
    fit_isolation_forest,
)
from reliability_platform.modeling.baselines import (
    _baseline_models,
)
from reliability_platform.modeling.conformal import (
    calibrate_trajectory_error,
    interval_metrics,
    prediction_intervals,
)
from reliability_platform.modeling.deep_learning import (
    LSTMModel,
    fit_lstm,
    predict_lstm,
    save_lstm_state,
)
from reliability_platform.modeling.explainability import explain_random_forest
from reliability_platform.modeling.features import FeatureDataset, engineer_features
from reliability_platform.modeling.metrics import grouped_metrics, regression_metrics
from reliability_platform.modeling.splitting import (
    MachinePartitions,
    split_by_machine,
)
from reliability_platform.modeling.survival import (
    SurvivalDataset,
    build_landmark_dataset,
    fit_weibull_aft,
    integrated_brier_score,
    kaplan_meier_brier,
    predict_survival,
    survival_concordance,
    transform_survival_covariates,
)
from reliability_platform.monitoring.lineage import RunLineage


def _dense(matrix: Any) -> npt.NDArray[np.float64]:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=float)


def _partition_counts(partitions: dict[str, pd.DataFrame]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for name, frame in partitions.items():
        identity = [
            column
            for column in ("source", "dataset_split", "machine_id")
            if column in frame
        ]
        result[name] = {
            "rows": int(len(frame)),
            "machines": int(frame[identity].drop_duplicates().shape[0]),
        }
    return result


def _inner_partition(training: pd.DataFrame, seed: int) -> MachinePartitions:
    return split_by_machine(
        training,
        fractions=(0.8, 0.1, 0.1),
        random_seed=seed + 1,
    )


def _metric_log(
    split_name: str,
    metrics: dict[str, float | int],
) -> None:
    mlflow.log_metrics(
        {f"{split_name}_{name}": float(value) for name, value in metrics.items()}
    )


def _report_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError(f"expected {name} report section to be a mapping")
    return value


def _report_number(value: object, name: str) -> float:
    if not isinstance(value, int | float):
        raise RuntimeError(f"expected numeric report value for {name}")
    return float(value)


def _feature_metrics(
    fitted: Any,
    feature_set: FeatureDataset,
    training: FeatureDataset,
) -> tuple[dict[str, object], npt.NDArray[np.float64]]:
    predictions = np.asarray(fitted.predict(feature_set.features), dtype=float)
    metrics = regression_metrics(feature_set.target, predictions)
    groups = grouped_metrics(
        feature_set.target,
        predictions,
        feature_set.metadata,
        training_metadata=training.metadata,
    )
    return {"overall": metrics, "by_group": groups}, predictions


def _raw_feature_columns(features: pd.DataFrame) -> list[str]:
    return [
        column
        for column in features.columns
        if column in {"cycle_number", "operating_condition"}
        or column.startswith("setting_")
        or (column.startswith("sensor_") and column.removeprefix("sensor_").isdigit())
    ]


def _fit_and_score_lstm(
    training: FeatureDataset,
    validation: FeatureDataset,
    *,
    window_size: int,
    epochs: int,
    random_seed: int,
) -> tuple[LSTMModel, dict[str, float | int], npt.NDArray[np.float64]]:
    fitted = fit_lstm(
        training.features,
        training.target,
        training.metadata,
        window_size=window_size,
        epochs=epochs,
        random_seed=random_seed,
    )
    predictions = predict_lstm(fitted, validation.features, validation.metadata)
    return fitted, regression_metrics(validation.target, predictions), predictions


def _fit_survival_selection(
    raw_splits: dict[str, pd.DataFrame],
    *,
    lineage: RunLineage,
    landmark_cycles: int,
    tracking_uri: str,
    experiment_name: str,
) -> tuple[dict[str, object], dict[str, Any]]:
    survival_sets = {
        name: build_landmark_dataset(frame, landmark_cycles=landmark_cycles)
        for name, frame in raw_splits.items()
    }
    training = survival_sets["fit"]
    validation = survival_sets["validation"]
    model, preprocessor, _ = fit_weibull_aft(training)
    horizons = np.unique(
        np.quantile(training.duration.to_numpy(dtype=float), [0.25, 0.5, 0.75])
    )
    validation_probabilities = predict_survival(
        model, preprocessor, validation.features, horizons
    )
    validation_brier = integrated_brier_score(
        validation_probabilities, validation.duration, validation.event_observed
    )
    validation_km_brier = kaplan_meier_brier(
        validation.duration,
        validation.event_observed,
        horizons,
        training.duration,
        training.event_observed,
    )
    selected = (
        "weibull_aft"
        if validation_brier <= validation_km_brier
        else "kaplan_meier_baseline"
    )
    run_client: dict[str, Any] = {
        "model": model,
        "preprocessor": preprocessor,
        "survival_sets": survival_sets,
        "horizons": horizons,
        "selected": selected,
    }

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="weibull_aft") as run:
        lineage.log(training.features)
        mlflow.log_params(
            {
                "model": "WeibullAFTFitter",
                "landmark_cycles": landmark_cycles,
                "training_machines": len(training.duration),
                "training_events": int(training.event_observed.sum()),
                "training_censored": int((~training.event_observed).sum()),
                "penalizer": 0.1,
            }
        )
        mlflow.log_metric("validation_integrated_brier", validation_brier)
        mlflow.log_metric("validation_kaplan_meier_brier", validation_km_brier)
        mlflow.log_param("selected_by_validation_brier", selected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weibull_aft.pkl"
            path.write_bytes(pickle.dumps((model, preprocessor)))
            mlflow.log_artifact(str(path), artifact_path="model")
        run_client["weibull_run_id"] = run.info.run_id

    with mlflow.start_run(run_name="kaplan_meier_baseline") as run:
        lineage.log(training.features)
        mlflow.log_params(
            {
                "model": "KaplanMeierFitter",
                "baseline": "non-personalized training survival curve",
                "selected_by_validation_brier": selected,
            }
        )
        mlflow.log_metric("validation_integrated_brier", validation_km_brier)
        run_client["km_run_id"] = run.info.run_id

    result: dict[str, object] = {
        "landmark_cycles": landmark_cycles,
        "time_origin": "cycle 1; covariates use only the first landmark_cycles rows",
        "duration": "last observed cycle",
        "event": "failure_event; absent failure labels are treated as right-censored",
        "censoring_note": (
            "Non-informative right censoring is assumed. C-MAPSS training and the "
            "current synthetic fleet are predominantly run-to-failure, so these "
            "data provide little empirical evidence about censoring behaviour."
        ),
        "interpretation": (
            "Survival probabilities are failure-free probabilities by time horizon, "
            "not cycle-level RUL point predictions."
        ),
        "validation": {
            "weibull_aft_integrated_brier": validation_brier,
            "kaplan_meier_integrated_brier": validation_km_brier,
            "selected": selected,
        },
        "horizons_cycles": [float(value) for value in horizons],
        "training_machines": len(training.duration),
        "training_events": int(training.event_observed.sum()),
        "training_censored": int((~training.event_observed).sum()),
    }
    return result, run_client


def _evaluate_survival_test(
    survival_report: dict[str, object],
    client: dict[str, Any],
    tracking_uri: str,
) -> dict[str, object]:
    model = client["model"]
    preprocessor = client["preprocessor"]
    horizons = client["horizons"]
    training: SurvivalDataset = client["survival_sets"]["fit"]
    testing: SurvivalDataset = client["survival_sets"]["test"]
    predictions = predict_survival(model, preprocessor, testing.features, horizons)
    aft_brier = integrated_brier_score(
        predictions, testing.duration, testing.event_observed
    )
    km_brier = kaplan_meier_brier(
        testing.duration,
        testing.event_observed,
        horizons,
        training.duration,
        training.event_observed,
    )
    median = model.predict_median(
        transform_survival_covariates(preprocessor, testing.features)
    )
    median_series = pd.Series(np.asarray(median, dtype=float).reshape(-1))
    aft_cindex = survival_concordance(
        testing.duration, testing.event_observed, median_series
    )
    km_cindex = float(
        concordance_index(
            testing.duration.to_numpy(dtype=float),
            np.ones(len(testing.duration), dtype=float),
            event_observed=testing.event_observed.to_numpy(dtype=bool),
        )
    )
    selected = str(client["selected"])
    km_model = KaplanMeierFitter().fit(
        training.duration, event_observed=training.event_observed
    )
    per_machine: list[dict[str, object]] = []
    for row_index in range(len(testing.metadata)):
        machine = testing.metadata.iloc[row_index]
        per_machine.append(
            {
                "machine_id": str(machine["machine_id"]),
                "duration_cycles": float(testing.duration.iloc[row_index]),
                "failure_observed": bool(testing.event_observed.iloc[row_index]),
                "predicted_median_lifetime_cycles": float(
                    median_series.iloc[row_index]
                ),
                "failure_probability_by_horizon": {
                    str(float(time)): float(1.0 - predictions.loc[time].iloc[row_index])
                    for time in horizons
                },
            }
        )
    result: dict[str, object] = {
        "weibull_aft": {
            "integrated_brier_score": aft_brier,
            "concordance_index": aft_cindex,
            "failure_probability_by_horizon": {
                str(float(time)): float(1.0 - predictions.loc[time].mean())
                for time in horizons
            },
            "test_events": int(testing.event_observed.sum()),
            "test_censored": int((~testing.event_observed).sum()),
            "per_machine_estimates": per_machine,
        },
        "kaplan_meier_baseline": {
            "integrated_brier_score": km_brier,
            "concordance_index": km_cindex,
            "failure_probability_by_horizon": {
                str(float(time)): float(1.0 - km_model.predict(float(time)))
                for time in horizons
            },
        },
        "selected_by_validation": selected,
    }
    mlflow.set_tracking_uri(tracking_uri)
    for key, run_id_key in (
        ("weibull_aft", "weibull_run_id"),
        ("kaplan_meier_baseline", "km_run_id"),
    ):
        with mlflow.start_run(run_id=client[run_id_key]):
            metrics = result[key]
            if not isinstance(metrics, dict):
                raise RuntimeError(f"missing test metrics for {key}")
            mlflow.log_metric(
                f"test_{key}_integrated_brier",
                float(metrics["integrated_brier_score"]),
            )
            mlflow.log_metric(
                f"test_{key}_concordance_index",
                float(metrics["concordance_index"]),
            )
    survival_report["test"] = result
    return result


def _log_candidate_test_metrics(
    tracking_uri: str,
    run_id: str,
    metrics: dict[str, object],
) -> None:
    overall = metrics["overall"]
    if not isinstance(overall, dict):
        return
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_id=run_id):
        _metric_log("test", overall)


def _fit_anomaly_report(
    feature_sets: dict[str, FeatureDataset],
    *,
    lineage: RunLineage,
    random_seed: int,
    tracking_uri: str,
    experiment_name: str,
) -> tuple[dict[str, object], dict[str, Any]]:
    training = feature_sets["fit"]
    detector, preprocessor = fit_isolation_forest(
        training.features,
        random_seed=random_seed,
    )
    validation_metrics, _ = evaluate_injected_sensor_shift(
        detector,
        preprocessor,
        feature_sets["validation"].features,
        feature_sets["validation"].metadata,
        training.features,
        random_seed=random_seed + 11,
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="isolation_forest_injected_shift") as run:
        lineage.log(training.features)
        mlflow.log_params(
            {
                "model": "IsolationForest",
                "n_estimators": 100,
                "contamination": "auto",
                "random_seed": random_seed,
                "training_rows": len(training.features),
            }
        )
        mlflow.log_metrics(
            {
                f"validation_{name}": float(value)
                for name, value in validation_metrics.items()
                if isinstance(value, int | float)
            }
        )
        mlflow.sklearn.log_model(
            sk_model=Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("detector", detector),
                ]
            ),
            artifact_path="detector",
        )
        client = {
            "detector": detector,
            "preprocessor": preprocessor,
            "run_id": run.info.run_id,
        }
    report: dict[str, object] = {
        "method": "Isolation Forest",
        "training_rows": len(training.features),
        "validation_injected_shift": validation_metrics,
        "evaluation_protocol": (
            "For each held-out machine, pair one unchanged cycle feature vector "
            "with a copy shifted by six training standard deviations in the first "
            "sensor. Scores are ranked only; this controlled scenario is not a "
            "calibrated fault or failure probability."
        ),
    }
    return report, client


def train_advanced_models(
    frame: pd.DataFrame,
    *,
    tracking_uri: str = "file:./mlruns",
    experiment_name: str = "reliability-phase4-advanced",
    random_seed: int = 42,
    window_size: int = 5,
    sequence_length: int = 20,
    epochs: int = 10,
    target_coverage: float = 0.9,
    landmark_cycles: int = 20,
    output_prefix: Path = Path("reports/generated/advanced"),
) -> dict[str, object]:
    """Train advanced models with validation selection and one final test pass."""
    validate_fleet(frame)
    if not 0 < target_coverage < 1:
        raise ValueError("target_coverage must be between 0 and 1")
    if sequence_length < 1 or epochs < 1 or landmark_cycles < 1:
        raise ValueError(
            "sequence_length, epochs, and landmark_cycles must be positive"
        )

    outer = split_by_machine(frame, random_seed=random_seed)
    inner = _inner_partition(outer.train, random_seed)
    calibration_frame = pd.concat([inner.validation, inner.test], ignore_index=True)
    raw_splits = {
        "fit": inner.train,
        "calibration": calibration_frame,
        "validation": outer.validation,
        "test": outer.test,
    }
    counts = _partition_counts(raw_splits)
    feature_sets = {
        name: engineer_features(raw, window_size=window_size)
        for name, raw in raw_splits.items()
    }
    training = feature_sets["fit"]
    validation = feature_sets["validation"]
    test = feature_sets["test"]
    lineage = RunLineage.from_frame(frame)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    report: dict[str, object] = {
        "dataset_rows": int(len(frame)),
        "random_seed": random_seed,
        "window_size": window_size,
        "sequence_length": sequence_length,
        "lstm_epochs": epochs,
        "target_coverage": target_coverage,
        "split_protocol": {
            "outer_machine_split": {"train": 0.7, "validation": 0.15, "test": 0.15},
            "training_machine_split": {
                "model_fit": 0.8,
                "conformal_calibration": 0.2,
            },
            "partitions": counts,
            "test_policy": (
                "Test machines are not used to fit, calibrate, or select models. "
                "They are evaluated once after validation-based selection."
            ),
        },
        "models": {},
        "conformal": {},
        "survival": {},
        "anomaly_detection": {},
        "explainability": {},
    }

    candidate_models = _baseline_models(training.features, random_seed=random_seed)
    candidate_predictions: dict[str, Any] = {}
    validation_results: dict[str, dict[str, float | int]] = {}
    run_ids: dict[str, str] = {}
    for name, model in candidate_models.items():
        with mlflow.start_run(run_name=name) as run:
            lineage.log(training.features)
            mlflow.log_params(
                {
                    "model": name,
                    "random_seed": random_seed,
                    "window_size": window_size,
                    "fit_machines": counts["fit"]["machines"],
                    "validation_machines": counts["validation"]["machines"],
                    "test_machines": counts["test"]["machines"],
                }
            )
            model.fit(training.features, training.target)
            validation_prediction = np.asarray(model.predict(validation.features))
            metrics = regression_metrics(validation.target, validation_prediction)
            _metric_log("validation", metrics)
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                input_example=training.features.head(3),
            )
            candidate_models[name] = model
            candidate_predictions[name] = model
            validation_results[name] = metrics
            run_ids[name] = run.info.run_id

    with mlflow.start_run(run_name="lstm_rul") as run:
        lineage.log(training.features)
        mlflow.log_params(
            {
                "model": "unidirectional_lstm",
                "random_seed": random_seed,
                "sequence_length": sequence_length,
                "epochs": epochs,
                "hidden_size": 32,
                "fit_machines": counts["fit"]["machines"],
            }
        )
        lstm, lstm_validation, _ = _fit_and_score_lstm(
            training,
            validation,
            window_size=sequence_length,
            epochs=epochs,
            random_seed=random_seed,
        )
        _metric_log("validation", lstm_validation)
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "lstm_state.pt"
            save_lstm_state(lstm, str(model_path))
            mlflow.log_artifact(str(model_path), artifact_path="model")
        candidate_predictions["lstm"] = lstm
        validation_results["lstm"] = lstm_validation
        run_ids["lstm"] = run.info.run_id

    selected_rul_model = min(
        validation_results,
        key=lambda model_name: validation_results[model_name]["rmse"],
    )
    report["model_selection"] = {
        "selected_rul_model": selected_rul_model,
        "selection_metric": "validation RMSE",
        "validation_rmse": {
            name: float(metrics["rmse"]) for name, metrics in validation_results.items()
        },
        "test_used_for_selection": False,
    }

    test_results: dict[str, dict[str, object]] = {}
    test_overall_metrics: dict[str, dict[str, float | int]] = {}
    test_predictions: dict[str, npt.NDArray[np.float64]] = {}
    for name, fitted in candidate_predictions.items():
        if name == "lstm":
            prediction = predict_lstm(fitted, test.features, test.metadata)
            test_metric = regression_metrics(test.target, prediction)
            groups = grouped_metrics(
                test.target,
                prediction,
                test.metadata,
                training_metadata=training.metadata,
            )
            result: dict[str, object] = {"overall": test_metric, "by_group": groups}
        else:
            result, prediction = _feature_metrics(fitted, test, training)
            if name == "random_forest":
                explanation_model = fitted
        test_results[name] = result
        overall = result.get("overall")
        if not isinstance(overall, dict):
            raise RuntimeError(f"missing overall test metrics for {name}")
        numeric_metrics: dict[str, float | int] = {}
        for metric_name, metric_value in overall.items():
            if isinstance(metric_value, int | float):
                numeric_metrics[metric_name] = metric_value
        test_overall_metrics[name] = numeric_metrics
        test_predictions[name] = prediction
        _log_candidate_test_metrics(tracking_uri, run_ids[name], result)

    models_report = {
        name: {
            "validation": validation_results[name],
            "test": test_results[name],
            "mlflow_run_id": run_ids[name],
        }
        for name in validation_results
    }
    report["models"] = models_report

    raw_columns = _raw_feature_columns(training.features)
    ablation_training = training.features.loc[:, raw_columns]
    ablation_validation = validation.features.loc[:, raw_columns]
    ablation_test = test.features.loc[:, raw_columns]
    ablation_model = _baseline_models(ablation_training, random_seed=random_seed)[
        "random_forest"
    ]
    with mlflow.start_run(run_name="random_forest_raw_feature_ablation") as run:
        lineage.log(ablation_training)
        mlflow.log_params(
            {
                "model": "RandomForestRegressor",
                "ablation": "raw sensors/settings/condition/cycle only",
                "random_seed": random_seed,
            }
        )
        ablation_model.fit(ablation_training, training.target)
        ablation_val = regression_metrics(
            validation.target,
            np.asarray(ablation_model.predict(ablation_validation)),
        )
        _metric_log("validation", ablation_val)
        ablation_prediction = np.asarray(ablation_model.predict(ablation_test))
        ablation_test_metrics = regression_metrics(test.target, ablation_prediction)
        _metric_log("test", ablation_test_metrics)
        mlflow.sklearn.log_model(ablation_model, artifact_path="model")
        feature_ablation = {
            "model": "random_forest_raw_feature_ablation",
            "included_features": raw_columns,
            "validation": ablation_val,
            "test": ablation_test_metrics,
            "test_used_for_selection": False,
            "mlflow_run_id": run.info.run_id,
        }
        report["feature_ablation"] = feature_ablation

    calibration = feature_sets["calibration"]
    selected_fitted = candidate_predictions[selected_rul_model]
    if selected_rul_model == "lstm":
        calibration_prediction = predict_lstm(
            selected_fitted, calibration.features, calibration.metadata
        )
    else:
        calibration_prediction = np.asarray(
            selected_fitted.predict(calibration.features), dtype=float
        )
    error_bound, calibration_machines = calibrate_trajectory_error(
        calibration.target,
        calibration_prediction,
        calibration.metadata,
        alpha=1 - target_coverage,
    )
    selected_prediction = test_predictions[selected_rul_model]
    lower, upper = prediction_intervals(
        selected_prediction,
        error_bound=error_bound,
    )
    coverage = interval_metrics(
        test.target,
        lower,
        upper,
        test.metadata,
        target_coverage=target_coverage,
        training_metadata=training.metadata,
    )
    coverage.update(
        {
            "alpha": 1 - target_coverage,
            "calibration_machines": calibration_machines,
            "trajectory_error_quantile_cycles": error_bound,
            "usefulness_note": (
                "Trajectory-level intervals target simultaneous coverage for all "
                "cycles of a new machine. Width is reported in RUL cycles; overly "
                "wide intervals indicate low operational usefulness."
            ),
        }
    )
    with mlflow.start_run(run_name="split_conformal_trajectory_intervals") as run:
        lineage.log(training.features)
        mlflow.log_params(
            {
                "selected_model": selected_rul_model,
                "target_coverage": target_coverage,
                "calibration_unit": "machine trajectory",
                "calibration_machines": calibration_machines,
            }
        )
        mlflow.log_metrics(
            {
                "target_coverage": target_coverage,
                "observed_row_coverage": _report_number(
                    coverage["observed_row_coverage"], "observed_row_coverage"
                ),
                "observed_trajectory_coverage": _report_number(
                    coverage["observed_trajectory_coverage"],
                    "observed_trajectory_coverage",
                ),
                "average_interval_width_cycles": _report_number(
                    coverage["average_interval_width_cycles"],
                    "average_interval_width_cycles",
                ),
                "calibration_error_bound_cycles": error_bound,
            }
        )
        coverage["mlflow_run_id"] = run.info.run_id
    report["conformal"] = coverage

    survival_report, survival_client = _fit_survival_selection(
        raw_splits,
        lineage=lineage,
        landmark_cycles=landmark_cycles,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )
    survival_test = _evaluate_survival_test(
        survival_report,
        survival_client,
        tracking_uri,
    )
    survival_report["model_selection"] = {
        "selected_by_validation_integrated_brier": survival_client["selected"],
        "test_used_for_selection": False,
    }
    report["survival"] = survival_report

    anomaly_report, anomaly_client = _fit_anomaly_report(
        feature_sets,
        lineage=lineage,
        random_seed=random_seed,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )
    anomaly_test, _ = evaluate_injected_sensor_shift(
        anomaly_client["detector"],
        anomaly_client["preprocessor"],
        test.features,
        test.metadata,
        training.features,
        random_seed=random_seed + 22,
    )
    with mlflow.start_run(run_id=anomaly_client["run_id"]):
        mlflow.log_metrics(
            {
                f"test_{name}": float(value)
                for name, value in anomaly_test.items()
                if isinstance(value, int | float)
            }
        )
    anomaly_report["test_injected_shift"] = anomaly_test
    anomaly_report["mlflow_run_id"] = anomaly_client["run_id"]
    report["anomaly_detection"] = anomaly_report

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    shap_report = explain_random_forest(
        explanation_model,
        test.features,
        test.target,
        output_prefix,
        random_seed=random_seed,
    )
    with mlflow.start_run(run_name="random_forest_shap_explanations") as run:
        lineage.log(test.features)
        mlflow.log_params(
            {
                "model": "phase4_random_forest",
                "method": "TreeExplainer tree-path-dependent",
                "explained_test_rows": shap_report["explained_test_rows"],
            }
        )
        mlflow.log_dict(shap_report, "shap_explanations.json")
        global_plot = shap_report.get("global_plot")
        individual_plot = shap_report.get("individual_plot")
        if not isinstance(global_plot, str) or not isinstance(individual_plot, str):
            raise RuntimeError("SHAP report is missing generated plot paths")
        mlflow.log_artifact(global_plot)
        mlflow.log_artifact(individual_plot)
        shap_report["mlflow_run_id"] = run.info.run_id
    report["explainability"] = shap_report

    comparison: list[dict[str, object]] = []
    for name in validation_results:
        test_metric = test_overall_metrics[name]
        comparison.append(
            {
                "model": name,
                "task": "cycle-level RUL regression",
                "validation_rmse": float(validation_results[name]["rmse"]),
                "test_rmse": float(test_metric["rmse"]),
                "test_mae": float(test_metric["mae"]),
                "test_asymmetric_score": float(test_metric["asymmetric_score"]),
            }
        )
    comparison.append(
        {
            "model": "random_forest_raw_feature_ablation",
            "task": "cycle-level RUL regression",
            "validation_rmse": float(ablation_val["rmse"]),
            "test_rmse": float(ablation_test_metrics["rmse"]),
            "test_mae": float(ablation_test_metrics["mae"]),
            "test_asymmetric_score": float(ablation_test_metrics["asymmetric_score"]),
        }
    )
    survival_validation = _report_mapping(
        survival_report["validation"], "survival validation"
    )
    weibull_test = _report_mapping(survival_test["weibull_aft"], "Weibull test")
    km_test = _report_mapping(
        survival_test["kaplan_meier_baseline"], "Kaplan-Meier test"
    )
    anomaly_validation = _report_mapping(
        anomaly_report["validation_injected_shift"], "anomaly validation"
    )
    comparison.extend(
        [
            {
                "model": "weibull_aft",
                "task": "right-censored survival",
                "validation_integrated_brier": float(
                    _report_number(
                        survival_validation["weibull_aft_integrated_brier"],
                        "weibull_aft_integrated_brier",
                    )
                ),
                "test_integrated_brier": _report_number(
                    weibull_test["integrated_brier_score"],
                    "weibull_aft test integrated_brier_score",
                ),
                "test_concordance_index": _report_number(
                    weibull_test["concordance_index"],
                    "weibull_aft test concordance_index",
                ),
            },
            {
                "model": "kaplan_meier_baseline",
                "task": "featureless right-censored survival",
                "validation_integrated_brier": float(
                    _report_number(
                        survival_validation["kaplan_meier_integrated_brier"],
                        "kaplan_meier_integrated_brier",
                    )
                ),
                "test_integrated_brier": _report_number(
                    km_test["integrated_brier_score"],
                    "Kaplan-Meier test integrated_brier_score",
                ),
                "test_concordance_index": _report_number(
                    km_test["concordance_index"],
                    "Kaplan-Meier test concordance_index",
                ),
            },
            {
                "model": "isolation_forest",
                "task": "injected-shift anomaly detection",
                "validation_roc_auc": _report_number(
                    anomaly_validation["roc_auc"], "validation anomaly roc_auc"
                ),
                "test_roc_auc": _report_number(
                    anomaly_test["roc_auc"], "test anomaly roc_auc"
                ),
                "test_average_precision": _report_number(
                    anomaly_test["average_precision"],
                    "test anomaly average_precision",
                ),
            },
        ]
    )
    report["comparison"] = comparison
    return report
