"""Landmark-based Weibull AFT survival analysis for machine lifetimes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from lifelines import KaplanMeierFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SurvivalDataset:
    """Machine-level survival times, event indicators, and early-life features."""

    features: pd.DataFrame
    duration: pd.Series[float]
    event_observed: pd.Series[bool]
    metadata: pd.DataFrame


@dataclass(frozen=True)
class SurvivalPreprocessor:
    """Train-fitted imputation/scaling for the AFT covariates."""

    columns: tuple[str, ...]
    imputer: SimpleImputer
    scaler: StandardScaler

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        aligned = features.reindex(columns=self.columns)
        new_dummy_columns = set(self.columns).difference(features.columns)
        if new_dummy_columns:
            aligned.loc[:, list(new_dummy_columns)] = 0.0
        values = self.scaler.transform(self.imputer.transform(aligned))
        return pd.DataFrame(values, columns=self.columns, index=features.index)


def build_landmark_dataset(
    frame: pd.DataFrame,
    *,
    landmark_cycles: int = 20,
) -> SurvivalDataset:
    """Summarize only the first landmark window and retain right-censoring.

    Duration is the last observed cycle. A machine is an observed event only
    when its input contains a failure_event; absent failure labels are treated
    as censored, never inferred from the last recorded observation.
    """
    if landmark_cycles < 1:
        raise ValueError("landmark_cycles must be positive")
    required = {"machine_id", "cycle"}
    if missing := required.difference(frame.columns):
        raise ValueError(
            f"survival data is missing columns: {', '.join(sorted(missing))}"
        )
    identity = [
        column
        for column in ("source", "dataset_split", "machine_id")
        if column in frame
    ]
    if frame.empty or frame[identity].isna().any().any():
        raise ValueError("survival data must contain machine identities")
    sensor_columns = sorted(
        (
            column
            for column in frame.columns
            if isinstance(column, str) and column.startswith("sensor_")
        ),
        key=lambda column: (
            0,
            int(column.removeprefix("sensor_")),
        )
        if column.removeprefix("sensor_").isdigit()
        else (1, column),
    )
    if not sensor_columns:
        raise ValueError("survival data must contain sensor_* columns")
    data = frame.sort_values([*identity, "cycle"], kind="stable")
    feature_rows: list[dict[str, object]] = []
    durations: list[float] = []
    events: list[bool] = []
    metadata_rows: list[dict[str, object]] = []

    for _, machine in data.groupby(identity, sort=False, dropna=False):
        early = machine.head(landmark_cycles)
        if len(early) < landmark_cycles:
            raise ValueError(
                f"machine trajectory has fewer than {landmark_cycles} landmark cycles"
            )
        features: dict[str, object] = {}
        for column in (*sensor_columns, "setting_1", "setting_2", "setting_3"):
            if column not in early:
                continue
            values = pd.to_numeric(early[column], errors="coerce")
            features[f"{column}_landmark_mean"] = float(values.mean())
            if column.startswith("sensor_"):
                features[f"{column}_landmark_std"] = float(values.std(ddof=0))
        if "machine_type" in machine:
            features["machine_type"] = str(machine["machine_type"].iloc[0])
        elif "operating_condition" in early:
            features["operating_condition"] = str(early["operating_condition"].iloc[0])

        feature_rows.append(features)
        durations.append(float(machine["cycle"].max()))
        events.append(
            bool(machine["failure_event"].astype(bool).any())
            if "failure_event" in machine
            else False
        )
        metadata_rows.append(
            {
                **{column: machine[column].iloc[0] for column in identity},
                **{
                    column: machine[column].iloc[0]
                    for column in ("machine_type", "operating_condition")
                    if column in machine
                },
            }
        )

    return SurvivalDataset(
        features=pd.DataFrame.from_records(feature_rows),
        duration=pd.Series(durations, dtype=float),
        event_observed=pd.Series(events, dtype=bool),
        metadata=pd.DataFrame.from_records(metadata_rows),
    )


def fit_survival_preprocessor(
    train_features: pd.DataFrame,
) -> tuple[SurvivalPreprocessor, pd.DataFrame]:
    """One-hot encode categories and fit numeric imputation/scaling on train."""
    data = train_features.copy()
    categorical = [
        column for column in data if not pd.api.types.is_numeric_dtype(data[column])
    ]
    data = pd.get_dummies(data, columns=categorical, dtype=float)
    if data.empty:
        raise ValueError("survival model requires at least one covariate")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    imputed = imputer.fit_transform(data)
    transformed = pd.DataFrame(
        scaler.fit_transform(imputed),
        columns=data.columns,
        index=data.index,
    )
    preprocessor = SurvivalPreprocessor(
        columns=tuple(data.columns),
        imputer=imputer,
        scaler=scaler,
    )
    return preprocessor, transformed


def _survival_matrix(
    preprocessor: SurvivalPreprocessor,
    features: pd.DataFrame,
) -> pd.DataFrame:
    data = features.copy()
    categorical = [
        column for column in data if not pd.api.types.is_numeric_dtype(data[column])
    ]
    data = pd.get_dummies(data, columns=categorical, dtype=float)
    return preprocessor.transform(data)


def transform_survival_covariates(
    preprocessor: SurvivalPreprocessor,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Transform covariates using the fitted training-only survival pipeline."""
    return _survival_matrix(preprocessor, features)


def integrated_brier_score(
    survival_probabilities: pd.DataFrame,
    durations: pd.Series[float],
    events: pd.Series[bool],
) -> float:
    """Average horizon-wise Brier score over known survival outcomes."""
    times = survival_probabilities.index.to_numpy(dtype=float)
    outcomes = durations.to_numpy(dtype=float)
    observed = events.to_numpy(dtype=bool)
    scores: list[float] = []
    for time in times:
        known = (observed & (outcomes <= time)) | (outcomes >= time)
        if not known.any():
            continue
        target_failure = (observed & (outcomes <= time))[known].astype(float)
        predicted_failure = (
            1.0 - survival_probabilities.loc[time].to_numpy(dtype=float)[known]
        )
        scores.append(float(np.mean(np.square(target_failure - predicted_failure))))
    if not scores:
        raise ValueError("no survival outcomes are known at requested horizons")
    return float(np.mean(scores))


def fit_weibull_aft(
    train: SurvivalDataset,
    *,
    penalizer: float = 0.1,
) -> tuple[WeibullAFTFitter, SurvivalPreprocessor, pd.DataFrame]:
    """Fit Weibull AFT with right-censored duration/event observations."""
    if not train.event_observed.any():
        raise ValueError("Weibull AFT requires at least one observed failure event")
    preprocessor, covariates = fit_survival_preprocessor(train.features)
    fit_frame = covariates.copy()
    fit_frame["duration"] = train.duration.to_numpy(dtype=float)
    fit_frame["event_observed"] = train.event_observed.to_numpy(dtype=int)
    model = WeibullAFTFitter(penalizer=penalizer)
    model.fit(
        fit_frame,
        duration_col="duration",
        event_col="event_observed",
        show_progress=False,
    )
    return model, preprocessor, covariates


def predict_survival(
    model: WeibullAFTFitter,
    preprocessor: SurvivalPreprocessor,
    features: pd.DataFrame,
    horizons: npt.NDArray[np.float64],
) -> pd.DataFrame:
    """Return estimated survival probabilities at specified cycle horizons."""
    covariates = _survival_matrix(preprocessor, features)
    return pd.DataFrame(model.predict_survival_function(covariates, times=horizons))


def survival_evaluation(
    durations: pd.Series[float],
    events: pd.Series[bool],
    survival_probabilities: pd.DataFrame,
) -> dict[str, object]:
    """Report censored concordance, survival Brier score, and horizon risks."""
    medians = survival_probabilities.index.to_numpy(dtype=float)
    # A model prediction at each horizon does not determine the median; the
    # caller adds the AFT median separately for concordance reporting.
    return {
        "integrated_brier_score": integrated_brier_score(
            survival_probabilities, durations, events
        ),
        "failure_probability_by_horizon": {
            str(float(time)): float(1.0 - survival_probabilities.loc[time].mean())
            for time in medians
        },
        "events": int(events.sum()),
        "censored": int((~events).sum()),
        "machines": int(len(durations)),
    }


def survival_concordance(
    durations: pd.Series[float],
    events: pd.Series[bool],
    predicted_median: pd.Series[float],
) -> float:
    """C-index using predicted median life as a higher-is-longer risk score."""
    return float(
        concordance_index(
            durations.to_numpy(dtype=float),
            predicted_median.to_numpy(dtype=float),
            event_observed=events.to_numpy(dtype=bool),
        )
    )


def kaplan_meier_brier(
    durations: pd.Series[float],
    events: pd.Series[bool],
    horizons: npt.NDArray[np.float64],
    training_durations: pd.Series[float],
    training_events: pd.Series[bool],
) -> float:
    """Featureless Kaplan-Meier survival baseline Brier score."""
    model = KaplanMeierFitter()
    model.fit(training_durations, event_observed=training_events)
    probabilities = pd.DataFrame(
        {
            index: [float(model.predict(float(time))) for time in horizons]
            for index in range(len(durations))
        },
        index=horizons,
    )
    return integrated_brier_score(probabilities, durations, events)


def survival_concordance_baseline(
    durations: pd.Series[float],
    events: pd.Series[bool],
) -> float:
    """Concordance for a non-personalized median-lifetime baseline."""
    return float(
        concordance_index(
            durations.to_numpy(dtype=float),
            np.ones(len(durations), dtype=float),
            event_observed=events.to_numpy(dtype=bool),
        )
    )
