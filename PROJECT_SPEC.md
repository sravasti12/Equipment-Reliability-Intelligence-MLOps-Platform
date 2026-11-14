# Equipment Reliability Intelligence & MLOps Platform
## Complete Project Specification for GitHub Copilot

---

# 0. Instructions for GitHub Copilot — READ FIRST

You are building this project in clearly defined phases. Follow these instructions throughout development.

### Development rules

1. Work on exactly one phase at a time, in numerical order. Never begin a new phase until the previous phase's acceptance criteria have been met.
2. Before implementing a phase, explain its architecture, proposed files, dependencies, and implementation plan.
3. After every completed task, update `PROGRESS.md`. Record completed tasks, decisions, problems, test results, and next steps.
4. Before ending any session, ensure `PROGRESS.md` accurately describes what is done, in progress, remaining, and blocked.
5. Write unit and integration tests for every relevant module.
6. Never fabricate experimental results, benchmark metrics, test results, or model improvements.
7. Use only public datasets and synthetic data. Never include confidential company or client information.
8. Use fixed random seeds, pinned dependencies, configuration files, type hints, and documentation.
9. Keep implementations modular and avoid unnecessary abstractions.
10. Do not generate the entire application in one session. Implement, test, review, and understand each phase.
11. Explain important technical decisions so the developer understands why an implementation is necessary.
12. Do not modify existing working functionality unnecessarily.
13. If a requirement is ambiguous, select the simplest technically sound approach and document the decision in `PROGRESS.md`.
14. Do not mark a task complete merely because code has been generated. It must be implemented, tested, and verified against its acceptance criteria.

### Strict architectural boundaries

This project is deliberately a traditional machine learning and MLOps engineering platform.

DO NOT introduce:
- Z3 or other formal constraint solvers.
- LangGraph or LangChain.
- LLMs or generative AI.
- Agentic AI workflows.
- AI-based autonomous decision-making.
- A second fault-injection or recovery-verification architecture resembling the OTA Campaign Verifier.
- Model registry stages (candidate, staging, production).
- Automated or gated model promotion.
- Model rollback.
- Shadow, champion-challenger, or A/B model deployment.

The maintenance scheduler must remain a transparent, deterministic heuristic.

Sensor missingness and data-quality handling are allowed as ordinary data-engineering requirements. They must not become a separate fault-injection or recovery-simulation subsystem.

---

# 1. Project Overview

Build an end-to-end industrial equipment reliability intelligence platform that analyzes sensor data from a fleet of heavy industrial machines, predicts equipment degradation and failures, quantifies prediction uncertainty, and helps prioritize preventive maintenance.

The platform should combine:

- Industrial sensor data engineering.
- Exploratory reliability analytics.
- Time-series machine learning.
- Remaining Useful Life (RUL) prediction.
- Survival analysis.
- Anomaly detection.
- Prediction uncertainty quantification.
- Model explainability.
- Cost-aware maintenance planning.
- Historical strategy backtesting.
- Model serving and deployment.
- Experiment tracking, data and model monitoring.

### Primary objective

Demonstrate the ability to develop, evaluate, deploy, and maintain an end-to-end machine learning system that converts industrial sensor data into measurable operational insights.

### Core engineering question

Can we develop a reliable predictive analytics system that estimates equipment degradation, communicates uncertainty, supports maintenance decisions, and maintains model quality throughout its lifecycle?

### Explicit non-goals

- No formal constraint optimization.
- No Z3 scheduling.
- No LLM or agent integration.
- No automated autonomous maintenance execution.
- No dedicated fault-injection or firmware-recovery subsystem.
- No model promotion, rollback, or deployment-slot logic.

---

# 2. Technology Stack

Use Python 3.11.

| Area | Technologies |
|---|---|
| Programming | Python 3.11 |
| Data processing | Pandas, NumPy |
| Storage | DuckDB, Parquet |
| Traditional ML | Scikit-learn |
| Gradient boosting | XGBoost or LightGBM |
| Deep learning | PyTorch |
| Survival analysis | Lifelines |
| Uncertainty | MAPIE or custom split-conformal implementation |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| Drift monitoring | Evidently |
| API | FastAPI, Pydantic |
| Dashboard | Streamlit or Plotly Dash |
| Testing | Pytest |
| Code quality | Ruff, Mypy, pre-commit |
| Deployment | Docker, Docker Compose |
| CI/CD | GitHub Actions |

Choose compatible, pinned dependency versions.

Avoid installing every optional library unless it is required for the current phase.

---

# 3. Data Sources

## 3.1 Primary dataset: NASA C-MAPSS

Use the publicly available NASA Commercial Modular Aero-Propulsion System Simulation dataset.

Dataset source:
https://data.nasa.gov/

Treat each engine as an individual machine with a degradation history.

The dataset contains:
- Engine identifiers.
- Operating conditions.
- Multiple sensor measurements.
- Operational cycles.
- Run-to-failure training trajectories.
- Test trajectories and corresponding remaining useful life labels.

Use the dataset to develop and evaluate RUL prediction models.

Preserve the distinction between training, validation, and test data.

Never use future information to construct features for past predictions.

## 3.2 Secondary dataset: Synthetic industrial fleet

Build a configurable synthetic fleet generator to supplement the NASA dataset.

The simulator should generate at least 100 machines with:

- Configurable degradation curves.
- Different operating conditions.
- Sensor noise.
- Different machine lifetimes.
- Missing sensor observations.
- Maintenance events.
- Failure events.
- Machine-specific characteristics.

Sensor missingness should be handled as a data-quality problem.

The synthetic dataset should allow reproducible experiments with fixed random seeds.

Do not claim that synthetic data represents actual industrial telemetry.

Clearly distinguish simulated results from results obtained using NASA data.

## 3.3 Data storage

Use:

- `data/raw/` for downloaded datasets.
- `data/processed/` for validated and transformed datasets.
- Parquet for efficient data storage.
- DuckDB for analytical queries.

Raw and processed datasets must be excluded from Git where appropriate.

Provide documented commands for obtaining and processing the datasets.

---

# 4. Repository Structure

Use the following structure, adjusting only when technically necessary.

```text
reliability-platform/
│
├── PROJECT_SPEC.md
├── PROGRESS.md
├── README.md
├── pyproject.toml
├── .pre-commit-config.yaml
├── .gitignore
│
├── configs/
│   ├── data.yaml
│   ├── features.yaml
│   ├── models.yaml
│   ├── training.yaml
│   ├── evaluation.yaml
│   ├── scheduling.yaml
│   └── monitoring.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── src/
│   ├── data/
│   │   ├── loaders.py
│   │   ├── simulator.py
│   │   ├── validation.py
│   │   └── storage.py
│   │
│   ├── features/
│   │   ├── windowing.py
│   │   └── engineering.py
│   │
│   ├── models/
│   │   ├── baselines.py
│   │   ├── rul.py
│   │   ├── anomaly.py
│   │   ├── survival.py
│   │   └── conformal.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── validation.py
│   │   └── backtesting.py
│   │
│   ├── scheduling/
│   │   ├── scheduler.py
│   │   └── cost_model.py
│   │
│   ├── serving/
│   │   ├── api.py
│   │   └── schemas.py
│   │
│   ├── dashboard/
│   │
│   └── monitoring/
│       ├── drift.py
│       ├── performance.py
│       ├── recommendation.py
│       └── data_quality.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── evaluation/
│
├── artifacts/
├── reports/
├── docs/
│
├── docker/
├── docker-compose.yml
│
└── .github/
    └── workflows/
        └── ci.yml
```

Do not put application logic inside notebooks.

Notebooks are exclusively for exploratory analysis and visualization.

---

# 5. Development Phases

## Phase 0: Project Setup

Tasks:

- Create repository structure.
- Configure Python 3.11.
- Set up dependency management using `uv` or pip.
- Pin dependencies.
- Configure Ruff and Mypy.
- Configure pre-commit.
- Configure Pytest.
- Create initial GitHub Actions workflow.
- Create README skeleton.
- Create PROGRESS.md using the supplied template.
- Configure `.gitignore`.

Acceptance criteria:

- Repository installs cleanly from scratch.
- Linting and type-checking work.
- CI successfully executes.
- Tests can be discovered and executed.
- No unnecessary application functionality is introduced.

---

## Phase 1: Data Engineering and Synthetic Fleet

Tasks:

### NASA dataset pipeline

- Download and load NASA C-MAPSS.
- Validate dataset schema.
- Identify machine IDs, operational cycles, operating conditions, and sensor columns.
- Handle missing and invalid observations.
- Store validated data in Parquet or DuckDB.
- Document the dataset structure.

### Synthetic fleet generator

Implement configurable degradation curves and machine characteristics.

Support:

- Different machine lifetimes.
- Operating conditions.
- Sensor noise.
- Missing observations.
- Maintenance events.
- Failure events.

Use reproducible random seeds.

### Data validation

Implement:

- Schema validation.
- Range validation.
- Duplicate detection.
- Missing-value analysis.
- Timestamp and cycle ordering.
- Machine-level consistency checks.

Acceptance criteria:

A single documented command should generate or process a validated dataset containing at least 100 synthetic machines.

NASA dataset loading and synthetic generation must have automated tests.

---

## Phase 2: Exploratory Data Analysis and Reliability Analytics

Tasks:

### Exploratory analysis

Create an EDA notebook investigating:

- Sensor behaviour throughout machine life.
- Sensor trends before failure.
- Degradation patterns.
- Differences between operating conditions.
- Machine-to-machine variability.
- Correlations between sensor measurements.

### Reliability metrics

Implement reproducible calculations for:

- Mean Time Between Failures (MTBF), where the data supports valid failure and exposure information.
- Failure rate by machine group.
- Downtime.
- Machine lifetime distributions.
- Maintenance frequency.
- Failure distribution.

Document at least five meaningful findings.

Every finding must be supported by actual analysis and visualizations.

Acceptance criteria:

- EDA notebook executes successfully.
- Reliability metrics are calculated using reusable Python modules.
- Findings are reproducible through scripts.
- No invented findings or results.

---

## Phase 3: Feature Engineering and Baseline Models

Tasks:

### Feature engineering

Implement:

- Rolling means.
- Rolling standard deviations.
- Trend features.
- Sensor variation.
- Operating-condition features.
- Health indicators.
- Time-window features.

Prevent future-data leakage.

### Dataset splitting

Use machine-level separation.

No machine trajectory or future observation may leak into an earlier prediction.

Create training, validation, and test datasets with clearly documented splitting rules.

Use chronological evaluation within machine trajectories where appropriate.

### Baseline models

Implement:

1. Constant-RUL baseline.
2. Linear regression.
3. Random forest or gradient boosting baseline.

### Evaluation metrics

Calculate:

- RMSE.
- MAE.
- Asymmetric RUL scoring.
- Performance by machine group.
- Performance across operating conditions.

Track experiments in MLflow.

Acceptance criteria:

- Baseline models train successfully.
- Feature generation is reproducible.
- Leakage tests pass.
- Metrics are logged to MLflow.
- Baseline performance is documented.

---

## Phase 4: Advanced Machine Learning and Uncertainty Quantification

This phase should demonstrate statistical and ML depth.

### 4.1 Remaining Useful Life prediction

Implement at least one temporal deep learning model using PyTorch.

Choose between:

- 1D Temporal CNN.
- LSTM.

Do not implement multiple deep architectures without a clear experimental reason.

Compare the advanced model against the Phase 3 baselines.

### 4.2 Anomaly detection

Implement one or more suitable methods:

- Isolation Forest.
- Autoencoder.

Use anomaly scores to identify unusual machine behaviour.

Evaluate the detection approach using available labels or carefully defined synthetic scenarios.

Do not treat an anomaly score as a calibrated failure probability.

### 4.3 Survival analysis

Use Lifelines to implement one suitable survival model:

- Weibull AFT.
- Cox Proportional Hazards.

Estimate failure probability over time.

Explain censoring assumptions and distinguish survival probabilities from RUL point predictions.

### 4.4 Uncertainty quantification

Implement split-conformal prediction intervals.

Report:

- Target coverage.
- Observed coverage.
- Average interval width.
- Coverage across relevant machine groups.
- Coverage under different operating conditions.

Evaluate whether the interval is useful, not merely whether its coverage is close to the target.

### 4.5 Explainability

Integrate SHAP for the tree-based model.

Show:

- Global feature importance.
- Individual prediction explanations.
- Sensor contributions.
- Limitations of the explanations.

### 4.6 Model comparison

Create an ablation and comparison table covering:

- Baseline models.
- Tree-based models.
- Deep learning model.
- Survival model.
- Anomaly detection approach where applicable.

Acceptance criteria:

- Every advanced model is compared against a meaningful baseline.
- Evaluation uses an untouched test dataset.
- Model selection is based on validation results.
- Test results are not used repeatedly to tune models.
- Prediction interval coverage and width are reported.
- All experiments are tracked in MLflow.

Important: Do not force advanced models to outperform simpler models. If a baseline performs better, report the result honestly and explain why.

---

## Phase 5: Heuristic Maintenance Scheduling and Business Backtesting

The purpose of this phase is to demonstrate how predictive analytics can influence operational decisions.

Do not introduce formal constraint solvers or complex optimization frameworks.

### 5.1 Cost model

Define configurable costs for:

- Planned maintenance.
- Unplanned equipment failure.
- Downtime per hour.
- Maintenance intervention.
- False maintenance alarms.

All cost assumptions must be documented and configurable.

Use synthetic cost assumptions unless authoritative public values are available.

### 5.2 Heuristic maintenance scheduler

Implement a deterministic scheduler.

The scheduler should:

1. Receive machine-level predictions.
2. Calculate risk using the available RUL estimates and conformal lower bounds.
3. Rank machines by maintenance urgency.
4. Consider a fixed maintenance crew capacity per shift.
5. Select the highest-priority machines.
6. Defer machines below a configurable risk threshold.

No Z3, formal optimization, or agentic decision-making.

The scheduler must remain explainable and reproducible.

### 5.3 Baseline maintenance strategies

Implement:

- Run-to-failure.
- Fixed-interval maintenance.
- ML-informed heuristic maintenance.

### 5.4 Historical backtesting

Replay machine histories chronologically.

At each simulated decision point, use only information available at that time.

Prevent future-data leakage in maintenance decisions.

Compare the strategies using:

- Total maintenance cost.
- Unplanned failures.
- Downtime hours.
- False-alarm rate.
- Maintenance interventions.
- Average lead time before failure.

### 5.5 Business evaluation

Create comparison tables and visualizations.

Explain:

- Whether the ML-informed approach reduces simulated operational cost.
- Whether it increases planned interventions.
- Whether it reduces unplanned downtime.
- What trade-offs exist.

Acceptance criteria:

A single command must reproduce the backtest and generate a comparison report.

The scheduler's limitations must be documented:

- Greedy prioritization.
- No machine dependency handling.
- Fixed crew capacity assumptions.
- No global optimality guarantee.
- Dependence on prediction quality and cost assumptions.

---

## Phase 6: Model Serving and API

Implement FastAPI endpoints.

### Required endpoints

`/predict`

Return:
- Machine ID.
- Predicted RUL.
- Risk score.
- Prediction interval.
- Model version.

`/health`

Return service and model availability.

`/explain`

Return relevant SHAP explanations.

`/schedule`

Return the heuristic maintenance priority list.

### Additional requirements

- Pydantic request and response validation.
- Error handling.
- Model loading.
- Model version identification.
- Model version is the MLflow run ID in API responses; no model switching is implemented.
- API tests.
- Dockerfile.
- Docker Compose.

Acceptance criteria:

The API must run inside a container and respond correctly to documented sample requests.

All endpoints must have automated tests.

---

## Phase 7: Interactive Dashboard

Build a Streamlit or Plotly Dash dashboard.

### Fleet overview

Display:

- Total machines.
- Healthy machines.
- Machines requiring attention.
- High-risk machines.
- Predicted RUL distribution.
- Maintenance priorities.

### Individual machine analysis

Display:

- Sensor trends.
- Historical degradation.
- Predicted RUL.
- Prediction intervals.
- Failure risk.
- SHAP explanations.

### Business analytics

Display:

- Downtime trends.
- Maintenance expenditure.
- Failure trends.
- Strategy comparison.
- Historical backtesting results.

### Model analytics

Display:

- Model run ID.
- Model evaluation metrics.
- Interval coverage over time.
- Drift summaries.
- Data-quality summaries.

Drift, calibration-over-time, and data-quality monitoring panels remain clearly
marked placeholders until Phase 8 implements those monitoring outputs. Do not
invent or simulate monitoring results in the dashboard.

Acceptance criteria:

Dashboard runs locally and through Docker Compose.

Save genuine application screenshots in `docs/`.

---

## Phase 8: Monitoring, Data Quality and Model Reliability Analytics

The goal is to show how a deployed prediction system is observed and judged
over time: is incoming data changing, are predictions still accurate, and are
uncertainty intervals still honest?

This phase observes and reports. It does not deploy, promote, switch, or roll
back models.

### 8.1 Experiment tracking and lineage

Use MLflow for tracking only. Record for every training run:

- Parameters and training configuration.
- Dataset identifier and version.
- Feature schema.
- Evaluation metrics, including interval coverage and width.
- Code revision and run ID.

The API loads the model from a single configured run ID or artifact path.
There is no staging, production tagging, or promotion workflow.

### 8.2 Data drift monitoring

Integrate Evidently to monitor:

- Sensor distributions.
- Operating conditions.
- Feature distributions.
- Missing-value rates.

Generate reproducible drift reports comparing a reference dataset with
controlled synthetic distribution shifts (gradual sensor offset, changed
operating-condition mix, increased missingness). This is a data-monitoring
experiment only.

### 8.3 Prediction monitoring

Track over simulated time:

- Predicted RUL distribution.
- Risk-score distribution.
- Prediction interval widths.
- Machine-level prediction trends.

### 8.4 Performance monitoring with delayed labels

When true failure times become known after the fact:

- Compute realized RUL errors (RMSE, MAE, asymmetric score).
- Compare against baseline models over the same period.
- Track error by machine group and operating condition.
- Flag statistically meaningful degradation, for example a rolling window
  versus the validation period.

### 8.5 Calibration monitoring

Track conformal interval coverage and average width over time using delayed
labels. Report when observed coverage falls meaningfully below target and
discuss why, including distribution shift and violations of the exchangeability
assumption.

### 8.6 Drift versus degradation analysis

Explicitly separate the two:

- Drift is a change in the data.
- Degradation is a measured increase in error or loss of coverage.

Include deterministic, measured scenarios showing all four cases:

1. No drift and no degradation.
2. Drift without degradation.
3. Degradation without obvious drift.
4. Drift with degradation.

Each scenario must be supported by measured drift and performance/calibration
results, not only its label. Drift metrics and degradation metrics must be
evaluated independently. Drift alone must never be reported as proof of poor
performance. Where machine counts make evidence weak, report uncertainty or an
inconclusive result rather than overclaiming.

### 8.7 Data-quality monitoring

Report incoming data quality over time: schema violations, out-of-range values,
missingness, stuck sensors, and duplicate timestamps.

### 8.8 Retraining recommendation report

Produce a deterministic, human-readable report that states:

- Which monitoring condition was triggered.
- Supporting evidence (drift metrics, error trend, coverage trend).
- A suggested action: no action, investigate, or retrain.

The recommendation is advisory; a person decides. Retraining, if done, is
simply re-running the normal training pipeline. Its metrics are tracked in
MLflow and compared offline in a report. No automated approval, promotion, or
replacement logic.

### 8.9 CI and dashboard integration

Extend GitHub Actions with fast monitoring tests and integration coverage,
Ruff, Mypy, and a Docker build without materially increasing CI runtime. Provide
a small/fast monitoring configuration for CI and a full configuration for local
runs and the dashboard. The dashboard reads saved monitoring outputs, does not
train models or invent monitoring values, and replaces the Phase 7 placeholders
only with measured results.

Acceptance criteria:

A single documented command runs a reproducible monitoring scenario:

1. Train an initial model and log it in MLflow.
2. Simulate incoming data in time windows.
3. Apply a controlled distribution shift.
4. Produce a drift report.
5. Compute delayed-label performance and calibration for each window.
6. Show at least one case of drift without degradation and one of real
   degradation.
7. Generate the retraining recommendation report.

All reported results must come from actual runs and saved outputs. The monitoring
model remains separate from the serving model. No Docker engine is required by
the test suite.

---

## Phase 9: Documentation, Reproducibility and Portfolio Presentation

### README

Include:

- Problem statement.
- Industrial relevance.
- Architecture diagram.
- Technology stack.
- Dataset information.
- Data pipeline.
- ML methodology.
- Model comparison.
- Uncertainty evaluation.
- Maintenance backtesting results.
- API documentation.
- Dashboard screenshots.
- Monitoring and reliability analytics.
- Reproduction instructions.
- Known limitations.
- Future improvements.

### Demonstration

Create a 2–3 minute demo video or GIF showing:

1. Fleet data and analytics.
2. RUL predictions.
3. Prediction intervals.
4. Maintenance prioritization.
5. Strategy comparison.
6. Model monitoring.
7. Drift report, delayed-label performance, calibration over time, and
   retraining recommendation.

### Final cleanup

- Remove unused code.
- Verify dependency pins.
- Run all tests.
- Verify Docker setup.
- Check GitHub Actions.
- Ensure documentation matches actual functionality.
- Add an appropriate open-source license.
- Prepare a concise technical results report.

Acceptance criteria:

A new developer must be able to clone the repository, follow the README, reproduce the main experiments, start the API, and view the dashboard.

---

# 6. Evaluation Requirements

These requirements apply throughout the project.

### Data evaluation

- Use machine-level separation.
- Prevent future-data leakage.
- Document missing-value handling.
- Preserve reproducibility.

### Model evaluation

- Compare models against baselines.
- Use validation data for model selection.
- Keep test data untouched until final evaluation.
- Report RMSE, MAE, and asymmetric scoring.
- Report prediction interval coverage and width.
- Evaluate performance across machine groups and operating conditions.

### Statistical evaluation

- Explain survival analysis assumptions.
- Explain censoring.
- Evaluate conformal coverage.
- Document limitations of uncertainty estimates.

### Business evaluation

Report ML metrics alongside:

- Maintenance expenditure.
- Downtime.
- Unplanned failures.
- False alarms.
- Maintenance interventions.
- Lead time.

### MLOps evaluation

Demonstrate:

- Experiment tracking.
- Lineage.
- Drift monitoring.
- Data-quality monitoring.
- Delayed-label performance monitoring.
- Calibration monitoring.
- Retraining recommendation.

Never claim production readiness based solely on a successful local experiment.

---

# 7. Definition of Done

The project is complete only when:

- All ten phases, including Phase 0, are completed.
- All acceptance criteria are met.
- Tests pass.
- GitHub Actions is green.
- Model evaluation is reproducible.
- Backtesting is reproducible.
- API and dashboard work.
- Monitoring scenario is reproducible and documented.
- Documentation is complete.
- PROGRESS.md is accurate.
- All reported results are backed by actual experiments.

---

# 8. Quality Rules

- Type hints on public functions.
- Docstrings on public functions and classes.
- No hard-coded paths.
- Configuration through YAML.
- Fixed random seeds.
- Pinned dependencies.
- No notebooks inside `src/`.
- Modular architecture.
- Meaningful exception handling.
- Unit and integration tests.
- No fabricated results.
- No unnecessary abstractions.
- No unnecessary AI components.

Use commit messages referencing the phase and task:

`P0: initialize project structure`

`P1: implement synthetic fleet generator`

`P3: add rolling feature engineering`

`P8: implement monitoring and reliability analytics`

---

# 9. PROGRESS.md Template

Create and maintain the following structure:

```markdown
# Progress Tracker

**Last updated:** YYYY-MM-DD
**Current phase:** Phase X: name
**Overall status:** X of 10 phases complete

## Status Summary

- Done:
- In progress:
- Next up:
- Blockers:

## Phase Checklist

Copy the tasks from PROJECT_SPEC.md.
Mark completed tasks only after verification.

## Decisions

| Date | Decision | Reason |
|------|----------|--------|

## Results Log

| Date | Experiment / Run | Key metrics | MLflow run ID |
|------|------------------|-------------|---------------|

## Session Log

### YYYY-MM-DD

- What was done:
- Tests executed:
- Results:
- Problems encountered:
- What is left for next session:

## Known Issues / Limitations

-
```

Update this file after every meaningful task.

Never leave inaccurate progress information.

---

# 10. Initial Instructions: Begin Phase 0

Before writing code:

1. Read this entire specification.
2. Inspect the current repository, if one exists.
3. Confirm the project directory and environment.
4. Explain the proposed Phase 0 architecture.
5. Identify the dependencies and their purpose.
6. Identify the files that will be created.
7. Explain the Phase 0 acceptance criteria.
8. Identify any setup risks on Windows.

Do not begin Phase 1.

Do not implement the entire application at once.

Wait for my approval before making the initial Phase 0 changes.

After approval, implement Phase 0, execute the checks, update PROGRESS.md, and report the actual results.

**The priority throughout development is engineering quality, statistical correctness, reproducibility, and demonstrable understanding — not the number of features or lines of generated code.**