# Progress Tracker

**Last updated:** 2026-11-14
**Current phase:** Phase 8.9: CI and dashboard integration
**Overall status:** Phases 0-7 locally verified; Phases 8.1-8.8 are implemented and focused tests pass

## Status Summary

- Done: Phases 0-7 are locally verified; Phases 8.1-8.8 add lineage, drift, prediction, delayed-label performance, calibration, measured drift/degradation scenarios, data-quality monitoring, and an advisory recommendation.
- Next up: Phase 8.9 CI/dashboard integration and the reproducible monitoring scenario.
- Blockers: By user decision, interval coverage and width remain unavailable in baseline, serving, and maintenance-backtest runs because those workflows do not currently calculate them; no new metric calculations will be added under 8.1. GitHub Actions run 43115068066 passed on the earlier main commit `6560fa7`; it does not include this working tree.

## Phase Checklist

### Phase 0: Project Setup

- [x] Create the minimal Python package and test layout.
- [x] Configure Python 3.11 and pinned development dependencies.
- [x] Configure Ruff and Mypy.
- [x] Configure pre-commit local hooks.
- [x] Configure Pytest and verify test discovery/execution.
- [x] Create initial GitHub Actions workflow.
- [x] Create README skeleton.
- [x] Create this progress tracker.
- [x] Configure `.gitignore`.

Acceptance criteria:

- [x] Local editable install succeeds in a fresh Python 3.11 virtual environment.
- [x] Linting and type-checking work.
- [x] CI successfully executes.
- [x] Tests can be discovered and executed.
- [x] No application functionality has been introduced.

### Phases 1-9

#### Phase 1: Data Engineering and Synthetic Fleet

- [x] Download and load NASA C-MAPSS.
- [x] Validate the NASA dataset schema and identify machine, cycle, setting, and sensor columns.
- [x] Handle missing and invalid observations.
- [x] Store validated datasets in Parquet.
- [x] Document the dataset structure and processing commands.
- [x] Generate configurable synthetic degradation curves and machine characteristics.
- [x] Support different machine lifetimes, operating conditions, noise, missing observations, maintenance events, and failures.
- [x] Use reproducible random seeds.
- [x] Implement schema, range, duplicate, missingness, cycle-order, and machine-consistency validation.
- [x] Add automated tests for NASA loading and synthetic generation.

Acceptance criteria:

- [x] A documented single command generates and validates at least 100 synthetic machines.
- [x] NASA loading and synthetic generation have automated tests.

#### Phase 2: Exploratory Data Analysis and Reliability Analytics

- [x] Add an executable notebook for sensor behaviour, trends, operating conditions, machine variability, and sensor correlations.
- [x] Add reusable lifetime, failure-rate, maintenance-frequency, MTBF, and downtime metrics with explicit data-availability rules.
- [x] Add a CLI command that writes JSON reliability reports.
- [x] Execute the notebook and include visualizations with at least five measured findings.
- [x] Document findings, commands, limitations, and reproducibility.
- [x] Add unit, CLI, and notebook integration tests.

Acceptance criteria:

- [x] EDA notebook executes successfully, including in headless CI.
- [x] Reliability metrics are calculated in reusable Python modules.
- [x] Findings are reproducible through the notebook and CLI.
- [x] Findings are measured from available NASA and synthetic datasets; unsupported MTBF and downtime are explicitly unavailable.
- [x] GitHub Actions passes for the Phase 2 commit.

#### Phase 3: Feature Engineering and Baseline Models

- [x] Create reproducible rolling means, rolling standard deviations, trends, sensor variation, health indicators, operating-condition, and time-window features.
- [x] Keep each trajectory's features causal: use only current and past observations; exclude RUL and event labels from predictors.
- [x] Split train, validation, and test by whole machine with deterministic 70%/15%/15% assignment.
- [x] Train constant-mean, linear-regression, and random-forest RUL baselines with train-only imputation and scaling.
- [x] Calculate RMSE, MAE, asymmetric NASA score, and test metrics by machine type and operating context where available.
- [x] Log model parameters, metrics, and fitted models to MLflow.
- [x] Document measured held-out baseline performance and add leakage, split, metric, CLI, and MLflow tests.

Acceptance criteria:

- [x] Baselines train successfully on NASA FD002 and the seeded synthetic fleet.
- [x] Feature generation and machine assignment are reproducible.
- [x] Automated tests verify no future-feature leakage and no machine overlap.
- [x] Metrics are logged to local MLflow experiments.
- [x] Baseline performance and split scope are documented.

#### Phase 4: Advanced Machine Learning and Uncertainty Quantification

- [x] Add one causal unidirectional PyTorch LSTM and compare it with Phase 3 RUL baselines.
- [x] Select RUL models using validation RMSE; keep the machine-disjoint test partition for final evaluation.
- [x] Add train-fitted Isolation Forest preprocessing and deterministic injected-sensor-shift evaluation.
- [x] Add landmark Weibull AFT survival analysis with explicit event/censoring rules and a Kaplan-Meier baseline.
- [x] Add machine-trajectory split-conformal RUL intervals and report row, trajectory, and group coverage plus width.
- [x] Add global and individual SHAP explanations with an explicit additivity check.
- [x] Track model, calibration, survival, anomaly, and explanation experiments in MLflow.
- [x] Document measured synthetic-fleet results and FD002 tracked evaluation results with limitations.
- [x] Add unit and end-to-end integration tests for uncertainty, survival, model training, reports, and MLflow.

Acceptance criteria:

- [x] Advanced models are compared with meaningful baselines.
- [x] Machine-disjoint test data is evaluated only after validation-based selection.
- [x] Conformal target/observed coverage and interval width are reported.
- [x] All advanced workflow runs are tracked in MLflow; the synthetic CLI completed end to end.
- [x] Local automated checks pass.
- [x] GitHub CI passes for the current main commit; run 43115068066 executed the full test, quality, Compose, and Docker build workflow.

#### Phase 5: Heuristic Maintenance Scheduling and Business Backtesting

- [x] Add configurable illustrative intervention, planned-maintenance, failure, downtime, and false-alarm costs.
- [x] Schedule deterministically by conformal lower-RUL risk with per-shift capacity constraints.
- [x] Train on training machines, calibrate on validation machines, and replay policies on held-out test machines.
- [x] Compare run-to-failure, recurring fixed-interval, and ML-informed heuristic policies chronologically.
- [x] Report interventions, false alarms, assumed prevented failures, unplanned failures, downtime, lead time, and total simulated cost.
- [x] Write a JSON report, comparison plot, and MLflow run through a dedicated CLI command.
- [x] Add unit and integration coverage; document defaults, limitations, and measured results.

Acceptance criteria:

- [x] No held-out RUL or event labels are available to the scheduling policy; outcomes are scored only after each action decision.
- [x] Capacity deferrals remain eligible for later scheduling, and fixed-interval service recurs per machine.
- [x] Censored or insufficiently observed interventions are reported as unverified, not false alarms.
- [x] Local automated checks pass and the complete synthetic-fleet CLI run is tracked in MLflow.
- [x] GitHub CI passes for the current main commit; run 43115068066 executed the full test, quality, Compose, and Docker build workflow.

#### Phase 6: Model Serving and API

- [x] Add validated `/health`, `/predict`, `/explain`, and `/schedule` endpoints.
- [x] Load one versioned MLflow run ID from a serving manifest and report it as the model version.
- [x] Train a serving random-forest artifact and calibrate trajectory intervals using machine-disjoint fit/calibration partitions.
- [x] Reuse the causal feature builder, conformal interval, and deterministic Phase 5 scheduler for API responses.
- [x] Add Pydantic request/response validation and explicit malformed-input/model-unavailable errors.
- [x] Add endpoint integration tests and sample requests.
- [x] Add Dockerfile, Compose service, model-health check, and deployment documentation; local Compose configuration and Docker image build passed.
- [x] Build and run the Docker image with all endpoints responding to documented sample requests.

Acceptance criteria:

- [x] All endpoints have automated tests, including prediction, explanations, priorities, and unavailable-model behavior.
- [x] Local API tests load an actual MLflow artifact and verify request/response contracts.
- [x] Locally pass both `docker compose config` and `docker buildx build --load`.
- [x] The container responds correctly to documented sample requests.
- [x] GitHub CI passes for the current main commit; run 43115068066 executed the full test, quality, Compose, and Docker build workflow.

#### Phase 7: Artifact-Backed Dashboard

- [x] Verify `streamlit==1.45.1` installation against the Python 3.11 pinned environment.
- [x] Implement fixture-tested data access for saved Parquet/JSON artifacts and the Phase 6 API.
- [x] Implement fleet overview, individual machine analysis, business analytics, and model analytics pages.
- [x] Add Streamlit AppTest smoke tests for every page; tests must not require Docker.
- [x] Label the data source on every page and keep synthetic fleet and NASA C-MAPSS outputs distinct.
- [x] Show an API-unavailable message while retaining artifact-backed page content.
- [x] Keep Phase 8 drift, calibration-over-time, and data-quality panels as explicit placeholders.
- [x] Add a dashboard Compose service that waits for the healthy API; update README instructions.
- [x] Run the dashboard locally and capture genuine screenshots in `docs/`.

Current progress:

- Data access is fixture-tested, with allowlisted Parquet/JSON loading and API access. Fleet overview, individual machine analysis, business analytics, and model analytics each have an AppTest smoke test. These tests run without Docker and confirm artifact-backed views survive an unavailable API.
- Synthetic-fleet and NASA C-MAPSS sources are labelled and kept separate. The model page displays only saved Phase 4 test results/run IDs and one-shot interval results; drift, calibration-over-time, and data-quality panels remain Phase 8 placeholders.
- `streamlit==1.45.1` installs cleanly with the pinned Python 3.11 environment; `pip check` is clean. The dashboard Compose service depends on healthy API and mounts processed data and reports read-only.
- Optimized fleet priority requests retain the first observation (sensor-variation baseline) and the configured latest five-cycle feature window. The feature fixture test confirms this produces the same current features as the full history. The API computes feature rows for all fleet machines together; a live Compose `/schedule` request returned 100 priorities in 1.16 seconds.
- Both final API and dashboard Docker images built successfully; `docker compose up --detach --wait --no-build` reached healthy API and running dashboard. All four dashboard pages rendered from saved outputs in the live Compose app. Genuine screenshots are saved in `docs/`.
- The prior full test run exposed an environment-dependent notebook PNG failure. The EDA notebook now explicitly selects the inline backend before importing pyplot; the complete suite passes with 66 tests. Remaining 251 warnings are non-fatal upstream deprecations: 244 Matplotlib/PyParsing, 3 Pydantic `min_items`, 1 Pydantic class-based config, 1 Starlette/AnyIO `BlockingPortal`, 1 Windows Proactor/ZMQ `add_reader`, and 1 Altair. No warning currently blocks Phase 7 acceptance.
- Final local quality checks pass: Ruff lint and format (59 files), strict Mypy (39 source files), `pip check`, `docker compose config`, and `git diff --check`. GitHub Actions has not run on this working tree; no Git state-changing operation was performed.

#### Phase 8: Monitoring, Data Quality and Model Reliability Analytics

- [ ] Record MLflow parameters, dataset/version, feature schema, metrics, code revision, and run lineage.
- [ ] Compare reference data with controlled sensor-offset, operating-condition-mix, and missingness changes using Evidently.
- [ ] Track prediction, risk-score, interval-width, and machine-level trends over simulated windows.
- [ ] Calculate delayed-label performance and compare it with baselines, including group/condition metrics and degradation checks.
- [ ] Monitor interval coverage and width against the conformal target over time.
- [ ] Demonstrate all four drift/degradation combinations; drift alone is not evidence of performance loss.
- [ ] Generate a deterministic advisory report with evidence and action (no action, investigate, or retrain).
- [ ] Report schema/range issues, missingness, stuck sensors, and duplicate timestamps.
- [ ] Extend CI with tests and Docker build for the monitoring workflow.

Acceptance criteria:

- [ ] A documented single command runs the monitoring scenario and logs the initial model in MLflow.
- [ ] The scenario simulates windows and controlled drift and produces a reproducible drift report.
- [ ] Delayed-label performance and calibration are reported for each window.
- [ ] Results show drift without degradation and actual degradation.
- [ ] A deterministic advisory retraining recommendation is generated.
- [ ] Every reported result is from an actual run.

## Decisions

| Date | Decision | Reason |
|------|----------|--------|
| 2026-11-11 | Use pip with pinned Phase 0 development dependencies. | `uv` is not installed; pip is available and avoids adding a bootstrap dependency. |
| 2026-11-11 | Use the project-specific repository and run GitHub Actions for CI. | User requested that GitHub Actions not be deferred and specified the target repository. |
| 2026-11-11 | Use NumPy/pandas for data generation and tabular processing, PyYAML for configuration, and PyArrow-backed Parquet for storage. | These are the minimum phase-specific dependencies needed; defer DuckDB until analytical queries are introduced. |
| 2026-11-11 | Preserve sensor NaNs and report their frequency instead of imputing them during ingestion. | Missingness is a data-quality concern; imputation belongs in a later modeling decision. |
| 2026-11-11 | Treat NASA test RUL as an evaluation target and keep test failure events unobserved. | Test labels describe future failure time and must not be confused with observed telemetry/features. |
| 2026-11-11 | Compute reliability rates from observed cycle-record exposure and only report MTBF or downtime when the required repeated events or measured hours exist. | Avoid implying unsupported reliability estimates from single terminal failures or event counts. |
| 2026-11-11 | Pin Matplotlib and notebook execution dependencies in the development extra. | Keep EDA visualizations and headless notebook execution reproducible without adding runtime dependencies. |
| 2026-11-11 | Pin scikit-learn 1.6.1 and MLflow 2.22.0 for the baseline models and experiment tracking; pin Pydantic 2.11.7 for a reproducible compatible MLflow dependency set. | Keep model training, metric logging, artifact serialization, and dependency resolution consistent. |
| 2026-11-11 | Fit sensor imputation/scaling and numeric operating-condition bin edges on training machines only. | Prevent held-out observations from influencing preprocessing or group definitions. |
| 2026-11-12 | Use one unidirectional LSTM, selected against Phase 3 baselines on validation RMSE. | Meet the temporal-model requirement without adding multiple deep architectures or tuning against test data. |
| 2026-11-12 | Calibrate conformal intervals on each calibration machine's maximum absolute cycle residual. | Treat a machine trajectory, rather than correlated cycle rows, as the calibration unit. |
| 2026-11-12 | Use TreeExplainer tree-path-dependent perturbation after interventional SHAP failed its additivity check on FD002. | The corrected explanation method passes an explicit contribution-sum check on synthetic and saved FD002 models. |
| 2026-11-12 | Treat maintenance costs and prevention effectiveness as configurable synthetic assumptions; permit only one ML-informed intervention per machine. | The project has no measured costs or post-maintenance counterfactual telemetry, so the backtest must not imply real-world treatment effects or repeatedly apply unsupported predictions. |
| 2026-11-12 | Serve one calibrated random-forest MLflow run, with its run ID and feature schema in a manifest; do not use model registry stages or switching. | Keep model identity explicit and use machine-disjoint calibration metadata without introducing promotion or rollback workflows. |

## Results Log

| Date | Experiment / Run | Key metrics | MLflow run ID |
|------|------------------|-------------|---------------|
| 2026-11-11 | Phase 0 local setup checks | Fresh `.venv` install succeeded; Pytest 1 passed; Ruff lint passed; Ruff format check passed; Mypy passed; pre-commit config validation passed; pip check passed. | N/A |
| 2026-11-11 | Phase 0 GitHub Actions | CI run 1 passed on Ubuntu/Python 3.11; install, pytest, Ruff lint, Ruff format, and Mypy steps all succeeded. [Run 43009344986](https://github.com/sravasti12/Equipment-Reliability-Intelligence-MLOps-Platform/actions/runs/43009344986). | N/A |
| 2026-11-11 | Phase 1 synthetic fleet | Default command generated a validated Parquet file with 11,935 rows, 100 machines, and 5 sensors; observed sensor missing fraction 0.020176; lifetimes ranged from 80 to 160 cycles. | N/A |
| 2026-11-11 | Phase 1 NASA C-MAPSS | Downloaded the public NASA archive and processed train/test for FD001-FD004. Rows/machines: FD001 20,631/100 and 13,096/100; FD002 53,759/260 and 33,991/259; FD003 24,720/100 and 16,596/100; FD004 61,249/249 and 41,214/248. All eight outputs contain 21 sensor columns and passed validation; no sensor missing values were observed. | N/A |
| 2026-11-11 | Phase 1 automated checks | Pytest: 16 passed; Ruff lint/format, Mypy, `pip check`, pre-commit config validation, and all four pre-commit hooks passed. [GitHub Actions run 43014898267](https://github.com/sravasti12/Equipment-Reliability-Intelligence-MLOps-Platform/actions/runs/43014898267) passed all steps. | N/A |
| 2026-11-11 | Phase 2 EDA | Executed the notebook against NASA FD001/FD002 and seed-42 synthetic data. It produced seven visualized findings: FD001 median observed lifetime 199 cycles (IQR 177-229.2); five displayed RUL slopes average -1.000; sensor 3 largest plotted decile shift +13.060; strongest absolute sensor correlation sensor 9/14 (r=0.963); setting-1 SD FD002 14.7474 vs FD001 0.0022; synthetic type B maintenance rate 8.802/1,000 cycles; missing fraction NASA 0.000% vs synthetic 2.018%. | N/A |
| 2026-11-11 | Phase 2 reliability reports | NASA FD001 report: 20,631 rows and 100 machines. Seed-42 synthetic report: 11,935 rows and 100 machines. MTBF is unavailable for both because no machine has recurrent observed failures; downtime is unavailable because neither dataset has measured downtime hours. | N/A |
| 2026-11-11 | Phase 2 automated checks | Pytest: 27 passed; Ruff lint/format, Mypy, `pip check`, and all four pre-commit hooks passed. The notebook emitted a non-fatal Windows Proactor/ZMQ selector warning. [GitHub Actions run 43023176262](https://github.com/sravasti12/Equipment-Reliability-Intelligence-MLOps-Platform/actions/runs/43023176262) passed all steps. | N/A |
| 2026-11-11 | Phase 3 NASA FD002 baseline | Held out 39 of 260 machines from the FD002 training file (8,070 test rows; 182/39/39 machines in train/validation/test). Test RMSE/MAE/asymmetric score: constant mean 67.594/56.315/918,376,631; linear 41.230/31.664/6,473,930; random forest 36.086/26.280/3,358,202. Random-forest RMSE ranged 35.105-36.495 over the training-defined setting_1/setting_2 test bins. | `phase3-cmapss-fd002` |
| 2026-11-11 | Phase 3 synthetic baseline | Seed-42 fleet held out 15 of 100 machines (1,772 test rows; 70/15/15 machines in train/validation/test). Test RMSE/MAE/asymmetric score: constant mean 36.909/31.257/118,118; linear 20.026/16.721/15,411; random forest 18.903/14.444/14,827. Random-forest RMSE by machine type: type A 23.992, type B 13.806, type C 14.865. | `phase3-synthetic-fleet-final` |
| 2026-11-11 | Phase 3 automated checks | Pytest: 37 passed; Ruff lint/format, Mypy, `pip check`, and all four pre-commit hooks passed. Local MLflow integration test verifies 3 completed runs, logged metrics/model artifacts, and non-negative model predictions. Remaining warnings are from MLflow's Pydantic models and the known Windows notebook/ZMQ event loop integration. GitHub CI awaits the user's push. | N/A |
| 2026-11-12 | Phase 4 seeded synthetic fleet | Full CLI completed on the seed-42 fleet (15 test machines, 1,772 rows). Validation selected random forest; test RMSE/MAE/asymmetric score: constant 36.982/31.330/123,660; linear 19.173/15.595/16,042; random forest 16.726/13.153/9,263; LSTM 22.139/17.541/26,324. At 90% target, row and trajectory coverage were both 100%, with 98.684-cycle average width from 14 calibration machines. Validation selected Kaplan-Meier (test Brier 0.214 vs. Weibull AFT 0.271). Injected-shift test ROC AUC 0.600 / average precision 0.609. SHAP maximum additivity residual: 9.6e-13. | `synthetic-phase4` |
| 2026-11-12 | Phase 4 NASA FD002 evaluation | Logged test metrics for a 39-machine / 8,070-row holdout from the FD002 training file; validation selected random forest (RMSE 35.057). Test RMSE/MAE/asymmetric score: constant 67.597/56.244/1,040,422,794; linear 40.124/31.161/4,819,382; random forest 35.494/25.923/2,770,600; LSTM 44.246/32.093/75,086,389. Target conformal coverage 90%, observed row/trajectory coverage 100%, average width 278.061 cycles (36 calibration machines). Validation selected Kaplan-Meier; test Brier was 0.207 for Kaplan-Meier and 0.197 for Weibull AFT. Injected-shift ROC AUC 0.570 / average precision 0.556. The initial CLI stopped before writing JSON at SHAP; after the fix, direct explanation of the saved FD002 model succeeded for 32 rows with max additivity residual 8.32e-10. | `cmapss-fd002-phase4` |
| 2026-11-12 | Phase 4 FD002 complete CLI rerun | Full `train-advanced` CLI completed on the 53,759-row FD002 training Parquet and wrote `reports/generated/fd002_advanced.json`. It reproduced the saved held-out results: random forest selected by validation (RMSE 35.057); test RMSE 35.494, MAE 25.923, asymmetric score 2,770,600; 100% observed trajectory coverage against 90% target with 278.061-cycle average width; SHAP and the remaining report artifacts completed. | `ce18ca5a63ff43b1803eeded4efc8bc5` |
| 2026-11-12 | Phase 4 automated checks | Full Pytest: 43 passed in 215.57 s; Ruff lint and format, Mypy (24 source files), `pip check`, and `git diff --check` passed. The focused advanced integration test completed the synthetic CLI and verified MLflow runs, plots, and SHAP additivity. The full FD002 CLI was not rerun after the SHAP fix; Phase 4 CI awaits push. | N/A |
| 2026-11-12 | Phase 5 synthetic maintenance backtest | Seed-42 fleet: 70 training / 15 calibration / 15 test machines; validation RMSE 21.427 cycles and 80% target trajectory-conformal bound 49.664 cycles. Run-to-failure: 585,000 simulated units, 15 failures, 360 downtime hours. Fixed interval: 198,000 units, 22 interventions, 10 false alarms, 12 assumed prevented failures, 3 failures, 72 downtime hours, 24.417-cycle mean lead. ML-informed: 612,000 units, 15 interventions, 14 false alarms, 1 assumed prevented failure, 14 failures, 336 downtime hours, 31-cycle mean lead. Results are assumption-dependent and not causal savings. | `e415d268d949484c9e763af19eda5827` |
| 2026-11-12 | Phase 5 automated checks | Full Pytest: 49 passed. Final Ruff lint, format check (44 files), strict Mypy (27 source files), `pip check`, and `git diff --check` all passed. GitHub CI has not run for local Phase 4/5 changes. | N/A |
| 2026-11-12 | Phase 6 serving model preparation | Seed-42 synthetic fleet; random forest fitted on training machines and calibrated on validation machines at 90% target. Manifest references MLflow run `05ba7e06f54c4a8fba45860a4bf2053d`. | `05ba7e06f54c4a8fba45860a4bf2053d` |
| 2026-11-12 | Phase 6 API integration tests | Full Pytest: 51 passed (2 API integration tests). Tests loaded an MLflow artifact and verified `/health`, `/predict`, `/explain` (SHAP additivity), `/schedule` capacity/priority ordering, duplicate-cycle rejection, and unavailable-model HTTP 503. After fixing local Windows MLflow file-URI handling for a Linux container, the 2 API integration tests passed again. Final Ruff lint/format, strict Mypy, `pip check`, `git diff --check`, and sample JSON validation passed. Docker/Compose verified locally: `docker compose config` and `docker buildx build --load --tag reliability-platform-api:phase6 .` succeeded; `docker compose up --build --wait` reached healthy. The documented sample returned RUL 120.276 cycles, interval [53.922, 186.631], risk 0, and SHAP additivity residual 6.96e-13. | N/A |
| 2026-11-12 | Phase 6 GitHub Actions | Run 43114036981 failed at `ruff format --check`: CI reported `src/reliability_platform/api/app.py` would be reformatted. Applied `ruff format` locally; full repository format check now passes (50 files). CI remains pending until this fix is pushed. | N/A |
| 2026-11-12 | Phase 6 CI formatting fix | Run 43115068066 passed on commit `6560fa7b63ad804ee5076bb9174bf5c857ae5050`. Every workflow step succeeded: dependency install, full pytest, Ruff lint/format, Mypy, Compose config, and Docker image build. | N/A |
| 2026-11-12 | Resume verification | Worktree matches `origin/main` at `6560fa7`; only the user-provided `PHASE_8_REVISION.md` is untracked. Full Pytest: 51 passed (250 warnings); Ruff lint and format passed (50 files); strict Mypy passed (32 source files); Docker 29.6.2, Compose config, and `docker build --tag reliability-platform-api:phase6-verification .` all passed. `PROJECT_SPEC.md` was aligned with the supplied monitoring-only Phase 8 revision; no Phase 7 implementation was started. | N/A |
| 2026-11-12 | Phase 7 data access (verified) | Fixture tests: 9 passed; Ruff lint/format, Mypy, and `pip check` passed. Streamlit 1.45.1 imports on Python 3.11.4; pip reports all pinned project requirements satisfy its dependencies. Earlier full-suite warnings: 244 Matplotlib/PyParsing deprecations, 3 Pydantic `min_items` deprecations, 1 Pydantic class-based-config deprecation, 1 Starlette/AnyIO `BlockingPortal` deprecation, and 1 Windows Jupyter/ZMQ Proactor `add_reader` runtime warning (250 total; none failed tests). | N/A |
| 2026-11-12 | Phase 7 fleet and machine pages | AppTest plus data-access suite: 11 passed. Both pages render fixture-backed data and source labels while the prediction API is unavailable. Ruff lint/format and Mypy passed. Streamlit emitted one upstream Altair deprecation warning during the chart smoke test. | N/A |
| 2026-11-12 | Phase 7 business and model pages | AppTest plus data-access suite: 13 passed. The business page uses only the saved synthetic maintenance report; the model page uses saved evaluation reports and retains the three Phase 8 placeholders without generating monitoring values. Ruff lint/format and Mypy passed. | N/A |
| 2026-11-12 | Phase 7 full project checks | Full Pytest: 63 passed, 1 failed, 251 warnings in 534.99 s. The only failure is the existing EDA notebook test expecting an `image/png` output; the executed notebook produced text/Markdown but no PNG with the local TkAgg Matplotlib backend. Ruff lint/format, strict Mypy (39 source files), `pip check`, and `git diff --check` passed. | N/A |
| 2026-11-14 | Phase 7 final local verification | Full Pytest: 66 passed, 251 warnings in 161.89 s. Making the EDA notebook select `%matplotlib inline` before importing pyplot fixed the nondeterministic PNG assertion; the targeted test also passed with `MPLBACKEND=TkAgg`. Ruff lint and format (59 files), strict Mypy (39 source files), `pip check`, and `git diff --check` passed. | N/A |
| 2026-11-14 | Phase 7 Compose runtime | Final API and dashboard images built successfully; Compose API health and dashboard startup passed. The live `/schedule` request returned 100 priorities for the synthetic fleet in 1.16 s. All four pages rendered from saved artifacts; screenshots saved in `docs/`. Compose config and read-only mounts were verified. | Serving model `05ba7e06f54c4a8fba45860a4bf2053d` |

## Session Log

### 2026-11-11

- What was done: Implemented NASA C-MAPSS loading, seeded synthetic fleet generation, data validation, Parquet I/O, CLI commands, configuration, and documentation.
- Tests executed: 16 pytest tests; Ruff lint and format; Mypy; `pip check`; pre-commit config validation and all hooks; default synthetic CLI; all eight NASA C-MAPSS subset/split processing and validation runs.
- Results: All local tests and quality checks passed. Synthetic output contains 100 machines and 11,935 rows. All FD001-FD004 train/test datasets passed validation. GitHub Actions run 43014898267 passed all steps.
- Problems encountered: Pytest's default system temp directory returned access denied on Windows; configuring `.pytest_tmp` inside the project made standard test execution pass. NASA archive contains a nested ZIP; README instructions account for both extraction steps.
- What is left for next session: Phase 2 was approved and completed; Phase 3 requires user approval.

- What was done: Added reusable reliability metrics and JSON reporting, an executable EDA notebook with seven measured visual findings, tests, development dependencies, and documentation.
- Tests executed: 27 pytest tests; Ruff lint/format, Mypy, `pip check`, and all four pre-commit hooks. Executed the notebook with local NASA FD001/FD002 and synthetic data and generated reliability reports for NASA FD001 and the synthetic fleet.
- Results: All local checks passed. Both reports used observed exposure and correctly reported MTBF and measured downtime as unavailable. Seven notebook findings are documented in the README and Results Log. GitHub Actions run 43023176262 passed all steps.
- Problems encountered: On Windows, Jupyter's ZMQ integration emits a non-fatal Proactor event-loop warning and registers a selector thread; notebook execution and tests still pass. The pre-commit system-Python hooks require the project `.venv\Scripts` directory on `PATH`.
- What is left for next session: Phase 2 is complete; request approval before Phase 3.

- What was done: Added causal cycle-level features, seeded machine-disjoint splits, three RUL baselines, train-fitted preprocessing, test metrics by machine/operating group, MLflow logging, CLI, tests, and measured documentation.
- Tests executed: 37 pytest tests; Ruff lint and format, Mypy, `pip check`, and all four pre-commit hooks. Trained all baselines on NASA FD002 training trajectories and the seed-42 synthetic fleet; confirmed all six MLflow runs completed.
- Results: All local checks passed. NASA FD002 held-out test RMSE: 67.594 (constant), 41.230 (linear), 36.086 (random forest). Synthetic held-out test RMSE: 36.909, 20.026, and 18.903 respectively. Machine and operating-context metrics and asymmetric scores are documented in the README and Results Log.
- Problems encountered: The FD002 baseline run is compute-intensive. MLflow 2.22 emits upstream Pydantic deprecation warnings, and the notebook integration retains the previously documented non-fatal Windows ZMQ/Proactor warning.
- What is left for next session: Phase 3 is complete locally; the user will push the changes and can verify GitHub CI. Request approval before Phase 4.

- What was done: Added the Phase 4 LSTM, Isolation Forest injected-shift evaluation, landmark Weibull AFT and Kaplan-Meier comparison, machine-level conformal intervals, SHAP explanations, comparison report, CLI, MLflow tracking, tests, and documentation.
- Tests executed: Full Pytest suite (43 passed); Ruff lint/format, Mypy, `pip check`, and `git diff --check`. Ran the complete advanced CLI on the seed-42 synthetic fleet. Ran FD002 evaluation and inspected tracked test metrics; validated corrected SHAP directly against its saved random forest model.
- Results: Synthetic CLI completed end to end; random forest was selected on validation (test RMSE 16.726) while the LSTM scored 22.139. Synthetic conformal coverage was 100% for rows and trajectories against a 90% target, at 98.684-cycle average width. Local type, lint, format, dependency, and all test checks pass.
- Problems encountered: The initial full FD002 CLI invocation stopped before writing its JSON because interventional SHAP failed the additivity check (maximum discrepancy 0.750 cycles). Tree-path-dependent SHAP was verified against the same saved FD002 model (32 explanations, maximum residual 8.32e-10), and the complete synthetic CLI plus integration test passed after the fix. The full FD002 CLI was not rerun. MLflow also warns that synthetic string-category input examples do not infer a model signature; model artifacts and metrics were logged. Tests emit upstream Pydantic, Matplotlib, and Pyparsing deprecation warnings.
- What is left for next session: Phase 4 is complete locally; the user will push these changes so GitHub CI can run. Wait for explicit approval before Phase 5.

### 2026-11-12

- What was done: Added configurable deterministic maintenance scheduling, per-machine recurring fixed-interval and run-to-failure policies, ML-informed replay with train/calibration/test machine separation, outcome/cost scoring, a report/plot, MLflow tracking, CLI integration, tests, and measured documentation.
- Tests executed: Full Pytest suite (49 passed); the corrected synthetic-fleet maintenance CLI completed and logged MLflow run `e415d268d949484c9e763af19eda5827`. Final Ruff lint, format check (44 files), strict Mypy (27 source files), `pip check`, and `git diff --check` passed.
- Results: On the seed-42 15-machine test fleet, the illustrative fixed-interval policy cost 198,000 units vs. 585,000 for run-to-failure, while the ML-informed policy cost 612,000 units. The fixed-interval result assumes 12 failures prevented; the ML policy had 14 false alarms out of 15 actions. These are simulated assumptions, not real-world savings.
- Problems encountered: The first fixed-interval replay repeatedly selected the same machines at global intervals. The policy was corrected to track each machine's own last service while retaining capacity-deferred candidates. MLflow warns that a model signature/input example is absent; the run and artifacts completed. Tests emit upstream Pydantic, Matplotlib, and Pyparsing warnings.
- What is left for next session: Push Phase 5 and verify GitHub CI as desired; Phase 6 requires explicit user approval.

- What was done: Added a CLI to train and calibrate a serving random forest, an MLflow run manifest, FastAPI health/predict/explain/schedule endpoints, Pydantic schemas, Dockerfile/Compose configuration, sample payload, and endpoint integration tests.
- Tests executed: Full Pytest suite: 51 passed; after the MLflow artifact-path fix, both API integration tests passed again. Ruff lint/format, strict Mypy, `pip check`, and `git diff --check` passed. Prepared the seed-42 serving artifact with MLflow run `05ba7e06f54c4a8fba45860a4bf2053d`. `docker compose config`, `docker buildx build --load --tag reliability-platform-api:phase6 .`, and `docker compose up --build --wait` all succeeded.
- Results: The live container reached healthy and served `/health`, `/predict`, `/explain`, and `/schedule` for the documented sample. It returned RUL 120.276 cycles, interval [53.922, 186.631], risk 0, and SHAP additivity residual 6.96e-13. The API reports the configured MLflow run ID and uses the Phase 5 deterministic priority ordering.
- Problems encountered: The first healthy-process attempt was marked unhealthy because Windows-produced MLflow file metadata referenced a host path unavailable in Linux. The runtime now resolves run artifacts from the mounted local file store. The Docker image uses the pinned CPU-only PyTorch wheel, avoiding unnecessary CUDA dependencies.
- What is left for next session: GitHub Actions includes Compose validation and image build; push the changes to run CI. Phase 7 requires explicit user approval.

- What was done: Resumed project verification, confirmed current local and origin commits match, confirmed Phase 6 CI run 43115068066 is green, reran all requested local checks, aligned `PROJECT_SPEC.md` with `PHASE_8_REVISION.md`, and corrected CI status/checklists in this tracker.
- Tests executed: Full Pytest: 51 passed (250 warnings); Ruff lint and format check (50 files); strict Mypy (32 source files); Docker engine check; `docker compose config`; `docker build --tag reliability-platform-api:phase6-verification .`; `git diff --check`.
- Results: All requested local checks passed. GitHub Actions run 43115068066 passed all steps on current `main`. Phases 0-6 are verified against their recorded acceptance criteria; no unmet earlier-phase acceptance criterion was found.
- Problems encountered: The full suite took 697 seconds and emitted 250 warnings, including upstream dependency deprecations. Docker build succeeded.
- What is left for next session: Phase 7 implementation plan is proposed in this session's response. Wait for explicit approval; do not begin Phase 7 or Phase 8 yet.

### 2026-11-14

- What was done: Completed the approved Phase 7 Streamlit dashboard, fixture-based data access and AppTest coverage, API outage fallback, source labels, Phase 8 placeholders, Compose health dependency, README guidance, and genuine screenshots from the running Compose app. Improved API fleet scheduling by engineering all machine features together and compacting dashboard requests to retain the first sensor-variation baseline plus the latest configured feature window.
- Tests executed: Full Pytest (66 passed, 251 warnings); dashboard and API targeted tests; Ruff lint/format, strict Mypy (39 source files), `pip check`, `git diff --check`; Docker Compose config, final API/dashboard image builds, Compose health startup, and live 100-machine schedule request.
- Results: Full suite and all local quality checks passed. The Compose API returned 100 priorities in 1.16 seconds. Four live page screenshots are in `docs/`. No Phase 8 work was started.
- Problems encountered: The first full run exposed an environment-dependent notebook PNG failure. Explicitly selecting the inline plotting backend before pyplot import made the notebook test pass deterministically, including with `MPLBACKEND=TkAgg`. Remaining dependency/platform deprecation warnings are listed above and do not fail tests.
- What is left for next session: User-managed Git operations and GitHub Actions verification for the current working tree. Follow the user's Git rule; do not start Phase 8 without approval.

## Known Issues / Limitations

- The enclosing `C:\Users\KIIT\.git` is unrelated; a separate project-local `.git` establishes the correct repository boundary.
- The system default Python is 3.14.4; use Python 3.11 explicitly for this project.
- NASA raw data and processed Parquet files are intentionally excluded from Git; use the documented download and processing commands to reproduce them.
- The Phase 4 FD002 complete CLI was rerun on 2026-11-12 and wrote `reports/generated/fd002_advanced.json`; measured results and run ID `ce18ca5a63ff43b1803eeded4efc8bc5` are recorded above and in the README.
- MLflow 2.22 logs a warning rather than inferring a model signature for synthetic examples with string `operating_condition` values; the model artifacts and metric runs are still created.
- Phase 5 costs and maintenance-prevention outcomes are synthetic assumptions; the held-out backtest is illustrative and does not establish causal treatment effectiveness or real-world savings.
- Phase 6 Docker/Compose was locally verified on 2026-11-12: `docker compose config`, Docker image build/load, Compose startup with health wait, and all documented API sample requests succeeded. Docker image installs CPU-only PyTorch to avoid unused CUDA runtime dependencies.
- Phase 7 local verification does not imply GitHub CI has passed for the current working tree. The user handles all Git operations; no add, commit, push, stash, checkout, or reset was run.
- `PHASE_8_REVISION.md` is user-provided and intentionally remains untracked. `PROJECT_SPEC.md` reflects its monitoring-only scope and the approved 8.7 data-quality / 8.8 recommendation numbering.

## Phase 8 Progress

### 8.1 Experiment tracking and lineage — complete

- [x] Add metadata-only MLflow lineage tags and a `metadata/lineage.json` artifact to baseline, advanced, serving, and maintenance-backtest runs.
- [x] Record dataset identifier/version, feature schema, code revision status, and MLflow run ID without changing model fitting, metric calculations, serving manifests, or report structures.
- [x] Preserve the additive-only constraint: log interval coverage/width only in workflows that already calculate them. By user decision, these metrics remain unavailable in baseline, serving, and maintenance-backtest runs.
- [x] 8.2 compatibility check: Evidently 0.7.23 resolved with the existing direct pins on Python 3.11.4, installed successfully, passed `pip check`, and passed an actual report API smoke test.
- [x] Add per-feature PSI drift reporting and separate missingness-rate changes with explicit thresholds; provide small/fast CI and full local monitoring configurations.
- [x] 8.3 Summarize actual model/API prediction outputs by window and source, including RUL, risk, interval width, and machine-level changes.
- [x] 8.4 Compare delayed-label RMSE, MAE, and asymmetric score against reference and same-window baselines with machine-level bootstrap confidence intervals.
- [x] 8.5 Monitor row and machine-trajectory interval coverage and width with machine-level confidence intervals and conservative small-sample conclusions.
- [x] 8.6 Demonstrate all four drift/degradation combinations with separate measured drift and delayed-label performance outputs; drift alone is not evidence of degradation.
- [x] 8.7 Report windowed schema violations, out-of-range values, missingness, stuck sensors, and duplicate timestamps.
- [x] 8.8 Generate a deterministic advisory report from drift, delayed-label performance, calibration, and data-quality evidence.
- [ ] 8.9 Integrate saved monitoring outputs into CI and the dashboard.

#### 2026-11-14 — Phase 8.1 work

- What was done: Added reproducible dataset fingerprints and feature-schema hashes, logged lineage tags/artifacts in existing MLflow runs, and corrected the 8.7/8.8 numbering in `PROJECT_SPEC.md`. No model behavior, metric calculations, report output formats, or serving artifacts were changed.
- Tests executed: Focused lineage and training integration tests: 8 passed, 249 warnings, 176 seconds. Warnings are existing upstream Pydantic, Starlette/AnyIO, Matplotlib, and PyParsing deprecations.
- Results: The lineage helper and all four training pathways passed their focused tests. Monitoring outputs, drift calculations, and dashboard integration have not been started.
- Decision: The user chose to preserve the strict additive-only constraint. Interval coverage/width remain absent rather than being fabricated or computed in violation of the requirement.
- What is left: Continue with Phase 8.3 prediction monitoring.

### Phase 8.2 — Data drift monitoring

- Dependency decision: Evidently 0.7.23 was verified with Python 3.11.4 and all existing direct pins. Pip's resolver dry run completed without conflicts; the actual editable install succeeded; `pip check` reported no broken requirements. Its documented `Report` and `DataDriftPreset(method="psi")` API was exercised on actual data and returned the expected per-feature PSI metrics. The release is pinned exactly and the report-shape behavior is covered by tests, so no custom fallback was needed.
- Tests executed: `tests/unit/test_monitoring_drift.py`: 3 passed, 10 warnings in 15.71 seconds. The warnings are upstream Evidently/NumPy and Litestar deprecations.
- Results: Drift reporting returns measured PSI, reference/current missingness rates, per-feature and overall drift flags, row counts, source labels, and library/version metadata. CI and full-local configuration files are present.
- What is left: Prediction monitoring (8.3), delayed-label performance (8.4), calibration (8.5), four deterministic drift/degradation scenarios (8.6), data quality (8.7), advisory recommendation (8.8), and CI/dashboard integration (8.9).

### Phase 8.3 — Prediction monitoring

- What was done: Added windowed summaries for predicted RUL, risk score, and interval width, plus per-machine deltas across windows. Reports preserve source labels and reject invalid or duplicate machine/window predictions.
- Tests executed: `tests/unit/test_monitoring_predictions.py`: 5 passed in 1.61 seconds.
- Results: The monitoring function consumes supplied model/API prediction rows only; it performs no training and does not modify serving artifacts. Unit fixtures verify aggregation, source separation, trend calculations, and validation.
- What is left: Generate predictions from the separate monitoring model and persist reports in the reproducible scenario (8.8/8.9).

### Phase 8.4 — Delayed-label performance monitoring

- What was done: Added per-window RMSE, MAE, and asymmetric-score reporting; comparison against a reference period and named baseline predictions; machine-type/operating-condition summaries; and 95% machine-level bootstrap intervals with fixed configurable seeds and resample counts.
- Statistical guard: Degradation is supported only when the RMSE-difference interval is above zero and both periods meet the configurable minimum machine count. Otherwise the report marks evidence inconclusive and explains why. The bootstrap resamples machines, not correlated cycle rows.
- Tests executed: `tests/unit/test_monitoring_performance.py`: 3 passed in 3.41 seconds.
- Results: Tests verify a measured degradation, baseline comparison, reproducibility, and a small-machine case where the confidence interval is present but the conclusion remains weak/inconclusive.
- What is left: Calibration/coverage monitoring (8.5), deterministic scenarios (8.6), data quality (8.7), recommendation (8.8), and CI/dashboard integration (8.9).

### Phase 8.5 — Calibration monitoring

- What was done: Added delayed-label row and machine-trajectory coverage, average interval width, and per-group summaries using the existing conformal interval metric implementation. Added fixed-seed, configurable machine-cluster bootstrap confidence intervals.
- Statistical guard: Coverage status uses the 95% trajectory-coverage interval and a configurable minimum machine count. Small samples and intervals overlapping target remain inconclusive.
- Tests executed: `tests/unit/test_monitoring_calibration.py`: 3 passed in 3.11 seconds.
- Results: Tests verify supported undercoverage with enough machines, descriptive intervals without a conclusion for a small sample, and the distinction between row and trajectory coverage.
- What is left: Four deterministic measured drift/degradation cases (8.6), data quality (8.7), advisory recommendation (8.8), and CI/dashboard integration (8.9).

### Phase 8.6 — Drift versus degradation scenarios

- What was done: Added a scenario evaluator that independently measures feature drift and delayed-label RMSE degradation, derives the case from those measurements, and rejects scenario sets that fail to demonstrate all four required combinations.
- Statistical guard: Drift alone never implies degradation. Performance degradation is supported only by the existing confidence-interval rule and minimum-machine threshold.
- Scenario-data note: The deterministic outcome perturbation is controlled synthetic data, not evidence about real equipment. Its output is based on computed metrics and must not be presented as a real-world degradation event.
- Tests executed: `tests/unit/test_monitoring_scenarios.py`: 2 passed, 10 upstream Evidently/NumPy/Litestar deprecation warnings, in 23.76 seconds.
- Results: The scenario report covered all four measured cases. The no-drift degradation case had an RMSE increase of 16.0 RUL units with a positive lower confidence bound; the drift-only case remained `no_degradation_detected`.
- What is left: Data-quality monitoring (8.7), advisory recommendation (8.8), and CI/dashboard integration (8.9).

### Phase 8.7 — Data-quality monitoring

- What was done: Added windowed checks for missing/unexpected schema columns, non-numeric sensor values, missingness thresholds, sensor values outside observed reference minima/maxima, repeated per-machine sensor readings, missing machine/timestamp identifiers, and duplicate machine-cycle timestamps. Added matching thresholds to the CI and full monitoring configs.
- Measurement note: Out-of-range checks use each reference sensor's observed min/max and are labeled as such; they are not physical operating limits. Missing cells and invalid non-numeric readings are reported separately.
- Tests executed: `tests/unit/test_monitoring_data_quality.py`: 3 passed in 3.21 seconds.
- Results: Focused fixtures verify each quality check, missingness accounting, and rejection of invalid window indices and thresholds.
- What is left: Advisory recommendation (8.8) and CI/dashboard integration (8.9).

### Phase 8.8 — Advisory retraining recommendation

- What was done: Added a deterministic, source-separated recommendation report that carries through measured drift metrics, RMSE confidence intervals, calibration coverage intervals, and data-quality findings. Supported delayed-label degradation or undercoverage recommends review of retraining; drift, data-quality issues, and inconclusive evidence recommend investigation. Clear evidence recommends no action.
- Safety guard: Recommendations are advisory only; the module neither trains nor switches models. Drift alone cannot result in a retraining recommendation. Each source retains an independent recommendation.
- Tests executed: `tests/unit/test_monitoring_recommendation.py`: 5 passed in 0.37 seconds.
- Results: Tests cover no action, drift-only investigation, supported degradation, inconclusive evidence, undercoverage, and source separation.
- What is left: Reproducible scenario command, saved monitoring outputs, dashboard integration, CI checks, and documentation (8.9).
