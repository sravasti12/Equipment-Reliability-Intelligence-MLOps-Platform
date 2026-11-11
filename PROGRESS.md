# Progress Tracker

**Last updated:** 2026-11-12
**Current phase:** Phase 6: Model Serving and API complete locally
**Overall status:** 7 of 10 phases complete

## Status Summary

- Done: Phases 0-5: project setup, data engineering, reliability analysis, leakage-aware features, baseline and advanced modeling, uncertainty, explainability, and heuristic maintenance backtesting.
- In progress: None.
- Next up: Wait for explicit approval before Phase 7.
- Blockers: GitHub CI has not run for the local Phase 4-6 changes. The user can push these changes to run CI.

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
- [ ] GitHub CI passes after the user pushes these changes.

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
- [ ] GitHub CI passes after the user pushes these changes.

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

Phase 7 is not started; see `PROJECT_SPEC.md`.

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
| 2026-11-12 | Phase 4 automated checks | Full Pytest: 43 passed in 215.57 s; Ruff lint and format, Mypy (24 source files), `pip check`, and `git diff --check` passed. The focused advanced integration test completed the synthetic CLI and verified MLflow runs, plots, and SHAP additivity. The full FD002 CLI was not rerun after the SHAP fix; Phase 4 CI awaits push. | N/A |
| 2026-11-12 | Phase 5 synthetic maintenance backtest | Seed-42 fleet: 70 training / 15 calibration / 15 test machines; validation RMSE 21.427 cycles and 80% target trajectory-conformal bound 49.664 cycles. Run-to-failure: 585,000 simulated units, 15 failures, 360 downtime hours. Fixed interval: 198,000 units, 22 interventions, 10 false alarms, 12 assumed prevented failures, 3 failures, 72 downtime hours, 24.417-cycle mean lead. ML-informed: 612,000 units, 15 interventions, 14 false alarms, 1 assumed prevented failure, 14 failures, 336 downtime hours, 31-cycle mean lead. Results are assumption-dependent and not causal savings. | `e415d268d949484c9e763af19eda5827` |
| 2026-11-12 | Phase 5 automated checks | Full Pytest: 49 passed. Final Ruff lint, format check (44 files), strict Mypy (27 source files), `pip check`, and `git diff --check` all passed. GitHub CI has not run for local Phase 4/5 changes. | N/A |
| 2026-11-12 | Phase 6 serving model preparation | Seed-42 synthetic fleet; random forest fitted on training machines and calibrated on validation machines at 90% target. Manifest references MLflow run `05ba7e06f54c4a8fba45860a4bf2053d`. | `05ba7e06f54c4a8fba45860a4bf2053d` |
| 2026-11-12 | Phase 6 API integration tests | Full Pytest: 51 passed (2 API integration tests). Tests loaded an MLflow artifact and verified `/health`, `/predict`, `/explain` (SHAP additivity), `/schedule` capacity/priority ordering, duplicate-cycle rejection, and unavailable-model HTTP 503. After fixing local Windows MLflow file-URI handling for a Linux container, the 2 API integration tests passed again. Final Ruff lint/format, strict Mypy, `pip check`, `git diff --check`, and sample JSON validation passed. Docker/Compose verified locally: `docker compose config` and `docker buildx build --load --tag reliability-platform-api:phase6 .` succeeded; `docker compose up --build --wait` reached healthy. The documented sample returned RUL 120.276 cycles, interval [53.922, 186.631], risk 0, and SHAP additivity residual 6.96e-13. | N/A |

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

## Known Issues / Limitations

- The enclosing `C:\Users\KIIT\.git` is unrelated; a separate project-local `.git` establishes the correct repository boundary.
- The system default Python is 3.14.4; use Python 3.11 explicitly for this project.
- NASA raw data and processed Parquet files are intentionally excluded from Git; use the documented download and processing commands to reproduce them.
- The full FD002 Phase 4 CLI has not yet been rerun after correcting SHAP; its initial model metrics are present in MLflow, and corrected SHAP was verified directly on the saved model. The full advanced CLI passed on the seeded synthetic fleet.
- MLflow 2.22 logs a warning rather than inferring a model signature for synthetic examples with string `operating_condition` values; the model artifacts and metric runs are still created.
- Phase 5 costs and maintenance-prevention outcomes are synthetic assumptions; the held-out backtest is illustrative and does not establish causal treatment effectiveness or real-world savings.
- Phase 6 Docker/Compose was locally verified on 2026-11-12: `docker compose config`, Docker image build/load, Compose startup with health wait, and all documented API sample requests succeeded. Docker image installs CPU-only PyTorch to avoid unused CUDA runtime dependencies.
