# Progress Tracker

**Last updated:** 2026-11-11
**Current phase:** Phase 3: Feature Engineering and Baseline Models complete; awaiting approval for Phase 4
**Overall status:** 4 of 10 phases complete

## Status Summary

- Done: Phases 0-3: project setup, data engineering, reliability analysis, leakage-aware features, and baseline modeling.
- In progress: None.
- Next up: Request approval to begin Phase 4: Advanced Machine Learning and Uncertainty Quantification.
- Blockers: None. GitHub CI for Phase 3 awaits the user's push.

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

Phases 4-9 are not started; see `PROJECT_SPEC.md`.

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

## Known Issues / Limitations

- The enclosing `C:\Users\KIIT\.git` is unrelated; a separate project-local `.git` establishes the correct repository boundary.
- The system default Python is 3.14.4; use Python 3.11 explicitly for this project.
- NASA raw data and processed Parquet files are intentionally excluded from Git; use the documented download and processing commands to reproduce them.
