# Equipment Reliability Intelligence & MLOps Platform

An educational, reproducible project for exploring equipment reliability,
remaining useful life (RUL) prediction, uncertainty, and maintenance planning.
The project follows the boundaries and phased plan in [PROJECT_SPEC.md](PROJECT_SPEC.md).

## Status

Phases 0-6 are implemented and locally verified: project setup, data
engineering, reliability analysis, leakage-aware feature engineering,
baseline and advanced RUL models, survival analysis, anomaly detection,
uncertainty quantification, explainability, and heuristic maintenance
backtesting. Phase 6 API endpoints and Docker container are locally verified.
Phase 7 dashboard pages and updated API are locally verified through Docker
Compose. GitHub Actions run 43115068066 passed for the earlier commit
`6560fa7`; CI has not yet run for the current working tree.

## Development setup

Use Python 3.11. On Windows, the Python launcher can select it explicitly:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the local checks:

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src
```

Runtime and development dependencies are pinned in `pyproject.toml`. Phase-specific
dependencies are added only when needed.

## Phase 1: Data engineering

### Generate a synthetic fleet

From the project root, install the package and run the documented default
configuration. It generates at least 100 simulated machines, validates the
result, and writes Parquet under `data/processed/`:

```powershell
python -m pip install -e ".[dev]"
reliability-platform generate-synthetic --config configs\data.yaml
```

The seed, fleet size, lifetime range, sensor count/noise, missingness, and
maintenance-event settings are in [configs/data.yaml](configs/data.yaml).
Synthetic observations are illustrative simulation only, not industrial
telemetry. Missing sensor values remain missing and are reported by validation;
the pipeline does not silently impute them.

### Process NASA C-MAPSS

Download the public C-MAPSS archive from NASA's
[Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
(Turbofan Engine Degradation Simulation). Extract it so the selected
`train_FD00*.txt`, `test_FD00*.txt`, and `RUL_FD00*.txt` files are directly
inside `data/raw/nasa_cmapss/`. On Windows:

```powershell
New-Item -ItemType Directory -Force data\raw\nasa_cmapss
Invoke-WebRequest -Uri "https://phm-datasets.s3.amazonaws.com/NASA/6.%2BTurbofan%2BEngine%2BDegradation%2BSimulation%2BData%2BSet.zip" -OutFile data\raw\CMAPSSData.zip
Expand-Archive -Path data\raw\CMAPSSData.zip -DestinationPath data\raw\nasa_archive -Force
Expand-Archive -Path "data\raw\nasa_archive\6. Turbofan Engine Degradation Simulation Data Set\CMAPSSData.zip" -DestinationPath data\raw\nasa_cmapss -Force
reliability-platform process-cmapss --input-dir data\raw\nasa_cmapss --subset FD001 --split train
reliability-platform process-cmapss --input-dir data\raw\nasa_cmapss --subset FD001 --split test
```

The loader expects the standard whitespace-delimited 26-column C-MAPSS files:
engine ID, cycle, three operating settings, and 21 sensors. Training RUL is
derived from each engine's final observed cycle. Test RUL labels are loaded
from NASA's label file and are targets for evaluation, not input features.
Test trajectories are not marked as observed failures.

## Phase 2: Exploratory reliability analysis

The reusable metrics in `src/reliability_platform/evaluation/reliability.py`
report observed lifetimes and censoring, failures per 1,000 observed machine
cycles, and maintenance events per 1,000 observed machine cycles. A C-MAPSS
training trajectory contains one terminal failure per engine, which is not
enough to estimate recurrent-failure MTBF. Neither C-MAPSS nor the synthetic
generator records measured downtime hours, so MTBF and downtime are explicitly
reported as unavailable rather than inferred. Test trajectories contribute
observed exposure but are not counted as failures.

Generate a machine-readable reliability report:

```powershell
reliability-platform analyze-reliability --input data\processed\cmapss_FD001_train.parquet --output reports\generated\fd001_reliability.json
```

Execute the EDA notebook and save a copy with computed charts/findings:

```powershell
reliability-platform run-eda --output reports\generated\exploratory_analysis.executed.ipynb
```

The source notebook is [notebooks/exploratory_analysis.ipynb](notebooks/exploratory_analysis.ipynb).
It analyzes actual FD001/FD002 processed data when available, otherwise clearly
falls back to the reproducible synthetic fleet. The notebook reports these
measured findings from the downloaded NASA FD001/FD002 training data and
seed-42 simulation:

1. FD001's 100 observed terminal failure lifetimes have median 199 cycles and
   interquartile range 177–229.2 cycles.
2. The mean fitted RUL-vs-cycle slope for five displayed completed FD001
   trajectories is -1.000 cycles per cycle, a target-label consistency check.
3. Among the selected plotted FD001 sensors, sensor 3 has the largest absolute
   first-to-last life-decile median shift: +13.060 sensor units.
4. The strongest absolute off-diagonal Pearson correlation in FD001 is between
   sensors 9 and 14 (r = 0.963).
5. FD002 has the larger setting-1 standard deviation (14.7474) than FD001
   (0.0022); this describes the simulated operating conditions, not a causal
   effect.
6. In the seeded synthetic fleet, type B has the highest observed maintenance
   event rate (8.802 per 1,000 machine cycles).
7. Sensor missingness is 0.000% in NASA FD001 training and 2.018% in the
   synthetic fleet; missing readings are not imputed.

Each finding is generated in the executed notebook beside its supporting
visualization. These descriptive results are specific to these datasets and do
not demonstrate predictive performance or represent actual industrial telemetry.

### Validate and test

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src
```

`pytest` also executes the EDA notebook headlessly as an integration test.

## Phase 3: Feature engineering and baseline RUL models

Train the constant-mean, linear-regression, and random-forest baselines on a
processed training dataset:

```powershell
reliability-platform train-baselines --input data\processed\cmapss_FD002_train.parquet --output reports\generated\fd002_baselines.json --experiment-name cmapss-fd002-baselines --tracking-uri file:./mlruns --seed 42 --window-size 5
```

The command assigns whole machines to deterministic 70%/15%/15% train,
validation, and test partitions. Machine IDs do not cross partitions. At each
cycle, features use only the current and preceding observations: raw sensors
and operating settings, trailing means and standard deviations, trailing
per-cycle linear trends, sensor deviations from the first recorded cycle,
an aggregate relative sensor-change health indicator, and the current cycle
number. Machine IDs, RUL labels, maintenance events, and failure events are not
predictors. Missing sensor values are imputed from training data only; linear
regression numeric features are also scaled on the training partition only.

The JSON report includes RMSE, MAE, and the asymmetric NASA C-MAPSS score for
train, validation, and test, plus test results by machine type and operating
context where available. For numeric settings, operating-context bins are
derived from training-partition quantiles. Each model and its parameters,
metrics, and fitted pipeline are logged to the named local MLflow experiment
(`mlruns/` by default). Use a different `--tracking-uri` to select another
MLflow tracking store.

With seed 42 and a five-observation window, the held-out results were:
The NASA result is the 39-machine holdout drawn from the FD002 training file,
not the separate official NASA test split.

| Dataset (test machines / rows) | Model | RMSE (cycles) | MAE (cycles) | Asymmetric score |
|---|---|---:|---:|---:|
| NASA FD002 (39 / 8,070) | Constant mean | 67.594 | 56.315 | 918,376,631 |
| NASA FD002 (39 / 8,070) | Linear regression | 41.230 | 31.664 | 6,473,930 |
| NASA FD002 (39 / 8,070) | Random forest | 36.086 | 26.280 | 3,358,202 |
| Synthetic fleet (15 / 1,772) | Constant mean | 36.909 | 31.257 | 118,118 |
| Synthetic fleet (15 / 1,772) | Linear regression | 20.026 | 16.721 | 15,411 |
| Synthetic fleet (15 / 1,772) | Random forest | 18.903 | 14.444 | 14,827 |

These are measured results for the stated seeded machine-level splits, not
cross-validation or evidence of production performance. The asymmetric score
is a sum and can be large because of its exponential penalty. On synthetic
test data, the random-forest RMSE ranged from 13.806 cycles for type B to
23.992 for type A (type C: 14.865). On NASA FD002, random-forest RMSE ranged
from 35.105 to 36.495 across training-quantile bins for settings 1 and 2.
The synthetic group results are simulation checks; NASA setting groups are
descriptive condition comparisons.

## Phase 4: Advanced models and uncertainty

Run the single-architecture advanced comparison on a processed training file:

```powershell
reliability-platform train-advanced --input data\processed\cmapss_FD002_train.parquet --output reports\generated\fd002_advanced.json --experiment-name cmapss-fd002-advanced --tracking-uri file:./mlruns --seed 42 --window-size 5 --sequence-length 20 --epochs 10 --target-coverage 0.9 --landmark-cycles 20
```

The workflow uses an unidirectional LSTM over causal cycle-level features,
Isolation Forest anomaly scores, a landmark Weibull AFT survival model,
machine-level split-conformal RUL intervals, and SHAP explanations for the
random forest. It reserves machines for fit, independent conformal calibration,
validation-based selection, and a final test set. Only validation RMSE selects
the RUL model; test metrics are calculated after selection and are not used to
tune it. Model hyperparameters are fixed by the command options and documented
defaults rather than tuned on the test data.

The conformal score is the largest absolute cycle residual for each calibration
machine, so the exchangeability unit is a machine trajectory, not its correlated
cycle rows. The report includes target and observed row/trajectory coverage,
average interval width, machine-type coverage, and coverage by operating
condition or training-defined setting bins. Very wide intervals are reported
as low utility rather than hidden by the coverage target.

The survival model uses only the first `--landmark-cycles` observations as
covariates, with last observed cycle as duration and explicit `failure_event`
as the event indicator; missing event labels are treated as right censoring.
It reports per-machine and horizon-level failure-free probabilities, Brier
scores, and concordance. Survival probabilities describe event-free probability
by time and are not RUL point predictions. C-MAPSS train and the seeded
synthetic fleet are predominantly run-to-failure, limiting empirical
assessment of censoring assumptions.

Isolation Forest is evaluated with a fixed, reproducible validation and test
scenario that shifts one held-out sensor value by six training standard
deviations per machine. This is a controlled anomaly-detection check, not a
measured fault label; anomaly scores are not failure probabilities. SHAP uses
TreeExplainer's tree-path-dependent perturbation; the report records the
maximum additivity residual and the run fails if explanations do not sum to
the underlying random-forest predictions. Global and individual plots are
written beside the JSON report as `*_shap_global.png` and
`*_shap_individual.png`. SHAP values explain model attribution, not causality
or a physical failure mechanism.

All model, survival, anomaly, conformal, and explanation runs are logged to the
selected MLflow experiment. The comparison report includes the Phase 3 baselines,
the LSTM, random-forest raw-feature ablation, Weibull AFT and Kaplan-Meier
survival baseline, and Isolation Forest scenario metrics. Advanced models are
not expected to outperform the simpler baselines.

### Phase 4 measured results

The seeded synthetic run completed the full CLI workflow with 15 held-out
machines (1,772 cycle rows). The NASA FD002 figures below are from completed
MLflow evaluation runs on a 39-machine, 8,070-row holdout from the FD002
training file; they are not results on NASA's separate official test split.

| Dataset | Model | Test RMSE (cycles) | Test MAE (cycles) | NASA asymmetric score |
|---|---|---:|---:|---:|
| NASA FD002 | Constant mean | 67.597 | 56.244 | 1,040,422,794 |
| NASA FD002 | Linear regression | 40.124 | 31.161 | 4,819,382 |
| NASA FD002 | Random forest | 35.494 | 25.923 | 2,770,600 |
| NASA FD002 | LSTM | 44.246 | 32.093 | 75,086,389 |
| NASA FD002 | Random forest raw-feature ablation | 36.772 | 26.613 | 5,066,116 |
| Synthetic (seed 42) | Constant mean | 36.982 | 31.330 | 123,660 |
| Synthetic (seed 42) | Linear regression | 19.173 | 15.595 | 16,042 |
| Synthetic (seed 42) | Random forest | 16.726 | 13.153 | 9,263 |
| Synthetic (seed 42) | LSTM | 22.139 | 17.541 | 26,324 |

Validation RMSE selected the random forest on both datasets; the LSTM did not
outperform the simpler model. For the synthetic run, the 90% target conformal
intervals covered 100% of test rows and complete trajectories, but averaged
98.684 RUL cycles wide (14 calibration machines), so the observed coverage
does not imply narrow or operationally useful intervals. The selected
featureless Kaplan-Meier survival baseline had a test integrated Brier score
of 0.214 versus 0.271 for Weibull AFT. The injected-shift anomaly check had
ROC AUC 0.600 and average precision 0.609; these are synthetic scenario
metrics, not real fault-detection performance. SHAP's maximum additivity
residual was below 1e-12.

On 2026-11-12, the full NASA FD002 advanced CLI was rerun end to end after the
SHAP fix and wrote `reports/generated/fd002_advanced.json`. It reproduced the
table above: validation selected random forest at 35.057-cycle RMSE; the
39-machine/8,070-row test partition had 100% row and trajectory coverage
against a 90% target, with 278.061-cycle average interval width. The run logged
the selected random-forest model as MLflow run
`ce18ca5a63ff43b1803eeded4efc8bc5` in experiment
`cmapss-fd002-advanced-resume`.

For FD002, validation selected the featureless Kaplan-Meier baseline
(validation Brier 0.207 versus 0.213 for Weibull AFT), while the test Brier
scores were 0.207 for Kaplan-Meier and 0.197 for Weibull AFT. The 90% target
conformal intervals covered all test rows and trajectories and averaged
278.061 cycles wide (36 calibration machines). The injected-shift anomaly
scenario had ROC AUC 0.570 and average precision 0.556. The lower Weibull test
score does not change validation-based model selection.

The FD002 CLI invocation reached and logged all model and test metrics but
stopped before writing the comparison JSON: interventional SHAP
explanations failed their additivity check by up to 0.750 RUL cycles. SHAP was
changed to tree-path-dependent perturbation and then verified on the same
saved FD002 random forest and holdout: 32 rows were explained, both plots were
created, and the maximum additivity residual was 8.32e-10. The full FD002 CLI
was not rerun after this correction; the corrected complete CLI is covered by
the synthetic integration test.

## Phase 5: Heuristic maintenance scheduling and backtesting

Run the chronological policy comparison on a processed, labelled fleet file:

```powershell
reliability-platform backtest-maintenance --input data\processed\synthetic_fleet.parquet --config configs\maintenance.yaml --output reports\generated\synthetic_maintenance_backtest.json --experiment-name phase5-synthetic-backtest --tracking-uri file:./mlruns
```

The command writes a JSON comparison and a companion PNG chart, and logs
configuration, strategy metrics, the fitted model, and the report to MLflow.
The default YAML makes all cost and policy assumptions explicit. Its costs
are illustrative synthetic units, not authoritative industrial estimates:
2,500 per planned maintenance, 500 per maintenance intervention, 15,000 per
unplanned failure, 24 downtime hours per failure at 1,000 per hour, and 1,500
per confirmed false alarm. Edit [configs/maintenance.yaml](configs/maintenance.yaml)
to change costs, crew capacity, prediction interval coverage, risk threshold,
fixed interval, or the assumed maintenance-effectiveness window.

Machines are separated into 70% training, 15% conformal calibration, and 15%
held-out test partitions. A Phase 3 random-forest RUL pipeline is fit only on
training machines. Its validation-machine maximum-trajectory residual
calibrates a conservative RUL lower bound. During chronological replay,
features for a decision use only the current and preceding cycles; test RUL
and failure labels are kept out of the scheduler and used only to score its
actions afterward. Crew capacity is enforced at each simulated cycle/shift.
The ML-informed policy greedily ranks eligible machines by risk, with ties
broken deterministically; the fixed-interval policy services each machine at
its configured interval and keeps capacity-deferred machines eligible. The ML
policy limits actions to one per machine because this
backtest has no post-maintenance counterfactual features.

The comparison reports simulated total cost, failures, downtime, false-alarm
rate, interventions, and lead time. A preventive intervention is assumed to
prevent a recorded failure only when scheduled strictly before it and within
the configured effectiveness window. This is a transparent counterfactual
assumption, not a measured maintenance treatment effect. Right-censored
machines are not counted as failures; interventions without adequate
follow-up are reported as unverified. Costs depend on the configurable
assumptions and the predictions. The greedy policy has no machine dependency
handling or global optimality guarantee, and crew capacity is fixed per shift.

### Phase 5 measured results

The seed-42 synthetic fleet contained 100 machines; the held-out backtest
scored 15 test machines (70 training / 15 calibration / 15 test). Validation
RUL RMSE was 21.427 cycles, and the 80% target trajectory-conformal error
bound was 49.664 cycles. Under the default synthetic costs and 40-cycle
maintenance-effectiveness assumption:

| Strategy | Simulated cost units | Planned interventions | False-alarm fraction | Assumed prevented failures | Unplanned failures | Downtime hours | Mean lead cycles |
|---|---:|---:|---:|---:|---:|---:|---:|
| Run to failure | 585,000 | 0 | 0.0% | 0 | 15 | 360 | N/A |
| Fixed interval | 198,000 | 22 | 45.5% | 12 | 3 | 72 | 24.4 |
| ML-informed heuristic | 612,000 | 15 | 93.3% | 1 | 14 | 336 | 31.0 |

In this replay, fixed-interval service has the lowest simulated cost and
downtime, but requires 22 interventions. The ML-informed policy does **not**
reduce cost against run-to-failure under these settings: its simulated cost is
612,000 units, with 14 false alarms among 15 interventions. Its conservative
trajectory error bound makes the lower-RUL risk signal highly cautious. The
apparent fixed-interval benefit and these ML-policy results are not measured
causal maintenance effects or evidence of real-world savings: they depend on
the configured synthetic costs and the assumption that an intervention
prevents a recorded failure only when scheduled within 40 cycles of it.

## Phase 6: Model serving API

First generate the reproducible synthetic fleet and prepare a serving model.
The preparation command fits a random-forest pipeline on training machines,
calibrates a machine-trajectory conformal interval on validation machines,
logs the artifact and metrics to MLflow, and writes the run ID and feature
schema to a manifest. Test machines are not used for fitting or calibration.

```powershell
reliability-platform generate-synthetic --output data\processed\synthetic_fleet.parquet
reliability-platform prepare-serving-model --input data\processed\synthetic_fleet.parquet --output reports\generated\serving_model.json --experiment-name phase6-serving --tracking-uri file:./mlruns --seed 42 --window-size 5 --target-coverage 0.9
docker compose up --build --wait
```

The Compose service mounts the local MLflow run store and serving manifest
read-only. Its health check waits for the configured model run to load. From
PowerShell, use the bundled [example request](examples/phase6_predict.json):

```powershell
curl.exe -fsS http://localhost:8000/health
curl.exe -fsS -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data-binary "@examples\phase6_predict.json"
curl.exe -fsS -X POST http://localhost:8000/explain -H "Content-Type: application/json" --data-binary "@examples\phase6_predict.json"
$machine = Get-Content examples\phase6_predict.json -Raw | ConvertFrom-Json
$schedule = @{ machines = @($machine) } | ConvertTo-Json -Depth 20
Invoke-RestMethod -Method Post -Uri http://localhost:8000/schedule -ContentType "application/json" -Body $schedule
docker compose down
```

`/predict` returns the machine ID, predicted RUL, calibrated interval, risk
score, and MLflow run ID as the model version. `/explain` returns top
tree-path-dependent SHAP contributions and checks their additivity.
`/schedule` applies the Phase 5 deterministic risk ranking and capacity limit.
`/health` reports service and model availability; model-dependent requests
return HTTP 503 when the manifest or artifact is unavailable, while malformed
or incomplete telemetry returns HTTP 422. Requests must provide unique cycle
numbers, sensor names, and any settings/operating-condition features required
by the manifest's training schema. The operating-condition value must match
the value encoding used during training.

The API serves one explicitly configured MLflow run; its run ID is not a
registry stage and there is no model switching or promotion behavior.
Intervals depend on the calibration population and its exchangeability
assumption. SHAP is limited to the supported tree pipeline and explains model
output, not causal effects. The maintenance schedule is a heuristic, not a
global optimizer.

## Phase 7: Reliability dashboard

The local Streamlit dashboard reads saved Parquet and JSON outputs from
`data/processed/` and `reports/generated/`; it does not train models or schedule
jobs. The four views are fleet overview, individual machine analysis, business
analytics for the saved synthetic maintenance backtest, and model analytics for
saved Phase 4 model comparisons. Each view labels its data source. Synthetic
fleet and NASA C-MAPSS results are kept separate.

Run the dashboard locally with saved artifacts:

```powershell
streamlit run src\reliability_platform\dashboard\app.py
```

The fleet overview and machine analysis can display their saved artifact-backed
content when the API is unavailable and will show an explicit status message.
The current serving model is trained on the synthetic fleet; NASA C-MAPSS rows
are never sent to it. Model analytics displays saved MLflow run IDs, test
metrics, and one-shot interval coverage/width. Drift, calibration-over-time,
and data-quality panels are Phase 8 placeholders and contain no monitoring
values.

To run the dashboard with the API through Compose, first generate the synthetic
fleet and prepare the serving model as shown in Phase 6, then start both
services:

```powershell
docker compose up --build --wait
```

Open `http://localhost:8501` for the dashboard and `http://localhost:8000` for
the API. Compose mounts processed data and reports read-only and starts the
dashboard only after the API health check succeeds. Shut down with
`docker compose down`. The dashboard can still render saved reports when its
API is unavailable in local development.

Genuine screenshots captured from the running Compose dashboard are available
in [`docs/`](docs/): [fleet overview](docs/fleet-overview.png),
[machine analysis](docs/machine-analysis.png),
[business analytics](docs/business-analytics.png), and
[model analytics](docs/model-analytics.png).

## Project principles

- Use public datasets and clearly labelled synthetic data only.
- Keep experiments reproducible and prevent future-data leakage.
- Report measured results honestly; do not infer results from generated code.
- Keep maintenance prioritization deterministic and explainable.
- Do not add LLMs, agentic workflows, or formal constraint solvers.

## Planned work

See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the architecture, phase plan,
acceptance criteria, and evaluation requirements. Progress and verified results
are recorded in [PROGRESS.md](PROGRESS.md).
