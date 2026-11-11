# Equipment Reliability Intelligence & MLOps Platform

An educational, reproducible project for exploring equipment reliability,
remaining useful life (RUL) prediction, uncertainty, and maintenance planning.
The project follows the boundaries and phased plan in [PROJECT_SPEC.md](PROJECT_SPEC.md).

## Status

Phases 0-2 are complete: project setup, the NASA C-MAPSS and synthetic data
pipelines, and exploratory reliability analysis. Phase 3 adds leakage-aware
features and baseline RUL models. No API or dashboard has been implemented.

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
