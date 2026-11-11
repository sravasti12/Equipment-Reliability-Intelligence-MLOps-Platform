# Equipment Reliability Intelligence & MLOps Platform

An educational, reproducible project for exploring equipment reliability,
remaining useful life (RUL) prediction, uncertainty, and maintenance planning.
The project follows the boundaries and phased plan in [PROJECT_SPEC.md](PROJECT_SPEC.md).

## Status

Phases 0-2 are complete: project setup, the NASA C-MAPSS and synthetic data
pipelines, and exploratory reliability analysis. No predictive models, API, or
dashboard have been implemented yet.

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
