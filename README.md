# Equipment Reliability Intelligence & MLOps Platform

An educational, reproducible project for exploring equipment reliability,
remaining useful life (RUL) prediction, uncertainty, and maintenance planning.
The project follows the boundaries and phased plan in [PROJECT_SPEC.md](PROJECT_SPEC.md).

## Status

Phase 0 (project setup) is complete. Phase 1 implements a NASA C-MAPSS loader,
a seeded synthetic fleet generator, data validation, and Parquet storage. No
models, API, or dashboard have been implemented yet.

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

### Validate and test

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src
```

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
