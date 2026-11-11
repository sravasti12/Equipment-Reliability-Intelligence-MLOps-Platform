# Equipment Reliability Intelligence & MLOps Platform

An educational, reproducible project for exploring equipment reliability,
remaining useful life (RUL) prediction, uncertainty, and maintenance planning.
The project follows the boundaries and phased plan in [PROJECT_SPEC.md](PROJECT_SPEC.md).

## Status

The project is in Phase 0 (project setup). No dataset pipeline, models, API, or
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

The optional `dev` dependencies are pinned in `pyproject.toml`. Phase-specific
runtime dependencies will be added only when needed.

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
