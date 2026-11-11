"""Reproducible execution of the exploratory reliability notebook."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def execute_eda_notebook(output: Path) -> Path:
    """Execute the EDA notebook and write an executed copy to ``output``."""
    project_root = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "pyproject.toml").is_file()
    )
    notebook_path = project_root / "notebooks" / "exploratory_analysis.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)  # type: ignore[no-untyped-call]
    executed_notebook = NotebookClient(
        notebook,
        kernel_name="python3",
        timeout=120,
        resources={"metadata": {"path": str(project_root)}},
    ).execute()
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(executed_notebook, output)  # type: ignore[no-untyped-call]
    return output


def main(argv: list[str] | None = None) -> None:
    """Execute the EDA notebook from the project command line."""
    parser = argparse.ArgumentParser(
        prog="reliability-platform run-eda",
        description="Execute the reproducible exploratory reliability notebook.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/exploratory_analysis.executed.ipynb"),
    )
    arguments = parser.parse_args(argv)
    output = execute_eda_notebook(arguments.output)
    print(f"Wrote executed EDA notebook to {output}")
