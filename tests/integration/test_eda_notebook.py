"""Execute the EDA notebook to verify its reproducibility."""

from pathlib import Path

import nbformat

from reliability_platform.evaluation.notebook import execute_eda_notebook


def test_exploratory_notebook_executes() -> None:
    project_root = Path(__file__).resolve().parents[2]
    output = execute_eda_notebook(
        project_root / ".pytest_tmp" / "eda" / "exploratory_analysis.ipynb"
    )
    executed = nbformat.read(output, as_version=4)

    source = "\n".join(
        cell.source for cell in executed.cells if cell.cell_type == "code"
    )
    assert source.count("Finding ") >= 7
    assert any(
        output.output_type == "display_data" and "image/png" in output.data
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )
