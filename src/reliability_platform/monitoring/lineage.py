"""Additive MLflow metadata for reproducible model runs."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd

_LOGGER = logging.getLogger(__name__)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _git_output(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(_REPOSITORY_ROOT), *arguments],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _code_revision() -> tuple[str, str]:
    for variable in ("GITHUB_SHA", "CI_COMMIT_SHA"):
        revision = os.environ.get(variable)
        if revision:
            return revision, variable

    revision = _git_output("rev-parse", "HEAD")
    if revision:
        return revision, "git"
    _LOGGER.warning(
        "Could not resolve a code revision; recording it as unavailable in MLflow."
    )
    return "unavailable", "unavailable"


def _dataset_version(frame: pd.DataFrame) -> str:
    schema = [
        {"name": str(column), "dtype": str(dtype)}
        for column, dtype in zip(frame.columns, frame.dtypes, strict=True)
    ]
    digest = hashlib.sha256(
        json.dumps(
            {"columns": schema, "row_count": len(frame)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    try:
        row_hashes = pd.util.hash_pandas_object(frame, index=False)
    except (TypeError, ValueError) as error:
        _LOGGER.warning(
            "Could not fingerprint the input dataset; recording its version as "
            "unavailable in MLflow: %s",
            error,
        )
        return "unavailable"
    digest.update(row_hashes.to_numpy().tobytes())
    return digest.hexdigest()


def _dataset_identifier(
    frame: pd.DataFrame,
    requested_identifier: str | None,
) -> str:
    if requested_identifier and requested_identifier != "unspecified":
        return requested_identifier
    if "source" in frame.columns:
        sources = sorted(str(value) for value in frame["source"].dropna().unique())
        if sources:
            return ",".join(sources)
    return "unspecified"


def _feature_schema(features: pd.DataFrame) -> dict[str, object]:
    columns = [
        {"name": str(column), "dtype": str(dtype)}
        for column, dtype in zip(features.columns, features.dtypes, strict=True)
    ]
    return {"format_version": 1, "columns": columns}


@dataclass(frozen=True)
class RunLineage:
    """Dataset and source revision metadata shared by related MLflow runs."""

    dataset_identifier: str
    dataset_version: str
    code_revision: str
    code_revision_source: str
    working_tree_status: str

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        dataset_identifier: str | None = None,
    ) -> RunLineage:
        revision, revision_source = _code_revision()
        git_status = _git_output("status", "--porcelain", "--untracked-files=no")
        return cls(
            dataset_identifier=_dataset_identifier(frame, dataset_identifier),
            dataset_version=_dataset_version(frame),
            code_revision=revision,
            code_revision_source=revision_source,
            working_tree_status=(
                "unknown" if git_status is None else "dirty" if git_status else "clean"
            ),
        )

    def log(self, features: pd.DataFrame) -> None:
        """Log lineage tags and a schema artifact in the active MLflow run."""
        active_run = mlflow.active_run()
        if active_run is None:
            raise RuntimeError("run lineage must be logged inside an active MLflow run")

        schema = _feature_schema(features)
        schema_json = json.dumps(schema, sort_keys=True, separators=(",", ":"))
        schema_hash = hashlib.sha256(schema_json.encode("utf-8")).hexdigest()
        metadata = {
            "dataset_identifier": self.dataset_identifier,
            "dataset_version_sha256": self.dataset_version,
            "code_revision": self.code_revision,
            "code_revision_source": self.code_revision_source,
            "working_tree_status": self.working_tree_status,
            "feature_schema_sha256": schema_hash,
            "run_id": active_run.info.run_id,
            "feature_schema": schema,
        }
        mlflow.set_tags(
            {
                "lineage.dataset_identifier": self.dataset_identifier,
                "lineage.dataset_version_sha256": self.dataset_version,
                "lineage.code_revision": self.code_revision,
                "lineage.code_revision_source": self.code_revision_source,
                "lineage.working_tree_status": self.working_tree_status,
                "lineage.feature_schema_sha256": schema_hash,
            }
        )
        mlflow.log_dict(metadata, "metadata/lineage.json")
