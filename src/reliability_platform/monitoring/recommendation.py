"""Deterministic, advisory retraining recommendations from monitoring reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

_ACTION_PRIORITY = {"no_action": 0, "investigate": 1, "retrain": 2}


def _records(report: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    value = report.get(key)
    if not isinstance(value, list):
        raise ValueError(f"monitoring report must contain a list of {key!r} objects")
    if not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"monitoring report must contain a list of {key!r} objects")
    return [item for item in value if isinstance(item, Mapping)]


def _source_labels(report: Mapping[str, object]) -> list[str]:
    sources = report.get("current_sources")
    if not isinstance(sources, list) or not all(
        isinstance(source, str) for source in sources
    ):
        return []
    if len(sources) > 1:
        raise ValueError("each drift report must contain only one current data source")
    return [source for source in sources if isinstance(source, str)]


def _mapping_value(
    mapping: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = mapping.get(key)
    return value if isinstance(value, Mapping) else {}


def _quality_window_has_source(
    window: Mapping[str, object],
    source: str,
) -> bool:
    sources = window.get("sources")
    return isinstance(sources, list) and source in sources


def _source_recommendation(
    source: str,
    drift_reports: list[Mapping[str, object]],
    performance_windows: list[Mapping[str, object]],
    calibration_windows: list[Mapping[str, object]],
    quality_windows: list[Mapping[str, object]],
) -> dict[str, object]:
    drift_evidence = [
        {
            "drift_detected": report["overall_drift_detected"],
            "drifted_columns": report.get("drifted_columns", []),
            "features": report.get("features", []),
        }
        for report in drift_reports
    ]
    performance_evidence = [
        {
            "window_index": window.get("window_index"),
            "machines": window.get("machines"),
            "metrics": window.get("metrics"),
            "rmse_difference": _mapping_value(
                window,
                "metric_deltas",
            ).get("rmse"),
            "status": window.get("degradation_status"),
            "evidence_note": window.get("evidence_note"),
        }
        for window in performance_windows
    ]
    calibration_evidence = [
        {
            "window_index": window.get("window_index"),
            "machines": window.get("machines"),
            "metrics": window.get("metrics"),
            "trajectory_coverage_interval": (
                _mapping_value(window, "confidence_intervals_95").get(
                    "trajectory_coverage"
                )
            ),
            "status": window.get("coverage_status"),
            "evidence_note": window.get("evidence_note"),
        }
        for window in calibration_windows
    ]
    quality_evidence = [
        {
            "window_index": window.get("window_index"),
            "quality_status": window.get("quality_status"),
            "schema_violations": window.get("schema_violations"),
            "missingness": window.get("missingness"),
            "out_of_range_values": window.get("out_of_range_values"),
            "stuck_sensors": window.get("stuck_sensors"),
            "duplicate_timestamp_rows": window.get("duplicate_timestamp_rows"),
        }
        for window in quality_windows
    ]

    drift_detected = any(
        report.get("overall_drift_detected") is True for report in drift_reports
    )
    degradation_supported = any(
        window.get("degradation_status") == "degradation_supported"
        for window in performance_windows
    )
    undercoverage_supported = any(
        window.get("coverage_status") == "undercoverage_supported"
        for window in calibration_windows
    )
    quality_issues = any(
        window.get("quality_status") == "issues_detected" for window in quality_windows
    )
    known_performance_statuses = {
        "degradation_supported",
        "no_degradation_detected",
        "inconclusive_insufficient_machines",
        "inconclusive_confidence_interval",
    }
    known_coverage_statuses = {
        "undercoverage_supported",
        "coverage_at_or_above_target",
        "inconclusive_insufficient_machines",
        "inconclusive_confidence_interval",
    }
    known_quality_statuses = {"issues_detected", "no_thresholds_exceeded"}
    incomplete_or_inconclusive = (
        not drift_reports
        or not performance_windows
        or not calibration_windows
        or not quality_windows
        or any(
            window.get("degradation_status") not in known_performance_statuses
            for window in performance_windows
        )
        or any(
            window.get("degradation_status")
            in {
                "inconclusive_insufficient_machines",
                "inconclusive_confidence_interval",
            }
            for window in performance_windows
        )
        or any(
            window.get("coverage_status") not in known_coverage_statuses
            for window in calibration_windows
        )
        or any(
            window.get("coverage_status")
            in {
                "inconclusive_insufficient_machines",
                "inconclusive_confidence_interval",
            }
            for window in calibration_windows
        )
        or any(
            window.get("quality_status") not in known_quality_statuses
            for window in quality_windows
        )
        or any(
            not isinstance(report.get("overall_drift_detected"), bool)
            for report in drift_reports
        )
    )

    if degradation_supported or undercoverage_supported:
        action = "retrain"
        rationale = (
            "Measured delayed-label degradation or supported interval undercoverage "
            "is present; review the evidence and consider rerunning the normal "
            "training pipeline."
        )
    elif drift_detected or quality_issues or incomplete_or_inconclusive:
        action = "investigate"
        rationale = (
            "A data change, data-quality issue, or insufficient/inconclusive evidence "
            "warrants investigation. Drift alone is not proof of poor model "
            "performance."
        )
    else:
        action = "no_action"
        rationale = (
            "No monitored condition currently supports further action; continue "
            "monitoring."
        )

    return {
        "source": source,
        "recommendation": action,
        "summary": f"{source}: {action.replace('_', ' ')} — {rationale}",
        "conditions": {
            "data_drift": drift_detected,
            "delayed_label_degradation": degradation_supported,
            "calibration_undercoverage": undercoverage_supported,
            "data_quality_issues": quality_issues,
            "evidence_incomplete_or_inconclusive": incomplete_or_inconclusive,
        },
        "evidence": {
            "drift": drift_evidence,
            "performance": performance_evidence,
            "calibration": calibration_evidence,
            "data_quality": quality_evidence,
        },
    }


def build_retraining_recommendation(
    drift_reports: Sequence[Mapping[str, object]],
    performance_report: Mapping[str, object],
    calibration_report: Mapping[str, object],
    data_quality_report: Mapping[str, object],
) -> dict[str, object]:
    """Create per-source recommendations without modifying models or reports."""
    drift_sources: dict[str, list[Mapping[str, object]]] = {}
    for report in drift_reports:
        labels = _source_labels(report)
        if not labels:
            raise ValueError("each drift report must identify its current data source")
        for source in labels:
            drift_sources.setdefault(source, []).append(report)

    performance_windows = _records(performance_report, "windows")
    calibration_windows = _records(calibration_report, "windows")
    quality_windows = _records(data_quality_report, "windows")
    sources = set(drift_sources)
    sources.update(
        str(window["source"])
        for window in performance_windows
        if isinstance(window.get("source"), str)
    )
    sources.update(
        str(window["source"])
        for window in calibration_windows
        if isinstance(window.get("source"), str)
    )
    for window in quality_windows:
        window_sources = window.get("sources")
        if isinstance(window_sources, list):
            sources.update(
                source for source in window_sources if isinstance(source, str)
            )
    if not sources:
        raise ValueError("monitoring reports must identify at least one source")

    recommendations = [
        _source_recommendation(
            source,
            drift_sources.get(source, []),
            [
                window
                for window in performance_windows
                if window.get("source") == source
            ],
            [
                window
                for window in calibration_windows
                if window.get("source") == source
            ],
            [
                window
                for window in quality_windows
                if _quality_window_has_source(window, source)
            ],
        )
        for source in sorted(sources)
    ]
    overall_action = max(
        (str(recommendation["recommendation"]) for recommendation in recommendations),
        key=lambda action: _ACTION_PRIORITY[action],
    )
    return {
        "schema_version": 1,
        "recommendation": overall_action,
        "summary": " ".join(
            str(recommendation["summary"]) for recommendation in recommendations
        ),
        "advisory_only": True,
        "source_recommendations": recommendations,
    }
