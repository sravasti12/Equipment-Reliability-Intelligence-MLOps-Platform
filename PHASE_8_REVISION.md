# Revision: Replace Phase 8 and align the rest of the spec

Purpose: make this project clearly different from the OTA Campaign Verifier. The OTA project owns **deployment safety** (versioned slots, activation, rollback, guarded promotion). This project owns **monitoring and statistical reliability of predictions**. So Phase 8 becomes a monitoring and data-quality phase, with no registry stages, promotion gates, rollback, or champion-challenger logic.

---

## A. Replace the whole of Phase 8 with this

## Phase 8: Monitoring, Data Quality and Model Reliability Analytics

The goal is to show how a deployed prediction system is observed and judged over time: is the incoming data changing, are predictions still accurate, and are the uncertainty intervals still honest?

This phase observes and reports. It does not deploy, promote, switch, or roll back models.

### 8.1 Experiment tracking and lineage
Use MLflow for tracking only. Record for every training run:
- Parameters and training configuration.
- Dataset identifier and version.
- Feature schema.
- Evaluation metrics, including interval coverage and width.
- Code revision and run ID.

The API loads the model from a single configured run ID or artifact path. There is no staging, production tagging, or promotion workflow.

### 8.2 Data drift monitoring
Integrate Evidently to monitor:
- Sensor distributions.
- Operating conditions.
- Feature distributions.
- Missing-value rates.

Generate reproducible drift reports comparing a reference dataset with controlled synthetic distribution shifts (gradual sensor offset, changed operating-condition mix, increased missingness). This is a data-monitoring experiment only.

### 8.3 Prediction monitoring
Track over simulated time:
- Predicted RUL distribution.
- Risk-score distribution.
- Prediction interval widths.
- Machine-level prediction trends.

### 8.4 Performance monitoring with delayed labels
When true failure times become known after the fact:
- Compute realized RUL errors (RMSE, MAE, asymmetric score).
- Compare against the baseline models over the same period.
- Track error by machine group and operating condition.
- Flag statistically meaningful degradation (for example, a rolling window versus the validation period).

### 8.5 Calibration monitoring
Track conformal interval coverage and average width over time using delayed labels. Report when observed coverage falls meaningfully below the target, and discuss why (distribution shift, exchangeability assumption violated).

### 8.6 Drift versus degradation analysis
Explicitly separate the two:
- Drift is a change in the data.
- Degradation is a measured increase in error or loss of coverage.

Include scenarios showing all four cases: no drift and no degradation; drift without degradation; degradation without obvious drift; drift with degradation. Drift alone must never be reported as proof of poor performance.

### 8.7 Retraining recommendation report
Produce a deterministic, human-readable report that states:
- Which monitoring condition was triggered.
- The supporting evidence (drift metrics, error trend, coverage trend).
- A suggested action: no action, investigate, or retrain.

The recommendation is advisory. A person decides. Retraining, if done, is simply re-running the normal training pipeline; its metrics are tracked in MLflow and compared offline in a report. No automated approval, promotion, or replacement logic.

### 8.8 Data-quality monitoring
Report incoming data quality over time: schema violations, out-of-range values, missingness, stuck sensors, and duplicate timestamps.

### 8.9 CI/CD
Extend GitHub Actions to run Ruff, Mypy, Pytest, integration tests, and a Docker build.

### Acceptance criteria
A single documented command runs a reproducible monitoring scenario:
1. Train an initial model and log it in MLflow.
2. Simulate incoming data in time windows.
3. Apply a controlled distribution shift.
4. Produce a drift report.
5. Compute delayed-label performance and calibration for each window.
6. Show at least one case of drift without degradation and one of real degradation.
7. Generate the retraining recommendation report.

All reported results must come from actual runs.

---

## B. Other edits needed so the spec stays consistent

1. **Section 0, "Strict architectural boundaries":** add these items to the DO NOT list:
   - Model registry stages (candidate, staging, production).
   - Automated or gated model promotion.
   - Model rollback.
   - Shadow, champion-challenger, or A/B model deployment.
2. **Section 1 (Overview):** change "Experiment tracking and model lifecycle management" to "Experiment tracking, data and model monitoring". Add to explicit non-goals: "No model promotion, rollback, or deployment-slot logic."
3. **Section 2 (Tech stack):** remove the "Model registry: MLflow Model Registry" row.
4. **Section 4 (Repo structure):** remove `monitoring/registry.py`; rename `monitoring/retraining.py` to `monitoring/recommendation.py`; add `monitoring/data_quality.py`.
5. **Phase 6 (API):** keep "model version" as the MLflow run ID in responses. Remove anything about model switching.
6. **Phase 7 (Dashboard), "Model analytics":** replace "Current model version / Model performance history" with "Model run ID, metrics, interval coverage over time, drift summaries, data-quality summaries".
7. **Phase 9 (Demo):** replace "Candidate model evaluation and promotion" with "Drift report, delayed-label performance, calibration over time, and retraining recommendation".
8. **Section 6 (MLOps evaluation):** replace the list with: experiment tracking, lineage, drift monitoring, data-quality monitoring, delayed-label performance monitoring, calibration monitoring, retraining recommendation.
9. **Section 7 (Definition of Done):** replace "MLOps lifecycle is demonstrated" with "Monitoring scenario is reproducible and documented".

---

## C. One-line instruction to give Copilot

> Read PHASE_8_REVISION.md. Replace Phase 8 in PROJECT_SPEC.md with the new version and apply the consistency edits in section B. Do not change any other phase. Then update PROGRESS.md to reflect the revised Phase 8 checklist.