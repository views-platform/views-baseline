# Technical Risk Register

| Register Info     | Details                              |
|-------------------|--------------------------------------|
| Project           | views-baseline                       |
| Owner             | Project maintainers                  |
| Last Updated      | 2026-05-19                           |
| Total Concerns    | 4                                    |
| Open Concerns     | 1                                    |
| Resolved Concerns | 3                                    |

---

## Tier Definitions

| Tier | Severity | Description |
|------|----------|-------------|
| 1 | Critical | Silent data corruption or model output correctness risk. Requires immediate attention. |
| 2 | High | Structural fragility that will cause failures under realistic change scenarios. |
| 3 | Medium | Maintainability or coupling issues that increase cost of change. |
| 4 | Low | Code quality concerns that do not affect correctness or reliability. |

---

## Open Concerns

### C-03: Duplicated index extraction across all 5 model classes

| Field | Value |
|-------|-------|
| ID | C-03 |
| Tier | 4 |
| Source | tech-debt-audit (2026-04-27) |
| Trigger | When adding a sixth baseline model class, the developer must copy the `self.time_idx = df.index.names[0]` / `self.entity_idx = df.index.names[1]` pattern and the `test_start = self.partition_dict["test"][0]` extraction — omitting either silently produces incorrect index handling |
| Location | `views_baseline/model/baseline.py:30-31,83-84,143-144,223-224,329-330` (index extraction), `baseline.py:44,82,101,142,168,222,261,328,382` (test_start extraction) |

All five model classes (`ZeroModel`, `LocfModel`, `AverageModel`, `ConflictologyModel`, `MixtureBaseline`) independently extract `time_idx`, `entity_idx`, and `test_start` in their `fit()` and `predict()` methods. The pattern is identical in every case. This is a DRY violation but deliberately minimal — each model is intentionally self-contained. The risk is low because the pattern is simple and well-tested across all models. Could be extracted to a helper or base class if a sixth model is added.

---

## Disagreements

(No disagreements registered yet.)

---

## Resolved Concerns

### C-02: `_evaluate_sweep` method has no test coverage — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-02 |
| Resolved | 2026-05-19 |
| Resolution | `test_manager_evaluate_sweep` added in `tests/test_baseline_manager.py`. Verifies that `_evaluate_sweep` loads data and delegates to `_generate_predictions`, returning identical output to direct model invocation. |

---

### C-04: `artifact_name` parameter silently ignored in evaluate/forecast — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-04 |
| Resolved | 2026-05-19 |
| Resolution | `if artifact_name:` branching implemented in both `_evaluate_model_artifact` and `_forecast_model_artifact`, following the stepshifter pattern. Tested by `test_uses_specified_artifact_timestamp` in `tests/test_falsification_ship_readiness.py`. |

---

### C-01: Version floor admits unreleased views-pipeline-core — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-01 |
| Resolved | 2026-04-28 |
| Resolution | views-pipeline-core bumped to `v2.3.0` (commit `0f87358`) on its `fix/C-59-cached-data-path-coupling` branch. Both branches ship together; the `>=2.3.0` floor in `pyproject.toml` now resolves correctly. |

---

## Register Conventions

- **ID format:** `C-xx` for concerns, `D-xx` for disagreements. IDs are permanent — gaps in numbering indicate merged or resolved entries
- **Sources:** `repo-assimilation`, `expert-review`, `test-review`, `falsification-audit`, `clean-architecture-review`, `pr-review`, `tech-debt-audit`, `incident`
- **Resolution:** Move to "Resolved Concerns" with resolution date and summary when addressed
- **Header counts:** Manually maintained — update whenever a concern is added or resolved
- **Cross-repo:** The parent risk for C-59 (filename coupling across engine repos) is tracked in the `views-pipeline-core` register. This register tracks views-baseline-specific risks only.
