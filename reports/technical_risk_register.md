# Technical Risk Register

| Register Info     | Details                              |
|-------------------|--------------------------------------|
| Project           | views-baseline                       |
| Owner             | Project maintainers                  |
| Last Updated      | 2026-04-27                           |
| Total Concerns    | 3                                    |
| Open Concerns     | 3                                    |
| Resolved Concerns | 0                                    |

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

### C-01: Version floor admits unreleased views-pipeline-core

| Field | Value |
|-------|-------|
| ID | C-01 |
| Tier | 2 |
| Source | falsification-audit (2026-04-27) |
| Trigger | When a CI runner or new contributor installs views-baseline from the `development` branch, `pip`/`poetry` resolves `views-pipeline-core>=2.3.0` — if `v2.3.0` has not been released yet, the install fails or falls back to `v2.2.0` which lacks `_get_cached_data_path()`, causing `AttributeError` at runtime |
| Location | `pyproject.toml:11` |

The `_get_cached_data_path()` method (commit `bbdba39` in views-pipeline-core) exists only on the `fix/C-59-cached-data-path-coupling` branch and is not in any tagged release through `v2.2.0`. The version floor was bumped from `>=2.0.0` to `>=2.3.0` to make the dependency explicit, but `v2.3.0` does not exist yet. This PR cannot be safely merged to `development` until views-pipeline-core ships a release containing the method. Currently mitigated by the fact that the views-pipeline monorepo typically installs from local checkouts, not from PyPI — but any CI pipeline or fresh clone would hit this.

See also pipeline-core C-59 (filename coupling across engine repos), D-12 (pass DataFrame to engines).

---

### C-02: `_evaluate_sweep` method has no test coverage

| Field | Value |
|-------|-------|
| ID | C-02 |
| Tier | 4 |
| Source | pr-review (2026-04-27), tech-debt-audit (2026-04-27) |
| Trigger | When a developer modifies `_evaluate_sweep()` logic (e.g., adding sweep-specific data transforms or changing the prediction call), there is no test to detect regressions |
| Location | `views_baseline/manager/baseline_manager.py:122-130` |

`_evaluate_sweep()` is a two-line method that loads data via `_get_cached_data_path()` and delegates to `_generate_predictions()`. Both paths it depends on are tested individually, and the method is trivially correct. However, it is the only public lifecycle method on `BaselineForecastingModelManager` with zero test coverage. Already noted as a Known Deviation in the CIC. If WandB sweeps are ever used with baseline models, a test should be added first.

---

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

(No resolved concerns yet.)

---

## Register Conventions

- **ID format:** `C-xx` for concerns, `D-xx` for disagreements. IDs are permanent — gaps in numbering indicate merged or resolved entries
- **Sources:** `repo-assimilation`, `expert-review`, `test-review`, `falsification-audit`, `clean-architecture-review`, `pr-review`, `tech-debt-audit`, `incident`
- **Resolution:** Move to "Resolved Concerns" with resolution date and summary when addressed
- **Header counts:** Manually maintained — update whenever a concern is added or resolved
- **Cross-repo:** The parent risk for C-59 (filename coupling across engine repos) is tracked in the `views-pipeline-core` register. This register tracks views-baseline-specific risks only.
