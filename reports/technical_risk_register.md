# Technical Risk Register

| Register Info     | Details                              |
|-------------------|--------------------------------------|
| Project           | views-baseline                       |
| Owner             | Project maintainers                  |
| Last Updated      | 2026-06-02                           |
| Total Concerns    | 13                                   |
| Open Concerns     | 10                                   |
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

See also C-05 (related: same code locations, different problem — C-03 is duplication, C-05 is validation absence).

---

### C-05: Unvalidated MultiIndex structure assumption across all model classes

| Field | Value |
|-------|-------|
| ID | C-05 |
| Tier | 2 |
| Source | repo-assimilation (2026-06-01) |
| Trigger | When the upstream data loader in views-pipeline-core changes the index level ordering, or when a contributor passes a flat-indexed or 3-level DataFrame to a model — all 5 models silently extract wrong index names via positional `df.index.names[0]`/`[1]` |
| Location | `views_baseline/model/baseline.py:30-31,83-84,143-144,223-224,329-330` |

Every model class extracts `self.time_idx = df.index.names[0]` and `self.entity_idx = df.index.names[1]` in `fit()` without validating that the DataFrame has a 2-level MultiIndex or that the levels are in the expected `(time, entity)` order. If the index levels are reversed, the model would set `self.time_idx` to the entity name and vice versa. Subsequent time-based filtering (e.g., `df.index.get_level_values(self.time_idx) < test_start`) would compare entity IDs against a month boundary, producing structurally valid but semantically wrong predictions with no error signal. Currently mitigated only by the convention that views-pipeline-core's data loader always produces `(time, entity)` ordering.

See also C-03 (related: same code locations, C-03 addresses duplication, C-05 addresses validation absence).

---

### C-06: `partition_dict` structure assumed but never validated

| Field | Value |
|-------|-------|
| ID | C-06 |
| Tier | 3 |
| Source | repo-assimilation (2026-06-01) |
| Trigger | When upstream changes the `partition_dict` schema (e.g., renames `"test"` key or changes tuple to a dict), all 5 models fail with `KeyError` or `TypeError` deep in `fit()`/`predict()` rather than at the validation boundary |
| Location | `views_baseline/model/baseline.py:44,82,101,142,168,222,261,328,382` |

All five model classes access `self.partition_dict["test"][0]` as the test-start boundary without any structural validation. The `ReproducibilityGate` validates hyperparameter keys but has no data-layer contracts — `partition_dict` is not audited. A malformed partition dict would cause loud failures (`KeyError`, `TypeError`) but at deep call sites inside `fit()` and `predict()`, far from the config boundary where such validation belongs. Currently mitigated by views-pipeline-core's data loader providing a consistent schema.

---

### C-07: No enforcement that `targets` columns exist in input DataFrame

| Field | Value |
|-------|-------|
| ID | C-07 |
| Tier | 3 |
| Source | repo-assimilation (2026-06-01) |
| Trigger | When a downstream config in views-models declares a target column name that does not exist in the loaded DataFrame — `fit()` fails with `KeyError` deep in groupby/slice logic rather than at the validation boundary |
| Location | `views_baseline/model/baseline.py:89,153,247,348` (`fit()` methods slicing `df[self.targets]` or `df.groupby(...)[self.targets]`) |

The config's `targets` list is passed through to models and used to slice DataFrame columns. Neither the `ReproducibilityGate`, `BaselineModelCatalog`, nor any model's `fit()` method verifies that these column names actually exist in the input DataFrame. The failure is loud (`KeyError`) but occurs deep in pandas groupby/slice operations, producing error messages that point to the DataFrame rather than the misconfigured target list. Currently mitigated only by the convention that views-models configs name columns that exist in the data loader's output.

See also C-06 (related: both are data-layer boundary validation gaps not covered by the ReproducibilityGate).

---

### C-08: Duplicated lazy `PredictionFrame` import in distributional predict() methods

| Field | Value |
|-------|-------|
| ID | C-08 |
| Tier | 4 |
| Source | repo-assimilation (2026-06-01) |
| Trigger | When views-pipeline-core renames or restructures the `data.prediction_frame` module path — both lazy import sites must be updated simultaneously, and omitting one produces a `ModuleNotFoundError` only when that specific model's `predict()` is exercised |
| Location | `views_baseline/model/baseline.py:259,380` |

`ConflictologyModel.predict()` and `MixtureBaseline.predict()` each contain an identical lazy import: `from views_pipeline_core.data.prediction_frame import PredictionFrame`. The duplication is mandated by ADR-002 (no module-level pipeline-core imports in `model/`) and acknowledged in ADR-002 and ADR-013. The risk is low — Python's module cache makes it functionally equivalent to a single site — but the pattern scales linearly with each new distributional model (ADR-012 Step 1 requires it). Currently mitigated by the fact that both sites are tested by the distributional model tests.

---

### C-09: Dependency topology rules enforced by convention only

| Field | Value |
|-------|-------|
| ID | C-09 |
| Tier | 3 |
| Source | repo-assimilation (2026-06-01) |
| Trigger | When a contributor adds a module-level `from views_pipeline_core import ...` in any `model/` file — the foundation-layer testability guarantee (ADR-002, ADR-013) is silently broken with no CI feedback |
| Location | `views_baseline/model/` (entire package — 4 source files) |

ADR-002 and ADR-013 define strict dependency topology rules: `model/` must have zero module-level imports from views-pipeline-core. These rules are enforced only by code review. No `import-linter` config, ruff plugin, or CI grep check exists. ADR-013 explicitly notes this as a known gap and suggests a CI lint rule. A single forbidden import would break the foundation-layer testability guarantee — `test_baseline.py`, `test_catalog.py`, and `test_protocol.py` would fail in environments where pipeline-core is absent or broken, and the failure would be attributed to pipeline-core rather than the import violation.

---

### C-10: `seed` parameter not included in ReproducibilityGate audit

| Field | Value |
|-------|-------|
| ID | C-10 |
| Tier | 3 |
| Source | repo-assimilation (2026-06-01) |
| Trigger | When a downstream config in views-models intends to use a specific seed but omits or misspells the `seed` key — the model silently uses default `seed=42`, producing valid but non-reproducible-relative-to-intent predictions with no warning |
| Location | `views_baseline/infrastructure/reproducibility_gate.py:27-33` (ALGORITHM_GENOMES), `views_baseline/model/baseline.py:201,309` (default seed=42 in constructors) |

The `seed` parameter for `ConflictologyModel` and `MixtureBaseline` has a default value of `42` in the constructor, following ADR-011's RNG determinism contract. However, `seed` is not listed in `ALGORITHM_GENOMES` and therefore not audited by `audit_manifest()`. If a downstream config intends `seed: 123` but misspells it as `seeed: 123`, the model silently falls back to `seed=42`. The predictions are internally consistent (same seed → same output) but differ from the intended config, violating the reproducibility guarantee at the experiment level. The `ReproducibilityGate` was designed to catch exactly this class of "present but wrong" configuration errors (see ADR-014 §4: None-value rejection), but `seed` was excluded because it has a sensible default.

---

### C-11: Ensemble manager incompatibility untracked in PredictionFrame migration plan

| Field | Value |
|-------|-------|
| ID | C-11 |
| Tier | 2 |
| Source | expert-review (2026-06-02) |
| Trigger | When baseline models switch to PredictionFrame output (GitHub #8) but the ensemble configuration still uses `EnsembleManager` (DataFrame-only) — ensemble evaluation fails with FileNotFoundError for all baseline constituents because it looks for `.parquet` files and finds `.npy` |
| Location | `views_baseline/manager/baseline_manager.py` (return types), `views_pipeline_core/managers/ensemble/ensemble.py:82` (DataFrame-only ensemble), `views_pipeline_core/managers/ensemble/prediction_frame_ensemble.py:109` (PF-only ensemble), GitHub issues #8-#11 |

GitHub issues #8–#11 migrate baseline output from DataFrame to PredictionFrame but do not track ensemble manager compatibility as a prerequisite. Pipeline-core has two separate ensemble managers: `EnsembleManager` (works exclusively with `pd.DataFrame`, loads `.parquet` prediction files) and `PredictionFrameEnsembleManager` (works exclusively with `PredictionFrame`, loads `.npy` files). After migration, baseline predictions can only be consumed by the PF ensemble. No issue verifies which ensemble currently consumes baseline outputs or tracks the ensemble config change as a dependency. This is the critical integration point — without it, the migration lands cleanly in views-baseline but breaks the monthly prediction cycle at ensemble evaluation time.

---

### C-12: `skip_predictions_delivery` config key added without effect analysis

| Field | Value |
|-------|-------|
| ID | C-12 |
| Tier | 3 |
| Source | expert-review (2026-06-02) |
| Trigger | When `skip_predictions_delivery: True` is added to baseline configs per GitHub issue #10 — prediction file delivery may be suppressed in the pipeline-core stage layer, causing downstream consumers to find no output files |
| Location | GitHub issue #10, `views_pipeline_core/managers/forecasting/stage.py` (delivery logic), `views_pipeline_core/managers/prediction/io.py` (save path) |

Issue #10 requires adding `skip_predictions_delivery: True` to all PredictionFrame baseline configs because pipeline-core's `CoreConfigSniffer` mandates this key when `prediction_format == "prediction_frame"`. However, views-hydranet — the only repo that completed this migration — does not use `skip_predictions_delivery` anywhere in its codebase. The flag's actual effect on baseline's simpler delivery path has not been analyzed. It may suppress the standard prediction file write in `ForecastingStage`, which baseline's ensemble integration may depend on. Adding config keys copied from a sniffer's mandatory list without understanding their runtime effect risks silently disabling file output.

See also C-11 (related: both concern the PredictionFrame migration's downstream effects).

---

### C-13: No sample-count validation in PredictionFrame concat aggregation

| Field | Value |
|-------|-------|
| ID | C-13 |
| Tier | 2 |
| Source | expert-review (2026-06-02) |
| Trigger | When a future ensemble config lists constituent models with different `n_samples` (e.g., a baseline point model producing `(N, 1)` alongside distributional models producing `(N, 64)`) and uses `aggregation: "concat"` — the ensemble produces `(N, K_total)` where the point model contributes 1/K_total of samples, being statistically ignored while appearing to be included |
| Location | `views_pipeline_core/managers/ensemble/prediction_frame_ensemble.py:78-97` (`_aggregate_prediction_frames` validates `n_rows` and `identifiers` but not `y_pred.shape[1]`) |

`_aggregate_prediction_frames()` performs raw `np.concatenate(axis=1)` for the `concat` method without checking that all constituent PredictionFrames have the same number of samples. A point model PF with shape `(N, 1)` concatenated with three 64-sample PFs produces `(N, 193)` where the point model contributes 0.5% of samples — statistically negligible but appearing as a full ensemble member. Downstream CRPS and calibration metrics are computed on this lopsided distribution, producing valid-looking but misleading results. The `arithmetic_mean` path is accidentally stricter (`np.stack` fails on shape mismatch). The older DataFrame-based `AggregationModule` (`aggregator.py:365-444`) has weighted resampling that handles heterogeneous counts correctly, but PFE lacks this. Currently theoretical — no mixed ensemble exists and baselines serve only as evaluation benchmarks, not ensemble constituents — but the guard should exist regardless.

See also C-11 (related: both concern the PredictionFrame migration's interaction with ensemble infrastructure).

---

## Disagreements

### D-01: Clean break vs. gradual dual-format migration for PredictionFrame adoption

| Field | Value |
|-------|-------|
| ID | D-01 |
| Source | expert-review (2026-06-02) |
| Perspectives | Feathers (gradual — the dual-format seam enables per-model testing and rollback if ensemble integration fails), Hickey/Ousterhout (clean break — baseline has 5 trivial models and 880 LOC, transient dual-format complexity is not justified, hydranet proved the direct switch works at larger scale), Beck (gradual only if integration-tested at each step; without an integration test, gradual is just slow risk accumulation) |
| Resolution | Unresolved. Recommend clean break given baseline's small surface area and hydranet's successful precedent (ADR-047). |

---

### D-02: Sample-count asymmetry — current defect vs. future concern

| Field | Value |
|-------|-------|
| ID | D-02 |
| Source | expert-review (2026-06-02) |
| Perspectives | Nygard/Kleppmann (design defect now — silent acceptance of invalid input should be fixed in pipeline-core regardless of current usage; the code validates 2 of 3 structural properties and the missing third enables silent data corruption), Feathers/Ousterhout (future concern — no ensemble triggers it today, golden_hour is homogeneous at 3×64, the fix is important but not urgent), Beck (write the test now — `test_aggregate_rejects_heterogeneous_sample_counts()` forces the design decision without requiring the full fix) |
| Resolution | Unresolved. Recommend fixing now — 5 lines of validation in pipeline-core, zero risk, prevents a class of silent corruption. See C-13. |

---

### D-03: Baseline migration deferral mechanism

| Field | Value |
|-------|-------|
| ID | D-03 |
| Source | expert-review (2026-06-02) |
| Perspectives | Feathers (event-triggered: un-defer baseline issues #8-#11 when golden_hour issue #126 closes — clear, concrete, no drift), Ousterhout (time-boxed: proceed independently if golden_hour hasn't shipped by a deadline — prevents indefinite blocking on another team's timeline), Hickey (permanent deferral until someone actually needs baselines in a PFE ensemble — don't build for hypothetical requirements; baselines currently serve only as evaluation benchmarks, not ensemble constituents) |
| Resolution | Unresolved. Recommend event-triggered with time-box fallback: "Un-defer when pipeline-core golden_hour #126 closes, or by 2026-07-15, whichever first." |

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
- **Cross-repo:** This register tracks views-baseline-specific risks only. Entries may reference external code locations (e.g., pipeline-core ensemble managers) when the risk is about baseline's interaction with that code. The root-cause fix may belong in another repo, but the risk to baseline is tracked here. The parent risk for C-59 (filename coupling across engine repos) is tracked in the `views-pipeline-core` register.
