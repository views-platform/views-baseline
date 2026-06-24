# Technical Risk Register

| Register Info     | Details                              |
|-------------------|--------------------------------------|
| Project           | views-baseline                       |
| Owner             | Project maintainers                  |
| Last Updated      | 2026-06-24                           |
| Total Concerns    | 23                                   |
| Open Concerns     | 20                                   |
| Resolved Concerns | 3                                    |

---

## Tier Definitions

| Tier | Severity | Description |
|------|----------|-------------|
| 1 | Critical | Silent data corruption or model output correctness risk. Requires immediate attention. |
| 2 | High | Structural fragility that will cause failures under realistic change scenarios. |
| 3 | Medium | Maintainability or coupling issues that increase cost of change. |
| 4 | Low | Code quality concerns that do not affect correctness or reliability. |

Tiers reflect **expected risk** (impact × likelihood), not impact alone. A silent-correctness risk with a low-likelihood trigger can sit at Tier 2 rather than Tier 1 — the narrative should state the impact-vs-likelihood reasoning when it does (see C-05, C-13).

---

## Open Concerns

> **Epic #22 status update (2026-06-24) — implemented in the working tree, pending commit/merge.**
> The views-frames adoption epic (#22; stories #23–#27; ADR-020) has been implemented and verified
> green (104 tests) in the canonical editable `views_pipeline` env. Effect on the entries below:
> - **C-16** — resolved in code: all `PredictionFrame` construction now goes through the single
>   `to_prediction_frames` seam against the `views_frames` leaf; zero `identifiers=` calls remain.
> - **C-08** — resolved in code: the three lazy-import/construction sites are consolidated into one.
> - **C-15** — resolved in code: dead `build_prediction_grid` deleted.
> - **C-19** — the half-migration/scattered-construction half is resolved (one seam; both distributional
>   sites migrated); the **OCP registry** half remains deferred per ADR-012.
> - **C-18** — mitigated: `level` is single-sourced from the declared `loa` and validated against the
>   index (`resolve_level`); the silent-mislabel path is closed at the boundary.
> - **C-05** — mitigated: the positional `(time, entity)` index assumption is now validated at the
>   model boundary by `resolve_level`.
> - **C-17** — partially addressed: `views-frames` is now declared and the pipeline-core pin set to
>   `>=3.0.0,<4.0.0` (matching `views-hydranet`); `poetry.lock`/PyPI-CI remain gated on the
>   platform-wide pipeline-core 3.0.0 publish (not a views-baseline code defect).
>
> These entries stay **Open** until the change is committed/merged, at which point they move to
> Resolved with dates (header counts updated then).

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

**Tier rationale (impact vs. likelihood):** The *impact* is Tier 1 — this is the register's clearest silent-correctness case (semantically wrong predictions, no error). It sits at Tier 2 because *likelihood* is low: the `(time, entity)` ordering is governed by a stable upstream convention. If that convention ever becomes configurable or a code path begins constructing model input DataFrames directly, re-evaluate for Tier 1.

See also C-03 (related: same code locations, C-03 addresses duplication, C-05 addresses validation absence).

**Forward link (FeatureFrame input, 2026-06-04):** this same direct consumption of the DataFrame `(time, entity)` MultiIndex is what makes views-baseline a *high-effort* consumer in the platform's FeatureFrame-input migration — point models read `df.index.names` / `df[targets]` directly, so an input-format switch breaks `fit()` unless adapted (unlike adapter-fronted engines such as hydranet/r2darts2). Adopting FeatureFrame input (which carries its own validation) could *subsume* this risk. Tracked at `views-platform/views-pipeline-core#161` (input path) and `views-platform/views-pipeline-core#162` (contract hardening); see ADR-019 (proposed, not implemented).

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

**Status update (review-rr 2026-06-04):** Investigation during the migration established that no ensemble lists any baseline model as a constituent (`models` list) — baselines appear only in `regression_point_baselines`/`regression_sample_baselines` for evaluation benchmarking. The realistic likelihood of a production ensemble break is therefore much lower than first assessed. Prerequisite tracked in GitHub issue #12. Keep open until the migration (PR #15) merges and issue #12 confirms the consuming ensemble's manager/format.

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

**Status update (review-rr 2026-06-04):** The PFE production roadmap (pipeline-core `2026-06-01_pfe_production_roadmap.md` §4.4) clarifies that `skip_predictions_delivery` controls **only Track B** (the Arrow/parquet write); Track A+ (`.npy` for PF-ensemble consumption) is always written. PR #76 applied `True` to all 9 point-model configs, matching the deployed ranger models. The runtime effect is now understood; residual concern is only confirming no current downstream consumer reads baseline Track-B parquet. Likelihood downgraded; keep open until that consumer check is done.

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

### C-14: Point baselines cannot satisfy views-models' PF readiness contract (cross-repo catch-22)

| Field | Value |
|-------|-------|
| ID | C-14 |
| Tier | 3 |
| Source | repo-assimilation (2026-06-05) |
| Trigger | When a contributor enables a point baseline (`ZeroModel`/`LocfModel`/`AverageModel`) for production by setting `prediction_format: "prediction_frame"` in its views-models config and runs the integration suite or `TestPFModelConfigReadiness` — the check fails (`n_posterior_samples` got `None`, `regression_targets` got `None`) because a deterministic model has no posterior samples |
| Location | `views_baseline/infrastructure/reproducibility_gate.py:24` (`CORE_GENOME`, no `n_posterior_samples`); `views-models` `tests/test_pfe_production_readiness.py::TestPFModelConfigReadiness` (`_discover_pf_models()` asserts `n_posterior_samples` positive int + `regression_targets` non-empty); evidenced by `integration_test_2026-06-04` |

A structural catch-22 spans the repo boundary. PR #15 made all baseline models — including the three deterministic point models — *always* return `dict[str, PredictionFrame]`. Consequently their config **must** declare `prediction_format: "prediction_frame"` (otherwise pipeline-core routes the returned `dict` through the DataFrame save path and crashes). But that declaration subjects them to views-models' readiness contract, which operationally defines `prediction_frame ⇒ sampled/distributional` and requires `n_posterior_samples` — a quantity deterministic point models cannot meaningfully supply. views-baseline's own `CORE_GENOME` does not include `n_posterior_samples`, so the contradiction is **invisible to this repo's gate and tests**; it surfaces only in the downstream suite as a loud test failure (not silent corruption). The resolution is cross-repo — a point-aware readiness check and a first-class point/stochastic content descriptor — tracked at views-models #81 and `views-platform/views-pipeline-core#159`; see ADR-018 ("What is explicitly NOT done here").

See also C-13 (related root cause: the ecosystem treating PredictionFrame as inherently sampled) and C-12 (related: PF-migration config keys adopted from a downstream sniffer's mandatory list).

---

### C-15: Dead `build_prediction_grid` DataFrame builder retained and tested post-migration

| Field | Value |
|-------|-------|
| ID | C-15 |
| Tier | 4 |
| Source | repo-assimilation (2026-06-05) |
| Trigger | When a contributor adds a sixth model or modifies output handling and reuses `build_prediction_grid()` believing it is a supported output path — they build on a helper that no production code path exercises after the PredictionFrame migration |
| Location | `views_baseline/model/helpers.py:12-48` (`build_prediction_grid`); `tests/test_helpers.py` (`test_build_prediction_grid_empty`) |

`build_prediction_grid()` — the legacy DataFrame builder that returns a MultiIndexed `pred_{target}` frame — is retained in `helpers.py` and still carries a unit test for its empty case, but is **called by nothing** in production after PR #15 (all models now use `build_prediction_frame()` → `dict[str, PredictionFrame]`). ADR-018 documents it as "retained for reference, called by nothing." The risk is purely drift: a maintainer reading `helpers.py` may infer a live DataFrame output path still exists, because the function is present and green-tested. No correctness or reliability impact; single-developer scope. Candidate for deletion or an explicit `@deprecated`/docstring marker disambiguating it from `build_prediction_frame()`.

---

### C-16: views-baseline un-migrated to the extracted `views-frames` PredictionFrame (governed break; DIP/SDP root cause)

| Field | Value |
|-------|-------|
| ID | C-16 |
| Tier | 2 |
| Source | repo-assimilation (2026-06-24) |
| Trigger | When any model's `predict()` executes against a `views-pipeline-core` that carries #188/PR #206 (the local editable checkout is already on `development` past #206; any clean install resolving to 3.0.0 likewise) — `PredictionFrame.__init__()` raises `TypeError: got an unexpected keyword argument 'identifiers'`, because the constructor is now the `views_frames` leaf `(y_pred, index: SpatioTemporalIndex(time, unit, level), metadata=None)` while the code passes `identifiers={...}` |
| Location | `views_baseline/model/helpers.py:136` (`build_prediction_frame`), `views_baseline/model/baseline.py:285` (`ConflictologyModel.predict`), `views_baseline/model/baseline.py:402` (`MixtureBaseline.predict`); contract also asserted in `README.md:38,70`; protocol return typed `-> dict` not `dict[str, PredictionFrame]` at `views_baseline/model/protocol.py:23,41` |

This is **not accidental dependency drift** but an un-adopted **governed platform migration**: the platform's Frame types were extracted into a standalone `views-frames` package (the "frames replace dataframes" program; ADR-017/018/019), and `views-pipeline-core` #188 (PR #206) retires its local `PredictionFrame` to re-export the `views_frames` leaf — a breaking change destined for pipeline-core **3.0.0**. The local editable `views-pipeline-core` checkout is already on `development` past #206 (its `prediction_frame.py` is now `from views_frames import PredictionFrame`), so the suite is **26/98 red in this env** even though pip metadata still reads `2.3.0`; the 72 green tests are exactly those that never construct a real frame. The failure is loud (`TypeError`) and total, not silent — hence Tier 2, not Tier 1. The **root cause is architectural, not a version number**: the model layer depends directly on the *concrete, volatile* platform leaf, constructing it inline at three sites, and baseline owns no boundary abstraction of its own (a DIP/SDP/SAP violation — high-level policy depending toward the least-stable component). That is why C-01, this break, and the pending FeatureFrame *input* migration are the same root cause surfacing repeatedly. The fix is a baseline-owned frames boundary (one adapter), not a pin bump alone. Specced and tracked by **views-baseline #21** (lockstep with pipeline-core #188).

See also C-01 (RESOLVED — its resolution rationale is invalidated here), C-08 (the lazy-import workaround the boundary adapter subsumes), C-05/C-14 (the same input/output complecting), C-17 (dependency-declaration hygiene), C-18 (level-inference correctness), C-19 (scattered construction / OCP), and D-04/D-06. Tracked by GitHub issue #21.

---

### C-17: `views-frames` undeclared and pin admits the breaking pipeline-core release

| Field | Value |
|-------|-------|
| ID | C-17 |
| Tier | 2 |
| Source | expert-review (2026-06-24) |
| Trigger | When `poetry update`/`poetry lock` or a clean install resolves dependencies — `views-pipeline-core` may resolve to the breaking `3.0.0` (admitted by `<=3.0.0`), and `views-frames` (now the owner of the core `PredictionFrame`/`SpatioTemporalIndex`/`SpatialLevel` types) is not declared at all, so the break is pulled in with no lockfile or contract-test guard |
| Location | `views_baseline/pyproject.toml:11-13` (`[tool.poetry.dependencies]`, single dep `views-pipeline-core >=2.3.0,<=3.0.0`; no `views-frames`) |

`pyproject.toml` declares exactly one runtime dependency and pins it `>=2.3.0,<=3.0.0` — a range whose upper bound *includes* the release (3.0.0) that retires the old `PredictionFrame` (C-16). The package that now owns the data structures flowing through the entire system, `views-frames`, is undeclared, so its version floats uncontrolled in any environment that has it transitively. There is no committed lockfile guarantee and no contract test that constructs the *real* leaf, so in a clean install pinned at 2.3.0 the break is invisible until 3.0.0 lands. This is a Stable-Dependencies / Stable-Abstractions failure: baseline depends toward an unmanaged, volatile surface. Mitigation (none applied yet): declare `views-frames` with a compatible range, bump the pipeline-core pin to `>=3.0.0,<4.0.0` per #21, commit the lockfile, and add one canary test that builds a real `views_frames.PredictionFrame` through the boundary adapter.

See also C-16 (the realized break), C-01 (RESOLVED — same pin-discipline class).

---

### C-18: Spatial-level mislabel if `SpatialLevel` is inferred from the positional entity index

| Field | Value |
|-------|-------|
| ID | C-18 |
| Tier | 2 |
| Source | expert-review (2026-06-24) |
| Trigger | When the #21 migration derives `SpatialLevel` from `df.index.names[1]` (e.g. `_level_from_entity_idx`: `"country_id"→CM`, else `PGM`) rather than from the declared `loa`, and a loader later renames the entity level (`priogrid_gid`/new alias), flips index order, or introduces a third level — the output `PredictionFrame` is stamped with the wrong `SpatialLevel` and no error is raised |
| Location | Prospective per #21 at `views_baseline/model/helpers.py:~138` and `views_baseline/model/baseline.py:285,402`; positional-index root at `views_baseline/model/baseline.py:30-31,83-84,143-144,223-224,329-330` |

The #21 spec derives the required `SpatialLevel` by string-matching the entity index name (`df.index.names[1]`). This is the same unvalidated positional-index assumption already tracked as C-05, but with a new and more dangerous consequence: the level is now *written into the output frame's identity*. A fall-through default (`else PGM`) means a renamed or reordered entity level produces a frame labelled `PGM` when it is `CM` (or vice versa) — a **silent mislabel** that downstream evaluation, aggregation, and joins consume as truth. Impact is Tier-1-shaped (silent output incorrectness, no signal); it sits at Tier 2 because the likelihood is gated on *how* #21 is implemented — the risk is fully preventable by single-sourcing the level. ADR-003 ("authority of declarations over inference") favours deriving from the declared `loa`; #21 chose inference; the two are independent sources of truth for one fact and can disagree. Mitigation: derive `level` once from a single source, and assert `df.index.names == (time_idx, entity_idx)` **and** `loa ⇔ entity_idx` agreement at the model boundary, raising on mismatch.

See also C-05 (acute new instance of the same positional-index root cause) and D-05 (the declared-vs-inferred level disagreement).

---

### C-19: Output construction scattered across three sites; model addition is modification, not extension (OCP)

| Field | Value |
|-------|-------|
| ID | C-19 |
| Tier | 3 |
| Source | expert-review (2026-06-24) |
| Trigger | When a contributor applies the #21 migration only to its two *listed* sites (`helpers.py`, `baseline.py:285`) — `MixtureBaseline.predict()` at `baseline.py:402` retains the old `identifiers=` constructor and ships half-migrated; and when adding a sixth model, three parallel registries (catalog dispatch dict, `ALGORITHM_GENOMES`, `_get_*` method) must be edited in lockstep or the model is silently unavailable/unvalidated |
| Location | `views_baseline/model/baseline.py:285,402` (two of three construction sites), `views_baseline/model/helpers.py:136` (the third); `views_baseline/model/catalog.py:24-30` + `views_baseline/infrastructure/reproducibility_gate.py:27` + `catalog.py:53-88` (three parallel registries) |

`PredictionFrame` is constructed at three independent sites and #21's location list names only two of them — an omission surface that ships a working point/Conflictology path while `MixtureBaseline` (`baseline.py:402`) breaks. Separately, adding a model is an Open/Closed violation: it requires coordinated edits to the catalog dispatch dict, the genome table, and a `_get_*` factory method (ADR-012 codifies the ritual). Both are the same structural issue — construction and registration logic spread across files instead of localized behind one seam. No correctness impact today beyond the half-migration risk; the cost is maintainability and a recurring edit surface for every platform change and every new model. Mitigation: collapse all frame construction into one boundary adapter (point and sample), and replace the three registries with a single registration table keyed by model name.

See also C-08 (the duplicated lazy import the adapter subsumes), C-16 (the migration that exposes the three sites), D-04.

---

### C-20: `views_frames.PredictionFrame` silently accepts zero-sample `(N, 0)` arrays

| Field | Value |
|-------|-------|
| ID | C-20 |
| Tier | 3 |
| Source | pr-review (2026-06-24) |
| Trigger | When any platform code constructs a `PredictionFrame` from a zero-width sample array — e.g. a degenerate `n_samples=0` config reaching a distributional model — `views_frames.PredictionFrame` accepts it (`sample_count == 0`) instead of raising, producing a meaningless empty-sample frame that downstream evaluation/aggregation consumes as if valid |
| Location | External: `views_frames.PredictionFrame` (no sample-count > 0 guard). Mitigated in this repo at `views_baseline/model/helpers.py` `to_prediction_frames` (raises on `y_pred.shape[1] == 0`); regression test `tests/test_baseline.py::test_conflictology_n_samples_zero_raises` |
| Narrative | The retired `views-pipeline-core` `PredictionFrame` rejected zero-sample arrays ("at least one sample column"); the extracted `views_frames` leaf dropped that validation (confirmed: `PredictionFrame(np.empty((2,0)), idx)` succeeds with `sample_count == 0`). For views-baseline this fail-loud guarantee (ADR-008) is **restored at the single construction seam** `to_prediction_frames`, so the repo's risk is mitigated. The platform-wide gap remains for any other code path that constructs `PredictionFrame` directly. Root-cause fix belongs in `views-frames` (a constructor guard); this entry tracks baseline's interaction and the cross-repo follow-up. |

See also C-13 (the related sample-axis validation gap in PFE concat aggregation) and C-16 (the views-frames migration that surfaced this).

---

### C-21: Governance docs still describe the removed `isinstance` manager dispatch (pre-existing drift)

| Field | Value |
|-------|-------|
| ID | C-21 |
| Tier | 4 |
| Source | pr-review (2026-06-24) |
| Trigger | When a contributor (human or AI) reads the manager-dispatch description in the docs below and writes or reviews code on the assumption that the manager routes via `isinstance(model, DistributionalBaselineModel)` — the manager has had a single type-uniform path since ADR-017 / PR #15, so the guidance is wrong |
| Location | `docs/ADRs/004_*.md:129`, `005_*.md:51`, `009_*.md:86-89`, `010_*.md:13,29,86`, `docs/CICs/BaselineForecastingModelManager.md:56`, `docs/contributor_protocols/hardened_protocol_template.md:166`, `carbon_based_agents.md:56` |

ADR-017 unified all models onto `dict[str, PredictionFrame]` output and removed the `isinstance(model, DistributionalBaselineModel)` dispatch from the manager (`_generate_predictions`); `distributional = True` is now a semantic marker only. ADR-010 itself records the removal, yet several other governance documents still describe the dispatch as live. This is **pre-existing drift** (it predates the views-frames epic #22) and is documentation-only — no code impact — but it misleads contributors, which matters under the silicon-agent protocol (ADR-007) where stale docs are treated as authoritative. Epic #22 corrected this where it already touched docs (silicon protocol, ADR-003, the two distributional CICs); the remaining sites are listed above. Out of scope for #22 (a bounded migration, not a doc sweep); tracked here for a dedicated governance-doc pass.

See also C-16 / ADR-020 (the epic that corrected the adjacent construction-site drift).

---

### C-22: Baseline PredictionFrames carry no provenance metadata

| Field | Value |
|-------|-------|
| ID | C-22 |
| Tier | 4 |
| Source | pr-review (2026-06-24) |
| Trigger | When a downstream consumer needs run identity (`model`, `run_type`, `seed`, `run_id`, `data_version`) attached to a baseline forecast — the frames built by `to_prediction_frames` carry an empty `FrameMetadata`, so provenance must be reconstructed from outside the frame |
| Location | `views_baseline/model/helpers.py` `to_prediction_frames` — `PredictionFrame(y_pred, index)` is constructed with no `metadata` argument |

`to_prediction_frames` is the single construction chokepoint (ADR-020), which makes it the natural — and only — place to stamp `FrameMetadata` (views-frames v1.4.0 added `run_id`/`data_version`) if downstream ever wants run identity carried on the frame itself. Not required for the #21 migration and not a correctness issue; recorded as an enhancement opportunity that the seam makes trivial to add later. No current consumer requests it.

---

### C-23: `value_fn` invoked once per horizon step though point baselines are constant across the horizon

| Field | Value |
|-------|-------|
| ID | C-23 |
| Tier | 4 |
| Source | pr-review (2026-06-24) |
| Trigger | When a point model's `value_fn` becomes non-trivial (e.g. a per-cell computation rather than a constant or a `.loc` lookup) — `build_prediction_frame` calls it `output_length` times per entity even though the value is identical across the forecast horizon, so the redundancy becomes a real cost |
| Location | `views_baseline/model/helpers.py` `build_prediction_frame` inner loop (`for _ in time_ids: values[idx, 0] = value_fn(cid, target)`) |

Point baselines are constant across the horizon, so `value_fn(cid, target)` returns the same value for every `tid`; calling it once per `(cid, target)` and broadcasting across `time_ids` would be equivalent. Correct as written and negligible today (the `value_fn`s are constants or single `.loc` lookups); flagged only so it is revisited if a point model with an expensive `value_fn` is ever added. Any change must preserve the entity→time→target fill order (ADR-011).

> **Status (2026-06-25):** resolved in the working tree by tech-debt-cleanup — `value_fn` is now evaluated once per `(entity, target)` and filled across the horizon (helpers.py `build_prediction_frame`), preserving the ADR-011 order and all values (104 tests green). Stays Open until merge.

---

## Disagreements

### D-01: Clean break vs. gradual dual-format migration for PredictionFrame adoption

| Field | Value |
|-------|-------|
| ID | D-01 |
| Source | expert-review (2026-06-02) |
| Perspectives | Feathers (gradual — the dual-format seam enables per-model testing and rollback if ensemble integration fails), Hickey/Ousterhout (clean break — baseline has 5 trivial models and 880 LOC, transient dual-format complexity is not justified, hydranet proved the direct switch works at larger scale), Beck (gradual only if integration-tested at each step; without an integration test, gradual is just slow risk accumulation) |
| Resolution | **Resolved in practice (2026-06-04).** Clean break implemented in PR #15: all point models switched directly to `dict[str, PredictionFrame]`, the isinstance dispatch was removed, and no dual-format path was introduced. Final on merge of PR #15. |

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
| Resolution | **Moot — overtaken by events (2026-06-04).** The deferral never took effect: the baseline migration proceeded directly (PR #15 in views-baseline, PR #76 in views-models) rather than waiting on golden_hour. The deferral-mechanism question (event-triggered vs time-boxed vs permanent) is therefore no longer live. Retained for audit trail. |

---

### D-04: Apply #21 site-by-site vs. route all construction through one boundary adapter

| Field | Value |
|-------|-------|
| ID | D-04 |
| Source | expert-review (2026-06-24) |
| Perspectives | Hickey/Ousterhout/Martin (one boundary module — the recurring break is a DIP/SDP/SAP violation; constructing the leaf at each site re-buys the same churn at the next platform change), Feathers/Beck (incremental is right, but via a *single* test-gated seam so the change is reviewable and the `:402` site cannot be missed), Nygard (either is acceptable provided dependency pins and a real-leaf canary test land) |
| Resolution | **Resolved toward the single boundary adapter (2026-06-24).** DIP/SDP/SAP decide it rather than preference; the maintainer confirmed a principle-driven framing (SOLID + component-cohesion + screaming architecture). Tracked by the views-frames-boundary epic (#21 and successors). See C-16, C-19. |

---

### D-05: Spatial level from observed entity index (#21) vs. declared `loa` (ADR-003)

| Field | Value |
|-------|-------|
| ID | D-05 |
| Source | expert-review (2026-06-24) |
| Perspectives | Kleppmann (derive from the *declared* `loa`, validated against the index — inference from `df.index.names[1]` risks a silent CM/PGM mislabel and conflates "declared" with "observed"), #21 author (derive from the *observed* `entity_idx` as the data's ground truth — pragmatic and local to the construction site) |
| Resolution | Unresolved. Recommend single-sourcing the level and validating `loa ⇔ entity_idx` agreement at the boundary, reconciled explicitly with ADR-003. See C-18. |

---

### D-06: How much to invest given ~880 LOC and a single maintainer

| Field | Value |
|-------|-------|
| ID | D-06 |
| Source | expert-review (2026-06-24) |
| Perspectives | Ousterhout/Hickey (the trivial size makes the clean restructure *cheap* — that is the argument *for* doing the full boundary + one-class-per-file restructure now, not against), YAGNI counter (it is a benchmark library; recurring small edits at each platform change may be an acceptable tax rather than a restructure) |
| Resolution | **Resolved by the maintainer (2026-06-24)** toward the full principle-driven restructure (SOLID + REP/CCP/CRP/ADP/SDP/SAP + screaming architecture / one-concept-per-file). Scoped as the views-frames-boundary epic. |

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
