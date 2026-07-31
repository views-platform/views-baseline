# Technical Risk Register

| Register Info     | Details                              |
|-------------------|--------------------------------------|
| Project           | views-baseline                       |
| Owner             | Project maintainers                  |
| Last Updated      | 2026-07-31                           |
| Total Concerns    | 38                                   |
| Open Concerns     | 12                                   |
| Resolved Concerns | 25                                   |
| Withdrawn         | 1 (C-27 — Tweedie removed)           |

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

## Causal Clusters

Root-cause groupings from the 2026-07-31 strategic review. Most are now resolved; the index
keeps the shared causes visible so a future regression is read against its cluster, not in
isolation.

| Cluster | Root cause | Entries | Status |
|---------|-----------|---------|--------|
| K1 — Frames boundary (DIP/SDP) | Model layer coupled to concrete platform data types; no baseline-owned seam | C-16, C-08, C-31, C-19, C-05, C-18 | Resolved via `to_prediction_frames` + `to_feature_frame` (ADR-019/020); only C-19 OCP half deferred (ADR-012) |
| K2 — numpy-port byte-identity (epic #47) | pandas→numpy rewrite risking silent forecast drift + weak guard tests | C-32, C-33, C-34, C-35, C-36, C-37, C-29 | Resolved & merged (PR #60); only C-30 (numpy pin) deferred |
| K3 — Seed / reproducibility | Seed not wired; determinism-only tests | C-10, C-24, C-25, C-29 | Resolved & merged (ADR-021, #31) |
| K4 — PredictionFrame-as-sampled ecosystem | Platform treats PF as inherently distributional; point models don't fit | C-11, C-13, C-14, C-20 | Open, cross-repo (pipeline-core / views-models) |
| K5 — Input-boundary validation | Boundary doesn't validate index order / partition_dict / targets | C-05, C-06, C-07 | Partly subsumed by K1; C-06/C-07 residual |
| K6 — Governance-doc drift | Docs lag code changes | C-21, C-28 | C-28 resolved; C-21 residual sites |
| K7 — Distribution / packaging | Cross-repo dependency on unpublished pipeline-core | C-38, C-17 | C-17 superseded; C-38 live, blocked on pipeline-core #319 |

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

> **Status (2026-07-19):** the distributional `predict()`-side duplication is reduced — the shared grid / RNG / entity→time→target fill scaffold now lives once in `helpers.sample_prediction_grid` (all four distributional models delegate; see C-19). **Still open:** the `fit()`-side `time_idx`/`entity_idx`/`test_start` extraction and the point-model (`ZeroModel`/`LocfModel`/`AverageModel`) `predict()` extraction remain duplicated — low value, deferred. (tech-debt-cleanup)

---

### C-05: Unvalidated MultiIndex structure assumption across all model classes

| Field | Value |
|-------|-------|
| ID | C-05 |
| Tier | 3 |
| Source | repo-assimilation (2026-06-01) |
| Trigger | When the upstream data loader in views-pipeline-core changes the index level ordering, or when a contributor passes a flat-indexed or 3-level DataFrame to a model — all 5 models silently extract wrong index names via positional `df.index.names[0]`/`[1]` |
| Location | `views_baseline/model/baseline.py:30-31,83-84,143-144,223-224,329-330` |

Every model class extracts `self.time_idx = df.index.names[0]` and `self.entity_idx = df.index.names[1]` in `fit()` without validating that the DataFrame has a 2-level MultiIndex or that the levels are in the expected `(time, entity)` order. If the index levels are reversed, the model would set `self.time_idx` to the entity name and vice versa. Subsequent time-based filtering (e.g., `df.index.get_level_values(self.time_idx) < test_start`) would compare entity IDs against a month boundary, producing structurally valid but semantically wrong predictions with no error signal. Currently mitigated only by the convention that views-pipeline-core's data loader always produces `(time, entity)` ordering.

**Tier rationale (impact vs. likelihood):** The *impact* is Tier 1 — this is the register's clearest silent-correctness case (semantically wrong predictions, no error). It sits at Tier 2 because *likelihood* is low: the `(time, entity)` ordering is governed by a stable upstream convention. If that convention ever becomes configurable or a code path begins constructing model input DataFrames directly, re-evaluate for Tier 1.

See also C-03 (related: same code locations, C-03 addresses duplication, C-05 addresses validation absence).

**Forward link (FeatureFrame input, 2026-06-04):** this same direct consumption of the DataFrame `(time, entity)` MultiIndex is what makes views-baseline a *high-effort* consumer in the platform's FeatureFrame-input migration — point models read `df.index.names` / `df[targets]` directly, so an input-format switch breaks `fit()` unless adapted (unlike adapter-fronted engines such as hydranet/r2darts2). Adopting FeatureFrame input (which carries its own validation) could *subsume* this risk. Tracked at `views-platform/views-pipeline-core#161` (input path) and `views-platform/views-pipeline-core#162` (contract hardening); see ADR-019 (proposed, not implemented).

> **Re-scope (2026-07-31, strategic review):** the FeatureFrame input boundary this entry's forward-link named as potentially subsuming it (**C-31**) is now resolved and merged — `to_feature_frame` validates loa↔level and rejects NaN / non-integer / overflow indices at the boundary. The positional-index root is largely closed on the frame path; residual exposure is only the raw-DataFrame dual-input path. **Downgraded to Tier 3 on 2026-07-31** on this evidence (positional-index root closed on the frame path; residual only on the raw-DataFrame dual-input path).

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

> **Partial mitigation (2026-07-31):** an AST guard now walks all `Import` nodes and fails CI on a runtime `import pandas` in the model layer (C-36 resolution, `tests/test_falsification_reorg_principles.py`) — concrete enforcement of one topology rule. The general “no module-level `views_pipeline_core` import in `model/`” rule remains convention-only (no import-linter). Kept Open.

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

> **Status (2026-07-19):** the **distributional output-construction / predict-scaffold** half is **resolved**. All four distributional models (`ConflictologyModel`, `MixtureBaseline`, `ParametricConflictology`, `ParametricHurdleConflictology`) now build output through one helper — `helpers.sample_prediction_grid(..., draw_cell)`, the distributional analogue of `build_prediction_frame` — which routes to the single `to_prediction_frames` seam; each `predict()` supplies only a per-cell `draw_cell(cid, target, rng)` closure. Behaviour-identical (byte-identity reproducibility tests, 174/174; ruff clean; `baseline.py` 586→529 LOC). The original half-migration trigger (`MixtureBaseline` old constructor) was already closed by the views-frames epic (ADR-020). **Still deferred:** the three-parallel-registries OCP half (catalog dispatch + `ALGORITHM_GENOMES` + `_get_*` factory) per ADR-012. (tech-debt-cleanup)

> **Re-confirmed (2026-07-20, falsify probe P2):** the PR-1 reorg did not change the OCP posture — `catalog.py` still hardcodes a 7-entry dispatch dict + seven `_get_*` methods, so adding a model remains a modification of existing files plus an `ALGORITHM_GENOMES` edit (not a pure extension). Guarded by `tests/test_falsification_reorg_principles.py::test_ocp_catalog_does_not_hardcode_model_registry` (strict xfail — ratchets when a single registration table replaces the three registries). Consolidation stays deferred per ADR-012 (catalog < ~10 models).

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

### C-27: Tweedie `lam` is unbounded when the derived index `p` clamps to the ceiling (OOM + unclamped magnitude)

| Field | Value |
|-------|-------|
| ID | C-27 |
| Tier | 3 — **WITHDRAWN 2026-07-18** |
| Source | review-diff (2026-07-18, epic #33 S9) |
| Trigger | When `fit_tweedie` is called on a **low-zero-rate, low-dispersion** window (so the derived `p = 2 − m²/(var·lam0)` exceeds 2 and clamps to `2 − eps`), making `lam = m²/(var·(2−p)) ≈ 1000·m²/var` — e.g. a new/low-dispersion regression target routed to `ParametricConflictology(family="tweedie")` |
| Location | `views_baseline/model/distributions.py` (`fit_tweedie` `lam` recompute; `sample_tweedie` `rng.gamma(..., size=total)` allocation); `views_baseline/model/baseline.py` (`ParametricConflictology.predict` native-zero path applies no clamp) |

> **WITHDRAWN (2026-07-18):** the Tweedie family was **removed** from views-baseline in S9 (code deleted, not fixed), so this risk no longer exists. Diagnosis that drove the removal: on the real pgm data the OOM/ceiling case (`p≥2`) fired **zero** times (max clamped λ = 0.3); what actually occurred was the *benign* `p≤1` clamp on ~40% of active cells — windows with 1–2 events in 36 months that a continuous family fundamentally cannot represent. The only honest handling of that infeasibility was either clamp (hides misspecification) or empirical fallback (turns the parametric model back into conflictology) — neither acceptable — so Tweedie was excluded from the baseline. See ADR-022 (Tweedie exclusion record) and `reports/closeness_experiment/FINDINGS.md` §S9. Original narrative retained below for provenance.

The Tweedie index `p` is derived from the window's mean/var/zero-rate and clamped to `(1+eps, 2−eps)` to stay in the compound Poisson-Gamma regime — but the clamp bounds only the *index*, not the rate. When `p` clamps to the ceiling, `lam` is recomputed as `1000·m²/var` and left unbounded. `sample_tweedie` then draws `rng.poisson(lam, size)` and allocates `rng.gamma(alpha, theta, size=total)` with `total = counts.sum()`; a pathological cell drives `lam` to 1e4–1e6 → a 1e7–1e9-element jump array → **OOM/hang**. Coupled second consequence: the native-zero path in `ParametricConflictology.predict` applies **neither** `clamp_log` **nor** `clamp_floor` (those guard the hurdle/log1p paths), so the same `lam` also yields an **unbounded emitted magnitude** — the failure class `EMIT_LOG_CEIL` exists to prevent. Did not fire in the S8 experiment because conflict windows are high-zero-rate/high-dispersion (`p` clamps toward 1, `lam` small); it is an atypical-but-realistic edge for a low-dispersion target. Tier 3 (not 2): needs an atypical window, no silent corruption in the intended zero-inflated regime, and the fix is a localized ceiling. **Not yet fixed** — a deliberate policy call: a single-sourced `_TWEEDIE_LAM_CEIL` (WARN on cap) bounds both the array size and the magnitude but under-preserves the mean in the already-approximate clamped-`p` regime. See also C-26 (the gumbel floor, fixed) as the sibling native-zero/positive-part numeric guard, and the `EMIT_LOG_CEIL` ceiling it mirrors.

---

### C-32: Byte-identity risk of the PR-2 numpy-on-FeatureFrame migration (silent forecast drift)

| Field | Value |
|-------|-------|
| ID | C-32 |
| Tier | 4 |
| Source | falsify (2026-07-20, PR-1 reorg audit — forward risk of the P1 remediation); plan (epic #47 PR-2) |
| Trigger | When PR-2 S6–S8 (issues #53–#55) rewrite `window_pool` and the point/Mixture/parametric `fit` aggregations from pandas (`groupby(...).tail`, `.xs`, sort order) to numpy-on-FeatureFrame — any change to entity ordering, tail-selection order, or float dtype silently shifts the seeded RNG stream and therefore the sampled `y_pred` |
| Location | `views_baseline/model/frames/pooling.py` (`window_pool`), the model `fit()` aggregations under `views_baseline/model/models/**`; gated by `tests/test_golden.py` (byte-identity) + `tests/test_pooling.py` |

The distributional models' output is reproducible only because a single seeded `np.random.default_rng` is advanced in a fixed **entity→time→target** order over pandas-derived pools (ADR-011). Re-deriving those pools with numpy changes nothing *if and only if* the entity order, the per-entity tail (last `window_months`) order, and the emitted float dtype are bit-for-bit preserved. A subtle divergence (e.g. numpy sort vs pandas `sort_index` stability, `unique()` ordering, `float64` vs `float32`) produces **different draws with no error signal** — wrong forecasts that pass every structural test. Impact is high (silent forecast incorrectness); likelihood is real during the rewrite; it is **not Tier 1** only because the golden characterization tests (C-29) provide a loud, total catch *if run at every step*. Mitigation: port one model at a time, gate each step on `test_golden.py` + `test_pooling.py` byte-identity, and treat any golden diff as a stop-the-line defect (not a regenerate-the-baseline event) until the change is proven order-preserving.

See also C-29 (the golden tests that gate this), C-30 (their numpy-Generator coupling), C-31 (the DIP migration this executes), C-11/C-25 (the RNG-order reproducibility contract this must preserve).

> **Decision (2026-07-20) — FeatureFrame is float32; golden regenerated once, deliberately.** S5 (the `to_feature_frame` adapter) surfaced that `views_frames.FeatureFrame` is a **float32** container by design (class contract; `coerce_values` casts; no float64 option), while the existing pools and golden tests are float64. Strict float64 byte-identity through a FeatureFrame is therefore impossible. The maintainer chose **FeatureFrame-canonical (float32)** with a **one-time, deliberate** golden regeneration to the platform's real precision (not a maintenance regen). To keep the migration safe despite the regen, the ordering/logic guard is **decoupled from the precision change**: (1) the ported numpy `window_pool` is proven byte-identical to the pandas `window_pool` on a **float64** panel (built directly from the df) — this pins entity order, tail order, and RNG advance independent of dtype (`test_pooling`, permanent); (2) only then is the model source switched to the float32 FeatureFrame and the model-level golden regenerated **once**, with the diff reviewed to confirm it is float32-rounding-only and nothing else. After the flip, `test_golden.py` pins the float32-canonical output and byte-identity holds going forward.

> **Discharged (2026-07-20, PR-2 S6–S10).** The migration landed with the ordering guard intact and **no golden regeneration was actually required**: the distributional golden fixtures use small-integer data (`(t*3+u*7)%9`), which is float32-exact, so routing it through the float32 FeatureFrame is lossless and `test_golden.py` stayed **byte-identical unchanged** — confirming the numpy port preserved entity/tail/RNG order and per-cell logic. The float64 ordering guard (`test_pooling::test_window_pool_arrays_matches_pandas_reference_in_float64`, using non-float32-exact values) independently pins the order/logic. The float32 precision change is therefore invisible on the golden set and only affects non-integer production magnitudes (the accepted trade-off). One deliberate behaviour change shipped alongside: `window_months <= 0` now fails loud (`ValueError`) at fit for all windowing models, replacing the pre-PR-2 accidental KeyError / silent-NaN. Risk retained at Tier 2 as documentation of the precision boundary; the acute trigger (a non-order-preserving rewrite) has passed.

> **Merged (2026-07-20, code-review #60 finding #6).** Two additions from the max-effort review: (1) the `_from_dataframe` code comment `"float64 block ... byte-identity with the pandas path"` (`model/frames/input.py`) is **now false** — `FeatureFrame.from_2d` immediately downcasts to float32, so the whole model layer (production df path included) computes on float32-rounded values; the comment is corrected in WS4. (2) The reach is a **reproducibility**, not merely precision, effect for non-float32-exact targets: `MixtureBaseline`'s global pool `v[v > 0]` and `ParametricHurdleConflictology`'s `pool == 0.0` / `pool > 0` zero-spike split are computed on float32 values, so a value that rounds across the 0 boundary changes the pool contents/size and reindexes the seeded `rng.choice` — the draws diverge from the pre-PR float64 path (still within the accepted FeatureFrame-canonical decision, but broader than "rounding").

> **Re-tiered 2→4 (2026-07-31, strategic review):** the acute non-order-preserving-rewrite trigger has passed — the numpy port merged and `test_golden.py` stayed byte-identical. Retained as documentation of the float32 precision boundary, not a live structural risk; kept Open as a standing boundary note.

---

### C-38: views-baseline 1.0.0 is published to PyPI but not installable until views-pipeline-core 3.0.0 ships there

| Field | Value |
|-------|-------|
| ID | C-38 |
| Tier | 3 |
| Source | release (2026-07-31, PyPI v1.0.0 publish — deliberate "publish now anyway" tradeoff) |
| Trigger | When anyone runs `pip install views-baseline` (or a `uv sync` / `uv pip install` that resolves it) from PyPI **before** `views-pipeline-core 3.0.0` is published there — dependency resolution fails hard because the required `views-pipeline-core>=3.0.0,<4.0.0` has no release on pypi.org (only 2.3.0 exists). The same unresolvable dependency keeps the `check` matrix in `run_tests.yml` red. |
| Location | `pyproject.toml:36` (`"views-pipeline-core>=3.0.0,<4.0.0"`); `.github/workflows/run_tests.yml` (`check` job `uv sync`, and the header note documenting the expected-red state); external blocker `views-platform/views-pipeline-core#319` (publish 3.0.0 to PyPI) |

views-baseline 1.0.0 was uploaded to PyPI via Trusted Publishing, reserving the name and version, **while one of its two hard dependencies is not on PyPI at all**. `views-frames>=1.3.0` resolves (it is published, 1.10.1); `views-pipeline-core>=3.0.0` does not — pipeline-core's newest PyPI release is 2.3.0, and 3.0.0 exists only in-repo (held from publication for non-technical reasons). Consequently the published artifact is **installable in name only**: every `pip install views-baseline` from a clean index errors at resolution, and the uv `check` CI job cannot `uv sync` for the same reason (documented, expected-red). The maintainer accepted this knowingly to reserve the release and complete the packaging pipeline ahead of the dependency; the working test gate remains the editable `views_pipeline` conda env, not the PyPI install.

**Tier rationale (impact vs. likelihood):** the failure is **loud** (a resolution error, never silent corruption) and has a single, known discharge condition, so it is not Tier 2 fragility. Blast radius today is small — no consumer is expected to `pip install` this yet (the conda env is the sanctioned path) — but it is a genuine cross-repo coupling that increases friction and confusion until cleared, hence Tier 3. **Discharge:** resolve when views-pipeline-core 3.0.0 is published to PyPI (`#319`) and a clean `pip install views-baseline==1.0.0` resolves; the `check` CI job goes green at the same moment. Tracked from this repo by the pandas/packaging follow-up issue and cross-repo by pipeline-core #319.

See also C-17 (the prior pin-hygiene entry — **inverted** here: 3.0.0 is now intentionally *required* rather than dangerously *admitted*), C-16 (the code break the `>=3.0.0` floor exists to require the fix for), and C-31/C-32 (the FeatureFrame-native work that made the frame path pandas-free, orthogonal to this distribution gap).

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
| Resolution | **Resolved (2026-07-31):** the epic #22 migration single-sourced `level` from the declared `loa` via `resolve_level`, validated against the index — the recommended reconciliation, decided in favour of Kleppmann / ADR-003 (declared over inferred). See C-18 (mitigated). |

---

### D-06: How much to invest given ~880 LOC and a single maintainer

| Field | Value |
|-------|-------|
| ID | D-06 |
| Source | expert-review (2026-06-24) |
| Perspectives | Ousterhout/Hickey (the trivial size makes the clean restructure *cheap* — that is the argument *for* doing the full boundary + one-class-per-file restructure now, not against), YAGNI counter (it is a benchmark library; recurring small edits at each platform change may be an acceptable tax rather than a restructure) |
| Resolution | **Resolved by the maintainer (2026-06-24)** toward the full principle-driven restructure (SOLID + REP/CCP/CRP/ADP/SDP/SAP + screaming architecture / one-concept-per-file). Scoped as the views-frames-boundary epic. |

---

### D-07: `seed` optional-with-default now vs required-in-genome now (the C-10 fix)

| Field | Value |
|-------|-------|
| ID | D-07 |
| Source | expert-review (2026-06-25) |
| Perspectives | Kleppmann / ADR-003 (declare `seed` in the genome and validate it — a reproducibility knob must not silently default), Feathers / Nygard (making it required breaks the ~9 downstream configs that omit `seed` → a config-time outage, C-24 → ship optional-with-default first and coordinate the required change separately), Hickey (both are lesser evils; give the gate an explicit "optional-with-default" notion to dissolve the dilemma) |
| Resolution | Recommend **optional-with-default now** (forward `seed` + single default), **required-in-genome later** via a coordinated views-baseline + views-models PR pair. See C-10, C-24. |

---

### D-08: Patch the one dropped param vs fix the parallel-list root (C-19 registry)

| Field | Value |
|-------|-------|
| ID | D-08 |
| Source | expert-review (2026-06-25) |
| Perspectives | Martin / GoF / Ousterhout (the dropped `seed` is a symptom of three hand-maintained parallel param lists — constructor signature ↔ `MODEL_GENOMES` ↔ factory body; the durable fix is the C-19 registration table that makes "audited" and "forwarded" one list), Feathers / Beck + the bounded mandate (patch `seed` now with a param-completeness test; don't balloon into the registry) |
| Resolution | Recommend **patch + param-completeness test now**; the C-19 registry remains the deferred durable fix (ADR-012 threshold). See C-19, C-10. |

---

### D-09: Where the default seed `42` lives

| Field | Value |
|-------|-------|
| ID | D-09 |
| Source | expert-review (2026-06-25) |
| Perspectives | Hickey / Kleppmann / Ousterhout (one module-level `DEFAULT_SEED` constant — single source of truth), minimal-diff view (inline `get("seed", 42)` is an acceptable two-line fix) |
| Resolution | Recommend a single `DEFAULT_SEED` constant — trivial and forecloses C-25. |

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

> **Archived from Open (2026-07-31, strategic review).** The entries below were resolved-and-merged to `main`/`development` (verified against the working tree) but had been left in Open by now-stale “stays Open until merge” caveats. Relocated verbatim — full narratives and status trails preserved. C-17 is resolved as *superseded by C-38*.

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

### C-10: `seed` is never forwarded from config to the distributional models (config/sweep seed silently ignored)

| Field | Value |
|-------|-------|
| ID | C-10 |
| Tier | 2 |
| Source | repo-assimilation (2026-06-01); mechanism corrected + re-tiered via model-review of white_ranger (2026-06-25) |
| Trigger | When any config or WandB sweep sets `seed` for `ConflictologyModel`/`MixtureBaseline` (e.g. white_ranger's sweep over `[42, 123, 456]`) — the value is silently dropped and the model always uses the hardcoded default `seed=42`. Seed sweeps are therefore inert (every leg identical), and any config-declared non-42 seed produces results that differ from the declared config with no warning |
| Location | `views_baseline/model/catalog.py` `_get_conflictology_model` / `_get_mixture_model` — **neither factory passes `seed=` to the constructor**; `views_baseline/model/baseline.py` (`seed: int = 42` default; `np.random.default_rng(self.seed)`); `views_baseline/infrastructure/reproducibility_gate.py` `ALGORITHM_GENOMES` (`seed` absent → not audited). Evidence: `views-models/models/white_ranger/configs/config_sweep.py` (inert `seed` sweep) |

**Mechanism corrected (2026-06-25):** the original entry framed this as "seed is not *audited*, so a *misspelled* key defaults." The actual defect is stronger — the catalog factory methods for both distributional models omit `seed` entirely, so `config["seed"]` **never reaches the model constructor**; both models always use the hardcoded default `42` regardless of what the config (correctly spelled or not) declares. Consequences: (1) a WandB **seed sweep is a no-op** — white_ranger sweeps `seed ∈ {42, 123, 456}` and every leg produces identical draws, which can support a false "results are seed-robust" conclusion; (2) any deployed baseline that declares `seed != 42` silently diverges from its own config. white_ranger is correct only by accident (its `seed: 42` equals the default). Re-tiered from 3 to 2: the failure is **silent** (no error), **already realized** (the sweep is inert now), and can produce **misleading experimental conclusions** about seed sensitivity. Fix is small: forward `seed=self.config.get("seed", 42)` in both factories, and add `seed` to `ALGORITHM_GENOMES` so it is audited (ADR-014 §4). Under investigation on a dedicated fix branch (2026-06-25).

See also C-14/ADR-014 (the reproducibility-gate contract this should extend to `seed`).

> **Status (2026-06-25):** fixed in the working tree (`fix/distributional-seed-not-forwarded`) under **ADR-021** — the catalog forwards `config["seed"]` strictly, `seed` is a required+audited `ALGORITHM_GENOMES` key, and a single `DEFAULT_SEED` sentinel backs direct construction. Enforced by `tests/test_falsification_seed_wiring.py`, catalog forwarding + param-completeness tests, and a gate rejects-missing-seed test (114 green). Stays Open until the **coordinated merge** with views-models #233 (declare `seed` in the 9 configs) lands together.
>
> **RESOLVED (2026-07-17):** merged — views-models#234 (declares `seed`, commit `d1187be`) then views-baseline#31 (genome + strict forward, merge `506523b`) landed on `development` in the safe order. All 20 distributional configs now declare `seed`; the fix is live.

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

> **SUPERSEDED (2026-07-31, strategic review):** the trigger references `poetry update`/`poetry lock`, but the repo migrated to hatchling / PEP 621 / uv this session, so that trigger is obsolete; and the mitigation this entry asked for — declare `views-frames`, bump the pin to `>=3.0.0,<4.0.0` — is now applied in `pyproject.toml`. The forward risk (the required 3.0.0 is absent from PyPI) is carried by **C-38**. Resolved as superseded, not by pin change alone.

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

### C-24: Promoting `seed` to a required genome key without updating downstream configs is a config-time outage

| Field | Value |
|-------|-------|
| ID | C-24 |
| Tier | 3 |
| Source | expert-review (2026-06-25) |
| Trigger | When the C-10 fix promotes `seed` to a **required** key in `ALGORITHM_GENOMES` (the ADR-003-correct end state) and ships **before** the distributional views-models configs that omit `seed` are updated — `audit_manifest` raises `MissingHyperparameterError` for every one of them, converting the silent C-10 bug into a hard config-time failure of all distributional baseline runs |
| Location | `views_baseline/infrastructure/reproducibility_gate.py` (`ALGORITHM_GENOMES` for `ConflictologyModel`/`MixtureBaseline`); downstream `views-models/models/{lucid_dream + the 8 MixtureBaseline models}/configs/config_hyperparameters.py` (currently omit `seed`) |

The correct long-term resolution of C-10 is to *declare* `seed` in the genome so `audit_manifest` validates it (ADR-003 declarations-over-inference; ADR-014 gate). But `seed` is currently optional-with-default, and ~9 distributional configs don't set it — making it required raises the gate for all of them at config time. This is a sequencing hazard, not a reason to avoid the genome change: forward `seed` as optional-with-default first (the C-10 code fix), then promote it to required only in a **coordinated views-baseline + views-models PR pair** after every distributional config sets `seed`. See also C-10 (the bug), C-14/ADR-014 (the gate contract), D-07.

> **Status (2026-06-25):** ADR-021 chose the declared-required end state directly (no optional-with-default interim), so the sequencing is now live: **views-models #233** filed to declare `seed` in the 9 configs that omit it; the views-baseline side (genome + strict forward) is ready on `fix/distributional-seed-not-forwarded`. **Must land together** — if views-baseline merges first, `audit_manifest` rejects those 9 configs (the intended loud failure, but only wanted post-config-update).
>
> **DISCHARGED (2026-07-17):** the coordinated merge completed in the safe order (views-models#234 first, then views-baseline#31) — no config-audit outage occurred.

---

### C-25: Fix that re-hardcodes the default seed `42` creates a third divergent copy

| Field | Value |
|-------|-------|
| ID | C-25 |
| Tier | 4 |
| Source | expert-review (2026-06-25) |
| Trigger | When the C-10 fix forwards `seed=self.config.get("seed", 42)` in the catalog and a later change alters the constructor default (`seed: int = 42` in `baseline.py`) without updating the catalog literal — catalog-built models silently use a different default than directly-built ones |
| Location | `views_baseline/model/baseline.py` (`seed: int = 42` in `ConflictologyModel.__init__` and `MixtureBaseline.__init__`) + the proposed `catalog.py` `get("seed", 42)` |

The default `42` already appears twice in `baseline.py`; the naive C-10 fix would add a third copy in `catalog.py`. Mitigation is trivial and should be part of the fix: introduce one module-level `DEFAULT_SEED = 42` referenced by both constructors and the catalog, so the default has a single source. See also C-10, D-09.

> **Status (2026-06-25):** resolved in the working tree — a single `DEFAULT_SEED = 42` in `baseline.py` backs both constructor defaults; the catalog reads `config["seed"]` strictly (no duplicated literal). `grep '= 42' views_baseline/model/` returns only the one definition.
>
> **RESOLVED (2026-07-17):** merged in views-baseline#31 (`506523b`).

---

### C-26: `ParametricHurdleConflictology` positive-part family can emit negative magnitudes (ℝ-support gumbel, no floor)

| Field | Value |
|-------|-------|
| ID | C-26 |
| Tier | 2 |
| Source | review-diff (2026-07-18, epic #33 S6) |
| Trigger | When a positive-part family whose support includes negatives (today `gumbel`; tomorrow any new `CONTINUOUS_FAMILIES` member, or a negative-capable transform) is fit and its samples are routed to output **without** passing through `clamp_floor` — the model emits negative conflict magnitudes, which no error signals and which silently distort the S8 closeness metrics against (non-negative) conflictology |
| Location | `views_baseline/model/baseline.py` (`ParametricHurdleConflictology.predict`, positive-part draw); `views_baseline/model/distributions.py` (`EMIT_FLOOR`/`clamp_floor`) |

The hurdle model's docstring calls `family` a "continuous **positive-part** family", but `scipy`/`numpy` `gumbel_r` has support on all of ℝ: its left tail draws below zero directly for `transform="none"`, and via `expm1(x)∈(-1,0)` for `x<0` under `transform="log1p"` (the `clamp_log`/`EMIT_LOG_CEIL` guard bounds only the *upper* tail). For a window with a heavy outlier the fitted gumbel `loc` goes negative and ≈40% of positive-part draws are negative pre-floor. A distributional baseline emitting negative fatalities violates the implicit non-negativity contract and biases the very metrics S8 uses to rank families. It went undetected because `test_hurdle_log1p_round_trips_to_raw_scale` asserted non-negativity but passed only *probabilistically* (~0.15% flake at seed 42). This is Tier 2 not Tier 1: the impact is silent output incorrectness, but the trigger is confined to the ℝ-support families and the fix closes it deterministically. See also C-20 (zero-sample PF), C-13 (sample-count validation) as sibling numeric-boundary guards.

> **Status (2026-07-18):** resolved in the working tree (same story, S6) — added a single-sourced `EMIT_FLOOR = 0.0` + `clamp_floor()` in `distributions.py` (mirror of `EMIT_LOG_CEIL`/`clamp_log`, WARN on floor), applied to the positive-part draws on the raw emitted scale in `ParametricHurdleConflictology.predict`. Guarded by `test_clamp_floor_*` unit tests and a deterministic `test_hurdle_gumbel_floors_negative_tail_to_nonnegative` (heavy-outlier window forces gumbel `loc<0` so the floor genuinely engages). Floor policy recorded in ADR-022 §3. 148 tests green, ruff clean.

---

### C-28: Governance-doc drift after the parametric-baseline epic (stale model/test counts + current-state claims)

| Field | Value |
|-------|-------|
| ID | C-28 |
| Tier | 3 |
| Source | review-base-docs (2026-07-18, epic #33 close-out) |
| Trigger | When a contributor or reviewer relies on a governance doc's current-state claim to scope work — e.g. reads `CICs/BaselineModelCatalog.md` to learn which classes the catalog imports/returns, or ADR-010/017/018 to confirm the return-type invariant, or a contributor-protocol "five model classes" enumeration — and acts on the stale count (under-tests a new model, mis-lists imports, mis-scopes a protocol touch) |
| Location | `docs/ADRs/{000,005,009,010,017,018}`, `docs/CICs/{BaselineModelCatalog,ReproducibilityGate,BaselineForecastingModelManager}.md`, `docs/INSTANTIATION_CHECKLIST.md`, `docs/contributor_protocols/{carbon_based_agents,silicon_based_agents,hardened_protocol_template}.md` |

Adding `ParametricConflictology` + `ParametricHurdleConflictology` (now **7** model classes, not 5) and running the epic (**165** tests across **14** files, not the documented "51 across 4") left standing current-state claims stale. **High** items are claims now factually wrong: "all 5/five baseline models return `dict[str, PredictionFrame]`" (ADR-010:27, ADR-017:35, ADR-018:27 — 7 do); the `BaselineModelCatalog` CIC's imports list (5 classes; code imports 7) and its "`list_models()` returns all 5" test-alignment rows; the `ReproducibilityGate` CIC's "All 5 model names are registered". **Medium** items are present-tense "five model classes" package descriptions and the (largely pre-existing) "51 tests across 4 files" counts. **Explicitly out of scope** (immutable per ADR-000): historical/decision records and dated snapshots that were accurate when written (ADR-002/003/008/011/020/021 model-pair references; ADR-010:11 "As of 2026-06-02…"; ADR-018:11 "98 tests"; ADR-020:18 "26/98 red"). See also C-21 (governance docs describing removed dispatch — the same drift class).

> **Status (2026-07-18):** resolved in the working tree — current-state claims corrected and brittle counts reworded to be durable (e.g. "every baseline model" rather than a hardcoded count) in the same pass; historical/dated records left intact per ADR-000. `docs/validate_docs.sh` green.

---

### C-29: No golden/characterization test pins the distributional sampling output (reproducibility tests are determinism-only)

| Field | Value |
|-------|-------|
| ID | C-29 |
| Tier | 2 |
| Source | test-review (2026-07-19) |
| Trigger | When a future refactor alters the RNG draw order or per-cell draw logic in `helpers.sample_prediction_grid` or a model's `draw_cell` closure while preserving output shape and marginal distribution — the change ships silently because every existing test still passes |
| Location | `tests/test_parametric.py`, `tests/test_baseline.py` (distributional predict tests); `views_baseline/model/helpers.py::sample_prediction_grid` |

The "reproducibility" tests (`test_reproducible_under_seed`, `test_no_hurdle_zinb_reproducible`, `test_hurdle_reproducible_under_seed`, `test_mixture_predict_reproducible`) build **two instances of the current code with the same seed and assert they agree** — verifying **determinism, not regression**. Both runs use whatever the current sampling does, so a draw-path change that keeps shape + marginal distribution passes all 174 tests. This is exactly the risk class of the `sample_prediction_grid` extraction (C-03/C-19 fix): its real safety came from mechanical faithfulness + the shape/distribution tests, not these self-referential "byte-identity" tests. Fix: a golden/characterization test — fixed seed + fixed tiny window → assert the exact `y_pred` array (or hash) for `ConflictologyModel`/`MixtureBaseline`/`ParametricConflictology`/`ParametricHurdleConflictology`. See also C-03/C-19 (the refactor this would guard), C-25/C-10 (the seed contract these depend on).

> **Status (2026-07-19):** **resolved in the working tree** — `tests/test_golden.py` added, pinning the exact `y_pred` of all four distributional models on a fixed seed + fixed window (`assert_array_equal` for the count families, `assert_allclose` for gamma). A draw-path change now fails loudly. Verified: 185 tests pass, ruff clean.

---

### C-31: Model layer depends on concrete pandas on the INPUT path; no FeatureFrame boundary (DIP — input side)

| Field | Value |
|-------|-------|
| ID | C-31 |
| Tier | 3 |
| Source | falsify (2026-07-20, PR-1 reorg principles audit — probe P1; folds P4) |
| Trigger | When an upstream producer switches the model input from a pandas `DataFrame` to a `views_frames.FeatureFrame` (the ADR-019 / PR-2 migration, issues #52–#58), every model `fit()`/`predict()` breaks because they read the concrete `df.index`/`groupby`/`.loc` API directly — there is no boundary adapter to absorb the switch |
| Location | `views_baseline/model/models/point/{zero,locf,average}.py`, `views_baseline/model/models/distributional/{conflictology,mixture,parametric_hurdle,parametric}.py`, `views_baseline/model/frames/pooling.py` — 8 modules `import pandas`; ~40 concrete-pandas call sites in `models/`. Guard: `tests/test_falsification_reorg_principles.py::test_dip_model_classes_do_not_import_pandas` (strict xfail) |

The PR-1 reorg established the screaming-architecture layout but is behaviour-preserving — it did **not** invert the pandas dependency. High-level model logic still imports and operates on the concrete pandas `MultiIndex` (`df.index.get_level_values`, `groupby(...).tail`, `.xs`, `.loc`), so the model layer depends toward a concrete, external data representation rather than a baseline-owned abstraction (a DIP violation on the *input* side — the mirror of C-16's now-resolved *output*-side leaf coupling). Consequence: baseline is a **high-effort** consumer of any platform input-format change (unlike adapter-fronted engines). This is the architectural gap ADR-019 / PR-2 (`to_feature_frame` adapter + numpy-on-FeatureFrame internals) is designed to close; a `strict=True` xfail ratchets it — it flips to a hard failure the moment PR-2 removes pandas from the model classes, forcing conversion to a permanent guard. **Secondary (P4, screaming architecture, Tier-4 cosmetic):** `frames/pooling.py` currently takes a `pd.DataFrame` and returns numpy dicts — it never touches a `views_frames` Frame, so its placement under `frames/` (the frame-boundary folder) screams the wrong responsibility until PR-2 S6 makes `window_pool` Frame-native. Mitigation: the PR-2 `to_feature_frame` boundary adapter (issue #52) confines pandas to one module; internals run on the FeatureFrame's numpy arrays.

See also C-16 (output-side DIP/SDP root cause, resolved via the `to_prediction_frames` seam), C-19 (the sibling OCP coupling), C-01 (direct MultiIndex consumption), ADR-019 (proposed → to be accepted in PR-2/S11). Same DIP/SDP root-cause family as C-16.

> **Resolved (2026-07-20, PR-2 S5–S9).** The DIP boundary is built: `to_feature_frame`
> (`model/frames/input.py`) is the single pandas reader; `window_pool` and every model
> fit/predict run on the FeatureFrame's numpy panel. No model module imports pandas at
> module scope (only under `TYPE_CHECKING`); pandas is confined to the lazy import inside
> `to_feature_frame`. The ratcheting guard flipped to a permanent passing test
> (`test_dip_model_layer_has_no_runtime_pandas_import`), and the folded P4 is resolved —
> `frames/pooling.py` is now Frame-native (`test_screaming_frames_package_is_only_about_frames`).
> ADR-019 accepted. Models accept `pd.DataFrame | FeatureFrame`; both routes are exactly
> equivalent (`test_dual_input.py`). Residual: the pure-FeatureFrame path has no upstream
> producer yet (architectural readiness, not a defect).

---

### C-33: Pooling models silently propagate NaN targets (NaN-skipping lost in the numpy port)

| Field | Value |
|-------|-------|
| ID | C-33 |
| Tier | 2 |
| Source | code-review (2026-07-20, PR #60 findings #2/#3) |
| Trigger | When `fit()` runs on a panel whose training window (`time <= train_end`) contains a NaN target value — a unit entering/leaving the panel, a missing boundary month, or a gappy feature. |
| Location | `views_baseline/model/frames/pooling.py:window_pool_arrays`; `views_baseline/model/models/point/locf.py:47`, `average.py:51` |

The pandas→numpy port dropped the pre-PR NaN tolerance. OLD `LocfModel` used `groupby(entity)[targets].last()` (last **non-null** value per target); the numpy port takes `window_pool(...,1)` then `pools[cid][t][0]` — the literal last row, NaN or not (and forces all targets to one row). OLD `AverageModel` used pandas `.tail(w)[targets].mean()` (skipna, divides by non-NaN count); the numpy port uses `np.ndarray.mean()`, which propagates NaN and divides by the full window. By extension the Conflictology/parametric/Mixture pools resample/fit on NaN too. A single NaN in a training window therefore becomes a NaN (or garbage) forecast **with no error signal**. Not silent enough to be Tier 1 — a NaN forecast surfaces as NaN evaluation metrics downstream — and likelihood is gated (VIEWS target panels are mostly dense), hence Tier 2. The one NaN-window test was replaced by a `window_months=0` fail-loud test, so NaN-in-window is now covered by neither the old nor the new suite. **Remediation (chosen: fail-loud):** a shared NaN guard in `window_pool_arrays` raises `ValueError` when a pooled value is NaN (WS1). Cross-ref C-32 (same numpy-port cluster), C-31.

> **Resolved (2026-07-20, re-review + fix).** A follow-up max-effort code-review found the WS1 guard was **incomplete**: `tail_pools` guarded only each entity's tail, but `MixtureBaseline`'s global pool consumes the *whole* train panel and silently dropped NaN there (`NaN > 0` is False) — empirically reproduced. Maintainer chose **fail-loud, whole-fit, everywhere**: the guard now also raises on any NaN in Mixture's global-pool source (`mixture.py` fit). Both tail and global paths fail loud; tested by `test_nan_in_training_window_fails_loud` (in-tail) + `test_mixture_nan_outside_window_fails_loud` (out-of-tail).

---

### C-34: predict/fit over-use the full input adapter — target-column precondition + wasted N×F lift

| Field | Value |
|-------|-------|
| ID | C-34 |
| Tier | 3 |
| Source | code-review (2026-07-20, PR #60 findings #1/#7/#8) |
| Trigger | When a caller passes `predict()` (or `ZeroModel.fit()`) a frame that omits a declared target column — a features-only or targets-not-yet-merged forecast frame; or when profiling a large-`pgm` predict. |
| Location | all 7 `views_baseline/model/models/**` predict/fit sites; `views_baseline/model/frames/input.py:_from_dataframe` |

Every `predict()` calls `to_feature_frame(df, targets)` even though it uses only `ff.index.level` (distributional) or `level` + predict-frame entities (point) — never the target **values**. Two consequences: (a) `_from_dataframe` reads `df[targets]` and raises `ValueError: missing required target column(s)` if a target column is absent — a **new precondition** the old index-only predict didn't have, which is worst for `ZeroModel` (whose entire purpose is data-independence, now data-dependent at both fit and predict — empirically verified to raise); (b) an N×F float64 block + N×F float32 block is built and discarded on every predict, ×7 models. Fails loud (not silent) and the standard VIEWS queryset carries the columns, so Tier 3 (contract widening + wasted work + test gap), not Tier 2. **Remediation:** a light `to_index(x,*,loa) -> (level, time, unit)` boundary that reads only the index; predict uses it; `ZeroModel.fit` stops requiring target columns (WS2). Cross-ref C-31 (the DIP boundary), C-07.

> **Resolved (2026-07-20, WS2 + re-review refinement).** `to_index` landed; point predict uses it, and the re-review added an even lighter `to_level(x,*,loa) -> SpatialLevel` for the 4 distributional predicts (they need only the level, not the identifier arrays `to_index` was building and discarding). Target-less predict is tested (`test_predict_does_not_require_target_columns`, now with a grid-shape assertion) and `ZeroModel` is data-independent again (`test_zero_model_is_data_independent`). Residual (accepted, C-34-adjacent): predict no longer errors on a *misnamed* target column — a silent-success where an error used to surface — the intended column-agnostic tradeoff.

---

### C-35: Input boundary missing guards — multi-sample frame truncated; float/NaN index corrupted

| Field | Value |
|-------|-------|
| ID | C-35 |
| Tier | 2 |
| Source | code-review (2026-07-20, PR #60 findings #4/#5) |
| Trigger | When a caller passes a directly-constructed `(N, F, S>1)` FeatureFrame as observed input (ADR-019 dual-input allows it); or a DataFrame whose time/entity index level is float-typed and contains NaN (common after a merge/reindex). |
| Location | `views_baseline/model/frames/input.py:_validate_feature_frame` (S), `_from_dataframe:100-101` (index cast), `panel:135` (S slice) |

Two silent boundary holes. (1) `_validate_feature_frame` checks loa↔level and target presence but **not** `sample_count`; `panel` then does `col[:, 0]`, so a multi-sample FeatureFrame is silently reduced to its first sample and every model fits/pools on sample 0 only — wrong result, no error. The df-lift path is always `S==1`, so only a directly-passed frame hits it. (2) `_from_dataframe` hard-casts both index levels with `.to_numpy(dtype=np.int64)` **before** views_frames' identifier validation runs, so a NaN in a float-typed index maps to a platform garbage id (e.g. `-9223372036854775808`) that flows into `entities_at`/`window_pool` and mis-keys the grid — bypassing the very NaN-rejection views_frames provides. Both are silent-corruption paths with low VIEWS likelihood (int ids, df path), hence Tier 2. **Remediation:** `S==1` guard in `_validate_feature_frame`; integer-dtype + NaN-free validation before the int64 cast in `_from_dataframe` (WS1).

> **Resolved (2026-07-20, re-review + fix).** The WS1 index guard was **partial** — `lvl.hasnans` caught NaN but not a fractional float (`2.9 -> 2`) or an out-of-int64-range value, which still silently truncated to a garbage id (the exact thing the guard's message claimed to prevent). Now `_index_arrays` requires an **exact integer cast** (`np.array_equal(raw.astype(int64), raw)`), rejecting NaN, non-integer, and overflow. Tested by `test_from_dataframe_rejects_nan_index` + `test_from_dataframe_rejects_fractional_index` + `test_rejects_multisample_featureframe`.

---

### C-36: PR-2 reorg guard tests under-verify their claims

| Field | Value |
|-------|-------|
| ID | C-36 |
| Tier | 3 |
| Source | code-review (2026-07-20, PR #60 findings #10/#11/#12) |
| Trigger | When a future change relies on these guards to catch a regression: a model reintroducing a float64 DataFrame fast-path; an in-`fit` `import pandas`; or a `catalog.py` registry refactored to a dict comprehension. |
| Location | `tests/test_dual_input.py:48`; `tests/test_falsification_reorg_principles.py:90` (DIP), `:116` (OCP) |

Three guards give false confidence. (1) The headline df≡FF equivalence test is near-tautological — both branches converge to the same float32 lift (`to_feature_frame`) and the dummy data is float32-exact integers, so a float64 df fast-path divergence would still compare equal. (2) The DIP AST guard scans only `ast.parse(...).body` (module top-level), so an in-function `import pandas` in a model — a genuine runtime DIP violation — is invisible and `offenders==[]` stays green. (3) The OCP strict-xfail measures the max size of any top-level dict **literal** in `catalog.py`; a still-hardcoded registry built via a dict comprehension or `dict(...)` drops the count to 0, flipping the strict-xfail to a false XPASS ("OCP resolved"). Maintainability/false-confidence, hence Tier 3. **Remediation:** non-float32-exact equivalence case; DIP guard walks all `Import` nodes; OCP check robust to comprehensions (WS3).

> **Re-review note (2026-07-20).** The WS3 float32-equivalence fix was itself **still tautological** — the re-review empirically proved the output `PredictionFrame` is always float32, so `out.values == np.float32(val)` holds regardless of the internal path. Fixed properly by asserting on the **fitted state** (`model.last_observations[cid][t]`, which carries the internal-path precision), not the output. The DIP-guard (now walks all `Import` nodes, excludes `frames/input.py`) and OCP-comprehension fixes stand; the OCP fix's residual limitation (a *legitimate* dict-comp OCP solution would also read as hardcoded, so the ratchet can't flip for that shape) is accepted since C-19 is deferred.

---

### C-37: pandas→numpy port residue (dead state, None-log, retained pool, double sort)

| Field | Value |
|-------|-------|
| ID | C-37 |
| Tier | 4 |
| Source | code-review (2026-07-20, PR #60 findings #9/#13/#14/#15) |
| Trigger | When a contributor extends a model and copies the dead `self.time_idx` assignment as if load-bearing, reads a fit log expecting the level, or profiles Mixture fit memory/CPU. |
| Location | all 7 `views_baseline/model/models/**`; `views_baseline/model/models/distributional/mixture.py:63`; `parametric.py:70`, `parametric_hurdle.py:74` |

Cleanup from the reorg: `self.time_idx` is assigned in every model's `fit()` but read nowhere; `self.entity_idx` only feeds a fit-log line that — because the reorg moved the assignment below the log — now always reads `on level: None`; `ParametricConflictology`/`Hurdle` retain `self.pools` on the fitted object though only `self.params` is read at predict (needlessly pins the entities×window array, including in pickled artifacts); `MixtureBaseline.fit` re-`lexsort`s and re-masks the same panel that `window_pool_arrays` already sorted for the local pool. No correctness impact — Tier 4. **Remediation:** WS4.

> **Resolved (2026-07-20, WS4 + re-review).** Dead `self.time_idx`/`self.entity_idx` deleted across all 7 models; fit logs now use `self.loa` (no more `on level: None`); parametric `self.pools` is a fit-local; `MixtureBaseline` derives local+global pools from one shared `sort_train_panel` (no re-sort). Re-review added: `tail_pools` builds+NaN-checks in one loop, the dead `block.size==0` guard removed, `_check_ff_level` dedups the loa↔level check across `to_index`/`to_feature_frame`, and a `train_test_boundary(partition_dict)` helper homes the `train_end = test_start-1` convention. Golden byte-identical throughout.

### C-06: `partition_dict` structure assumed but never validated — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-06 |
| Resolved | 2026-07-31 |
| Resolution | `train_test_boundary` (`model/grid.py`) now validates the partition-dict shape (a dict with a `'test'` key whose value is a non-empty `(start, end)` of ints) and raises `ValueError` at the model boundary; the 4 distributional `predict()`s were routed through it, closing the former Pattern-B inline `self.partition_dict['test'][0]` bypass so every fit/predict site is guarded once. Tests: `tests/test_grid.py`. |

---

### C-07: No enforcement that `targets` columns exist in input DataFrame — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-07 |
| Resolved | 2026-07-31 |
| Resolution | Target-presence is validated on the fit path of all 6 non-Zero models — `_from_dataframe` raises `missing required target column(s)` and `_validate_feature_frame` raises `missing required target feature(s)` (`model/frames/input.py`, C-34 work). Predict is deliberately column-agnostic (C-34) and `ZeroModel` deliberately data-independent, so the deep-`KeyError` path this entry described is closed with no remaining code gap. Covered by `tests/test_input.py` (`*_rejects_missing_target_*`). |

---

### C-21: Governance docs still describe the removed `isinstance` manager dispatch — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-21 |
| Resolved | 2026-07-31 |
| Resolution | Doc sweep completed. Active docs edited in place — `contributor_protocols/carbon_based_agents.md`, `hardened_protocol_template.md`, `CICs/ConflictologyModel.md`, `INSTANTIATION_CHECKLIST.md`; immutable ADR bodies given prepended Status notes per ADR-000 — `ADRs/009` (Boundary 3) and `ADRs/004`. The register's stale location list is corrected: `ADRs/005:51` (legitimate protocol-conformance text) and `CICs/BaselineForecastingModelManager.md:56` (already correct) were NOT drift. `validate_docs.sh` green; remaining `isinstance` mentions are historical/decision records or protocol-conformance statements. |

---

### C-30: Golden tests are coupled to the numpy Generator version (no numpy pin) — RESOLVED

| Field | Value |
|-------|-------|
| ID | C-30 |
| Resolved | 2026-07-31 |
| Resolution | `numpy>=1.26,<3` declared in `pyproject.toml` `[project] dependencies`, matching sibling house style (views-frames, views-datafactory) and making numpy a first-class direct dependency (imported across ~10 runtime modules). The golden-test/Generator-stream coupling note stays in `tests/test_golden.py`; the `<3` ceiling matches siblings and those streams are stream-stable (NEP-19). |

## Register Conventions

- **ID format:** `C-xx` for concerns, `D-xx` for disagreements. IDs are permanent — gaps in numbering indicate merged or resolved entries
- **Sources:** `repo-assimilation`, `expert-review`, `test-review`, `falsification-audit`, `clean-architecture-review`, `pr-review`, `tech-debt-audit`, `incident`
- **Resolution:** Move to "Resolved Concerns" with resolution date and summary when addressed
- **Header counts:** Manually maintained — update whenever a concern is added or resolved
- **Cross-repo:** This register tracks views-baseline-specific risks only. Entries may reference external code locations (e.g., pipeline-core ensemble managers) when the risk is about baseline's interaction with that code. The root-cause fix may belong in another repo, but the risk to baseline is tracked here. The parent risk for C-59 (filename coupling across engine repos) is tracked in the `views-pipeline-core` register.
