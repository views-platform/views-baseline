# ADR-020: Single views-frames Construction Seam and the Spatial-Level Contract

**Status:** Accepted
**Date:** 2026-06-24
**Deciders:** Simon, VIEWS platform team
**Consulted:** repo-assimilation + expert-code-review (risk register C-16–C-19, D-04–D-06)
**Informed:** views-models maintainers (pipeline-core 3.0.0 alignment)

---

## Context

The platform extracted its Frame types into a standalone **`views-frames`** package, and `views-pipeline-core` #188 (PR #206) retires its local `PredictionFrame` to re-export the `views_frames` leaf. The constructor changed:

- **old:** `PredictionFrame(y_pred, identifiers={"time": ..., "unit": ...})`
- **new:** `PredictionFrame(y_pred, index=SpatioTemporalIndex(time, unit, level))` — the spatial `level` (CM/PGM) is now a **required** part of every frame's identity, and there is no `identifiers=` keyword.

This is a breaking change destined for **pipeline-core 3.0.0**. views-baseline still calls the old signature at **three** sites — `model/helpers.py` (`build_prediction_frame`, point models), `model/baseline.py` (`ConflictologyModel.predict`, `MixtureBaseline.predict`) — each with its own lazy `PredictionFrame` import. Against a post-#188 pipeline-core the suite is **26/98 red** (risk register **C-16**).

This is the **second** time this class of break has occurred (C-01 in April was the first). The root cause is not a version number: model logic constructs the *concrete, volatile* platform leaf inline at multiple sites and owns no single construction seam — a Dependency-Inversion / Stable-Dependencies weakness. The narrow constructor swap is filed as **#21**; this ADR authorizes the *correct, bounded* fix instead.

A decision is needed **now** because the suite cannot be green under the pipeline-core release the platform is shipping, and because the change touches "sensitive areas" that, per `docs/contributor_protocols/silicon_based_agents.md`, require an explicit decision recorded in an ADR before implementation.

---

## Decision

**1. One construction seam.** All `PredictionFrame` construction is consolidated into a single function, `to_prediction_frames(...)`, in `model/helpers.py`. It is the *only* place in the repository that imports and constructs `views_frames` types. Its shape:

```python
def to_prediction_frames(
    y_pred_by_target: dict[str, np.ndarray],   # each (N, S): S=1 point, S=n_samples distributional
    time: np.ndarray,                          # (N,) int64
    unit: np.ndarray,                          # (N,) int64
    level: "SpatialLevel",
) -> dict[str, "PredictionFrame"]:
    from views_frames import PredictionFrame, SpatioTemporalIndex
    index = SpatioTemporalIndex(time=time.astype("int64"), unit=unit.astype("int64"), level=level)
    return {t: PredictionFrame(y, index) for t, y in y_pred_by_target.items()}
```

`build_prediction_frame` (point models) becomes a thin wrapper over it; `ConflictologyModel.predict` and `MixtureBaseline.predict` route their `(N, n_samples)` arrays through it instead of constructing inline. The dead `build_prediction_grid` (the pre-PredictionFrame DataFrame builder, "called by nothing" per ADR-018) is deleted.

**2. The spatial-level contract.** The required `level` is derived from the **declared** `loa` (`SpatialLevel(loa)`), and **validated** against the observed entity index — never inferred from the index as the sole source. Per **ADR-003**, the declaration is authoritative; the observed `entity_idx` is a cross-check that must *agree* (`country_id`⇔CM, `priogrid_*`⇔PGM), raising a descriptive error at the model boundary on mismatch or on a non-`(time, entity)` index.

**3. The lazy import is consolidated, not multiplied.** This amends ADR-002's lazy-import rule: there is now **one** function-scoped `views_frames` import site (`to_prediction_frames`), not three. `model/` remains free of module-level frame imports.

**In scope:** the seam, the level contract, the dependency declaration + pin + lockfile (see ADR references), the protocol return-type tightening to `dict[str, PredictionFrame]`, deletion of `build_prediction_grid`.

**Explicitly out of scope:** one-concept-per-file splitting of `baseline.py`, new packages, renaming `infrastructure/`, and **OCP consolidation of the three-registry model-addition path** (deferred per ADR-012 until the catalog exceeds ~10 models). The FeatureFrame *input* migration (ADR-017, ADR-019) and ensemble compatibility remain separate clusters.

---

## Rationale

- **Dependency Inversion / Stable Dependencies.** A library should depend on the volatile platform leaf through one boundary it controls, not at every call site. One seam means the *next* platform change to the frame constructor is a one-function edit, not a three-site hunt — directly preventing the C-01/C-16 recurrence and the half-migration risk (C-19) latent in #21's two-of-three site list.
- **Declarations over inference (ADR-003).** Writing `level` into the frame's identity makes a wrong level a *silent* mislabel consumed downstream as truth (C-18). Deriving from the declared `loa` and validating against the index keeps the authoritative source declarative while still catching loader drift loudly.
- **Bounded, not a refactor.** The maintainer's constraint (2026-06-24) is "done right, zero tech debt, but not a full repo refactor." Consolidating construction and deleting dead code removes debt without restructuring the package; the procedural sprawl elsewhere is left untouched.

---

## Considered Alternatives

### Alternative A: Patch each construction site (the #21 plan as written)
- **Pros:** Smallest diff; familiar.
- **Cons:** Leaves three sites and three imports; #21's location list names only two, risking a half-migrated `MixtureBaseline` (C-19); re-buys the same churn at the next platform change.
- **Reason for rejection:** Treats the symptom, not the DIP/SDP root cause. Revisit never — superseded by this ADR.

### Alternative B: Infer `level` from the entity index name only
- **Pros:** Local to the construction site; no `loa` threading.
- **Cons:** A renamed/reordered entity level silently stamps the wrong `SpatialLevel` (C-18); violates ADR-003 (inference as sole source).
- **Reason for rejection:** Silent output mislabel is the exact failure class ADR-003 exists to prevent.

### Alternative C: A dedicated boundary/adapter package (and one-class-per-file restructure)
- **Pros:** Maximal separation; "screaming architecture."
- **Cons:** Touches ADR-002 topology broadly, moves import paths, and exceeds the maintainer's bounded-scope decision.
- **Reason for rejection:** Out of scope now. Revisit if the model layer grows or the input (FeatureFrame) migration forces a boundary package anyway.

### Alternative D: Consolidate the three-registry model-addition path (OCP)
- **Reason for rejection / deferral:** ADR-012 deliberately deferred this until >10 models (currently 5). Honored; recorded as a needs-decision, no code.

---

## Consequences

### Positive
- One construction site; zero `identifiers=` calls remain. Closes C-16 (behavior), C-08 (duplicated lazy import), C-15 (dead builder), C-19 (scattered construction / half-migration).
- `level` is single-sourced and validated; C-18 closed, C-05 (positional-index assumption) mitigated by the boundary check.
- The next frame-constructor change is a one-function edit.

### Negative
- One more indirection inside `helpers.py` (`build_prediction_frame` → `to_prediction_frames`).
- The lazy-import pattern remains non-obvious (now in one place; ADR-002's tradeoff stands, reduced in surface).
- The level cross-check adds a boundary assertion the models did not previously make (intended — it is the C-18 guard).

---

## Implementation Notes

- **Seam:** `views_baseline/model/helpers.py` — add `to_prediction_frames`; make `build_prediction_frame` delegate; delete `build_prediction_grid`.
- **Level:** `resolve_level(loa, entity_idx) -> SpatialLevel` (one place); raise on `loa`⇔`entity_idx` mismatch or non-`(time, entity)` index. Lazy `views_frames.SpatialLevel` import.
- **Models:** route `ConflictologyModel.predict` and `MixtureBaseline.predict` through the seam; **preserve** entity→time→target RNG order (ADR-011), `distributional` markers, entity-drop warnings, `require_entities`. Edit `baseline.py` with targeted replacements (anti-truncation rule).
- **Protocol:** `model/protocol.py` `predict()` return → `dict[str, PredictionFrame]`.
- **Dependencies:** declare `views-frames`; bump `views-pipeline-core` to `>=3.0.0,<4.0.0`; commit `poetry.lock`; add a real-leaf canary test. (Story S4 / #26.)
- **Governance correction:** the now-stale `silicon_based_agents.md` (Forbidden #3 "point predict returns `pd.DataFrame`"; the `isinstance` dispatch sensitivity) and ADR-003 §2 / ADR-002's two-site lazy-import note are corrected to match the post-ADR-017 single uniform path and this ADR's single seam.
- **Verify:** whether one shared `SpatioTemporalIndex` object may back all per-target frames or must be per-frame (current code `.copy()`s identifier arrays per target).

Tracked as epic **#22**; stories **#23–#27**; checklist **#28**. Subsumes **#21**.

---

## Validation & Monitoring

- `grep -rn "identifiers=" views_baseline/` returns nothing; exactly one `views_frames` construction call exists.
- `conda run -n views_pipeline python -m pytest -q` and `poetry run pytest tests/ -v` green under pipeline-core 3.0.0, including `test_mixture_predict_reproducible` / `test_conflictology_predict_reproducible` (RNG order preserved).
- A real-leaf canary test asserts `y_pred` shape and the resulting `SpatialLevel`; it fails loudly if the leaf API shifts again — the early-warning this ADR exists to provide.
- A `loa`⇔`entity_idx` mismatch raises at the boundary (red-team test).

---

## Open Questions

- May a single `SpatioTemporalIndex` instance back all per-target frames, or must each frame own its own? Resolved during S3 implementation against the `views_frames` contract.
- When the FeatureFrame **input** migration (ADR-017 §input, ADR-019) lands, should a symmetric input seam be introduced — and at that point does a dedicated boundary package (Alternative C) become justified?
- Revisit OCP registry consolidation (Alternative D) when the catalog exceeds ~10 models (ADR-012).

---

## References

- Epic #22; stories #23–#27; tracking #28; subsumes #21.
- `views-pipeline-core` #188 (PR #206) — local `PredictionFrame` retired, re-exports `views_frames` leaf → pipeline-core 3.0.0.
- ADR-002 (topology / lazy-import rule — amended here), ADR-003 (declarations over inference — basis of the level contract), ADR-010 (PredictionFrame as universal output), ADR-011 (RNG determinism), ADR-012 (model addition / OCP deferral), ADR-013 (pipeline-core coupling), ADR-017 (frames replace dataframes), ADR-018 (`build_prediction_grid` documented dead).
- Risk register: C-05, C-08, C-15, C-16, C-17, C-18, C-19; disagreements D-04, D-05, D-06.
- `docs/contributor_protocols/silicon_based_agents.md` (corrected by this ADR).
