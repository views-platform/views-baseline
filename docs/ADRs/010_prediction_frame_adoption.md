# ADR-010: PredictionFrame as Universal Output Format

**Status:** Accepted (reconciled 2026-07-27; revised 2026-06-02; supersedes the original 2026-03-13 decision)
**Date:** 2026-06-02 (decision) · 2026-07-27 (text reconciled to shipped code — see Reconciliation note)
**Deciders:** Project maintainers

---

## Context

All **seven** baseline model classes return `dict[str, PredictionFrame]` from `predict()`:

- **Point models** (`ZeroModel`, `LocfModel`, `AverageModel`) — `y_pred` shape `(N, 1)`, a single deterministic value per cell.
- **Distributional models** (`ConflictologyModel`, `MixtureBaseline`, `ParametricConflictology`, `ParametricHurdleConflictology`) — `y_pred` shape `(N, n_samples)`.

Previously (ADR-010 original, 2026-03-13), only distributional models returned `PredictionFrame` while point models returned `pd.DataFrame`. That created a dual dispatch in the manager (`isinstance(model, DistributionalBaselineModel)`) and two code paths in pipeline-core. The migration to a universal `PredictionFrame` output was motivated by:

1. **12 of 21 deployed baseline models already used PredictionFrame** — all `MixtureBaseline` and `ConflictologyModel` configs in views-models had `prediction_format: "prediction_frame"`.
2. **The dual dispatch was fragile** — pipeline-core's external dispatch (config-based) and the baseline manager's internal dispatch (isinstance-based) aligned by coincidence, not by contract.
3. **Hydranet completed the same migration** (ADR-047) as a clean break with no dual-format period.

---

## Decision

**`PredictionFrame` is the canonical output format for ALL baseline models. No DataFrame output path exists.**

### What changed

1. All seven model classes return `dict[str, PredictionFrame]` from `predict()`.
2. Point models build their output via `build_prediction_frame()` in `model/frames/output.py` — the `(N, 1)` counterpart of the distributional `sample_prediction_grid()` in the same module.
3. Both helpers route through the single construction seam `to_prediction_frames()` (`model/frames/output.py`), the **one** place the `views_frames` leaf is constructed (ADR-020).
4. The `isinstance(model, DistributionalBaselineModel)` dispatch in the manager is **removed**. `_generate_predictions()` and `_forecast_model_artifact()` handle every model through one uniform path, because every model returns the same `dict[str, PredictionFrame]` type.
5. The `BaselineModel` protocol's `predict()` return type is `dict[str, PredictionFrame]`.
6. `prediction_format` is a required key in `ReproducibilityGate.Config.CORE_GENOME`.

### What stays the same

- The `DistributionalBaselineModel` protocol is retained for **semantic classification only** — it marks models carrying `distributional = True` (all four distributional classes set it; point models do not). It is **not** used for output dispatch.
- The internal structure of `PredictionFrame` / `SpatioTemporalIndex` is owned by `views_frames` (re-exported through views-pipeline-core).

---

## Rationale

- The old DataFrame-with-lists format was structurally ambiguous: a `pd.DataFrame` with list-valued cells cannot be distinguished from a malformed point forecast without inspecting cell contents. This violates the declarations-over-inference principle (ADR-003).
- `PredictionFrame` is a purpose-built container for forecast output. It carries explicit shape metadata (`y_pred.shape == (N, n_samples)`, with `n_samples == 1` a legitimate point case) and a `SpatioTemporalIndex` (`time`, `unit`, and the spatial `level`), making the structure self-describing.
- Lazy import preserves the testability constraint: the `views_frames` leaf is imported **inside** the two frame seams — `to_feature_frame()` (input, `model/frames/input.py`) and `to_prediction_frames()` (output, `model/frames/output.py`) — so `model/` stays importable without the platform leaf present (ADR-002/ADR-013).
- A single output format with no fallback path reduces code surface area and eliminates the risk of a consumer silently receiving the wrong format.

---

## Considered Alternatives

### Alternative A: Dual output (PredictionFrame + DataFrame fallback)

- **Pros:** Backwards compatible; consumers without the `views_frames` leaf get DataFrames.
- **Cons:** Two code paths to maintain and test; violates ADR-003 (would require inference to detect which format was produced).
- **Reason for rejection:** The old DataFrame path was dead code. Reviving it adds complexity with no current consumer.

### Alternative B: Keep point models on DataFrame

- **Pros:** Point forecasts avoid a `views_frames` dependency in their predict path.
- **Cons:** Non-uniform output type forces the fragile dual dispatch back into the manager and pipeline-core.
- **Reason for rejection:** Adopted the opposite — point models now emit `(N, 1)` PredictionFrames through the same seam. The `n_samples == 1` degenerate case is explicit and self-describing, and the lazy-import seam keeps `model/` testable without the leaf.

---

## Consequences

### Positive

- Single output format for all models — no ambiguity, no dispatch.
- The manager is uniform: `_generate_predictions()` / `_forecast_model_artifact()` consume `dict[str, PredictionFrame]` from every model with one code path.
- The lazy-import seam keeps `model/` independently testable.

### Negative

- `views_frames` (re-exported via views-pipeline-core) is a runtime dependency for every model's `predict()`. If the `PredictionFrame` API changes upstream, all models are affected — caught by the golden/characterization suite.
- Point and distributional models share a return type but differ in `y_pred` width (`1` vs `n_samples`); downstream consumers that concatenate frames must respect `sample_count` (see the ensemble `sample_count` guard, pipeline-core #160).

---

## Implementation Notes

The decision is fully implemented. Key locations:

- `views_baseline/model/frames/output.py` — `to_prediction_frames()` (single seam), `build_prediction_frame()` (point, `(N, 1)`), `sample_prediction_grid()` (distributional, `(N, n_samples)`).
- `views_baseline/model/models/point/{zero,locf,average}.py` — point models call `build_prediction_frame()`.
- `views_baseline/model/models/distributional/{conflictology,mixture,parametric,parametric_hurdle}.py` — distributional models fill via `sample_prediction_grid()`; each sets `distributional = True`.
- `views_baseline/manager/baseline_manager.py` — `_generate_predictions()` and `_forecast_model_artifact()` handle all models uniformly (no `isinstance` dispatch).
- `views_baseline/model/protocol.py` — `BaselineModel` (universal) and `DistributionalBaselineModel` (classification marker, `distributional: bool`).

No migration is required. This ADR codifies the existing state.

---

## Validation & Monitoring

- `tests/test_baseline.py` — point and distributional `predict()` return `dict[str, PredictionFrame]` with the correct keys and shapes (e.g. point `(N, 1)` assertions).
- `tests/test_golden.py` — pins exact distributional `y_pred` values (byte-identity characterization).
- `tests/test_baseline_manager.py` — manager evaluate/forecast/sweep return `dict[str, (list of) PredictionFrame]` for both point and distributional models through the uniform path.
- `tests/test_protocol.py` — `isinstance` checks confirm distributional models satisfy `DistributionalBaselineModel` (classification, not dispatch).
- Failure mode: if `views_frames` changes the `PredictionFrame` constructor signature, `to_prediction_frames()` raises at runtime — caught by the existing suite.

---

## Open Questions

- Should the `dict[str, PredictionFrame]` return type be formalized as a named type alias (e.g. `Prediction`) for documentation clarity?

*(The former open question — "should point models migrate to PredictionFrame?" — is resolved: they did, via `build_prediction_frame()`.)*

---

## Reconciliation note (2026-07-27)

The decision (universal `PredictionFrame`, no DataFrame path, no dispatch) has been correct since 2026-06-02, but the surrounding prose had drifted after the epic #47 reorg and epic #33 (parametric models): it said "5 models" (now 7), cited `helpers.py` / `baseline.py:270/403` paths the reorg deleted, and its Consequences/Implementation-Notes/Open-Questions still described the removed `isinstance` dispatch as present and the point-model migration as pending. This revision reconciles every section to the shipped code (issue #67, epic #66). No decision changed — this is a text-only reconciliation.

---

## References

- **ADR-017: Frames Replace DataFrames as Operational Containers** — the broader platform direction (universal `PredictionFrame` output incl. `n_samples == 1`; `FeatureFrame` as the input half). Read alongside this ADR.
- **ADR-020: Single `views_frames` construction seam** — `to_prediction_frames()`.
- ADR-002 / ADR-013: Topology and dependency rules (lazy-import mandate).
- ADR-003: Authority of Declarations over Inference.
- `views_baseline/model/frames/output.py` — construction seam and builders.
- `views_baseline/model/protocol.py` — `BaselineModel`, `DistributionalBaselineModel`.
- views-pipeline-core ADR-033: PredictionFrame as canonical distributional output.
