# ADR-010: PredictionFrame as Universal Output Format

**Status:** Accepted (revised 2026-06-02; supersedes the original 2026-03-13 decision)
**Date:** 2026-06-02
**Deciders:** Project maintainers

---

## Context

As of 2026-06-02, all 5 baseline model classes return `dict[str, PredictionFrame]` from `predict()`. Point models produce `y_pred` with shape `(N, 1)` (single deterministic value); distributional models produce `(N, n_samples)`.

Previously (ADR-010 original, 2026-03-13), only distributional models returned PredictionFrame while point models returned `pd.DataFrame`. This created a dual dispatch in the manager (`isinstance(model, DistributionalBaselineModel)`) and two code paths in pipeline-core. The migration was motivated by:

1. **12 of 21 deployed baseline models already used PredictionFrame** — all MixtureBaseline and ConflictologyModel configs in views-models had `prediction_format: "prediction_frame"`.
2. **The dual dispatch was fragile** — pipeline-core's external dispatch (config-based) and the baseline manager's internal dispatch (isinstance-based) aligned by coincidence, not by contract.
3. **Hydranet completed the same migration** (ADR-047) as a clean break with no dual-format period.

---

## Decision

**`PredictionFrame` is the canonical output format for ALL baseline models. No DataFrame output path exists.**

### What changed

1. All 5 model classes return `dict[str, PredictionFrame]` from `predict()`.
2. Point models use `build_prediction_frame()` in `model/helpers.py` — a new helper that mirrors `build_prediction_grid()` but outputs PredictionFrame with `y_pred` shape `(N, 1)`.
3. The `isinstance(model, DistributionalBaselineModel)` dispatch in the manager is removed. One code path for all models.
4. The `BaselineModel` protocol's `predict()` return type is `dict` (was `pd.DataFrame`).
5. `prediction_format` is added to `ReproducibilityGate.Config.CORE_GENOME`.

### What stays the same

- `DistributionalBaselineModel` protocol is retained for semantic classification (marks models with `distributional = True`), but is no longer used for dispatch.
- `build_prediction_grid()` is retained in `helpers.py` for backward compatibility but is no longer called by any model class.
- The internal structure of `PredictionFrame` is owned by views-pipeline-core.

---

## Rationale

- The old DataFrame-with-lists format was structurally ambiguous: a `pd.DataFrame` with list-valued cells cannot be distinguished from a malformed point forecast without inspecting cell contents. This violates the declarations-over-inference principle (ADR-003).
- `PredictionFrame` is a purpose-built container for distributional predictions. It carries explicit shape metadata (`y_pred.shape == (N, n_samples)`) and named identifiers, making the structure self-describing.
- Lazy import preserves the testability constraint: `model/` tests can run without views-pipeline-core by mocking or skipping the import. The lazy import is isolated to exactly 2 call sites (`ConflictologyModel.predict()` line 270, `MixtureBaseline.predict()` line 403 in `baseline.py`).
- No fallback path reduces code surface area and eliminates the risk of a consumer silently receiving the wrong format.

---

## Considered Alternatives

### Alternative A: Dual output (PredictionFrame + DataFrame fallback)

- **Pros:** Backwards compatible; consumers without views-pipeline-core get DataFrames.
- **Cons:** Two code paths to maintain and test; violates ADR-003 (would require inference to detect which format was produced).
- **Reason for rejection:** The old DataFrame path was dead code. Reviving it adds complexity with no current consumer.

### Alternative B: Migrate point models to PredictionFrame too

- **Pros:** Uniform output type across all models.
- **Cons:** `PredictionFrame` is designed for distributional outputs (samples dimension); point forecasts are a degenerate case (`n_samples=1`). Would force views-pipeline-core dependency into the point-model predict path.
- **Reason for rejection:** Not rejected permanently — deferred as an open question. Currently not needed.

---

## Consequences

### Positive

- Single output format for distributional models — no ambiguity.
- Manager dispatch is clean: `isinstance` check against the protocol determines routing.
- Lazy import pattern keeps `model/` independently testable.

### Negative

- views-pipeline-core is a runtime dependency for distributional `predict()`. If the `PredictionFrame` API changes upstream, both distributional models break.
- Point and distributional models have different return types, requiring the manager to handle two dispatch branches.

---

## Implementation Notes

The decision is already implemented. Key locations:

- `views_baseline/model/baseline.py` — `ConflictologyModel.predict()` and `MixtureBaseline.predict()` both lazy-import `PredictionFrame` and return `dict[str, PredictionFrame]`.
- `views_baseline/manager/baseline_manager.py` — `_generate_predictions()` and `_forecast_model_artifact()` dispatch on `isinstance(model, DistributionalBaselineModel)`.
- `views_baseline/model/protocol.py` — `DistributionalBaselineModel` protocol requires `distributional: bool` attribute.

No migration is required. This ADR codifies the existing state.

---

## Validation & Monitoring

- `tests/test_baseline.py` — distributional model tests verify `predict()` returns `dict[str, PredictionFrame]` with correct keys and shapes.
- `tests/test_baseline_manager.py` — manager tests verify correct dispatch for both point and distributional models.
- `tests/test_protocol.py` — `isinstance` checks confirm distributional models satisfy the protocol.
- Failure mode: if views-pipeline-core changes the `PredictionFrame` constructor signature, distributional `predict()` calls will raise `TypeError` at runtime. This is caught by the existing test suite.

---

## Open Questions

- Should point forecast models also migrate to `PredictionFrame`? This would unify the output interface but introduces a views-pipeline-core dependency into the point-model path. A separate ADR should be written if this is pursued.
- Should the `dict[str, PredictionFrame]` return type be formalized as a type alias (e.g., `DistributionalPrediction`) for documentation clarity?

---

## References

- **ADR-017: Frames Replace DataFrames as Operational Containers** — the broader platform direction this decision is part of (universal PredictionFrame output incl. `n_samples == 1`; FeatureFrame as the input half). Read alongside this ADR.
- ADR-002: Topology and Dependency Rules (lazy import mandate)
- ADR-003: Authority of Declarations over Inference (protocol dispatch)
- `views_baseline/model/baseline.py` — `ConflictologyModel`, `MixtureBaseline`
- `views_baseline/model/protocol.py` — `DistributionalBaselineModel`
- `views_baseline/manager/baseline_manager.py` — dispatch logic
- views-pipeline-core ADR-033: PredictionFrame as canonical distributional output
