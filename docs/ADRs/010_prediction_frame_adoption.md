# ADR-010: PredictionFrame Adoption and DataFrame Deprecation

**Status:** Accepted
**Date:** 2026-03-13
**Deciders:** Project maintainers

---

## Context

The views-baseline package produces two structurally different output types depending on the model category:

- **Point forecast models** (ZeroModel, LocfModel, AverageModel) return a `pd.DataFrame` indexed by `(time, entity)` with columns `pred_{target}`. These are built by `build_prediction_grid` in `model/helpers.py`.
- **Distributional models** (ConflictologyModel, MixtureBaseline) return `dict[str, PredictionFrame]` — one `PredictionFrame` per target, each containing a `y_pred` array of shape `(N, n_samples)` and an `identifiers` dict with `time` and `unit` arrays.

Historically, the distributional models returned `pd.DataFrame` with list-valued cells (each cell containing a list of `n_samples` draws). This was removed as dead code following the adoption of `PredictionFrame` upstream in views-pipeline-core (ADR-033 in that project).

The current state works but the decision has not been formally recorded: is `PredictionFrame` the permanent canonical output for distributional models? Should point models also migrate?

---

## Decision

**`PredictionFrame` is the canonical output format for distributional models. No DataFrame fallback is provided or planned.**

### In scope

1. `ConflictologyModel.predict()` returns `dict[str, PredictionFrame]` — one PF per target.
2. `MixtureBaseline.predict()` returns `dict[str, PredictionFrame]` — same structure.
3. `PredictionFrame` is lazy-imported inside `predict()` methods to keep `model/` testable without views-pipeline-core installed (ADR-002 topology constraint).
4. The manager dispatches on `isinstance(model, DistributionalBaselineModel)` to handle the two output types (ADR-003 declaration principle).

### Out of scope

- Point forecast models (ZeroModel, LocfModel, AverageModel) continue to return `pd.DataFrame` via `build_prediction_grid`. Migration to `PredictionFrame` is not part of this decision.
- The internal structure of `PredictionFrame` (column names, metadata) is owned by views-pipeline-core, not this project.

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

- ADR-002: Topology and Dependency Rules (lazy import mandate)
- ADR-003: Authority of Declarations over Inference (protocol dispatch)
- `views_baseline/model/baseline.py` — `ConflictologyModel`, `MixtureBaseline`
- `views_baseline/model/protocol.py` — `DistributionalBaselineModel`
- `views_baseline/manager/baseline_manager.py` — dispatch logic
- views-pipeline-core ADR-033: PredictionFrame as canonical distributional output
