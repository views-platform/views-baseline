# ADR 004: Rules for Evaluation and Stability

**Status:** Accepted
**Date:** 2026-03-13
**Deciders:** Project maintainers

---

## Context

views-baseline contains components at different stages of maturity and with different downstream audiences. All models now emit `dict[str, PredictionFrame]` (ADR-017) — point models a single-sample `(N,1)`, distributional models `(N, n_samples)`. Both output types changed from `pd.DataFrame` during development. The manager is coupled to views-pipeline-core, which evolves independently.

Without explicit stability classifications and rules for when those classifications trigger action, contributors have no shared framework for deciding whether a proposed change is safe, requires a migration path, or requires an ADR update. This ADR establishes that framework.

It also records the current state of test coverage, because the test suite is the primary mechanism for detecting stability regressions, and its gaps are consequential.

---

## Decision

### Stability Classifications

Each component carries one of two stability labels:

- **Stable** — The public interface (constructor signature, `fit()` inputs, `predict()` outputs) is a contract. Changes to a stable component's interface require a migration path and a new ADR. Internal implementation changes (e.g., algorithmic improvements that do not change output shape or type) are permitted without an ADR.
- **Evolving** — The component's interface may change between releases. Changes should be documented in commit messages and may require coordinated updates to dependent components, but do not require a new ADR unless they alter the topology or ontology.

### Component Stability Map

| Component | Stability | Rationale |
|---|---|---|
| `ZeroModel` | Stable | Output is `dict[str, PredictionFrame]`, `y_pred` shape `(N, 1)` (ADR-017); consumed by downstream ensembles |
| `LocfModel` | Stable | Same as above |
| `AverageModel` | Stable | Same as above; `window_months` is a declared config parameter |
| `ConflictologyModel` | Evolving | Output `dict[str, PredictionFrame]`, `(N, n_samples)`; `PredictionFrame` schema is upstream-owned |
| `MixtureBaseline` | Evolving | Newer model; `lambda_mix` and `global_pool` semantics are still being evaluated |
| `BaselineModel` (protocol) | Stable | Universal contract — all models return `dict[str, PredictionFrame]`; changes ripple to all implementing classes |
| `DistributionalBaselineModel` (protocol) | Stable | Semantic marker for sampled models (`distributional = True`); no longer drives manager dispatch (ADR-017) |
| `BaselineModelCatalog` | Stable | `get_model()` and `list_models()` are consumed by the manager |
| `build_prediction_frame` | Stable | Active output builder; called by the three point model classes (`build_prediction_grid` retained but unused) |
| `BaselineForecastingModelManager` | Evolving | Tightly coupled to `views-pipeline-core`; any breaking change upstream propagates here |

### Reproducibility Contract

The RNG seeding arithmetic used by `ConflictologyModel` and `MixtureBaseline` is a reproducibility contract:

```python
rng = np.random.default_rng(self.seed)
```

Both models initialise the RNG from `self.seed` at the start of each `predict()` call, and generate samples in a fixed order: entity → time step → target. This means that for a given `(seed, data, sequence_number)` triple, the output is deterministic. The test `test_mixture_predict_reproducible` validates this.

This contract has a subtlety: the current implementation does not incorporate `sequence_number` into the seed. Two calls with different `sequence_number` values but the same `seed` will produce identical sample *distributions* (the RNG is re-initialised from `self.seed` each time). Whether this is intentional or a gap is recorded in Open Questions.

The seed arithmetic is considered part of the stable interface for these models in the sense that changing it would alter the reproducibility of existing results. Any change to seeding behaviour must be documented as a breaking change.

### Trigger Conditions for ADR Updates

An ADR update (or a new ADR) is required when:

1. **views-pipeline-core API changes** — any change to `ForecastingModelManager`, `ModelPathManager`, `read_dataframe`, `generate_model_file_name`, `_get_cached_data_path()`, or `PredictionFrame` that requires a code change in `manager/baseline_manager.py` or in the distributional `predict()` methods.
2. **New model class additions** — a new class in `baseline.py` requires updating `MODEL_GENOMES` in `catalog.py`, the stability map in this ADR, and the ontology in ADR 001. If the new class introduces a new output type, it also requires updating ADR 003.
3. **PredictionFrame schema changes** — changes to the `y_pred` shape convention, `identifiers` key names, or constructor signature in `views-pipeline-core.data.prediction_frame` affect both distributional models and any downstream consumer.
4. **Protocol changes** — adding or removing attributes or methods from `BaselineModel` or `DistributionalBaselineModel`.
5. **Stability reclassification** — promoting an evolving component to stable, or demoting a stable component to evolving.

---

## Rationale

### Why point models are stable

`ZeroModel`, `LocfModel`, and `AverageModel` are the simplest possible baselines. They serve as reference points in evaluation: a model that cannot outperform LOCF has a fundamental problem. Downstream ensemble managers may call these models directly or use their outputs as features. Changing their output format — even in a seemingly minor way, such as changing column naming from `pred_y` to `y_pred` — would silently break ensemble consumers that do not validate their inputs.

### Why distributional models are evolving

`ConflictologyModel` and `MixtureBaseline` were added later and their integration with the rest of the VIEWS pipeline is still being worked out. The output format changed from `DataFrame` to `dict[str, PredictionFrame]` during development (visible in the git history). `PredictionFrame` itself is defined upstream and its schema is not owned by this repository. Marking these as evolving reflects the reality of their situation rather than aspirational stability.

### Why the manager is evolving

The manager does not contain domain logic; it translates between the pipeline infrastructure and the model layer. Its stability is therefore downstream of `views-pipeline-core`'s stability, which this project does not control. Since ADR-017, `_generate_predictions` has a single path: every model returns `dict[str, PredictionFrame]`, accumulated into `dict[str, list[PredictionFrame]]` — no `isinstance` dispatch. The manager's sensitivity is now purely to `views-pipeline-core` changes in how it consumes `dict[str, list[PredictionFrame]]` (and the `prediction_format` routing).

### Why the seed arithmetic matters

Reproducibility is a first-class property of forecasting baselines. If a model produces different samples on two runs with identical inputs, it is not a reliable reference point for evaluation. The `default_rng(seed)` pattern from NumPy's `Generator` API provides this guarantee. It is documented here so that future contributors do not "simplify" the seeding (e.g., by using `np.random.seed()` which is global state, or by not resetting the RNG on each `predict()` call).

---

## Considered Alternatives

**No formal stability classification; rely on semantic versioning.** The project does not currently have a release process with formal version numbers. Stability classification provides the same signal without requiring a release infrastructure.

**Mark all components as evolving.** Honest for the distributional models but undersells the stability obligations on the point models, which have been in use and tested for longer and have downstream consumers.

**Mark all components as stable.** Would be aspirational rather than descriptive for the distributional models and the manager. Misleading stability labels are worse than no labels.

**Use deprecation warnings rather than ADR trigger conditions.** Appropriate for library APIs with external users. For a project of this size with internal consumers, the ADR trigger list is a lighter-weight mechanism.

---

## Consequences

**Positive:**
- Contributors have a checklist for when an ADR update is required.
- The stability map makes explicit that changes to `ZeroModel`, `LocfModel`, and `AverageModel` have downstream consequences that changes to distributional models do not (yet) have at the same level.
- The reproducibility contract is documented so that future refactors do not inadvertently break it.

**Negative:**
- The stability map must be kept in sync with the code and with ADR 001. If a new model is added without updating this ADR, the map becomes stale.
- "Evolving" is not a promise of instability; it is a signal that changes are more likely. It may set the wrong expectation for consumers of the distributional models.

---

## Implementation Notes

### Current test coverage state (as of 2026-03-13)

The test suite contains 51 tests across 4 files:

| File | Tests | Coverage |
|---|---|---|
| `tests/test_baseline.py` | Point models (ZeroModel, LocfModel, AverageModel), distributional models (ConflictologyModel, MixtureBaseline), build_prediction_grid | Comprehensive for fit/predict contracts; no edge case tests for degenerate inputs |
| `tests/test_catalog.py` | `BaselineModelCatalog`: get_model, list_models, missing-key validation, unknown model | Comprehensive for catalog validation |
| `tests/test_protocol.py` | Protocol conformance: which classes satisfy BaselineModel and DistributionalBaselineModel | Comprehensive for protocol checks |
| `tests/test_baseline_manager.py` | Manager: evaluate (ZeroModel, LocfModel), forecast (LocfModel, ZeroModel), setup, train/artifact | Point model paths only |

**Known gap: manager distributional path is untested.**

`_evaluate_model_artifact` and `_forecast_model_artifact` are tested only with `ZeroModel` and `LocfModel`. The `isinstance(model, DistributionalBaselineModel)` branch in `_generate_predictions` has no test. If the distributional dispatch is broken (e.g., by a protocol change), the test suite will not catch it.

This gap is accepted as a known risk. Closing it requires either a real `views-pipeline-core` `PredictionFrame` in the test environment (already available, since distributional model tests use it), or a mock. The test infrastructure in `test_baseline_manager.py` already demonstrates how to mock `read_dataframe` and bypass `__init__`; the pattern is reusable for a distributional manager test.

### Reproduciblity test

`tests/test_baseline.py::test_mixture_predict_reproducible` validates that two `MixtureBaseline` instances with the same `seed` produce identical `y_pred` arrays:

```python
np.testing.assert_array_equal(r1[target].y_pred, r2[target].y_pred)
```

No equivalent test exists for `ConflictologyModel`. This is a gap.

---

## Validation & Monitoring

- The trigger conditions list above defines when this ADR must be revisited.
- The stability map should be reviewed when any of the following occur: a views-pipeline-core release, a new model PR, or a PredictionFrame schema change PR.
- The manager distributional path gap should be tracked as a known issue until a test is written.
- The reproducibility contract should be validated by a test for both distributional models (currently only `MixtureBaseline` has this test).

---

## Open Questions

- **Sequence number and RNG.** The current seed arithmetic is `np.random.default_rng(self.seed)` with no incorporation of `sequence_number`. This means predictions for different sequence numbers (different forecast windows) draw from identically-seeded RNG instances. Is this intentional (each forecast window is independently reproducible from the same seed) or a gap (different sequence numbers should produce independent samples)? The answer affects whether the reproducibility contract needs to include `sequence_number`.
- **When should distributional models be promoted to stable?** A reasonable criterion: once the `PredictionFrame` schema is declared stable upstream and at least one ensemble consumer has been validated against distributional baseline output.
- **Should the untested manager distributional path block production use?** Currently it does not. The distributional models themselves are tested; only the manager routing is not. This is a judgement call that depends on how the distributional models are invoked in practice.

---

## References

- ADR 001: Ontology of the Repository (stability annotations)
- ADR 002: Topology and Dependency Rules (manager coupling)
- ADR 003: Authority of Declarations over Inference (entity-drop warnings, degenerate input gap)
- `views_baseline/model/baseline.py` — RNG seeding in `ConflictologyModel.predict()` and `MixtureBaseline.predict()`
- `tests/test_baseline.py` — `test_mixture_predict_reproducible`
- `tests/test_baseline_manager.py` — manager test coverage and the distributional gap
- `views-pipeline-core` — upstream source of `PredictionFrame`, `ForecastingModelManager`, and pipeline conventions
