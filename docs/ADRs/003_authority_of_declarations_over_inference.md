# ADR 003: Authority of Declarations over Inference

**Status:** Accepted
**Date:** 2026-03-13
**Deciders:** Project maintainers

---

## Context

Forecasting code contains many quantities that could, in principle, be derived from data at runtime: the train/test boundary could be inferred from the index range, the model type could be guessed from the shape of its output, the target columns could be detected by scanning the DataFrame. Each of these inferences feels convenient at the point of writing but creates a latent fragility: the code silently produces wrong results when the data or the model behave unexpectedly, and the bug is often not caught until prediction time — or later.

The alternative is to require that every structurally significant quantity be declared explicitly and validated against that declaration rather than inferred from ambient state. This ADR records that principle, its scope in views-baseline, and the degree to which it is currently implemented.

---

## Decision

**Declarations are authoritative. Inference from data or output shape is forbidden for the quantities listed below.**

### Declared Quantities and Where They Live

#### 1. Config Genomes — required keys per model

`MODEL_GENOMES` in `BaselineModelCatalog` is the authoritative declaration of which config keys each model requires:

```python
MODEL_GENOMES = {
    "ZeroModel": [],
    "LocfModel": [],
    "AverageModel": ["window_months"],
    "ConflictologyModel": ["window_months"],
    "MixtureBaseline": [],
}
```

`get_model(name)` validates against this list before constructing the model:

```python
missing = [k for k in self.MODEL_GENOMES[model_name] if k not in self.config]
if missing:
    raise ValueError(
        f"Model '{model_name}' requires config keys {missing} but they are missing"
    )
```

The catalog never infers what a model needs by inspecting the model's constructor signature or by attempting construction and catching `KeyError`.

#### 2. Protocol Dispatch — `distributional` class attribute

> **Status note (ADR-017, ADR-020):** Since PR #15 every baseline model returns `dict[str, PredictionFrame]`
> and the manager has a **single, type-uniform path** — it no longer dispatches on
> `isinstance(model, DistributionalBaselineModel)`. The `distributional` attribute is retained as a
> **semantic marker** (declaration of intent per ADR-012), which keeps this section's "declare, don't
> infer" principle intact even though the runtime dispatch it describes has been removed. ADR-020 adds
> a second declared quantity to this list: the spatial **`level`**, derived from the declared `loa`
> and validated against the observed `entity_idx` (never inferred as the sole source).

Whether a model returns a `dict[str, PredictionFrame]` or a `pd.DataFrame` is declared by the model class via a class attribute:

```python
class ConflictologyModel:
    distributional = True

class MixtureBaseline:
    distributional = True
```

The `DistributionalBaselineModel` protocol requires this attribute. The manager dispatches on `isinstance(model, DistributionalBaselineModel)`, which checks for the presence of `distributional` at runtime via the `@runtime_checkable` protocol — not by inspecting the return type of `predict()` or calling `predict()` and checking what comes back.

Point forecast models (`ZeroModel`, `LocfModel`, `AverageModel`) do not carry a `distributional` attribute. They are not instances of `DistributionalBaselineModel`. The manager's `isinstance` check resolves this correctly without any conditional on output shape.

#### 3. Train/Test Boundary — `partition_dict["test"][0]`

The boundary between training data and the forecast horizon is declared in `partition_dict` and passed explicitly to every model at construction time:

```python
ZeroModel(targets=targets, partition_dict=partition_dict, loa=loa)
```

Every model reads `self.partition_dict["test"][0]` to find `test_start`. `fit()` methods slice training data as `df[... < test_start]`. `predict()` methods compute `prediction_start = test_start + sequence_number`.

No model infers the boundary from the data (e.g., by taking `df.index.get_level_values(time_idx).max()` and treating it as the last training step). This is deliberate: in a live pipeline, the full DataFrame passed to `fit()` may include data past the declared train end, and inferring from the index would silently use the wrong boundary.

#### 4. Target Columns — explicit `targets` list

Every model is constructed with an explicit `targets: List[str]` parameter. Models never scan the DataFrame for numeric columns or look for columns matching a pattern. Output columns are named `pred_{target}` for each declared target, constructed explicitly in `build_prediction_grid` via:

```python
row.update({f"pred_{target}": value_fn(cid, target) for target in targets})
```

### Forbidden Inferences

The following patterns are explicitly disallowed:

| Forbidden pattern | Why |
|---|---|
| Inferring model type from output shape (e.g., `if isinstance(result, dict)` without a protocol check) | Fragile: a point model could theoretically return a dict |
| Guessing target columns by scanning DataFrame dtypes or column names | Silent failure if column names change |
| Inferring `window_size` from `len(df)` or from the index range | Would silently change behaviour when data length changes |
| Inferring train/test boundary from `df.index.max()` | Wrong when the full dataset extends past the declared train end |
| Calling `predict()` and inspecting the return type to determine dispatch | Inference from output, not from declaration |

---

## Rationale

Baseline forecasting models sit at the boundary between research code (which often embraces flexibility and inference) and production pipelines (which require predictability). Once a baseline is in the pipeline:

- The data passed to `fit()` and `predict()` may change without the model knowing.
- The model may be called in a context where the caller assumes a specific output shape.
- Bugs that produce plausible-but-wrong outputs (e.g., LOCF carrying forward the wrong observation because the boundary was inferred incorrectly) can pass undetected through automated evaluation metrics.

Explicit declaration + validation at construction or call time surfaces these problems early and loudly, rather than silently at prediction time.

The `MODEL_GENOMES` validation is an example of this principle applied to configuration: rather than letting a `KeyError` bubble up from inside `_get_average_model()` when `config["window_months"]` is absent, the catalog checks at `get_model()` time and raises a descriptive `ValueError`. The error message names the missing key, which is far more actionable than a generic `KeyError`.

---

## Considered Alternatives

**Inspect constructor signatures to derive required keys.** Python's `inspect.signature` could extract required parameters from `AverageModel.__init__`. Rejected because it couples `MODEL_GENOMES` to implementation details of the constructors; a refactor that adds a default value to `months` would silently change the validation behaviour.

**Use type annotations to enforce config shape (e.g., TypedDict).** Would provide IDE-level validation. Rejected as over-engineered for the current codebase; config dicts are assembled by the pipeline infrastructure and type-checking them at the pipeline boundary is out of scope for this library.

**Use duck typing in the manager: call `predict()` and check what it returns.** Would work mechanically. Rejected because it runs the model to determine its type, which is expensive and side-effectful. Protocol dispatch is free.

**Check `hasattr(model, 'distributional')` instead of `isinstance(model, DistributionalBaselineModel)`.** Functionally equivalent for the current models. Rejected because `isinstance` against a `@runtime_checkable` protocol checks all required attributes simultaneously and is more readable as an intent declaration.

---

## Consequences

**Positive:**
- Missing config keys are caught at model construction time, not inside `fit()` or `predict()`.
- Manager dispatch is correct by construction: adding a new distributional model that sets `distributional = True` is all that is required for the manager to route it correctly.
- Models are not coupled to the shape of the data they receive (beyond requiring `MultiIndex(time, entity)`).
- The train/test boundary is a single source of truth per model instance; it cannot drift between `fit()` and `predict()` calls.

**Negative:**
- The principle is partially implemented. Two known gaps exist:
  1. **Entity-drop warnings, not errors.** When `LocfModel`, `AverageModel`, or the distributional models drop entities at predict time (because those entities are absent from the fitted state), they issue a `logger.warning` but continue. The declaration principle would call for raising an error or, at minimum, making the dropped-entity behaviour explicit in the protocol. This is a known limitation, not a resolved decision.
  2. **No input guards for degenerate parameters.** `window_months=0`, `n_samples=0`, and `lambda_mix > 1.0` are not validated at construction time. They would produce silent incorrect results or numpy errors at `fit()`/`predict()` time.
- `MODEL_GENOMES` and the actual constructor parameters can drift. If a new required parameter is added to `AverageModel.__init__` but not to `MODEL_GENOMES["AverageModel"]`, the validation will miss it. This is a maintenance obligation.

---

## Implementation Notes

The three sites where this principle is most concretely enacted:

**Config genome validation** (`model/catalog.py`, `get_model()`)

**Protocol discriminator attribute** (`model/baseline.py`, `ConflictologyModel` and `MixtureBaseline` class bodies)

**Partition dict reading** (`model/baseline.py`, every `fit()` and `predict()` method):
```python
test_start = self.partition_dict["test"][0]
```

**Entity-drop warning (current implementation)** (`model/helpers.py`, `filter_entities()`):
```python
if len(filtered) < n_before:
    logger.warning(
        f"{model_name}: {n_before - len(filtered)} entities dropped"
    )
```

This warning is emitted by the shared `filter_entities` helper, called from `LocfModel.predict()`, `AverageModel.predict()`, `ConflictologyModel.predict()`, and `MixtureBaseline.predict()`. It is consistent but does not raise. Whether it should raise is an open question (see below).

---

## Validation & Monitoring

- `test_catalog.py::test_catalog_missing_required_key_raises` validates that `AverageModel` raises `ValueError` when `months` is absent from config.
- `test_catalog.py::test_catalog_unknown_model_raises` validates that unknown model names raise `ValueError`.
- Protocol dispatch is tested implicitly through the manager tests: `_evaluate_model_artifact` and `_forecast_model_artifact` are tested for point models only. The distributional dispatch path in the manager is untested (see ADR 004).
- No tests currently cover the entity-drop warning path. A test that fits on a subset of entities and predicts on a larger set would catch regressions in this behaviour.

---

## Open Questions

- **Entity-drop warnings: accepted as-is.** Entity drops at predict time (entities absent from fitted state) warn via `logger.warning` but do not raise. This is the correct behaviour: entity drops are a normal operational condition (some entities may lack sufficient history), not an invariant violation. Raising would break the pipeline during routine operation. The warnings provide observability consistent with ADR-008. No change is planned.
- **DataFrame schema validation: accepted as-is.** There is no explicit check that the input DataFrame has a 2-level MultiIndex or that target columns exist. This is acceptable because pandas will raise `KeyError` or `IndexError` naturally on malformed input. Adding explicit guards would be defensive boilerplate with no practical benefit — the failure modes are already loud.
- Should degenerate parameter values (`window_months <= 0`, `n_samples <= 0`, `lambda_mix < 0 or > 1`) be validated in `__init__` with a `ValueError`? This is straightforward to implement and would close a known gap.
- Should `MODEL_GENOMES` validation be extended to check value types (e.g., `months` must be a positive integer), or is key presence sufficient?

---

## References

- ADR 001: Ontology of the Repository
- ADR 002: Topology and Dependency Rules
- `views_baseline/model/catalog.py` — `MODEL_GENOMES` and `get_model()`
- `views_baseline/model/baseline.py` — `distributional` attribute, `partition_dict["test"][0]` usage, entity-drop warnings
- `views_baseline/model/protocol.py` — `DistributionalBaselineModel` protocol
- `views_baseline/manager/baseline_manager.py` — `isinstance(model, DistributionalBaselineModel)` dispatch
- `tests/test_catalog.py` — config validation tests
