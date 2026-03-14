# ADR 001: Ontology of the Repository

**Status:** Accepted
**Date:** 2026-03-13
**Deciders:** Project maintainers

---

## Context

views-baseline contains five model classes, two protocols, one factory, one manager, and one helper module. Although that is a small surface area, the entities serve qualitatively different purposes and have different authority relationships, stability expectations, and dependency rules. Without a shared vocabulary for these categories, contributors must infer the ontology from file layout alone — which produces inconsistent mental models and leads to decisions (where to add a new model, what may depend on what) being made by analogy rather than by principle.

A formal ontology provides the vocabulary. It does not constrain what the code can do; it names what the code already is.

---

## Decision

The following six ontological categories are recognised in this repository. Every source file and every class belongs to exactly one category. The category membership listed here is authoritative.

---

### Core Ontological Categories

#### 1. Point Forecast Models

- **Purpose:** Deterministic, single-value predictions. For each (entity, time) cell in the forecast horizon, produce one scalar per target column.
- **Classes:** `ZeroModel`, `LocfModel`, `AverageModel`
- **File:** `views_baseline/model/baseline.py`
- **Interface contract:** `fit(df) -> self`, `predict(df, sequence_number, output_length) -> pd.DataFrame` with `MultiIndex(time, entity)` and columns `pred_{target}`.
- **Authority:** Authoritative — these classes produce the final prediction values for point-forecast use cases.
- **Stability:** Stable. Changes to their output format or fit/predict signatures would break downstream ensemble consumers.

#### 2. Distributional Forecast Models

- **Purpose:** Probabilistic, multi-sample predictions. For each (entity, time) cell, produce `n_samples` draws per target.
- **Classes:** `ConflictologyModel`, `MixtureBaseline`
- **File:** `views_baseline/model/baseline.py`
- **Interface contract:** `fit(df) -> self`, `predict(df, sequence_number, output_length) -> dict[str, PredictionFrame]`. Both classes carry a class attribute `distributional = True` which is the protocol discriminator.
- **Authority:** Authoritative — these classes produce the final prediction values for distributional use cases.
- **Stability:** Evolving. The output format changed from `DataFrame` to `PredictionFrame` during development and may change again as the `PredictionFrame` schema evolves upstream.

#### 3. Model Protocols

- **Purpose:** Structural typing contracts that allow runtime dispatch without import coupling. Callers can check `isinstance(model, DistributionalBaselineModel)` without importing concrete model classes.
- **Classes:** `BaselineModel`, `DistributionalBaselineModel`
- **File:** `views_baseline/model/protocol.py`
- **Implementation detail:** Both protocols are decorated `@runtime_checkable`. `DistributionalBaselineModel` adds the `distributional: bool` attribute requirement to the protocol, which serves as the explicit discriminator used by `_generate_predictions` and `_forecast_model_artifact` in the manager.
- **Authority:** Authoritative — these protocols define the model interface that the manager and any future caller must program against.
- **Stability:** Stable. Protocol changes require coordinated updates to all implementing classes.

#### 4. Model Factory

- **Purpose:** Config-validated model instantiation. The factory owns the mapping from string algorithm names to constructor calls, and enforces that required config keys are present before constructing a model.
- **Classes:** `BaselineModelCatalog`
- **File:** `views_baseline/model/catalog.py`
- **Implementation detail:** `MODEL_GENOMES` is a dict mapping each model name to its list of required config keys. `get_model(name)` raises `ValueError` if the name is unknown or if required keys are missing. Optional parameters (e.g., `n_samples`, `lambda_mix`, `window_months` for `MixtureBaseline`) are handled inside the factory methods via `config.get(key, default)` and are not listed in `MODEL_GENOMES`.
- **Authority:** Authoritative — owns model construction. No other code should instantiate model classes directly in production paths.
- **Stability:** Stable. The catalog's public API (`get_model`, `list_models`) is consumed by the manager.

#### 5. Pipeline Integration

- **Purpose:** Orchestrates the model lifecycle (load data, fit, predict, save artifact, sweep) within the VIEWS pipeline infrastructure. Translates between the pipeline's conventions (run types, artifact paths, partition dicts from `views-pipeline-core`) and the model layer's interface.
- **Classes:** `BaselineForecastingModelManager`
- **File:** `views_baseline/manager/baseline_manager.py`
- **Implementation detail:** Extends `ForecastingModelManager` from `views-pipeline-core`. Overrides five methods: `_train_model_artifact`, `_setup_model_and_data`, `_generate_predictions`, `_evaluate_model_artifact`, `_forecast_model_artifact`. Imports 6 names from `views-pipeline-core`: `PipelineConfig`, `generate_model_file_name`, `read_dataframe`, `ForecastingModelManager`, `ModelPathManager`, and the `DistributionalBaselineModel` protocol (the last is from the local `model/` layer, not pipeline-core).
- **Authority:** Derived — the manager delegates all prediction logic to the model layer. It adds no domain knowledge; it only routes.
- **Stability:** Evolving. Tightly coupled to `views-pipeline-core`; any breaking change there propagates here.

#### 6. Prediction Builders

- **Purpose:** Shared output construction for point forecast models. Provides the canonical implementation of the (entity × time) grid expansion so that `ZeroModel`, `LocfModel`, and `AverageModel` do not each contain duplicated DataFrame assembly logic.
- **Functions:** `build_prediction_grid`
- **File:** `views_baseline/model/helpers.py`
- **Implementation detail:** Takes `time_idx`, `entity_idx`, `loa_ids`, `time_ids`, `targets`, and a `value_fn` callable. Returns a `pd.DataFrame` with `MultiIndex([time_idx, entity_idx])` and columns `pred_{target}`. Handles the empty-entity edge case explicitly.
- **Authority:** Derived — helpers serve the point forecast models, not the other way around.
- **Stability:** Stable. The function signature and output contract are depended on by three model classes.

---

## Rationale

Separating the ontology from the physical file layout makes the categories legible to contributors who read documentation before code. The categories chosen reflect actual authority and dependency relationships in the codebase:

- **Point vs. Distributional** is a functional split, not an implementation convenience. The two categories have different output types, different manager dispatch paths, and different stability expectations.
- **Protocols** are separate from implementations because they define contracts that cross the `model/`–`manager/` boundary without creating an import cycle.
- **Factory** is a distinct category because it owns validation logic; it is not merely a convenience wrapper.
- **Pipeline Integration** is explicitly marked as derived because the manager adds no prediction intelligence — this naming prevents future contributors from adding domain logic there.
- **Prediction Builders** is separated from Point Forecast Models because helpers are utilities, not models, and have a different axis of change.

---

## Considered Alternatives

**Flatten everything into "model classes" and "infrastructure."** This is the implicit ontology in many small ML repositories. Rejected because it obscures the protocol/factory/helper distinctions and does not name the derived/authoritative split explicitly.

**Use abstract base classes instead of protocols.** Would have made the ontological categories explicit in code via inheritance. Rejected in favour of `@runtime_checkable` protocols because protocols allow structural subtyping — a model class satisfies `DistributionalBaselineModel` by having the right attributes and methods, without any base-class import in `model/`.

**Separate files per model class.** Would have made the ontological boundaries physically visible. Rejected as over-engineered for five classes that are closely related and frequently read together.

---

## Consequences

**Positive:**
- New model classes can be classified by referring to this document. A new distributional model goes in `model/baseline.py`; a new pipeline adapter goes in `manager/`.
- The derived/authoritative split makes it clear that the manager should not grow domain logic.
- The stability annotations set expectations about which changes require coordinated downstream updates.

**Negative:**
- The ontology must be kept in sync with the code. If a new category emerges (e.g., a separate evaluation helper), this ADR must be updated.
- The vocabulary is local to this repository and not shared with views-pipeline-core or other VIEWS components.

---

## Implementation Notes

The physical file layout maps to the ontological categories as follows:

```
views_baseline/
  model/
    baseline.py      → Point Forecast Models + Distributional Forecast Models
    protocol.py      → Model Protocols
    catalog.py       → Model Factory
    helpers.py       → Prediction Builders
  manager/
    baseline_manager.py → Pipeline Integration
```

---

## Validation & Monitoring

- When a new class is added to `baseline.py`, the contributor should declare in the PR description which ontological category it belongs to and why.
- If the manager begins to contain logic that belongs to a model category (e.g., computing a target column name, applying a threshold), that is a signal that the derived/authoritative split is eroding.
- Protocol changes should trigger a review of all implementing classes to confirm continued conformance.

---

## Open Questions

- Should `PredictionFrame` (imported from `views-pipeline-core`) be considered a seventh category ("External Data Contracts") in this ontology, or is it sufficient to note it as an upstream dependency?
- As distributional models evolve, should `ConflictologyModel` and `MixtureBaseline` be split into separate files to make their independent evolution paths clearer?

---

## References

- ADR 000: Use of ADRs
- ADR 002: Topology and Dependency Rules
- ADR 003: Authority of Declarations over Inference
- `views_baseline/model/baseline.py` — Point and Distributional model implementations
- `views_baseline/model/protocol.py` — Protocol definitions
- `views_baseline/model/catalog.py` — Factory implementation
- `views_baseline/model/helpers.py` — Prediction builder
- `views_baseline/manager/baseline_manager.py` — Pipeline integration
