# ADR-012: Model Addition Protocol and Catalog Registration

**Status:** Accepted
**Date:** 2026-03-13
**Deciders:** Project maintainers

---

## Context

Adding a new baseline model to views-baseline requires coordinated changes across multiple files. There is no single document that enumerates all the touch points. Contributors have had to reverse-engineer the registration path by reading existing model implementations.

This is error-prone: a model that is defined in `baseline.py` but not registered in `catalog.py` will be importable but invisible to the pipeline. A model registered in the catalog but missing from the test parametrization will have no automated coverage. The protocol tests will not catch it unless the parametrize list is updated.

This ADR serves as the authoritative checklist for adding a new model.

---

## Decision

**Adding a new baseline model requires changes to all of the following files, in order:**

### Step 1: Define the model class

**File:** `views_baseline/model/baseline.py`

- Define a class with `fit(self, df: pd.DataFrame)` and `predict(self, df: pd.DataFrame, sequence_number: int, output_length: int)` methods.
- **All models return `dict[str, PredictionFrame]` from `predict()`** (ADR-010, ADR-017) — point models produce `y_pred` shape `(N, 1)`, distributional models `(N, n_samples)`.
  - **Point models:** build the result with `build_prediction_frame()` from `model/helpers.py`. Do **not** use `build_prediction_grid()` — it is retained for reference but no longer used by any model.
  - **Distributional models:** set `distributional = True` as a class attribute (semantic marker; no longer used for manager dispatch) and construct `PredictionFrame` objects directly.
- `PredictionFrame` is lazy-imported inside the helper / `predict()` — never at module level (ADR-002, ADR-013). For point models this happens inside `build_prediction_frame()`.
- **Call `require_entities(entity_ids, "<ModelName>")` before building output** — fail loud if no entities remain (ADR-008-style explicit failure; uniform across all seven models).
- Accept `targets`, `partition_dict`, and `loa` as constructor parameters (universal). Accept model-specific parameters (e.g., `window_months`, `n_samples`, `seed`) as additional constructor parameters.
- Follow the entity → time → target iteration order for any RNG-consuming loops (ADR-011).

### Step 2: Register in the catalog

**File:** `views_baseline/model/catalog.py`

1. Add an import for the new class at the top of the file.
2. Add an entry to `MODEL_GENOMES` listing required config keys (beyond `targets`). For a **stochastic** model this MUST include `seed` — it is a required, audited genome key with no magic default (ADR-021). For a **parametric-family** model (ADR-022) it MUST also include `family` and `transform`; the constructor is responsible for failing loud on an unsupported family or an illegal `family×transform` combination.
3. Add a factory method `_get_<model_name>()` that reads config and constructs the model.
4. Add the model name → factory method mapping in `self.models` inside `__init__`.

### Step 3: Write model tests

**File:** `tests/test_baseline.py`

- Test `fit()` + `predict()` basic behaviour: return type is `dict[str, PredictionFrame]`, keys match targets, `y_pred` shape is `(N, 1)` for point models or `(N, n_samples)` for distributional.
- If RNG-based: add a reproducibility test (call `predict()` twice, assert bitwise equality).
- If the model drops entities: test the warning path, and test the empty-entity case raises a descriptive `ValueError` via `require_entities`.

### Step 4: Add catalog tests

**File:** `tests/test_catalog.py`

- Add the model name to any parametrized catalog tests (e.g., `test_catalog_returns_correct_type`).
- If the model requires specific config keys: add a test that missing keys raise `ValueError`.

### Step 5: Add protocol tests

**File:** `tests/test_protocol.py`

- Add the model class to the parametrized `isinstance` check for `BaselineModel` (all models). Distributional models additionally satisfy `DistributionalBaselineModel` (the `distributional = True` marker) — though the manager no longer dispatches on it.

### Step 6: Update documentation

**File:** `README.md`

- Add the model to the appropriate section (Point Forecast Models or Distributional Models).
- Include constructor signature and a brief description.

### Step 7: Update governance documents (if applicable)

- **ADR-001** — Add the model to the ontology if it introduces a new category or changes membership.
- **ADR-004** — Add the model to the stability classification (Stable or Evolving).
- **docs/CICs/** — Write a Class Intent Contract if the model is non-trivial (ADR-006).

---

## Rationale

- A checklist reduces the risk of partial registration (model exists but is unreachable via the catalog, or reachable but untested).
- The ordering (define → register → test → document) matches the natural development flow and ensures each step can be verified before moving to the next.
- Requiring protocol tests ensures that new models satisfy the structural contract the manager relies on (every model returns `dict[str, PredictionFrame]`).

---

## Considered Alternatives

### Alternative A: Auto-discovery via metaclass or decorator

Automatically register models by scanning `baseline.py` for classes that implement the protocol.

- **Pros:** No manual registration step; impossible to forget.
- **Cons:** Implicit magic; harder to debug; `MODEL_GENOMES` (required config keys) cannot be inferred from the class alone; violates ADR-003 (inference over declaration).
- **Reason for rejection:** The manual checklist is 7 steps for an infrequent operation. The explicitness cost is low and the debuggability benefit is high.

### Alternative B: Single-file model registration

Define the model, its genome, and its factory in one file.

- **Pros:** All registration info co-located.
- **Cons:** Would require restructuring the existing `catalog.py` / `baseline.py` split. Not worth the churn for the current 5-model catalog.
- **Reason for rejection:** Current structure works. Revisit if the catalog grows past ~10 models.

---

## Consequences

### Positive

- New contributors have a clear, ordered checklist.
- Partial registration is detectable: if a model is in `baseline.py` but not in `catalog.py`, the catalog tests will not cover it; if it is in the catalog but not in `test_protocol.py`, protocol coverage is missing.
- The checklist makes code review more systematic — reviewers can verify each step.

### Negative

- 7 files must be touched for a new model. This is inherent complexity, not accidental — but it is a non-trivial onboarding cost.
- The checklist must be maintained. If a new file is added to the registration path (e.g., a validation schema), this ADR must be updated.

---

## Implementation Notes

The current 7 models all follow this protocol:

| Model | baseline.py | catalog.py | test_baseline.py | test_catalog.py | test_protocol.py | README.md |
|---|---|---|---|---|---|---|
| ZeroModel | yes | yes | yes | yes | yes | yes |
| LocfModel | yes | yes | yes | yes | yes | yes |
| AverageModel | yes | yes | yes | yes | yes | yes |
| ConflictologyModel | yes | yes | yes | yes | yes | yes |
| MixtureBaseline | yes | yes | yes | yes | yes | yes |
| ParametricConflictology | yes | yes | yes (`test_parametric.py`) | yes | yes | yes |
| ParametricHurdleConflictology | yes | yes | yes (`test_parametric.py`) | yes | yes | yes |

No action is required beyond writing this ADR.

---

## Validation & Monitoring

- The protocol tests in `test_protocol.py` parametrize over a list of model classes. If a new model is added to `baseline.py` but not to this list, it will lack protocol coverage. A future improvement could auto-discover model classes, but this is not currently implemented.
- Code review should verify that PRs adding a new model touch all 6+ files listed above.

---

## Open Questions

- Should there be a CI check that verifies all classes in `baseline.py` are registered in `catalog.py`? This would catch partial registration automatically.
- Should `MODEL_GENOMES` be co-located with the model classes (e.g., as a class attribute) rather than centralized in the catalog?

---

## References

- ADR-001: Ontology of the Repository
- ADR-002: Topology and Dependency Rules (lazy import mandate)
- ADR-003: Authority of Declarations over Inference (MODEL_GENOMES validation)
- ADR-004: Rules for Evaluation and Stability (stability classification)
- ADR-006: Intent Contracts for Non-Trivial Classes (CIC requirement)
- ADR-011: RNG Determinism Contract (iteration order for distributional models)
- `views_baseline/model/baseline.py` — model definitions
- `views_baseline/model/catalog.py` — catalog registration
- `tests/test_baseline.py`, `tests/test_catalog.py`, `tests/test_protocol.py` — test suites
