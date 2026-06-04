# ADR-013: views-pipeline-core Coupling Management

**Status:** Accepted
**Date:** 2026-03-13
**Deciders:** Project maintainers

---

## Context

views-baseline depends on views-pipeline-core for pipeline integration (model management, data loading, configuration) and for the `PredictionFrame` output type used by distributional models. This coupling is unavoidable — views-baseline is a plugin in the VIEWS pipeline — but it must be managed deliberately because:

1. **Import-time breakage.** Module-level imports from views-pipeline-core mean that any API change in the base package breaks views-baseline at import time, not at call time. This makes failures loud (good) but also means a views-pipeline-core release can break views-baseline even if no views-baseline code changed.
2. **Testability.** The `model/` package must be independently testable without views-pipeline-core installed (for fast local development and CI isolation). This constrains where views-pipeline-core imports can appear.
3. **Coupling surface area.** The more symbols imported from views-pipeline-core, the more breakage vectors exist. The coupling surface should be documented and minimized.

---

## Decision

**The coupling management strategy is: zero module-level imports from views-pipeline-core in `model/`, lazy imports for `PredictionFrame` in distributional `predict()` methods, and explicit module-level imports in `manager/` and `tests/`.**

### Coupling inventory

#### `model/` package — ZERO module-level imports

- `model/baseline.py` — no module-level views-pipeline-core imports.
- `model/catalog.py` — no module-level views-pipeline-core imports.
- `model/protocol.py` — no module-level views-pipeline-core imports.
- `model/helpers.py` — no module-level views-pipeline-core imports.

#### `model/` — 3 lazy `PredictionFrame` import sites (all inside function bodies)

```python
# model/baseline.py — ConflictologyModel.predict()
from views_pipeline_core.data.prediction_frame import PredictionFrame

# model/baseline.py — MixtureBaseline.predict()
from views_pipeline_core.data.prediction_frame import PredictionFrame

# model/helpers.py — build_prediction_frame()  (used by ALL point models since ADR-010/ADR-017)
from views_pipeline_core.data.prediction_frame import PredictionFrame
```

These are the only points where `model/` touches views-pipeline-core. Since ADR-010/ADR-017 unified all models onto PredictionFrame output, the point models reach pipeline-core too — via `build_prediction_frame()` in `helpers.py`. All three sites are still lazy (inside function bodies), so `model/` remains importable without views-pipeline-core; the dependency is incurred only when `predict()` is actually called.

#### `manager/baseline_manager.py` — 4 module-level imports

```python
from views_pipeline_core.files.utils import generate_model_file_name, read_dataframe
from views_pipeline_core.managers.model import ForecastingModelManager, ModelPathManager
```

These are acceptable: `manager/` is the pipeline integration layer and is expected to couple tightly to views-pipeline-core. It is not independently testable by design. The manager also depends on `_get_cached_data_path()` from the base class at runtime.

#### `tests/` — mixed

- `tests/conftest.py` — imports `PredictionFrame` (assertion helper for PF output) and `ConfigurationManager`.
- `tests/test_baseline.py` — exercises `predict()` for **all** models, so all now reach `PredictionFrame` via `build_prediction_frame` / distributional `predict()`. **This file now requires views-pipeline-core** (see the narrowed guarantee below).
- `tests/test_baseline_manager.py` — module-level `ConfigurationManager` import (required to construct manager instances).
- `tests/test_catalog.py` — no views-pipeline-core imports (catalog construction only; does not call `predict()`).
- `tests/test_protocol.py` — no views-pipeline-core imports (`isinstance` protocol checks only).

### Rules

1. **`model/` must have zero module-level imports from views-pipeline-core.** Any new import in `model/` that touches views-pipeline-core must use the lazy import pattern (import inside the function that needs it).
2. **`manager/` may import freely from views-pipeline-core** at module level. This is the integration boundary.
3. **Tests may import from views-pipeline-core** as needed, but should prefer lazy imports when testing model code to preserve the testability guarantee.
4. **New lazy import sites must be documented** in this ADR's coupling inventory.

---

## Rationale

- The `model/` zero-**module-level**-import rule keeps `model/` *importable* without views-pipeline-core. **Narrowed guarantee (ADR-010/ADR-017):** since all models now emit PredictionFrame, calling any model's `predict()` reaches pipeline-core. `tests/test_catalog.py` and `tests/test_protocol.py` (which only construct/inspect, never `predict()`) still run without pipeline-core; `tests/test_baseline.py` no longer does, because every model's `predict()` now builds a PredictionFrame. The original "run the whole model test suite without pipeline-core" property no longer holds — this is an accepted cost of unifying on PredictionFrame.
- Module-level imports in `manager/` are acceptable because the manager is inherently a views-pipeline-core adapter. Testing it requires views-pipeline-core anyway (via `ConfigurationManager`, `ForecastingModelManager`).
- Lazy imports in `predict()` are a pragmatic compromise: distributional output requires `PredictionFrame`, but the import cost is paid only at prediction time, not at module load time.

---

## Considered Alternatives

### Alternative A: Abstract PredictionFrame behind a local interface

Define a `views_baseline.model.prediction.DistributionalOutput` class that wraps the data, and convert to `PredictionFrame` only in the manager.

- **Pros:** `model/` has zero runtime dependency on views-pipeline-core.
- **Cons:** Extra abstraction layer; the conversion would duplicate `PredictionFrame` construction logic; the manager would need to know the internal structure of `DistributionalOutput`.
- **Reason for rejection:** Over-engineered. The lazy import achieves the same testability goal with no additional code.

### Alternative B: Pin views-pipeline-core to an exact version

- **Pros:** Eliminates surprise breakage.
- **Cons:** Requires manual version bumps for every upstream release; prevents consuming bugfixes.
- **Reason for rejection:** A version range (`>=2.0.0,<4.0.0`) is more practical. Exact pinning is appropriate only for production lockfiles, not for library dependencies.

### Alternative C: Import everything at module level in model/

- **Pros:** Simpler code (no lazy imports).
- **Cons:** Breaks independent testability of `model/`. Any views-pipeline-core change breaks all model imports.
- **Reason for rejection:** Directly violates the testability requirement (ADR-002, ADR-005).

---

## Consequences

### Positive

- `model/` remains *importable* without views-pipeline-core; catalog/protocol tests still run in isolation. (Model `predict()` tests now require pipeline-core — see the narrowed guarantee under Rationale.)
- The coupling inventory is explicit and auditable. New coupling is visible in code review.
- Import-time failures in `manager/` are loud and immediate, which is the correct failure mode for integration code.

### Negative

- Lazy imports add 2 lines of boilerplate per distributional model.
- The coupling inventory in this ADR must be kept in sync with the code. If a new import is added, this ADR should be updated.
- views-pipeline-core API changes can still break `manager/` at import time. Mitigation: version range constraints in `pyproject.toml`.

---

## Implementation Notes

The coupling management strategy is already implemented. This ADR codifies it.

To verify the zero-import rule for `model/`:

```bash
grep -r "from views_pipeline_core" views_baseline/model/ | grep -v "def predict"
```

This should return no results. Any match indicates a module-level import that violates the rule.

To verify lazy imports are inside function bodies:

```bash
grep -n "from views_pipeline_core" views_baseline/model/baseline.py
```

Both results should be inside `predict()` methods (indented, after `def predict`).

---

## Validation & Monitoring

- The test suite implicitly validates the coupling strategy: if a module-level import is added to `model/`, tests that don't need views-pipeline-core will fail when run in isolation.
- CI should run model tests (`test_baseline.py`, `test_catalog.py`, `test_protocol.py`) in an environment where views-pipeline-core is installed, but the zero-import rule means they *could* run without it (modulo distributional predict tests).
- Failure mode: a contributor adds `from views_pipeline_core import X` at the top of `model/baseline.py`. This should be caught in code review by checking against this ADR.

---

## Open Questions

- Should a CI lint rule enforce the zero-import constraint for `model/`? A simple grep-based check would catch violations automatically.
- Should the version range for views-pipeline-core be documented here or only in `pyproject.toml`?
- If views-pipeline-core splits into sub-packages, the coupling inventory will need restructuring.

---

## References

- ADR-002: Topology and Dependency Rules (import direction constraints)
- ADR-005: Testing as Mandatory Critical Infrastructure (testability requirement)
- ADR-010: PredictionFrame Adoption (lazy import motivation)
- `views_baseline/model/baseline.py` — lazy import sites
- `views_baseline/manager/baseline_manager.py` — module-level imports
- `pyproject.toml` — views-pipeline-core version constraint
