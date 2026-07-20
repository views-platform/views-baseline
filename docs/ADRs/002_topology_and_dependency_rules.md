# ADR 002: Topology and Dependency Rules

**Status:** Accepted
**Date:** 2026-03-13
**Deciders:** Project maintainers

> **Amendment (PR-1 reorg, 2026-07, issues #48–#51):** the SOLID / screaming-architecture
> reorganization refined the `model/` layer into responsibility-separated modules. The
> **rules and their intent are unchanged** — the graph is still acyclic and directed
> `leaf-utilities → construction-seam → models → catalog → manager`; only the file names and
> their number grew. The single ex-`helpers.py` node is now the leaf utilities
> (`spatial.py`, `grid.py`, `distributions/`, `frames/pooling.py`, `defaults.py`) plus the
> construction seam (`frames/output.py`); `baseline.py` is now the per-class files under
> `model/models/{point,distributional}/`. The Decision, Forbidden-Imports, and
> Implementation-Notes sections below have been updated to the post-reorg paths; the
> Rationale and Alternatives are unchanged. Acyclicity was re-verified after the reorg
> (`import views_baseline.model.catalog` and the sub-packages import cleanly; no cycles).

---

## Context

views-baseline has two top-level sub-packages: `model/` and `manager/`. These serve different purposes and have different external dependency profiles. Without explicit rules about what may import what, the dependency graph is at risk of collapsing: a convenience import in the wrong direction creates a cycle, pulls in `views-pipeline-core` into code that should be pure, and makes unit testing harder.

The project also has a specific constraint: `PredictionFrame` (from `views-pipeline-core`) is required by the distributional models' `predict()` output, but `model/` should otherwise remain free of pipeline-core imports. Resolving this tension requires a deliberate rule rather than ad-hoc decisions at each touch point.

This ADR records the dependency topology and the rules that enforce it.

---

## Decision

### Dependency Graph

The allowed import relationships are:

```
numpy, pandas, scipy
      |
      v
leaf utilities:  model/spatial.py   model/grid.py   model/defaults.py
                 model/distributions/{transforms,families → registry}
      |
      v
model/frames/input.py    (to_feature_frame — the ONE pandas reader; ADR-019/DIP boundary;
                          pandas + views_frames imported lazily inside the functions)
      |
      v
model/frames/pooling.py  (window_pool on a FeatureFrame; imports frames/input.panel)
model/frames/output.py   (construction seam; imports grid + spatial)
      |
      v
model/models/point/{zero,locf,average}.py
model/models/distributional/{conflictology,mixture,parametric,parametric_hurdle}.py
      |
      v
model/catalog.py          model/protocol.py
      |                          |
      +----------+---------------+
                 |
                 v
         manager/baseline_manager.py
                 |
                 v
         views-pipeline-core / views-frames
```

Written as explicit rules:

1. **Leaf utilities** — `model/spatial.py`, `model/grid.py`, `model/defaults.py`, and the `model/distributions/` package — import only `numpy`/`scipy`, the Python standard library, and (within `distributions/`) each other in the direction `transforms, families → registry`. No `pandas`. No imports from `model/models/`, `model/frames/`, `model/catalog.py`, `model/protocol.py`, or `manager/`.
2. **`model/frames/input.py`** is the **single pandas reader** (`to_feature_frame`, the ADR-019 / DIP boundary). It imports `model/spatial.py`, `numpy`, and — **lazily, inside the functions** — `pandas` and the `views_frames` leaf. Nothing else in `model/` imports `pandas` at module scope (guarded by a falsification test). `model/frames/pooling.py` imports `frames/input.panel`; `model/frames/output.py` imports `model/grid.py`, `model/spatial.py`, and lazily the `views_frames` leaf. None of these import from `model/models/`, `model/catalog.py`, `model/protocol.py`, or `manager/`.
3. **Model classes** (`model/models/point/*`, `model/models/distributional/*`) import from `model/frames/{input,output,pooling}`, `model/distributions`, `model/grid.py`, `model/spatial.py`, `model/defaults.py`, `numpy`, and the standard library — **not `pandas`** (imported only under `TYPE_CHECKING` for the `pd.DataFrame | FeatureFrame` hint). They do not import from `model/catalog.py`, `model/protocol.py`, or `manager/`.
4. **`model/catalog.py`** imports from the model sub-packages (`model/models/point`, `model/models/distributional`) and from `infrastructure/reproducibility_gate.py` (the genome source of truth). It does not import from `model/protocol.py` or `manager/`.
5. **`model/protocol.py`** imports `pandas` and the Python standard library only. It does not import from any other `model/` file or from `manager/`.
6. **`manager/baseline_manager.py`** may import from `model/catalog.py`, `model/protocol.py`, and from `views-pipeline-core`. It does not import from the model classes or the leaf utilities directly.

### The `PredictionFrame` Lazy Import Rule

> **Amended by ADR-020 (2026-06-24):** `PredictionFrame` now comes from the `views_frames` leaf
> (re-exported by `views-pipeline-core` ≥3.0.0, #188), and all construction is **consolidated into
> a single function-scoped import site** — `to_prediction_frames`, now in `model/frames/output.py`
> (was `model/helpers.py` before the PR-1 reorg). The two per-model lazy imports described below
> are replaced by that one site; the distributional models no longer construct `PredictionFrame`
> inline. This answers the first Open Question at the foot of this ADR. The rule that `model/`
> carries no module-level frame import is unchanged.

`PredictionFrame` from `views-pipeline-core` is required inside the `predict()` methods of `ConflictologyModel` and `MixtureBaseline`, but `model/` must not carry `views-pipeline-core` as a hard module-level import. The resolution is:

```python
# Inside predict() only — not at module top-level
from views_pipeline_core.data.prediction_frame import PredictionFrame
```

This import is deferred to call time. The cost is a minor runtime overhead on first call (mitigated by Python's module cache). The benefit is that `model/` can be imported, tested, and used in environments where `views-pipeline-core` is not installed, as long as the distributional `predict()` path is not exercised.

### Forbidden Imports

The following import directions are explicitly forbidden:

| Forbidden | Reason |
|---|---|
| `model/` importing from `manager/` | Would create an upward dependency; model layer must not know about pipeline infrastructure |
| A leaf utility (`model/grid.py`, `model/spatial.py`, `model/distributions/*`, `model/frames/pooling.py`) or `model/frames/output.py` importing from `model/models/` | Utilities and the construction seam are serve-the-models code; importing from the models they serve inverts the hierarchy |
| `model/protocol.py` importing from `model/models/` | Protocols define structural contracts; knowing the concrete implementations would make them implementation-aware |
| `model/catalog.py` importing from `manager/` | Factory lives in the model layer; importing from the manager layer is an upward dependency |

---

## Rationale

### Why `model/` is the foundation layer

The point forecast models (`ZeroModel`, `LocfModel`, `AverageModel`) have no need for pipeline-core and should remain testable without it. The 51-test suite depends on this: `test_baseline.py` and `test_catalog.py` import only from `views_baseline.model.*` and can run without a pipeline-core installation beyond what the distributional model tests need.

Keeping `model/` pure also means it could, in principle, be used outside the VIEWS pipeline (e.g., in a notebook or standalone script) without dragging in pipeline infrastructure.

### Why `PredictionFrame` is lazy-imported

The alternative is to make `views-pipeline-core` a hard dependency of `model/`. This was rejected because:

- It couples the model layer's importability to pipeline-core's availability.
- It makes the dependency declaration misleading: `model/` does not need pipeline-core to do its work; it only needs it to wrap its output.
- Lazy imports at function scope are an established Python pattern for optional or conditional heavy dependencies.

The tradeoff is that a missing `views-pipeline-core` installation is only surfaced at `predict()` call time for distributional models, not at import time. This is accepted as a known limitation; the two distributional model tests that call `predict()` will fail with a clear `ModuleNotFoundError` if pipeline-core is absent.

### Why `manager/` imports only `catalog.py` and `protocol.py` from `model/`

The manager needs to instantiate models (via the catalog) and to dispatch on model type (via the protocol). It does not need to import concrete model classes directly because the catalog handles construction and the protocol handles type checking. If the manager imported `ZeroModel` directly, it would be bypassing the factory and partially duplicating the catalog's responsibilities.

---

## Considered Alternatives

**Allow `model/` to import `views-pipeline-core` at module level.** This would simplify the distributional `predict()` methods (no deferred import required). Rejected because it couples the foundation layer to the pipeline infrastructure, defeating the purpose of the layered architecture.

**Use a `TYPE_CHECKING` guard instead of a deferred import.** `from __future__ import annotations` plus `if TYPE_CHECKING: from views_pipeline_core...` would avoid the runtime overhead. Rejected because the import is not purely for type annotations; `PredictionFrame` is instantiated at runtime inside `predict()`. `TYPE_CHECKING` guards are for type annotations only.

**Merge `model/` and `manager/` into a single package.** Would remove the need for inter-package import rules. Rejected because it conflates domain logic (model layer) with infrastructure (pipeline integration), making it harder to test, replace, or reuse either independently.

**Enforce rules via a linting plugin (e.g., `import-linter`).** Would make the rules machine-checked. Not currently implemented; the rules are enforced by convention and code review. This is a known gap.

---

## Consequences

**Positive:**
- The model layer (`model/`) can be imported and unit-tested without `views-pipeline-core` present (for point model tests).
- The dependency graph is acyclic and has a clear direction of dependence: leaf utilities → construction seam → models → catalog → manager.
- The manager's imports (`catalog.py`, `protocol.py`, pipeline-core) are minimal and explicit, making the coupling surface visible.
- Adding a new point forecast model requires no changes to the manager or the protocol.

**Negative:**
- The lazy `PredictionFrame` import is non-obvious to readers unfamiliar with the pattern. It requires a comment to explain.
- Dependency rules are currently enforced by convention only. A contributor could add a forbidden import without immediate mechanical feedback.
- If `views-pipeline-core` ever restructures its `data.prediction_frame` module path, both `ConflictologyModel.predict()` and `MixtureBaseline.predict()` break simultaneously — the two deferred import sites must be updated together.

---

## Implementation Notes

The actual import statements are:

**`model/frames/output.py` (top of file — the construction seam):**
```python
import numpy as np
from views_baseline.model.grid import (
    build_identifier_arrays, build_time_grid, filter_entities, require_entities,
)
from views_baseline.model.spatial import resolve_level
```

**A point model, e.g. `model/models/point/locf.py` (top of file):**
```python
import pandas as pd
from views_baseline.model.frames.output import build_prediction_frame
from views_baseline.model.grid import build_time_grid, filter_entities, require_entities
from views_baseline.model.spatial import resolve_level
```

**A distributional model, e.g. `model/models/distributional/conflictology.py` (top of file):**
```python
import pandas as pd
from views_baseline.model.defaults import DEFAULT_SEED
from views_baseline.model.frames.output import sample_prediction_grid
from views_baseline.model.frames.pooling import window_pool
```

**`model/frames/output.py` (inside `to_prediction_frames()` — the single `views_frames` construction site, ADR-020):**
```python
from views_frames import PredictionFrame, SpatioTemporalIndex
```

**`model/catalog.py` (top of file):**
```python
from views_baseline.model.models.distributional import (
    ConflictologyModel, MixtureBaseline, ParametricConflictology, ParametricHurdleConflictology,
)
from views_baseline.model.models.point import AverageModel, LocfModel, ZeroModel
```

**`manager/baseline_manager.py` (top of file):**
```python
from views_pipeline_core.files.utils import generate_model_file_name, read_dataframe
from views_pipeline_core.managers.model import ForecastingModelManager, ModelPathManager
from views_baseline.model.catalog import BaselineModelCatalog
```
*(Since ADR-017 the manager no longer imports `DistributionalBaselineModel` — it has a single, type-uniform prediction path.)*

---

## Validation & Monitoring

- Any PR that adds an import to a `model/` file should be reviewed against the forbidden imports table.
- The test suite implicitly validates the foundation-layer rule: if `model/` acquired a hard pipeline-core import, distributional model tests would fail in environments where pipeline-core is correctly installed but for the wrong reason (module-level import failure rather than `predict()` failure).
- Consider adding an `import-linter` or `flake8-import-order` configuration in a future CI step to enforce these rules mechanically.

---

## Open Questions

- Should the lazy import pattern be extracted into a utility (e.g., a `_get_prediction_frame()` function in a shared module) to make the single import site easier to update when the upstream path changes?
- If a third distributional model is added, should the lazy import be consolidated at the `catalog.py` or `manager/` level instead of being repeated in each model's `predict()` method?

---

## References

- ADR 001: Ontology of the Repository
- ADR 003: Authority of Declarations over Inference
- `views_baseline/model/frames/output.py`, `views_baseline/model/frames/pooling.py`
- `views_baseline/model/grid.py`, `views_baseline/model/spatial.py`, `views_baseline/model/defaults.py`
- `views_baseline/model/distributions/` (package: `transforms.py`, `families.py`, `registry.py`)
- `views_baseline/model/models/point/`, `views_baseline/model/models/distributional/`
- `views_baseline/model/catalog.py`
- `views_baseline/model/protocol.py`
- `views_baseline/manager/baseline_manager.py`
