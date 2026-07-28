# Logging and Observability Standard — views-baseline

**Last reviewed:** 2026-03-13
**Related ADRs:** ADR-003, ADR-005, ADR-008

---

## Overview

This standard records the logging conventions in use in views-baseline and the gaps that
are accepted as known technical debt. It is grounded in the actual logging patterns in the
codebase as of 2026-03-13.

For the architectural principles behind these choices, see:
- [ADR-008: Observability and Explicit Failure](../ADRs/008_observability_and_explicit_failure.md)
- [ADR-003: Authority of Declarations over Inference](../ADRs/003_authority_of_declarations_over_inference.md)

---

## Logger Instantiation

Several modules create named loggers using the standard Python `logging` module, each via
`logger = logging.getLogger(__name__)`:

```text
views_baseline/model/models/point/zero.py         → views_baseline.model.models.point.zero
views_baseline/model/models/point/locf.py         → views_baseline.model.models.point.locf
views_baseline/model/models/point/average.py      → views_baseline.model.models.point.average
views_baseline/model/grid.py                      → views_baseline.model.grid
views_baseline/model/distributions/transforms.py  → views_baseline.model.distributions.transforms
views_baseline/model/distributions/families.py    → views_baseline.model.distributions.families
views_baseline/manager/baseline_manager.py        → views_baseline.manager.baseline_manager
```

The distributional model modules (`model/models/distributional/`), `catalog.py`,
`protocol.py`, `spatial.py`, `frames/` (output seam + pooling), and
`distributions/registry.py` do not log.

**Rule:** Always use `logging.getLogger(__name__)` at module level. Do not use the root
logger (`logging.getLogger()`). Do not create a logger with a hardcoded name string.

---

## Log Levels in Use

### INFO

INFO messages confirm that control flow reached the expected code path. They carry the
entity index name and algorithm name where relevant.

**In the point model modules (`views_baseline/model/models/point/zero.py`, `locf.py`, `average.py`):**

| Location | Message pattern |
|----------|----------------|
| `ZeroModel.predict()` | `"Currently running a {entity_idx} model"` |
| `LocfModel.fit()` | `"Fitting LastObservationModel on level: {entity_idx}"` |
| `LocfModel.predict()` | `"Generating LOCF predictions on level: {entity_idx}"` |
| `AverageModel.fit()` | `"Fitting AverageModel on level: {entity_idx}"` |
| `AverageModel.predict()` | `"Generating average predictions on level: {entity_idx}"` |

**In `views_baseline/manager/baseline_manager.py`:**

| Location | Message pattern |
|----------|----------------|
| `__init__()` | `"Initializing BaselineModelManager"` |
| `_train_model_artifact()` | `"Saved baseline artifact: {filename}"` |
| `_setup_model_and_data()` | `"Model type is {algorithm}"` |
| `_evaluate_model_artifact()` | `"Evaluating baseline model artifact"` |
| `_evaluate_model_artifact()` | `"Generating predictions for {eval_type} evaluation"` |
| `_forecast_model_artifact()` | `"Generating forecasts"` |

INFO messages confirm execution stages. They do not contain structured fields and are not
intended for machine parsing.

### WARNING

WARNING messages are the most operationally significant log output in the package. They
indicate that prediction output is incomplete because entities present in the input data
were absent from the model's fitted state.

**Emitted by the shared `filter_entities` helper in `views_baseline/model/grid.py`** (logger
`views_baseline.model.grid`), once per model that drops entities. The triggering model is
identified by the `model_name` argument threaded through the call:

| Triggering model | Message pattern |
|----------|----------------|
| `LocfModel.predict()` | `"LocfModel: {n} entities dropped"` |
| `AverageModel.predict()` | `"AverageModel: {n} entities dropped"` |
| `ConflictologyModel.predict()` | `"ConflictologyModel: {n} entities dropped"` (via `sample_prediction_grid`) |
| `MixtureBaseline.predict()` | `"MixtureBaseline: {n} entities dropped"` (via `sample_prediction_grid`) |

Each message includes the model class name (the `model_name` argument) and the count of
dropped entities — enough to tell whether the problem is in fitting (wrong train period),
data (missing entities), or configuration (wrong partition boundary). The message is
uniform across models: `filter_entities` is a single shared helper, so it does not name the
per-model fitted-state attribute.

**Rule:** Any model that can drop entities at predict time must emit a WARNING with the
model name and the dropped count. Do not use INFO for entity drops.

### ERROR and CRITICAL

**No ERROR or CRITICAL log messages exist anywhere in the codebase as of 2026-03-13.**

This is a known gap (documented in ADR-008). The most significant silent failure case is:
`ConflictologyModel.predict()` and `MixtureBaseline.predict()` return an empty dict `{}`
if all entities are filtered out. This is a complete prediction failure that is not signalled
at ERROR level. The caller receives `{}` with no indication that something went wrong.

This gap is accepted as technical debt. If it causes operational problems in production,
the fix is to add `logger.error(...)` before returning `{}` in the empty-entity case.

---

## Two-Tier Observability Principle

From ADR-008, the project follows two complementary principles:

1. **Fail loud on configuration errors.** Any problem detectable at setup time — unknown
   model name, missing required config key — must raise a named exception with a
   descriptive message. `BaselineModelCatalog.get_model()` is the primary implementation
   of this principle.

2. **Log at WARNING on data-driven deviations.** Problems arising from data (missing
   entities, empty history pools) are logged at WARNING level. Prediction continues for
   the entities that are not affected.

Entity-drop events sit at the boundary between these two principles. They are currently
handled as WARNING (log, continue) rather than ValueError (raise, abort). This is a known
deviation from the fail-loud principle, documented in ADR-003 and ADR-008. The rationale
is robustness in production: a model that drops a handful of entities should not abort an
entire pipeline run. The warning ensures the deviation is visible.

---

## No Structured Logging

Log messages are unstructured strings. There is no `run_id`, `model_id`, `loa`, or
`sequence_number` attached to any log message as a structured field.

**Implications:**
- Log messages from concurrent pipeline runs cannot be correlated without external context.
- Log aggregation tools cannot filter by model type or evaluation run without parsing text.
- Entity-drop warnings for different models in the same run are indistinguishable except by
  the model name embedded in the message text.

This is a known limitation accepted as technical debt. Structured logging would require
adopting a structured logging library or establishing a convention for embedding context
fields. Neither is currently implemented.

**Do not introduce structured logging** (e.g., `structlog`, JSON-formatted messages) without
a PR discussion. Changing the log output format is a potentially breaking change for any
downstream log consumer.

---

## No Run-Level Context

No log message in the package carries a `run_id`, `sequence_number`, or equivalent
run-level identifier. A pipeline that runs multiple baselines in sequence will produce
interleaved WARNING messages that are distinguished only by the model class name.

This is an accepted limitation. In a single-model pipeline run, the model class name in the
WARNING message is sufficient. In multi-model runs, the pipeline orchestrator is responsible
for adding run-level context if needed.

---

## What Must Not Be Removed

The following logging calls must not be removed, downgraded, or replaced with silent
continuations. They are the primary operational signals available to pipeline operators:

1. The shared entity-drop WARNING call in `filter_entities` (`views_baseline/model/grid.py`), which fires for all four models that can drop entities
2. The artifact-save INFO in `_train_model_artifact()` (confirms artifact was written)
3. The algorithm INFO in `_setup_model_and_data()` (confirms which model is running)

Removing any of these reduces the observability of the pipeline below the current accepted
baseline. Any such change requires a PR discussion referencing ADR-008.

---

## Test Coverage of Logging

**No tests currently assert that WARNING messages are emitted.**

The entity-drop warning code paths are exercised only incidentally through tests that happen
to trigger entity drops. The warnings can be removed or miscoded without CI detecting the
failure. This is a known gap from ADR-005.

A test that verifies warning emission would use `pytest`'s `caplog` fixture:

```python
def test_locf_warns_on_entity_drop(caplog):
    # fit on a subset of entities, predict on the full set
    with caplog.at_level(logging.WARNING, logger="views_baseline.model.grid"):
        model.predict(full_df, sequence_number=0)
    assert any("entities dropped" in r.message for r in caplog.records)
```

Writing such tests is a valued contribution. They belong in the Green team category
(ADR-005) as invariant-enforcement tests.

---

## Adding Logging to New Code

When adding logging to new model classes or manager methods:

1. **Use the module logger.** `logger = logging.getLogger(__name__)` at module level.
2. **One INFO on fit entry.** Name the model and the entity index level.
3. **One INFO on predict entry.** Name the model and the entity index level.
4. **One WARNING per entity-drop event.** After the entity filter, compute `n_dropped` and
   log it. Include the model name, the count, and the storage attribute name.
5. **No logging inside loops.** One aggregate message per method call, not one per entity
   or time step.
6. **No print statements.** All output goes through the logging system.
