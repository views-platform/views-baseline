# ADR-008: Observability and Explicit Failure

- **Status:** Accepted
- **Date:** 2026-03-13
- **Deciders:** Project maintainers

---

## Context

views-baseline is used within a larger pipeline. When a baseline model silently produces wrong output — empty predictions, NaN-filled arrays, a shorter-than-expected entity list — the error may not surface until a downstream consumer notices anomalous results. At that point the cause is opaque: was it a data issue, a configuration issue, a code bug, or a transient failure?

The project currently has two classes of observability mechanism in place: Python logging and explicit `ValueError` raises. These are inconsistently applied. Some failure modes are logged with a warning; others fail silently; one category (config validation) raises immediately. This ADR records what is actually in place, what is not, and the principles that should govern future additions.

---

## Decision

The project adopts two complementary principles:

**Fail loud on configuration errors.** Any problem that can be detected at setup time — before data is loaded or predictions are generated — must raise a named exception with a message that identifies what is wrong. Silently proceeding with a broken configuration is not acceptable.

**Log at the appropriate level on data-driven deviations.** Problems that arise from data (missing entities, empty history pools, dropped observations) are logged at WARNING level. They are not exceptions because the rest of the prediction run can proceed for the entities that are not affected. They are not silent because the caller must be able to see that the output is incomplete.

---

## Current Logging Implementation

### Module-level loggers

Two modules create named loggers:

- `views_baseline/model/baseline.py`: `logger = logging.getLogger(__name__)` → logger name `views_baseline.model.baseline`
- `views_baseline/manager/baseline_manager.py`: `logger = logging.getLogger(__name__)` → logger name `views_baseline.manager.baseline_manager`

No other module in the package uses logging.

### INFO-level logging

The following INFO-level messages are emitted during normal operation:

- `ZeroModel.predict()`: `"Currently running a {entity_idx} model"` — logged on every predict call.
- `LocfModel.fit()`: `"Fitting LastObservationModel on level: {entity_idx}"`
- `LocfModel.predict()`: `"Generating LOCF predictions on level: {entity_idx}"`
- `AverageModel.fit()`: `"Fitting AverageModel on level: {entity_idx}"`
- `AverageModel.predict()`: `"Generating average predictions on level: {entity_idx}"`
- `BaselineForecastingModelManager.__init__()`: `"Initializing BaselineModelManager"`
- `BaselineForecastingModelManager._train_model_artifact()`: `"Saved baseline artifact: {filename}"`
- `BaselineForecastingModelManager._setup_model_and_data()`: `"Model type is {algorithm}"`
- `BaselineForecastingModelManager._evaluate_model_artifact()`: two messages: `"Evaluating baseline model artifact"` and `"Generating predictions for {eval_type} evaluation"`
- `BaselineForecastingModelManager._forecast_model_artifact()`: `"Generating forecasts"`

These messages confirm that control flow reached the expected branch. They carry no structured data beyond the entity index name and algorithm name.

### WARNING-level logging

The following WARNING-level messages are emitted when entities present at `train_end` are absent from the fitted model state:

- `LocfModel.predict()`: `"LocfModel: {n} entities dropped (missing from last_observations)"`
- `AverageModel.predict()`: `"AverageModel: {n} entities dropped (missing from mean)"`
- `ConflictologyModel.predict()`: `"ConflictologyModel: {n} entities dropped (missing from hist_per_entity)"`
- `MixtureBaseline.predict()`: `"MixtureBaseline: {n} entities dropped (missing from local_pool)"`

These are the most operationally significant log messages in the package. They indicate that the prediction output covers fewer entities than the input data, which is a data quality signal. The messages include the model name and the count of dropped entities.

### ERROR and CRITICAL level logging

No ERROR or CRITICAL level log messages exist anywhere in the codebase as of 2026-03-13.

This is a known gap. There are code paths that reach a semantically error-like state — for example, `ConflictologyModel.predict()` and `MixtureBaseline.predict()` return an empty dict `{}` if `entities_with_history` is empty after entity filtering. This is a complete prediction failure, but it is not logged at ERROR level; it returns silently.

---

## Current Fail-Loud Implementation

### BaselineModelCatalog.get_model()

`get_model()` raises `ValueError` in two cases:

1. The requested model name is not in `self.models` (unknown model): raises `ValueError` with the unknown name and the list of available names in the message.
2. A required config key listed in `MODEL_GENOMES[model_name]` is absent from `self.config`: raises `ValueError` naming the missing keys.

This is the primary fail-loud boundary in the project. It fires at model construction time, before any data is loaded or any computation occurs.

### What is not validated

The catalog validates key presence but not value validity. The following degenerate configurations are accepted without error:

- `window_months=0` — `AverageModel` and `ConflictologyModel` will compute means and histories over zero rows; `tail(0)` returns an empty DataFrame; `AverageModel` stores NaN means that silently propagate into predictions.
- `output_length=0` — all models accept this at predict time; they return an empty DataFrame or empty dict without warning.
- `partition_dict` with `test[0]` larger than the maximum time index in the data — entity list at `train_end` is empty; predictions are silently empty.
- `targets` referencing columns absent from the input DataFrame — raises `KeyError` at fit time with no contextual message about which model or target caused the failure.

---

## No Structured Logging

Log messages are unstructured strings. There is no `run_id`, `model_id`, `loa`, or `sequence_number` attached to any log message as a structured field. This means:

- Log messages from concurrent pipeline runs cannot be correlated without external context.
- Log aggregation tools (e.g., structured log search) cannot filter by model type or evaluation run without parsing message text.

This is a known limitation and an accepted deviation from the stated principle of making failures observable. Structured logging would require either adopting a structured logging library or establishing a convention for embedding context in message strings. Neither has been done.

---

## Consequences

**Positive:**

- Configuration errors (unknown model name, missing required key) fail immediately with a message that identifies the problem. A misconfigured pipeline does not proceed silently to data loading and prediction.
- Entity-drop events are visible in logs at WARNING level. A caller can configure their logging handler to surface these.
- The INFO-level trail provides a minimal execution trace for debugging.

**Negative / Risks:**

- Silent failure on complete entity loss: if all entities are dropped before prediction (e.g., due to a partition boundary mismatch), distributional models return `{}` without any ERROR-level signal. The caller receives an empty result with no indication that something went wrong.
- Degenerate parameter values (`window_months=0`, `output_length=0`) produce incorrect output without any warning. This violates the fail-loud principle for what are effectively configuration errors.
- No run-level context in log messages makes it difficult to correlate a warning with the specific model configuration that triggered it in a pipeline that runs multiple baselines.
- No tests assert that WARNING messages are emitted. The warning paths can be broken without CI detecting the failure.

**Accepted gaps:**

The structured logging gap and the absence of ERROR-level signalling on complete prediction failure are accepted as known technical debt. They are documented here so they can be addressed deliberately if operational debugging proves difficult.
