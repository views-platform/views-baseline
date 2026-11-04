# Class Intent Contract: BaselineForecastingModelManager

**Date:** 2026-03-17 (updated 2026-11-05, #120)
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-001, ADR-002, ADR-006, ADR-008, ADR-009, ADR-014, ADR-016

---

## Purpose

`BaselineForecastingModelManager` orchestrates the full lifecycle of a baseline model within the VIEWS pipeline by extending `views_pipeline_core`'s `ForecastingModelManager`. It is the single entry point for the pipeline to set up, train, evaluate, and forecast using any model registered in `BaselineModelCatalog`. It does not implement model logic itself; instead, it delegates model selection and validation to the catalog, delegates data loading to `_data_loader`, and delegates config management to `_config_manager`.

All models now return `dict[str, PredictionFrame]` from `predict()`. The manager has a single code path — no type-based dispatch.

---

## Non-Goals

- Does not implement any prediction algorithm. All model logic lives in the seven model classes.
- Does not own data loading. Input is read from disk inside `_load_source()`, dispatching on the declared `data_format`: a pandas DataFrame via `read_dataframe` (legacy default) or a `views_frames.FeatureFrame` via `load_frame_cache` when the model declares `data_format: feature_frame` (issue #64). Both come from `views_pipeline_core`.
- Does not define the configuration schema. Config structure is the responsibility of the pipeline configuration layer.
- Does not sweep hyperparameters in any non-trivial way; `_evaluate_sweep` delegates entirely to `_generate_predictions`.
- Does not stream the sweep path. `_evaluate_sweep` stays batch (all origins in memory); pipeline-core's sweep does not use the streaming sink.
- Does not bound the memory of pipeline-core's scoring step, which runs after evaluation returns and reloads the saved origins.

---

## Responsibilities and Guarantees

**`_load_source()`**:
- The single input-sourcing seam (issue #64). Dispatches on the declared `_data_format` (set by the base class during data fetching; absent → `dataframe`, i.e. legacy behavior).
- `feature_frame` → returns a `views_frames.FeatureFrame` via `load_frame_cache(self._get_cached_frame_path())`; the frame flows straight into `fit`/`predict` with no pandas in the manager (models are frame-native since ADR-019).
- Any other value → returns a `pandas.DataFrame` via `read_dataframe(self._get_cached_data_path())` (byte-identical legacy path).
- The frame branch uses the **loud** `_get_cached_frame_path()` getter (raises `RuntimeError` if the frame cache was never populated). There is deliberately no silent fallback from the frame path to pandas — a frames-declared config that lost its cache fails loud (pipeline-core register C-214).

**`_setup_model_and_data()`**:
- Calls `ReproducibilityGate.Config.audit_manifest(self.config)` as a precondition — raises `MissingHyperparameterError` if core or algorithm-specific keys are missing or `None`.
- Reads `config["level"]` and `_data_loader.partition_dict`.
- Instantiates `BaselineModelCatalog` and calls `catalog.get_model(config["algorithm"])`.
- Reads the training input from disk via `self._load_source()` (a DataFrame or FeatureFrame per the declared `data_format`; path set by the base class during data fetching).
- Calls `model.fit(df_source)`.
- Returns `(model, df_source)`.

**`_train_model_artifact()`**:
- Calls `_setup_model_and_data()`.
- Pickles the fitted model to `{path_artifacts}/{run_type}_model_{timestamp}.pkl` via `generate_model_file_name`.
- Returns the fitted model. The artifact is required by the downstream ensemble manager to resolve artifact paths.

**`_iter_predictions(model, df, eval_type)`**:
- The single per-origin loop. Calls `_resolve_evaluation_sequence_number(eval_type)` to determine the iteration count, then yields `(seq_num, model.predict(df=df, sequence_number=seq_num, output_length=config["time_steps"]))` one origin at a time.
- Keeps no reference to an origin once it is yielded.

**`_generate_predictions(model, df, eval_type)`**:
- Batch form: collects `_iter_predictions` into `dict[str, list[PredictionFrame]]`. Every origin is in memory at once. Used by `_evaluate_model_artifact` and `_evaluate_sweep`.

**`_prepare_evaluation(artifact_name)`**:
- Shared set-up for both evaluation paths: resolves the artifact path and persists its timestamp (as below), sets `self.model` via `_setup_model_and_data()`, and returns the input data.

**`_evaluate_model_artifact(eval_type, artifact_name)`**:
- If `artifact_name` is provided, resolves the artifact path as `self._model_path.artifacts / artifact_name`. Otherwise, resolves the latest artifact path via `self._model_path.get_latest_model_artifact_path(run_type=...)`. Raises `FileNotFoundError` if no artifact `.pkl` exists for the run type.
- Extracts the 15-character timestamp from the artifact filename stem and persists it via `self._config_manager.add_config({"timestamp": ...})`.
- Calls `_setup_model_and_data()` then `_generate_predictions()` (via `_prepare_evaluation`).
- Returns the accumulated predictions.

**`_evaluate_model_artifact_streaming(eval_type, artifact_name, origin_sink)`** (#120):
- Overrides pipeline-core's default, which calls the batch method first and so holds all origins (13 × 7.2 GB at pgm `land`, S = 256: 93 GB, an OOM at a 87.5 GiB container limit).
- Same set-up as `_evaluate_model_artifact` (`_prepare_evaluation`), then calls `origin_sink(seq_num, pf_dict)` for each origin from `_iter_predictions`, before the next origin is predicted, and drops its own reference afterwards. With a sink that frees what it saves (pipeline-core's does), peak memory is one origin.
- Output is identical to the batch path's, origin by origin (each `predict()` seeds its own RNG).
- Logs once at the end: the number of origins and the process's peak RSS so far (`ru_maxrss`). This covers prediction and saving, not the scoring that follows.
- This is the path pipeline-core's evaluation uses for `prediction_format: prediction_frame`.

**`_forecast_model_artifact(artifact_name)`**:
- If `artifact_name` is provided, resolves the artifact path as `self._model_path.artifacts / artifact_name`. Otherwise, resolves the latest artifact path via `self._model_path.get_latest_model_artifact_path(run_type=...)`. Raises `FileNotFoundError` if no artifact `.pkl` exists for the run type.
- Extracts the 15-character timestamp from the artifact filename stem and persists it via `self._config_manager.add_config({"timestamp": ...})`.
- Calls `_setup_model_and_data()`.
- Calls `model.predict(df=df_source, sequence_number=0, output_length=output_length)` and returns `dict[str, PredictionFrame]`. Every model — point and distributional — returns this same type; there is no `isinstance` dispatch (ADR-010/ADR-017).

**`_evaluate_sweep(eval_type, model)`**:
- Reads input from disk via `_load_source()` (the same dispatch as `_setup_model_and_data()`).
- Calls `_generate_predictions(model, df, eval_type)`.
- Returns predictions.

---

## Inputs and Assumptions

| Source | What is consumed | Notes |
|---|---|---|
| `model_path` (constructor) | `ModelPathManager` instance | Provides `.artifacts` path. `.data_raw` is no longer accessed directly (data path comes from base class `_cached_data_path`). |
| `self.config` | `dict` | Set via property inherited from base class. Must contain `"run_type"`, `"algorithm"`, and every `CORE_GENOME` key — `"steps"`, `"time_steps"`, `"prediction_format"`, `"regression_targets"`, `"level"` — all audited by `audit_manifest` before the catalog is built. (`targets` was retired upstream in pipeline-core #380; see #85. `run_type` is read one line after the audit but is NOT audited — it is injected at runtime, not declared in a config file; see C-48.) |
| `_data_loader.partition_dict` | `dict` | Must contain `"test"` key. |
| `_config_manager` | `ConfigurationManager` | Used by base class and accessed directly in `_evaluate_model_artifact` and `_forecast_model_artifact` via `add_config()` to persist the artifact timestamp. |

The `config` dict is populated by the base class `_config_manager` before any lifecycle method is called.

---

## Outputs and Side Effects

Output type is **uniform** across point and distributional models (ADR-010): all `predict()`-driven methods return `PredictionFrame`s, differing only in `y_pred` width (`(N, 1)` point / `(N, n_samples)` distributional).

**Downstream consumption (verified, issue #69):** these frames are consumed by pipeline-core's `PredictionFrameEnsembleManager`, whose `_aggregate_prediction_frames` requires constituents to share `sample_count` — a point `(N, 1)` frame pools with other single-sample frames, and mixing it with a distributional `(N, n_samples>1)` frame fails loud (pipeline-core #160 / C-205), never a silent unbalanced pool. Cross-**level** `point-broadcast` (cm→pgm reconciliation) is a separate mechanism, not constituent pooling. Pinned by `tests/test_ensemble_consumption.py`.

| Method | Output type (all models) | Side effects |
|---|---|---|
| `_train_model_artifact()` | Fitted model instance | Writes `.pkl` file to `artifacts/` |
| `_evaluate_model_artifact()` | `dict[str, list[PredictionFrame]]` | Persists artifact timestamp in config via `add_config` |
| `_evaluate_model_artifact_streaming()` | `None`; each origin goes to `origin_sink` | Persists artifact timestamp; logs origin count and peak RSS |
| `_forecast_model_artifact()` | `dict[str, PredictionFrame]` | Persists artifact timestamp in config via `add_config` |
| `_setup_model_and_data()` | `(model, df_source)` | None |
| `_iter_predictions()` | generator of `(int, dict[str, PredictionFrame])` | None |
| `_generate_predictions()` | `dict[str, list[PredictionFrame]]` | None |
| `_evaluate_sweep()` | `dict[str, list[PredictionFrame]]` | None (reads disk) |

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| `config["algorithm"]` unknown to catalog | `ValueError` from `BaselineModelCatalog.get_model()` | Descriptive error listing available names. |
| Required config key missing for the algorithm | `ValueError` from catalog | Lists missing keys. |
| Cached data path not set (data fetching not run) | `RuntimeError` from `_get_cached_data_path()` | Raised if `_execute_data_fetching()` has not run before a lifecycle method accesses data. Message: "No cached data path available." |
| Frames-declared model with no populated frame cache | `RuntimeError` from `_get_cached_frame_path()` | `_load_source()` on the `feature_frame` branch uses the loud getter — a lost/absent frame cache fails loud, never silently falling back to pandas (C-214). |
| Data file not found on disk | `FileNotFoundError` from `read_dataframe` (DataFrame) / from `load_frame_cache` (FeatureFrame) | No explicit error handling in manager; both readers are fail-loud upstream. |
| Config missing core or algorithm HP keys | `MissingHyperparameterError` (crash) | Raised by `ReproducibilityGate.Config.audit_manifest()` before catalog construction. |
| `config` missing `"run_type"`, `"level"`, or `"algorithm"` | `KeyError` (crash) | No explicit validation (algorithm absence is caught by the gate with "Missing required key: 'algorithm'"). |
| Model `fit()` raises | Propagates to caller | No wrapping. |
| No artifact `.pkl` for the run type on disk | `FileNotFoundError` from `get_latest_model_artifact_path()` | Raised in `_evaluate_model_artifact` and `_forecast_model_artifact` before setup. New precondition introduced by ADR-016 fix. |
| `artifact_name` points to non-existent file | No error at resolve time | The path is constructed but not checked for existence; the timestamp is still extracted from the filename stem. Downstream failure occurs only if the artifact is loaded (baseline models don't load artifacts, so this is benign). |
| Distributional code path in `_generate_predictions` or `_forecast_model_artifact` called with novel model | No error | Now tested; see Test Alignment. |

The manager emits `logger.info()` messages at lifecycle boundaries: initialisation, artifact save, model type selection, evaluation entry (batch and streaming), prediction generation (batch and streaming), the streaming summary (origin count and peak RSS), and forecast entry (9 messages total in `baseline_manager.py`). Model-level logging (INFO, WARNING) also propagates from the model classes through the module logger hierarchy.

---

## Boundaries and Interactions

**Inherits from:** `views_pipeline_core.managers.model.ForecastingModelManager`

**Imports from `views_pipeline_core`:**
| Symbol | Purpose |
|---|---|
| `generate_model_file_name` | Generates timestamped artifact filename. |
| `read_dataframe` | Reads the viewser DataFrame from disk (legacy `dataframe` path in `_load_source`). |
| `load_frame_cache` | Reads the `FeatureFrame` directory cache from disk (`feature_frame` path in `_load_source`, issue #64). |
| `DATA_FORMAT_DATAFRAME`, `DATA_FORMAT_FEATURE_FRAME` | The declared-format literals `_load_source` dispatches on. |
| `ForecastingModelManager` | Base class providing `config` property, `_config_manager`, `_data_loader`, `_resolve_evaluation_sequence_number`, etc. |
| `ModelPathManager` | Type of the `model_path` constructor argument. |
| `ConfigurationManager` | Type used in tests to construct `_config_manager`. |

**Imports from project:**
| Symbol | Purpose |
|---|---|
| `ReproducibilityGate` | Config validation gate — `audit_manifest()` called in `_setup_model_and_data()`. |
| `BaselineModelCatalog` | Model factory. |
| `DistributionalBaselineModel` | Protocol marking distributional models (`distributional: bool`) — semantic classification, not output dispatch. |

**Does not depend on** the frame builders (`build_prediction_frame` / `sample_prediction_grid` in `model/frames/output.py`) or any individual model class directly.

---

## Examples of Correct Usage

The manager is not typically instantiated directly outside tests. Pipeline integration:

```python
from views_baseline.manager.baseline_manager import BaselineForecastingModelManager

manager = BaselineForecastingModelManager(model_path=model_path_manager)
manager.config = {
    "run_type": "calibration",
    "level": "pg_id",
    "algorithm": "LocfModel",
    "regression_targets": ["y1", "y2"],
    # CORE_GENOME keys — audit_manifest raises before the catalog is reached without them
    "steps": [*range(1, 37)],
    "time_steps": 36,
    "prediction_format": "prediction_frame",
}
# Training
model = manager._train_model_artifact()   # fits, pickles, returns model

# Evaluation
preds = manager._evaluate_model_artifact(eval_type="temporal")
# preds is dict[str, list[PredictionFrame]] for all models (point and distributional)

# Forecasting
forecast = manager._forecast_model_artifact()
# forecast is dict[str, PredictionFrame] for all models
```

Testing pattern (from `test_baseline_manager.py`):

```python
mgr = BaselineForecastingModelManager.__new__(BaselineForecastingModelManager)
mgr._config_manager = ConfigurationManager(...)
mgr._sweep = False
mgr.config = config
mgr._model_path = SimpleNamespace(
    artifacts=Path("dummy_artifacts_path"),
    get_latest_model_artifact_path=lambda run_type: Path(
        f"dummy_artifacts_path/{run_type}_model_20260101_120000.pkl"
    ),
)
mgr._data_loader = SimpleNamespace(partition_dict=partition_dict)
mgr._cached_data_path = Path("dummy_raw_path") / "cached_df.parquet"
mgr._resolve_evaluation_sequence_number = lambda eval_type: config.get("sequence_numbers", 1)
monkeypatch.setattr(bm, "read_dataframe", lambda path: base_df)
preds = mgr._evaluate_model_artifact(eval_type="temporal")
```

---

## Examples of Incorrect Usage

```python
# Using an algorithm name not registered in the catalog
manager.config = {
    "run_type": "eval", "level": "pg_id", "algorithm": "SVR",
    "regression_targets": ["y1"], "steps": [1], "time_steps": 36,
    "prediction_format": "prediction_frame",
}
manager._evaluate_model_artifact(eval_type="temporal")
# MissingHyperparameterError: "Unknown algorithm 'SVR'. Available: [...]"
# (the GATE rejects it first — the catalog's own ValueError is unreachable in a
#  pipeline run, though still raised on the direct-API path)

# Missing required config key
manager.config = {
    "run_type": "eval", "level": "pg_id", "algorithm": "AverageModel",
    "regression_targets": ["y1"], "steps": [1], "time_steps": 36,
    "prediction_format": "prediction_frame",
}
# Missing "window_months"
manager._evaluate_model_artifact(eval_type="temporal")
# MissingHyperparameterError: "Algorithm 'AverageModel' requires missing parameters: ['window_months']"

# Indexing _forecast_model_artifact by position (it returns dict[str, PredictionFrame], keyed by target)
forecast = manager._forecast_model_artifact()
forecast[0]   # KeyError — index by target key (e.g. forecast["y1"]), not by integer position
```

---

## Test Alignment

File: `tests/test_baseline_manager.py`

| Test | What it verifies |
|---|---|
| `test_manager_evaluate_uses_zero_model` | `_evaluate_model_artifact` with `ZeroModel` returns a list of 2 DataFrames that are identical to what `ZeroModel` produces directly for `sequence_number=0` and `sequence_number=1`. |
| `test_manager_evaluate_uses_locf_model` | Same structure for `LocfModel`; confirms `config["algorithm"]` dispatch is respected. |
| `test_manager_forecast_uses_baseline_model` | `_forecast_model_artifact` with `LocfModel` returns a DataFrame equal to direct `LocfModel.predict(sequence_number=0)`. |
| `test_manager_forecast_respects_algorithm_choice` | `ZeroModel` and `LocfModel` produce different (non-equal) forecasts on the same data. |
| `test_manager_setup_returns_model_and_data` | `_setup_model_and_data()` returns `(ZeroModel instance, DataFrame)` with correct shapes. |
| `test_manager_train_saves_artifact` | Pickle file is written to `artifacts/`, filename matches pattern `calibration_model_{YYYYMMDD}_{HHMMSS}.pkl`, and the unpickled object is a valid `ZeroModel`. |
| `test_manager_evaluate_distributional_model` | `_evaluate_model_artifact` with `ConflictologyModel` returns `dict[str, list[PredictionFrame]]` with correct keys and list length matching `sequence_numbers`. |
| `test_manager_forecast_distributional_model` | `_forecast_model_artifact` with `ConflictologyModel` returns `dict[str, PredictionFrame]` with correct keys and types. |
| `test_manager_evaluate_sweep` | `_evaluate_sweep` loads data, delegates to `_generate_predictions`, and returns a list of DataFrames identical to direct model output. |
| `test_manager_setup_flows_feature_frame_from_cache` | A `data_format: feature_frame` model routes `_setup_model_and_data` through `load_frame_cache`; a `FeatureFrame` (not a DataFrame) flows into `fit`, and `read_dataframe` is never called (issue #64). |
| `test_manager_forecast_via_frame_cache` | End-to-end `_forecast_model_artifact` from the frame cache yields `PredictionFrame`s with no pandas. |
| `test_manager_evaluate_sweep_via_frame_cache` | The sweep read-site also serves the frame cache with no pandas. |
| `test_manager_frame_declared_missing_cache_fails_loud` | A frames-declared model with an absent cache raises `RuntimeError` — never a silent pandas fallback (C-214). |
| `test_manager_legacy_format_never_calls_frame_loader` | An explicitly `dataframe`-declared model uses `read_dataframe` and never the frame loader. |
| `test_manager_default_format_is_dataframe` | With no `_data_format` set, the manager defaults to the pandas path (byte-identical legacy behavior). |
| `test_streaming_hands_each_origin_to_the_sink_before_predicting_the_next` | Origin i reaches the sink before origin i+1 is predicted: nothing is accumulated and emitted later (#120). |
| `test_streaming_never_calls_the_batch_path` | The streaming path never builds the batch dict. |
| `test_streaming_releases_each_origin_after_the_sink` | Once a sink drops an origin's PredictionFrames, no reference to them survives in the manager. |
| `test_streaming_output_equals_batch_output` | Streamed output equals batch output exactly, origin by origin, so results are comparable across the change. |
| `test_streaming_logs_origin_count_and_peak_rss` | One log line with the origin count and peak RSS. |
| `test_manager_gate_rejects_incomplete_config` | `_setup_model_and_data()` raises `MissingHyperparameterError` when core keys are missing. (In `test_reproducibility_gate.py`.) |

File: `tests/test_falsification_timestamp_contract.py`

| Test | What it verifies |
|---|---|
| `test_evaluate_persists_artifact_timestamp_in_config` | `_evaluate_model_artifact` extracts the artifact timestamp and persists it in config via `add_config`. |
| `test_forecast_persists_artifact_timestamp_in_config` | `_forecast_model_artifact` extracts the artifact timestamp and persists it in config via `add_config`. |
| `test_timestamp_is_not_datetime_now` | The persisted timestamp comes from the artifact stem, not `datetime.now()`. |

File: `tests/test_falsification_merge_regression.py`

| Test | What it verifies |
|---|---|
| `test_setup_model_and_data_does_not_stamp_timestamp` | `_setup_model_and_data` no longer mutates `config["timestamp"]` (CIC drift guard). |
| `test_evaluate_raises_when_no_artifact_exists` | `_evaluate_model_artifact` raises `FileNotFoundError` when no artifact `.pkl` exists on disk. |
| `test_forecast_raises_when_no_artifact_exists` | `_forecast_model_artifact` raises `FileNotFoundError` when no artifact `.pkl` exists on disk. |

File: `tests/test_falsification_ship_readiness.py`

| Test | What it verifies |
|---|---|
| `test_uses_specified_artifact_timestamp[_evaluate_model_artifact]` | When `artifact_name` is provided, `_evaluate_model_artifact` extracts timestamp from the specified artifact, not the latest. |
| `test_uses_specified_artifact_timestamp[_forecast_model_artifact]` | When `artifact_name` is provided, `_forecast_model_artifact` extracts timestamp from the specified artifact, not the latest. |

---

## Evolution Notes

- The distributional code paths in `_generate_predictions` and `_forecast_model_artifact` are now covered by `test_manager_evaluate_distributional_model` and `test_manager_forecast_distributional_model`.
- `_evaluate_sweep` is now covered by `test_manager_evaluate_sweep`.
- **#120 (2026-11-05):** evaluation streams one origin at a time (`_evaluate_model_artifact_streaming`). Verified on the pod on all ten views-models#695 models at S = 64, global `land`: all exited 0. RSS was 2.3 GiB during prediction and 11.3 GiB later, in pipeline-core's scoring step (sampled with `ps`, not a traced peak). The scoring reload is the next memory ceiling and is pipeline-core's (C-54).
- Manager log messages are currently all `INFO` level. If `WARNING` or `ERROR` messages are added, they should follow the two-tier observability pattern (ADR-008).

---

## Known Deviations

None.
