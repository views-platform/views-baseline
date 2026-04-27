# Class Intent Contract: BaselineForecastingModelManager

**Date:** 2026-03-17
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-001, ADR-002, ADR-006, ADR-008, ADR-009, ADR-014

---

## Purpose

`BaselineForecastingModelManager` orchestrates the full lifecycle of a baseline model within the VIEWS pipeline by extending `views_pipeline_core`'s `ForecastingModelManager`. It is the single entry point for the pipeline to set up, train, evaluate, and forecast using any model registered in `BaselineModelCatalog`. It does not implement model logic itself; instead, it delegates model selection and validation to the catalog, delegates data loading to `_data_loader`, and delegates config management to `_config_manager`.

The manager also serves as the dispatch layer between point models (which return `list[pd.DataFrame]`) and distributional models (which return `dict[target, list[PredictionFrame]]`), routing output based on `isinstance(model, DistributionalBaselineModel)`.

---

## Non-Goals

- Does not implement any prediction algorithm. All model logic lives in the five model classes.
- Does not own data loading. Data is read from disk via `read_dataframe` (from `views_pipeline_core`) inside `_setup_model_and_data()`.
- Does not define the configuration schema. Config structure is the responsibility of the pipeline configuration layer.
- Does not sweep hyperparameters in any non-trivial way; `_evaluate_sweep` delegates entirely to `_generate_predictions`.

---

## Responsibilities and Guarantees

**`_setup_model_and_data()`**:
- Calls `ReproducibilityGate.Config.audit_manifest(self.config)` as a precondition — raises `MissingHyperparameterError` if core or algorithm-specific keys are missing or `None`.
- Reads `config["run_type"]`, `config["level"]`, and `_data_loader.partition_dict`.
- Instantiates `BaselineModelCatalog` and calls `catalog.get_model(config["algorithm"])`.
- Stamps `config["timestamp"]` with the current datetime.
- Reads the training DataFrame from disk via `self._get_cached_data_path()` (path set by the base class during data fetching).
- Calls `model.fit(df)`.
- Returns `(model, df)`.

**`_train_model_artifact()`**:
- Calls `_setup_model_and_data()`.
- Pickles the fitted model to `{path_artifacts}/{run_type}_model_{timestamp}.pkl` via `generate_model_file_name`.
- Returns the fitted model. The artifact is required by the downstream ensemble manager to resolve artifact paths.

**`_generate_predictions(model, df, eval_type)`**:
- Calls `_resolve_evaluation_sequence_number(eval_type)` to determine iteration count.
- If `isinstance(model, DistributionalBaselineModel)`: iterates over sequence numbers, calls `model.predict(df=df, sequence_number=seq_num)` for each, accumulates `dict[target, list[PredictionFrame]]`.
- Otherwise: iterates and accumulates `list[pd.DataFrame]`.

**`_evaluate_model_artifact(eval_type, artifact_name)`**:
- Calls `_setup_model_and_data()` then `_generate_predictions()`.
- Returns the accumulated predictions.

**`_forecast_model_artifact(artifact_name)`**:
- Calls `_setup_model_and_data()`.
- Calls `model.predict(df=df_source, sequence_number=0, output_length=output_length)` using keyword arguments. Both distributional and point paths use the same calling convention.
- Returns the prediction result directly (not wrapped in a list).

**`_evaluate_sweep(eval_type, model)`**:
- Reads the DataFrame from disk using the same path logic as `_setup_model_and_data()`.
- Calls `_generate_predictions(model, df, eval_type)`.
- Returns predictions.

---

## Inputs and Assumptions

| Source | What is consumed | Notes |
|---|---|---|
| `model_path` (constructor) | `ModelPathManager` instance | Provides `.artifacts` path. `.data_raw` is no longer accessed directly (data path comes from base class `_cached_data_path`). |
| `self.config` | `dict` | Set via property inherited from base class. Must contain `"run_type"`, `"level"`, `"algorithm"`, `"targets"`. |
| `_data_loader.partition_dict` | `dict` | Must contain `"test"` key. |
| `_config_manager` | `ConfigurationManager` | Used by base class; not directly accessed in overridden methods. |

The `config` dict is populated by the base class `_config_manager` before any lifecycle method is called.

---

## Outputs and Side Effects

| Method | Output type (point) | Output type (distributional) | Side effects |
|---|---|---|---|
| `_train_model_artifact()` | Fitted model instance | Fitted model instance | Writes `.pkl` file to `artifacts/` |
| `_evaluate_model_artifact()` | `list[pd.DataFrame]` | `dict[str, list[PredictionFrame]]` | None (reads disk) |
| `_forecast_model_artifact()` | `pd.DataFrame` | `dict[str, PredictionFrame]` | None (reads disk) |
| `_setup_model_and_data()` | `(model, df)` | `(model, df)` | Mutates `config["timestamp"]` |
| `_generate_predictions()` | `list[pd.DataFrame]` | `dict[str, list[PredictionFrame]]` | None |
| `_evaluate_sweep()` | `list[pd.DataFrame]` | `dict[str, list[PredictionFrame]]` | None (reads disk) |

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| `config["algorithm"]` unknown to catalog | `ValueError` from `BaselineModelCatalog.get_model()` | Descriptive error listing available names. |
| Required config key missing for the algorithm | `ValueError` from catalog | Lists missing keys. |
| Data file not found on disk | `FileNotFoundError` from `read_dataframe` | No explicit error handling in manager. |
| Config missing core or algorithm HP keys | `MissingHyperparameterError` (crash) | Raised by `ReproducibilityGate.Config.audit_manifest()` before catalog construction. |
| `config` missing `"run_type"`, `"level"`, or `"algorithm"` | `KeyError` (crash) | No explicit validation (algorithm absence is caught by the gate with "Missing required key: 'algorithm'"). |
| Model `fit()` raises | Propagates to caller | No wrapping. |
| Distributional code path in `_generate_predictions` or `_forecast_model_artifact` called with novel model | No error | Now tested; see Test Alignment. |

The manager emits `logger.info()` messages at lifecycle boundaries: initialisation, artifact save, model type selection, evaluation entry, prediction generation, and forecast entry (6 messages total in `baseline_manager.py`). Model-level logging (INFO, WARNING) also propagates from the model classes through the module logger hierarchy.

---

## Boundaries and Interactions

**Inherits from:** `views_pipeline_core.managers.model.ForecastingModelManager`

**Imports from `views_pipeline_core`:**
| Symbol | Purpose |
|---|---|
| `generate_model_file_name` | Generates timestamped artifact filename. |
| `read_dataframe` | Reads the viewser DataFrame from disk. |
| `ForecastingModelManager` | Base class providing `config` property, `_config_manager`, `_data_loader`, `_resolve_evaluation_sequence_number`, etc. |
| `ModelPathManager` | Type of the `model_path` constructor argument. |
| `ConfigurationManager` | Type used in tests to construct `_config_manager`. |

**Imports from project:**
| Symbol | Purpose |
|---|---|
| `ReproducibilityGate` | Config validation gate — `audit_manifest()` called in `_setup_model_and_data()`. |
| `BaselineModelCatalog` | Model factory. |
| `DistributionalBaselineModel` | Protocol used for `isinstance()` dispatch. |

**Does not depend on** `build_prediction_grid` or any individual model class directly.

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
    "targets": ["y1", "y2"],
}
# Training
model = manager._train_model_artifact()   # fits, pickles, returns model

# Evaluation
preds = manager._evaluate_model_artifact(eval_type="temporal")
# preds is list[pd.DataFrame] for point models

# Forecasting
forecast = manager._forecast_model_artifact()
# forecast is pd.DataFrame for point models
```

Testing pattern (from `test_baseline_manager.py`):

```python
mgr = BaselineForecastingModelManager.__new__(BaselineForecastingModelManager)
mgr._config_manager = ConfigurationManager(...)
mgr._sweep = False
mgr.config = config
mgr._model_path = SimpleNamespace(artifacts=...)
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
manager.config = {"run_type": "eval", "level": "pg_id", "algorithm": "SVR", "targets": ["y1"]}
manager._evaluate_model_artifact(eval_type="temporal")
# ValueError: "Model 'SVR' is not in the catalog. Available: ..."

# Missing required config key
manager.config = {"run_type": "eval", "level": "pg_id", "algorithm": "AverageModel", "targets": ["y1"]}
# Missing "window_months"
manager._evaluate_model_artifact(eval_type="temporal")
# ValueError: "Model 'AverageModel' requires config keys ['window_months'] but they are missing"

# Expecting a list from _forecast_model_artifact (it returns a single prediction, not a list)
forecast = manager._forecast_model_artifact()
forecast[0]   # TypeError if forecast is a DataFrame (point model)
              # or KeyError if accessing by integer on a dict (distributional model)
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
| `test_manager_gate_rejects_incomplete_config` | `_setup_model_and_data()` raises `MissingHyperparameterError` when core keys are missing. (In `test_reproducibility_gate.py`.) |

---

## Evolution Notes

- The distributional code paths in `_generate_predictions` and `_forecast_model_artifact` are now covered by `test_manager_evaluate_distributional_model` and `test_manager_forecast_distributional_model`.
- `_evaluate_sweep` is also untested. If WandB sweeps are used with baseline models, tests are needed.
- Manager log messages are currently all `INFO` level. If `WARNING` or `ERROR` messages are added, they should follow the two-tier observability pattern (ADR-008).

---

## Known Deviations

- **`_evaluate_sweep` is untested:** No test exercises this method.
- **`config["timestamp"]` mutation in `_setup_model_and_data()`:** The manager stamps the config dict with the current time as a side effect of setup. This is a mutation of shared state that callers may not expect.
