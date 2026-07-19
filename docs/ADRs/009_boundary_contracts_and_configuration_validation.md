# ADR-009: Boundary Contracts and Configuration Validation

- **Status:** Accepted
- **Date:** 2026-03-13
- **Deciders:** Project maintainers

---

## Context

views-baseline sits between two external systems: the VIEWS pipeline configuration and data layer on its input side, and `views-pipeline-core` on its integration side. It also has internal boundaries between its own components: config flows through the catalog into model constructors, and data flows through model methods into the manager's prediction loop.

When assumptions about these boundaries are implicit, they fail silently at runtime rather than at construction time. An incorrect config key passes through `BaselineModelCatalog` construction without error; the error surfaces only when `get_model()` is called. A DataFrame with the wrong index structure passes through `fit()` without error; the error surfaces as an unexpected `KeyError` or `IndexError` deep in the prediction loop.

This ADR records the boundary contracts that exist, where they are enforced, and where they are assumed but not checked.

---

## Decision

Each module boundary is documented with an explicit contract. Where validation exists in code, it is described. Where validation is absent, the gap is recorded as accepted debt. No new runtime validation code is introduced by this ADR — it records the current state.

---

## Boundary 1: Config → Catalog

**Location:** `BaselineModelCatalog.__init__()` and `BaselineModelCatalog.get_model()`

**What is validated:**

`get_model(model_name)` checks two things before constructing a model:

1. That `model_name` is a key in `self.models`. If not, raises `ValueError` naming the unknown model and listing available options.
2. That every key in `MODEL_GENOMES[model_name]` is present in `self.config`. If not, raises `ValueError` naming the missing keys.

`MODEL_GENOMES` encodes the required keys per model:

```python
MODEL_GENOMES = {
    "ZeroModel":          [],
    "LocfModel":          [],
    "AverageModel":       ["window_months"],
    "ConflictologyModel": ["window_months", "n_samples"],
    "MixtureBaseline":    ["window_months", "lambda_mix", "n_samples"],
}
```

All model-specific parameters are required and must be explicitly declared in the model's `config_hyperparameters.py`. No defaults are applied by factory methods.

**What is not validated:**

- Value ranges: `window_months=0`, `window_months=-1`, `lambda_mix=2.0`, `n_samples=0` all pass validation.
- `targets` is required by all models but is not listed in `MODEL_GENOMES` (it is assumed to always be present). If `targets` is missing from config, construction of the model object will raise a `KeyError` from within the factory method, not from the validation check.
- `partition_dict` is passed directly to model constructors without any structural check. The expected structure is `{"test": (start, end)}` where `start` and `end` are integer month IDs, but this is not enforced.

---

## Boundary 2: Catalog → Model

**Location:** `BaselineModelCatalog._get_*` factory methods

**What is validated:**

Each factory method maps config keys to constructor parameters. The mapping is explicit and readable:

- `config["targets"]` → `targets=`
- `config["window_months"]` → `window_months=` (for `AverageModel`, `ConflictologyModel`, and `MixtureBaseline`)
- `config["lambda_mix"]` → `lambda_mix=` (for `MixtureBaseline`)
- `config["n_samples"]` → `n_samples=` (for `ConflictologyModel` and `MixtureBaseline`)

The model constructor receives fully resolved values. It does not receive the config dict and cannot be misconfigured by key aliasing errors.

**What is not validated:**

- Constructor parameters are not range-checked. `window_months=0` is passed directly to `AverageModel(window_months=0)` without error.
- Type is not enforced. `config["window_months"] = "18"` (a string) would be passed as `window_months="18"` and cause a `TypeError` only when `tail()` is called during fit.

---

## Boundary 3: Model → Pipeline (Protocol Dispatch)

**Location:** `BaselineForecastingModelManager._generate_predictions()` and `_forecast_model_artifact()`

**What is validated:**

The manager dispatches on model type using `isinstance(model, DistributionalBaselineModel)`:

```python
if isinstance(model, DistributionalBaselineModel):
    # returns dict[str, list[PredictionFrame]]
    ...
else:
    # returns list[pd.DataFrame]
    ...
```

`DistributionalBaselineModel` is a `@runtime_checkable` Protocol. The `isinstance` check passes if and only if the model object has a `distributional` attribute and implements the expected method signatures structurally. `ConflictologyModel` and `MixtureBaseline` both set `distributional = True` as a class attribute, which satisfies the protocol check.

This dispatch is the primary type-safety mechanism at the model-to-pipeline boundary. It ensures that the manager handles distributional and point-forecast outputs through different code paths.

**What is not validated:**

- The contents of the returned dict from distributional models are not validated. If a model returns a dict with the wrong `PredictionFrame` shape or unexpected keys, the manager passes it upstream without checking.
- The `predictions` structure returned from `_generate_predictions` is not validated before being returned to the caller. The caller (in `views-pipeline-core`) is assumed to handle the structure correctly.

---

## Boundary 4: Data → Model

**Location:** `fit()` and `predict()` methods of all seven model classes

**What is assumed (not validated):**

All seven models assume without checking:

1. The input `df` has a `MultiIndex` with at least two levels. `df.index.names[0]` is the time index, `df.index.names[1]` is the entity index.
2. The time index values are integers (month IDs).
3. `df.index.names[1]` matches `self.loa` (the level-of-analysis string). This is not checked; a mismatch would produce predictions with a misnamed entity index.
4. `self.partition_dict["test"][0]` is a valid integer that is present in or adjacent to the time index values in `df`. If `test_start` is past the end of the data, `entity_ids` at `train_end` is empty and predictions are silently empty.
5. The columns in `self.targets` are present in `df`. If a target is missing, the failure surface varies by model: `KeyError` at groupby time for `LocfModel` and `AverageModel`, `KeyError` during history extraction for `ConflictologyModel`, `KeyError` during pool construction for `MixtureBaseline`.

**No schema validation exists.** There is no `pandera`, `pydantic`, or manual index-structure check at the entry point of any model method.

---

## Boundary 5: Manager → views-pipeline-core

**Location:** `views_baseline/manager/baseline_manager.py`

**What is imported from views-pipeline-core:**

Four symbols are imported at module load time:

```python
from views_pipeline_core.files.utils import generate_model_file_name, read_dataframe
from views_pipeline_core.managers.model import ForecastingModelManager, ModelPathManager
```

One additional symbol is imported lazily inside methods:

```python
from views_pipeline_core.data.prediction_frame import PredictionFrame
```

`PredictionFrame` is imported lazily (inside `ConflictologyModel.predict()` and `MixtureBaseline.predict()`) to avoid a hard dependency on `views-pipeline-core` when only point-forecast models are used. This is a deliberate design choice.

**What is assumed:**

- `ForecastingModelManager` provides `_resolve_evaluation_sequence_number(eval_type)`, `_config_manager`, `_sweep`, `_get_cached_data_path()`, and `config` setter. These are used by `BaselineForecastingModelManager` but their contracts are defined in the base class, not in this package.
- `ModelPathManager` provides `.artifacts` attribute that is a `pathlib.Path`-like object. `.data_raw` is no longer accessed directly by the manager (data path comes from the base class `_cached_data_path`).
- `generate_model_file_name(run_type, file_extension=".pkl")` returns a filename string in the format `{run_type}_model_{timestamp}.pkl`.
- `partition_dict` is accessed via `self._data_loader.partition_dict`. The structure is assumed to match what model constructors expect (`{"test": (start, end)}`), but this is not checked.

**What is not validated:**

- The structure of `partition_dict` as returned by `self._data_loader` is not checked before being passed to `BaselineModelCatalog`.
- The file produced by `read_dataframe` is not validated before being passed to `model.fit()`.

---

## Summary of Known Gaps

| Boundary | What is missing |
|---|---|
| Config → Catalog | No validation that `targets` is present; no value-range validation for numeric params |
| Catalog → Model | No constructor-level range checks; no type coercion |
| Model → Pipeline | No validation of distributional output structure or shape |
| Data → Model | No MultiIndex structure validation; no column presence check; no time-range sanity check |
| Manager → core | `partition_dict` structure assumed; `read_dataframe` output assumed valid |

These gaps are accepted as known technical debt. They are documented here rather than fixed because adding schema validation at every boundary would add a meaningful maintenance surface to a package whose value proposition is simplicity.

---

## Consequences

**Positive:**

- Configuration errors that can be caught structurally (unknown model name, missing required key) fail immediately with clear messages.
- The catalog-to-model factory boundary is explicit and readable: every parameter mapping is a named argument in a factory method, not a `**kwargs` pass-through.
- Protocol dispatch at the model-to-pipeline boundary is type-checked at runtime using `isinstance`, ensuring that distributional and point-forecast outputs are handled through the correct code path.

**Negative / Risks:**

- Degenerate numeric parameters (`window_months=0`, `output_length=0`) are not caught at configuration time. They produce incorrect output without error or warning.
- Missing DataFrame columns produce `KeyError` at fit or predict time with no contextual message identifying which model, target, or boundary was involved.
- An incorrectly structured `partition_dict` (e.g., missing the `"test"` key) will raise a `KeyError` during model prediction, not during model construction or manager initialisation.
- The absence of input DataFrame validation means that a data loading error upstream (wrong file, wrong index) will surface as an obscure error message deep inside model logic rather than at the boundary where the bad data enters.
