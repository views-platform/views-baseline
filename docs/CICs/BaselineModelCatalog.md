# Class Intent Contract: BaselineModelCatalog

**Date:** 2026-03-13
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-001, ADR-006, ADR-009

---

## Purpose

`BaselineModelCatalog` is a factory that translates a flat configuration dictionary into a fully instantiated (but unfitted) baseline model object. It centralises the mapping from model names to constructor calls, performs genome-based validation of required configuration keys before instantiation, and provides a single authoritative registry of available baseline models. This separates the concerns of "which model to build" from both the model classes themselves and the manager that drives the training pipeline.

---

## Non-Goals

- Does not fit any model. The returned object is always unfitted.
- Does not own the model lifecycle (training, evaluation, forecasting). That is `BaselineForecastingModelManager`'s responsibility.
- Does not validate the semantic correctness of config values (e.g., whether `months` is positive). It validates only key presence, not value validity.
- Does not cache or reuse model instances between calls.

---

## Responsibilities and Guarantees

- **`__init__(config, partition_dict, loa)`**: Stores all three arguments. Constructs the `self.models` dispatch dict mapping model name strings to private factory methods.
- **`get_model(model_name)`**:
  1. Raises `ValueError` if `model_name` is not in `self.models`, with a message listing available names.
  2. Raises `ValueError` listing all missing required keys if any key from `MODEL_GENOMES[model_name]` is absent from `self.config`.
  3. Calls the appropriate factory method and returns the instantiated model.
- **`list_models()`**: Returns a list of all registered model name strings.
- **`MODEL_GENOMES`** is a class-level dict mapping each model name to the list of config keys that must be present for that model. All model-specific parameters are required and must be explicitly declared in the model's `config_hyperparameters.py` in views-models. No defaults are applied by factory methods.

**Genome contents:**

| Model | Required keys |
|---|---|
| `ZeroModel` | (none) |
| `LocfModel` | (none) |
| `AverageModel` | `"window_months"` |
| `ConflictologyModel` | `"window_months"`, `"n_samples"` |
| `MixtureBaseline` | `"window_months"`, `"lambda_mix"`, `"n_samples"` |

---

## Inputs and Assumptions

| Parameter | Type | Notes |
|---|---|---|
| `config` | `dict` | Must contain `"targets"` (List[str]) for all models. Model-specific keys per `MODEL_GENOMES`. |
| `partition_dict` | `dict` | Passed directly to each model constructor. Must contain `"test"` key. |
| `loa` | `str` | Level-of-analysis string. Passed directly to each model constructor. |

The catalog stores a reference to `config` (not a copy). Mutations to `config` after catalog construction will be visible in subsequent `get_model()` calls.

---

## Outputs and Side Effects

- **`get_model()`**: Returns an unfitted model instance. Raises `ValueError` on validation failure. No side effects.
- **`list_models()`**: Returns `list[str]`. No side effects.
- No logging. No I/O.

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| Unknown `model_name` | `ValueError` with message listing available names | `"Model '{model_name}' is not in the catalog. Available: ..."` |
| Required config key missing | `ValueError` listing missing keys | `"Model '{model_name}' requires config keys {missing} but they are missing"` |
| `config` missing `"targets"` | `KeyError` (crash) | No explicit validation; `targets=self.config["targets"]` raises. |
| `partition_dict` missing `"test"` | Deferred to model | Not validated in catalog; surfaces when the model calls `partition_dict["test"]`. |

---

## Boundaries and Interactions

- **Imports:** `ZeroModel`, `LocfModel`, `AverageModel`, `ConflictologyModel`, `MixtureBaseline` from `views_baseline.model.baseline`.
- **Imported by:** `BaselineForecastingModelManager._setup_model_and_data()`.
- **No external runtime dependencies** beyond the five model classes.
- The config key `"window_months"` maps directly to the constructor parameter `window_months` for all models that use it.

---

## Examples of Correct Usage

```python
from views_baseline.model.catalog import BaselineModelCatalog

partition_dict = {"test": (493, 528)}

# ZeroModel — no model-specific keys required
config = {"targets": ["y1", "y2"]}
catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa="pg_id")
model = catalog.get_model("ZeroModel")

# AverageModel — requires "window_months"
config = {"targets": ["y1", "y2"], "window_months": 6}
catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa="pg_id")
model = catalog.get_model("AverageModel")
assert model.window_months == 6

# MixtureBaseline — all model-specific params required
config = {"targets": ["y1"], "window_months": 18, "lambda_mix": 0.05, "n_samples": 256}
catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa="pg_id")
model = catalog.get_model("MixtureBaseline")
assert model.window_months == 18
assert model.lambda_mix == 0.05

# List available models
names = catalog.list_models()
assert "ConflictologyModel" in names
```

---

## Examples of Incorrect Usage

```python
# Unknown model name
catalog.get_model("BayesianModel")
# ValueError: "Model 'BayesianModel' is not in the catalog. Available: ..."

# Missing required key for AverageModel
config = {"targets": ["y1"]}  # no "window_months"
catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa="pg_id")
catalog.get_model("AverageModel")
# ValueError: "Model 'AverageModel' requires config keys ['window_months'] but they are missing"

# Missing "targets" in config (not in MODEL_GENOMES — crashes rather than raises ValueError)
config = {"window_months": 6}
catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa="pg_id")
catalog.get_model("AverageModel")   # KeyError: "targets" inside factory method

# Mutating config after construction affects subsequent get_model calls
config = {"targets": ["y1"]}
catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa="pg_id")
config["targets"] = []   # mutation visible inside catalog
model = catalog.get_model("ZeroModel")
assert model.targets == []   # unexpected empty targets
```

---

## Test Alignment

Files: `tests/test_catalog.py` (7 tests) and `tests/test_baseline.py` (5 tests)

| Test | File | What it verifies |
|---|---|---|
| `test_catalog_lists_all_models` | `test_catalog.py` | `list_models()` returns all 5 expected names. |
| `test_catalog_returns_zero_model` | `test_catalog.py` | `get_model("ZeroModel")` returns correct type; `targets`, `partition_dict`, `loa` are wired correctly. |
| `test_catalog_returns_locf_model` | `test_catalog.py` | Same structural check for `LocfModel`. |
| `test_catalog_returns_average_model` | `test_catalog.py` | `window_months` wired from `config["window_months"]`. |
| `test_catalog_returns_conflictology_model` | `test_catalog.py` | `window_months` wired from `config["window_months"]`. |
| `test_catalog_raises_for_unknown_model` | `test_catalog.py` | `ValueError` with model name in message. |
| `test_catalog_returns_mixture_model` | `test_catalog.py` | All three required params correctly wired from config. |
| `test_catalog_missing_n_samples_for_conflictology_raises` | `test_catalog.py` | `ValueError` when `n_samples` missing for `ConflictologyModel`. |
| `test_catalog_missing_keys_for_mixture_raises` | `test_catalog.py` | `ValueError` when `window_months` missing for `MixtureBaseline`. |
| `test_catalog_get_zero_model` | `test_baseline.py` | `get_model("ZeroModel")` returns `ZeroModel` instance. |
| `test_catalog_get_average_model` | `test_baseline.py` | `window_months` wired from `config["window_months"]`. |
| `test_catalog_unknown_model_raises` | `test_baseline.py` | Error message contains the unknown name. |
| `test_catalog_missing_required_key_raises` | `test_baseline.py` | `"window_months"` appears in error message when missing for `AverageModel`. |
| `test_catalog_list_models` | `test_baseline.py` | Set of returned names matches expected 5 models. |

---

## Evolution Notes

- To add a new model, register it in both `MODEL_GENOMES` and `self.models`, and add a `_get_*` factory method. All model-specific parameters must be listed in `MODEL_GENOMES` — the catalog does not apply defaults.
- If `"targets"` should also be validated (currently not in genomes), it can be added as a universal required key checked outside the model-specific genome logic.

---

## Known Deviations

- The `"targets"` key is universally required for all models but is not listed in any `MODEL_GENOMES` entry. A missing `"targets"` key surfaces as a `KeyError` inside the factory method rather than a descriptive `ValueError` from the genome check.
- The catalog stores a reference to `config`, not a copy. Post-construction mutations to the dict affect factory behaviour.
