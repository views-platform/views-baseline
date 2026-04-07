# Class Intent Contract: ReproducibilityGate

**Date:** 2026-04-07
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-009, ADR-014

---

## Purpose

`ReproducibilityGate` is the canonical definition of which hyperparameters each baseline algorithm requires. It centralises two contracts — `CORE_GENOME` (keys required by all models) and `ALGORITHM_GENOMES` (keys required per algorithm) — in a single importable location. The `audit_manifest()` method enforces these contracts at runtime by raising `MissingHyperparameterError` on any violation.

The class is importable by downstream packages (e.g. views-models) so they can validate their `config_hyperparameters.py` files in CI without running the full pipeline.

---

## Non-Goals

- Does not validate hyperparameter *values* (types, ranges, semantics). Only validates *presence* and *non-None*.
- Does not implement temporal or data integrity gates. Baseline models have no TimeSeries objects, no optimizers, and no loss functions.
- Does not own model instantiation. That remains with `BaselineModelCatalog`.

---

## Responsibilities and Guarantees

**`Config.CORE_GENOME`**: Class attribute. List of config keys required by every baseline model: `["steps", "time_steps"]`.

**`Config.ALGORITHM_GENOMES`**: Class attribute. Dict mapping each algorithm name to its list of additional required config keys. This dict is the single source of truth; `BaselineModelCatalog.MODEL_GENOMES` is an alias to it.

**`Config.audit_manifest(config)`**: Static method. Validates a config dict in four sequential checks:
1. All `CORE_GENOME` keys are present in `config`.
2. `config["algorithm"]` is a key in `ALGORITHM_GENOMES`.
3. All algorithm-specific keys for the declared algorithm are present.
4. No required key (core + algorithm-specific) has value `None`.

If any check fails, raises `MissingHyperparameterError` with a message identifying the missing or null keys. Logs the error at `ERROR` level before raising.

---

## Inputs and Assumptions

| Parameter | Type | Notes |
|---|---|---|
| `config` | `dict` | Must contain `"algorithm"` key. Expected to contain all keys declared in `CORE_GENOME` and the relevant `ALGORITHM_GENOMES` entry. |

---

## Outputs and Side Effects

- **`audit_manifest()`**: Returns `None` on success. Raises `MissingHyperparameterError` on failure. Emits `ERROR` log on failure. No other side effects.

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| Missing core key | `MissingHyperparameterError` (crash) | Lists all missing core keys. |
| Unknown algorithm | `MissingHyperparameterError` (crash) | Names the unknown algorithm and lists available ones. |
| Missing algorithm-specific key | `MissingHyperparameterError` (crash) | Lists missing keys for the declared algorithm. |
| Required key set to `None` | `MissingHyperparameterError` (crash) | Lists all `None`-valued required keys. |
| `config` missing `"algorithm"` key | `MissingHyperparameterError` (crash) | Explicit guard: `"Missing required key: 'algorithm'"`. |

All failures are loud and immediate. No silent fallbacks.

---

## Boundaries and Interactions

- **Depends on:** `views_baseline.infrastructure.exceptions.MissingHyperparameterError`.
- **No external dependencies** beyond standard library and logging.
- **Imported by:** `BaselineModelCatalog` (uses `ALGORITHM_GENOMES` as `MODEL_GENOMES`).
- **Imported by:** `BaselineForecastingModelManager` (calls `audit_manifest()` in `_setup_model_and_data()`).
- **Importable by:** `views-models` tests for static config validation.

---

## Examples of Correct Usage

```python
from views_baseline.infrastructure.reproducibility_gate import ReproducibilityGate

# Valid config — passes silently
config = {
    "algorithm": "AverageModel",
    "targets": ["y1"],
    "steps": [*range(1, 37)],
    "time_steps": 36,
    "window_months": 60,
}
ReproducibilityGate.Config.audit_manifest(config)  # no error

# Downstream validation in views-models
CORE_PARAMS = set(ReproducibilityGate.Config.CORE_GENOME)
ALGO_PARAMS = ReproducibilityGate.Config.ALGORITHM_GENOMES

hp = get_hp_config()  # from config_hyperparameters.py
missing = CORE_PARAMS - set(hp.keys())
assert not missing, f"Config missing core params: {missing}"
```

---

## Examples of Incorrect Usage

```python
# Missing core key — raises
config = {"algorithm": "ZeroModel", "targets": ["y1"], "time_steps": 36}
ReproducibilityGate.Config.audit_manifest(config)
# MissingHyperparameterError: Missing core parameters: ['steps']

# None value — raises
config = {"algorithm": "ZeroModel", "targets": ["y1"], "steps": None, "time_steps": 36}
ReproducibilityGate.Config.audit_manifest(config)
# MissingHyperparameterError: Mandatory parameters set to None: ['steps']

# Unknown algorithm — raises
config = {"algorithm": "SVR", "targets": ["y1"], "steps": [1], "time_steps": 36}
ReproducibilityGate.Config.audit_manifest(config)
# MissingHyperparameterError: Unknown algorithm 'SVR'. Available: [...]
```

---

## Test Alignment

File: `tests/test_reproducibility_gate.py`

| Test | What it verifies |
|---|---|
| `test_core_genome_is_list_of_strings` | `CORE_GENOME` is a non-empty list of strings. |
| `test_algorithm_genomes_covers_all_catalog_models` | All 5 model names are registered in `ALGORITHM_GENOMES`. |
| `test_audit_manifest_accepts_valid_zero_model_config` | Valid minimal config passes without error. |
| `test_audit_manifest_accepts_valid_mixture_config` | Valid config with all algorithm-specific keys passes. |
| `test_audit_manifest_rejects_missing_core_key` | Missing `steps` raises `MissingHyperparameterError`. |
| `test_audit_manifest_rejects_missing_algorithm_key` | Missing `window_months` for `AverageModel` raises. |
| `test_audit_manifest_rejects_unknown_algorithm` | Unknown algorithm name raises. |
| `test_gate_genomes_match_catalog_genomes` | `BaselineModelCatalog.MODEL_GENOMES` is the same object as `ALGORITHM_GENOMES`. |
| `test_manager_gate_rejects_incomplete_config` | End-to-end: manager rejects config missing core keys. |
| `test_downstream_import_contract` | Gate is importable and exposes expected attributes. |
| `test_none_value_injection` | Required key set to `None` raises. |
| `test_empty_string_algorithm` | Empty-string algorithm raises. |
| `test_extra_keys_ignored` | Surplus keys do not cause errors. |

---

## Evolution Notes

- If `Temporal` or `Data` gates are added for baseline models, they should follow the nested-class pattern (`ReproducibilityGate.Temporal`, `ReproducibilityGate.Data`) established in views-r2darts2.
- If hyperparameter *value* validation is needed (e.g. `window_months > 0`, `lambda_mix` in `[0, 1]`), it should be added as a separate `audit_values()` method, not mixed into `audit_manifest()`.
- The exception hierarchy currently contains only `ReproducibilityError` and `MissingHyperparameterError`. Additional exception types should be added only when a new gate type requires them.

---

## Known Deviations

None.
