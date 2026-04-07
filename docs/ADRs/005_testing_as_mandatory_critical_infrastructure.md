# ADR-005: Testing as Mandatory Critical Infrastructure

- **Status:** Accepted
- **Date:** 2026-03-17
- **Deciders:** Project maintainers

---

## Context

views-baseline produces forecasts that feed into the VIEWS conflict prediction pipeline. Errors in baseline predictions propagate silently downstream: a miscalculated entity average or a broken distributional sample shape will not raise an exception at the pipeline boundary — it will produce wrong numbers. Because baselines serve as the correctness reference against which more complex models are benchmarked, an incorrect baseline is worse than no baseline.

The project is also small enough that informal verification is tempting. The five model classes fit in a single file; the manager is a thin wrapper around a base class. Without a forcing function, testing would be treated as optional and deferred.

This ADR records the decision that testing is not optional infrastructure but a first-class deliverable, defines the three-team testing model used in this project, and documents the current coverage state honestly, including known gaps.

---

## Decision

All non-trivial logic in views-baseline must be covered by automated tests that run in CI on every push and pull request. Tests are categorised into three teams with different mandates. The test suite is the primary mechanism for verifying that the codebase does what it claims to do.

---

## Three-Team Testing Model

### Green Team — Stability and Correctness

Green team tests verify that each model produces the output it is specified to produce, under normal conditions, for the inputs it was designed to handle.

**What is covered:**

`tests/test_baseline.py` (36 tests) covers all five model classes:

- `ZeroModel`: verified to produce all-zero DataFrames at both `pg_id` and `country_id` levels of analysis; `sequence_number` offset is verified to shift the time index by the correct number of steps.
- `LocfModel`: verified to carry the last training-period observation forward into every forecast step; explicitly tests that fit on unsorted time data still selects the temporally last value, not the positionally last value.
- `AverageModel`: verified to compute per-entity means over the correct `window_months` window; `sequence_number` offset verified.
- `ConflictologyModel`: verified to return a `dict[str, PredictionFrame]` with shape `(n_entities * output_length, n_samples)`; verified that every sampled value for a given entity is drawn exclusively from that entity's history window; reproducibility under identical seeds verified.
- `MixtureBaseline`: verified local pool extraction, global pool positivity and causality, `lambda_mix=0.0` produces only local samples, `lambda_mix=1.0` causes zero-history entities to receive positive samples from the global pool, reproducibility under identical seeds.
- `build_prediction_grid` helper: shape, column names, and empty-input edge case tested directly.

`tests/test_catalog.py` (9 tests) covers `BaselineModelCatalog`:

- All five model names return correctly typed and correctly parameterised instances (including `MixtureBaseline` with all three required params).
- Unknown model name raises `ValueError` with the name in the message.
- Missing required config key raises `ValueError` naming the key (tested for `ConflictologyModel` and `MixtureBaseline`).

`tests/test_protocol.py` (10 tests) covers protocol conformance:

- All five model classes satisfy `isinstance(model, BaselineModel)`.
- `ConflictologyModel` and `MixtureBaseline` satisfy `isinstance(model, DistributionalBaselineModel)`.
- `ZeroModel`, `LocfModel`, and `AverageModel` do not satisfy `DistributionalBaselineModel` (negative conformance).

`tests/test_helpers.py` (5 tests) covers shared helper functions:

- `build_time_grid`: basic grid and offset verification.
- `filter_entities`: no-drop and drop-with-warning cases (warning emission verified via `caplog`).
- `build_identifier_arrays`: entity→time ordering per ADR-011.

`tests/test_reproducibility_gate.py` (7 green team tests) covers `ReproducibilityGate`:

- `CORE_GENOME` and `ALGORITHM_GENOMES` structural sanity (non-empty, all strings, all 5 models registered).
- `audit_manifest()` accepts valid configs for both minimal (ZeroModel) and maximal (MixtureBaseline) cases.
- `audit_manifest()` rejects missing core keys, missing algorithm-specific keys, and unknown algorithm names.

These tests form the CI backbone. A green build on this set is the minimum bar for merging any change.

### Beige Team — Realistic Misuse

Beige team tests exercise the code from the outside, as a downstream caller would, including dependency injection and integration across module boundaries. They are permitted to monkeypatch, but they patch at realistic seams rather than disabling the code under test.

**What is covered:**

`tests/test_baseline_manager.py` (8 tests) covers `BaselineForecastingModelManager` with monkeypatched file I/O (`read_dataframe`) and a manually constructed manager that bypasses disk and `wandb`:

- `_evaluate_model_artifact` with `ZeroModel`: returns a list of prediction DataFrames of the correct length, with values matching a directly instantiated ZeroModel.
- `_evaluate_model_artifact` with `LocfModel`: same contract, verifying that `config["algorithm"]` is respected.
- `_forecast_model_artifact` with `LocfModel`: returns a single DataFrame matching direct model output.
- `_forecast_model_artifact` algorithm switching: confirms ZeroModel and LocfModel produce different results on non-degenerate data.
- `_setup_model_and_data`: returns the correct model type and a DataFrame with the correct shape.
- `_train_model_artifact`: writes a `.pkl` file to the artifacts directory with a name matching `calibration_model_YYYYMMDD_HHMMSS.pkl`; the unpickled object is a valid fitted `ZeroModel` with correct `targets`.
- `_evaluate_model_artifact` with `ConflictologyModel`: returns `dict[str, list[PredictionFrame]]` with correct keys and list length.
- `_forecast_model_artifact` with `ConflictologyModel`: returns `dict[str, PredictionFrame]` with correct keys and types.

`tests/test_baseline.py` (2 beige team tests) captures entity-drop warnings:

- `test_locf_entity_drop_warning`: injects an unknown entity at `train_end` and asserts `WARNING` log via `caplog`.
- `test_average_entity_drop_warning`: same pattern for `AverageModel`.

`test_conflictology_matches_mixture_lambda_zero` (in `test_baseline.py`) is a cross-model equivalence test: it verifies that `ConflictologyModel` and `MixtureBaseline(lambda_mix=0.0)` populate their respective history pools with identical values for each entity and target, confirming the two models share a common conceptual base.

`tests/test_reproducibility_gate.py` (3 beige team tests) covers cross-module integration:

- `test_gate_genomes_match_catalog_genomes`: identity check that `BaselineModelCatalog.MODEL_GENOMES` is the same object as `ReproducibilityGate.Config.ALGORITHM_GENOMES`.
- `test_manager_gate_rejects_incomplete_config`: end-to-end — the manager raises `MissingHyperparameterError` when core keys are missing from config.
- `test_downstream_import_contract`: the gate is importable and exposes `Config`, `CORE_GENOME`, `ALGORITHM_GENOMES`, and `audit_manifest`.

### Red Team — Adversarial Inputs

Red team tests probe behaviour under degenerate or hostile inputs that a caller could plausibly provide.

**What is covered:**

`tests/test_baseline.py` (5 tests) probes degenerate and hostile inputs:

- `test_average_model_window_months_zero_produces_nan`: `window_months=0` silently produces all-NaN predictions.
- `test_conflictology_window_months_zero_raises`: `window_months=0` raises `KeyError` during `fit()`.
- `test_mixture_window_months_zero_raises`: `window_months=0` raises `ValueError` during `predict()`.
- `test_conflictology_n_samples_zero_raises`: `n_samples=0` raises `ValueError` from `PredictionFrame` validation.
- `test_predict_before_fit_raises`: `predict()` before `fit()` crashes (`AttributeError`/`TypeError`/`KeyError`).

`tests/test_reproducibility_gate.py` (3 red team tests) probes adversarial config inputs:

- `test_none_value_injection`: required key present but set to `None` raises `MissingHyperparameterError`.
- `test_empty_string_algorithm`: empty-string algorithm raises `MissingHyperparameterError`.
- `test_extra_keys_ignored`: surplus keys in config do not cause errors.

**Still untested:**

- `output_length=0` — empty time list; model returns empty DataFrame without error or warning.
- `partition_dict` with `test[0]` beyond the input DataFrame — empty entity list; silently empty predictions.
- `targets` referencing columns not in the DataFrame — `KeyError` at fit time; no test verifies the error.
- Single-level (non-MultiIndex) DataFrame — `IndexError`; untested.

---

## Coverage Gaps (Known and Documented)

The following areas have no test coverage as of 2026-03-17:

1. **`_evaluate_sweep`:** The method that loads data and calls `_generate_predictions` for a WandB sweep iteration has no test coverage at all.

2. **`output_length` variety:** Most tests use `output_length=36` or a small explicit value in the 4–5 range. No test checks behaviour at a semantically different value (e.g., `output_length=1`, `output_length=72`).

3. **Single-target distributional models:** Most tests use the `targets` fixture which provides two targets (`["y1", "y2"]`). No test exercises the single-target case for distributional models to verify `dict` key cardinality.

---

## Consequences

**Positive:**

- Regressions in model arithmetic are caught before merge.
- Protocol conformance is verified structurally, not just by documentation.
- Manager integration is tested without requiring live pipeline dependencies.
- The three-team framing creates a shared vocabulary for discussing what is and is not tested.

**Negative / Risks:**

- Remaining red team gaps mean untested edge cases (empty `output_length`, out-of-range partition, wrong index levels) could still produce empty or NaN output silently.

**Accepted debt:**

The gaps listed above are accepted as known technical debt, not oversights. They are recorded here so they can be prioritised deliberately rather than discovered in production.
