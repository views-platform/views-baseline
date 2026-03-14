# ADR-005: Testing as Mandatory Critical Infrastructure

- **Status:** Accepted
- **Date:** 2026-03-13
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

`tests/test_baseline.py` (28 tests) covers all five model classes:

- `ZeroModel`: verified to produce all-zero DataFrames at both `pg_id` and `country_id` levels of analysis; `sequence_number` offset is verified to shift the time index by the correct number of steps.
- `LocfModel`: verified to carry the last training-period observation forward into every forecast step; explicitly tests that fit on unsorted time data still selects the temporally last value, not the positionally last value.
- `AverageModel`: verified to compute per-entity means over the correct `window_months` window; `sequence_number` offset verified.
- `ConflictologyModel`: verified to return a `dict[str, PredictionFrame]` with shape `(n_entities * output_length, n_samples)`; verified that every sampled value for a given entity is drawn exclusively from that entity's history window.
- `MixtureBaseline`: verified local pool extraction, global pool positivity and causality, `lambda_mix=0.0` produces only local samples, `lambda_mix=1.0` causes zero-history entities to receive positive samples from the global pool, reproducibility under identical seeds.
- `build_prediction_grid` helper: shape, column names, and empty-input edge case tested directly.

`tests/test_catalog.py` (7 tests) covers `BaselineModelCatalog`:

- All five model names return correctly typed and correctly parameterised instances.
- Unknown model name raises `ValueError` with the name in the message.
- Missing required config key raises `ValueError` naming the key.

`tests/test_protocol.py` (10 tests) covers protocol conformance:

- All five model classes satisfy `isinstance(model, BaselineModel)`.
- `ConflictologyModel` and `MixtureBaseline` satisfy `isinstance(model, DistributionalBaselineModel)`.
- `ZeroModel`, `LocfModel`, and `AverageModel` do not satisfy `DistributionalBaselineModel` (negative conformance).

These tests form the CI backbone. A green build on this set is the minimum bar for merging any change.

### Beige Team — Realistic Misuse

Beige team tests exercise the code from the outside, as a downstream caller would, including dependency injection and integration across module boundaries. They are permitted to monkeypatch, but they patch at realistic seams rather than disabling the code under test.

**What is covered:**

`tests/test_baseline_manager.py` (6 tests) covers `BaselineForecastingModelManager` with monkeypatched file I/O (`read_dataframe`) and a manually constructed manager that bypasses disk and `wandb`:

- `_evaluate_model_artifact` with `ZeroModel`: returns a list of prediction DataFrames of the correct length, with values matching a directly instantiated ZeroModel.
- `_evaluate_model_artifact` with `LocfModel`: same contract, verifying that `config["algorithm"]` is respected.
- `_forecast_model_artifact` with `LocfModel`: returns a single DataFrame matching direct model output.
- `_forecast_model_artifact` algorithm switching: confirms ZeroModel and LocfModel produce different results on non-degenerate data.
- `_setup_model_and_data`: returns the correct model type and a DataFrame with the correct shape.
- `_train_model_artifact`: writes a `.pkl` file to the artifacts directory with a name matching `calibration_model_YYYYMMDD_HHMMSS.pkl`; the unpickled object is a valid fitted `ZeroModel` with correct `targets`.

`test_conflictology_matches_mixture_lambda_zero` (in `test_baseline.py`) is a cross-model equivalence test: it verifies that `ConflictologyModel` and `MixtureBaseline(lambda_mix=0.0)` populate their respective history pools with identical values for each entity and target, confirming the two models share a common conceptual base.

### Red Team — Adversarial Inputs

Red team tests probe behaviour under degenerate or hostile inputs that a caller could plausibly provide.

**Current state: absent.**

No red team tests exist in the current suite. The following adversarial cases are untested:

- `window_months=0` passed to `AverageModel` or `ConflictologyModel` — behaviour is undefined; `tail(0)` returns an empty DataFrame, producing NaN means that propagate silently into predictions.
- `output_length=0` — `range(start, start + 0)` produces an empty time list; the model returns an empty DataFrame without error or warning.
- A `partition_dict` with `test[0]` beyond the end of the input DataFrame — entity list at `train_end` is empty; predictions are silently empty.
- A config with `targets` referencing columns not present in the input DataFrame — will raise a `KeyError` at fit time, but no test documents this or verifies the error message.
- An input DataFrame with a single-level index (not a MultiIndex) — `df.index.names[1]` raises `IndexError`; untested.

---

## Coverage Gaps (Known and Documented)

The following areas have no test coverage as of 2026-03-13:

1. **Manager distributional path:** `_evaluate_model_artifact` and `_forecast_model_artifact` with a distributional model (`ConflictologyModel` or `MixtureBaseline`) are untested. The dispatch branch `isinstance(model, DistributionalBaselineModel)` in `_generate_predictions` has never been exercised by a test.

2. **`_evaluate_sweep`:** The method that loads data and calls `_generate_predictions` for a WandB sweep iteration has no test coverage at all.

3. **Entity-drop warning emission:** `LocfModel`, `AverageModel`, `ConflictologyModel`, and `MixtureBaseline` all emit `logger.warning()` when entities present at `train_end` are missing from the fitted state. No test captures a log record and asserts that the warning was emitted. The warning paths are exercised only incidentally.

4. **`output_length != 36`:** Most tests use the default `output_length=36` or a small explicit value in the 4–5 range. No test checks behaviour at a non-default value that differs from the default in a semantically meaningful way (e.g., `output_length=1`, `output_length=72`).

5. **Multiple targets:** Most tests use the `targets` fixture which provides two targets (`["y1", "y2"]`). No test exercises the single-target case for distributional models to verify `dict` key cardinality.

---

## Consequences

**Positive:**

- Regressions in model arithmetic are caught before merge.
- Protocol conformance is verified structurally, not just by documentation.
- Manager integration is tested without requiring live pipeline dependencies.
- The three-team framing creates a shared vocabulary for discussing what is and is not tested.

**Negative / Risks:**

- The distributional manager path is unverified. A silent breakage in `_generate_predictions` for `ConflictologyModel` or `MixtureBaseline` would not be caught by CI.
- Red team gaps mean degenerate parameter combinations silently produce empty or NaN-filled output with no error. Users who pass `window_months=0` will receive incorrect predictions without any indication.
- No logging assertion tests means the entity-drop warning could be removed or miscoded without CI failing.

**Accepted debt:**

The gaps listed above are accepted as known technical debt, not oversights. They are recorded here so they can be prioritised deliberately rather than discovered in production.
