# Hardened Contributor Protocol — views-baseline

**Applies to:** Any contributor (human or AI) making changes to numerical or stochastic code.
**Last reviewed:** 2026-03-13
**Related ADRs:** ADR-003, ADR-004, ADR-005, ADR-007

---

## Overview

This protocol defines the hardened practices for contributing to the ML and numerical
components of views-baseline. It applies specifically to:

- The seven model classes under `views_baseline/model/models/point/` and `views_baseline/model/models/distributional/`
- The output-construction helpers in `views_baseline/model/frames/output.py`
- Any new distributional model that uses NumPy random number generation

It extends the standard contributor protocols with additional requirements for
reproducibility, config validation, and numerical correctness.

---

## 1. Reproducibility Contract

### The RNG pattern

`ConflictologyModel` and `MixtureBaseline` use NumPy's `Generator` API for reproducible
stochastic output. The contract is:

```python
rng = np.random.default_rng(self.seed)
```

This is called at the **start of each `predict()` invocation**, not once at construction.
The result is that for a given `(seed, data, sequence_number)` combination, the output is
deterministic across multiple `predict()` calls with the same inputs.

**Do not use:**
- `np.random.seed()` — global state, not thread-safe
- `random.seed()` — wrong module
- Any RNG initialisation that is not `np.random.default_rng(seed)`
- RNG initialisation at `__init__` time rather than at `predict()` time (this would make
  reproducibility depend on the number of prior `predict()` calls)

### Iteration order is part of the contract

The order in which entities, time steps, and targets are iterated determines which random
draws map to which prediction cells. This order must not change without a deliberate decision
and an update to ADR-004 (reproducibility contract section).

Current order: **entity → time step → target** in all distributional models.

### Adding a new distributional model

If you add a new distributional model:

1. Use `np.random.default_rng(self.seed)` at the start of `predict()`.
2. Document the iteration order explicitly in a comment.
3. Write a reproducibility test analogous to `test_mixture_predict_reproducible`:
   two instances with the same `seed` must produce identical `y_pred` arrays.
4. Add a `distributional = True` class attribute (the protocol discriminator, per ADR-003).

### Known limitation: sequence_number not in seed

The current seed arithmetic does not incorporate `sequence_number`. Two calls to
`ConflictologyModel.predict()` or `MixtureBaseline.predict()` with different `sequence_number`
values but the same `seed` will produce samples from identically-seeded RNG instances.
Whether this is intentional is an open question tracked in ADR-004 and candidate ADR-011.

Do not "fix" this unilaterally. It is a documented known limitation, not an oversight.

---

## 2. Config Validation

### MODEL_GENOMES

`MODEL_GENOMES` in `BaselineModelCatalog` is the authoritative declaration of required config
keys per model:

```python
MODEL_GENOMES = {
    "ZeroModel":          [],
    "LocfModel":          [],
    "AverageModel":       ["window_months"],
    "ConflictologyModel": ["window_months", "n_samples"],
    "MixtureBaseline":    ["window_months", "lambda_mix", "n_samples"],
}
```

Rules:
- All model-specific parameters **must** be listed in `MODEL_GENOMES`. The catalog does not
  apply defaults — all values must be explicitly declared in `config_hyperparameters.py`.
- Parameters (those with defaults) are **not** listed in `MODEL_GENOMES`. They are
  accessed via `config.get(key, default)` in the factory method.
- `targets` and `partition_dict` are universal inputs (all models require them) but are not
  listed in `MODEL_GENOMES` — this is a known gap documented in ADR-009.

When adding a new model, update `MODEL_GENOMES` in the same commit as the constructor. A
`MODEL_GENOMES` entry that does not match the constructor is a silent validation gap.

### Value range validation

The catalog validates key presence, not value ranges. The following degenerate values are not
caught at construction time and will produce incorrect output silently:

- `window_months=0` — `tail(0)` returns an empty DataFrame; NaN means propagate
- `n_samples=0` — returns arrays with zero samples
- `lambda_mix < 0` or `lambda_mix > 1` — undefined mixture semantics
- `output_length=0` (passed to `predict()`) — returns an empty prediction silently

These are documented as accepted gaps in ADR-008 and ADR-009. Do not add value-range
validation without a PR discussion, as it adds maintenance surface.

---

## 3. File Structure — One Class Per File

Since the PR-1 reorganization (issues #48–#51), each baseline model class lives in its own
module. The point models (`ZeroModel`, `LocfModel`, `AverageModel`) live one-per-file under
`views_baseline/model/models/point/`, and the distributional models (`ConflictologyModel`,
`MixtureBaseline`, `ParametricConflictology`, `ParametricHurdleConflictology`) under
`views_baseline/model/models/distributional/`. The shared support code lives in `frames/`
(output seam + pooling), `distributions/`, `grid.py`, and `spatial.py`.

This supersedes the earlier single-file layout, whose rationale was recorded in ADR-001:

> Separate files per model class would have made the ontological boundaries physically
> visible. Rejected as over-engineered for seven classes that are closely related and
> frequently read together.

**Consequence for contributors:** When modifying a model module, read the full file before
making changes. Use targeted edits (the Edit tool's string-replacement mode, not full-file
rewrite) to avoid accidentally truncating class or function definitions.

The Anti-Truncation Rule from the silicon protocol applies: any AI-generated change that
shortens a model module by more than the lines being intentionally deleted is suspect.

---

## 4. Testing Taxonomy

The three-team testing model from ADR-005 maps to the codebase as follows:

### Green team — stability and correctness

Covers: `test_baseline.py`, `test_catalog.py`, `test_protocol.py`

Every new model class must have Green team tests that verify:
- Output shape and type (DataFrame vs `dict[str, PredictionFrame]`)
- Output column names (`pred_{target}`)
- Output MultiIndex names and time range
- Correct arithmetic (e.g., for `AverageModel`: output equals the mean of the declared window)
- `sequence_number` offset is correctly applied

For distributional models, additionally verify:
- `y_pred` shape is `(n_entities * output_length, n_samples)`
- All sampled values for a given entity come from that entity's history
- Reproducibility under identical seeds

### Beige team — realistic integration

Covers: `test_baseline_manager.py`

When adding a new model, consider whether a Beige team test is needed to verify the manager
consumes the new model's `dict[str, PredictionFrame]` output correctly. Note that since
ADR-017 the manager has a single type-uniform path — every model returns the same type, so
there is no `isinstance(model, DistributionalBaselineModel)` dispatch branch in
`_generate_predictions` to test (`distributional = True` is a semantic marker, not a dispatch
discriminator).

### Red team — adversarial inputs

Currently absent. Valued contributions include tests for:
- `window_months=0` (expect graceful failure or documented behaviour)
- `output_length=0` (expect empty output with no error, or a ValueError)
- Missing target columns in the input DataFrame
- Single-level index passed to `fit()` (expect `IndexError` or a descriptive `ValueError`)

Do not add Red team tests that change current silent-failure behaviour to loud-failure
without a PR discussion. The desired failure mode for each case must be agreed first.

---

## 5. Point Forecast vs Distributional Model Checklist

When adding a new model, use the appropriate checklist.

### Point forecast model checklist

- [ ] Class in its own module under `views_baseline/model/models/point/`
- [ ] `fit(df) -> self` — stores index names and per-entity statistics
- [ ] `predict(df, sequence_number, output_length) -> pd.DataFrame` with `MultiIndex(time, entity)` and `pred_{target}` columns
- [ ] Uses `build_prediction_frame` from `frames/output.py` (do not re-implement grid construction)
- [ ] No `distributional` class attribute
- [ ] Emits `logger.info` on `fit()` and `predict()`
- [ ] Emits `logger.warning` if entities are dropped at predict time
- [ ] Entry in `MODEL_GENOMES` (even if `[]`)
- [ ] Factory method `_get_<name>_model()` in `BaselineModelCatalog`
- [ ] Green team tests: shape, values, sequence_number, entity coverage
- [ ] CIC in `docs/CICs/<ClassName>.md`
- [ ] ADR-001 ontology updated
- [ ] ADR-004 stability map updated

### Distributional model checklist

All of the above (but the class module lives under `views_baseline/model/models/distributional/`, not `point/`), plus:
- [ ] `distributional = True` class attribute
- [ ] `predict()` returns `dict[str, PredictionFrame]` — one key per target
- [ ] `PredictionFrame` imported lazily inside `predict()`, not at module top-level
- [ ] `np.random.default_rng(self.seed)` called at start of `predict()`
- [ ] Iteration order documented in a comment
- [ ] Reproducibility test: two instances with same seed produce identical `y_pred`
- [ ] ADR-002 lazy import rule applied and documented
- [ ] ADR-003 `distributional` attribute documented as the protocol discriminator

---

## 6. No Numerical Airlock

views-baseline does not use deep learning, gradients, or GPU computation. There is no
need for:
- Float32 downcasting
- Mixed-precision training
- CUDA device management
- Gradient checkpointing
- Framework-specific random state (PyTorch, TensorFlow, JAX)

All numerical computation uses NumPy and pandas. The reproducibility contract is fully
covered by `np.random.default_rng(seed)`. Do not introduce any deep learning framework
dependency.

---

## 7. Logging in Numerical Code

From ADR-008, logging in model methods follows this pattern:

- `logger.info(...)` — at `fit()` entry and `predict()` entry, confirming which model and
  entity index is in use
- `logger.warning(...)` — when entities are dropped, with the count: e.g.,
  `f"ConflictologyModel: {n} entities dropped (missing from hist_per_entity)"`
- No `logger.error(...)` or `logger.critical(...)` currently used

Do not log inside inner loops (entity or time step loops). Log once at method entry and
once for any aggregate anomaly (entity drops). Logging inside loops produces unbounded
output for large datasets.
