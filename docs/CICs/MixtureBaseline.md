# Class Intent Contract: MixtureBaseline

**Date:** 2026-03-17
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-001, ADR-003, ADR-005, ADR-006, ADR-008, ADR-009

---

## Purpose

`MixtureBaseline` is a distributional baseline that addresses the zero-probability trap in conflict forecasting. For entities with locally zero or near-zero historical values, a pure local resampler such as `ConflictologyModel` can never predict conflict onset. `MixtureBaseline` mixes local (per-entity) history with a global pool of all positive training values using a Bernoulli mixing coefficient `lambda_mix`. With probability `lambda_mix`, each of the `n_samples` draws comes from the global positive pool; with probability `1 - lambda_mix`, it comes from the entity's local history.

The class attribute `distributional = True` routes the output through the distributional code path in `BaselineForecastingModelManager`.

---

## Non-Goals

- Does not learn `lambda_mix` from data. It is a fixed hyperparameter.
- Does not perform temporal weighting within the local or global pool.
- Does not use a formal mixture model (no EM, no kernel density estimation). The mixing is purely a Bernoulli draw per sample.
- Does not validate `lambda_mix` to be in `[0, 1]` or `window_months` to be positive.

---

## Responsibilities and Guarantees

**`fit(df)`**:
- Filters `df` to `time_idx < test_start` (strict), sorts, and derives `entity_ids` from the single row at `train_end`.
- Builds `self.local_pool`: `dict[entity_id, dict[target, np.ndarray]]` — for each entity, the last `window_months` training values per target.
- Builds `self.global_pool`: `dict[target, np.ndarray]` — for each target, all training values across all entities and all training timesteps that are strictly positive (`> 0`). The causal bound is `time_idx < test_start`, so `train_end` is the last included timestep.
- Returns `self`.

**`_sample(cid, target, rng)`**:
- Draws `n_samples` values by mixing local and global pools.
- If `global_pool[target]` is empty (all training values are zero or non-positive), falls back to local-only sampling with no error or warning.
- Uses `rng.random(n_samples) < lambda_mix` to determine which samples come from the global pool (vectorised Bernoulli draw).

**`predict(df, sequence_number, output_length)`**:
- Constructs entity list from `entity_ids` filtered to those in `local_pool`.
- Logs `WARNING` on entity drops.
- Raises a descriptive `ValueError` via `require_entities` if no entities have pools (fail-loud — consistent with all baseline models).
- Returns `dict[str, PredictionFrame]` — one PF per target. Shape `(N, n_samples)` where `N = len(entities_with_pool) * output_length`.
- RNG is re-initialised via `np.random.default_rng(self.seed)` at the start of each `predict()` call. Iteration order is entity→time→target for RNG consistency.

---

## Inputs and Assumptions

| Parameter | Type | Notes |
|---|---|---|
| `targets` | `List[str]` | Must exist as columns in `df`. |
| `window_months` | `int` | Number of trailing training months for local pool. Must be positive (not validated). |
| `lambda_mix` | `float` | Mixing probability toward global pool. Expected range `[0.0, 1.0]` (not validated). |
| `n_samples` | `int` | Number of draws per prediction cell. Must be positive (not validated). |
| `partition_dict` | `dict` | Must contain `"test"` with tuple `(test_start, test_end)`. |
| `loa` | `str` | Stored; not used in computation. |
| `seed` | `int` | Base seed for `np.random.default_rng`. **Required, audited genome key** (ADR-021 / C-10): the catalog forwards `config["seed"]` and it must be declared in config. `DEFAULT_SEED` (42) is a single-sourced sentinel for direct/test construction only, never the production path. |
| `df` (fit/predict) | `pd.DataFrame` | 2-level MultiIndex; level 0 = time, level 1 = entity. |
| `sequence_number` | `int` | Forecast window start offset. |
| `output_length` | `int` | Number of forecast timesteps. Required — no default; passed from `config["time_steps"]` by the manager. |

---

## Outputs and Side Effects

- **`fit()`**: Returns `self`. Sets `self.time_idx`, `self.entity_idx`, `self.entity_ids`, `self.local_pool`, `self.global_pool`. No external side effects.
- **`predict()`**: Returns `dict[str, PredictionFrame]`. Raises `ValueError` (via `require_entities`) if no entities have pools. Emits `WARNING` on entity drops. `PredictionFrame` is imported lazily from `views_pipeline_core.data.prediction_frame`. No other external side effects.

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| Target column missing from `df` | `KeyError` (crash) | No validation. |
| `df` missing 2-level MultiIndex | `IndexError` (crash) | No validation. |
| `partition_dict` missing `"test"` | `KeyError` (crash) | No validation. |
| `lambda_mix` outside `[0, 1]` | Undefined behaviour | Values `< 0` mean all draws come from the local pool in practice; values `> 1` mean all draws come from global. |
| `window_months <= 0` | Empty local pool arrays | `tail(0)` produces empty arrays; sampling from empty raises `ValueError` in numpy at predict time. |
| `global_pool` empty for a target | Silent fallback | `_sample()` uses local-only path. No warning. |
| Entities absent from `local_pool` at predict time | `WARNING` log | Dropped from output. |
| No entities with pools | `ValueError` (crash) | `require_entities` raises a descriptive error. Fail-loud — consistent with all baseline models; an empty result would otherwise surface as a `StopIteration` deep in pipeline-core's evaluation path. |
| `views_pipeline_core` not installed | `ImportError` at predict time | Lazy import defers this error. |

---

## Boundaries and Interactions

- **Depends on:** `numpy` for RNG and array operations.
- **Output construction (ADR-020):** routes through the single seam `to_prediction_frames` in `model/helpers.py` (lazy `views_frames` leaf import inside the function). No inline construction; `build_prediction_grid` is deleted.
- **Predict scaffold (ADR-011):** `predict()` delegates the shared distributional shell (entity→time→target fill, seeded RNG, entity drop/`require_entities`, seam construction) to `helpers.sample_prediction_grid(..., draw_cell)`, supplying `self._sample` (signature `(cid, target, rng)`) as the per-cell `draw_cell`.
- **Instantiated by:** `BaselineModelCatalog._get_mixture_model()`. All model-specific params (`window_months`, `lambda_mix`, `n_samples`) are required config keys — the catalog reads them via `self.config[key]` with no defaults.
- **Dispatched by:** `BaselineForecastingModelManager._generate_predictions()` — a single type-uniform path since ADR-017; `distributional = True` is a semantic marker, not an `isinstance` dispatch discriminator.
- **Pool equivalence with `ConflictologyModel`:** When `lambda_mix=0.0`, all draws come from the local pool, which is constructed by the same logic as `ConflictologyModel.hist_per_entity`. This is verified by `test_conflictology_matches_mixture_lambda_zero`.

---

## Examples of Correct Usage

```python
from views_baseline.model.baseline import MixtureBaseline

partition_dict = {"test": (493, 528)}
model = MixtureBaseline(
    targets=["y1", "y2"],
    window_months=4,
    lambda_mix=0.05,
    n_samples=100,
    partition_dict=partition_dict,
    loa="pg_id",
    seed=42,
)
model.fit(df)
result = model.predict(df=df, sequence_number=0, output_length=5)

# result is dict[str, PredictionFrame]
pf = result["y1"]
assert pf.y_pred.shape == (n_entities * 5, 100)

# Lambda=0: local only — all samples within local pool values
model_local = MixtureBaseline(..., lambda_mix=0.0, ...)
model_local.fit(df)
r = model_local.predict(df=df, sequence_number=0, output_length=1)
for i in range(r["y1"].y_pred.shape[0]):
    uid = r["y1"].identifiers["unit"][i]
    local_vals = set(model_local.local_pool[uid]["y1"].tolist())
    assert set(r["y1"].y_pred[i].tolist()).issubset(local_vals)

# Lambda=1: global only — all samples from positive pool
model_global = MixtureBaseline(..., lambda_mix=1.0, ...)
model_global.fit(df)
r = model_global.predict(df=df, sequence_number=0, output_length=1)
# Entity 3 (all-zero local pool) now receives positive samples from the global pool
```

Reproducibility:

```python
kwargs = dict(targets=["y1"], window_months=4, lambda_mix=0.05, n_samples=50,
              partition_dict=partition_dict, loa="pg_id", seed=123)
m1 = MixtureBaseline(**kwargs)
m1.fit(df)
r1 = m1.predict(df=df, sequence_number=0, output_length=2)

m2 = MixtureBaseline(**kwargs)
m2.fit(df)
r2 = m2.predict(df=df, sequence_number=0, output_length=2)

np.testing.assert_array_equal(r1["y1"].y_pred, r2["y1"].y_pred)  # deterministic
```

---

## Examples of Incorrect Usage

```python
# lambda_mix > 1: all Bernoulli draws evaluate True, so all samples come from global pool
model = MixtureBaseline(..., lambda_mix=1.5, ...)
# No error raised, but behaviour is equivalent to lambda_mix=1.0 (all global)

# window_months=0: empty local pool, numpy ValueError at predict time
model = MixtureBaseline(..., window_months=0, ...)
model.fit(df)
model.predict(df=df, sequence_number=0, output_length=5)   # ValueError from rng.choice on empty array

# Expecting DataFrame instead of dict
result = model.predict(df=df, sequence_number=0, output_length=5)
result.values   # AttributeError: dict has no .values

# Calling _sample() with an entity not in local_pool
model._sample(99999, "y1", rng)   # KeyError: 99999 not in local_pool
```

---

## Test Alignment

File: `tests/test_baseline.py`

| Test | What it verifies |
|---|---|
| `test_mixture_fit_extracts_local_pool` | Local pool contains exactly the last `window_months` training values per entity. Entity with all-zero values has an all-zero local pool. |
| `test_mixture_fit_extracts_global_pool` | Global pool contains only positive values; entity with all-zero values contributes nothing; pool size equals expected count. |
| `test_mixture_fit_global_pool_causal` | Maximum value in global pool is causally bounded: no values from `test_start` or later. |
| `test_mixture_fit_returns_self` | `fit()` returns the model instance. |
| `test_mixture_predict_shape` | Output is `dict[str, PredictionFrame]` with correct keys and shape `(n_entities * output_length, n_samples)`. |
| `test_mixture_predict_respects_sequence_number` | Time identifiers span the correct window relative to `sequence_number`. |
| `test_mixture_predict_lambda_zero_local_only` | All samples for each entity are subsets of that entity's local pool. Entity 3 (all-zero) produces only zero samples. |
| `test_mixture_predict_lambda_one_global_only` | Entity 3 (all-zero local pool) produces only positive samples when `lambda_mix=1.0`. |
| `test_mixture_predict_reproducible` | Identical kwargs and seed produce bit-identical `y_pred` arrays across two independent model instances. |

---

## Evolution Notes

- A natural extension is `lambda_mix` as a per-entity or per-target parameter (e.g., entities with sparse history receive higher `lambda_mix`). This would require changing the `_sample()` signature and is best introduced as a subclass.
- The global pool currently holds raw values. A future variant might transform these (e.g., log scale) before sampling. This should be a separate class rather than a parameter, to keep the sampling semantics clear.
- If `views_pipeline_core` becomes a guaranteed hard dependency, moving `PredictionFrame` to a module-level import would improve debuggability.

---

## Known Deviations

- No validation that `lambda_mix` is in `[0, 1]`. Values outside this range silently produce out-of-spec but numerically valid (if semantically wrong) results.
- No validation that `window_months > 0`. Invalid values surface as numpy errors during `predict()`.
- Global pool causal bound is `time_idx < test_start` (strict), which excludes `train_end` from contributing to the global pool. This means the single most recent training timestep is included in local pools but not the global pool. This asymmetry is documented here but not in the source code.
- `_sample()` does not emit a warning when falling back to local-only due to an empty global pool. The fallback is silent.
- `PredictionFrame` is imported lazily inside `predict()`, deferring import errors to prediction time rather than construction time.
