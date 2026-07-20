# Class Intent Contract: MixtureBaseline

**Date:** 2026-03-17
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-001, ADR-003, ADR-005, ADR-006, ADR-008, ADR-009, ADR-011, ADR-019, ADR-020

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
- Normalizes the input via `to_feature_frame(df, loa, targets)` (the ADR-019 pandas boundary; `pd.DataFrame` or `FeatureFrame`), records `time_idx`/`entity_idx` from `ff.index.level.index_names`, and reads the `(time, unit, values)` numpy panel via `panel(ff, targets)` — no pandas.
- Builds `self.local_pool`: `dict[entity_id, dict[target, np.ndarray]]` — for each entity, the last `window_months` training values per target, via the shared `window_pool_arrays(time, unit, values, ...)` (a stable `(entity, time)` `np.lexsort`, identical to `ConflictologyModel`). Also stores `self.entity_ids` (entities present at `train_end`).
- Builds `self.global_pool`: `dict[target, np.ndarray]` — for each target, all strictly-positive (`> 0`) values across the whole training panel (`time <= train_end`), in `(entity, time)`-sorted numpy order (matching the pre-PR-2 sorted `train_df[t].values` order so the global `rng.choice` draws stay byte-identical, ADR-011). `train_end` (`= test_start - 1`) is the last included timestep.
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
| `window_months` | `int` | Number of trailing training months for local pool. Must be `>= 1` — a non-positive value fails loud at fit (`window_pool_arrays`). |
| `lambda_mix` | `float` | Mixing probability toward global pool. Expected range `[0.0, 1.0]` (not validated). |
| `n_samples` | `int` | Number of draws per prediction cell. Must be positive (not validated). |
| `partition_dict` | `dict` | Must contain `"test"` with tuple `(test_start, test_end)`. |
| `loa` | `str` | Declared level of analysis (`"cm"` or `"pgm"`). Now **used**, not just stored: `to_feature_frame` resolves it to a `SpatialLevel`, validates the input's spatial level against it, and carries it into every output `PredictionFrame` (ADR-003/ADR-020). |
| `seed` | `int` | Base seed for `np.random.default_rng`. **Required, audited genome key** (ADR-021 / C-10): the catalog forwards `config["seed"]` and it must be declared in config. `DEFAULT_SEED` (42) is a single-sourced sentinel for direct/test construction only, never the production path. |
| `df` (fit/predict) | `pd.DataFrame \| FeatureFrame` | Normalized at entry via `to_feature_frame`. A DataFrame needs a 2-level MultiIndex (level 0 = time, level 1 = entity) matching `loa`; a `FeatureFrame` must carry the matching `level` and the `targets` features. |
| `sequence_number` | `int` | Forecast window start offset. |
| `output_length` | `int` | Number of forecast timesteps. Required — no default; passed from `config["time_steps"]` by the manager. |

---

## Outputs and Side Effects

- **`fit()`**: Returns `self`. Sets `self.time_idx`, `self.entity_idx`, `self.entity_ids`, `self.local_pool`, `self.global_pool`. No external side effects.
- **`predict()`**: Returns `dict[str, PredictionFrame]`. Raises `ValueError` (via `require_entities`) if no entities have pools. Emits `WARNING` on entity drops. `PredictionFrame`/`SpatioTemporalIndex` are lazy-imported from `views_frames` inside the `to_prediction_frames` seam. No other external side effects.

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| Target column missing / feature absent from a `FeatureFrame` | `ValueError` (crash) | `to_feature_frame` checks target presence at the input boundary (ADR-019). |
| DataFrame index names do not match `loa` (incl. non-2-level index) | `ValueError` (crash) | `to_feature_frame` → `resolve_level` raises at entry. |
| `partition_dict` missing `"test"` | `KeyError` (crash) | No validation. |
| `lambda_mix` outside `[0, 1]` | Undefined behaviour | Values `< 0` mean all draws come from the local pool in practice; values `> 1` mean all draws come from global. |
| `window_months <= 0` | `ValueError` at **fit** (crash) | `window_pool_arrays` fails loud with `"window_months must be >= 1, got ..."`. Previously the empty local pool surfaced only later as a `rng.choice` `ValueError` at predict — now caught deliberately (ADR-019). |
| `global_pool` empty for a target | Silent fallback | `_sample()` uses local-only path. No warning. |
| Entities absent from `local_pool` at predict time | `WARNING` log | Dropped from output. |
| No entities with pools | `ValueError` (crash) | `require_entities` raises a descriptive error. Fail-loud — consistent with all baseline models; an empty result would otherwise surface as a `StopIteration` deep in pipeline-core's evaluation path. |
| `views_frames` not installed | `ImportError` at predict time | Lazy import in the `to_prediction_frames` seam defers this error. |

---

## Boundaries and Interactions

- **Depends on:** `numpy` for RNG and array operations; `views_baseline.model.frames.input` (`to_feature_frame`, `panel` — the single pandas boundary, ADR-019); `views_baseline.model.frames.pooling.window_pool_arrays` (the numpy windowing core, used directly so the same panel feeds the global-pool build); `views_baseline.model.frames.output.sample_prediction_grid`. pandas is not a runtime dependency — confined to `to_feature_frame`, imported only under `TYPE_CHECKING`.
- **Output construction (ADR-020):** routes through the single seam `to_prediction_frames` in `model/frames/output.py` (lazy `views_frames` leaf import inside the function). No inline construction; `build_prediction_grid` is deleted.
- **Predict scaffold (ADR-011):** `predict()` delegates the shared distributional shell (entity→time→target fill, seeded RNG, entity drop/`require_entities`, seam construction) to `frames.output.sample_prediction_grid(..., draw_cell)`, supplying `self._sample` (signature `(cid, target, rng)`) as the per-cell `draw_cell`. The spatial `level` is resolved once at the input boundary and passed in as `level=ff.index.level` (PR-2 changed `sample_prediction_grid`'s signature to take `level`, not `(loa, index_names)`).
- **Instantiated by:** `BaselineModelCatalog._get_mixture_model()`. All model-specific params (`window_months`, `lambda_mix`, `n_samples`) are required config keys — the catalog reads them via `self.config[key]` with no defaults.
- **Dispatched by:** `BaselineForecastingModelManager._generate_predictions()` — a single type-uniform path since ADR-017; `distributional = True` is a semantic marker, not an `isinstance` dispatch discriminator.
- **Pool equivalence with `ConflictologyModel`:** When `lambda_mix=0.0`, all draws come from the local pool, which is constructed by the same logic as `ConflictologyModel.hist_per_entity`. This is verified by `test_conflictology_matches_mixture_lambda_zero`.

---

## Examples of Correct Usage

```python
from views_baseline.model.models.distributional import MixtureBaseline

partition_dict = {"test": (493, 528)}
model = MixtureBaseline(
    targets=["y1", "y2"],
    window_months=4,
    lambda_mix=0.05,
    n_samples=100,
    partition_dict=partition_dict,
    loa="pgm",
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
              partition_dict=partition_dict, loa="pgm", seed=123)
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

# window_months=0 now fails loud at fit (previously a deferred rng.choice ValueError at predict)
model = MixtureBaseline(..., window_months=0, ...)
model.fit(df)   # ValueError: window_months must be >= 1, got 0.

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
- If `views_frames` becomes a guaranteed hard dependency, moving the `PredictionFrame`/`FeatureFrame` imports to module level (out of the `to_prediction_frames` / `to_feature_frame` seams) would improve debuggability.

---

## Known Deviations

- No validation that `lambda_mix` is in `[0, 1]`. Values outside this range silently produce out-of-spec but numerically valid (if semantically wrong) results.
- `window_months <= 0` now fails loud with a descriptive `ValueError` at **fit** (via `window_pool_arrays`), no longer a deferred numpy error at predict.
- Global pool causal bound is `time <= train_end` (`= time < test_start`), the **same** bound the local pool uses — so `train_end` (the last training timestep) contributes to both pools. There is no local/global asymmetry.
- `FeatureFrame` storage is `float32` by design (C-32), so both the local pool (resampled window values) and the global pool (positive training values) are `float32`-rounded relative to raw `float64`. The entity order, per-entity tail order, global-pool `(entity, time)` order, and RNG-advance order are preserved exactly (the byte-identity contract, ADR-011), and a DataFrame and the `FeatureFrame` built from it produce identical output. Reproducibility across two model instances built on the same input therefore still holds bit-for-bit. This also preserves the byte-identical `local_pool` equivalence with `ConflictologyModel.hist_per_entity`.
- `_sample()` does not emit a warning when falling back to local-only due to an empty global pool. The fallback is silent.
- `PredictionFrame`/`SpatioTemporalIndex` are imported lazily inside the `to_prediction_frames` seam at predict time, deferring import errors to prediction rather than construction time.
