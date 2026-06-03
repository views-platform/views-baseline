# Class Intent Contract: ConflictologyModel

**Date:** 2026-03-17
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-001, ADR-003, ADR-005, ADR-006, ADR-008, ADR-009

---

## Purpose

`ConflictologyModel` is a distributional climatology baseline. For each entity, it treats the last `window_months` of training observations as an empirical distribution and produces `n_samples` i.i.d. bootstrap resamples (with replacement) per prediction cell. This gives a full predictive distribution that reflects the historical variability of conflict intensity for each location, without pooling across entities or weighting by recency. It serves as the distributional counterpart to `AverageModel`.

The class attribute `distributional = True` is used by `BaselineForecastingModelManager` and `DistributionalBaselineModel` protocol dispatch to route output handling correctly.

---

## Non-Goals

- Does not mix pools across entities. Each entity's samples come exclusively from its own local history.
- Does not weight observations by recency within the window.
- Does not produce a point prediction. The output is a `dict[str, PredictionFrame]`, not a `pd.DataFrame`.
- Does not validate `window_months` or `n_samples` at construction time.

---

## Responsibilities and Guarantees

- `fit(df)` extracts the last `window_months` rows per entity from the training period (`time_idx <= train_end`), sorts by `[entity_idx, time_idx]`, and stores the result as `self.hist_per_entity`: a `dict[entity_id, dict[target, np.ndarray]]`. Also stores `self.entity_ids` (entities present at `train_end`). Returns `self`.
- `predict(df, sequence_number, output_length)` returns `dict[str, PredictionFrame]` — one `PredictionFrame` per target. Each PF has:
  - `y_pred` of shape `(N, n_samples)` where `N = len(entities_with_history) * output_length`.
  - `identifiers` dict with `"time"` and `"unit"` arrays of length `N`.
- Samples are drawn using `np.random.default_rng(self.seed)`. The RNG is re-initialised fresh on each `predict()` call, making results deterministic given the same `seed` and `sequence_number`. RNG state is advanced in entity→time→target order to ensure consistent ordering.
- Entities with no history (absent from `hist_per_entity`) are dropped with a `WARNING` log.
- If no entities have history, raises a descriptive `ValueError` via `require_entities` (fail-loud — consistent with all baseline models).

---

## Inputs and Assumptions

| Parameter | Type | Notes |
|---|---|---|
| `targets` | `List[str]` | Column names in `df`. Must exist at fit time. |
| `window_months` | `int` | Number of trailing training months per entity to retain. Must be positive (not validated). |
| `partition_dict` | `dict` | Must contain `"test"` key with tuple `(test_start, test_end)`. |
| `loa` | `str` | Stored; not used in computation. |
| `n_samples` | `int` | Number of bootstrap draws per prediction cell. Required — no default. |
| `seed` | `int` | Base seed for `np.random.default_rng`. Default: 42. |
| `df` (fit/predict) | `pd.DataFrame` | 2-level MultiIndex; level 0 = time, level 1 = entity. |
| `sequence_number` | `int` | Offset from `test_start` for the prediction window start. |
| `output_length` | `int` | Number of forecast timesteps. Required — no default; passed from `config["time_steps"]` by the manager. |

Note: `fit()` uses `time_idx <= train_end` (inclusive) for the training filter, whereas the point models use `< test_start`. These are equivalent since `train_end = test_start - 1`, but the inclusive bound is explicit here.

---

## Outputs and Side Effects

- **`fit()`**: Returns `self`. Sets `self.time_idx`, `self.entity_idx`, `self.hist_per_entity`, `self.entity_ids`. No external side effects.
- **`predict()`**: Returns `dict[str, PredictionFrame]`. Raises `ValueError` (via `require_entities`) if no entities have history. Emits `WARNING` on entity drops. No external side effects. `PredictionFrame` is imported lazily from `views_pipeline_core.data.prediction_frame` at call time.

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| Target column missing from `df` | `KeyError` (crash) | No validation. |
| `df` missing 2-level MultiIndex | `IndexError` (crash) | No validation. |
| `partition_dict` missing `"test"` | `KeyError` (crash) | No validation. |
| `window_months <= 0` | `tail(0)` returns empty arrays | Sampling from empty array raises `ValueError` inside numpy during `predict()`. |
| `n_samples <= 0` | `ValueError` from numpy | `rng.choice(..., size=0)` succeeds but `size < 0` raises. |
| Entities with empty history at fit time | Silently excluded from `hist_per_entity` | Only entities whose `xs()` call returns non-empty data are stored. |
| Entities absent from `hist_per_entity` at predict time | `WARNING` log | Dropped from output. |
| No entities have history | `ValueError` (crash) | `require_entities` raises a descriptive error. Fail-loud — consistent with all baseline models; an empty result would otherwise surface as a `StopIteration` deep in pipeline-core's evaluation path. |
| `views_pipeline_core` not installed | `ImportError` (crash at predict time) | Lazy import defers this to `predict()`. |

---

## Boundaries and Interactions

- **Depends on:** `numpy` (`np.random.default_rng`, `np.array`, `rng.choice`).
- **Lazy import:** `views_pipeline_core.data.prediction_frame.PredictionFrame` — imported inside `predict()`, not at module load time. This means import failures surface at prediction time, not construction time.
- **Does not depend on** `build_prediction_grid`; output construction is inline.
- **Instantiated by:** `BaselineModelCatalog._get_conflictology_model()`.
- **Dispatched by:** `BaselineForecastingModelManager._generate_predictions()` via `isinstance(model, DistributionalBaselineModel)`.
- **Pool equivalence:** `hist_per_entity` is constructed identically to `MixtureBaseline.local_pool`. This equivalence is verified by `test_conflictology_matches_mixture_lambda_zero`.

---

## Examples of Correct Usage

```python
from views_baseline.model.baseline import ConflictologyModel

partition_dict = {"test": (493, 528)}
model = ConflictologyModel(
    targets=["y1", "y2"],
    window_months=4,
    partition_dict=partition_dict,
    loa="pg_id",
    n_samples=64,
    seed=42,
)
model.fit(df)
result = model.predict(df=df, sequence_number=0, output_length=5)

# result is dict[str, PredictionFrame]
pf = result["y1"]
assert pf.y_pred.shape == (n_entities * 5, 64)

# All samples come from local history
for i in range(pf.y_pred.shape[0]):
    uid = pf.identifiers["unit"][i]
    history_vals = set(model.hist_per_entity[uid]["y1"].tolist())
    assert set(pf.y_pred[i].tolist()).issubset(history_vals)
```

---

## Examples of Incorrect Usage

```python
# window_months=0: empty history arrays lead to numpy ValueError during predict
model = ConflictologyModel(targets=["y1"], window_months=0, partition_dict=partition_dict, loa="pg_id")
model.fit(df)
model.predict(df=df, sequence_number=0, output_length=5)   # ValueError from rng.choice on empty array

# Expecting a DataFrame instead of dict
result = model.predict(df=df, sequence_number=0, output_length=5)
result.values   # AttributeError: dict has no .values

# Assuming RNG state carries over between predict calls
r1 = model.predict(df=df, sequence_number=0, output_length=5)
r2 = model.predict(df=df, sequence_number=0, output_length=5)
# r1 and r2 ARE identical (RNG re-seeded each call) — but this is a feature, not a bug.
# Do NOT assume r1 != r2 for different sequence_numbers; they can coincidentally match
# if the drawn samples happen to be the same from the same history.
```

---

## Test Alignment

File: `tests/test_baseline.py`

| Test | What it verifies |
|---|---|
| `test_conflictology_model_resamples_from_history` | Output is `dict[str, PredictionFrame]`, shape is `(n_entities * output_length, n_samples)`, and all sampled values are elements of the entity's local history window. |
| `test_conflictology_model_respects_sequence_number` | Time identifiers in the PredictionFrame span `[test_start + seq_num, test_start + seq_num + output_length - 1]`. |
| `test_conflictology_matches_mixture_lambda_zero` | `hist_per_entity` pools are element-wise equal (after sorting) to `MixtureBaseline.local_pool` when built on the same data, confirming pool construction parity. |

---

## Evolution Notes

- If temporal weighting within the window is required, a new class (e.g., `WeightedClimatologyModel`) is preferable to adding optional parameters here. This preserves the "pure i.i.d. resample from flat history" semantics.
- The lazy `PredictionFrame` import could be moved to module level if `views_pipeline_core` becomes a hard install-time dependency. Currently, the lazy import avoids requiring the package when only point models are used.
- Adding `sequence_number` as an input to the RNG seed (i.e., `default_rng(seed + sequence_number)`) would make samples vary across sequence steps. This is a design choice currently left to the caller to work around via different seeds.

---

## Known Deviations

- No validation of `window_months > 0` or `n_samples > 0`. Invalid values surface as numpy errors during `predict()`, not as descriptive `ValueError`s at construction or fit time.
- `PredictionFrame` is imported inside `predict()`, not at module load time or construction time. Import failures (e.g., `views_pipeline_core` not installed) are deferred to prediction time.
- `fit()` uses `time_idx <= train_end` (inclusive upper bound on training data) while the point models use `time_idx < test_start`. These are semantically equivalent but the inconsistency makes code comparison slightly harder.
