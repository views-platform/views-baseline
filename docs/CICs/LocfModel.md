# Class Intent Contract: LocfModel

**Date:** 2026-03-17
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-001, ADR-005, ADR-006, ADR-008

---

## Purpose

`LocfModel` (Last Observation Carried Forward) is a persistence baseline: it carries the final training observation for each entity forward as a constant prediction across the entire forecast horizon. It represents the hypothesis that "the most recent measurement is the best available predictor," which is a useful reference for targets with strong autocorrelation such as conflict event counts.

---

## Non-Goals

- Does not average across time. Only the single last observation before `test_start` is retained per entity per target.
- Does not interpolate or extrapolate.
- Does not produce distributional output. Returns `dict[str, PredictionFrame]` with `y_pred` shape `(N, 1)` — a single deterministic value per cell, not multiple samples.

---

## Responsibilities and Guarantees

- `fit(df)` filters `df` to rows where `time_idx < test_start`, sorts by `[entity_idx, time_idx]`, then computes `groupby(entity_idx)[targets].last()`. The result is stored as `self.last_observations` (a DataFrame indexed by entity, columns = targets). Returns `self`.
- `predict(df, sequence_number, output_length)` determines `entity_ids` from rows at `train_end`, filters to those present in `last_observations`, and returns `dict[str, PredictionFrame]` via `build_prediction_frame` where every cell for entity `cid` and target `t` has the value `last_observations.loc[cid, t]`.
- If any entities in `entity_ids` are absent from `last_observations`, a `WARNING` is logged with the count of dropped entities.
- The returned dict has the same structure contract as all baseline models: one key per target, each value a `PredictionFrame` with `identifiers` dict containing `"time"` and `"unit"` arrays.
- `self.time_idx` is `None` before `fit()` is called — callers can use this as a pre-fit sentinel.

---

## Inputs and Assumptions

| Parameter | Type | Notes |
|---|---|---|
| `targets` | `List[str]` | Must exist as columns in `df` at fit time. |
| `partition_dict` | `dict` | Must contain `"test"` key with tuple `(test_start, test_end)`. |
| `loa` | `str` | Stored for logging context; not used in computation. |
| `df` (fit) | `pd.DataFrame` | 2-level MultiIndex; level 0 = time, level 1 = entity. Data may be unsorted. |
| `df` (predict) | `pd.DataFrame` | Same structure; used only to extract entity IDs at `train_end`. |
| `sequence_number` | `int` | Shifts the prediction window start relative to `test_start`. |
| `output_length` | `int` | Number of forecast timesteps. Required — no default; passed from `config["time_steps"]` by the manager. |

The model handles unsorted input data correctly because `fit()` explicitly calls `sort_index(level=[entity_idx, time_idx])` before applying `.last()`. Positional ordering of rows in the input DataFrame does not affect which value is selected as the last observation.

---

## Outputs and Side Effects

- **`fit()`**: Returns `self`. Sets `self.time_idx`, `self.entity_idx`, `self.last_observations`. Emits an `INFO` log message. No external side effects.
- **`predict()`**: Returns `dict[str, PredictionFrame]` with `y_pred` shape `(N, 1)`. Emits an `INFO` log and, if entities are dropped, a `WARNING`. No external side effects.

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| `targets` column missing from `df` | `KeyError` (crash) | No validation that targets exist in `df`. |
| `df` missing 2-level MultiIndex | `IndexError` (crash) | No structural validation. |
| `partition_dict` missing `"test"` key | `KeyError` (crash) | No validation. |
| Entities in `entity_ids` absent from `last_observations` | `WARNING` log | Entity is silently dropped from predictions. |
| All entities dropped | Silent empty dict | `build_prediction_frame` returns an empty dict. |
| `predict()` called before `fit()` | `AttributeError` | `self.last_observations` is `None`; `cid in None` raises. |

---

## Boundaries and Interactions

- **Depends on:** `views_baseline.model.helpers.build_prediction_frame`. `PredictionFrame` is lazy-imported from `views-pipeline-core` inside the helper.
- **External dependency:** `views-pipeline-core` (via `PredictionFrame`, lazy-imported at call time).
- **Instantiated by:** `BaselineModelCatalog._get_locf_model()`.

---

## Examples of Correct Usage

```python
from views_baseline.model.baseline import LocfModel

partition_dict = {"test": (493, 528)}
model = LocfModel(targets=["y1", "y2"], partition_dict=partition_dict, loa="pg_id")
model.fit(df)
preds = model.predict(df=df, sequence_number=0, output_length=36)

# Each cell holds the last training value for that entity
train_df = df[df.index.get_level_values("month_id") < 493]
last_obs = train_df.groupby(level="pg_id")[["y1", "y2"]].last()
assert preds.loc[(493, 1), "pred_y1"] == last_obs.loc[1, "y1"]
```

Checking pre-fit sentinel:

```python
model = LocfModel(targets=["y1"], partition_dict=partition_dict, loa="pg_id")
assert model.time_idx is None   # safe pre-fit check
```

---

## Examples of Incorrect Usage

```python
# Targets not present in DataFrame
model = LocfModel(targets=["nonexistent"], partition_dict=partition_dict, loa="pg_id")
model.fit(df)          # KeyError: "nonexistent"

# Calling predict before fit
model = LocfModel(targets=["y1"], partition_dict=partition_dict, loa="pg_id")
model.predict(df=df, sequence_number=0)   # AttributeError: last_observations is None

# Assuming positional last row is the temporally last row
# (safe — LocfModel sorts internally, so this is NOT a user error to worry about,
#  but passing pre-sorted data does not break anything)
```

---

## Test Alignment

File: `tests/test_baseline.py`

| Test | What it verifies |
|---|---|
| `test_locf_model_uses_last_observation` | Prediction values equal the per-entity last training observation; time range and index structure are correct. |
| `test_locf_model_respects_sequence_number` | Output time range shifts correctly with `sequence_number=1`. |
| `test_locf_model_time_idx_is_not_tuple_before_fit` | `time_idx` is `None` before `fit()` is called. |
| `test_locf_model_fit_handles_unsorted_data` | With deliberately unsorted input, `.last()` selects the temporally last row (month 492), not the positionally last row. |

---

## Evolution Notes

- If a decay variant is desired (e.g., last observation multiplied by a decay factor per timestep), a new class is preferable to adding optional parameters here, to keep this class's semantics unambiguous.
- If validation of target columns is added to `fit()`, both the happy-path and the `KeyError` case should be updated in tests.

---

## Known Deviations

- No validation that `targets` exist as columns in `df`. Missing targets raise a bare `KeyError` from pandas.
- Entity filtering in `predict()` is performed against `last_observations`, which is keyed on entities present in training. An entity present at `train_end` but with no training rows (an edge-case possible if the DataFrame is constructed unusually) would produce a `KeyError` inside `value_fn` during grid building.
