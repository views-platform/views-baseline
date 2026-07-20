# Class Intent Contract: LocfModel

**Date:** 2026-03-17
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-001, ADR-005, ADR-006, ADR-008, ADR-019, ADR-020

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

- `fit(df)` normalizes the input via `to_feature_frame(df, loa, targets)` (the ADR-019 pandas boundary), records `time_idx`/`entity_idx` from `ff.index.level.index_names`, and computes the last observation per entity as the **window-of-1** `window_pool(ff, targets, 1, train_end)` on the frame's numpy panel (no pandas). The result is stored as `self.last_observations`, a nested dict `{entity -> {target -> last observed value}}` (not a pandas object). Returns `self`.
- `predict(df, sequence_number, output_length)` normalizes the input, determines `entity_ids` via `entities_at(unit, time, train_end)` (first-appearance order), filters to those present in `last_observations` (`filter_entities`), and returns `dict[str, PredictionFrame]` via `build_prediction_frame` (with `level = ff.index.level`) where every cell for entity `cid` and target `t` has the value `last_observations[cid][t]`.
- If any entities in `entity_ids` are absent from `last_observations`, a `WARNING` is logged with the count of dropped entities.
- The returned dict has the same structure contract as all baseline models: one key per target, each value a `PredictionFrame` with `identifiers` dict containing `"time"` and `"unit"` arrays.
- `self.time_idx` is `None` before `fit()` is called — callers can use this as a pre-fit sentinel.

---

## Inputs and Assumptions

| Parameter | Type | Notes |
|---|---|---|
| `targets` | `List[str]` | Must exist as columns in `df` at fit time. |
| `partition_dict` | `dict` | Must contain `"test"` key with tuple `(test_start, test_end)`. |
| `loa` | `str` | Declared level of analysis (`"cm"` or `"pgm"`). Now **used**, not just logged: `to_feature_frame` resolves it to a `SpatialLevel` and validates the input's spatial level against it (ADR-003/ADR-020). |
| `df` (fit) | `pd.DataFrame \| FeatureFrame` | Normalized at entry via `to_feature_frame`. A DataFrame needs a 2-level MultiIndex (level 0 = time, level 1 = entity) matching the declared `loa`; data may be unsorted. A `FeatureFrame` must carry the matching `level` and the `targets` features. |
| `df` (predict) | `pd.DataFrame \| FeatureFrame` | Same structure; used only to extract entity IDs at `train_end`. |
| `sequence_number` | `int` | Shifts the prediction window start relative to `test_start`. |
| `output_length` | `int` | Number of forecast timesteps. Required — no default; passed from `config["time_steps"]` by the manager. |

The model handles unsorted input data correctly because `window_pool` stably sorts the numpy panel by `(entity, time)` (via `np.lexsort`) before taking each entity's last row. Positional ordering of rows in the input does not affect which value is selected as the last observation.

---

## Outputs and Side Effects

- **`fit()`**: Returns `self`. Sets `self.time_idx`, `self.entity_idx`, `self.last_observations`. Emits an `INFO` log message. No external side effects.
- **`predict()`**: Returns `dict[str, PredictionFrame]` with `y_pred` shape `(N, 1)`. Emits an `INFO` log and, if entities are dropped, a `WARNING`. No external side effects.

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| `targets` column missing from `df` / feature absent from a `FeatureFrame` | `ValueError` (crash) | `to_feature_frame` checks target presence at the input boundary and raises a descriptive error (ADR-019). |
| DataFrame index names do not match the declared `loa` (incl. non-2-level index) | `ValueError` (crash) | `to_feature_frame` → `resolve_level` raises at entry. |
| `partition_dict` missing `"test"` key | `KeyError` (crash) | No validation. |
| Entities in `entity_ids` absent from `last_observations` | `WARNING` log | Entity is silently dropped from predictions. |
| All entities dropped | `ValueError` (crash) | `require_entities` raises a descriptive error. Fail-loud: a prediction over zero entities cannot satisfy the evaluation contract (an empty result would otherwise surface as a `StopIteration` deep in pipeline-core). |
| `predict()` called before `fit()` | `TypeError` (crash) | `self.last_observations` is `None`; `filter_entities` evaluates `cid in None`, which raises. |

---

## Boundaries and Interactions

- **Depends on:** `views_baseline.model.frames.input` (`to_feature_frame`, `panel` — the single pandas boundary, ADR-019), `views_baseline.model.frames.pooling.window_pool` (numpy windowing), `views_baseline.model.frames.output.build_prediction_frame`, and `views_baseline.model.grid` (`entities_at`, `filter_entities`, `require_entities`, `build_time_grid`). pandas is not a runtime dependency of this model — it is confined to `to_feature_frame` and imported only under `TYPE_CHECKING`.
- **External dependency:** `views_frames` — `FeatureFrame`/`SpatioTemporalIndex` lazy-imported inside `to_feature_frame`; `PredictionFrame`/`SpatioTemporalIndex` inside the `to_prediction_frames` seam, at call time.
- **Instantiated by:** `BaselineModelCatalog._get_locf_model()`.

---

## Examples of Correct Usage

```python
from views_baseline.model.models.point import LocfModel

partition_dict = {"test": (493, 528)}
model = LocfModel(targets=["y1", "y2"], partition_dict=partition_dict, loa="pgm")
model.fit(df)                # model.fit(ff) also works — dual input
preds = model.predict(df=df, sequence_number=0, output_length=36)

# Fitted state is a nested dict; each cell holds the last training value for that entity.
# (float32-rounded via the FeatureFrame — see C-32 note in Known Deviations.)
assert model.last_observations[1]["y1"] == <last observed y1 for entity 1 before test_start>

# preds is dict[str, PredictionFrame]; the first cell of entity 1 carries that value
pf = preds["y1"]
```

Checking pre-fit sentinel:

```python
model = LocfModel(targets=["y1"], partition_dict=partition_dict, loa="pgm")
assert model.time_idx is None   # safe pre-fit check
```

---

## Examples of Incorrect Usage

```python
# Targets not present in DataFrame
model = LocfModel(targets=["nonexistent"], partition_dict=partition_dict, loa="pgm")
model.fit(df)          # ValueError: DataFrame is missing required target column(s) ['nonexistent']

# Calling predict before fit
model = LocfModel(targets=["y1"], partition_dict=partition_dict, loa="pgm")
model.predict(df=df, sequence_number=0, output_length=36)   # TypeError: last_observations is None

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
| `test_locf_model_fit_handles_unsorted_data` | With deliberately unsorted input, `window_pool`'s stable `(entity, time)` sort selects the temporally last row (month 492), not the positionally last row. |

---

## Evolution Notes

- If a decay variant is desired (e.g., last observation multiplied by a decay factor per timestep), a new class is preferable to adding optional parameters here, to keep this class's semantics unambiguous.
- If validation of target columns is added to `fit()`, both the happy-path and the `KeyError` case should be updated in tests.

---

## Known Deviations

- Missing targets are now caught at the input boundary: `to_feature_frame` raises a descriptive `ValueError` (not a bare pandas `KeyError`).
- `FeatureFrame` storage is `float32` by design (C-32), so the carried-forward last observation is `float32`-rounded relative to the raw `float64` value. The entity/tail selection **order** is preserved exactly, and a DataFrame and the `FeatureFrame` built from it produce identical output. This is an accepted precision trade-off.
- Entity filtering in `predict()` is performed against `last_observations` via `filter_entities`, which drops (with a `WARNING`) any entity present at `train_end` but absent from the fitted state. Such entities never reach `value_fn`, so no `KeyError` arises there.
