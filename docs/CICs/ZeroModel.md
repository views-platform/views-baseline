# Class Intent Contract: ZeroModel

**Date:** 2026-03-13
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-001, ADR-005, ADR-006, ADR-019, ADR-020

---

## Purpose

`ZeroModel` is the simplest conceivable baseline: it returns a constant 0.0 for every prediction cell across the full forecast horizon. It exists to provide a lower-bound reference point against which all other baselines are benchmarked. It also validates the prediction pipeline end-to-end without introducing any learned behaviour that could mask infrastructure problems.

---

## Non-Goals

- Does not learn from training data in any sense. `fit()` records only the `(time, entity)` index names from the normalized `FeatureFrame`; no statistics are computed. `predict()` does not depend on `fit()` state at all — it is stateless (see Responsibilities).
- Does not produce distributional output. Returns `dict[str, PredictionFrame]` with `y_pred` shape `(N, 1)` — a single deterministic value per cell, not multiple samples.
- Does not perform any entity filtering beyond reading which entities were present at `train_end`.

---

## Responsibilities and Guarantees

- `fit(df)` normalizes the input via `to_feature_frame(df, loa, targets)` (the ADR-019 pandas boundary) and records `time_idx` and `entity_idx` from `ff.index.level.index_names`. Returns `self`.
- `predict(df, sequence_number, output_length)` normalizes the input via `to_feature_frame`, reads the `(time, unit)` numpy panel via `panel(ff, targets)`, and constructs the entity list with `entities_at(unit, time, train_end)` (first-appearance order; `train_end = partition_dict["test"][0] - 1`). It then builds and returns `dict[str, PredictionFrame]` via `build_prediction_frame` with `value_fn = lambda cid, target: 0.0` and `level = ff.index.level`. `predict()` derives everything from its input argument and reads no fitted state, so it does not require `fit()` to have run first.
- The returned dict has one key per target. Each `PredictionFrame` has `y_pred` shape `(N, 1)` with all values exactly `0.0`, and `identifiers` with `"time"` and `"unit"` arrays.
- The time range in the output is `[test_start + sequence_number, test_start + sequence_number + output_length - 1]` inclusive.

---

## Inputs and Assumptions

| Parameter | Type | Notes |
|---|---|---|
| `targets` | `List[str]` | Column names in `df` that should be predicted. |
| `partition_dict` | `dict` | Must contain key `"test"` mapping to a tuple `(test_start, test_end)`. |
| `loa` | `str` | Declared level of analysis (the `SpatialLevel` value, `"cm"` or `"pgm"`). Now **used**, not just logged: `to_feature_frame` resolves it to a `SpatialLevel` and validates the input's spatial level against it (ADR-003/ADR-020). |
| `df` (fit) | `pd.DataFrame \| FeatureFrame` | Normalized at entry via `to_feature_frame`. A DataFrame must have a 2-level MultiIndex (level 0 = time, level 1 = entity) whose names match the declared `loa`; a `FeatureFrame` must carry the matching `level` and the `targets` features. |
| `df` (predict) | `pd.DataFrame \| FeatureFrame` | Same structure; used only to read entity IDs at `train_end` (via `entities_at`). |
| `sequence_number` | `int` | Offset from `test_start` at which the forecast window begins. |
| `output_length` | `int` | Number of time steps to forecast. Required — no default; passed from `config["time_steps"]` by the manager. |

`predict()` is stateless — it does not require `fit()` to have run first (it reads no fitted state). If `predict()` is called before `fit()`, `entity_idx` is `None` and appears only as `None` in the INFO log line; the prediction itself is unaffected.

---

## Outputs and Side Effects

- **`fit()`**: Returns `self`. Sets `self.time_idx` and `self.entity_idx`. No external side effects.
- **`predict()`**: Returns `dict[str, PredictionFrame]`. Each PredictionFrame has `y_pred` shape `(N, 1)` with all values `0.0`, and `identifiers` dict with `"time"` and `"unit"` arrays. Emits an `INFO` log line via module logger.

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| DataFrame index names do not match the declared `loa` (incl. a flat / non-2-level index) | `ValueError` (crash) | `to_feature_frame` → `resolve_level` raises a descriptive error at entry (declarations are authoritative, ADR-003). |
| A passed `FeatureFrame`'s `level` disagrees with `loa`, or it lacks a required target | `ValueError` (crash) | `to_feature_frame` validates the frame at entry. |
| `partition_dict` missing `"test"` key | `KeyError` (crash) | No explicit validation. |
| No entities present at `train_end` | `ValueError` (crash) | `require_entities` raises a descriptive error naming the cause. Fail-loud: the evaluation path cannot proceed with zero entities (an empty result would otherwise surface as a `StopIteration` deep in pipeline-core). |

The model never emits `WARNING` or `ERROR` log messages.

---

## Boundaries and Interactions

- **Depends on:** `views_baseline.model.frames.input` (`to_feature_frame`, `panel` — the single pandas boundary, ADR-019), `views_baseline.model.frames.output.build_prediction_frame`, and `views_baseline.model.grid` (`entities_at`, `build_time_grid`, `require_entities`). pandas is not a runtime dependency of this model — it is confined to `to_feature_frame` and imported only under `TYPE_CHECKING` here.
- **External dependency:** `views_frames` — `FeatureFrame`/`SpatioTemporalIndex` are lazy-imported inside `to_feature_frame`, and `PredictionFrame`/`SpatioTemporalIndex` inside the `to_prediction_frames` seam, at call time.
- **Not imported by models.** Instantiated exclusively through `BaselineModelCatalog._get_zero_model()`.

---

## Examples of Correct Usage

```python
from views_baseline.model.models.point import ZeroModel

partition_dict = {"test": (493, 528)}
model = ZeroModel(targets=["y1", "y2"], partition_dict=partition_dict, loa="pgm")
model.fit(df)                                         # records index names (model.fit(ff) also works — dual input)
result = model.predict(df=df, sequence_number=0, output_length=36)
assert all(pf.y_pred == 0.0 for pf in result.values())  # all targets zero
```

Using a non-zero `sequence_number` for rolling evaluation:

```python
preds_seq2 = model.predict(df=df, sequence_number=2, output_length=36)
# time range starts at 493 + 2 = 495
```

---

## Examples of Incorrect Usage

```python
# Omitting output_length (a required argument)
model = ZeroModel(targets=["y1"], partition_dict=partition_dict, loa="pgm")
model.predict(df=df, sequence_number=0)   # TypeError: missing output_length
# (Note: predict does NOT require fit — it is stateless and derives everything from df.)

# Passing a flat (non-MultiIndex) DataFrame
flat_df = df.reset_index()
model.fit(flat_df)                        # ValueError: index names do not match declared loa

# Passing a partition_dict without "test"
bad_model = ZeroModel(targets=["y1"], partition_dict={"train": (1, 492)}, loa="pgm")
bad_model.predict(df=df, sequence_number=0, output_length=36)   # KeyError: "test"
```

---

## Test Alignment

File: `tests/test_baseline.py`

| Test | What it verifies |
|---|---|
| `test_zero_model_predicts_zeros_pgm` | All values are 0.0, columns match `pred_{target}`, time range covers `[test_start, test_start + output_length - 1]`, index names match `pg_id` loa. |
| `test_zero_model_predicts_zeros_cm` | Same assertions for `country_id` loa, confirming the model is loa-agnostic. |
| `test_zero_model_respects_sequence_number` | Output time range shifts by `sequence_number` relative to `test_start`. |

---

## Evolution Notes

- If a warm-start zero model is ever needed (e.g., zeroing only specific targets), a subclass is preferable to modifying `value_fn` in place, to keep this class's contract stable.
- Structural input validation now lives at the `to_feature_frame` boundary (ADR-019), which raises a descriptive `ValueError` for index/level/target mismatches. Any additional pre-fit checks should follow the same fail-loud-with-a-clear-message pattern, and tests should be updated accordingly.

---

## Known Deviations

- Structural input errors (index-name / level / target mismatch) now surface as a descriptive `ValueError` from `to_feature_frame` at the input boundary (ADR-019), rather than a bare pandas `IndexError`. A `partition_dict` missing `"test"` still surfaces as a bare `KeyError`.
- Because `FeatureFrame` storage is `float32` by design (C-32), pooled/observed magnitudes are `float32`-rounded — immaterial for `ZeroModel` (it emits a constant `0.0` regardless of the observed values), but noted for consistency with the other baselines.
- Entity filtering is based on which entities appear at the single row `train_end`. Entities present in training but absent from that specific timestep are silently excluded from all predictions. This matches the behaviour of all other point models in this package but is not explicitly documented at the call site.
