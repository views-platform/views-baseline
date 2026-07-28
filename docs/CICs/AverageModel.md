# Class Intent Contract: AverageModel

**Date:** 2026-03-17
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-001, ADR-005, ADR-006, ADR-009, ADR-019, ADR-020

---

## Purpose

`AverageModel` is a windowed-mean baseline: it computes the arithmetic mean of the last `window_months` training observations per entity per target, then carries that mean forward as a constant prediction across the forecast horizon. It represents the hypothesis that the near-term trend is best summarised by recent mean activity, and is a stronger baseline than LOCF when individual observations are noisy.

---

## Non-Goals

- Does not weight observations by recency (e.g., no exponential weighting).
- Does not produce distributional output. Returns `dict[str, PredictionFrame]` with `y_pred` shape `(N, 1)` — a single deterministic value per cell, not multiple samples.
- Does not validate that `window_months` is less than the available training history length. (A non-positive `window_months` **is** now rejected — see Failure Modes.)

---

## Responsibilities and Guarantees

- `fit(df)` normalizes the input via `to_feature_frame(df, loa, targets)` (the ADR-019 pandas boundary), records `time_idx`/`entity_idx` from `ff.index.level.index_names`, and computes the per-entity trailing-window mean as the **mean of the shared `window_pool(ff, targets, window_months, train_end)`** on the frame's numpy panel (no pandas). The result is stored as `self.mean`, a nested dict `{entity -> {target -> trailing-window mean}}` (not a pandas object). Returns `self`.
- `predict(df, sequence_number, output_length)` normalizes the input, determines `entity_ids` via `entities_at(unit, time, train_end)` (first-appearance order), filters to those present in `self.mean` (`filter_entities`), and returns `dict[str, PredictionFrame]` via `build_prediction_frame` (with `level = ff.index.level`) where every cell for entity `cid` and target `t` has the value `self.mean[cid][t]`.
- If entities in `entity_ids` are absent from `self.mean`, a `WARNING` is logged with the count of dropped entities.
- Output structure contract is identical to all baseline models: one key per target, each value a `PredictionFrame` with `identifiers` dict containing `"time"` and `"unit"` arrays.

---

## Inputs and Assumptions

| Parameter | Type | Notes |
|---|---|---|
| `targets` | `List[str]` | Must exist as columns in `df`. |
| `window_months` | `int` | Number of trailing months to average. Must be `>= 1` — a non-positive value fails loud at fit (`window_pool`). |
| `partition_dict` | `dict` | Must contain `"test"` key with tuple `(test_start, test_end)`. |
| `loa` | `str` | Declared level of analysis; resolved to a `SpatialLevel` and validated against the index by `to_feature_frame`. |
| `df` (fit/predict) | `pd.DataFrame \| FeatureFrame` | Normalized at entry via `to_feature_frame`. A DataFrame needs a 2-level MultiIndex (level 0 = time, level 1 = entity) matching `loa`; a `FeatureFrame` must carry the matching `level` and the `targets` features. |
| `sequence_number` | `int` | Forecast window start offset from `test_start`. |
| `output_length` | `int` | Number of forecast timesteps. Required — no default; passed from `config["time_steps"]` by the manager. |

When `window_months` exceeds the number of training rows available for an entity, the numpy tail slice (`block[-window_months:]`) silently returns all available rows. The mean is computed over fewer observations than requested without any warning.

---

## Outputs and Side Effects

- **`fit()`**: Returns `self`. Sets `self.time_idx`, `self.entity_idx`, `self.mean`. Emits an `INFO` log. No external side effects.
- **`predict()`**: Returns `dict[str, PredictionFrame]` with `y_pred` shape `(N, 1)`. Emits an `INFO` log and, if entities are dropped, a `WARNING`. No external side effects.

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| Target column missing / feature absent from a `FeatureFrame` | `ValueError` (crash) | `to_feature_frame` checks target presence at the input boundary (ADR-019). |
| DataFrame index names do not match `loa` (incl. non-2-level index) | `ValueError` (crash) | `to_feature_frame` → `resolve_level` raises at entry. |
| `partition_dict` missing `"test"` | `KeyError` (crash) | No validation. |
| `window_months <= 0` | `ValueError` at **fit** (crash) | `window_pool` fails loud with `"window_months must be >= 1, got ..."`. Previously this was silent all-`NaN` predictions — now caught deliberately (ADR-019). |
| `window_months` exceeds entity history length | Silent degradation | Mean computed over all available rows; no warning emitted. |
| Entities absent from `self.mean` at predict time | `WARNING` log | Entity dropped from output. |
| All entities dropped | `ValueError` (crash) | `require_entities` raises a descriptive error. Fail-loud: a prediction over zero entities cannot satisfy the evaluation contract (an empty result would otherwise surface as a `StopIteration` deep in pipeline-core). |

---

## Boundaries and Interactions

- **Depends on:** `views_baseline.model.frames.input` (`to_feature_frame`, `panel` — the single pandas boundary, ADR-019), `views_baseline.model.frames.pooling.window_pool` (numpy windowing), `views_baseline.model.frames.output.build_prediction_frame`, and `views_baseline.model.grid` (`entities_at`, `filter_entities`, `require_entities`, `build_time_grid`). pandas is not a runtime dependency of this model — it is confined to `to_feature_frame` and imported only under `TYPE_CHECKING`.
- **External dependency:** `views_frames` — `FeatureFrame`/`SpatioTemporalIndex` lazy-imported inside `to_feature_frame`; `PredictionFrame`/`SpatioTemporalIndex` inside the `to_prediction_frames` seam, at call time.
- **Instantiated by:** `BaselineModelCatalog._get_average_model()`, which reads `config["window_months"]`.

---

## Examples of Correct Usage

```python
from views_baseline.model.models.point import AverageModel

partition_dict = {"test": (493, 528)}
model = AverageModel(
    targets=["y1", "y2"],
    window_months=6,
    partition_dict=partition_dict,
    loa="pgm",
)
model.fit(df)                # model.fit(ff) also works — dual input
preds = model.predict(df=df, sequence_number=0, output_length=36)

# Fitted state is a nested dict of trailing-window means (float32-rounded via the
# FeatureFrame — see C-32 note in Known Deviations).
expected_mean = <mean of entity 1's last 6 y1 observations before test_start>
assert model.mean[1]["y1"] == pytest.approx(expected_mean)

# preds is dict[str, PredictionFrame]; every cell of entity 1's y1 frame carries that mean.
pf = preds["y1"]
```

---

## Examples of Incorrect Usage

```python
# window_months=0 now fails loud at fit (previously it silently produced NaN predictions)
model = AverageModel(targets=["y1"], window_months=0, partition_dict=partition_dict, loa="pgm")
model.fit(df)   # ValueError: window_months must be >= 1, got 0.

# window_months larger than training length: no error, just shorter window
model = AverageModel(targets=["y1"], window_months=10000, partition_dict=partition_dict, loa="pgm")
model.fit(df)   # uses all available training rows; no warning

# Target not in DataFrame
model = AverageModel(targets=["nonexistent"], window_months=6, partition_dict=partition_dict, loa="pgm")
model.fit(df)   # ValueError: DataFrame is missing required target column(s) ['nonexistent']
```

---

## Test Alignment

File: `tests/test_baseline.py`

| Test | What it verifies |
|---|---|
| `test_average_model_uses_mean_of_last_n_months` | Prediction values equal the independently computed mean of the last `window_months` training rows per entity; full output structure (columns, index names, time range) is verified. |
| `test_average_model_respects_sequence_number` | Output time range shifts correctly with `sequence_number=2`. |

---

## Evolution Notes

- If recency-weighted averaging is added, it should be a separate class (e.g., `ExponentialAverageModel`) rather than a parameter on this class, to keep the `AverageModel` contract simple.
- The `window_months <= 0` guard is now enforced centrally in `window_pool`/`window_pool_arrays` (raising `ValueError` at fit), shared by all pooling baselines — no per-class check needed.

---

## Known Deviations

- `window_months <= 0` now fails loud with a `ValueError` at fit (via `window_pool`). Previously it silently produced `NaN` predictions with no log message — that degenerate path was closed in PR-2 (ADR-019).
- `FeatureFrame` storage is `float32` by design (C-32), so the pooled values (and thus the trailing-window mean) are `float32`-rounded relative to raw `float64`. The entity/tail selection **order** is preserved exactly, and a DataFrame and the `FeatureFrame` built from it produce identical output — an accepted precision trade-off.
- When `window_months` exceeds the available entity history, the numpy tail slice silently returns fewer rows than requested. No warning is emitted, so callers have no way to detect that the window was truncated.
