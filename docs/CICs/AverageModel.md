# Class Intent Contract: AverageModel

**Date:** 2026-03-13
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-001, ADR-005, ADR-006, ADR-009

---

## Purpose

`AverageModel` is a windowed-mean baseline: it computes the arithmetic mean of the last `window_months` training observations per entity per target, then carries that mean forward as a constant prediction across the forecast horizon. It represents the hypothesis that the near-term trend is best summarised by recent mean activity, and is a stronger baseline than LOCF when individual observations are noisy.

---

## Non-Goals

- Does not weight observations by recency (e.g., no exponential weighting).
- Does not produce distributional output. The class attribute `distributional` is absent.
- Does not validate that `window_months` is positive or that it is less than the available training history length.

---

## Responsibilities and Guarantees

- `fit(df)` filters to `time_idx < test_start`, sorts by `[entity_idx, time_idx]`, then applies `groupby(entity_idx).apply(lambda g: g.tail(window_months)[targets].mean())`. The result is stored as `self.mean` (a DataFrame indexed by entity, columns = targets). Returns `self`.
- `predict(df, sequence_number, output_length)` determines `loa_ids` from rows at `train_end`, filters to those present in `self.mean`, and returns a prediction grid where every cell for entity `cid` and target `t` has the value `self.mean.loc[cid, t]`.
- If entities in `loa_ids` are absent from `self.mean`, a `WARNING` is logged with the count of dropped entities.
- Output DataFrame structure contract is identical to `ZeroModel` and `LocfModel`: MultiIndex `(time_idx, entity_idx)`, columns `pred_{target}`, sorted.

---

## Inputs and Assumptions

| Parameter | Type | Notes |
|---|---|---|
| `targets` | `List[str]` | Must exist as columns in `df`. |
| `window_months` | `int` | Number of trailing months to average. Must be positive (not validated). |
| `partition_dict` | `dict` | Must contain `"test"` key with tuple `(test_start, test_end)`. |
| `loa` | `str` | Stored for logging; not used in computation. |
| `df` (fit/predict) | `pd.DataFrame` | 2-level MultiIndex; level 0 = time, level 1 = entity. |
| `sequence_number` | `int` | Forecast window start offset from `test_start`. |
| `output_length` | `int` | Number of forecast timesteps. Required — no default; passed from `config["time_steps"]` by the manager. |

When `window_months` exceeds the number of training rows available for an entity, `tail()` silently returns all available rows. The mean is computed over fewer observations than requested without any warning.

---

## Outputs and Side Effects

- **`fit()`**: Returns `self`. Sets `self.time_idx`, `self.entity_idx`, `self.mean`. Emits an `INFO` log. No external side effects.
- **`predict()`**: Returns a `pd.DataFrame`. Emits an `INFO` log and, if entities are dropped, a `WARNING`. No external side effects.

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| Target column missing from `df` | `KeyError` (crash) | No validation. |
| `df` missing 2-level MultiIndex | `IndexError` (crash) | No validation. |
| `partition_dict` missing `"test"` | `KeyError` (crash) | No validation. |
| `window_months <= 0` | Undefined behaviour | `tail(0)` returns empty; `.mean()` on empty returns `NaN`. Predictions silently become `NaN`. |
| `window_months` exceeds entity history length | Silent degradation | Mean computed over all available rows; no warning emitted. |
| Entities absent from `self.mean` at predict time | `WARNING` log | Entity dropped from output. |
| All entities dropped | Silent empty DataFrame | Valid empty result from `build_prediction_grid`. |

---

## Boundaries and Interactions

- **Depends on:** `views_baseline.model.helpers.build_prediction_grid`.
- **No external dependencies** beyond standard library, pandas, and numpy (transitively via pandas).
- **Instantiated by:** `BaselineModelCatalog._get_average_model()`, which reads `config["window_months"]`.

---

## Examples of Correct Usage

```python
from views_baseline.model.baseline import AverageModel

partition_dict = {"test": (493, 528)}
model = AverageModel(
    targets=["y1", "y2"],
    window_months=6,
    partition_dict=partition_dict,
    loa="pg_id",
)
model.fit(df)
preds = model.predict(df=df, sequence_number=0, output_length=36)

# Verify one cell
train_df = df[df.index.get_level_values("month_id") < 493]
train_df = train_df.sort_index(level=["pg_id", "month_id"])
expected_mean = train_df.xs(1, level="pg_id").tail(6)["y1"].mean()
assert preds.loc[(493, 1), "pred_y1"] == pytest.approx(expected_mean)
```

---

## Examples of Incorrect Usage

```python
# window_months=0 silently produces NaN predictions
model = AverageModel(targets=["y1"], window_months=0, partition_dict=partition_dict, loa="pg_id")
model.fit(df)
preds = model.predict(df=df, sequence_number=0)
# preds contains NaN — no error is raised

# window_months larger than training length: no error, just shorter window
model = AverageModel(targets=["y1"], window_months=10000, partition_dict=partition_dict, loa="pg_id")
model.fit(df)   # uses all available training rows; no warning

# Target not in DataFrame
model = AverageModel(targets=["nonexistent"], window_months=6, partition_dict=partition_dict, loa="pg_id")
model.fit(df)   # KeyError: "nonexistent"
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
- Adding a validation check `if window_months <= 0: raise ValueError(...)` in `__init__` or `fit()` would be a safe, backwards-compatible improvement with no behaviour change for valid inputs.

---

## Known Deviations

- No validation that `window_months > 0`. Passing `window_months=0` silently produces `NaN` predictions without any log message.
- When `window_months` exceeds the available entity history, `tail()` silently returns fewer rows than requested. No warning is emitted, so callers have no way to detect that the window was truncated.
