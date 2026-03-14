# Class Intent Contract: ZeroModel

**Date:** 2026-03-13
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-001, ADR-005, ADR-006

---

## Purpose

`ZeroModel` is the simplest conceivable baseline: it returns a constant 0.0 for every prediction cell across the full forecast horizon. It exists to provide a lower-bound reference point against which all other baselines are benchmarked. It also validates the prediction pipeline end-to-end without introducing any learned behaviour that could mask infrastructure problems.

---

## Non-Goals

- Does not learn from training data in any sense. `fit()` records only the index names from the DataFrame; no statistics are computed.
- Does not produce distributional output. The class attribute `distributional` is absent; the manager dispatches it through the point-model code path.
- Does not perform any entity filtering beyond reading which entities were present at `train_end`.

---

## Responsibilities and Guarantees

- `fit(df)` records `time_idx` and `entity_idx` from `df.index.names[0]` and `df.index.names[1]` respectively. Returns `self`.
- `predict(df, sequence_number, output_length)` constructs the entity list from rows where `time_idx == train_end` (where `train_end = partition_dict["test"][0] - 1`), then builds and returns a prediction grid via `build_prediction_grid` with `value_fn = lambda cid, target: 0.0`.
- The returned DataFrame always has columns `pred_{target}` for each target in `self.targets`, with all values exactly `0.0`.
- The time range in the output is `[test_start + sequence_number, test_start + sequence_number + output_length - 1]` inclusive.
- The MultiIndex of the output has names `[time_idx, entity_idx]` and is sorted.

---

## Inputs and Assumptions

| Parameter | Type | Notes |
|---|---|---|
| `targets` | `List[str]` | Column names in `df` that should be predicted. |
| `partition_dict` | `dict` | Must contain key `"test"` mapping to a tuple `(test_start, test_end)`. |
| `loa` | `str` | Level-of-analysis string (e.g. `"pg_id"`, `"country_id"`). Stored but not used beyond logging. |
| `df` (fit) | `pd.DataFrame` | Must have a 2-level MultiIndex. Level 0 is the time index, level 1 is the entity index. |
| `df` (predict) | `pd.DataFrame` | Same DataFrame passed to `fit()`; used only to read entity IDs at `train_end`. |
| `sequence_number` | `int` | Offset from `test_start` at which the forecast window begins. |
| `output_length` | `int` | Number of time steps to forecast. Required — no default; passed from `config["time_steps"]` by the manager. |

`fit()` must be called before `predict()`. If called in the wrong order, `time_idx` and `entity_idx` will be `None` and `predict()` will raise an `AttributeError` internally.

---

## Outputs and Side Effects

- **`fit()`**: Returns `self`. Sets `self.time_idx` and `self.entity_idx`. No external side effects.
- **`predict()`**: Returns a `pd.DataFrame` with MultiIndex `(time_idx, entity_idx)` and columns `pred_{target}` for all targets. All prediction values are `0.0`. Emits an `INFO` log line via module logger.

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| `df` passed to `fit()` or `predict()` lacks a 2-level MultiIndex | `IndexError` (crash) | No explicit validation is performed. |
| `partition_dict` missing `"test"` key | `KeyError` (crash) | No explicit validation. |
| No entities present at `train_end` | Silent empty DataFrame | `build_prediction_grid` returns a valid empty DataFrame with correct columns and index names. |

The model never emits `WARNING` or `ERROR` log messages.

---

## Boundaries and Interactions

- **Depends on:** `views_baseline.model.helpers.build_prediction_grid` — the only function called by `predict()`.
- **No external dependencies** beyond the standard library and pandas.
- **Not imported by models.** Instantiated exclusively through `BaselineModelCatalog._get_zero_model()`.
- Does not interact with `views_pipeline_core` directly.

---

## Examples of Correct Usage

```python
from views_baseline.model.baseline import ZeroModel

partition_dict = {"test": (493, 528)}
model = ZeroModel(targets=["y1", "y2"], partition_dict=partition_dict, loa="pg_id")
model.fit(df)                                         # records index names
preds = model.predict(df=df, sequence_number=0)       # returns zero-filled DataFrame
assert (preds.values == 0.0).all()
```

Using a non-zero `sequence_number` for rolling evaluation:

```python
preds_seq2 = model.predict(df=df, sequence_number=2, output_length=36)
# time range starts at 493 + 2 = 495
```

---

## Examples of Incorrect Usage

```python
# Calling predict before fit
model = ZeroModel(targets=["y1"], partition_dict=partition_dict, loa="pg_id")
model.predict(df=df, sequence_number=0)   # AttributeError: time_idx is None

# Passing a flat (non-MultiIndex) DataFrame
flat_df = df.reset_index()
model.fit(flat_df)                        # IndexError: index.names[1] does not exist

# Passing a partition_dict without "test"
bad_model = ZeroModel(targets=["y1"], partition_dict={"train": (1, 492)}, loa="pg_id")
bad_model.fit(df)
bad_model.predict(df=df, sequence_number=0)   # KeyError: "test"
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
- If explicit pre-fit validation is added, it should raise `ValueError` with a clear message rather than letting a downstream `IndexError` propagate, and tests should be updated accordingly.

---

## Known Deviations

- No input validation on DataFrame structure. Structural errors surface as `IndexError` or `KeyError` rather than a descriptive `ValueError`.
- Entity filtering is based on which entities appear at the single row `train_end`. Entities present in training but absent from that specific timestep are silently excluded from all predictions. This matches the behaviour of all other point models in this package but is not explicitly documented at the call site.
