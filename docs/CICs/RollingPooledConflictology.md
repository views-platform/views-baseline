# Class Intent Contract: RollingPooledConflictology

**Date:** 2026-11-14
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-008, ADR-011, ADR-019, ADR-021, ADR-024

---

## Purpose

`RollingPooledConflictology` is `PooledConflictology` with one change: **at rolling origin `k` the area
windows end at `train_end + k`**, not at `train_end`.

That is Matryoshka's information rule, and `RollingConflictology`'s (#126). A comparison of a single level
with Matryoshka then differs only in the level, not in how recent the data is (#165). `PooledConflictology`
draws every origin from `train_end`, so it compares with Matryoshka only at origin 0 (ADR-024, amendment
2026-11-14).

---

## Non-Goals

- **Does not move the area assignment.** A cell keeps the area `fit` read up to `train_end`. Codes after
  `train_end` are never read, so the window is the only thing that moves (decided on #165).
- Does not refit anything per origin: there is nothing to fit.
- Does not mix levels, weight by distance, or weight months by recency (as `PooledConflictology`).

---

## Responsibilities and Guarantees

- `__init__` and `fit` are `PooledConflictology`'s, unchanged. That covers the `pool_by` checks, the area
  codes, `NO_AREA`, and the WARNING.
- `predict(df, sequence_number=k, output_length)`:
  - `k == 0`: draws from the fitted pools. The output is **byte-identical** to `PooledConflictology`'s.
  - `k > 0`: takes each forecast cell's last `window_months` rows up to `train_end + k`, after checking
    the months after `train_end` (below), with `frames.pooling.rolling_window_pools`, the function
    `RollingConflictology` uses too. It then pools them by the fitted areas (`PooledConflictology._pools`).
  - Draws as `PooledConflictology` does (`rng.choice` through `sample_prediction_grid`, seeded by `seed`)
    and returns `dict[str, PredictionFrame]` of shape `(N · output_length, n_samples)`.
- With every cell in its own area, the output equals `RollingConflictology`'s at every origin.
- **Common random numbers:** the same seed picks the same pool positions at every origin, as long as every
  pool keeps its length, which means every forecast cell has a full window.

---

## Inputs and Assumptions

| Parameter | Type | Notes |
|---|---|---|
| `targets`, `window_months`, `partition_dict`, `loa`, `n_samples`, `pool_by`, `seed` | | As `PooledConflictology`. |

**Assumption:** the `df` passed to `predict` holds every forecast cell's targets up to `train_end + k`.
pipeline-core passes the full data at every origin.

**Cost:** the per-origin rebuild of the pools is small next to the drawing. On a synthetic pgm-sized panel
(65,000 cells, 3 targets, `window_months` 36, S = 64), `predict` took about 56 s with `"map"` and about
52 s with a country column. The time was the same at origins 0, 1 and 12, so the rebuild costs well under a
second (#167; laptop timing, 2026-11-14).

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| A forecast cell is missing a month after `train_end` inside the window, `max(train_end+1, train_end+k−window_months+1) .. train_end+k` | `ValueError` naming the model, the cells and the first missing months | Shared with `RollingConflictology` (`rolling_window_pools`). A gap would silently pull older months back into the window. |
| NaN in a target in those months | `ValueError` naming the target, the cell and the months | The same check. |
| `sequence_number < 0` | `ValueError` naming the model | |
| Everything `PooledConflictology` refuses | As `PooledConflictology` | Inherited; the messages and the log lines name `RollingPooledConflictology` (`model_name`). |

---

## Boundaries and Interactions

- **Depends on:** `PooledConflictology` (subclass; `_pools`), `frames.pooling.rolling_window_pools`,
  `frames.input.to_level`, `frames.output.sample_prediction_grid`.
- **Instantiated by:** `BaselineModelCatalog._get_rolling_pooled_conflictology()`.
- **Genome:** `["window_months", "n_samples", "seed", "pool_by"]`, the same as `PooledConflictology`'s.
  The moving window is the class, not a key, so the existing `PooledConflictology` configs are unchanged.

---

## Examples of Correct Usage

```python
from views_baseline.model.models.distributional import RollingPooledConflictology

m = RollingPooledConflictology(
    targets=["lr_sb_best"], window_months=36, partition_dict={"test": (457, 504)},
    loa="pgm", n_samples=64, pool_by="c_id", seed=42,
).fit(df)                        # df carries lr_sb_best and c_id
pf = m.predict(df=df, sequence_number=5, output_length=36)["lr_sb_best"]   # window ends at 461
```

## Examples of Incorrect Usage

```python
# The data stops at train_end: origin 1 raises instead of silently using the train_end window.
m.predict(df=df_up_to_train_end, sequence_number=1, output_length=36)
```

---

## Test Alignment

All in `tests/test_rolling_pooled_conflictology.py` unless noted. The input encodes each value's source
(`y1 = month × 100 + cell`).

| Test | What it verifies |
|---|---|
| `test_each_cell_draws_from_all_of_its_areas_window_at_the_origin_and_nothing_else` (k = 1, 3, 12) | Exactly the area's windows ending at `train_end + k`; −1 cells use only their own; −3727 is a real area. |
| `test_map_pools_every_cells_window_at_the_origin` | `"map"` at origins 1 and 12. |
| `test_the_window_really_moves` | The newest month drawn is `train_end + k`, not `train_end`. |
| `test_origin_zero_is_pooled_conflictology_byte_for_byte` | Origin 0 equals `PooledConflictology`. |
| `test_one_cell_areas_are_rolling_conflictology_at_every_origin` | One cell per area equals `RollingConflictology` at k = 0, 1, 5, 12. |
| `test_areas_stay_as_fitted_at_train_end` | An area change after `train_end` is not read. |
| `test_a_gap_after_train_end_inside_the_window_raises_naming_the_cell` | A missing row and a NaN each raise, and only once inside the window. |
| `test_negative_sequence_number_raises`, `test_dataframe_and_featureframe_give_the_same_draws` | Bad origin; both input types. |
| `test_the_catalog_requires_pool_by_and_forwards_every_key` | No silent default; every key forwarded. |
| `test_every_model_in_the_family_names_itself_in_its_output_errors`, `test_rolling_conflictology_names_itself_in_its_origin_errors`, `test_pooled_conflictology_names_itself_in_its_errors_and_logs` | `model_name` reaches every error, log line and output path of the four conflictology models. |
| `test_the_shared_window_read_finds_a_gap_even_with_unsorted_entity_ids` | `rolling_window_pools` does not depend on the order of `entity_ids`. |
| `test_forecasts_are_labelled_with_the_origins_months` | Origin k's forecasts carry months `test_start + k ...`, not origin 0's. |
| `test_a_cell_that_first_appears_after_train_end_is_neither_forecast_nor_pooled` | The forecast cells are fixed at `fit`, at every origin. |
| `test_the_area_pool_is_its_cells_windows_in_ascending_cell_order` | The exact pool array, so a seed draws the same values. |
| `test_a_nan_after_train_end_names_the_first_cell_its_months_and_the_cell_count`, `test_a_gap_earlier_in_the_window_raises`, `test_inherited_errors_and_warnings_name_this_model` | The errors name the first bad cell and this model. |
| Shared lists in `test_protocol.py`, `test_dual_input.py`, `test_catalog.py`, `test_falsification_seed_wiring.py`, `test_reproducibility_gate.py` (incl. `test_rolling_pooled_conflictology_genome_is_exact`), `test_config_conformance.py`, `test_baseline.py` | Protocol, input types, catalog, seed wiring, exact genome, a derived conformance fixture. |

---

## Evolution Notes

- **Evolving** (ADR-004).
- The conformance fixture is derived (light_strider with `algorithm` swapped and `pool_by: "c_id"`) until
  views-models ships a config for this model (#171).
