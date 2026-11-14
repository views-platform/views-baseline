# Class Intent Contract: PooledConflictology

**Date:** 2026-11-12
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-008, ADR-011, ADR-019, ADR-021, ADR-024

---

## Purpose

`PooledConflictology` is `ConflictologyModel` with one change: **a cell draws from the window of every
cell in its area**, not only from its own.

The area is set by the `pool_by` column: for example the country (`c_id`) or the admin-1 unit. With
`pool_by="map"`, every cell is in one area. The window is conflictology's own: each cell contributes the
last `window_months` up to `train_end` that `ConflictologyModel` would draw from.

It exists for ADR-024's comparisons (#151). A reviewer will ask whether **one level on its own**, for
example just the country or just admin 1, does as well as Matryoshka. Each such level is this model with a
different `pool_by`.

---

## Non-Goals

- Does not mix levels. Whether *adding* a level helps is a question for Matryoshka's dolls (#150).
- Does not weight cells by distance or months by recency.
- Does not move its window with the forecast origin. Like `ConflictologyModel`, it draws every origin from
  the window ending at `train_end`.
- Does not shrink a cell's own history towards the area's (#112 does).

---

## Responsibilities and Guarantees

- `fit(df)`:
  - pools the targets with the shared `window_pool_arrays`, so each cell's rows are exactly
    `ConflictologyModel`'s;
  - reads each forecast cell's area code from the `pool_by` column, in every month up to `train_end`. The
    code must be a finite integer and the same in every such month;
  - builds one pool per area and target: the windows of the area's forecast cells, concatenated in
    ascending cell order. The cells of an area share that one array (no copy per cell);
  - sets `self.entity_ids` (the cells present at `train_end`), `self.area` (cell → code) and
    `self.hist_per_entity` (cell → its area's pool).
- **A cell without an area** (code `NO_AREA = -1`, the VIEWS data's code for no country and for no admin-1
  unit) draws from its own window, as `ConflictologyModel` does. `fit` logs one WARNING with how many
  such cells there are (ADR-008). Other negative codes are real areas: the Azores admin-1 codes are
  −3727 to −3730 (views-datafactory ADR-043).
- `predict(df, sequence_number, output_length)` draws as `ConflictologyModel` does (`rng.choice` over the
  pool, through `sample_prediction_grid`, seeded by `seed`, entity → time → target) and returns
  `dict[str, PredictionFrame]` of shape `(N · output_length, n_samples)`.
- With every cell in its own area, the output is byte-identical to `ConflictologyModel`'s.

---

## Inputs and Assumptions

| Parameter | Type | Notes |
|---|---|---|
| `targets`, `window_months`, `partition_dict`, `loa`, `n_samples`, `seed` | | As `ConflictologyModel`. |
| `pool_by` | `str` | The area column, or `"map"` (no column needed). Must not be a target. No default (ADR-021). |

**Assumption:** the `pool_by` column is in the input that `fit` gets. The country column (`c_id`, from
views-datafactory's `gaul0_code`) is already in views-models' pgm input. For admin 1, views-datafactory
already provides `gaul1_code`; a views-models queryset adds it in one line, next to `gaul0_code`
(views-models#728).

**Cost:** one pool per area. With `"map"` the pool holds every forecast cell's window: at pgm `land`, about
65,000 × 36 values per target, about 19 MB in float64. Drawing costs the same as `ConflictologyModel`.

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| `pool_by` empty, not a string, or a target | `ValueError` at construction | |
| The `pool_by` column is missing from the input | `ValueError` from `panel` naming the column | |
| An area code is NaN, not an integer, or 2**24 or more in size, up to `train_end` | `ValueError` naming the column, a cell, its value and the number of rows | The input is stored as float32, which holds integers exactly only below 2**24; larger codes could merge two areas silently. Real codes are far below (GAUL admin-1: −3730 to 4109). |
| A cell's area code changes up to `train_end` | `ValueError` naming the count and the first cell | A cell must belong to one area. Changes after `train_end` are not read. |
| A cell has no area (−1) | WARNING with the count; the cell draws from its own window | Simon's decision, recorded on #151 (comment of 2026-11-12): mostly small islands, 76 cells for `c_id`. |
| NaN in a target in the window, `window_months <= 0`, bad input | `ValueError` | As `ConflictologyModel` (`tail_pools`, `to_feature_frame`). |

---

## Boundaries and Interactions

- **Depends on:** `frames.pooling.window_pool_arrays`, `frames.input` (`panel`, `to_feature_frame`,
  `to_level`), `frames.output.sample_prediction_grid`, `grid.train_test_boundary`. Subclass of
  `ConflictologyModel` (its `__init__` checks are inherited).
- **Instantiated by:** `BaselineModelCatalog._get_pooled_conflictology()`.
- **Genome:** `["window_months", "n_samples", "seed", "pool_by"]`.

---

## Examples of Correct Usage

```python
from views_baseline.model.models.distributional import PooledConflictology

m = PooledConflictology(
    targets=["lr_sb_best"], window_months=36, partition_dict={"test": (457, 504)},
    loa="pgm", n_samples=64, pool_by="c_id", seed=42,
).fit(df)                       # df carries lr_sb_best and c_id
pf = m.predict(df=df, sequence_number=0, output_length=36)["lr_sb_best"]
```

## Examples of Incorrect Usage

```python
# The area column is not in the data: fit raises, naming 'admin1_code'.
PooledConflictology(..., pool_by="admin1_code").fit(df_without_admin1)
```

---

## Test Alignment

All in `tests/test_pooled_conflictology.py` unless noted. The test input encodes each value's source
(`y1 = month × 100 + cell`), so the cell and month every draw came from can be read back exactly.

| Test | What it verifies |
|---|---|
| `test_each_cell_draws_from_all_of_its_areas_window_and_nothing_else` | Every cell draws from exactly its area's window, all of it; a negative code (−3727) is a real area. |
| `test_cells_without_an_area_draw_from_their_own_window_and_are_warned_about`, `test_no_warning_when_every_cell_has_an_area` | Two −1 cells each draw only from their own window (not pooled together); one WARNING with the count, and none when every cell has an area. |
| `test_map_pools_every_cell_and_needs_no_area_column` | `"map"` pools every cell and reads no area column. |
| `test_window_stays_at_train_end_for_later_origins` | Later origins still draw from the window ending at `train_end`. |
| `test_cells_of_one_area_share_one_pool_not_copies` | One array per area, shared; the pool size is cells × window. |
| `test_a_cell_absent_at_train_end_is_neither_forecast_nor_pooled` | Only forecast cells enter a pool. |
| `test_a_one_cell_area_is_conflictology` | With one cell per area, the output equals `ConflictologyModel`'s exactly. |
| `test_dataframe_and_featureframe_give_the_same_draws` | Both input types give the same draws, with an area column. |
| `test_an_area_code_that_is_not_an_integer_raises_naming_the_cell` (NaN, 10.5, 2**24; month 490 and `train_end`) | Bad codes raise and name the cell, up to and including `train_end`. |
| `test_a_cell_that_changes_area_raises`, `test_area_changes_after_train_end_are_ignored` | One area per cell, up to `train_end` only. |
| `test_a_missing_area_column_raises`, `test_pool_by_must_be_a_column_that_is_not_a_target` | Bad `pool_by`. |
| `test_the_catalog_requires_pool_by` | A config without `pool_by` is rejected; no silent default (ADR-021). |
| Shared lists in `test_protocol.py`, `test_dual_input.py`, `test_catalog.py`, `test_falsification_seed_wiring.py`, `test_reproducibility_gate.py` (incl. `test_pooled_conflictology_genome_is_exact`), `test_config_conformance.py`, `test_baseline.py` | Protocol, both input types, catalog forwarding, seed wiring, exact genome, a derived conformance fixture. |

---

## Evolution Notes

- **Evolving** (ADR-004).
- The conformance fixture is derived (light_strider with `algorithm` swapped and `pool_by: "c_id"`) until
  views-models ships a config for this model.
