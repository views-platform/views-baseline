# Class Intent Contract: NeighbourhoodConflictology

**Date:** 2026-11-18
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-008, ADR-011, ADR-019, ADR-021, ADR-024

---

## Purpose

`NeighbourhoodConflictology` is conflictology whose window is pooled over a **disc of cells**: each cell's
forecast is drawn from the last `window_months` of every forecast cell within grid distance `radius` of
it, itself included, all with equal weight.

It is the single level for Matryoshka's `near` doll (ADR-024, amendment 2026-11-18). With `radius` 5 the
disc is that doll's neighbourhood plus the cell, because both come from `matryoshka.geometry.neighbour_table`.

---

## Non-Goals

- Does not weight by distance. The kernel is Matryoshka's method, not the level.
- Does not mix levels, or weight months by recency.
- Does not move the window with the rolling origin: that is `RollingNeighbourhoodConflictology`.
- No country-month version: the disc comes from priogrid ids.

---

## Responsibilities and Guarantees

- `__init__` requires `radius`, an integer in 0..359. A bool, float, string, None, negative value or
  360 and above raises. From 360 on, the disc would reach round the date line onto itself and count a
  cell twice.
- `fit(df)`:
  - raises unless `loa` is pgm;
  - takes conflictology's windows (`ConflictologyModel.fit`, `window_pool`);
  - builds each cell's disc from the forecast cells, in ascending cell order.
- `predict(df, sequence_number, output_length)` draws `n_samples` values per cell, month and target
  through `sample_prediction_grid`, seeded by `seed`.
- **The draws are those of `rng.choice` over the disc's windows concatenated in ascending cell order.**
  The pool is not built. `rng.choice(total)` uses the random stream exactly as `rng.choice(pool)` does,
  and each position is mapped to its cell and month. Memory: one (N, `window_months`) array per target.
- `radius` 0 is `ConflictologyModel` byte for byte.
- A cell with no forecast cell within `radius` draws from its own window.
- A cell with a short history adds only the months it has.

---

## Inputs and Assumptions

| Parameter | Type | Notes |
|---|---|---|
| `targets`, `window_months`, `partition_dict`, `loa`, `n_samples`, `seed` | | As `ConflictologyModel`. |
| `radius` | int ≥ 0 | Grid cells, straight-line distance between cell centres, no latitude correction (#141). Columns wrap at the date line; rows stop at the poles. 5 is Matryoshka's `NEAR_RADIUS`. |

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| `radius` not an integer in 0..359 | `ValueError` naming the model | At construction. `neighbour_table` checks it too. |
| `loa` not pgm | `ValueError` naming the model | At `fit`, as in `MatryoshkaConflictology`, so the shared protocol tests can construct it. |
| Everything `ConflictologyModel` refuses | As `ConflictologyModel` | |

---

## Boundaries and Interactions

- **Depends on:** `ConflictologyModel` (subclass), `matryoshka.geometry.neighbour_table`,
  `frames.output.sample_prediction_grid`, `frames.input.to_level`, `spatial.spatial_level`.
- **Instantiated by:** `BaselineModelCatalog._get_neighbourhood_conflictology()`.
- **Genome:** `["window_months", "n_samples", "seed", "radius"]`.
- **Subclassed by:** `RollingNeighbourhoodConflictology`, which overrides `_windows_for_origin`.

---

## Examples of Correct Usage

```python
from views_baseline.model.models.distributional import NeighbourhoodConflictology

m = NeighbourhoodConflictology(
    targets=["lr_sb_best"], window_months=36, partition_dict={"test": (457, 504)},
    loa="pgm", n_samples=64, radius=5, seed=42,
).fit(df)
pf = m.predict(df=df, sequence_number=0, output_length=36)["lr_sb_best"]
```

## Examples of Incorrect Usage

```python
NeighbourhoodConflictology(..., radius=5.0)   # raises: radius must be an integer
NeighbourhoodConflictology(..., loa="cm", radius=5).fit(df)   # raises: pgm only
```

---

## Test Alignment

All in `tests/test_neighbourhood_conflictology.py` unless noted. The input encodes each value's source
(`y1 = month × 100 + i`); the cells are real priogrid ids at the date line, at the pole row and far apart.

| Test | What it verifies |
|---|---|
| `test_each_cell_draws_from_all_of_its_discs_window_and_nothing_else` (radius 1, 2, 5) | Exactly the disc's windows. |
| `test_the_draws_are_rng_choice_over_the_discs_windows_in_ascending_cell_order` | The exact draws against a concatenated-pool reference, with a short-history cell. |
| `test_radius_zero_is_conflictology_byte_for_byte` | radius 0 ≡ `ConflictologyModel`. |
| `test_the_radius_5_disc_is_matryoshkas_neighbourhood_plus_the_cell` | The disc ≡ `build_geometry(...).neighbours` plus the cell, cell for cell, across the date line. |
| `test_a_cell_with_no_neighbours_draws_from_its_own_window`, `test_a_short_history_cell_is_pooled_with_only_the_months_it_has` | Edge cases. |
| `test_a_bad_radius_raises_naming_the_model`, `test_fit_at_country_level_raises_naming_the_model` | Settings. |
| `test_dataframe_and_featureframe_give_the_same_draws`, `test_the_catalog_requires_radius_and_forwards_every_key` | Input types; catalog. |
| `test_each_target_draws_only_from_its_own_column` (both classes) | Two targets never mix (PR #189 guard audit). |
| `tests/test_matryoshka.py::test_neighbour_table_*` | The disc itself, against brute force at radius 1, 2, 5. |
| Shared lists in `test_protocol.py`, `test_dual_input.py`, `test_catalog.py`, `test_falsification_seed_wiring.py`, `test_reproducibility_gate.py`, `test_config_conformance.py`, `test_baseline.py` | Protocol, input types, catalog, seed wiring, exact genome, a derived conformance fixture. |

---

## Evolution Notes

- **Evolving** (ADR-004).
- The conformance fixture is derived (light_strider with `algorithm` swapped and `radius: 5`) until
  views-models ships `tight_canvas` (#186).
- `neighbour_table` lives in the `matryoshka/` package although it is a grid concept. Moving it was left
  out of #178 on purpose.
