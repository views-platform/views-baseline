# Class Intent Contract: RollingNeighbourhoodConflictology

**Date:** 2026-11-18
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-008, ADR-011, ADR-019, ADR-021, ADR-024

---

## Purpose

`RollingNeighbourhoodConflictology` is `NeighbourhoodConflictology` with one change: **at rolling origin `k`
the disc's windows end at `train_end + k`**, not at `train_end`.

That is Matryoshka's information rule, and `RollingConflictology`'s. It is the neighbourhood single level
to compare with Matryoshka at every origin (ADR-024, amendment 2026-11-18).

---

## Non-Goals

- Does not move the disc. The disc is built at `fit` from the cells present at `train_end`.
- Everything `NeighbourhoodConflictology` does not do.

---

## Responsibilities and Guarantees

- `__init__` and `fit` are `NeighbourhoodConflictology`'s.
- `predict(df, sequence_number=k, output_length)`:
  - `k == 0`: draws from the fitted windows; byte-identical to `NeighbourhoodConflictology`;
  - `k > 0`: reads each forecast cell's last `window_months` rows up to `train_end + k` with
    `frames.pooling.rolling_window_pools` (the reader `RollingConflictology` and
    `RollingPooledConflictology` use), then draws over the same discs.
- `radius` 0 is `RollingConflictology` byte for byte at every origin.
- **Common random numbers:** the same seed picks the same pool positions at every origin, as long as every
  disc keeps its total length, which means every forecast cell has a full window.
- It overrides only `_windows_for_origin`. The siblings copy `predict` instead; here `predict` holds the
  disc draw, which should exist once.

---

## Inputs and Assumptions

As `NeighbourhoodConflictology`. **Assumption:** the `df` passed to `predict` holds every forecast cell's
targets up to `train_end + k`; pipeline-core passes the full data at every origin.

**Cost:** on a synthetic pgm-sized panel (64,800 cells, 3 targets, `window_months` 36, S = 64, radius 5),
`predict` took about 93–112 s per origin and peaked at 2.5 GiB; about 1.8× `RollingPooledConflictology`
(#182; laptop timing, 2026-11-18; `reports/matryoshka/LOG.md`).

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| A forecast cell is missing a month, or has a NaN, after `train_end` inside the window | `ValueError` naming the model and the cell | From `rolling_window_pools`. A gap would silently pull older months back into the window. |
| `sequence_number < 0` | `ValueError` naming the model | |
| Everything `NeighbourhoodConflictology` refuses | As `NeighbourhoodConflictology` | |

---

## Boundaries and Interactions

- **Depends on:** `NeighbourhoodConflictology` (subclass; `_windows`), `frames.pooling.rolling_window_pools`.
- **Instantiated by:** `BaselineModelCatalog._get_rolling_neighbourhood_conflictology()`.
- **Genome:** `["window_months", "n_samples", "seed", "radius"]`, the same as `NeighbourhoodConflictology`'s.

---

## Examples of Correct Usage

```python
from views_baseline.model.models.distributional import RollingNeighbourhoodConflictology

m = RollingNeighbourhoodConflictology(
    targets=["lr_sb_best"], window_months=36, partition_dict={"test": (457, 504)},
    loa="pgm", n_samples=64, radius=5, seed=42,
).fit(df)
pf = m.predict(df=df, sequence_number=5, output_length=36)["lr_sb_best"]   # windows end at 461
```

## Examples of Incorrect Usage

```python
# The data stops at train_end: origin 1 raises instead of silently using the train_end window.
m.predict(df=df_up_to_train_end, sequence_number=1, output_length=36)
```

---

## Test Alignment

In `tests/test_neighbourhood_conflictology.py`.

| Test | What it verifies |
|---|---|
| `test_rolling_each_cell_draws_from_its_discs_window_ending_at_the_origin` (k = 1, 3, 12) | Exactly the disc's windows ending at `train_end + k`. |
| `test_rolling_origin_zero_is_the_fixed_window_model_byte_for_byte` | Origin 0 ≡ `NeighbourhoodConflictology`. |
| `test_rolling_radius_zero_is_rolling_conflictology_byte_for_byte` (k = 0, 1, 5, 12) | radius 0 ≡ `RollingConflictology`. |
| `test_rolling_a_gap_inside_the_window_raises_naming_the_model` | A missing row or NaN raises, only once inside the window. |
| `test_rolling_negative_sequence_number_raises`, `test_rolling_forecasts_are_labelled_with_the_origins_months`, `test_rolling_the_disc_stays_as_fitted` | Bad origin; labels; a cell appearing after `train_end` is neither forecast nor pooled. |
| `test_the_catalog_requires_radius_and_forwards_every_key`, and the shared lists | As `NeighbourhoodConflictology`. |

---

## Evolution Notes

- **Evolving** (ADR-004).
- The conformance fixture is derived until views-models ships `tight_sailcloth` (#186).
