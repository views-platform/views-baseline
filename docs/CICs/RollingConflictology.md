# Class Intent Contract: RollingConflictology

**Date:** 2026-11-05 (revised after review, same day)
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-008, ADR-011, ADR-019, ADR-021, ADR-024

---

## Purpose

`RollingConflictology` is `ConflictologyModel` with one change: **its window moves with each rolling
forecast origin.**

`ConflictologyModel` forecasts every origin from one window, the last `window_months` up to the last
training month (`train_end`). So when it forecasts from origin 12, its information is already 12 months
older than it needs to be. `RollingConflictology` forecasts origin `k` from the `window_months` ending at
`train_end + k`: the month just before that origin's first forecast month (`test_start + k`). That is the
same information rule views-stepshifter uses in the same rolling evaluation (its target series is sliced to
`train_end + 1 + k`, exclusive), so no forecast month is ever inside its own window.

It exists as the **control for ADR-024** (`MatryoshkaConflictology`, issue #126). It separates two causes of
stale forecasts: a window that is out of date at the origin (which this model removes), and information
that ages within one forecast as it looks further ahead (which it does not).

---

## Non-Goals

- Does not change anything within one origin. All forecast months of one origin are drawn from the same
  window, so a forecast 1 month ahead and 36 months ahead are identical in distribution. That is
  ADR-024's job.
- Does not borrow from neighbouring cells, and does not weight months by recency.
- Does not fall back to an older window when data is missing. It stops with an error instead.

---

## Responsibilities and Guarantees

- It is a **subclass of `ConflictologyModel`**: `__init__` and `fit` are inherited, not copied. So
  `fit(df)` fixes `self.entity_ids` (entities present at `train_end`) and the origin-0 pools
  `self.hist_per_entity` exactly as conflictology does, and a later change to conflictology cannot
  silently break the paired comparison.
- `predict(df, sequence_number, output_length)`:
  - **origin 0** uses the fitted pools, so its output is byte-identical to `ConflictologyModel`'s;
  - **origin k > 0** reads the window directly from `df`: for every fitted cell, the values at months
    `train_end + k − window_months + 1` to `train_end + k`, by (cell, month) position, in one pass over
    the panel. On complete data this equals `window_pool_arrays` anchored at `train_end + k` (tested),
    but it never scans the panel once per cell. Measured on the 24.9M-row pgm `land` panel: about
    0.5 s per origin, against about 14 minutes for the shared helper;
  - cells that are not forecast (not present at `train_end`) are never pooled, so they cannot affect
    the run;
  - draws exactly as `ConflictologyModel` does (`rng.choice` over the pool, through
    `sample_prediction_grid`, seeded by `seed`), and returns `dict[str, PredictionFrame]` of shape
    `(N, n_samples)`.
- **It never reads a forecast month.** The newest month in origin `k`'s window is `train_end + k`; its first
  forecast month is `train_end + k + 1`.

---

## Inputs and Assumptions

| Parameter | Type | Notes |
|---|---|---|
| `targets`, `window_months`, `partition_dict`, `loa`, `n_samples`, `seed` | | As `ConflictologyModel`; the audited genome is the same: `window_months`, `n_samples`, `seed`. |

**Assumption:** the `df` passed to `predict` contains the observed months up to `train_end + k` for every
cell. In the pipeline's evaluation runs it does: the cached data covers the whole partition (months
121–504 for calibration). In a forecasting run, only origin 0 is used, which needs nothing beyond
`train_end`.

**Cost:** for every origin after the first, it converts the input once (about 3 s at pgm `land`) and
reads the window (about 0.5 s). Twelve origins add well under a minute to an evaluation.

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| Any month of the window `train_end + k − window_months + 1 .. train_end + k` missing for a forecast cell (the data ends early, or a gap anywhere in the window) | `ValueError` naming the window, how many cells, and example cells with their missing months | A silent fall back to older months would turn this control back into conflictology without anyone noticing. Note the difference: `ConflictologyModel` keeps the last `window_months` *rows*, so it completes on such data. This model checks calendar months and stops. |
| NaN in the window for a forecast cell | `ValueError` naming the target, the cell and the month(s) | Points at the post-`train_end` month, not at the training data. |
| Bad input, NaN in the window, `window_months <= 0` | `ValueError` | As `ConflictologyModel` (`window_pool`, `to_feature_frame`). |

---

## Boundaries and Interactions

- **Depends on:** `frames.pooling` (`window_pool`, `window_pool_arrays`), `frames.input` (`panel`,
  `to_feature_frame`, `to_level`), `frames.output.sample_prediction_grid`, `grid.train_test_boundary`.
- **Instantiated by:** `BaselineModelCatalog._get_rolling_conflictology()`.
- **Genome:** `["window_months", "n_samples", "seed"]`.
- It is the only baseline that reads observed months after `train_end`.

---

## Examples of Correct Usage

```python
from views_baseline.model.models.distributional import RollingConflictology

m = RollingConflictology(
    targets=["lr_sb_best"], window_months=36, partition_dict={"test": (457, 504)},
    loa="pgm", n_samples=256, seed=42,
).fit(df)
pf = m.predict(df=df, sequence_number=12, output_length=36)["lr_sb_best"]
# draws come from months 433..468; the first forecast month is 469
```

## Examples of Incorrect Usage

```python
# Passing only the training data and asking for a later origin: there is nothing to slide onto.
m.fit(train_only_df).predict(train_only_df, sequence_number=12, output_length=36)  # ValueError
```

---

## Test Alignment

All in `tests/test_rolling_conflictology.py` unless noted. The test input encodes each value's month
(`y1 = month × 10 + entity`), so the month every draw came from can be read back exactly.

| Test | What it verifies |
|---|---|
| `test_origin_zero_is_byte_identical_to_conflictology` | Origin 0 equals `ConflictologyModel` exactly: values, times and units. |
| `test_window_ends_at_train_end_plus_k` (k = 0, 1, 5, 12, 30) | Origin k draws from exactly months `train_end + k − window_months + 1` to `train_end + k`, and never from a forecast month. |
| `test_conflictology_does_not_move_its_window` | The contrast: `ConflictologyModel` still draws only up to `train_end` at origin 12. |
| `test_fails_loud_when_the_data_does_not_reach_the_origin` | `ValueError` naming the first missing month. |
| `test_fails_loud_when_one_cell_is_missing_at_the_origin` | `ValueError` when a single cell lacks the anchor month. |
| `test_reproducible_and_seed_sensitive` | Same seed, same output; different seed, different output. |
| `test_each_target_draws_from_its_own_history_at_every_origin` | With two targets, each target's draws come from its own history at k > 0 (re-pooling never mixes targets). |
| `test_entity_set_is_fixed_at_fit_even_if_a_cell_appears_later` | A cell that first appears after `train_end` never enters the forecast, and one `predict()` call cannot change a later one. |
| `test_same_draw_indices_as_conflictology_at_every_origin` (k = 1, 12) | The same seed picks the same pool positions as `ConflictologyModel` at every origin, so each draw is conflictology's shifted by exactly k months: the comparison stays paired. |
| `test_pools_equal_window_pool_arrays_on_a_complete_panel` (k = 1, 12, 30) | The direct window read equals the shared pooling helper on complete data. |
| `test_fails_loud_on_a_gap_earlier_in_the_window` | A cell present at the anchor but missing an earlier window month stops the run. |
| `test_nan_after_train_end_names_the_month` | The NaN error names the cell and the post-`train_end` month. |
| `test_a_cell_that_is_never_forecast_cannot_stop_the_run` | A NaN in a cell that first appears after `train_end` is ignored. |
| `test_is_a_conflictology_subclass` | `__init__` and `fit` are conflictology's own, not copies. |
| Shared lists in `test_protocol.py` (which now also pins `distributional is True` for every distributional model), `test_dual_input.py`, `test_catalog.py`, `test_falsification_seed_wiring.py`, `test_reproducibility_gate.py` (incl. `test_rolling_conflictology_genome_is_exact`), `test_config_conformance.py` | Protocol, both input types, catalog forwarding, seed wiring, exact genome, a derived conformance fixture. |

---

## Evolution Notes

- **Evolving** (ADR-004).
- The conformance fixture is derived (light_strider with `algorithm` swapped) until views-models ships a
  config for this model.
- If ADR-024's comparison shows this control already removes most of the staleness, that is a finding
  for #82: whether `ConflictologyModel` itself should slide.
