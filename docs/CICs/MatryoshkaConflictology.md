# Class Intent Contract: MatryoshkaConflictology

**Date:** 2026-11-07
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-008, ADR-011, ADR-019, ADR-021, ADR-024 (with its amendment of 2026-11-06)

---

## Purpose

`MatryoshkaConflictology` is conflictology that answers one extra question: **which set of past
observations should a cell's forecast draw from?** Each forecast draw is a value that was really
observed. It is picked in three steps:

1. **A doll:** the cell itself, its neighbourhood (cells within 5 grid cells, closer ones more
   likely), the rest of its country, or the rest of the map. Doll `k` is picked ∝
   `weight[k] · exp(−h / memory[k])`, where `h` is how many months after the last observed month the
   forecast month lies. Inner dolls forget faster, so the forecast opens outward as it looks further
   ahead.
2. **A month** ∝ `exp(−(c − s) / memory[k])`, over every history month `s` up to the last observed
   month `c`.
3. **A cell in that doll:** a neighbour at distance `d` ∝ `exp(−d / ℓ)`; uniform within the country
   and map shells.

There are **8 knobs** per target (3 weights, the cell's fixed at 1; 4 memories; the length scale `ℓ`).
They are fitted once, by backtest on data up to `train_end`. The data itself runs up to each forecast
origin: origin `k` draws from history ending at `c = train_end + k`.

Plain conflictology with a moving window (`RollingConflictology`) is the special case where only the
cell has weight, nothing fades, and the history is the last 36 months.

---

## Non-Goals

- Does not forecast at country level (cm); `fit` raises for any level other than pgm.
- Does not invent values: no smoothing between observed values (ADR-024, amendment: KDE rejected).
- Does not separate *whether* violence happens from *how big* it is (#137, rejected). In a quiet
  place the size of a possible outbreak comes from the world pool, which is dominated by ongoing wars.
- Does not refit its knobs at predict time; only the data moves with the origin.
- Does not fill or skip missing data. A gap anywhere in the panel stops the run.

---

## Responsibilities and Guarantees

- `fit(df)`:
  - reads the targets and `country_column` for months ≤ `train_end`;
  - requires a **complete panel**: every cell in every month from the first month in `df` to
    `train_end`, exactly once, with no NaN;
  - requires each cell's country code to be an integer that never changes (≤ 0 means no country);
  - per target, searches the 8 knobs with L-BFGS-B on the exact log score over 7 size classes and
    its exact slope (`knob_fit.py`). The backtest has `backtest_cutoffs` cutoffs, 12 months apart,
    the last one `backtest_horizon` months before `train_end`. Each is scored on the
    `backtest_horizon` months after it;
  - with `stability_refits`, repeats the search with each cutoff left out in turn;
  - stores the result in `self.knobs[target]`: `weight`, `memory`, `length_scale`, `score`,
    `start_score`, `floored`, `converged`, `message`, `steps`, `evaluations`, `seconds`, and
    `leave_one_cutoff_out` (with stability refits). `seconds` is wall-clock time, the one field
    that differs between identical fits. It also stores `self.entity_ids` (ascending priogrid
    ids), `self.country` and `self.history_start`.
- `predict(df, sequence_number, output_length)`:
  - reads the targets for months ≤ `c = train_end + sequence_number`. The panel must be complete up
    to `c`, over the same cells as at fit, and must start in the same month as the fit history;
  - forecast month `t = test_start + sequence_number + j` is drawn with `h = t − c = j + 1`;
  - **never draws a month after `c`**, and never a forecast month;
  - returns `dict[str, PredictionFrame]`, one per target. Each has shape
    `(N · output_length, n_samples)`, with rows in entity → time order.
- **Draws are exactly the fitted distribution.** The sampler (`draw.py`) and the exact probabilities
  (`forecast_probability.py`) are checked against each other in the tests.
- **RNG (ADR-011):**
  - one fresh `default_rng(seed)` per `predict` call;
  - the seed does not depend on `sequence_number`;
  - three uniforms per sample, consumed in entity → time → target order. Cells are drawn in blocks
    of about `BLOCK_UNIFORMS` uniforms (128 MiB), which consume the same stream in the same order,
    so the block size changes memory and speed only (tested).
- **No zero-probability pick.** Row lookups are capped at the row's last positive entry, so floating
  rounding of the row shift cannot reach a sea neighbour or a month past `c` (code review of
  `702f17d`).

---

## Inputs and Assumptions

| Parameter | Type | Notes |
|---|---|---|
| `targets`, `partition_dict`, `loa`, `n_samples`, `seed` | | As the other distributional baselines. `loa` must be `pgm`. |
| `country_column` | `str` | The column with each cell's GAUL country code. In views-models' pgm input this is `c_id` (renamed from `gaul0_code`). It must not be a target. |
| `backtest_cutoffs` | `int ≥ 1` | Number of backtest cutoffs; 4 in the prototype. |
| `backtest_horizon` | `int ≥ 1` | Months scored after each cutoff; 36 in the prototype. |
| `stability_refits` | `bool` | Refit with each cutoff left out in turn (needs ≥ 2 cutoffs). This multiplies the fit time by about `backtest_cutoffs + 1`. |

**Fixed by ADR-024, not configured:**
- the four dolls and the 5-cell radius;
- the seven size classes;
- the knob limits: weights 1e-6–1e3, memories 1–600 months, `ℓ` 0.25–10 cells;
- the starting values;
- the search settings: `ftol = 1e-13`, `gtol = 1e-9`, at most 200 steps;
- at least 120 months of history before the earliest cutoff.

**The history is all the data passed.** There is no window. With the calibration input that is months
121 up to the origin. Passing a shorter or longer history changes the forecast.

**Assumption:** in the pipeline's evaluation runs, the `df` passed to `predict` holds the observed months
up to `train_end + k` for every cell, as for `RollingConflictology`.

**Cost**, measured 2026-11-07 at pgm `land` on the pinned calibration input
(`reports/matryoshka/results/package_check.json`):
- **Fit,** three targets, no stability refits: 621 s, with a peak of 5.4 GiB. Stability refits multiply
  the search time by about 5.
- **Predict,** one origin, all cells, 36 months: 35 s for one target at S = 64. Draw time scales with
  targets × S, so the full 3 targets × S = 256 is roughly 7 minutes per origin.
- **Output:** the output alone is about 7 GB per origin at that size, the same as conflictology's.

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| A missing or duplicated cell-month, or non-consecutive months, up to `train_end` (fit) or `c` (predict) | `ValueError` with counts | The shells and their sizes assume a complete panel. |
| NaN in a target or the country column | `ValueError` naming the column, a cell and a month | |
| A negative target value | `ValueError` naming the cell and month (and `classify` refuses negatives) | A negative value would otherwise index the largest size class and silently bias the fit. |
| Predict data starting at a different month from the fit history | `ValueError` naming both months | The history is all the data passed, so a different start would draw from a different distribution than the one fitted. |
| A cell's country changes over time, or is not an integer | `ValueError` | |
| Fewer than 120 months before the earliest cutoff | `ValueError` with the months available | ADR-024: at least 10 years. |
| `stability_refits` with a single cutoff | `ValueError` | |
| `loa` other than `pgm` | `ValueError` at fit | |
| `predict` before `fit`; `sequence_number < 0`; data ending before `c`; different cells from fit | `ValueError` | |
| `country_column` among the targets | `ValueError` at construction | |
| The search stops without converging | **Not raised.** `knobs[t]["converged"]` is False and `message` says why | Reported, not hidden; the evaluation report shows it. |

---

## Boundaries and Interactions

- **Package:** `views_baseline/model/models/distributional/matryoshka/`, one idea per module:
  - `geometry` (the grid, neighbours, country);
  - `size_classes`;
  - `doll_counts` (sparse per-month class counts, shell sizes);
  - `forecast_probability` (the exact probabilities, the score and its slope);
  - `knob_fit` (the search);
  - `draw` (the sampler);
  - `model`.
- **Depends on:** `frames.input` (`to_feature_frame`, `panel`, `to_level`), `frames.output`
  (`to_prediction_frames`), `grid`, `spatial`, and scipy (sparse matrices, `minimize`).
- **Instantiated by:** `BaselineModelCatalog._get_matryoshka()`.
- **Genome:** `["n_samples", "seed", "country_column", "backtest_cutoffs", "backtest_horizon",
  "stability_refits"]`.
- **Ported from** the prototype in `reports/matryoshka/prototype/` (branch
  `feat/127-matryoshka-prototype`). `reports/matryoshka/package_check.py` checks on real data that the
  package's knobs equal the prototype's.

---

## Examples of Correct Usage

```python
from views_baseline.model.models.distributional import MatryoshkaConflictology

m = MatryoshkaConflictology(
    targets=["lr_sb_best", "lr_ns_best", "lr_os_best"], partition_dict={"test": (457, 492)},
    loa="pgm", n_samples=256, country_column="c_id", backtest_cutoffs=4, backtest_horizon=36,
    stability_refits=True, seed=42,
).fit(df)                      # df: months 121..492 or later, lr_*_best and c_id
m.knobs["lr_sb_best"]["memory"]
pf = m.predict(df=df, sequence_number=12, output_length=36)["lr_sb_best"]
# draws come from months 121..468; the first forecast month is 469
```

## Examples of Incorrect Usage

```python
# The country column is missing: fit cannot build the country doll.
m.fit(df[["lr_sb_best"]])                                   # ValueError (missing column)
# A panel with a gap: no silent filling.
m.fit(df.drop(df.index[0]))                                 # ValueError (complete panel)
```

---

## Test Alignment

All in `tests/test_matryoshka.py`. The fixture has three cells without a country, apart, and holes.

| Test | What it verifies |
|---|---|
| `test_shells_are_disjoint_cover_every_cell_and_have_the_counted_sizes` | The four shells, built by brute force from pairwise distances, are disjoint, cover every cell, and have the sizes the counts use. |
| `test_doll_shares_equal_a_direct_kernel_weighted_average` | Every doll's class shares equal a direct (kernel-weighted) average over its cells, to 1e-10. |
| `test_exact_probabilities_match_an_independent_three_step_draw` | The closed form equals 200,000 three-step draws from the brute-force shells, for a full and a corner cell. |
| `test_exact_slope_matches_nudged_differences` | The exact slope equals nudged differences (rtol 1e-6, no absolute slack). |
| `test_kept_positions_skip_excluded_stretches`, `test_sampler_shells_are_exactly_the_brute_force_shells` | The sampler's uniform shell picks reach exactly the brute-force country and map shells, each cell once. |
| `test_sampler_matches_the_exact_probabilities` (h = 1, 12) | What predict draws equals what the fit scores. |
| `test_forecast_opens_outward_with_distance` | The cell's share shrinks and the map's grows with `h`. |
| `test_row_lookup_never_overflows_onto_a_zero_probability_entry` | The largest uniform `rng.random` returns, in the last row of a 256-row block, stops at the last positive entry. |
| `test_a_panel_where_no_cell_has_a_country_still_works` | Counts, shares and draws with no country at all. |
| `test_special_case_is_conflictology_with_a_moving_window` | Cell only, no fading, 36 months: draws are uniform over the cell's last 36 values. |
| `test_fit_beats_the_cell_only_special_case_on_a_planted_neighbour_effect` | On data with spreading hotspots, the fit converges, beats the cell-only case, and gives the neighbourhood weight. |
| `test_backtest_cutoffs_are_spaced_and_end_a_horizon_before_train_end` | 456, 4, 36 gives 384, 396, 408, 420 (the prototype's design). |
| `test_predict_never_draws_a_month_after_the_history_end` | Marked future values never appear. |
| `test_predict_output_grid_and_shape`, `test_same_seed_same_output_and_seed_matters`, `test_block_size_changes_nothing`, `test_dataframe_and_featureframe_inputs_give_identical_output`, `test_stability_refits_leave_each_cutoff_out` | Output grid, reproducibility, ADR-011 order, both input types, stability refits. |
| `test_fails_loud_*` (incl. a negative value), `test_classify_rejects_negative_values`, `test_predict_fails_loud`, `test_predict_fails_loud_when_the_history_starts_elsewhere`, `test_stability_refits_need_two_cutoffs`, `test_country_column_must_not_be_a_target` | The failure table above. |
| `test_each_doll_draws_exactly_from_its_shell_with_the_right_cell_weights` (each doll), `test_a_cell_without_a_country_never_picks_the_country_doll`, `test_cdf_ends_exactly_at_one_from_the_last_positive_entry` | With values that encode their cell, every draw lands in its doll's brute-force shell in the right proportions (a full cell, a corner and a no-country cell). |
| `test_predict_uses_history_end_c_and_months_ahead_h` | With values that encode month and cell, the share drawn from the cell follows `P(cell \| h = j + 1)` per step, and its months follow `exp(−(c − s)/m)` up to and including `c`. |
| `test_rng_order_is_entity_then_time_then_target` | With two targets, predict equals one sampler call per (cell, month, target) in that order (ADR-011). |
| `test_the_120_month_history_boundary`, `test_stability_refits_are_searches_without_the_left_out_cutoff`, `test_outcomes_are_the_months_after_each_cutoff`, `test_log_score_equals_the_samplers_mix_on_a_grid_with_empty_shells` | The history boundary exactly, the refits, the scored months, and that fit and sampler use the same doll mix with empty shells. |
| `test_near_last_is_each_cells_farthest_present_neighbour`, `test_predict_rejects_a_negative_value_in_its_history`, `test_start_score_is_the_score_at_the_starting_values` | The overflow cap's per-cell limit, predict's own negative check, and the reported starting score. |
| `test_geometry_wraps_columns_at_the_date_line`, `test_geometry_rejects_unsorted_cells`, `test_fails_loud_on_duplicate_cell_months`, `test_fails_loud_on_non_consecutive_months`, `test_fails_loud_on_a_non_integer_country_code`, `test_fails_loud_when_the_data_ends_before_train_end`, `test_nan_in_the_predict_history_is_named`, `test_fit_names_nan_before_classifying`, `test_predict_fails_loud_on_same_count_but_different_cells` | Geometry edges and every fail-loud check, each by its own message. |
| Shared lists in `test_protocol.py`, `test_catalog.py`, `test_falsification_seed_wiring.py`, `test_reproducibility_gate.py` (incl. `test_matryoshka_genome_is_exact`), `test_baseline.py`, `test_config_conformance.py` | Protocol, catalog forwarding, seed wiring, exact genome, a derived conformance fixture. The shared dual-input and seed-output lists use a 60-month panel without a country column, so those two checks are in `test_matryoshka.py`. |

---

## Evolution Notes

- **Evolving** (ADR-004).
- The conformance fixture is derived (light_strider with `algorithm` swapped and the Matryoshka keys)
  until views-models ships a config for this model (#127 stage 4).
- Known loose knobs (#140, accepted): the one-sided map memory sits at the 600-month ceiling; the
  non-state country weight and memory trade off against each other.
