# Class Intent Contract: InterpolatedQuantileConflictology

**Date:** 2026-11-03
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-006, ADR-008, ADR-011, ADR-012, ADR-019, ADR-020, ADR-021, ADR-023

---

## Purpose

`InterpolatedQuantileConflictology` is smoothed climatology **B** (ADR-023, issue #106). It is
`ConflictologyModel` (A) with one change: the positive values in a cell's window are drawn from
their **linearly interpolated quantile function** instead of being resampled. So it can draw the
counts *between* the observed ones, which A cannot.

It is also an uncertainty prototype (epic #104): the network analogue is a **quantile head**
that interpolates between predicted quantiles of the positive part.

---

## Non-Goals

- Does **not** change the zero mass. `P(Y=0)` is the window's empirical zero rate, as in A.
- Does **not** extrapolate. Positive draws stay in `[min, max]` of the observed positives,
  by decision (ADR-023 §3). It cannot forecast a count larger than the window has seen.
- Does **not** transform (no `log1p`), and has **no tuning parameter**: the plotting position
  `(i − 1)/(m − 1)` is its definition.
- Does **not** pool across entities.

---

## Responsibilities and Guarantees

- `fit(df)` normalizes the input with `to_feature_frame` and takes each cell's window from the
  shared `window_pool`, so the pool is **byte-identical to A's**. Each (entity, target) cell is
  classified by `smoothing.classify_window`:
  - `smooth`: at least `MIN_POSITIVES_FOR_SMOOTHING` (= 2) positive values, not all equal.
    Stores the zero rate, the sorted positives `p(1) ≤ … ≤ p(m)` and the positions
    `(i − 1)/(m − 1)`.
  - `all_zero`, `too_few_positives`, `no_spread`: stores the pool; drawn exactly as A draws it.
  
  Counts per kind go to `self.fallback_counts`, reset on every `fit()`.
- `predict(df, sequence_number, output_length)` returns `dict[str, PredictionFrame]` with
  `y_pred` shape `(N, n_samples)`, through `sample_prediction_grid` (fresh
  `default_rng(seed)`, entity → time → target; ADR-011).
- **Per smoothed cell:** `is_positive = rng.random(n) >= zero_rate`; then one
  `U ~ Uniform(0,1)` per positive draw, `x = interp(U, positions, positives)`, and the draw is
  `max(floor(x + 0.5), 1)`.
- **Integer output.** Every draw is in {0, 1, 2, …}, and a positive draw is ≥ 1.
- **One aggregate WARN per `predict()`** naming how many cells were not smoothed, by kind
  (ADR-008). Silent when every cell was smoothed.

---

## Inputs and Assumptions

| Parameter | Type | Notes |
|---|---|---|
| `targets` | `list[str]` | Must exist at fit time. |
| `window_months` | `int` | Trailing months per entity (shared `window_pool`); `<= 0` fails loud. |
| `partition_dict` | `dict` | Must contain `"test"`; `train_end = test_start − 1`. |
| `loa` | `str` | Declared level, validated at `to_feature_frame`. |
| `n_samples` | `int` | Draws per cell. |
| `seed` | `int` | Required, audited genome key (ADR-021). `DEFAULT_SEED` is a direct-construction sentinel only. |

Assumes the target is a count. A fractional positive value below 0.5 would be raised to 1 by
the floor, not dropped to 0, so the zero mass is still kept.

---

## Outputs and Side Effects

- **`fit()`**: returns `self`; sets `entity_ids`, `cells`, `fallback_counts`.
- **`predict()`**: returns `dict[str, PredictionFrame]`; one WARN if any cell fell back.
  Reproducible given `seed`.

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| Window with < 2 positives, all-zero, or all positives equal | Drawn as A + one aggregate WARN | Not an error; counted in `fallback_counts`. |
| NaN in the window | `ValueError` at fit | `window_pool` (C-33). |
| `window_months <= 0` | `ValueError` at fit | `window_pool`. |
| Missing target / level mismatch | `ValueError` | `to_feature_frame` (ADR-019). |
| No entities | `ValueError` | `require_entities`. |

---

## Boundaries and Interactions

- **Depends on:** `numpy`; `frames.input`, `frames.pooling.window_pool`,
  `frames.output.sample_prediction_grid`, `grid.train_test_boundary`;
  `models.distributional.smoothing` (shared with C and E).
- **Instantiated by:** `BaselineModelCatalog._get_interpolated_quantile()`.
- **Genome:** `ALGORITHM_GENOMES["InterpolatedQuantileConflictology"] = ["window_months", "n_samples", "seed"]`.

---

## Examples of Correct Usage

```python
from views_baseline.model.models.distributional import InterpolatedQuantileConflictology

model = InterpolatedQuantileConflictology(
    targets=["lr_sb_best"], window_months=36, partition_dict={"test": (457, 504)},
    loa="pgm", n_samples=256, seed=42,
)
pf = model.fit(df).predict(df=df, sequence_number=0, output_length=36)["lr_sb_best"]
print(model.fallback_counts)   # how many cells were not smoothed, by reason
```

---

## Examples of Incorrect Usage

```python
# Reading B's output as "can forecast an unseen maximum" — it cannot (no extrapolation).
# Comparing B to A on an occurrence metric and reading a difference as real — the zero
# mass is equal by construction, so any difference is Monte-Carlo noise (EVALUATION_PLAN §2).
```

---

## Test Alignment

| Test | What it verifies |
|---|---|
| `test_smoothed_climatology.py::test_b_pool_is_as` | Pool identical to A's. |
| `::test_b_zero_mass_equals_as` | Zero fraction within 5 SE of the window's zero rate, for A and B. |
| `::test_b_draws_are_integers_and_positive_draws_at_least_one` | Integer output. |
| `::test_b_never_leaves_the_observed_positive_range` | No extrapolation. |
| `::test_b_fills_the_gaps_a_cannot` | B draws counts between observed ones; A does not. |
| `::test_b_interpolates_on_plotting_positions` | Plotting positions and linear interpolation. |
| `::test_b_counts_each_fallback_and_warns_once`, `::test_b_fallback_cells_are_drawn_exactly_as_a_draws_them` | Each fallback branch fires, is counted, warns once, and equals A. |
| `test_golden.py::test_golden_interpolated_quantile` | Exact draws at a fixed seed. |
| catalog / protocol / gate / dual-input / seed-wiring / conformance | Wiring (ADR-012). |

---

## Evolution Notes

- **Evolving** (ADR-004).
- A smooth tail extension (e.g. Bernstein, Cheng 1995) was rejected for Phase 1; revisit only if
  #109 shows the bound on observed values is B's main failure (ADR-023 Alternatives).
- The conformance fixture is derived from light_strider's until views-models ships a B config
  (#109).
