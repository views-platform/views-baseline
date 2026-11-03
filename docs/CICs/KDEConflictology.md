# Class Intent Contract: KDEConflictology

**Date:** 2026-11-03
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-006, ADR-008, ADR-011, ADR-012, ADR-019, ADR-020, ADR-021, ADR-023

---

## Purpose

`KDEConflictology` is smoothed climatology **C** (ADR-023, issue #107). It is
`ConflictologyModel` (A) with one change: a positive draw comes from a **Gaussian kernel density
estimate** of the window's positive values, reflected at 0.5 and rounded to a count. Unlike A
and B, it can draw counts above the window's maximum. How far above is set by `bandwidth`.

As an uncertainty prototype (epic #104), the network analogue is a **hurdle head with a
smooth positive density**, for example a mixture-density head whose components sit at
observed-like magnitudes.

---

## Non-Goals

- Does **not** change the zero mass. `P(Y=0)` is the window's empirical zero rate, as in A.
- Does **not** use a discrete count kernel: rejected in ADR-023 (Senga Kiessé & Durrieu 2024).
- Does **not** transform (no `log1p`) and does **not** adapt the bandwidth to scale. A
  scale-relative or adaptive bandwidth is D (#110).
- Does **not** have a default `bandwidth`.

---

## Responsibilities and Guarantees

- `__init__` validates `bandwidth`. It must be a positive finite real number (not `bool`), or
  `"sj"` or `"silverman"`. Anything else raises `ValueError`.
- `fit(df)` takes each cell's window from the shared `window_pool`, so the pool is identical to
  A's, and classifies each cell with `smoothing.classify_window`.
  - For a `smooth` cell it computes `h`. A number is used as given, the same for every cell.
    `"sj"` is the Sheather–Jones (1991) solve-the-equation bandwidth on the cell's positives.
    `"silverman"` is R's `bw.nrd0` on the cell's positives.
  - A selector that returns no positive bandwidth marks the cell `selector_failed`.
  - Cells that are not smoothed are drawn exactly as A draws them.
  - Counts go in `fallback_counts`, which is reset on every `fit()`.
- `predict()` returns `dict[str, PredictionFrame]` of shape `(N, n_samples)`, built through
  `sample_prediction_grid` (ADR-011).
  - Per smoothed cell, it first draws `is_positive = rng.random(n) >= zero_rate`.
  - Then `draw_kde_positives` draws, for the positive draws only, `k` uniform indices into
    the positives, then `k` standard normals. It sets `x = p + h·z`, reflects
    `x < 0.5 → 1 − x`, and rounds with `round_positive`.
- **Integer output.** Positive draws are ≥ 1. Reflection at 0.5 puts no mass on 0.
- **One aggregate WARN per `predict()`** names the cells that were not smoothed, by kind,
  including `selector_failed`.

---

## Inputs and Assumptions

| Parameter | Type | Notes |
|---|---|---|
| `targets`, `window_months`, `partition_dict`, `loa`, `n_samples` | | As `ConflictologyModel`. |
| `bandwidth` | `float \| str` | Required, audited genome key (ADR-021). A number in count units, global across cells, or `"sj"` / `"silverman"` per cell. |
| `seed` | `int` | Required, audited genome key. |

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| Invalid `bandwidth` | `ValueError` at `__init__` | Covers unknown names, ≤ 0, non-finite values and `bool`. |
| Selector undefined | Drawn as A, plus the aggregate WARN | For example, SJ when the IQR is 0 or there is no root. Counted as `selector_failed`. |
| Window with < 2 positives, all-zero, or no spread | Drawn as A, plus the aggregate WARN | Shared rule (`smoothing.py`). |
| NaN in the window / `window_months <= 0` / bad input | `ValueError` | `window_pool` / `to_feature_frame`. |

---

## Known bias (state it in every report)

Reflection removes the O(1) boundary loss but leaves an **O(h) bias** wherever the density's
slope at the boundary is non-zero (Marron & Ruppert 1994). Conflict counts decay steeply from 1,
so C moves mass from 1 onto 2, 3 and higher. As a result, `P(Y=1 | Y≥1)` is below the window's
share of 1s, and the gap grows with `h`. `"silverman"` oversmooths (Jones, Marron & Sheather
1996) and is the expected worst choice.

---

## Boundaries and Interactions

- **Depends on:** numpy only. The SJ selector is implemented in `kde.py` with exact pairwise
  sums, which is cheap for windows of a few dozen values. **No new runtime dependency.**
- **Shared:** `smoothing.py`, with B and E. E (#108) will reuse `draw_kde_positives` and
  `cell_bandwidth`.
- **Instantiated by:** `BaselineModelCatalog._get_kde()`.
- **Genome:** `["window_months", "n_samples", "seed", "bandwidth"]`.

---

## Examples of Correct Usage

```python
from views_baseline.model.models.distributional import KDEConflictology

model = KDEConflictology(
    targets=["lr_sb_best"], window_months=36, partition_dict={"test": (457, 504)},
    loa="pgm", n_samples=256, bandwidth="sj", seed=42,
)
pf = model.fit(df).predict(df=df, sequence_number=0, output_length=36)["lr_sb_best"]
print(model.fallback_counts)   # includes selector_failed for cells SJ cannot handle
```

## Examples of Incorrect Usage

```python
KDEConflictology(..., bandwidth="scott")   # ValueError: unknown selector
KDEConflictology(..., bandwidth=0)         # ValueError: must be positive
# Reading a lower P(Y=1 | Y≥1) than A as a finding about the data: it is C's known boundary bias.
```

---

## Test Alignment

All in `tests/test_smoothed_climatology.py` unless noted.

| Test | What it verifies |
|---|---|
| `test_c_pool_is_as`, `test_c_zero_rate_is_the_windows_exactly`, `test_c_zero_mass_equals_as` | Pool and zero mass are A's. |
| `test_c_draws_are_integers_and_positive_draws_at_least_one` | Integer output at small, medium and huge `h`. |
| `test_c_reflects_at_one_half` | The draw law matches the analytic P(1), P(2) for p = 1, h = 1. |
| `test_c_boundary_bias_moves_mass_off_one` | The known bias exists and grows with `h`. |
| `test_silverman_is_bw_nrd0`, `test_sj_matches_the_normal_optimum`, `test_sj_is_narrower_than_silverman_on_bimodal_data`, `test_sj_undefined_on_zero_iqr` | The selectors. |
| `test_c_each_bandwidth_runs_on_a_three_positive_window`, `test_c_selector_failure_falls_back_to_a_and_is_counted`, `test_c_counts_fallbacks_and_warns_once` | No silent failure; fallbacks are counted and equal A. |
| `test_c_accepts_valid_bandwidths`, `test_c_rejects_invalid_bandwidths`, `test_catalog.py` bandwidth tests | Validation and the genome. |
| `test_golden_c_with_zeros` | Exact draws at a fixed seed. |

---

## Evolution Notes

- **Evolving** (ADR-004).
- Open (ADR-023): whether `"sj"` is stable enough on 2–20 positive values. #109 reports the
  `selector_failed` count.
- The conformance fixture is derived until views-models ships a C config (#109).
