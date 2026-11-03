# Class Intent Contract: EmpiricalKDEMixtureConflictology

**Date:** 2026-11-03
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-006, ADR-008, ADR-011, ADR-012, ADR-019, ADR-020, ADR-021, ADR-023

---

## Purpose

`EmpiricalKDEMixtureConflictology` is smoothed climatology **E** (ADR-023, issue #108). It is the
mixture `F = (1−α)·F_A + α·F_C` of conflictology (A) and its KDE smoothing (C). Each draw is an
A-draw with probability `1−α` and a C-draw with probability `α`. It mixes **distributions**,
not draws: E never averages values.

`α` works as a **spread dial** from A (α = 0) to C (α = 1). Mixing can only widen spread
(Gneiting & Ranjan 2013, Thm 3.1). E is also the generator for the hydranet α-sweep
(conditions on #109). As an uncertainty prototype (epic #104), the network analogue is a
**mixture head** with an empirical / resampling component and a smooth component.

---

## Non-Goals

- Does **not** change the zero mass, since both components keep A's.
- Is **not** byte-identical to A at α = 0 or to C at α = 1. It equals them **in distribution**,
  because it always consumes all three random streams.
- Does **not** recalibrate the pool. The beta-transformed linear pool is out of scope
  (ADR-023 §3).
- Has **no** default for `alpha` or `bandwidth`.

---

## Responsibilities and Guarantees

- `__init__` validates both parameters and raises `ValueError` otherwise:
  - `alpha` must be a finite real number in [0, 1], not `bool`.
  - `bandwidth` follows C's rule (`kde.validate_bandwidth`).
- `fit(df)`:
  - takes the shared `window_pool` (identical to A's);
  - fits every cell with `kde.fit_kde_cell`, the same state C fits, which keeps the pool in
    every cell;
  - counts the kind of each cell in `fallback_counts`.
- `predict()`: per cell and call, in fixed order:
  1. `U` = `n_samples` uniforms;
  2. the A-draws: `rng.choice(pool, n_samples)`;
  3. the C-draws: `kde.draw_kde_cell`.

  Output is `where(U < alpha, C, A)`. The whole grid is built through
  `sample_prediction_grid` (ADR-011).
- **Common random numbers.** No random number E consumes depends on `alpha`. So at a fixed
  seed, raising `alpha` only switches individual draws from their A value to their C value, and
  the switched set grows with `alpha`. An α-sweep is therefore a coupled, paired design.
- **Integer output**: A draws are window values; C draws are counts ≥ 1 or 0.
- A cell C does not smooth is drawn from A in both slots. It is counted in **one aggregate
  WARN** per `predict()`.

---

## Inputs and Assumptions

| Parameter | Type | Notes |
|---|---|---|
| `targets`, `window_months`, `partition_dict`, `loa`, `n_samples` | | As `ConflictologyModel`. |
| `alpha` | `float` | Required, audited genome key. The weight on C. |
| `bandwidth` | `float \| str` | Required, audited genome key, as C. |
| `seed` | `int` | Required, audited genome key. |

**Cost:** about three times A's per-cell sampling, because it draws three streams.

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| Invalid `alpha` or `bandwidth` | `ValueError` at `__init__` | Also raised via the catalog. |
| Cell not smoothed by C | Drawn from A in both slots, plus the aggregate WARN | Counted by kind, including `selector_failed`. |
| NaN in the window / `window_months <= 0` / bad input | `ValueError` | `window_pool` / `to_feature_frame`. |

---

## Boundaries and Interactions

- **Depends on:** `kde` (`fit_kde_cell`, `draw_kde_cell`, `validate_bandwidth`) and `smoothing`
  (`draw_fallback`, `warn_fallbacks`). It has no copy of C's logic, so if C changes, E changes
  with it. That is intended: E is defined as a mixture *with C*.
- **Instantiated by:** `BaselineModelCatalog._get_empirical_kde_mixture()`.
- **Genome:** `["window_months", "n_samples", "seed", "alpha", "bandwidth"]`.

---

## Examples of Correct Usage

```python
from views_baseline.model.models.distributional import EmpiricalKDEMixtureConflictology

# An α-sweep at a fixed seed is a paired design: only alpha changes.
for alpha in [i / 10 for i in range(11)]:
    m = EmpiricalKDEMixtureConflictology(
        targets=["lr_sb_best"], window_months=36, partition_dict={"test": (457, 504)},
        loa="pgm", n_samples=256, alpha=alpha, bandwidth="sj", seed=42,
    )
    pf = m.fit(df).predict(df=df, sequence_number=0, output_length=36)["lr_sb_best"]
```

## Examples of Incorrect Usage

```python
# Changing the seed together with alpha throws away the coupling: differences across α
# then carry Monte-Carlo noise of the full stream, not just the switched draws.
# Expecting E(alpha=0) to equal ConflictologyModel byte for byte: it is equal in distribution only.
```

---

## Test Alignment

All in `tests/test_smoothed_climatology.py` unless noted.

| Test | What it verifies |
|---|---|
| `test_e_coupling_raising_alpha_only_switches_a_draws_to_c_draws` | Common random numbers: each draw is its A or C value, switches are monotone in α, and the switched share is ≈ α. |
| `test_e_at_alpha_zero_is_a_in_distribution`, `test_e_at_alpha_one_is_c_in_distribution` | The endpoint invariants. |
| `test_e_pool_and_zero_rate_are_as`, `test_e_zero_mass_equals_as` | Pool and zero mass are A's, at α ∈ {0, 0.5, 1}. |
| `test_e_mixing_widens_spread` | Mixing adds positive values A cannot draw. |
| `test_e_accepts_valid_alpha`, `test_e_rejects_invalid_alpha`, `test_e_rejects_invalid_bandwidth`; catalog alpha tests in `test_catalog.py` | Validation and genome. |
| `test_e_counts_fallbacks_and_warns_once`, `test_e_fallback_cells_draw_only_window_values` | Fallbacks. |
| `test_golden_e_with_zeros`, `test_e_reproducible_and_seed_sensitive` | Exact draws and seed wiring. |

---

## Evolution Notes

- **Evolving** (ADR-004).
- If #109 shows E is too wide, the known remedy is the beta-transformed linear pool
  (Gneiting & Ranjan 2013), which is out of scope for now.
- The conformance fixture is derived until views-models ships an E config (#109).
