# S8 — Closeness-to-conflictology experiment: findings

**Epic #33 / ADR-022 · Story S8 (#41)** · date 2026-07-18
Harness: `views_baseline/evaluation/closeness.py` (S3). Runner: `reports/closeness_experiment/run.py`. Raw: `results.json`.

## Setup

- **Data:** `white_ranger/data/raw/calibration_viewser_df.parquet` (real pgm), targets `lr_sb_best`, `lr_ns_best`, `lr_os_best`.
- **Config (real white_ranger):** `window_months=36`, calibration partition `test=(457,504)` → `train_end=456`, `loa="pgm"`.
- **Fidelity setting:** `n_samples=1000` per cell (ADR-022 operationalises closeness at ~1000 draws; independent of the deployed `n_samples=64`).
- **Reference / null:** conflictology(seed 1) as reference; conflictology(seed 2) vs reference is the **same-model null floor**.
- **Entity subsample:** all 1197 window-active entities + 4000 random inactive = 5197 (reproducible, `SUBSAMPLE_SEED=20260718`), so the low/high strata are populated rather than drowned by the 99.5%-zero majority.
- **Strata (by reference per-cell mean, `lr_sb_best`):** zero n=4413, low (<1) n=489, high (≥1) n=295, all n=5197.
- **Pre-registered δ (ADR-022 §4):** C2ST ≤ 0.55; Wasserstein ≤ null-median + 0.5.

## Result 1 — everyone is indistinguishable on the mass of the data

On the **zero** and **all** strata (≈99% of cells) every candidate sits at C2ST ≈ 0.500 and is judged **equivalent** to conflictology on all three targets. The zero-spike is matched exactly by every family (`clamp_floor` + point-mass/hurdle zero-rate). The action is entirely in the **high-activity** stratum.

## Result 2 — high-activity C2ST (closest to 0.500 wins)

| candidate | lr_sb high | lr_ns high | lr_os high |
|---|---|---|---|
| conflictology-null | 0.499 | 0.499 | 0.500 |
| **nb / none** | **0.578** | 0.560 | **0.556** |
| lognormal / none | 0.608 | 0.567 | 0.571 |
| lognormal / log1p | 0.607 | 0.566 | 0.572 |
| gamma / none | 0.610 | 0.562 | 0.571 |
| gamma / log1p | 0.607 | 0.563 | 0.576 |
| **gumbel / none** | 0.604 | **0.555** | 0.568 |
| gumbel / log1p | 0.608 | 0.557 | 0.572 |

By C2ST, **nb (no-hurdle) is best or tied-best on 2/3 targets**; gumbel/none best on `lr_ns`. No candidate reaches the δ band (0.55) on high activity → **no family is perfectly indistinguishable from conflictology on active cells** — exactly as ADR-022 predicted for a smooth fit to a tiny discrete window.

## Result 3 — high-activity Wasserstein / energy (lower is closer) *disagrees with C2ST*

| candidate | lr_sb W (E) | lr_ns W (E) | lr_os W (E) |
|---|---|---|---|
| conflictology-null | 0.446 | 0.351 | 0.280 |
| nb / none | 1.540 (0.327) | 1.407 (0.297) | 1.032 (0.263) |
| lognormal / none | 1.454 (0.230) | 1.187 (0.182) | 0.876 (0.185) |
| **gamma / none** | **1.069 (0.220)** | **0.954 (0.177)** | **0.685 (0.172)** |
| gumbel / none | 1.170 (0.239) | 0.954 (0.201) | 0.816 (0.191) |
| lognormal / log1p | 6.546 (0.316) | 3.057 (0.245) | 2.009 (0.224) |
| gamma / log1p | 1.893 (0.246) | 1.485 (0.190) | 1.170 (0.188) |
| gumbel / log1p | 2.221 (0.265) | 1.723 (0.204) | 1.327 (0.203) |

By **magnitude fidelity (Wasserstein & energy), `gamma/none` is the clear winner across all three targets**, gumbel/none second. nb is among the *worst* on Wasserstein despite winning on C2ST.

### Why C2ST and Wasserstein disagree (the key insight)

- **C2ST (1-NN)** rewards matching conflictology's *discrete integer support*. `nb` emits integer counts, like conflictology's resampled window values, so a nearest-neighbour classifier struggles to separate them → low C2ST. Continuous families are trivially separable by their non-integer support → higher C2ST, *regardless* of how well they match the magnitude.
- **Wasserstein/energy** reward matching the *magnitude distribution* (spread, tail). `gamma/none` fits the positive-part shape best → lowest transport cost.

So the two metrics answer different questions: C2ST ≈ "can a classifier tell them apart at all" (support-sensitive); Wasserstein/energy ≈ "how far apart are the magnitudes." Both were pre-registered; we report both rather than collapsing them.

## Result 4 — `transform="log1p"` strictly *hurts* on active cells

For every family and every target, `log1p` inflates high-activity Wasserstein (e.g. lognormal 1.45 → **6.55**) and energy vs `none`. The per-sample `expm1` detransform amplifies the heavy right tail of conflict magnitudes (large log-space draws → very large raw draws). **`transform="none"` dominates `log1p`** on active cells; log1p should not be a default. (log1p is still a legal, audited genome option — it just loses here.)

## Result 5 — the non-negativity floor (C-26) is load-bearing on real data

The gumbel runs logged dozens of `floored … at EMIT_FLOOR=0.0` warnings (up to 26 samples/cell). Without the S6 floor these would have been **negative fatalities**, silently biasing every metric above. The C-26 fix is exercised by real data, not just the synthetic test.

## Ranking & recommendation

| Goal | Best family | Notes |
|---|---|---|
| Literal indistinguishability (C2ST, ADR headline) | **nb** (`ParametricConflictology`, `none`) | best/near-best high-activity C2ST on 2/3 targets; shares conflictology's discrete support; ~0.01 behind hurdle families on the sparse *low* stratum |
| Magnitude fidelity (Wasserstein/energy) | **gamma / none** (`ParametricHurdleConflictology`) | lowest transport cost on all 3 targets |
| Robust all-rounder / uncertainty (Vesco-motivated) | **gumbel / none** | 2nd on Wasserstein, best C2ST on `lr_ns`; right-skew tail |
| — avoid — | any `log1p` | strictly worse on active cells |

**Overall:** for a single "closest to conflictology" default, **`nb` (no-hurdle)** best satisfies the ADR's C2ST-indistinguishability framing; **`gamma/none`** is the pick when magnitude fidelity matters more than support-matching. Neither is equivalent on high-activity cells — that residual gap is the expected cost of a smooth parametric fit and is the natural target for **S9 (Tweedie)**, whose native zero-inflation + continuous positive part may close both metrics at once. This ranking must be **re-run including Tweedie in S9**, and the recommendation carried into the **S10** CICs/README.

---

## S9 — Tweedie evaluated and **excluded** (negative result)

Tweedie (compound Poisson-Gamma, `1<p<2`) was implemented and measured as a native-zero, no-hurdle family: the index `p` is derived from the window's mean/var/**zero-rate** (three moments → three params) so the emitted zero mass matches the empirical zero-rate exactly. It was the only single family modelling the zero-spike **and** a continuous positive part together. **It has been removed from the baseline.** The evidence below is the exclusion rationale.

### What the measurement showed (before removal)

On high-activity cells Tweedie was competitive on *magnitude* — best/near-best Wasserstein & energy across targets (e.g. `lr_ns` W=0.879, best; energy 0.170/0.217 best on `lr_ns`/`lr_sb`), clearly better than `nb` there. But on **C2ST it lost to `nb`** (0.611/0.565/0.577 vs nb 0.578/0.560/0.556) for the same reason every continuous family does: non-integer support is trivially separable from conflictology's resampled integers.

### Why it was excluded (the decisive finding)

A **family-feasibility** check on the real pgm windows: Tweedie(`1<p<2`) can match a window's (mean, variance, zero-rate) only when `m² < var·(−ln zero_rate)`. On the S8 subsample:

| target | active cells | feasible (`1<p<2`) | infeasible (`p≤1`, clamped) | catastrophic (`p≥2`) | max clamped λ |
|---|---|---|---|---|---|
| lr_sb | 773 | 447 | 326 | **0** | 0.2 |
| lr_ns | 380 | 207 | 173 | **0** | 0.1 |
| lr_os | 562 | 255 | 307 | **0** | 0.3 |

- **The OOM/heavy-tail case (`p≥2`) never occurred** — conflict windows are too zero-inflated/overdispersed to reach it (so the earlier C-27 OOM risk was theoretical; it is now withdrawn).
- **~40% of active cells are *infeasible* (`p≤1`)** — windows with **1–2 events in 36 months**. A continuous family fundamentally cannot represent "exactly one event, ever." Every *other* family degrades gracefully here (point-mass/Poisson) and stays parametric; Tweedie hits a hard feasibility wall.

The only two honest ways past that wall were **(a) clamp `p`** — silently approximates a fit that doesn't exist (hides misspecification, violates ADR-021 "no silent magic"), or **(b) empirical fallback** — resample the window, i.e. *become conflictology* on 40% of cells, which is no longer the parametric generative model this epic is building. Neither is acceptable, so Tweedie was dropped.

### Ranking without Tweedie (final)

| Goal | Best family |
|---|---|
| Literal 1-NN indistinguishability (ADR C2ST framing) | **`nb`** (`ParametricConflictology`) |
| Magnitude fidelity via hurdle / Vesco uncertainty | **`gamma/none`**, then **`gumbel/none`** (`ParametricHurdleConflictology`) |
| — avoid — | any `log1p` (strictly worse on active cells) |

**Bottom line:** with Tweedie removed, the closeness picture is: **`nb` wins indistinguishability** (matches conflictology's discrete support), **`gamma/none` wins magnitude fidelity**, and no smooth family is equivalent on high-activity cells — the irreducible cost of fitting a ≤36-point discrete window. Tweedie remains a sensible candidate for a *future data-rich* (pooled/covariate) model where each fit sees enough observations that the feasibility wall disappears. Carry the `nb` / `gamma` defaults into the **S10** CICs/README.

## Reproduce

```
conda run -n views_pipeline python reports/closeness_experiment/run.py
```
Deterministic given the fixed subsample + model seeds. Console prints the C2ST table; `results.json` holds full stratified Wasserstein/energy/C2ST percentiles + equivalence verdicts.
