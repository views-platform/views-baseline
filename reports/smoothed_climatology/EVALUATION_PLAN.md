# Evaluation plan: smoothed climatology baselines (epic #104, Phase 1)

**Status:** pre-registered on 2026-11-03, **before any model exists or any result is seen**. Change
it only by a dated amendment at the bottom. Never edit it to match a result.

**Governs:** #109, the Phase-1 experiment.
**Decision:** ADR-023.
**Lessons source:** `views-hydranet/reports/2026-09-30_evaluation_learning_report.md` (cited as
"HN §x"). It is a living document, so re-read it before running and record the revision date
used.

---

## 1. What this experiment is, and is not

- **It is a measurement.** For each of B, C and E, it measures where its predictive distribution
  differs from A's (`ConflictologyModel`). The answer is broken down by part of the distribution
  (zeros, small counts, large counts, spread) and translated into what it suggests for neural
  output heads.
- **It is not a ranking.** No metric decides a winner and nothing is promoted (ADR-023 §4; Simon,
  2026-11-02). A statement of the form "X is better" may appear only as "X differs from A in
  direction d on metric tuple t, beyond its noise floor".

## 2. The structural facts the design rests on

1. **A, B, C and E have identical zero mass by construction** (ADR-023 §2). Any metric that reads
   only occurrence must show **no difference** beyond Monte-Carlo noise. This is the built-in
   control: the HN §8 row-17 requirement for a prediction on a property the intervention does not
   act on.
2. **The models also share pool, cells and placement.** So placement-only metrics (FSS-type, AP on
   occurrence) are expected to show no difference either. HN §9.2 found that 14 of 20 columns read
   the mean field alone.
3. **All differences live in the positive part:** shape, spread and tail.

## 3. Setup

| item | value |
|---|---|
| where | views-models, as clones of `white_ranger`; only `algorithm` and its genome keys vary (#33 precedent) |
| level / cells | pgm, `white_ranger`'s queryset; **record the cell count and region name** (`land` = 64,818 vs `land_gaul` = 64,742 differ) |
| window | `window_months = 36` for every model |
| partitions | calibration **and** validation, reported side by side; no partition is privileged, since nothing is selected |
| transform | count space; no log, no normalisation |
| draws | **S = 256 for every model, A included** (HN §2.3, §8 ★4: unequal S biases CRPS and quantises AP differently) |
| seeds | **10 seeds, the same list for every model** (paired; HN §2.5). The noise is Monte-Carlo only, since nothing is trained |
| C/E knobs | `bandwidth` ∈ {a small numeric grid, `"sj"`, `"silverman"`}; E's `alpha` ∈ {0, 0.1, …, 1.0} at the C bandwidth closest to A on the §5 control metrics |
| CRPS estimator | **fair (ensemble-size-adjusted) CRPS** (Ferro 2008), stated in every table; plug-in vs fair moved CRPSS by 18% at S=16 (HN §9.4c) |

Scoring runs offline on the saved prediction files. Every metric is a full
**(column, population, transform, estimator)** tuple (HN §9.4b), named as such in every table.

## 4. Populations

| name | definition | conditioning | role |
|---|---|---|---|
| `all` | all cell-months | none | report |
| `hist_active` | cells with ≥ 1 positive value **in their pool window** | **observed history** (proper; HN §8.3) | report |
| `hist_zero` | cells with no positive value in the window | observed history | report |
| `events` | cell-months where the **outcome** is positive | **outcome** (forecaster's dilemma, Lerch2017) | ⛔ **diagnostic only**, never compared as a verdict |

Never condition on a model-inferred state (HN §8.3).

## 5. Metric tuples

Each tuple has a **predicted direction for B/C/E vs A**, recorded now. "—" means no prediction.

**Zeros / occurrence (control).**

| # | column | population | sensitive to | predicted B/C/E vs A |
|---|---|---|---|---|
| Z1 | `act_ratio`: Σ 1(draw>0) / Σ 1(y>0), draw-averaged | `all` | occurrence | **equal** (control) |
| Z2 | Brier on P(Y>0) = positive-draw fraction | `all`, `hist_active` | occurrence | **equal** (control) |
| Z3 | AP on P(Y>0) | `all` | occurrence rank / placement | **equal** (control) |

**Overall (reported, not ranked; HN §2.2, §3.2).**

| # | column | population | sensitive to | predicted |
|---|---|---|---|---|
| O1 | fair CRPS | `all` | mostly zeros | ≈ equal; its **dynamic range across A/B/C/E is reported** before anything is read off it |
| O2 | fair CRPS + `zero_share_of_gap` vs A | `hist_active` | positive part | — |

**Positive part: level and shape.**

| # | column | population | sensitive to | predicted |
|---|---|---|---|---|
| P1 | `mass_ratio`: Σ E[Y] / Σ y (ratio of sums) | `hist_active` | level | B ≈ A; C/E slightly above A (reflection moves mass up) |
| P2 | W₁ between the pooled positive draws and the pooled observed positives | `hist_active` | **marginal severity shape** (not conditional calibration; HN §8.3) | — |
| P3 | `size_ratio`: **median** of guess/truth | `events` | level on truth-positives | diagnostic only |
| P4 | fair CRPS | `events` | positive-part fit | diagnostic only |

**Spread / calibration.**

| # | column | population | sensitive to | predicted |
|---|---|---|---|---|
| S1 | randomized PIT histogram (Czado 2009) and **PIT variance** (1/12 = right spread) | `hist_active` | spread | A **under-dispersed** (#33); C and E (as α rises) wider than A |
| S2 | central 50% / 90% interval coverage | `hist_active` | spread | A below nominal; C/E closer |

**Tail.**

| # | column | population | sensitive to | predicted |
|---|---|---|---|---|
| T1 | Brier on P(Y ≥ k), k ∈ {10, 100, 1000} | `hist_active` | tail, threshold-based (**not** quantile-based: QS99 sits inside the zero mass; HN §1 am. 2) | above a cell's window maximum, A = B = 0 (neither can exceed it); C and E > 0 |
| T2 | threshold-weighted CRPS, weight 1(y ≥ 100) (proper; Gneiting2011) | `hist_active` | tail | diagnostic only |
| T3 | AP on P(Y ≥ 100) | `all` | tail ranking | — |

The **zero metrics never use outcome-conditioned populations.** The `events` rows are
explanations, never verdicts.

## 6. Noise floor and what counts as "a difference"

- **Seed SD.** For every tuple and model, compute the metric per seed and report the seed SD.
  - This is **Monte-Carlo noise only**: these models are deterministic given history.
  - ⚠️ It is **not comparable** to hydranet's retrained-model floors (HN §9.4b), and every table
    that could be mixed with theirs must say so.
- **Paired differences.** Report the difference to A per seed (same seed for both), as a mean with
  a 90% interval across seeds.
- **When a difference is reported.** A difference is reported as such only if its paired interval
  excludes zero. Otherwise the result is "no resolvable difference at S=256, 10 seeds", which is a
  valid finding.
- **Control failure.** If **Z1–Z3 show a resolvable difference**, the experiment is **VOID** until
  explained, because the zero mass is supposed to be identical by construction (HN §2.11: VOID is
  a third outcome).

## 7. Degenerate-forecast checks, run through the actual scorer before real data (HN §8 row 5)

| forecast | must show |
|---|---|
| all-zero | Z1 → 0, P1 → 0; **O1 may look good** (the expected trap), and the report must show it |
| point mass at A's per-cell mean (no spread) | S1 PIT variance far from 1/12; S2 coverage ≈ 0 |
| C with a huge bandwidth | S1 over-dispersed (PIT variance < 1/12); T1 too high at large k |
| A vs A at a different seed | every tuple's difference inside its noise floor (the null) |

If a check does not fire as stated, that metric is dropped from the report and the drop is
recorded.

## 8. Report structure (#109)

1. Setup actually used: region and cell count, S, seeds, the HN revision date, and fallback counts
   per model (cells below `MIN_POSITIVES_FOR_SMOOTHING`, selector failures).
2. Control check (Z1–Z3) **first**.
3. Per model, per tuple: value, seed SD, paired difference to A. Calibration and validation side by
   side.
4. Predicted vs observed direction for every row of §5, with misses stated plainly.
5. **What this means for network output heads**, per estimator:
   - which part it improved or worsened;
   - the candidate head (#104 table);
   - which representations should **not** be tried, and why.
6. What the experiment cannot see:
   - spatial and temporal dependence (every metric here is marginal);
   - whether the target itself is right (HN §5.5).

## 9. Separate and not part of this plan

The **hydranet α-sweep** (#109, conditions recorded there) uses hydranet's frozen ruler,
`land` = 64,818 cells, months 457–469 and S = 256. It delivers a seed × α × column table. Its
numbers are properties of the views-baseline generator and are never mixed into §5's tables.

---

## Amendments

*(none)*
