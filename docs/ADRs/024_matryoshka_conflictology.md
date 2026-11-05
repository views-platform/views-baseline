# ADR-024: MatryoshkaConflictology: a climatology that opens outward as its information ages

**Status:** Proposed (draft for discussion)
**Date:** 2026-11-05
**Deciders:** Simon
**Consulted:** views-models session (Phase-1 measurements); views-hydranet session (evaluation lessons)
**Informed:** views-models maintainers

---

## Context

`ConflictologyModel` (**A**) forecasts each cell by resampling its own last `window_months` values.
ADR-023 added three smoothings of A (B, C, E). The Phase-1 run of all ten views-models#695 models
(results and corrections on #109, #82, #122) found:

1. **"Where" is solved by each cell's own history.** AP ≈ 0.355 against a prevalence of 0.39% (about
   91× chance), in every arm.
2. **Smoothing is not the lever.** The arms differ by at most 1.09–1.29× on any properly conditioned
   axis. B, C and E inherit A's main defect unchanged.
3. **The defect is staleness.** The pool is fixed at `train_end` for every origin (#82), so a cell's
   P(y>0) is the same at every lead, while reality reverts: in the top predicted bin (~0.85), the
   realised rate falls from 0.64 at h = 1 to 0.60 at h = 18 and 0.28 at h = 36. At origin_12, "h = 1" is
   already 13 months after the window ends. The pattern holds on all three arms tested, in two bins.
   It is not an end-of-data artefact: month 504's prevalence is normal.
4. **Cold cells are under-confident.** ~1,800 conflict cells had predicted P = 0 exactly: a cell whose
   own window has no positives can never fire.

Both defects are the same mechanism. The forecast does not know that a cell's own past loses
relevance with lead time, and has nothing to fall back on when it does.

**Second purpose (as in ADR-023): an uncertainty prototype.** The question "how fast does a cell's own
history stop predicting it, and what should replace it" is exactly what a learned model should beat.
Its answer, a few fitted time constants, is a reference a network head can be compared against.

## Decision

Add **`MatryoshkaConflictology`**: a climatology that draws from nested spatial rings, each with its
own memory. As the forecast looks further ahead, the inner rings' information ages and the forecast
**opens outward** to larger, slower-changing rings.

**The dolls.** Ring k = 0 is the cell itself. Ring k = 1..K are its k-th order queen-contiguity
neighbours on the PRIO-GRID (computed from `priogrid_id = (row − 1) × 720 + col`, verified on all
64,818 `land` cells). An outermost ring is a broad level (country, or the whole map; see Open
Questions).

**The weight** of a past observation of ring k in month s, when forecasting target month t:

```
w(k, s; t) = ring_weight[k] × exp(−(t − s) / memory[k]) / n_k(s)
```

- **Age is measured to the target month t**, not to the end of the history. So the same observation
  weighs less at longer lead, and lead-dependence needs no separate parameter. Because memories
  differ by ring, the **mix** shifts outward as the target recedes.
- **`n_k(s)` averages within the ring.** Ring k has about 8k cells; without averaging, outer rings
  would swamp the cell itself.
- **Expected ordering, not enforced:** memory[0] < memory[1] < … (hotspots cool faster than regions).
  The fit decides; an inverted fit is a finding, not an error.

**The draw (per cell, target, target month, sample):** pick a past observation with probability
∝ w, and return its value. This is A's resampling, weighted: zeros and sizes come from the same
draw, every draw is a value that actually occurred (so integers, with no smoothing in value space),
and P(y>0) is the weighted share of positive observations.

**A is a special case.** With ring_weight = [1, 0, …], memory[0] → ∞ and the history cut at
`window_months`, Matryoshka equals A in distribution. The comparison with A is therefore a test of
the two added ideas (rings, decay), nothing else.

**The information cutoff stays at `train_end`**, as in A, so the two are comparable. Whether to
condition on months up to each origin instead is #82's question, and the sliding-window control
below answers it separately.

**Knobs are fitted, not tuned by hand.** With K = 2 plus the broad level, there are 4 memories and 3
free ring weights. They are shared across the whole map and fitted by rolling backtest **inside the
training partition**. The objective is a proper score on occurrence by lead (Brier or log loss, so
calibration is rewarded), with size checked by CRPS on positive cells. **Not CRPS alone** (#122). The
fitted values are reported; they are the scientific output.

**First step, before Matryoshka: a sliding-window control.** A with its window ending at each origin
(#82's other option). It is small to build and separates "the window is stale at the origin" from
"lead time within an origin". If it fixes most of the staleness, Matryoshka's lead mechanism has less
to do, and that is worth knowing first.

## Rationale

- **It targets the measured defect,** not a hypothesised one. Smoothing addressed a gap that turned
  out not to matter; staleness is measured, on all arms, in a properly conditioned check.
- **One mechanism, three open issues.** It joins lead-awareness (#82), temporal decay (#111) and spatial
  pooling (#112) into one weighting, instead of three models whose interactions would be untested.
- **Minimal and interpretable.** No covariates; ≤ 7 fitted numbers, each with a physical meaning (a
  memory in months, a ring's share). It nests A exactly.
- **No new values invented.** Unlike C, every draw happened somewhere nearby and recently. That keeps
  integers natural and keeps the C boundary bias (ADR-023) out.

## Considered Alternatives

### Independent time and space knobs
One decay for time, one mix for space. **Rejected:** the mix between cell and neighbours would be the
same at h = 1 and h = 36, which is exactly the flaw measured. The rings must differ in memory.

### Hierarchical Bayesian partial pooling
Shrink each cell toward its region with fitted variances. More principled, but much more machinery,
and it does not by itself make the shrinkage grow with lead. **Deferred;** Matryoshka's fitted
memories are a cheap first answer to whether that machinery would pay.

### Sliding-window A only
**Kept as the control,** not the model. Within one origin it still predicts the same P at h = 1 and
h = 36.

### More smoothing variants (D, #110)
**Parked:** Phase-1 shows smoothing does not touch the defect.

## Consequences

### Positive
- No cell is certain of zero unless its whole neighbourhood and region are quiet.
- Lead-dependent forecasts from a climatology, a stronger yardstick for learned models.
- The fitted memories are interpretable findings in themselves.

### Negative
- **Fitting is new for this repo.** Every existing baseline is fit-free apart from its window. A
  backtest loop adds code and run time, and needs its own tests.
- **Heavier prediction.** The draw depends on the target month, so it is per cell-month, not per cell.
  Ring averages per month must be precomputed on the 360 × 720 grid.
- **Misplaced mass spreads further.** A cell-month that holds a whole region's annual toll (UCDP event
  463131: ~113,000 in one cell, month 504; views-datafactory#533) would now enter its neighbours'
  draws too. This is fixed upstream; this repo does not winsorise.

## Implementation Notes

- New class in `views_baseline/model/models/distributional/`, registered in the catalog, with exact
  genome keys (ADR-021: no defaults for the fitted or structural keys).
- Structural keys: number of rings K, the broad level's definition, the history cap, the backtest
  design. Fitted values are stored on the model and reported, not configured.
- Neighbours are computed from `priogrid_id`. pgm only at first; cm has no geometry here (Open
  Questions).
- The same RNG and seeding discipline as the other distributional models (`sample_prediction_grid`).
- Evaluation memory is handled by the streaming path (#121).

## Validation & Monitoring

- **Equivalence:** with the special-case parameters, matches A in distribution (a property test, as
  for E at α = 0).
- **Measured as Phase-1 was:** against A and the sliding-window control at equal S. Report:
  - reliability of P(y>0) **per lead** (the check that exposed the defect);
  - AP (must not get worse);
  - size conditional on the forecast;
  - CRPS split by true-zero and true-positive cells.
  
  All of these are reported with and without the cells flagged by views-datafactory#533, until that
  fix lands.
- **Report-only,** as ADR-023: a table of properties, not a declared winner.

## Open Questions

1. **The broad level:** country (borders matter for conflict) or the whole map? Decided by the
   backtest; both are cheap. Note that in the datafactory pull, `c_id` is a GAUL code, not a VIEWS
   country id (views-models#660).
2. **K, the number of neighbour rings:** 2 is the proposal; the backtest can test 1 and 3.
3. **Lead vs calendar month** is not yet separated in the evidence: does the decay follow months since
   the history ended, or the calendar? Pending the models session's check on another origin. The design
   assumes the former.
4. **The history cap:** with decay, the full history can be used, but a cap bounds cost and the
   influence of early, sparser data.
5. **cm level:** no ring geometry here. Either cell-only with decay at cm, or a country-contiguity source.
6. **Per-target memories** (sb / ns / os) or shared? Shared is simpler; violence types may age
   differently.

## References

- #82 (fixed vs sliding window), #109 (Phase-1 results and corrections), #111, #112, #122 (scoring).
- ADR-021 (no magic defaults), ADR-023 (smoothed climatology; report-only evaluation).
- views-models `reports/2026-11-04_baseline_phase1/` (measurements, including `cond_fcst.py`).
- views-datafactory#533 (misplaced aggregate events).
