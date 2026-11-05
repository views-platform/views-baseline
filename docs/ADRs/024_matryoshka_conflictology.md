# ADR-024: MatryoshkaConflictology: a climatology that opens outward as its information ages

**Status:** Proposed (draft for discussion)
**Date:** 2026-11-05
**Deciders:** Simon
**Consulted:** views-models session (Phase-1 measurements, lead-vs-calendar and cliff checks); views-hydranet session (evaluation lessons)
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
3. **The forecasts are too narrow at both ends, and the zero end gets worse with age.**
   - *Hot cells:* where conflictology predicts P(y>0) ≥ 0.70, conflict happens about 0.73 as often
     as predicted, i.e. ~27% over-confidence. Per month it is flat over months 457–500.
   - *Quiet cells:* ~1,800 conflict cells had predicted P = 0 exactly. A cell whose own window has no
     positives can never fire.
   - *Age:* a cell's own quiet history stops being evidence of quiet as it ages. Measured by refitting
     at six cutoffs (below); conditioned on the forecast (P = 0 exactly), not on the outcome.

**Evidence on age (2026-11-05, #109).** With one pool fixed at `train_end`, lead and calendar month are
the same variable (lead = month − 456), so Phase-1 could not test age. Its apparent "fading over the
horizon" (0.75 → 0.70 → 0.33) was a flat line plus a **global** drop in Sep–Dec 2021 (months 501–504,
rest of world 0.73 → 0.34; not Ethiopia, not views-datafactory#533; cause open). **It is withdrawn as
evidence.**

The test that does separate them: conflictology refitted with the pool ending at 396, 408, …, 456, every
later month scored up to 492 (2021 excluded), and age separated from calendar year by a two-way fit. The
fit assumes no cutoff-specific effect; checked by dropping the earliest and then the latest cutoff.

| stratum (conditioned on the forecast) | age effect by year 7 | all cutoffs / −396 / −456 |
|---|---|---|
| quiet (P = 0), sb: conflict per 1,000 cell-months | rises from ~0.4 | +0.89 / +0.88 / +0.94 |
| quiet, ns | rises | +0.62 / +0.62 / +0.69 |
| quiet, os | rises | +0.42 / +0.39 / +0.43 |
| hot (P ≥ 0.5), os: realised / predicted | falls | −0.33 / −0.51 / −0.29 |
| hot, ns (year 3) | falls | −0.38 / −0.29 / −0.60 |
| hot, sb (year 3) | about nil | −0.07 / −0.03 / −0.08 |

- **The quiet-cell gradient is robust in sign and size** (within ~11% across cutoff sets; tens of
  thousands of cell-months per entry). It can carry a fitted value.
- **The hot-cell fading is robust in sign only** (its size moves 75–100% when one cutoff is dropped).
  It supports memories that differ by violence type, not their length.
- **sb hot cells have no age effect,** only the constant offset. That is a calibration defect this
  design is not aimed at. Pulling them toward their rings may reduce it; that is not promised.
- Calendar effects (e.g. the late-2021 drop) remain. A climatology cannot foresee them.

Scripts: views-baseline session scratch `cutoff_check.py` (to be committed with the backtest); the
models session's `lead_vs_calendar.py` and `eth_step.py` (views-models `docs/baseline-phase1-campaign`,
`a001aba7`).

The mechanism behind the age effect: the forecast does not know that a cell's own past loses relevance
with age, and has nothing to fall back on when it does.

**Second purpose (as in ADR-023): an uncertainty prototype.** The question "how fast does a cell's own
history stop predicting it, and what should replace it" is exactly what a learned model should beat.
Its answer, a few fitted time constants, is a reference a network head can be compared against.

## Decision

Add **`MatryoshkaConflictology`**: a climatology that draws from nested spatial rings, each with its
own memory. As the forecast looks further ahead, the inner rings' information ages and the forecast
**opens outward** to larger, slower-changing rings.

**The dolls (pgm).**
- **Ring k = 0** is the cell itself.
- **Rings k = 1, 2** are its first- and second-order queen-contiguity neighbours on the PRIO-GRID,
  computed from `priogrid_id = (row − 1) × 720 + col` (verified on all 64,818 `land` cells).
  **Two rings**, decided 2026-11-05.
- **The country** is the next doll (`c_id`, one per cell).
- **The whole map** is the outermost doll, as a backstop. It must earn a non-zero weight in the fit;
  it always serves the ~76 `land` cells with no country.

**At cm level, no rings for now:** the country itself, with memory decay, then the whole map. A
region or theatre doll, or country contiguity, can be added later.

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

**Each doll has its own free weight and memory** (decided 2026-11-05):
- the cell: a memory, with its weight fixed at 1 as the reference;
- ring 1, ring 2, the country and the whole map: a weight and a memory each.

That is **9 knobs per target**. A geometric rule (ring_weight[k] = ρ^k, memory[k] = memory[0] × γ^k)
would save only one knob with two rings, so it is not imposed. The fit reports whether the free values
happen to follow it, with the Currents-of-Conflict ρ ≈ 0.4 as the check. The rule is reconsidered only
if a later version adds more rings.

**Separate knobs per violence type** (sb, ns, os), decided 2026-11-05: they plausibly persist and
spread differently, and the model already runs per target. A shared fit is reported as a check, and
taken instead if it does as well.

**Starting values and plausible ranges** come from von der Maase (2025), *The Currents of Conflict*
(arXiv:2506.06828):
- Spatial Matérn-3/2 lengthscale ℓ = 0.72° over cell centroids. That gives a ring-average correlation
  of ≈ 0.58 (ring 1) and ≈ 0.24 (ring 2), i.e. **ρ ≈ 0.4**, with ring 3 ≤ 0.1.
- Temporal lengthscales ≈ **4 months** (short-term trend) and ≈ **122 months** (long-term trend).

Those were fitted on log fatalities and the 60 most conflict-heavy cells. So they seed the backtest
and bound a plausible range; they are not fixed. A fit that lands near them is independent
confirmation; one that lands far away is a finding.

**Knobs are fitted, not tuned by hand.** They are shared across the whole map and fitted by rolling
backtest **inside the training partition**. The objective is a proper score on occurrence by lead (Brier or log loss, so
calibration is rewarded), with size checked by CRPS on positive cells. **Not CRPS alone** (#122). The
fitted values are reported; they are the scientific output.

- **Report stability, not just values:** refit each knob with each backtest cutoff left out in turn.
  A memory pinned down mainly by hot cells is expected to be unstable (see the age evidence), and is
  reported as uncertain.
- **Print n beside every number,** so a reader can see which figures are solid.
- **Report every evaluation window ending at 504 with and without months 501–504.**

**First step, before Matryoshka: a sliding-window control.** A with its window ending at each origin
(#82's other option). It is small to build and separates "the window is stale at the origin" from
"lead time within an origin". If it fixes most of the staleness, Matryoshka's lead mechanism has less
to do, and that is worth knowing first.

## Rationale

- **It targets the measured defect,** not a hypothesised one. Smoothing addressed a gap that turned
  out not to matter. The quiet-cell age effect is measured by a design that separates age from
  calendar, conditioned on the forecast, and robust to the choice of cutoffs.
- **One mechanism, three open issues.** It joins lead-awareness (#82), temporal decay (#111) and spatial
  pooling (#112) into one weighting, instead of three models whose interactions would be untested.
- **Minimal and interpretable.** No covariates; 9 fitted numbers per violence type, each with a physical meaning (a
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
- Neighbours are computed from `priogrid_id`. pgm first; cm uses country → whole map, without rings.
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

Decided 2026-11-05 (Simon):
- the country is the outer doll, with the whole map as backstop;
- two neighbour rings;
- separate knobs per violence type;
- no rings at cm for now;
- **one memory per doll in v1**;
- **free weight and memory per doll in v1** (no geometric ring rule);
- **at least 10 years of history**;
- **the country doll is the FAO GAUL 2024 country (`gaul0_code`)**, the platform's country identity
  (views-datafactory ADR-025). It reaches the data as `c_id`, and is fixed per cell over time.
  hydranet (`bright_starship`) and the chrome_box/crate members (e.g. `ashen_sparrow`) use the same
  mapping. The legacy, time-varying academic country id (G&W / CShapes) is **not** used. In the
  Phase-1 `land` pull: 202 countries; 76 of 64,818 cells have `c_id` ≤ 0 (no country) and draw from
  the whole-map doll instead. Any join to data keyed by another country id (ISO3, legacy ids) needs
  an explicit crosswalk first.
- **fading follows the age of the information (time since the data ended), per violence type.**
  Confirmed by refitting at multiple cutoffs (see Context: Evidence on age). Strong for quiet cells;
  for hot cells, robust in sign only; nil for sb hot cells.

**History length.** Use **at least 120 months** of history, by default all available history (from month
121, i.e. ~28 years at `train_end` = 456), and let the decay do the weighting. A cap is a cost knob, not a
modelling one. The literature supports a long reach:
- Currents of Conflict's long trend has ℓ ≈ 122 months.
- Conflict effects linger for at least 20–25 years, and 20 years of uninterrupted peace lowers the risk
  of major conflict by 84% for a Liberia-like case (Hegre, Nygård & Ræder 2017).
- Hegre et al. (2013) represent the conflict history as **log** years since conflict. That decays more
  slowly than exponentially, so a 120-month exponential memory at the country doll may still forget too
  fast (weight e⁻² ≈ 0.14 at 20 years). The backtest should report where the fitted country memory lands.

Nothing in v1's design is still open; the remaining unknowns are what the backtest measures.

**For v2:**
- **two memories in the inner doll**: a short burst memory (~4 months in Currents of Conflict) on top of
  a long-run level;
- **a region/theatre doll**, and rings at cm (country contiguity);
- **a slower-than-exponential fade** (e.g. log time, after Hegre et al. 2013), if the fitted country
  memory runs to its limit.

## References

- #82 (fixed vs sliding window), #109 (Phase-1 results and corrections), #111, #112, #122 (scoring).
- ADR-021 (no magic defaults), ADR-023 (smoothed climatology; report-only evaluation).
- von der Maase, S. (2025). *The Currents of Conflict: Decomposing Conflict Trends with Gaussian Processes.* arXiv:2506.06828. Table 2 (ℓ_SCE = 0.72°, Matérn 3/2) and the temporal lengthscales (≈ 4.09 and 122.38 months).
- views-models `reports/2026-11-04_baseline_phase1/` (measurements, including `cond_fcst.py`).
- views-datafactory#533 (misplaced aggregate events); views-datafactory ADR-025 (country identity: GAUL).
- Hegre, H., Karlsen, J., Nygård, H. M., Strand, H. & Urdal, H. (2013). Predicting Armed Conflict, 2010–2050. *International Studies Quarterly* 57(2): 250–270 (conflict history as log years in state).
- Hegre, H., Nygård, H. M. & Ræder, R. F. (2017). Evaluating the scope and intensity of the conflict trap. *Journal of Peace Research* 54(2): 243–261 (effects linger ≥ 25 years; 20 years of peace lowers the risk 84%).
