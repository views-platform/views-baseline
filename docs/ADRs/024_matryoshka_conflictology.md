# ADR-024: MatryoshkaConflictology: a climatology that opens outward as its information ages

**Status:** Accepted
**Date:** 2026-11-05
**Deciders:** Simon
**Consulted:** views-models session (Phase-1 measurements, lead-vs-calendar and cliff checks); views-hydranet session (evaluation lessons)
**Informed:** views-models maintainers

---

## In short

Our simplest probabilistic baseline, conflictology, forecasts a grid cell by replaying that cell's own
recent past. That works well for *where* conflict happens. But it has a blind spot: it treats a cell's
past as equally informative whether we forecast next month or four years ahead. A cell that has been
quiet is forecast as *certainly* quiet, however old that quiet history is. Our measurements show that
this certainty gets steadily less justified as the information ages.

`MatryoshkaConflictology` fixes this by also listening to the cell's surroundings, like a set of nested
Russian dolls:

- the **cell** itself is the smallest doll;
- around it, its **nearest neighbours**;
- then the **next ring** of neighbours;
- then its **country**;
- and finally the **whole map**.

Each doll has its own **memory**: how quickly its past stops being informative. A single cell's history
goes stale fastest; a country's or the world's changes slowly. The further ahead we forecast, the more
the inner dolls' information has faded, so the forecast naturally **opens outward** and relies more on
the larger dolls.

Every forecast value is still a value that actually happened somewhere nearby, at some point in the
past. Plain conflictology is the special case where only the innermost doll is used and nothing fades.

---

## Terms used in this document

- **Cell:** one PRIO-GRID cell, 0.5° × 0.5° (about 55 km across at the equator).
- **pgm / cm:** the two levels we forecast: grid cell by month (pgm), and country by month (cm).
- **Conflictology (model "A"):** the existing baseline. For each cell it keeps the last 36 months of
  observed fatalities and forecasts by drawing at random from them.
- **The window, or pool:** those 36 past months a cell's forecast draws from. Today it always ends at the
  last training month (`train_end`, month 456 = December 2017 in the calibration setup).
- **P(fire):** the forecast's probability that a cell sees any conflict (more than zero fatalities) in a
  given month. For conflictology this is simply the share of the window's months that had conflict.
- **Hot cells / quiet cells:** cells the forecast thinks very likely to see conflict (P(fire) of 50% or
  more), and cells it thinks certain not to (P(fire) exactly 0).
- **Calibration:** whether forecast probabilities come true as often as they say. If the forecast says
  80% for many cells, conflict should happen in about 80% of them.
- **Age of information:** how long ago the observed months that a forecast relies on happened.
- **Violence types:** sb = state-based, ns = non-state, os = one-sided violence. Each is forecast
  separately.
- **Knob:** a setting of the model that is estimated from data, not chosen by hand.

---

## Context: why we need this

### What the first round of smoothed baselines taught us

ADR-023 added three "smoothed" versions of conflictology (B, C and E). All ten resulting models were run
in views-models (issue views-models#695), and the results were examined in detail on issues #109, #82
and #122. Three lessons came out of that:

1. **Each cell's own history is already very good at saying *where* conflict happens.** All ten models
   rank cells about 91 times better than chance. A new baseline does not need to improve this.

2. **Smoothing was not the right lever.** The ten models barely differ from one another on any fair
   comparison (at most 1.09–1.29×). Smoothing changes the shape of the forecast slightly, but all three
   smoothed versions inherit conflictology's main weakness unchanged.

3. **The forecasts are too narrow at both ends, and the quiet end gets worse with age.**
   - *Hot cells are over-confident.* Where conflictology says 70% or more, conflict happens only about
     0.73 times as often as predicted, i.e. about 27% too confident. This is roughly constant over time.
   - *Quiet cells are treated as certain.* About 1,800 cells that did see conflict had been forecast at
     *exactly* 0%. A cell whose own window had no conflict can never be forecast to have any.
   - *And that certainty becomes less justified as the information ages.* This is the key finding,
     explained next.

### How we know that age matters (and what we first got wrong)

Our first reading of the Phase-1 results said that "the forecast fades as it looks further ahead". That
reading turned out to be wrong, for an instructive reason:

- Conflictology uses **one window, ending at month 456, for every forecast**. So "how far past the
  window" a forecast month lies is the same thing as "which calendar month it is" (month minus 456).
  With that setup, it is impossible to tell an effect of *age* from an effect of *particular months*.
- Looking month by month, the apparent fading was actually **flat for 44 months, followed by a sudden
  worldwide drop in September–December 2021**. That drop is real and global (not caused by Ethiopia or by
  the misplaced records in views-datafactory#533), but its cause is unknown. Either way, it is not age.
  **That first reading is therefore withdrawn as evidence.**

To separate age from the calendar properly, we re-fitted conflictology with its window ending at **six
different points**, one year apart (months 396, 408, 420, 432, 444 and 456). Now the same calendar month
is forecast from information of different ages, and the two effects can be told apart. We scored every
later month up to 492, **leaving out all of 2021**, so the late-2021 drop could not affect the result.
We then checked the result again after dropping the earliest cutoff, and again after dropping the latest.

What we found, by how old the information is (year 1 = the first year after the window ends):

| which cells (grouped by what the forecast said) | what happens as the information ages | by year 7: all cutoffs / without earliest / without latest |
|---|---|---|
| quiet cells, state-based: conflicts per 1,000 cell-months | **rises steadily** from about 0.4 | +0.89 / +0.88 / +0.94 |
| quiet cells, non-state | rises steadily | +0.62 / +0.62 / +0.69 |
| quiet cells, one-sided | rises steadily | +0.42 / +0.39 / +0.43 |
| hot cells, one-sided: how often forecasts come true | falls | −0.33 / −0.51 / −0.29 |
| hot cells, non-state (year 3) | falls | −0.38 / −0.29 / −0.60 |
| hot cells, state-based (year 3) | about no change | −0.07 / −0.03 / −0.08 |

How to read this:

- **For quiet cells, the effect is clear and solid.** The older the information, the more often a cell
  forecast as "certainly quiet" actually sees conflict. For state-based violence the rate roughly
  triples over seven years. The numbers barely move when a cutoff is dropped (within about 11%), and
  each rests on tens of thousands of cell-months. This is solid enough to estimate a memory length from.
- **For hot cells, the direction is solid but the size is not.** One-sided and non-state forecasts in hot
  cells do fade with age, but dropping a single cutoff changes the size by 75–100%. This tells us the
  violence types should have *different* memories, but not *how long* those memories are.
- **State-based hot cells show no age effect at all**, only the constant ~27% over-confidence. That is a
  different problem (calibration), which this design is not aimed at. Borrowing from neighbours may
  soften it, but we do not promise that.
- **Some errors belong to particular periods**, like the late-2021 drop. No climatology can foresee
  those.

These results were grouped by *what the forecast said*, not by *what happened*. That distinction matters:
grouping by what happened produces misleading patterns even for a perfect forecast (regression to the
mean), a trap that cost us one retraction during the analysis.

The underlying problem, in one sentence: **the forecast does not know that a cell's own past loses
relevance as it ages, and it has nothing to fall back on when that happens.**

### A second purpose

As with ADR-023, these baselines also serve as working sketches for the uncertainty part of deep-learning
models. The question "how fast does a cell's own history stop predicting it, and what should replace it?"
is exactly what a learned model ought to answer better. Matryoshka's answer, a handful of fitted memory
lengths, gives such a model a clear, interpretable reference to beat.

---

## Decision

We add a new baseline, **`MatryoshkaConflictology`**.

### The dolls

At grid-cell level (pgm), from the inside out:

1. **The cell itself.**
2. **Ring 1:** the 8 cells touching it, including diagonally.
3. **Ring 2:** the 16 cells touching ring 1.
4. **The country** the cell belongs to.
5. **The whole map.** This is a backstop: it only gets weight if the data shows it helps. It is also the
   only outer doll for the 76 cells that have no country (mostly small islands).

Neighbours can be computed directly from each cell's ID, because `priogrid_id = (row − 1) × 720 + col`.
We verified this on all 64,818 land cells, so no new data is needed.

At **country level (cm)** there are no rings for now: the dolls are the country itself, then the whole
map. Neighbouring countries, or a region/theatre doll, can be added in a later version.

### What "country" means here

The country doll uses the **FAO GAUL 2024 country code** (`gaul0_code`). This is the platform's agreed
country identity (views-datafactory ADR-025), and it arrives in our data as the `c_id` column. Each cell
belongs to the same country in every month. The same mapping is used by hydranet (`bright_starship`) and
by the members of the chrome_box/crate ensembles (e.g. `ashen_sparrow`).

We deliberately do **not** use the older academic country codes (Gleditsch & Ward / CShapes), which
change over time. In the current land data there are 202 countries. If this model's country level is
ever joined to data keyed by another country code (ISO3, or the old codes), that needs an explicit
translation table first.

### How the forecast is made

To forecast one cell for one future month, the model looks at every past observation in every doll and
gives it a **weight**:

```
weight = (how much this doll counts) × (how much this observation has faded) / (number of cells in this doll that month)
```

or, in symbols, for an observation from doll k in month s, when forecasting month t:

```
w(k, s; t) = ring_weight[k] × exp(−(t − s) / memory[k]) / n_k(s)
```

Each part, in plain words:

- **How much this doll counts** (`ring_weight[k]`): how important this level is overall. The cell itself
  is the reference and counts 1; the other dolls' weights are estimated from data.
- **How much the observation has faded** (`exp(−age / memory)`): an observation that is `memory` months
  old keeps about 37% of its weight; one that is twice as old keeps about 14%; and so on. Each doll has
  its own memory. We expect the cell's memory to be the shortest and the world's the longest, but the
  data decides. If it comes out the other way, that is a finding, not an error.
- **Age is counted up to the month being forecast**, not up to the end of the history. So the further
  ahead we forecast, the older every observation is, and the more the short-memory inner dolls have
  faded relative to the long-memory outer ones. The forecast "opens outward" automatically, without any
  separate setting for lead time.
- **Divided by the number of cells in the doll:** ring 2 has 16 cells, a country may have thousands.
  Without this, large dolls would drown out the cell itself simply by having more cells. Dividing makes
  each doll count as one voice, however many cells it has.

**Then the model draws:** it picks one past observation at random, with probability proportional to its
weight, and uses that observation's value as the forecast draw. It repeats this as many times as the
number of draws the configuration asks for. So:

- every draw is a real observed value (whole numbers of fatalities, nothing invented between them);
- the chance of conflict is simply the weighted share of past observations that had conflict;
- the size of conflict, when it happens, comes from the same draw.

### A small worked example

Take a cell that has been quiet for the last ten years, in a country fighting a war elsewhere.

- **Conflictology today:** its window has no conflict, so it forecasts 0% for every month ahead, forever.
- **Matryoshka, one month ahead:** the cell's own quiet months are recent and carry most of the weight,
  so the forecast is close to zero, but not exactly zero: the neighbours and the country contribute a
  little.
- **Matryoshka, three years ahead:** the cell's own history has faded much more than the country's, so
  the country's war contributes a larger share of the draws. The forecast now gives a small but real
  chance of conflict, which is what our measurements say it should.

### Plain conflictology is a special case

If only the cell itself counts (all other doll weights zero), nothing fades (an infinitely long memory),
and the history is cut at 36 months, Matryoshka is exactly conflictology. That means comparing the two
tests precisely the two new ideas, the outer dolls and the fading, and nothing else.

### Where the history ends

Like conflictology, the model's history ends at the last training month (`train_end`). That keeps the
two directly comparable. Whether forecasts should instead use all months up to each forecast's start
date is a separate question (issue #82), which the control model below answers.

### How much history

The model uses **at least 10 years** of history, and by default **all of it** (from 1990, i.e. about 28
years when the history ends in 2017). The fading takes care of making old data count less; a cut-off would
only be a way to save computing time. The literature supports looking far back:

- Conflict effects linger for at least 20–25 years, and 20 years of uninterrupted peace lowers the risk
  of a new major conflict by 84% for a case like Liberia (Hegre, Nygård & Ræder 2017).
- The long-term trend in von der Maase (2025) has a timescale of about 122 months.
- Hegre et al. (2013) describe conflict history through the *logarithm* of years since the last conflict,
  which fades more slowly than our exponential fade. With a 10-year memory, a 20-year-old conflict keeps
  only about 14% of its weight, which may be too little at country level. The fit will show where the
  country's memory lands; if it keeps running into its upper limit, that points to a slower fade
  (see "For a later version").

### The knobs

**Each doll has its own weight and its own memory**, estimated from data:

| doll | weight | memory |
|---|---|---|
| the cell | fixed at 1 (the reference) | estimated |
| ring 1 | estimated | estimated |
| ring 2 | estimated | estimated |
| the country | estimated | estimated |
| the whole map | estimated | estimated |

That is **9 knobs**, and there is a separate set for each violence type (state-based, non-state,
one-sided), because the evidence shows they age differently. A single shared set is also fitted as a
check, and used instead if it does just as well.

We considered a simpler rule where each ring outward gets a fixed fraction of the weight and a fixed
multiple of the memory. With only two rings that saves just one knob, so we do not impose it. Instead the
fit reports whether the freely estimated values happen to follow such a rule.

### Starting points from the literature

von der Maase (2025), *The Currents of Conflict*, estimates how conflict relates across space and time
using Gaussian processes. Its estimates give sensible starting values and plausible ranges:

- **Space:** a length scale of 0.72° (a Matérn 3/2 kernel over cell centres). Translated to our rings,
  that means ring 1 relates to the cell about 0.58 as strongly as the cell itself, and ring 2 about
  0.24. So each ring outward keeps roughly 40% of the previous one's influence, and a third ring would add
  less than 10%. This independently supports using two rings.
- **Time:** a short-term timescale of about 4 months and a long-term one of about 122 months.

These were estimated on log fatalities in the 60 most conflict-heavy cells, so they describe hot regions,
not the whole map. We use them as starting points and sanity bounds, not as fixed values. If our fit lands
near them, that is independent confirmation; if it lands far away, that difference is itself a finding.

### How the knobs are estimated

The knobs are estimated by **backtesting within the training period**: pretend the history ended at an
earlier month, forecast the following months, compare with what actually happened, and repeat for several
such cutoffs. The knob values that forecast best are kept.

"Forecast best" is judged mainly on whether the **chance of conflict is honest** at each forecast
distance (using a proper score such as the Brier score or log loss, which reward calibration), with the
**size** of conflict checked separately on cells that had conflict. CRPS alone is not used to choose,
because we found it can prefer the wrong model (#122).

The fitted values are reported in full. They are the scientific result, not just settings. When reporting
them:

- **Show how stable each knob is:** refit with each backtest cutoff left out in turn. A memory that is
  mostly determined by hot cells is expected to be unstable, and should be reported as uncertain.
- **Show how many cells each number rests on**, so a reader can see which numbers are solid.
- **For any evaluation that ends in December 2021 (month 504), report results with and without the last
  four months**, because of the unexplained worldwide drop there.

### The first step: a simpler control

Before building Matryoshka, we build a much simpler **control**: plain conflictology whose window ends at
each forecast's start date instead of always at month 456 (issue #126). It is quick to build. It shows how
much of the problem is simply that the window is out of date at the start of the forecast, as opposed to
how far ahead the forecast looks. If the control already fixes most of the problem, that is important to
know before building the full model (issue #127).

---

## Why this design

- **It targets a problem we measured, not one we guessed.** Smoothing addressed a gap that turned out not
  to matter. The age effect in quiet cells is measured with a method that separates age from calendar
  months, groups cells by what the forecast said, and holds up when the cutoffs change.
- **One idea covers three open issues.** It combines forecasts that know how old their information is
  (#82), older data counting less (#111) and borrowing from neighbours (#112) into a single weighting,
  rather than three separate models whose interactions would never be tested.
- **It stays small and readable.** No outside data, 9 knobs per violence type, and each knob has a plain
  meaning: a memory in months, or how much a doll counts.
- **It never invents values.** Unlike model C, every draw is something that actually happened nearby.
  Forecasts stay whole numbers, and C's known bias at small counts (ADR-023) does not apply.

---

## Alternatives we considered

### One setting for time and one for space

A single fading rate for all history, plus a single mix between the cell and its surroundings.
**Rejected:** the mix would then be the same one month ahead as four years ahead, which is exactly the
problem we measured. The dolls need different memories for the forecast to open outward.

### A full hierarchical Bayesian model

Pull each cell towards its region using a statistical model with estimated variances. More principled,
but much more machinery, and it does not by itself make the pull grow with forecast distance.
**Deferred:** Matryoshka's fitted memories are a cheap first test of whether that machinery would pay off.

### Only the moving-window control

**Kept as the control, not as the model.** Within a single forecast it still predicts the same chance one
month ahead as three years ahead.

### More smoothing variants (model D, #110)

**Parked:** the Phase-1 results show that smoothing does not touch the actual problem.

---

## Consequences

### What we gain

- No cell is forecast as *certainly* quiet unless its neighbours, its country and the world give no reason
  to expect conflict.
- Forecasts that depend on how far ahead they look, from a simple baseline. That makes a stronger and more
  honest yardstick for learned models.
- The fitted memory lengths are findings in their own right, for example "one-sided violence stays
  informative for about X months".

### What it costs

- **Estimating knobs is new for this repo.** Every existing baseline only has a window length. A backtest
  loop adds code and running time, and needs its own tests.
- **Forecasting is heavier.** The weights depend on the month being forecast, so the draw has to be
  prepared for every cell *and* every forecast month, not just every cell. Ring averages per month need to
  be computed in advance on the 360 × 720 grid.
- **Misplaced records spread further.** Some records in the data put a whole region's annual death toll
  into one cell and one month. One such record puts about 113,000 deaths from the Tigray war into a single
  cell in December 2021 (UCDP event 463131; views-datafactory#533). With Matryoshka, such a record would
  also reach its neighbours' forecasts. This is being fixed in views-datafactory; this repo does not trim
  or cap values to work around it.

---

## Implementation notes

- The new class goes in `views_baseline/model/models/distributional/`, registered in the model catalog,
  with explicit configuration keys and no hidden defaults (ADR-021).
- Structural choices (number of rings, the outer dolls, how much history, the backtest design) are
  configuration. The fitted knob values are stored on the model and reported, not configured.
- Neighbours are computed from `priogrid_id`. Grid level first; country level uses country → whole map,
  without rings.
- Random draws follow the same seeding discipline as the other distributional models
  (`sample_prediction_grid`), so results are reproducible.
- Memory use during evaluation is handled by the streaming evaluation added in 1.1.1 (#121).

---

## How we will check it

- **Equivalence check:** with the special-case settings (only the cell, no fading, a 36-month history),
  Matryoshka must match conflictology in distribution. This is a test, in the same way model E at α = 0
  was tested against conflictology.
- **Measured the same way as Phase-1:** compared with conflictology and with the moving-window control,
  using the same number of draws. We report:
  - whether the chance of conflict is honest **at each forecast distance** (the check that revealed the
    problem);
  - how well it ranks where conflict happens (this must not get worse);
  - size of conflict, grouped by what the forecast said;
  - CRPS, split into cells with and without conflict.

  All of these are reported with and without the cells affected by views-datafactory#533, until that fix
  is in.
- **Report-only**, as in ADR-023: we publish a table of properties, not a declared winner.

---

## Decisions made while drafting

All settled by Simon on 2026-11-05:

- The country is the outer doll, with the whole map as backstop.
- Two rings of neighbours.
- A separate set of knobs for each violence type.
- No rings at country level for now.
- One memory per doll in this first version.
- A free weight and memory for every doll (no fixed ring rule).
- At least 10 years of history, by default all of it.
- The country is the GAUL 2024 country code (`gaul0_code`), not the older academic codes.
- Fading follows the **age of the information**, per violence type. This is confirmed by the
  multiple-cutoff check: strong for quiet cells, solid in direction only for hot cells, and absent for
  state-based hot cells.

Nothing in the design of this first version is still open. What remains unknown is what the backtest is
there to measure.

### For a later version

- **Two memories for the cell itself:** a short one for bursts that die out within months (about 4
  months in von der Maase 2025) on top of a long one for the cell's long-run level.
- **A region or theatre doll**, and rings at country level (using which countries border which).
- **A slower fade than exponential**, for example based on the logarithm of time (after Hegre et al.
  2013), if the fitted country memory keeps hitting its upper limit.

---

## References

- Issues in this repo: #82 (fixed vs moving window), #109 (Phase-1 results and corrections), #111
  (temporal weighting), #112 (spatial pooling), #122 (scoring), #126 (moving-window control), #127
  (building Matryoshka).
- ADR-021 (no hidden defaults), ADR-023 (smoothed climatology baselines; report-only evaluation).
- views-models `reports/2026-11-04_baseline_phase1/` on branch `docs/baseline-phase1-campaign`
  (commit `a001aba7`): the Phase-1 measurements, including `cond_fcst.py`, `lead_vs_calendar.py` and
  `eth_step.py`. The multiple-cutoff check (`cutoff_check.py`) will be committed here together with the
  backtest.
- views-datafactory#533 (misplaced aggregate records); views-datafactory ADR-025 (country identity: GAUL).
- von der Maase, S. (2025). *The Currents of Conflict: Decomposing Conflict Trends with Gaussian
  Processes.* arXiv:2506.06828. Table 2 (spatial length scale 0.72°, Matérn 3/2) and the temporal
  timescales (about 4.09 and 122.38 months).
- Hegre, H., Karlsen, J., Nygård, H. M., Strand, H. & Urdal, H. (2013). Predicting Armed Conflict,
  2010–2050. *International Studies Quarterly* 57(2): 250–270.
- Hegre, H., Nygård, H. M. & Ræder, R. F. (2017). Evaluating the scope and intensity of the conflict trap.
  *Journal of Peace Research* 54(2): 243–261.
