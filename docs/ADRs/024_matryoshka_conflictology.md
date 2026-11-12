# ADR-024: MatryoshkaConflictology: a climatology that opens outward as its information ages

**Status:** Accepted. Amended 2026-11-05 (data up to each forecast origin; see "Where the history ends") and 2026-11-06 (the two rings replaced by one smooth neighbourhood, and the design made final after the prototype; see "Amendment 2026-11-06: what the prototype settled")
**Date:** 2026-11-05
**Deciders:** Simon
**Consulted:** views-baseline #109 and #82 (Phase-1 results and the age checks); views-models `reports/2026-11-04_baseline_phase1/sparse_support_question.md` (branch `docs/baseline-phase1-campaign`, commit `a001aba7`); views-hydranet `reports/2026-09-30_evaluation_learning_report.md` (evaluation lessons)
**Informed:** UFFAC research team (Gudmond, Celine, Mike, Håvard)

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
- around it, its **neighbourhood**: nearby cells, the closest ones counting most;
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
2. **The neighbourhood** (amended 2026-11-06, #139; it replaces the two square rings of the first
   version): the land cells within 5 cells of it, measured in a straight line between cell centres (up to
   80 cells). Within this doll, a neighbour at distance `d` is picked with probability proportional to
   `exp(−d / ℓ)`, so closer cells count more. The length scale `ℓ` is a fitted knob.
3. **The rest of the country:** the cell's country, minus the cell and its neighbourhood.
4. **The rest of the map:** everything else. This is a backstop: it only gets weight if the data shows
   it helps. For the 76 cells that have no country (mostly small islands), it is the only outer doll.

**Each doll is a shell: it leaves out the dolls inside it.** So no cell is ever counted twice. Without
this, the country would contain the cell and its neighbourhood, and the whole map would contain
everything, so the cell's own history would be counted several times over. A shell can be empty: for
example, a small island may have no land cells within 5 cells, and a small country may lie entirely within
the neighbourhood. An empty shell simply takes no part in the forecast.

Neighbours can be computed directly from each cell's ID, because `priogrid_id = (row − 1) × 720 + col`.
We verified this on all 64,818 land cells, so no new data is needed.

At **country level (cm)** there is no neighbourhood for now: the dolls are the country itself, then the
rest of the map. A neighbouring-countries doll was considered for the grid level and dropped (#138; see the
2026-11-06 amendment).

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

To forecast one cell for one future month, the model picks one past observation at random and uses its
value as a forecast draw. It does this as many times as the configuration asks for. All the modelling is
in **how likely each past observation is to be picked**. That is decided in three steps.

Notation: the history ends at month `c`, the last month observed when the forecast is made (for rolling
origin `k` that is `train_end + k`; see "Where the history ends"). We are forecasting month `t` (so the
forecast looks `t − c` months ahead), and `s` is the month a past observation comes from.

**Step 1: pick a doll.** Doll `k` is picked with probability proportional to

```
ring_weight[k] × exp(−(t − c) / memory[k])
```

- `ring_weight[k]` is how much the doll counts when we forecast the month right after the history ends.
  The cell itself is the reference: its weight is fixed at 1.
- `exp(−(t − c) / memory[k])` is how that say shrinks the further ahead we forecast. A doll with a short
  memory loses its say quickly; one with a long memory keeps it. For example, with a memory of 12 months, a
  doll keeps 92% of its say one month ahead, 37% twelve months ahead and 5% three years ahead.

Because the inner dolls have shorter memories than the outer ones, their share shrinks faster, and the
forecast **opens outward** as we look further ahead.

**Step 2: pick a month within that doll.** Month `s` is picked with probability

```
exp(−(c − s) / memory[k]) / Z_k        where  Z_k = sum over all history months s of exp(−(c − s) / memory[k])
```

Recent months are more likely than old ones, at the doll's own pace. Dividing by `Z_k` makes these
probabilities add up to exactly 1, so this step only decides *which* months the doll's say goes to, never
*how much* say the doll has.

**Step 3: pick a cell within that doll in that month.** For the country and the map this is uniform: if
the shell has `n_k(s)` cells that month, each has probability `1 / n_k(s)`. For the neighbourhood, a
neighbour at distance `d` has probability `exp(−d / ℓ) / Σ exp(−d' / ℓ)`, the sum running over the
cell's land neighbours (amended 2026-11-06). Either way, every doll counts as one voice however many cells
it has; without that, a large country would drown out the cell itself.

**Put together,** the probability of drawing the observation of cell `i` in doll `k` in month `s` is
proportional to

```
w(k, s, i; t) = ring_weight[k] × exp(−(t − c) / memory[k]) × exp(−(c − s) / memory[k]) / Z_k / n_k(s)
```

(for the neighbourhood, `1 / n_k(s)` is replaced by the distance weight of step 3)

(the forecast divides by the sum of `w` over everything, so the probabilities add up to 1).

**Why `Z_k` matters.** Without it, a doll's total say would grow with its memory: summed over 28 years of
history, a doll with a 120-month memory would carry about 25 times the say of a doll with a 4-month
memory at the same weight (113 against 4.5). "How much a doll counts" and "how quickly it forgets" would
then be tangled together, and the fit could not tell them apart. With `Z_k`, each doll's total say is
exactly `ring_weight[k] × exp(−(t − c) / memory[k])`, whatever its memory and however long the history.

**What a forecast draw then is:** a real observed value, so a whole number of fatalities, with nothing
invented between observed values. The chance of conflict is simply the share of the probability that
falls on observations with conflict. The size of conflict, when there is some, comes from the same draw.

**Checked numerically.** `reports/matryoshka/matryoshka_math_check.py` verifies on a small synthetic
grid that:
- each doll's total say is exactly `ring_weight × exp(−(t − c)/memory)`, independent of memory and
  history length;
- the cell's share falls and the outer dolls' share rises as the forecast looks further ahead;
- the shells never overlap and together cover the map;
- the three-step draw gives exactly the probabilities of the combined formula;
- the special case below is exactly conflictology.

### A small worked example

Take a cell that has been quiet for the last ten years, in a country fighting a war elsewhere.

- **Conflictology today:** its window has no conflict, so it forecasts 0% for every month ahead, forever.
- **Matryoshka, one month ahead:** the cell's own quiet months carry most of the say, so the forecast is
  close to zero, but not exactly zero: the neighbours and the rest of the country contribute a little.
- **Matryoshka, three years ahead:** the cell's own say has shrunk much more than the country's, so the
  country's war contributes a larger share of the draws. The forecast now gives a small but real chance of
  conflict, which is what our measurements say it should.

### Plain conflictology is a special case

Give every doll except the cell a weight of zero, let the cell's memory be infinite (nothing fades), and
use only the last 36 months of history. Then step 1 always picks the cell; in step 2 every month has
`exp(0) = 1`, so `Z = 36` and each month has probability `1/36`; step 3 has only one cell. That is a
uniform draw over the cell's last 36 values, which is exactly conflictology with a moving window
(`RollingConflictology`, whose window ends at each forecast origin like Matryoshka's history). Fix the
history end at `train_end` as well, and it is plain `ConflictologyModel`. Comparing Matryoshka with
`RollingConflictology` therefore tests precisely the two new ideas, the outer dolls and the fading, and
nothing else; comparing with `ConflictologyModel` adds the effect of using the latest data.

### Where the history ends

**Decided 2026-11-05 (amendment): the knobs are fitted once; the data is the latest at each forecast
origin.**

The production regime trains models once a year, up to the first out-of-sample forecasting window, and
uses the saved artifacts for inference for the rest of the year. Conflictology mirrors that by fixing its
window at the last training month for every forecast origin. But what a frozen artifact freezes differs
between learned models and climatologies:

- **A learned model freezes its parameters, not its inputs.** views-stepshifter is trained up to
  `train_end`, but when forecasting from origin `k` it still reads the observed data up to `train_end + k`.
- **For a climatology, the data is the model.** Freezing conflictology's window also freezes its inputs,
  which denies it information the learned models get. And "retraining" a baseline costs seconds.

So Matryoshka follows the learned models' regime:

- **What is frozen, like a trained model:** the knobs (the weights and memories). They are fitted once,
  by backtest on data up to `train_end`, and stored in the artifact.
- **What is fresh at each origin, like a learned model's inputs:** the observed data. For rolling origin
  `k`, the history ends at `c = train_end + k`, the month just before that origin's first forecast month.
  No forecast month is ever part of its own history. In a forecasting run there is one origin, and `c` is
  the last observed month.

Age is still measured to the month being forecast, so the forecast still opens outward the further ahead
it looks within one origin. The reasoning is recorded on #82.

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

| doll | weight | memory | other |
|---|---|---|---|
| the cell | fixed at 1 (the reference) | estimated | |
| the neighbourhood | estimated | estimated | length scale `ℓ`, estimated |
| the rest of the country | estimated | estimated | |
| the rest of the map | estimated | estimated | |

That is **8 knobs** (amended 2026-11-06; the first version had 9, with two rings), and there is a separate
set for each violence type (state-based, non-state, one-sided), because the evidence shows they age
differently. The fitted memories confirm this (see the amendment below). A single shared set, planned
here as a check, was not fitted in the prototype; the separate sets are kept.

The fit searches within fixed limits: weights 1e-6 to 1,000, memories 1 to 600 months, `ℓ` 0.25 to 10
cells. **The 600-month memory limit is a deliberate ceiling** (#140): a memory at 600 means "the whole
history, roughly equally", which is longer than the 28 years of history can tell apart.

### Starting points from the literature

von der Maase (2025), *The Currents of Conflict*, estimates how conflict relates across space and time
using Gaussian processes. Its estimates give sensible starting values and plausible ranges:

- **Space:** a length scale of 0.72° (a Matérn 3/2 kernel over cell centres). The neighbourhood's `ℓ`
  is directly comparable. The fitted values are 0.40° (state-based), 0.68° (non-state) and 0.49°
  (one-sided), so non-state matches closely, and the other two are somewhat shorter.
- **Time:** a short-term timescale of about 4 months and a long-term one of about 122 months.

These were estimated on log fatalities in the 60 most conflict-heavy cells, so they describe hot regions,
not the whole map. We use them as starting points and sanity bounds, not as fixed values. If our fit lands
near them, that is independent confirmation; if it lands far away, that difference is itself a finding.

### How the knobs are estimated

The knobs are estimated by **backtesting within the training period**: pretend the history ended at an
earlier month, forecast the following months, compare with what actually happened, and repeat for several
such cutoffs. The knob values with the best score are kept.

**The score used for fitting: a log score over size classes.** Each cell-month's outcome is put into one
of seven classes:

> 0 · 1 · 2–5 · 6–25 · 26–100 · 101–1,000 · more than 1,000 fatalities

For each cell-month, the model's forecast gives a probability for each class. The score is the logarithm
of the probability the model gave to the class that actually happened, averaged over all cell-months,
forecast distances and backtest cutoffs. Higher is better.

Why this score, and not CRPS or the Brier score (lessons from the hydranet evaluation report,
`views-hydranet/reports/2026-09-30_evaluation_learning_report.md`, §0 and §2–§3):

- **It is one proper score over every possible outcome.** A proper score is one that a forecaster cannot
  improve by saying something other than what they believe. The report's rule is that *one* such score,
  over all outcomes, does the choosing, and everything else is a diagnostic. Scores computed only on cells
  that had conflict must never choose: a forecaster who always predicts conflict would win them (the
  "forecaster's dilemma", Lerch et al. 2017).
- **It does not drown in zeros.** About 99.7% of cell-months have no conflict. Scores that average a
  *distance* over all cells, such as CRPS and the Brier score, then mostly measure the zeros: in hydranet,
  four models sat within 0.0007 CRPS of each other while their ability to locate conflict differed 13-fold,
  and CRPS picked the worst of our ten baselines in Phase-1 (#122). A log score instead charges heavily for
  having said "almost impossible" about something that happened.
- **It punishes exactly the failure we want to fix.** A cell forecast as *certainly* quiet (probability
  0) that then sees conflict gets a very bad score. That forces the outer dolls to earn their weight.
- **Classes keep it finite and robust.** Scoring exact counts would give an infinitely bad score whenever
  the model had never seen that exact count. Classes avoid that, and one misplaced record, like the 113,000
  Tigray deaths in a single cell, costs one bad class instead of dominating the score (it made up 38% of
  every model's CRPS in Phase-1). If a class never occurred anywhere in the history of a violence type, it
  is merged with its neighbour for that type, so no class can have probability zero for every cell. (In
  the training data every class occurs for every violence type, so no merge has been needed. The rarest
  is more than 1,000 fatalities for non-state violence, with 10 cell-months.)

**The fitting score is never used to judge the model.** The hydranet report's rule (§8, row 16): *no
quantity in the training objective may serve as a promotion criterion*. The log score is reported, but the
model is judged on the separate checks below, on months the fit never saw.

The fitted values are reported in full. They are the scientific result, not just settings. When reporting
them:

- **Show how stable each knob is:** refit with each backtest cutoff left out in turn. A memory that is
  mostly determined by hot cells is expected to be unstable, and should be reported as uncertain.
- **Show how many cells each number rests on**, so a reader can see which numbers are solid.
- **For any evaluation that ends in December 2021 (month 504), report results with and without the last
  four months**, because of the unexplained worldwide drop there.

### The comparison model: conflictology with a moving window

`RollingConflictology` (views-baseline 1.2.0, issue #126) is plain conflictology whose window ends at each
forecast origin, the same information rule as above. It is the right comparison for Matryoshka: both see
the same data at each origin, so any difference comes from the dolls and the fading, not from fresher
data. It also answers #82's own question, how much plain conflictology gains just from using the latest
data. It is not a prerequisite for building Matryoshka (#127); the two can be run side by side.

### Single levels on their own (amendment 2026-11-12, #151)

A reviewer will also ask whether one level on its own (just the country, just admin 1, just the map)
does as well. `PooledConflictology` answers that: conflictology whose window is pooled over every cell
in the cell's area (`pool_by`). Whether *adding* a level to the others helps is the other question; that
needs Matryoshka with dolls switched off (#150).

---

## Why this design

- **It targets a problem we measured, not one we guessed.** Smoothing addressed a gap that turned out not
  to matter. The age effect in quiet cells is measured with a method that separates age from calendar
  months, groups cells by what the forecast said, and holds up when the cutoffs change.
- **One idea covers three open issues.** It combines forecasts that know how old their information is
  (#82), older data counting less (#111) and borrowing from neighbours (#112) into a single weighting,
  rather than three separate models whose interactions would never be tested.
- **It stays small and readable.** No outside data, 8 knobs per violence type, and each knob has a plain
  meaning: a memory in months, how much a doll counts, or how far the neighbourhood reaches.
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
  prepared for every cell *and* every forecast month, not just every cell. Neighbourhood sums per month
  need to be computed in advance on the 360 × 720 grid.
- **Misplaced records spread further.** Some records in the data put a whole region's annual death toll
  into one cell and one month. One such record puts about 113,000 deaths from the Tigray war into a single
  cell in December 2021 (UCDP event 463131; views-datafactory#533). With Matryoshka, such a record would
  also reach its neighbours' forecasts. This is being fixed in views-datafactory; this repo does not trim
  or cap values to work around it.

---

## Implementation notes

- The new class goes in `views_baseline/model/models/distributional/`, registered in the model catalog,
  with explicit configuration keys and no hidden defaults (ADR-021).
- The backtest design is configuration. The history is not: it is all the data passed, from its first
  month (recorded at fit as `history_start`, and checked at predict). How much history to use is
  decided by the input pull, not by a setting. The dolls, the 5-cell neighbourhood
  radius, the size classes and the knob limits are fixed by this ADR, not configured. The fitted knob
  values are stored on the model and reported, not configured.
- Neighbours are computed from `priogrid_id`. Each doll is a shell that excludes the dolls inside it.
  Grid level first; country level uses country → rest of the map, without a neighbourhood.
- Random draws follow the same seeding discipline as the other distributional models
  (`sample_prediction_grid`), so results are reproducible.
- Memory use during evaluation is handled by the streaming evaluation added in 1.1.1 (#121).

---

## How we will check it

**Before anything else: the equivalence check.** With the special-case settings (only the cell, no
fading, a 36-month history), Matryoshka must match conflictology in distribution. This is a test, in the
same way model E at α = 0 was tested against conflictology.

**Then the evaluation**, on the calibration test period (months the fit never saw), using hydranet's
frozen scoring code (`lodestar_score.py`) rather than our own re-implementations. Matryoshka is always
compared with **the moving-window control (`RollingConflictology`, the like-for-like comparison) and plain
conflictology**, with the **same number of draws** and
the **same random seeds** (a paired comparison: the report shows pairing makes differences about six
times easier to detect).

What we report, each answering one question, side by side and never folded into a single number:

| question | what we report |
|---|---|
| Does the chance of conflict come true as often as it says, **at each forecast distance**? | Calibration of P(conflict), grouped by what the forecast said, per forecast-distance band. This is the check that revealed the problem, and the main thing Matryoshka must improve. |
| Does it still find **where** conflict happens? | Average precision (AP). It must not get worse. First we check that AP actually separates the models being compared; the report's rule is to check a metric's range before ranking on it. |
| Does it predict conflict **as often as it happens**? | How often it fires compared with reality. |
| When there is conflict, **how big**? | The typical (median) and the average size, reported separately: a few huge cells can flatter the average. These are grouped by what the forecast said where possible; anything grouped by what happened is a diagnostic only. |
| What does CRPS say, and **is any CRPS gain real**? | CRPS split into cells with and without conflict, plus hydranet's check of how much of a gain comes from correctly predicted zeros (a gain that is mostly zeros is an artefact). |

All of these are reported **with and without** the cells affected by views-datafactory#533 until that fix
is in, and with and without months 501–504.

**Report-only**, as in ADR-023: we publish a table of properties, not a declared winner.

---

## Decisions made while drafting

All settled by Simon on 2026-11-05:

- The country is the outer doll, with the whole map as backstop. Each doll is a **shell** that leaves out
  the dolls inside it, so nothing is counted twice.
- Each doll's fade is **normalised** (`Z_k`), so a doll's weight and its memory mean separate things.
  Checked numerically (`reports/matryoshka/matryoshka_math_check.py`).
- The knobs are fitted on a **log score over seven size classes**, and that score is never used to judge
  the model.
- Two rings of neighbours. *(Replaced on 2026-11-06 by one smooth neighbourhood; see the amendment.)*
- A separate set of knobs for each violence type.
- No rings (now: no neighbourhood) at country level for now.
- One memory per doll in this first version.
- A free weight and memory for every doll (no fixed ring rule).
- At least 10 years of history, by default all of it.
- The country is the GAUL 2024 country code (`gaul0_code`), not the older academic codes.
- **Knobs frozen at `train_end`, data up to each forecast origin** (amendment, see "Where the history
  ends"): the climatology analogue of a yearly-trained model that reads the latest inputs.
- Fading follows the **age of the information**, per violence type. This is confirmed by the
  multiple-cutoff check: strong for quiet cells, solid in direction only for hot cells, and absent for
  state-based hot cells.

Nothing in the design of this first version is still open. What remains unknown is what the backtest is
there to measure.

### For a later version

- **Two memories for the cell itself:** a short one for bursts that die out within months (about 4
  months in von der Maase 2025) on top of a long one for the cell's long-run level.
- **A slower fade than exponential**, for example based on the logarithm of time (after Hegre et al.
  2013). The one-sided rest-of-map memory sits at the 600-month ceiling in every fit (#140); that is the
  case this would address. Accepted as it is for now.

The region doll and the neighbouring-countries doll that were listed here have been tested or dropped; see
the amendment below.

---

## Amendment 2026-11-06: what the prototype settled

The design above was built as a prototype (`reports/matryoshka/`, branch `feat/127-matryoshka-prototype`)
and fitted on real data before any package code. Every result and decision is in the running log
`reports/matryoshka/LOG.md`, and on the issues of epic #135. Simon declared the design final on
2026-11-06 (LOG.md, commit `187ce44`). **The package build (#127) follows this amendment where it differs
from the text above.**

**The pictures** are in the notebook `notebooks/matryoshka_conflictology.ipynb`. It explains what the model
does and how, with the package code, and draws every plot when it is run: the dolls of a cell, the
neighbourhood kernel, how the mix of dolls changes with distance, the fit, the sampler against the exact
probabilities, the out-of-sample maps against conflictology, and the size of a possible outbreak in quiet
places. No graphics are committed (#144). The prototype's own map scripts are in
`reports/matryoshka/prototype/` (`map_forecasts.py`, `map_conditional_mean.py`, `compare_maps.py`).

### What changed, and why

- **The two square rings became one smooth neighbourhood (#139, kept).**
  - The rings left visible square halos around active cells in the forecast maps.
  - The fitted ring weights, per cell, already fell off smoothly with distance.
  - A neighbourhood with `exp(−d / ℓ)` within 5 cells scores better than the rings for state-based and
    one-sided violence (+5.3e-5 and +3.1e-5 in mean log score) and ties for non-state (+1.5e-6). It also
    has one knob fewer.
- **A province (admin-1) doll was tested and dropped (#136).** On top of the rings and again on top of the
  smooth neighbourhood, every score difference was within the noise (about 6e-6). On the smooth
  neighbourhood, state-based switched the province off entirely (weight 0). It would also have needed a new data column in views-models.
- **Separating "whether violence happens" from "how big it is" was dropped (#137).**
  - In quiet places the outer dolls give a large size (about 25 deaths given any). This is because the
    world's violence is dominated by ongoing wars, while recent outbreaks in quiet places are smaller
    (about 10; `prototype/describe_onset_sizes.py`).
  - Choosing sizes by whether a place was quiet before would add a new kind of rule. The model's idea is
    only *which past observations to draw from*. The large size is the honest answer of the world pool.
- **A neighbouring-countries doll was dropped (#138).** The neighbourhood already crosses borders, and the
  province test showed that another administrative doll adds nothing.
- **Distances are not corrected for latitude (#141).** The fitted `ℓ` is under one cell, so the correction
  would change practically nothing.
- **Smoothing the drawn values (a kernel density, Silverman's bandwidth) was considered and rejected.**
  - It widens the pool but does not move its centre.
  - Matryoshka has no gap in what it can draw: every cell reaches every observed value through the outer
    dolls.
- **Loosely determined knobs are accepted (#140):**
  - the one-sided map memory at the ceiling;
  - the non-state country weight and memory, which trade off against each other.

  Their ranges are reported, not hidden.

### How the knobs are found

- **Search:** L-BFGS-B over the logarithms of the knobs, with the exact slope of the score (not slopes
  estimated by nudging each knob).
- **Stopping thresholds:** the defaults stopped too early, so they are strict: `ftol = 1e-13`,
  `gtol = 1e-9`, at most 200 steps.
- **Exact class probabilities:** the scoring works these out exactly from counts per doll and month; no
  sampling.
- **Storage:** only the cell-months with deaths are stored (sparse), since nearly all are zero.
- **Cost:** about 1 minute and about 4 GiB per violence type at pgm `land`, with 4 backtest cutoffs (384,
  396, 408, 420) and 36 months scored after each.
- **Checks in the prototype**, each tested by deliberately breaking the code:
  - the shells, sizes and class shares against a brute-force construction on a small grid;
  - the exact probabilities against 200,000 simulated three-step draws;
  - the exact slope against nudged differences (relative tolerance 1e-6);
  - the expected value against direct sums.

### The fitted knobs (training data up to month 456)

| | state-based | non-state | one-sided |
|---|---|---|---|
| weight: neighbourhood / country / map | 0.95 / 0.18 / 0.046 | 2.80 / 0.38 / 0.21 | 1.72 / 0.47 / 0.028 |
| memory in months: cell / neighbourhood / country / map | 13.5 / 13.4 / 11.6 / 24.8 | 15.6 / 11.6 / 80.1 / 29.1 | 23.1 / 13.8 / 22.1 / 600 (ceiling) |
| `ℓ` (cells; degrees) | 0.80; 0.40° | 1.35; 0.68° | 0.98; 0.49° |
| mean log score (conflictology) | −0.0151 (−0.0407) | −0.0055 (−0.0164) | −0.0067 (−0.0196) |
| observed outcomes given a near-zero chance (conflictology) | 0 (11,456) | 0 (5,139) | 0 (5,861) |

**Stability.** Refitting with each backtest cutoff left out in turn:

| | state-based | non-state | one-sided |
|---|---|---|---|
| cell memory (months) | 12–15 | 15–17 | 17–28 |
| `ℓ` (cells) | 0.77–0.83 | 1.23–1.65 | 0.95–0.99 |
| country memory (months) | 10–14 | 11–196 (traded against its weight, 0.30–2.03) | 18–25 |
| map memory (months) | 20–36 | 25–31 | 600 in every refit |

Everything else is in `reports/matryoshka/PROTOTYPE_v5_kernel.md` and `results/v5_kernel/`.

**Reading:**
- The cell and its neighbourhood forget at about the same pace, 12–24 months, for every type.
- Non-state violence leans hardest on its neighbourhood and reaches furthest.
- The world backstop gets little weight, and for one-sided violence it remembers the whole history.

The log scores are fitting scores and do not judge the model. Judging is the evaluation below
("How we will check it").

---

## Amendment 2026-11-12: choosing which dolls to use (#153)

**Why.** A paper on Matryoshka has to show whether each level earns its place: does the country doll
add anything on top of the cell and its neighbourhood, does the map doll? The fair test fits smaller
versions with the same code, the same knob search and the same backtest. Single levels on their own are
a different question, answered by `PooledConflictology` (#151).

**Decision.** `MatryoshkaConflictology` takes a required setting, `dolls`: the names of the dolls to
use, in the order `cell`, `near`, `country`, `map`. All four is the model above.

- **The cell is always used.** Its weight is the reference the other weights are measured against
  (#150, comment of 2026-11-12).
- **An off doll's cells move outward** to the next doll that is on (Simon, 2026-11-12; epic #153):
  - without `near`, the country doll holds the cell's same-country neighbours, and the map the others;
  - without `country`, the map doll holds the cell's country;
  - without `map`, its cells are in no doll, since the map is outermost.

  The alternative, dropping an off doll's cells altogether, was rejected. "No country" would then also
  hide the country's cells from the map doll, which mixes two effects.
- **An off doll has an empty shell**, so the exact probability gives it no weight and the draw never
  picks it. Its knobs are **left out of the search**, not pinned at a small value. They are stored as
  weight 0 and memory NaN, and the length scale as NaN when `near` is off.
- **No default** (ADR-021): a config without `dolls` is rejected.

**What it guarantees.** For every subset that includes the cell, the tests show:
- the shells are disjoint and have the counted sizes;
- the draw samples exactly the distribution the search scores;
- the slope of the free knobs is exact.

With all four dolls, the output is byte-identical to 1.3.0 (`GOLDEN_MATRYOSHKA`), and the fit time is
unchanged (`reports/matryoshka/LOG.md`, 2026-11-12).

**Sources:** issues #150, #151, #153–#157.

## References

- Issues in this repo: #82 (fixed vs moving window), #109 (Phase-1 results and corrections), #111
  (temporal weighting), #112 (spatial pooling), #122 (scoring), #126 (moving-window control), #127
  (building Matryoshka); epic #135 and its steps #136–#142 (the prototype decisions).
- `notebooks/matryoshka_conflictology.ipynb`: the explainer, with every plot drawn when it is run (#144).
- `reports/matryoshka/LOG.md` (the running log of the prototype) and `reports/matryoshka/PROTOTYPE_v5_kernel.md`
  (the final fit), branch `feat/127-matryoshka-prototype`.
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
- Lerch, S., Thorarinsdottir, T. L., Ravazzolo, F. & Gneiting, T. (2017). Forecaster's dilemma: Extreme
  events and forecast evaluation. *Statistical Science* 32(1): 106–127.
- views-hydranet `reports/2026-09-30_evaluation_learning_report.md` (the living evaluation report; read
  2026-11-05, last changed 2026-11-04), §0, §2.2–2.4, §3.2, §8 (row 16) and §8.3.
- Hegre, H., Nygård, H. M. & Ræder, R. F. (2017). Evaluating the scope and intensity of the conflict trap.
  *Journal of Peace Research* 54(2): 243–261.
