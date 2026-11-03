# ADR-023: Smoothed Climatology Baselines — Estimator Design and Report-Only Evaluation

**Status:** Proposed
**Date:** 2026-11-03
**Deciders:** Simon, VIEWS platform team
**Consulted:** views-hydranet session (evaluation lessons, α-sweep conditions); paper library (11 papers, see References)
**Informed:** views-models maintainers

---

## Context

`ConflictologyModel` (call it **A**) forecasts each cell by resampling its last `window_months`
observed values. ADR-022 tested *parametric* replacements for that empirical law; epic #104 tests
*non-parametric smoothing* of it instead:

- **B**, an interpolated empirical quantile function;
- **C**, a zero-preserving kernel density estimate (KDE);
- **E**, a mixture of A and C with weight α.

These baselines have **two purposes**:

1. **As baselines.** Do they forecast better than A?
2. **As uncertainty prototypes.** Which part of the fatality distribution does each one capture
   (zeros, small counts, large counts, spread), and what neural-network output head would carry
   that idea?

**A decision is needed now for two reasons.**

**First, the papers settle several design choices that would otherwise be made ad hoc in code.**
Read 2026-11-02; findings recorded on #107 and #108.

**Second, scoring is an unsolved problem on this data, and the earlier plan was naive.** The
living document `views-hydranet/reports/2026-09-30_evaluation_learning_report.md` shows:

- overall CRPS cannot rank models on a ~99.7%-zero field;
- outcome-conditioned scores cannot select models;
- most metric columns read only the mean field.

A, B, C and E share the same pool, the same cells and (nearly) the same mean. **They differ only
in shape and spread**, which is the region the standard battery sees worst.

Epic #33's verdicts (ADR-022; `reports/dwarf_forecast_evaluation/FINDINGS.md`) rest on the
platform's CRPS / MCR profile. Under the lessons above, those rankings are weaker evidence than
they read as. That is noted here, not re-litigated.

---

## Decision

### 1. Three models — ontology cat. 2 (Distributional), stability Evolving

| ID | Class | One-line law |
|---|---|---|
| B | `InterpolatedQuantileConflictology` | zero mass = empirical zero rate; positives drawn from the linearly interpolated quantile function of the window's positive values |
| C | `KDEConflictology` | zero mass = empirical zero rate; positives drawn from a Gaussian KDE of the window's positive values, reflected at 0.5 and rounded to integers |
| E | `EmpiricalKDEMixtureConflictology` | per draw: with probability 1−α an A draw, with probability α a C draw |

All three:

- are `distributional = True` and return `dict[str, PredictionFrame]` `(N, n_samples)` through the
  single `to_prediction_frames` seam (ADR-020);
- take their pool from the **shared `window_pool()`** (`model/frames/pooling.py`), so pools are
  byte-identical to A's (as in ADR-022);
- follow the ADR-011 RNG contract: fresh `default_rng(seed)` per `predict()`, iterating
  entity → time → target.

### 2. Binding rules shared by all three

- **Zero mass is separate and equals A's.** `P(Y=0)` is the window's empirical zero rate in every
  estimator. Smoothing acts on the positive values only.
  - *Why:* this was the decisive factor in #33 (ZINB) and in Huang 2023 (Bernoulli-Gamma).
  - *Consequence:* **B, C and E cannot differ from A on any zero / occurrence metric** except by
    Monte-Carlo noise. The evaluation uses this as a built-in control (§5).
- **Integer, non-negative output.** Every draw is a count in {0, 1, 2, …}; a positive draw is ≥ 1.
  - *Why:* occurrence metrics count draws > 0, and a fractional value in (0, 1) would count as
    "firing". Without this, B and C would change the activation rate by interpolation or kernel
    leakage alone.
- **Degenerate windows fall back to the empirical law, and say so.**
  - empty window: as A;
  - all-zero window: point mass at 0;
  - fewer than `MIN_POSITIVES_FOR_SMOOTHING` positive values, or zero spread among them: resample
    the positive values as A does.
  
  `MIN_POSITIVES_FOR_SMOOTHING = 2` is a single named constant. Fallbacks are counted and reported
  once per `predict()` as an aggregate WARN (ADR-008), never once per cell.
- **No transform.** Phase 1 smooths on the count scale only. A `log1p` variant is **out of scope**:
  #33 measured log1p as catastrophic (the `expm1` tail blow-up).

### 3. Per-model rules

**B — interpolated quantile.**

- **Positions.** Sort the positive values `p(1) ≤ … ≤ p(m)`. Use plotting positions
  `u_i = (i−1)/(m−1)` and linear interpolation `Q₊(u)`.
- **Drawing.** A positive draw is `round(Q₊(U))` with `U ~ Uniform(0,1)`.
- **No extrapolation.** Draws stay inside `[min, max]` of the observed positives, the honest
  default. Huang 2023 found that ECDF interpolation *reduces* resampling's gaps but cannot exceed
  the observed range. That limitation is kept and measured, not patched.
- **No new parameter.** The plotting position is the definition of B, not a tunable.

**C — zero-preserving KDE.**

- **Sampling.** Pick a positive value `p` uniformly from the window. Draw `x = p + h·Z`, with
  `Z ~ N(0,1)`. If `x < 0.5`, reflect it to `1 − x`. Then round to the nearest integer.
- **What that sampling equals.**
  - Rounding is exactly the same as integrating the reflected density over the bins
    `[k−½, k+½)`, `k ≥ 1`.
  - Reflection at 0.5 (the lower edge of count 1's bin) puts no mass on 0, so the zero mass stays
    exactly A's.
  - No grid is needed.
- **Residual bias, which the docstring must state.** Reflection removes the O(1) boundary loss but
  leaves O(h) bias wherever the density slope at the boundary is non-zero (Marron & Ruppert 1994).
  For our data it is steep there.
- **Bandwidth `bandwidth`.** A **required, audited genome key** (ADR-021). Its value is one of:
  - a positive number `h` in count units, global across cells;
  - `"sj"` (Sheather–Jones plug-in, per cell);
  - `"silverman"`, which is allowed but documented as oversmoothing (Jones, Marron & Sheather
    1996). Combined with Huang 2023's "overly wide spreads", it is the expected worst choice.
  
  There is no silent default.
- **Selector failure.** Any per-cell selector that fails or returns a non-positive value
  (`"sj"` / `"silverman"` on tiny or tied windows) falls back to the empirical rule above,
  counted in the same WARN.
- **Discrete count kernels are rejected** (Senga Kiessé & Durrieu 2024):
  - at k = 1 the discrete kernel is exactly as good as the raw frequencies, i.e. no better than A;
  - it smooths only ±k counts, so it cannot smooth the tail;
  - its support is all integers.

**E — empirical–KDE mixture.**

- **What it is.** `F_α = (1−α)·F_A + α·F_C`: a mixture of **distributions**, not an average of
  draws.
- **Genome keys.** `alpha ∈ [0, 1]` and `bandwidth` (as for C) are required, audited keys.
- **Common random numbers.**
  - For each draw, E consumes, in fixed order, one `U ~ Uniform(0,1)`, one A-draw and one C-draw.
    It keeps the C-draw iff `U < α`.
  - The stream is the same for every α, so **an α-sweep at a fixed seed is a coupled
    (paired) design**: raising α only switches draws from A to C.
  - *Why:* it costs nothing, it makes α-to-α differences far less noisy, and the hydranet α-sweep
    needs exactly this (#108, #109).
- **Invariants.** At `α = 0`, E equals A **in distribution**; at `α = 1`, it equals C in
  distribution. Both are tested.
  - Byte-identity with A is **not** required: E consumes extra random numbers.
- **Known property.** Mixing distinct forecasts can only *widen* spread, never sharpen it
  (Gneiting & Ranjan 2013, Thm 3.1). So α is effectively a spread dial between A and C.
- **Out of scope:** the beta-transformed pool (BLP), a known fix if E proves too wide.

### 4. Evaluation is report-only — no winner is declared

Decided by Simon, 2026-11-02 (#105, option b).

- The Phase-1 experiment (#109) **measures where each estimator differs from A** (zeros, small
  counts, large counts, spread) under the constraints of the evaluation plan
  `reports/smoothed_climatology/EVALUATION_PLAN.md`.
- No metric is designated as a ranking criterion. No baseline is "promoted".
- When the hydranet minimal-battery work (its report §9) yields a validated metric set, the same
  outputs are re-scored with it.

### 5. Scope

**In:**
- the three classes;
- catalog and genome wiring;
- golden and invariant tests;
- the evaluation plan;
- the Phase-1 experiment in views-models (#109);
- the hydranet α-sweep under the conditions recorded on #109.

**Out:**
- D (adaptive KDE, #110);
- F/G (temporal weighting, #111);
- H (spatial pooling, #112);
- log1p variants, discrete kernels, BLP;
- any change to the behaviour of `ConflictologyModel`, `MixtureBaseline` or the ADR-022 models.

---

## Rationale

- **Zero mass equal to A's** isolates the experiment. Every difference between A and B/C/E is then
  attributable to the positive part, which is the question. It also hands the evaluation a free
  control.
- **Integer output** keeps occurrence well defined. It is what the observed data are, and it lets
  the same scorer treat A, B, C and E identically.
- **Reflect-then-round C** is the simplest correct construction: three lines of sampling, no grid,
  exact zero mass. Its known bias is stated, not hidden.
- **Report-only evaluation** follows from the evidence. No metric set is known to rank correctly on
  this field, and the second purpose (prototyping output heads) needs attribution, not a
  leaderboard.

---

## Considered Alternatives

### B with zeros included in the interpolation
- **Pros:** one formula, no hurdle.
- **Cons:** interpolating between the last zero and the first positive creates fractional draws in
  (0, p(1)), silently raising the activation rate.
- **Rejected:** it would change the zero mass, the one thing every estimator must hold fixed.

### B with a smooth tail extension (e.g. Bernstein, Cheng 1995)
- **Rejected for Phase 1:** keep B the plain interpolation Huang 2023 tested. Revisit if #109 shows
  the bound on observed values is B's main failure.

### C via a discrete associated kernel (Senga Kiessé & Durrieu 2024)
- **Rejected:** no better than A at k = 1, blind to the tail, integer support leaks below 1.
- **Revisit as:** a small-count-only variant, if the bulk proves to be where C helps.

### C via grid evaluation and integration of the density
- **Equivalent in law** to reflect-then-round sampling, but adds a grid, its resolution, and a
  normaliser.
- **Rejected** on simplicity.

### A pre-declared ranking metric set (#105 option a)
- **Rejected:** no metric set is known to rank correctly on this data, so a winner would be an
  artifact of the choice.

### Wait for hydranet's minimal battery (#105 option c)
- **Not chosen as the gate:** it doesn't exist yet.
- **Kept as the follow-up:** re-score the same outputs once it does.

---

## Consequences

### Positive
- Three thin estimators on one shared pool and one seam. Differences between them are attributable
  by construction.
- A coupled α-sweep that hydranet can use as a population ordered on dispersion. Hydranet's
  synthetic forecasts have no dispersion axis.

### Negative / accepted
- **No "winner"**, so no baseline is promoted from this epic.
- **C's fixed numeric bandwidth is global across cells.** It is wide for small counts and narrow
  for large ones. That is the known weakness of fixed-bandwidth KDE on heavy tails, and the reason
  D (#110) exists.
- **B cannot exceed the observed range,** by choice.
- **The epic #33 verdicts are weaker evidence than they read as** (see Context). Re-scoring them is
  possible later and out of scope here.

---

## Implementation Notes

- **Order:** this ADR and the evaluation plan (#105) → B (#106) ‖ C (#107) → E (#108) → the
  experiment (#109).
- **Per-class bookkeeping.** Each class follows ADR-012's model-addition protocol in its own PR:
  catalog, `ALGORITHM_GENOMES`, the stability map in ADR-004, the ontology in ADR-001, a CIC, and
  tests. The ADR-001/004 updates land with the classes, not here.
- **Single-sourced constant** in the module that hosts the shared fallback: the name
  `MIN_POSITIVES_FOR_SMOOTHING`.
- **No new runtime dependency** unless `"sj"` needs one. If it does, that dependency is recorded
  in the C PR, not assumed here.

---

## Validation & Monitoring

**Every estimator:**
- the pool is byte-identical to A's (`window_pool` test);
- the zero mass equals A's within Monte-Carlo error on a synthetic window;
- draws are integers ≥ 0, and positive draws are ≥ 1;
- each fallback branch fires and is counted;
- a golden test pins the draws for a fixed seed.

**B:**
- every draw lies within the observed positive range.

**C:**
- predicted `P(Y=1 | Y≥1)` against the empirical share of 1s on a steep-decay synthetic window,
  which shows the boundary bias;
- each bandwidth option runs on a 3-positive window without failing silently.

**E:**
- equals A in distribution at α = 0, and C at α = 1;
- the coupling holds: at a fixed seed, raising α only switches draws from A to C.

**Reopen this ADR if:**
- any estimator changes the zero mass;
- any estimator emits a non-integer value;
- the evaluation plan is used to declare a winner.

---

## Open Questions

- Whether `"sj"` is stable enough on 2–20 positive values to be a usable per-cell selector. #109
  reports how many cells hit the fallback.
- Whether C's global fixed bandwidth should instead be scale-relative (proportional to the cell's
  positive median). Deferred to the D work (#110); not decided here.
- Which hydranet columns score the α-sweep. To be read off its separation matrix when E exists.

---

## References

**Governance:**
- ADR-001, ADR-004, ADR-008, ADR-011, ADR-012, ADR-020, ADR-021, ADR-022.

**Issues:**
- views-baseline epic #104, and #105–#112.

**Evaluation lessons:**
- `views-hydranet/reports/2026-09-30_evaluation_learning_report.md`. A living document: re-read
  it, don't cite it from memory.

**Papers (in the paper library):**
- Huang et al. 2023 (doi:10.1029/2023WR034942)
- Senga Kiessé & Durrieu 2024 (10.1016/j.spl.2024.110078)
- Marron & Ruppert 1994 (10.1111/j.2517-6161.1994.tb02006.x)
- Jones, Marron & Sheather 1996 (10.1080/01621459.1996.10476701)
- Gneiting & Ranjan 2013 (10.1214/13-EJS823)
- Cheng 1995 (10.1016/0167-7152(94)00190-J)
- Ferro 2008 (ensemble-size effect on CRPS)
