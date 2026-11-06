# Matryoshka conflictology: running log

What we tried, what came out, and what we decided, in the order it happened. Issue #127, ADR-024.
Newest entries at the bottom. Numbers come from the files named in each entry.

## 2026-11-05: prototype built (Stage 0)

- Code in `prototype/`, one idea per file:
  - `grid_geometry.py`: the grid and each cell's neighbours;
  - `size_classes.py`: the seven size classes;
  - `doll_counts.py`: how many cells of each doll fall in each class, month by month;
  - `forecast_probability.py`: from settings to forecast probabilities and the fitting score, worked out exactly rather than by drawing samples;
  - `fit_knobs.py`: the search for the best settings;
  - `write_report.py`: turns the results into `PROTOTYPE.md`.
- **Check that the formula matches the three-step draw** (`prototype/check_against_sampling.py`). Method: on a small made-up grid, 200,000 simulated draws were compared with the exact formula. Passed: drawn [0.4444 0.1038 …] against exact [0.4446 0.1032 …].
- **Data.** The pinned calibration input pull (md5 `d3636988b1085af74a08555c347ec5fe`), with all 64,818 land cells. Only months 121–456 are used, so nothing after the last training month enters.
- **Backtest.** History ends at months 384, 396, 408 and 420. Each is scored on the 36 months that follow.
- **Measured costs:**
  - loading the data: 9–11 s;
  - doll counts: about 20 s per violence type;
  - one scoring: 12–14 s the first time, then 1.4 s when only the weights change and 2.3 s when a memory changes.

## 2026-11-06: first real fit, state-based violence (`results/lr_sb_best.json`)

| Group of cells | Weight | Memory (months) |
|---|---|---|
| the cell | 1 (fixed) | 14.6 |
| ring 1 | 0.551 | 14.9 |
| ring 2 | 0.248 | 14.3 |
| rest of country | 0.247 | 13.1 |
| rest of map | 0.031 | 96.4 |

- **Score: −0.01520 for Matryoshka.** Conflictology (36-month window, cell only) scores −0.04067. The cell's whole history on its own scores −0.03455. The starting values already scored −0.01616.
- **Outcomes given a near-zero chance (below 1e-12) that then happened:** 0 for Matryoshka, against 11,456 for conflictology and 8,017 for the cell's whole history. Most of conflictology's worse score comes from these misses. The count is therefore the fairer comparison, because the score itself depends on the 1e-12 floor.
- **The rings shrink at about the expected rate.** Ring 1 to the cell is 0.55, and ring 2 to ring 1 is 0.45. Currents of Conflict suggests about 0.4.
- **Memory.** The cell, the rings and the country all forget at about the same pace (13–15 months). Only the rest of the map remembers much longer (96 months). The country's memory is far below the 600-month limit, so the question of very long memories (logged for v2) does not arise for this violence type.
- **The search settled properly:** "projected gradient ≤ tolerance".
- **Cost: 2,434 s (about 40 minutes) and 530 scorings.** This is over the 10-minute budget in the plan's gate. Peak memory was 3.19 GB.
  - **Cause.** The search estimates each step's direction by nudging each of the 9 settings in turn, so every step costs about 10 scorings.
  - **Planned fix.** Calculate the direction exactly from the score's formula. Expected effect: about ten times faster, with the same answer.
  - **Simon's call.** It would be nice to get the time down, but it is not a blocker.
- **Size classes seen in the history** (cell-months): the class above 1,000 has 74 entries for this violence type.

## 2026-11-06: non-state violence (`results/lr_ns_best.json`)

| Group of cells | Weight | Memory (months) |
|---|---|---|
| the cell | 1 (fixed) | 13.6 |
| ring 1 | 0.591 | 15.0 |
| ring 2 | 0.318 | 16.0 |
| rest of country | 1.871 | 11.5 |
| rest of map | 0.220 | 36.0 |

- **Score: −0.00547 for Matryoshka.** Conflictology scores −0.01638 and the cell's whole history −0.01477. The starting values scored −0.00565.
- **Near-zero misses: 0 for Matryoshka,** against 5,139 for conflictology and 4,405 for the cell's whole history.
- **The rings shrink at about the expected rate.** Ring 1 to the cell is 0.59, and ring 2 to ring 1 is 0.54. That is close to the state-based values.
- **This violence type leans much more on the country.** The rest of the country gets weight 1.87 (state-based: 0.25), and the rest of the map gets 0.22 (state-based: 0.03).
  - One reading: non-state conflict is rarer and moves around within a country more.
  - Not tested yet. Under the wait-for-the-stability-check rule, this is a candidate, not a finding.
- **Memory.** The cell, the rings and the country all sit at 11–16 months, as for state-based. The map is shorter (36 months).
- **The search settled properly:** "projected gradient ≤ tolerance".
- **Cost: 2,668 s (about 44 minutes), 580 scorings, peak memory 3.19 GB.**
- **Size classes.** The class above 1,000 has only 10 entries (cell-months) for non-state violence. That is thin, but the fit was not floored anywhere, so merging it with the class below is not needed yet.

## 2026-11-06: one-sided violence fit stopped, not failed

- **What happened.** The run was stopped by Claude Code, not by an error, while it was on one-sided violence. It happened while the session was idle and the computer was low on memory: 31 GB total, 2 GB free. The fit itself peaked at 3.2 GB, and other programs held most of the rest.
- **Still to do:**
  - one-sided violence;
  - the stability check (each history end left out in turn).
- Waiting for Simon's go-ahead before starting it again.

## 2026-11-06: where the fitting time goes (correction to the 10× estimate)

- **Method.** Each part of one scoring was timed on real data, one-sided violence, with the starting settings. The machine had memory to spare at the time.
- **Results:**

  | Part | Time |
  |---|---|
  | the cell | 0.2 s |
  | ring 1 | 0.70 s |
  | ring 2 | 0.74 s |
  | rest of country | 1.0 s |
  | rest of map | 0.97 s |
  | combining into the score | 0.49 s |
  | **total** | **about 4.1 s** |

- **Of that, about 0.7 s per ring is wasted.** The stored counts are converted to decimal numbers again on every scoring (`astype(np.float32)` of a 0.57 GB array each time).
- **Correction.** The earlier "about ten times faster" from exact step directions was wrong. A step costs about 12–15 s now and would cost about 8 s with exact directions, so about 2× faster.
- **Expected combined speed-up: about 3–5×,** from exact directions plus converting the counts only once. To be measured.
- **The full runs were also slowed by the machine itself.** Their spare memory on disk (swap) was full, and the average scoring took 4.6 s against 4.1 s measured here.

## 2026-11-06: speed changes (Simon: "go ahead with the speed changes")

- **Committed the prototype first,** as it stood (`983badf`). The first results moved to `results/v1_nudged_slopes/`, so the new fit can be compared with them.
- **Change 1: convert the counts once.** `doll_counts.py` now stores the counts as float32, class first, `(K, M, N)`, so each scoring multiplies them by the time weights directly.
  - Cost: about 0.6 GB more memory per array, about 2.4 GB in total.
- **Change 2: the exact slope.** `forecast_probability.py` (`log_score_slope`, `time_weight_slopes`) computes the slope of the score from the formula; the derivation is in the module docstring. The search (`fit_knobs.py`) uses it (`jac=True`) instead of nudging each setting.
  - How the memory slope is computed: it is the same weighted sum as the shares, with the month weights replaced by their slope. Both are stacked into one pass over the counts.
  - The maximum number of search steps went from 60 to 200. The old fits stopped because they had converged, not because they ran out of steps, so this changes nothing for them.
- **Checks:**
  - `check_against_sampling.py` gives exactly the same numbers as before the change.
  - `check_slope.py` (new) compares the exact slope with small nudges on a synthetic grid. Passed.
    - In float32, the memory slopes for the rest of the country and the rest of the map differed from the nudges by 4% and 30%.
    - Redone in float64 (a scratch copy with float32 replaced by float64) with nudges from 1e-2 to 1e-5, they agree to a relative gap of 1.2e-8 at the best nudge size. So the exact formula is right, and the gap was rounding noise in the nudges.
    - Side effect: the old search's directions for those two memories were noisy too.
  - Not yet strong: `check_slope.py` in float32 uses a loose tolerance (atol 1e-4), which the two smallest slopes (about 3e-4 and 3e-5) cannot meaningfully fail. The float64 run is the real check, and it is not yet saved as a file.
- **Running:** a refit of all three violence types (`fit_run2.log`). State-based and non-state are to be compared with `v1_nudged_slopes/`.

## 2026-11-06: refit with the exact slope (`results/v2_default_stopping/`)

**Run time.** Every type reached the search's own "converged" test ("projected gradient ≤ tolerance").

| Type | Steps | Scorings | Fit time | Peak memory |
|---|---|---|---|---|
| state-based | 11 | 13 | 88 s | 4.7 GB |
| non-state | 14 | 20 | 138 s | 5.9 GB |
| one-sided | 8 | 15 | 102 s | 5.9 GB |

- **Before:** 2,434 s with 530 scorings (state-based) and 2,668 s with 580 (non-state).
- **About 25× faster.** That is more than the 3–5× estimated, because the search also needs far fewer steps once its directions are exact.
- **One scoring with slope, including the counts:** about 6.5 s for the first one.
- **Within the 10-minute gate.**

**Scores: the same as before, to 6 digits.**

| Type | New | Old |
|---|---|---|
| state-based | −0.015195 | −0.015196 |
| non-state | −0.005469 | −0.005467 |

**But some settings moved:**

| Setting | Old | New |
|---|---|---|
| state-based ring 1 weight | 0.551 | 0.582 |
| state-based map memory (months) | 96 | 101 |
| non-state ring 2 memory (months) | 16 | 20 |
| non-state map memory (months) | 36 | 50 |
| non-state country weight | 1.87 | 2.12 |

Two possible readings:
- **(a) A flat score.** The score barely changes along some directions, so those settings are weakly determined by the data.
- **(b) Stopped too early.** The default thresholds (gtol 1e-5) are coarse for slopes of about 1e-4.

The test: refit with ftol 1e-13 and gtol 1e-9 (now in `fit_knobs.py`).

**One-sided violence (first fit):**

| Group of cells | Weight | Memory (months) |
|---|---|---|
| the cell | 1 (fixed) | 21.3 |
| ring 1 | 0.652 | 16.5 |
| ring 2 | 0.334 | 16.8 |
| rest of country | 0.598 | 19.7 |
| rest of map | 0.023 | 145 |

- **Score: −0.00671 for Matryoshka,** against −0.01955 for conflictology and −0.01634 for the cell's whole history.
- **Near-zero misses: 0 for Matryoshka,** against 5,861 for conflictology and 4,233 for the cell's whole history.
- **Size classes.** The class above 1,000 has 58 entries (cell-months).

## 2026-11-06: strict stopping thresholds (`results/v3_strict_stopping/`)

**Run time:** 5–7 minutes per type, still inside the gate. The search now stops on "relative reduction of the score ≤ ftol".

| Type | Steps | Scorings | Fit time |
|---|---|---|---|
| state-based | 29 | 61 | 408 s |
| non-state | 29 | 44 | 306 s |
| one-sided | 18 | 42 | 333 s |

**Scores improved, but only a little:**

| Type | Strict | Default stopping | Gain |
|---|---|---|---|
| state-based | −0.0151878 | −0.0151954 | 7.6e-6 |
| non-state | −0.0054628 | −0.0054692 | 6.4e-6 |
| one-sided | −0.0067079 | −0.0067138 | 5.9e-6 |

**But several settings moved a lot:**

| Setting | Default stopping | Strict |
|---|---|---|
| state-based map memory (months) | 101 | 26 |
| non-state ring 1 weight | 0.66 | 1.25 |
| non-state map memory (months) | 50 | 26 |
| one-sided ring 1 weight | 0.65 | 1.07 |

**Reading: both (a) and (b) were true.**
- (b) The default thresholds stopped the search early.
- (a) The score is very flat along these directions: large changes in a setting buy gains of about 1e-5. Such settings are weakly determined by the data.
- The stability runs (each backtest period left out in turn), started with the strict thresholds, will show how much they wobble.

**What stays stable across all three fits (v1, v2, v3):**
- the cell's memory: 14–15 months for state-based and non-state, 21–24 for one-sided;
- the country memory: 11–21 months;
- the score;
- zero near-zero misses.

The ring and map settings are the ones that wander.

## 2026-11-06: forecast maps (`maps/`, `prototype/map_forecasts.py`)

- **Setup.** History up to month 456, the last month the fit could see. Forecasts for months 457–492, which the fit never used, so the maps are out of sample. Knobs from `v3_strict_stopping`. Each cell shows the chance of at least one death, averaged over the 36 months, next to what happened and next to conflictology.
- **Out of sample, not in sample.** A first version used history up to month 420, which would have mapped months the fit was scored on. It was changed before running.
- **What the maps show:**
  - **Conflictology is speckled:** only cells that had deaths in the last 36 months get any probability. Everything else is exactly zero, including many cells where deaths then happened.
  - **Matryoshka has a country-wide floor plus halos.** The floor comes from the country doll: a whole country gets one level. The halos come from the rings: square, because the rings are squares of grid cells. Hot spots match conflictology's.
  - **For non-state violence,** Matryoshka spreads probability across the whole of Mexico and Brazil, where the outcome was scattered events.
  - **For one-sided violence,** quiet regions fall below 0.0001 (map weight 0.038).
- **Things to look at:**
  - **The country floor is uniform across a country,** however far a cell is from the fighting. This shows clearly in large countries (DR Congo, Sudan, Brazil). It is by design in v1.
  - **The square halos** are visible around single isolated events.

## 2026-11-06: conditional-mean maps (`maps/conditional_mean_*.png`)

Simon asked for the expected deaths in a cell, given at least one.

**Method.**
- **Matryoshka:** exact, from the doll mix. Σ_h E[value] / Σ_h P(value > 0), pooled over forecast months 457–492, with knobs from `v3_strict_stopping`.
- **New code:**
  - `prototype/doll_value_sums.py`: the value counterpart of the doll counts;
  - `prototype/check_value_sums.py`: with value = 1(class = j), the doll mean equals the class-j share for every doll and class. Passed.
- **Conflictology:** the mean over the non-zero months of the cell's last 36 months.
- **What happened:** the mean over the non-zero months of 457–492.

**What the maps show: in remote quiet places Matryoshka's conditional mean is higher, about 20–30 deaths, than in active places, about 3–10.**
- **Cause.** In a cell with no recent deaths, almost all non-zero draws come from the rest-of-map doll. Those draws carry the size of events in the world's war zones.

**Check** (`prototype/check_conditional_mean_by_group.py`, months 457–492). "Quiet before" means no deaths in months 421–456.

| Type | Group | Cells with deaths after | Observed mean given > 0 (median) | Matryoshka median in those cells | Matryoshka median over the whole group |
|---|---|---|---|---|---|
| state-based | quiet before | 527 | 8.6 (3) | 8.8 | 24.8 |
| state-based | active before | 947 | 17.7 (6) | 11.1 | 9.2 |
| non-state | quiet before | 519 | 7.8 (3) | 12.2 | 21.8 |
| non-state | active before | 322 | 15.1 (4) | 10.3 | 10.4 |
| one-sided | quiet before | 658 | 7.8 (2) | 6.5 | 26.3 |
| one-sided | active before | 480 | 5.6 (2) | 4.3 | 4.5 |

**Reading:**
- **Where new violence actually broke out, Matryoshka's size forecast is about right.** Those cells were mostly near existing conflict, where the rings and country dominate.
- **Across the whole quiet group it says about 22–26 deaths.** That is about 3× what new outbreaks actually bring (mean about 8, median 2–3). The cells concerned get a tiny chance of any deaths, so it barely moves the scores, but the shape is wrong: "if anything happens in Canada, expect a war-zone-sized month".
- **For cells that were active before, sb and ns,** the observed mean (15–18) is above Matryoshka's median forecast (10–11). This compares a pooled mean with a median across cells, so it is only a hint, not a measurement.

**For the evaluation and for v2:**
- ADR-024's "size conditioned on the forecast" check should show this.
- A design question for Simon: should the map doll (and perhaps the country doll) bring only occurrence and not size, or bring size from quiet-before places only?

## 2026-11-06: what the conditional mean means in quiet places (discussion with Simon)

**Simon's point.** Matryoshka's large conditional mean in remote quiet places is not wrong in kind. With nothing local to go on, borrowing from the rest of the map is a defensible answer to "if something happened here, how big?"

**Conflictology has no answer at all.** In a cell with no deaths in its window, all draws are 0, so P(≥1) = 0 and the conditional mean is 0/0, undefined. That is the grey in its panels. It is not saying "small" or "uncertain": the question does not exist for it. A plus for Matryoshka.

**Level, not kind.** The fallback answers "what does violence look like anywhere in the world?" The world pool is dominated by sustained wars, which puts it at about 25. "What does violence look like when a quiet place erupts?" gives an observed mean of about 8 (median 2–3) in months 457–492.

**Caveat.** Most quiet-before cells that erupted were near existing conflict. Remote (Canada-like) eruptions are too rare for this check to speak about them specifically.

**v2 option.** Keep the fallback, but have it draw sizes from quiet-before-then-erupted cases rather than from all non-zero months.

## 2026-11-06: the way ahead agreed (epic #135)

Simon agreed the order: province first, then occurrence vs size, then neighbouring countries. His point on step 2: occurrence and magnitude are two different processes.

- **#136: province (admin-1) group** between the rings and the country.
  - Data: GAUL admin-1 in views-datafactory `data/raw/gaul_admin/gaul1_code.parquet`.
  - On our land cells: 2,433 provinces, none crossing a `c_id` border, median 6 cells per province.
- **#137: separate whether violence happens from how big it is** in the outer groups. Sizes would come from quiet-places-that-erupted.
- **#138: neighbouring-countries group** between the country and the rest of the map.
- **Method for each step:** test in the prototype first, compare with the previous version, log, then keep or drop.
