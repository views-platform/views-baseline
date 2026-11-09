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

## 2026-11-06: stability check (`results/*.json`, `PROTOTYPE.md`)

**Method.** For each violence type, the knobs were refitted four times, each time leaving one backtest period out, with strict stopping. Each refit took 2.5–4 minutes.

**Deterministic.** The main fit repeated `v3_strict_stopping` exactly.

**State-based: stable.**

| Setting | Range across the four refits |
|---|---|
| memory: cell, rings, country | 11–17 months |
| memory: map | 21–40 months |
| weight: ring 1 | 0.46–0.65 |
| weight: ring 2 | 0.20–0.29 |
| weight: country | 0.16–0.35 |
| weight: map | 0.026–0.070 |

**Non-state: mostly stable, with two trade-offs.**
- **Stable:**
  - memory: cell 14–16, ring 1 10–13, map 23–31 months;
  - weight: ring 1 0.95–1.52.
- **Leaving out 396:** ring 2 is almost switched off (weight 0.10), and its memory runs to 85 months. With so little weight, its memory hardly matters.
- **Leaving out 408:** the country swaps "short memory, high weight" (11 months, 2.3) for "long memory, low weight" (87 months, 0.54). Two near-equivalent ways to explain the same data.
- **Leaving out 420:** the search ended "ABNORMAL" (the line search could not improve further at the precision limit). The score (−0.005319) is in line with the others.

**One-sided: stable except the map memory.**
- **Stable:**
  - memory: cell 18–30, rings 12–17, country 17–23 months;
  - weight: ring 1 0.95–1.11, ring 2 0.40–0.62, country 0.58–0.74, map 0.025–0.043.
- **Not stable: map memory,** 164, 426, and twice the 600-month ceiling. With a map weight of about 0.03, the data barely constrains it. This is the "very long memory" question logged for v2 (log-time kernel), appearing for one-sided violence only.

**Reading against the plan's gate** ("≲ 10 minutes per run and the knobs are stable"):
- **Time: passed.** About 5–7 minutes for a full fit.
- **Stability: passed for every setting that carries real weight.**
  - The settings that wander are those whose group gets little weight (non-state ring 2 when it is switched off; the one-sided map), plus one weight–memory trade-off (non-state country).
  - In both cases, the fit hardly cares.
  - For the package: either accept it (the frozen knobs are still a good fit) or pin such memories. That is a decision for Simon.

## 2026-11-06: #136 province group, implementation and checks

Plan approved by Simon: "investigate and plan first, then execute".

**Data facts:**
- **Codes.** `c_id` equals GAUL0 for every land cell. GAUL admin-1 comes from views-datafactory `data/raw/gaul_admin/gaul1_code.parquet`, passed as an explicit argument.
- **"No province".**
  - `-1` means unassigned. `-3727`…`-3730` are the synthetic Azores island provinces (views-datafactory ADR-043), and they count as real provinces.
  - Rule: a cell has a province iff its code ≠ −1 and it has a country.
  - `build_geometry` raises if a province would span two countries, or if a cell has a province but no country.
- **Sizes.** Per province the median is 6 cells, but 80% of cells (52,116) are in provinces larger than the 5×5 ring box.

**Model.** Six groups: cell, ring 1, ring 2, rest of province, rest of country, rest of map. Province ⊂ country, so:
- province rest = province total − inner cells in the same province;
- country rest = country total − inner cells in the same country − province rest;
- the map is unchanged.

That gives 11 settings. The starting values put the province between ring 2 and the country (weight 0.12, memory 36).

**Not a strict extension of v3.** With province weight 0, the province's cells drop out of every group, so it is not the 5-group model. Comparison with v3 is model against model.

**Code.** One list (`DOLLS`) drives the number of groups. The touched files are `grid_geometry.py`, `doll_counts.py`, `doll_value_sums.py`, `forecast_probability.py`, `fit_knobs.py` (results now go to `results/v4_province/`), the map scripts (knobs and output folders are now arguments), `write_report.py` (reads the group names from the JSON) and `check_conditional_mean_by_group.py` (tidied into a proper script).

**Checks, all passed:**
- **`check_against_sampling.py`, rebuilt.** On a synthetic grid with two countries, three provinces, a cell without a province and a cell without a country:
  1. the six shells, built independently from distance, province and country, are disjoint and cover every cell, for every cell;
  2. the shell sizes equal `DollCounts.sizes`;
  3. every group's shares equal a direct average over its cells, for every cell;
  4. the three-step draw matches the exact formula for a cell with a province shell (39 cells) and for one without.
- **The check can fail.** Three deliberate breaks were each caught, applied to a scratch copy:
  - the country rest without subtracting the province: caught ("country shares disagree");
  - the province rest without subtracting the inner cells: caught;
  - the country shell size without subtracting the province: caught ("sizes disagree").
- **`check_slope_float64.py` (new).** It runs `check_slope.py` on a float64 copy with rtol 1e-6, closing the weak-check note from earlier today.
  - The largest relative gap over the 11 slopes is 6.2e-9.
  - Dropping the memory-through-month-weights term in a scratch copy is caught (gap 1.8).
- **`check_value_sums.py`** passes with provinces.

**Egypt check** (input pull, one-sided violence, Egypt `c_id` 120, months 400–456):
- 21 non-zero cell-months and 517 deaths in total.
- 311 of them (60%) are a single cell-month: North Sinai, month 455 (November 2017, the Sinai mosque attack).
- Next largest: Kalyoubia 54 (month 403), Beni Suef 29, Behera 29.
- **Guess confirmed.** One event dominates Egypt's one-sided history right before the history ends.

**Running:** the 6-group fit plus stability for all three violence types (`fit_province.log`).

## 2026-11-06: #136 result, the province group adds almost nothing (`results/v4_province/`, `maps/province/`)

**Fit.** All three types converged. The loader confirms: 76 cells without a province (all of them also without a country), and the four Azores codes in use.

| | state-based | non-state | one-sided |
|---|---|---|---|
| score v4 (6 groups) | −0.0151838 | −0.0054606 | −0.0067095 |
| score v3 (5 groups) | −0.0151878 | −0.0054628 | −0.0067079 |
| difference | +4.0e-6 | +2.3e-6 | **−1.6e-6** (worse) |
| near-zero misses | 0 | 0 | 0 |
| fit time | 417 s | 173 s | 335 s |
| peak memory | 5.4 GB | 7.2 GB | 7.2 GB |

The differences are at the level of the noise between stopping rules seen earlier (about 6e-6). One-sided is slightly worse. That is possible because v4 does not contain v3 as a special case.

**Fitted province weight and memory:**

| Type | Province weight | Province memory | Country weight (v4) | Country weight (v3) |
|---|---|---|---|---|
| state-based | 0.063 | 12.6 months | 0.235 | 0.274 |
| non-state | 0.33 | 10.1 months | 2.02 | 2.32 |
| one-sided | 0.049 | 72 months | 0.587 | 0.66 |

The rings and cell are unchanged from v3.

**Per-cell pull** (`prototype/check_province_pull.py`): (province weight / province shell size) ÷ (country weight / country shell size), over the 59,971 cells with both shells.
- Median: 2.3 for state-based, 1.4 for non-state, 0.7 for one-sided.
- Median shell sizes: province 162 cells, country 2,829 cells.
- So a province cell counts at most about twice as much as any other cell of the country. The fit treats being in the same province as worth little beyond the rings and the country.

**Stability:** similar to v3.
- One-sided province memory is unstable: 18–114 months, and the map memory hits the 600 ceiling once.
- The non-state country weight/memory trade-off reappears when 408 or 420 is left out (memory 98 and 67).

**Maps and size check:**
- The maps are visually almost unchanged from v3.
- **Egypt is still nearly one block in the one-sided size map:** North Sinai median 54 against 43 for the rest of Egypt.
- The quiet-before / active-before size table is practically unchanged (one-sided quiet-before, whole group: median 32.7, against 26.3 in v3).

**Reading:**
- **The fitting score sees no value in the province level.** The score is about how likely each size class is, and it is dominated by cells near conflict, where the cell and rings decide. The province shell (median 162 cells) mostly repeats what the rings and the country already say.
- **The Egypt artefact is a size problem.** One huge event inflates the sizes drawn from Egypt. The province split does not touch it because the fit does not reward it, and the occurrence-focused score barely sees it. That is #137's territory (size separate from occurrence), not #136's.
- **Caveat:** the fitting score is not the judge. The evaluation ruler (ADR-024) could still show a difference, but nothing here suggests it will.

**Recommendation to Simon: drop the province group for v1.** Revert the model change, but keep the commit for the record. Reasons:
- no measurable gain;
- one-sided slightly worse;
- two more settings, one of which is unstable;
- a new cross-repo data dependency (`gaul1_code` in the views-models input pull).

**Keep:**
- the stronger checks from this step: the rebuilt `check_against_sampling.py` and `check_slope_float64.py`;
- the parametrised map scripts and report.

**Revisit provinces** only if #137's size pools turn out to need a sub-country level.

## 2026-11-06: map colour floor, and the 5- vs 6-group comparison maps

**Simon's question on the one-sided map:** why does Saudi Arabia show a few coloured cells but nothing for the rest of the country, and why are some single cells coloured with no rings around them?

**Cause: the grey cut-off of the colour scale (0.0001), not the model.** Checked on the v4 one-sided forecasts:

| Saudi Arabia (713 cells) | Chance of ≥ 1 death |
|---|---|
| the 6 cells with past one-sided deaths (latest month 434) | median 6.4e-3 |
| their ring-1 neighbours | median 3.7e-4 |
| the rest of the country | median 5.9e-5, under the cut-off, so drawn grey but not zero |

- **Isolated coloured cells.** For one-sided violence, 38 cells clear 0.0001 while all their neighbours fall below it (neighbours about 5e-5 to 9e-5). These are cells whose last event was years ago, for example in Madagascar, Zimbabwe, Peru and Indonesia.
- **Fix to the maps.** The colour floor is now 1e-6 (`map_forecasts.py`, `FLOOR`; `compare_maps.py`). Grey now means exactly 0 or below 1e-6.

**Comparison maps** (`maps/province/compare_*.png`, from `prototype/dump_version.py` and `prototype/compare_maps.py`; the arrays are in `results/v3.npz` and `results/v4.npz`): 5 groups | 6 groups | ratio.

| | Median ratio (6 / 5 groups) | Cells changed by > 10% |
|---|---|---|
| chance of ≥ 1 death | 0.98–0.995 | 4.5–7.1% |
| conditional mean, state-based and non-state | about 1.00 | 1–3% |
| conditional mean, one-sided | **1.25** | **84%** |

- **The chance maps change by whole provinces.** The ratio maps show provinces lighting up or dimming as blocks: for example a northern Niger province up, a northern Chad province down. Those changes do not improve the fitting score.
- **The one-sided size shift is broad, not local.** It comes from the refit (the map memory moved from 174 to 268 months, and the weights shifted), not from provinces as such.

## 2026-11-06: #139 smooth distance kernel, implementation and checks

**Simon's call.** The square halos are an artefact of the two hard-edged rings. Agreed plan:
- replace the rings with one smooth distance kernel;
- park the province group (#136) and re-test it on top of the kernel model;
- step 0 of epic #135.

**Model.** Four groups: cell, neighbourhood, rest of country, rest of map.
- **Neighbourhood:** every land cell at grid distance 0 < d ≤ 5 cells. A neighbour is drawn with probability ∝ exp(−d / ℓ).
- **Settings: 8,** down from 9:
  - weights: near, country, map;
  - memories: cell, near, country, map;
  - the length scale ℓ.
- **Starting values:** near weight 0.6 (about the two rings' sum); ℓ = 1.44 cells (Currents of Conflict, 0.72°), bounded to 0.25–10 cells.
- **Distance:** straight-line distance on the grid, with no latitude correction (stated in #139).

**Code** (`grid_geometry.py`, `doll_counts.py`, `forecast_probability.py`, `doll_value_sums.py`, `fit_knobs.py`, map scripts, report):
- Neighbours are grouped in 13 distance bands (1, √2, 2, …, 5); the kernel recombines the bands at each scoring.
- **Only non-zero cell-months are stored,** as sparse matrices. Class 0 is "1 − the rest".
- Everything is now float64. The province is still loaded and validated, but no group uses it (parked).
- **Speed:**
  - counts: 20 s → 1 s;
  - one scoring with all slopes: 1.8 s (the first, uncached);
  - peak memory: 4 GB.
- Results go to `results/v5_kernel/`.

**Checks, all passed and all shown to fail when broken:**
- **`check_against_sampling.py`, rebuilt for the kernel.** It runs on a 16 × 22 grid with sea holes and a cell without a country. Shells and kernel weights are built by brute force from pairwise distances, independently of the offset table. It confirms, for every cell:
  - the shells are disjoint and cover every cell;
  - the shell sizes match;
  - every group's shares equal the direct kernel-weighted averages (to 1e-10);
  - the three-step draw (neighbour ∝ exp(−d/ℓ)) matches the exact formula, for a cell with a full neighbourhood (80 cells) and a corner cell (22).
- **Deliberate breaks, each caught:**
  - uniform instead of kernel weights;
  - the country rest without subtracting the inner cells;
  - the country size without subtracting the neighbourhood;
  - scrambled neighbour bands.
- **`check_slope.py`.** Strict by default now (rtol 1e-6, float64 throughout), which replaces `check_slope_float64.py`.
  - Largest gap over the 8 slopes, including the length scale: 8.9e-9.
  - Dropping the −q·dZ/Z term of the length slope is caught (gap 1.7).
- **`check_value_sums.py`** passes for the kernel model.

**Running:** the main fit plus stability for all three types (`fit_kernel.log`).

## 2026-11-06: #139 result, the smooth kernel is better, simpler and faster (`results/v5_kernel/`, `maps/kernel/`)

**Fit.** All three violence types converged.

| | state-based | non-state | one-sided |
|---|---|---|---|
| score, kernel | −0.0151347 | −0.0054613 | −0.0066772 |
| score, square rings (v3) | −0.0151878 | −0.0054628 | −0.0067079 |
| difference | **+5.3e-5** | +1.5e-6 (tie) | **+3.1e-5** |
| near-zero misses | 0 | 0 | 0 |
| length scale ℓ | 0.80 cells = 0.40° | 1.35 cells = **0.68°** | 0.98 cells = 0.49° |
| ℓ across the four leave-one-out refits | 0.77–0.83 | 1.23–1.65 | 0.95–0.99 |
| fit time | 61 s | 51 s | 69 s |

- **The gains are real.** For state-based and one-sided they are 5–10× the noise between stopping rules (about 6e-6). They come with one setting fewer (8, against 9).
- **The non-state length scale matches the literature.** At 0.68° it is close to Currents of Conflict's 0.72°. State-based and one-sided prefer a tighter neighbourhood (0.40–0.49°).
- **Fitting is about 5× faster** than with the rings: about 1 minute per fit.

**Fitted settings:**

| Type | Near weight | Country weight | Map weight | Memory (months): cell / near / country / map |
|---|---|---|---|---|
| state-based | 0.95 | 0.18 | 0.046 | 13.5 / 13.4 / 11.6 / 24.8 |
| non-state | 2.80 | 0.38 | 0.21 | 15.6 / 11.6 / 80 / 29 |
| one-sided | 1.72 | 0.47 | 0.028 | 23.1 / 13.8 / 22.1 / 600 (ceiling) |

**Stability:**
- **State-based:** stable throughout.
- **One-sided:** stable, but the map memory sits at the 600-month ceiling in all five fits. Stably at the ceiling means "remember everything" for the rest of the map, which is the logged v2 log-time question.
- **Non-state:** the country memory is unstable (11–196 months). The country weight/memory trade-off seen before is still there.

**Plots** (`maps/kernel/`):
- **`neighbour_pull.png`.** The pull of one neighbouring cell, relative to the cell itself, by distance: the rings' staircase against the kernel's straight line on a log scale. The kernel passes through the middle of the old steps, then keeps going smoothly instead of dropping to zero after ring 2. (The rings are drawn at their ring number, so their x position is approximate.)
- **`forecast_maps_*.png`:** the halos are now round and fade smoothly; no square boxes.
- **`compare_chance_*.png`** (rings | kernel | ratio). The kernel model is red in round bands 3–5 cells out from hot spots, where the rings gave nothing beyond ring 2, and blue in the old square corners.
  - **Non-state:** the country floors are much lower (median ratio 0.58), because the fit moved weight from the country (2.3 → 0.38) to the neighbourhood (2.8). Egypt goes up (country memory 80 months, so old events count).
- **Size check:** practically unchanged. One-sided quiet-before, whole group: median 41 (v3: 26; v4: 33). The one-sided map memory at the ceiling makes the world fallback even more war-weighted, which is #137's problem.

**Recommendation: keep the kernel.** Next: re-test the province group on top of it (#136), then #137.

## 2026-11-06: everything locked; follow-ups filed as issues

Simon: "set up new issues before you do new stuff… we do not want to lose the ball."

- **Branch `feat/127-matryoshka-prototype` is pushed** at `eef6cf6`. All results, maps, reports and the code are committed.
- **Filed under epic #135:**
  - **#140:** settings the data barely pins down (one-sided map memory at the ceiling; non-state country weight/memory trade-off). Decide after #137.
  - **#141:** latitude-correct distances in the kernel. Low priority.
  - **#142:** amend ADR-024 and #127's build plan to the prototype's final design. Blocks the package build.
- **Epic #135 updated:** #139 done and kept; #136 is next (re-test on top of the kernel). #127 has a pointer to #142.

## 2026-11-06: #136 province re-test on top of the kernel, implementation

**Model.** Five groups: cell, neighbourhood (kernel), rest of province, rest of country, rest of map. Province ⊂ country, with the same arithmetic as the first province test, now with "inner" = cell + kernel neighbourhood. That gives 10 settings:
- weights for near, province, country and map;
- 5 memories;
- the length scale.

The province starts with weight 0.05 and memory 36 months. Results go to `results/v6_kernel_province/`.

**Checks:** all three pass with provinces in the synthetic grids (three provinces, a cell without a province, a cell without a country):
- shells disjoint and complete, sizes and shares direct, sampling;
- slope gap 8.8e-9;
- value sums.

**Deliberate breaks, each caught:**
- the country rest without subtracting the province;
- the province rest without subtracting the inner cells.

**Running:** the main fit plus stability (`fit_kernel_province.log`).

## 2026-11-06: #136 re-test result, the province adds nothing on top of the kernel either (`results/v6_kernel_province/`)

**Fit.** All three types converged.

| | state-based | non-state | one-sided |
|---|---|---|---|
| score, kernel + province | −0.0151362 | −0.0054575 | −0.0066784 |
| score, kernel (v5) | −0.0151347 | −0.0054613 | −0.0066772 |
| difference | −1.6e-6 (worse) | +3.9e-6 | −1.2e-6 (worse) |
| province weight | **0.000** (switched off) | 0.30 | 0.034 |
| province memory (months) | 1.0 (meaningless at weight 0) | 9.7 | 192 |
| fit time | 521 s | 249 s | 328 s |

- **Every difference is within the noise level** of about 6e-6. State-based switches the province off entirely; one-sided gives it almost nothing.
- **The province memory is unstable across the leave-one-out refits:** state-based 1–22 months, one-sided 82–439 months.
- **Everything else matches v5:** the length scale, near weight and other memories.
- **The fit is 5–9× slower** than the kernel alone.

**Maps** (`maps/kernel_province/compare_*.png`, kernel | kernel + province | ratio):
- **Only 2–4.5% of cells change their chance by more than 10%,** and 1–3.5% their size.
- **The visible province blocks** (for example in Niger and Algeria) come from the country shell now *excluding* the cell's own province. That holds even where the province weight is 0, so a cell in a quiet province of a violent country borrows from the violent rest. This is not a province signal.

**Reading.** With the smooth kernel the neighbourhood already covers what the province could add. The province is not worth two settings, a cross-repo data dependency (`gaul1_code` in the views-models input pull), or the slower fit.

**Recommendation to Simon:**
- **Close #136 as not planned.**
- **Revert the model to the kernel-only version (v5),** keeping the province commits (`fa4273d`, `2957a20`) for the record.
- The extended checks stay valid for the kernel model once the province is removed.

## 2026-11-06: #136 closed (Simon: close and revert)

- **The prototype code is restored to the kernel-only model** (`prototype/` as of `eef6cf6`: four groups, 8 settings). All three checks pass again.
- **The province code stays in git history:**
  - `fa4273d`: province with the square rings;
  - `2957a20`: province on the kernel.
- **The results stay** in `results/v4_province/` and `results/v6_kernel_province/`, with maps in `maps/province/` and `maps/kernel_province/`.
- **No issue in views-models for `gaul1_code` is needed.**
- **Next: #137** (whether violence happens vs how big it is).

## 2026-11-06: #137 groundwork, how big are outbreaks compared with all violence? (`prototype/describe_onset_sizes.py`)

Training data only (≤ 456). An *onset* is a violent cell-month whose cell had no deaths in the 36 months before. A *remote onset* also had no deaths in its neighbourhood (≤ 5 cells) in those months.

| Type | Period | All violent months: mean (median) | Onsets | Remote onsets |
|---|---|---|---|---|
| state-based | 157–456 | 21.8 (4), n=42,950 | 14.0 (3), n=4,861 | 17.2 (4), n=247 |
| state-based | 421–456 | 25.0 (6), n=8,208 | **10.0 (4)**, n=677 | 5.9 (3), n=29 |
| non-state | 157–456 | 21.0 (5), n=8,119 | 20.4 (4), n=2,195 | 23.8 (6), n=250 |
| non-state | 421–456 | 21.4 (6), n=2,209 | **10.6 (4)**, n=410 | 11.4 (3), n=41 |
| one-sided | 157–456 | 53.0 (2), n=19,678 | 59.2 (2), n=3,836 | 21.3 (3), n=346 |
| one-sided | 421–456 | 9.8 (2), n=2,787 | 8.5 (2), n=611 | 11.9 (3), n=56 |

**Reading:**
- **In recent years, outbreaks are about half the size of all violence** for state-based and non-state (10 against 21–25). That matches the observed out-of-sample level for quiet places (about 8). So an onset pool should help these two types.
- **Over the whole history the gap is small or absent.** Non-state is the same, and one-sided outbreaks are even larger (the 1990s massacres). So the onset pool only helps if the pool is weighted toward recent months.
- **One-sided's oversized fallback is a memory problem, not an onset problem.** Recent one-sided violence is small (mean 9.8) either way. Matryoshka gives 26–41 because the map memory sits at 600 months, so the 1990s dominate. That is #140, and an onset pool with the same memory would not fix it.
- **Remote onsets are too rare to build a pool from** (29–56 in the last three years). So the onset definition stays on the cell's own history.
- **Class shares** (what the score sees) differ only modestly between onsets and all violence. Expect a small score change; judge the step mostly on the size check and the maps.

## 2026-11-06: model declared done (Simon)

**Simon's framing.** Matryoshka is conflictology that answers one extra question: *which set of past observations do we draw from?* Judged by that, the model is done.

- **Final design: v5 kernel** (`results/v5_kernel/`).
  - Four groups: the cell, a smooth neighbourhood (≤ 5 cells, ∝ exp(−d/ℓ)), the rest of the country, the rest of the map.
  - 8 settings; the fit takes about 1 min.
- **Nothing is broken.**
  - The large "if something happens" value in quiet places is what the world pool honestly gives.
  - The one-sided map memory at 600 months means "the whole history, roughly equally"; 600 is stated as a deliberate ceiling.
- **No low-hanging fruit left:**
  - **#137 closed.** A size-by-outbreak rule is a new kind of machinery, and the gain would be in sizes only.
  - **#138 closed.** The kernel already crosses borders.
  - **#141 closed.** ℓ is under one cell.
  - **#140 decided: accept.** Report the ranges.
- **Smoothing (Silverman KDE) considered and rejected.**
  - It widens the pool but does not move its centre, so it does not help the Canada level. On the log scale it nudges the mean up.
  - Matryoshka has no support gap: the outer groups reach every observed value, which is why there are 0 near-zero misses.
- **Next:**
  - #142: amend ADR-024 and #127's plan to the v5 design;
  - the package build (#127 stages 1–4);
  - release 1.3.0;
  - evaluation against conflictology in views-models.

## 2026-11-06: #142 done, ADR-024 amended and #127's plan updated

- **ADR-024** (`docs/ADRs/024_matryoshka_conflictology.md`):
  - the two rings are replaced by the smooth neighbourhood throughout;
  - 8 knobs, with their limits and the 600-month ceiling;
  - a new section, "Amendment 2026-11-06: what the prototype settled", gives each decision with its issue, the fitting method, the fitted knobs and their stability.
  - `validate_docs.sh` passes.
- **#127's body now carries the build plan** for stages 1–4 on the final design:
  - a `matryoshka/` sub-package ported file by file from the prototype;
  - the configuration keys;
  - the prototype checks to port as tests. The value-sum check is not ported: the package draws samples and does not compute expected values in closed form.
  - `scipy` must be declared in `pyproject.toml`; it is not declared today.

## 2026-11-07: #127 stages 1–2, the package build

**Code.** The new `views_baseline/model/models/distributional/matryoshka/` sub-package is ported module by module from the prototype:
- `geometry`, `size_classes`, `doll_counts`, `forecast_probability` and `knob_fit`;
- `draw` (new): the three-step sampler;
- `model` (new).

It is wired into the catalog and the gate. The genome is `n_samples`, `seed`, `country_column`, `backtest_cutoffs`, `backtest_horizon` and `stability_refits`. `scipy` is now declared in `pyproject.toml`. The docs are written: the CIC, the README, the CIC index, and the ADR-001 and ADR-004 entries.

**Departures from #127's plan, decided while building:**
- **No `history_months` key.** The history is all the data passed, as in the prototype. Fit fails loud with fewer than 120 months before the earliest cutoff.
- **A required `country_column` key** (`c_id` in views-models), instead of a hard-coded column name.
- **Predict draws a block of 256 cells at once.** One call per (cell, month, target) took about 90 µs, or about 10 minutes per origin. The block draw consumes the same random stream in the same entity → time → target order (ADR-011), so the block size cannot change the output (tested).
- **Shell draws are exact, with no rejection sampling.** Cells are put in country-major order, and the j-th cell of a shell is found by skipping its excluded stretches.

**Tests** (`tests/test_matryoshka.py`, 27):
- the four prototype checks, ported: brute-force shells, direct shares, the independent three-step draw, and the strict slope check;
- the sampler's shells against brute force, exhaustively;
- the sampler against the exact probabilities, at h = 1 and h = 12;
- the conflictology special case, a planted neighbour effect, and no draws after the origin;
- loud failures, the seed, the block size, and both input types;
- the shared lists updated.

The full suite: 565 passed, 1 xfailed.

**Real data** (`package_check.py` → `results/package_check.json`):
- **The package's knobs equal the prototype's v5 exactly** (relative gap 0, score gap 0) for all three types, all converged. The values are integers, so the FeatureFrame's float32 storage is exact here.
- **Fit:** 621 s for the three types, peak 5.4 GiB. The prototype's search alone took 50–70 s per type. This run also builds the counts, scores the starting values, and ran beside other sessions. The difference is not investigated.
- **Predict, one origin:** 35 s for one type at S = 64. Scaled to 3 types × S = 256, that is about 7 minutes per origin.
- **The first attempt was stopped** by Claude Code for low system memory: it had predicted all 3 types × S = 256 at once, about 7 GB of output. The rerun used the smaller predict.

## 2026-11-07: #127 stage 3, the three reviews of 702f17d and the fixes

**review-diff:**
- no bugs in the core logic;
- 2 warnings: negative target values are not rejected, and leftover "ring" naming with an inconsistent `> 0` filter between the score and its slope;
- 8 suggestions.

**code-review (high):**
1. **Float rounding in the row lookup, silent but rare.** Rows are shifted apart by 2·row, so for the largest uniform `rng.random` returns (u ≈ 1 − 2⁻⁵³), `u + shift` can round onto the row's final 1.0. A neighbourhood draw could then land on a zero-weight sea offset. This happens in about 1 of 2⁴⁵ draws, roughly 1e-4 times per origin.
2. **Negative values:** a value below 0 scores as the largest class.
3. **A panel where no cell has a country** crashes.
4. **Predict's history start is not checked against the fit's.**
5. Efficiency points, including a cache that holds up to 41 entries of dead arrays.

**falsify guard audit** (independent, in a worktree):
- 59 mutations: 37 caught, 22 survived, 2 of them equivalent.
- **Strong:** the exact maths, the slopes, and the catalog and genome wiring.
- **Weak:**
  - which cell the sampler picks, because the fixture's values are i.i.d. across cells;
  - c and h, which were only guarded in one direction;
  - the RNG order with only one target;
  - the 120-month boundary;
  - the stability refits;
  - half the fail-loud checks, two of which passed only because an unrelated message matched.

**Fixes:**
- **Row lookups are capped at the row's last positive entry.** The new test fails without the cap, checked by mutation.
- **Negative values are rejected** in `classify` and in fit and predict.
- **No-country panels work.** An extra zero column handles the cells without a country.
- **Predict requires the fit's history start** (`history_start`).
- **The cache keeps one entry per doll.** Peak memory went from 5.4 to 4.3 GiB.
- **The starting score comes from the search's first evaluation.**
- **Smaller fixes:** "ring" renamed to weight everywhere, the `> 0` filter dropped, the unused `window` parameter removed, a cheaper completeness check (`bincount`), a block size that scales with S (`BLOCK_UNIFORMS`), the redundant `require_entities` dropped, and the seed-wiring comment moved.
- **22 new tests aimed at the survivors:**
  - encoded values (month × N + cell) show exactly which doll, cell and month each draw came from;
  - an exact ADR-011 stream check with two targets;
  - the 120/119-month boundary;
  - refits equal direct searches;
  - the score's mix equals the sampler's on a grid with empty shells;
  - the date-line wrap;
  - one test per fail-loud check, matched on its own message;
  - the fixture now has three cells without a country.
- **Not done:** the duplicate NEAR work in `near_length_slope`. It is efficiency only, and keeping it leaves the code equal to the checked prototype.

**Suite:** 592 passed, 1 xfailed. **Real data, rerun after the fixes:** the knobs still equal the prototype's v5 exactly for all three types. Fit 616 s, peak 4.3 GiB. Predict for one type at S = 64 took 41 s.

## 2026-11-07: #127 stage 3, guard re-audit of aefe448 and the last gaps

**The same independent auditor** re-applied the survivors and ran 13 fresh mutations:
- 20 of the earlier survivors are now caught.
- Equivalent mutants, which cannot change behaviour:
  - DC2/DC3 and N9: the country-less arithmetic is masked by `np.where(has_country, …)`; the output is bit-identical;
  - G3: a row above the grid always gives an id ≤ 0, which can never match a cell;
  - D9/D10: probably equivalent, since the overflow cap covers them.
- **Real gaps that remained, each closed by a new test, with every mutation re-applied and caught:**
  - **D7:** the country draw without `block_start`. The tested cell sat in the block starting at 0; now a cell in the second country is checked too.
  - **N2:** `near_last` one offset early; now checked directly per cell.
  - **N5/N7:** predict's own negative check. With it lost, predict returned negative draws silently; now tested.
  - **N12:** `start_score` was not checked; now compared with `Objective.score` at the starting values.
  - **G1:** the date-line wrap. The old test only counted neighbours, and without the wrap, col 0's offset (+1, −1) aliases onto the col-719 id. It is now checked by offset.

**Suite:** 595 passed, 1 xfailed.

## 2026-11-07: no committed graphics; an explainer notebook instead (#144)

**Simon:** graphics are not committed. Instead, there is a notebook that explains the model and draws the plots when it is run.

- **`notebooks/matryoshka_conflictology.ipynb`** (a new top-level folder) uses the package code throughout, and is committed with its outputs cleared.
  - **Part 1, synthetic, runs in seconds:**
    - the dolls of a cell;
    - the neighbourhood kernel;
    - the doll mix by forecast distance;
    - the month weights;
    - the fit;
    - the sampler against the exact probabilities;
    - the out-of-sample maps against conflictology;
    - the conflictology special case.
  - **Part 2, real data, optional:** the fitted knobs, the out-of-sample P(any) maps against conflictology, and P(> 25 deaths | any).
- **Run 2026-11-07, both parts, state-based for Part 2, as a script in `views_pipeline`:**
  - the knobs equal ADR-024's table;
  - months 457–492: 1,120 cell-months with deaths that conflictology gave exactly 0%, against 0 observed classes that Matryoshka gave < 1e-12.
- **Correction found while writing it.** "The forecast opens outward" holds for state-based and non-state violence. For one-sided violence, the neighbourhood forgets faster (14 months) than the cell (23), so the cell's own share *grows* with distance. The notebook says so.
- **The PR moves to a clean branch from development, without the PNG maps, `.npz` dumps and stray `.pyc` files** (25 + 4 + 8 files, about 20 MB). The prototype branch stays on the remote, so the commit hashes cited here still resolve. The map files named in older entries of this log exist only there.
- **`.gitignore`:** the `**/notebooks/` rule is removed, so `notebooks/` is tracked as in views-datafactory (Simon, 2026-11-07, after comparing views-hydranet, which has the rule and no notebooks, with views-datafactory, which has a tracked `notebooks/` that its ADR-053 cites). Unlike datafactory, outputs are cleared: no graphics. `.pyc` was already ignored; it got in through `git add -f`.
- **Clean branch `feat/127-matryoshka`,** from `origin/development`, with every changed file of the prototype branch except the binaries. It was checked identical file by file; only the 37 binary files differ. 72 files, 676 KB, against about 20 MB.
- **The notebook is now linted.** `notebooks/` is no longer ignored, so ruff checks it. It was formatted with `ruff format`, and three long lines were shortened by hand. Part 1 was re-run after that.

## 2026-11-07: release prepared; stage 4 filed

- **PR #145** (the model, the notebook, no graphics) goes to development; CI is green. #143 is closed.
- **PR #146** (`release/1.3.0`, the bump 1.2.0 → 1.3.0) is to be merged after #145. Goldens pass (5), and the full suite: 595 passed, 1 xfailed.
- **views-models#717** is the stage-4 evaluation model next to `plain_burlap` and `rolling_burlap`. The config drops `window_months` and adds `country_column: c_id`, `backtest_cutoffs: 4`, `backtest_horizon: 36` and `stability_refits: True`. It is not to start before 1.3.0 is on PyPI.

## 2026-11-09: saved artifacts (#148)

The baseline manager refits in evaluate and forecast, and ignores `--artifact_name` (#147, finding 1). views-hydranet, views-r2darts2 and views-stepshifter all load the named artifact, or else the latest one. Decision (Simon): Matryoshka gets one load function in the hydranet style, and the artifact flag is an error for models without an artifact. Filed as #148. It is separate from PR #145.

## 2026-11-09: code review #147 fixed on PR #145

- **Draw (findings 4, 9, 13).** The row search now compares whole numbers: k = u · 2^53 (exact for rng.random) against ceil(cdf · 2^53). So a pick depends only on its own uniform and row, never on its position in a block. The new test caught a first version that rounded the CDF down. The month step reuses `time_weights`. Each sample works out a cell only for the doll it picked. Same synthetic grid (8,000 cells × 36 months × 256 samples): 15.5 s before, 11.7 s after. Real data not re-timed.
- **Checks (2, 5, 6).** Inf and non-integer targets are rejected (in the model and in `classify`). Settings are type-checked at construction; empty targets and `output_length < 1` fail loudly.
- **Logging (3).** INFO per target with the knobs and score; WARNING on non-convergence or floored probabilities.
- **Tests (7, 8, 10, 11).** Per-target knobs; a golden for the draw path; one has-a-country rule (`in_a_country`), with -1 in the fixtures; the C-34 exception pinned in test_dual_input.
- **Docs (12, 14, 15, low).** "Opens outward" is a finding, not a property (one-sided: near memory 14 < cell 23). ADR-002 rule 3 now allows scipy. ADR-024: the history is the data, not a setting. ADR-001 and the Rolling CIC no longer say "only baseline after train_end". CIC cost numbers corrected (616 s, 4.3 GiB, 41 s). output.py docstring. A test that notebooks have no outputs. Input arrays freed before the draw. The unfitted-cell message names the cells.
- Finding 1 moved to #148.

## 2026-11-09: review of the #147 fixes, and a real-data re-timing

- **Reviews of 62bd614** (review-diff, code-review high, falsify guard; clean contexts). No guard was decorative. Fixed in 84fd824:
  - The sampler rejects knobs out of range: a hand-set zero cell weight crashed the draw.
  - Infinite country codes fail; failed stability refits are logged; numpy bools are accepted.
  - The row search finds each chunk's elements by sorting, not with a mask per chunk.
  - Tests closing every real survivor of the guard audit: floored = 1; refit warnings; different y and z values; a golden that reaches every doll, with a √2 neighbour and a country block not at 0; two 0- and two −1-coded cells; no images in notebooks.
  - Each survivor was re-applied and is now caught.
- **Real data** (`results/package_check.json`, input md5 d3636988…):
  - The knobs equal the prototype's v5 exactly (gap 0) for all three types.
  - Fit 594 s, peak 4.3 GiB.
  - Predict, one target, S = 64: 49–52 s new against 86–91 s for the code before #147, timed alternately in one session (`results/predict_timing.json`, `predict_timing.py`). The 61.8 s in `package_check.json` is a separate run straight after the fit; the 41 s of 2026-11-07 was a quieter machine.
- Process note: a mutation script restored files with `git checkout` while fixes were uncommitted, and wiped two files. They were restored from a stash made just before, and checked identical. Mutation checks now run on committed trees only.

## 2026-11-09: second review round (of 84fd824)

- review-diff, code-review high (with the skill's own background pass) and a falsify guard audit, in clean contexts. The code-review worked on an exported copy, so the guard audit's mutations could not reach it. No correctness bug reachable with fitted knobs; no decorative guard.
- **Fixed:**
  - Hand-set knobs under which no doll can be drawn (all weights fade to 0, or their sum overflows) now fail loudly per cell. They used to fall through to the map silently.
  - The knob check now covers the number of knobs, every doll's position, NaN and inf. A zero cell weight is allowed: it is a valid mix.
  - Tested (the code already did it): a refit's own floored count is warned about.
  - The row search skips the sort when the rows are already in order (as in the draw).
  - `stability_refits` is checked, then stored as `bool`.
  - Notebook tests also refuse image links, HTML images and inline SVG.
  - The CIC failure and test tables are complete; the timing has its own source file.
- **Survivors judged equivalent, left alone:** chunk size `2**63 // span` (still fits int64), a smaller chunk (speed only).

## 2026-11-09: third review round (of 49364fb)

- All three reviews: no correctness bug with fitted knobs; no decorative guard. Fixed:
  - Knobs under which no doll can be drawn now fail before predict draws anything. One cell per pattern of empty shells is checked, at most 16.
  - The message names the priogrid id and the months ahead.
  - An infinite length scale is rejected.
  - The dead first assignment of `stability_refits` is removed.
- **Tests strengthened where the audit found them weak:**
  - a zero cell weight and infinite memories are honoured, not only accepted;
  - warning order for a search that is both unconverged and floored;
  - error messages checked on cells whose position differs from their index.
  - The image check is now broad (any image file name, embedding HTML, `url(`, `data:`), with a test that no image file is tracked.
- **Timing:** the CIC says the figures were taken at 84fd824. The script now prints which code it imported.
