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
