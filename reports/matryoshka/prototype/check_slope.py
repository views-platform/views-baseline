"""Check that the exact slope of the score equals what small nudges of each knob give.

Small synthetic grid with provinces (as in check_against_sampling.py). For each knob, nudge it up and
down and compare (score_up − score_down) / (2 · nudge) with `log_score_slope`.

In float32 (the prototype's working precision) the nudged values carry rounding noise, so the
tolerance here is loose; `check_slope_float64.py` runs this same file in float64 with a tight one.
Run: python check_slope.py [rtol atol nudge]
"""

from __future__ import annotations

import sys

import numpy as np
from doll_counts import DOLLS, build_doll_counts
from fit_knobs import Objective
from forecast_probability import outcome_classes
from grid_geometry import N_COLS, NO_PROVINCE, build_geometry
from size_classes import classify

RTOL, ATOL, NUDGE = (float(x) for x in sys.argv[1:4]) if len(sys.argv) == 4 else (1e-2, 1e-4, 1e-3)
KNOBS = [f"weight {d}" for d in DOLLS[1:]] + [f"memory {d}" for d in DOLLS]

rng = np.random.default_rng(1)
rows, cols = np.meshgrid(np.arange(100, 110), np.arange(200, 214), indexing="ij")
cells = np.sort((rows * N_COLS + cols + 1).ravel())
r, c = (cells - 1) // N_COLS, (cells - 1) % N_COLS
country = np.where(c < 206, 7, 9)
province = np.where(country == 7, np.where(r < 105, 71, 72), 91)
province[5] = NO_PROVINCE
country[3], province[3] = 0, NO_PROVINCE
geo = build_geometry(cells, country, province)
months = np.arange(100, 190)
values = rng.choice([0, 0, 0, 0, 1, 3, 12, 40, 300, 2000], size=(months.size, cells.size))
classes = classify(values)
dc = build_doll_counts(classes, months, geo)
cutoffs = np.array([150, 162])
obj = Objective(dc, cutoffs, outcome_classes(classes, months, cutoffs, 24))

theta = np.log([0.6, 0.3, 0.4, 0.5, 0.2, 3.0, 6.0, 12.0, 20.0, 40.0, 200.0])
assert theta.size == len(KNOBS)
exact = -obj(theta)[1]
numeric = np.empty(theta.size)
for j in range(theta.size):
    step = NUDGE * np.eye(theta.size)[j]
    numeric[j] = -(obj(theta + step)[0] - obj(theta - step)[0]) / (2 * NUDGE)

for name, e, n in zip(KNOBS, exact, numeric):
    print(f"{name:16s} exact {e: .9f}   nudged {n: .9f}")
gap = np.max(np.abs(exact - numeric) / np.abs(exact))
print(f"largest relative gap {gap:.1e} (rtol {RTOL}, atol {ATOL}, nudge {NUDGE})")
assert np.allclose(exact, numeric, rtol=RTOL, atol=ATOL), "exact slope disagrees with nudges"
print("OK: exact slope matches the nudges")
