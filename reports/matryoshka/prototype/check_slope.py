"""Check that the exact slope of the score equals what small nudges of each knob give.

Small synthetic grid (two countries, one cell without a country). For each knob, nudge it up and
down and compare (score_up − score_down) / (2 · nudge) with `log_score_slope`.

The prototype works in float64 (sparse counts, #139), so the nudges are precise and the default
tolerance is tight (rtol 1e-6, no absolute slack). This replaces check_slope_float64.py.
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

RTOL, ATOL, NUDGE = (float(x) for x in sys.argv[1:4]) if len(sys.argv) == 4 else (1e-6, 0.0, 1e-4)
KNOBS = [f"weight {d}" for d in DOLLS[1:]] + [f"memory {d}" for d in DOLLS] + ["length scale"]

rng = np.random.default_rng(1)
rows, cols = np.meshgrid(np.arange(100, 114), np.arange(200, 220), indexing="ij")
cells = np.sort((rows * N_COLS + cols + 1).ravel())
c = (cells - 1) % N_COLS
country = np.where(c < 209, 7, 9)
country[3] = 0
geo = build_geometry(cells, country, np.full(cells.size, NO_PROVINCE))
months = np.arange(100, 190)
values = rng.choice([0, 0, 0, 0, 1, 3, 12, 40, 300, 2000], size=(months.size, cells.size))
classes = classify(values)
dc = build_doll_counts(classes, months, geo)
cutoffs = np.array([150, 162])
obj = Objective(dc, cutoffs, outcome_classes(classes, months, cutoffs, 24))

theta = np.log([0.8, 0.5, 0.2, 3.0, 8.0, 40.0, 200.0, 1.7])
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
