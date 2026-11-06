"""Check that the exact slope of the score equals what small nudges of each knob give.

Small synthetic grid (as in check_against_sampling.py). For each of the 9 knobs, nudge it up and down
by a little and compare (score_up − score_down) / (2 · nudge) with `log_score_slope`.
Run: python check_slope.py
"""

from __future__ import annotations

import numpy as np

from doll_counts import build_doll_counts
from fit_knobs import Objective
from forecast_probability import outcome_classes
from grid_geometry import N_COLS, build_geometry
from size_classes import classify

KNOBS = ["w ring1", "w ring2", "w country", "w map",
         "m cell", "m ring1", "m ring2", "m country", "m map"]

rng = np.random.default_rng(1)
rows, cols = np.meshgrid(np.arange(100, 110), np.arange(200, 214), indexing="ij")
cells = np.sort((rows * N_COLS + cols + 1).ravel())
country = np.where(((cells - 1) % N_COLS) < 206, 7, 9)
country[3] = 0                                                  # one cell with no country
geo = build_geometry(cells, country)
months = np.arange(100, 190)
values = rng.choice([0, 0, 0, 0, 1, 3, 12, 40, 300, 2000], size=(months.size, cells.size))
classes = classify(values)
dc = build_doll_counts(classes, months, geo)
cutoffs = np.array([150, 162])
obj = Objective(dc, cutoffs, outcome_classes(classes, months, cutoffs, 24))

theta = np.log([0.6, 0.3, 0.5, 0.2, 3.0, 6.0, 12.0, 40.0, 200.0])
_, minus_slope = obj(theta)
nudge = 1e-3
numeric = np.empty(9)
for j in range(9):
    up, down = theta.copy(), theta.copy()
    up[j] += nudge
    down[j] -= nudge
    numeric[j] = -(obj(up)[0] - obj(down)[0]) / (2 * nudge)

exact = -minus_slope
for name, e, n in zip(KNOBS, exact, numeric):
    print(f"{name:10s} exact {e: .6f}   nudged {n: .6f}")
assert np.allclose(exact, numeric, rtol=1e-2, atol=1e-4), "exact slope disagrees with nudges"
print("OK: exact slope matches the nudges")
