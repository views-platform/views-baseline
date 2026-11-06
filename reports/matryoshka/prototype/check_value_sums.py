"""Check doll_value_sums against doll_counts: with value = 1(class == j), the doll's mean value must
equal its share of class j, for every doll and class. Run: python check_value_sums.py
"""

from __future__ import annotations

import numpy as np

from doll_counts import build_doll_counts
from doll_value_sums import build_doll_value_sums, doll_mean_values
from forecast_probability import doll_shares, time_weights
from grid_geometry import N_COLS, build_geometry
from size_classes import N_CLASSES, classify

rng = np.random.default_rng(2)
rows, cols = np.meshgrid(np.arange(100, 109), np.arange(200, 212), indexing="ij")
cells = np.sort((rows * N_COLS + cols + 1).ravel())
country = np.where(((cells - 1) % N_COLS) < 205, 7, 9)
country[3] = 0
geo = build_geometry(cells, country)
months = np.arange(100, 160)
classes = classify(rng.choice([0, 0, 0, 1, 3, 12, 40, 300, 2000], size=(months.size, cells.size)))
dc = build_doll_counts(classes, months, geo)
W = time_weights(months, np.array([140, 150]), 9.0)
for j in range(N_CLASSES):
    dv = build_doll_value_sums((classes == j).astype(float), months, geo, dc.sizes)
    for doll in range(5):
        assert np.allclose(doll_mean_values(dv, W, doll), doll_shares(dc, W, doll)[..., j],
                           atol=1e-5), (j, doll)
print("OK: value sums reproduce the class shares for every doll and class")
