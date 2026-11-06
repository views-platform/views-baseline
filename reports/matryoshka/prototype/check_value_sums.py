"""Check doll_value_sums against doll_counts: with value = 1(class == j), the doll's mean value must
equal its share of class j, for every doll and class. Run: python check_value_sums.py
"""

from __future__ import annotations

import numpy as np
from doll_counts import DOLLS, build_doll_counts
from doll_value_sums import build_doll_value_sums, doll_mean_values
from forecast_probability import doll_shares, time_weights
from grid_geometry import N_COLS, NO_PROVINCE, build_geometry
from size_classes import N_CLASSES, classify

rng = np.random.default_rng(2)
rows, cols = np.meshgrid(np.arange(100, 114), np.arange(200, 218), indexing="ij")
cells = np.sort((rows * N_COLS + cols + 1).ravel())
c = (cells - 1) % N_COLS
r = (cells - 1) // N_COLS
country = np.where(c < 208, 7, 9)
province = np.where(country == 7, np.where(r < 107, 71, 72), 91)
province[5] = NO_PROVINCE
country[3], province[3] = 0, NO_PROVINCE
geo = build_geometry(cells, country, province)
months = np.arange(100, 160)
classes = classify(rng.choice([0, 0, 0, 1, 3, 12, 40, 300, 2000], size=(months.size, cells.size)))
dc = build_doll_counts(classes, months, geo)
W = time_weights(months, np.array([140, 150]), 9.0)
length = 1.7
for j in range(N_CLASSES):
    dv = build_doll_value_sums((classes == j).astype(float), months, geo, dc.band_sizes, dc.sizes)
    for doll in range(len(DOLLS)):
        assert np.allclose(doll_mean_values(dv, W, doll, length),
                           doll_shares(dc, W, doll, length)[..., j], atol=1e-10), (j, doll)
print("OK: value sums reproduce the class shares for every doll and class")
