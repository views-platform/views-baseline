"""Per-cell pull of the province vs the country group, and the Egypt one-sided split (#136).

Run: python check_province_pull.py <input.parquet> <gaul1_code.parquet>  (knobs from ../results/v4_province)
"""
# Per-cell pull of the province vs the country group: (w_province / n_province) / (w_country / n_country),
# for cells where both shells are non-empty; plus Egypt one-sided conditional mean, North Sinai vs elsewhere.
import json
import sys

import numpy as np

sys.path.insert(0, ".")

from doll_counts import COUNTRY, PROVINCE, build_doll_counts
from doll_value_sums import build_doll_value_sums
from fit_knobs import TARGETS, load
from grid_geometry import build_geometry
from map_conditional_mean import CUTOFF, HORIZON, matryoshka_conditional_mean
from size_classes import classify

months, cells, country, province, values = load(sys.argv[1], sys.argv[2], last_month=CUTOFF + HORIZON)
geo = build_geometry(cells, country, province)
for t in TARGETS:
    v = values[t]; dc = build_doll_counts(classify(v), months, geo)
    w = np.array(json.load(open(f"../results/v4_province/{t}.json"))["fit"]["weight"])
    both = (dc.sizes[PROVINCE] > 0) & (dc.sizes[COUNTRY] > 0)
    ratio = (w[PROVINCE] / dc.sizes[PROVINCE][both]) / (w[COUNTRY] / dc.sizes[COUNTRY][both])
    print(f"{t}: cells with both shells {both.sum()}; per-cell pull province/country: median {np.median(ratio):.2f}, "
          f"10–90% {np.percentile(ratio,10):.2f}–{np.percentile(ratio,90):.2f}; median shell sizes province {np.median(dc.sizes[PROVINCE][both]):.0f}, country {np.median(dc.sizes[COUNTRY][both]):.0f}")
    if t == "lr_os_best":
        dv = build_doll_value_sums(v, months, geo, dc.sizes)
        m = matryoshka_conditional_mean(dc, dv, json.load(open(f"../results/v4_province/{t}.json"))["fit"])
        eg = country == 120; sinai = eg & (province == province[cells == 174667][0])
        print(f"  Egypt one-sided conditional mean: North Sinai median {np.nanmedian(m[sinai]):.0f}, rest of Egypt median {np.nanmedian(m[eg & ~sinai]):.0f}")
    del dc
