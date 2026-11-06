# Dump Matryoshka's chance of >=1 death and conditional mean (months 457-492) for one prototype version.
# usage: python dump_version.py <proto_dir> <knobs_dir> <out.npz> <input.parquet> [gaul1.parquet]
import sys, json, inspect, numpy as np
proto, knobs_dir, out, inp = sys.argv[1:5]
sys.path.insert(0, proto)
from fit_knobs import TARGETS, load
from grid_geometry import build_geometry
from doll_counts import build_doll_counts
from doll_value_sums import build_doll_value_sums
from size_classes import classify
from map_forecasts import matryoshka_any_death
from map_conditional_mean import CUTOFF, HORIZON, matryoshka_conditional_mean
if len(sys.argv) > 5:
    months, cells, country, province, values = load(inp, sys.argv[5], last_month=CUTOFF + HORIZON)
    geo = build_geometry(cells, country, province)
else:
    months, cells, country, values = load(inp, last_month=CUTOFF + HORIZON)
    geo = build_geometry(cells, country)
res = {"cells": cells}
for t in TARGETS:
    v = values[t]; dc = build_doll_counts(classify(v), months, geo)
    dv = build_doll_value_sums(v, months, geo, dc.sizes)
    k = json.load(open(f"{knobs_dir}/{t}.json"))["fit"]
    res[f"{t}_p"] = matryoshka_any_death(dc, k); res[f"{t}_m"] = matryoshka_conditional_mean(dc, dv, k)
    del dc, dv
np.savez(out, **res); print("saved", out)
