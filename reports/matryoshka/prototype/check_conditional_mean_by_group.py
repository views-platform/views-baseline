# Quiet-before cells vs active-before cells: observed vs Matryoshka conditional mean (months 457-492).
import sys, json, numpy as np
sys.path.insert(0, ".")
from map_conditional_mean import *
months, cells, country, values = load(sys.argv[1], last_month=CUTOFF + HORIZON)
geo = build_geometry(cells, country)
last36 = (months > CUTOFF - 36) & (months <= CUTOFF); after = (months > CUTOFF) & (months <= CUTOFF + HORIZON)
for t in TARGETS:
    v = values[t]; dc = build_doll_counts(classify(v), months, geo); dv = build_doll_value_sums(v, months, geo, dc.sizes)
    m = matryoshka_conditional_mean(dc, dv, json.loads((KNOBS / f"{t}.json").read_text())["fit"])
    quiet = (v[last36] == 0).all(0); va = v[after]; nz = va > 0
    for name, grp in (("quiet before", quiet), ("active before", ~quiet)):
        cellsel = grp & nz.any(0)
        obs = va[:, grp].sum() / nz[:, grp].sum()
        med_obs = np.median(va[:, grp][nz[:, grp]])
        print(f"{LABEL[t]:12s} {name:13s}: cells with deaths after {cellsel.sum():5d}; observed mean given >0 {obs:7.1f} (median {med_obs:4.0f}); "
              f"Matryoshka forecast in those cells: median {np.nanmedian(m[cellsel]):6.1f}; over all such cells of the group: median {np.nanmedian(m[grp]):6.1f}")
    del dc, dv
