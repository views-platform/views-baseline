"""Size check: observed vs Matryoshka conditional mean, for cells quiet before vs active before.

Data in:   the calibration input pull; GAUL admin-1 per cell; fitted knobs in <knobs_dir>/{target}.json.
Transform: "quiet before" = no deaths in the 36 months up to CUTOFF (456). For months 457–492 (out of
           sample): the observed mean given at least one death (pooled; and the median), next to
           Matryoshka's conditional mean (median over the cells that had deaths, and over the group).
Out:       printed table (LOG.md 2026-11-06).

Run: python check_conditional_mean_by_group.py <input.parquet> <gaul1_code.parquet> <knobs_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from doll_counts import build_doll_counts
from doll_value_sums import build_doll_value_sums
from fit_knobs import TARGETS, load
from grid_geometry import build_geometry
from map_conditional_mean import CUTOFF, HORIZON, LABEL, matryoshka_conditional_mean
from size_classes import classify


def main(path: str, province_path: str, knobs_dir: Path) -> None:
    months, cells, country, province, values = load(path, province_path,
                                                    last_month=CUTOFF + HORIZON)
    geo = build_geometry(cells, country, province)
    last36 = (months > CUTOFF - 36) & (months <= CUTOFF)
    after = (months > CUTOFF) & (months <= CUTOFF + HORIZON)
    for target in TARGETS:
        v = values[target]
        dc = build_doll_counts(classify(v), months, geo)
        dv = build_doll_value_sums(v, months, geo, dc.sizes)
        knobs = json.loads((knobs_dir / f"{target}.json").read_text())["fit"]
        forecast = matryoshka_conditional_mean(dc, dv, knobs)
        quiet = (v[last36] == 0).all(0)
        va = v[after]
        nz = va > 0
        for name, group in (("quiet before", quiet), ("active before", ~quiet)):
            had_deaths = group & nz.any(0)
            observed = va[:, group].sum() / nz[:, group].sum()
            observed_median = np.median(va[:, group][nz[:, group]])
            print(f"{LABEL[target]:12s} {name:13s}: cells with deaths after {had_deaths.sum():5d}; "
                  f"observed mean given >0 {observed:7.1f} (median {observed_median:4.0f}); "
                  f"Matryoshka in those cells: median {np.nanmedian(forecast[had_deaths]):6.1f}; "
                  f"over the whole group: median {np.nanmedian(forecast[group]):6.1f}", flush=True)
        del dc, dv


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
