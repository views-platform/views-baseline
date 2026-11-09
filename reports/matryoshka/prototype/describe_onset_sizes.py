"""Sizes of outbreaks versus all violent cell-months, as input to the #137 design.

Onset: a non-zero cell-month whose cell had no deaths in the preceding WINDOW months.
Remote onset: an onset whose land neighbourhood (grid distance ≤ NEAR_RADIUS) also had no deaths in
those months. Months before HISTORY_START + WINDOW cannot be classified, so periods start there.
Training data only (≤ TRAIN_END).
Run: python describe_onset_sizes.py <input.parquet> <gaul1_code.parquet>
"""

from __future__ import annotations

import sys

import numpy as np
from fit_knobs import load
from grid_geometry import build_geometry

WINDOW = 36
PERIODS = [(157, 456), (421, 456)]


def describe(x: np.ndarray) -> str:
    return (f"n={x.size:6d} mean={x.mean():6.1f} median={np.median(x):4.0f} "
            f"p90={np.quantile(x, 0.9):5.0f}")


def main(input_path: str, province_path: str) -> None:
    months, cells, country, province, values = load(input_path, province_path)
    nbr = build_geometry(cells, country, province).neighbours
    for target, V in values.items():
        V = np.asarray(V, dtype=float)
        M, N = V.shape
        nz = V > 0
        cum = np.vstack([np.zeros((1, N)), np.cumsum(nz, axis=0)])
        prior = np.zeros_like(V)                       # non-zero months in the preceding window
        prior[WINDOW:] = cum[WINDOW:M] - cum[: M - WINDOW]
        padded = np.concatenate([prior, np.zeros((M, 1))], axis=1)
        idx = np.where(nbr >= 0, nbr, N)
        near_prior = sum(padded[:, idx[:, o]] for o in range(nbr.shape[1]))
        print(target)
        for lo, hi in PERIODS:
            rows = (months >= lo) & (months <= hi)
            v, n, p, q = V[rows], nz[rows], prior[rows], near_prior[rows]
            print(f"  months {lo}-{hi}: all    ", describe(v[n]))
            print(f"  months {lo}-{hi}: onset  ", describe(v[n & (p == 0)]))
            print(f"  months {lo}-{hi}: remote ", describe(v[n & (p == 0) & (q == 0)]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
