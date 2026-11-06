"""Check that the closed-form class probabilities equal what the three-step draw produces.

Small synthetic grid: draw doll → month → cell many times exactly as ADR-024 describes, and compare
the resulting class frequencies with `forecast_probability`. Run: python check_against_sampling.py
"""

from __future__ import annotations

import numpy as np

from doll_counts import build_doll_counts
from forecast_probability import doll_shares, log_score, outcome_classes
from grid_geometry import N_COLS, build_geometry
from size_classes import N_CLASSES, classify

rng = np.random.default_rng(0)
rows, cols = np.meshgrid(np.arange(100, 108), np.arange(200, 210), indexing="ij")
cells = np.sort((rows * N_COLS + cols + 1).ravel())
country = np.where(((cells - 1) % N_COLS) < 205, 7, 9)
country[3] = 0                                                  # one cell with no country
geo = build_geometry(cells, country)
months = np.arange(100, 160)
values = rng.choice([0, 0, 0, 0, 1, 3, 12, 40, 300, 2000], size=(months.size, cells.size))
classes = classify(values)
dc = build_doll_counts(classes, months, geo)

cutoffs = np.array([150])
memory = np.array([3.0, 6.0, 12.0, 40.0, 200.0])
ring_weight = np.array([1.0, 0.6, 0.3, 0.5, 0.2])
q = np.stack([doll_shares(dc, cutoffs, k, memory[k]) for k in range(5)])
assert np.allclose(q.sum(axis=3)[:, 0][dc.sizes > 0], 1.0, atol=1e-5), "shares must sum to 1"

# shell membership, built independently of doll_counts
r, c = (cells - 1) // N_COLS, (cells - 1) % N_COLS
cheb = np.maximum(abs(r[:, None] - r[None, :]), abs(c[:, None] - c[None, :]))
same = (country[:, None] == country[None, :]) & (country[:, None] > 0)
shells = [cheb == 0, cheb == 1, cheb == 2, same & (cheb > 2), ~same & (cheb > 2)]

i, h, S = 27, 5, 200_000
t = cutoffs[0] + h
hist = months[months <= cutoffs[0]]
a = np.array([ring_weight[k] * np.exp(-h / memory[k]) if shells[k][i].any() else 0.0
              for k in range(5)])
dolls = rng.choice(5, size=S, p=a / a.sum())
draws = np.empty(S, np.int64)
for k in range(5):
    sel = dolls == k
    w = np.exp(-(cutoffs[0] - hist) / memory[k])
    ms = rng.choice(hist.size, size=sel.sum(), p=w / w.sum())
    members = np.nonzero(shells[k][i])[0]
    cs = members[rng.integers(0, members.size, sel.sum())]
    draws[sel] = classes[ms + (hist[0] - months[0]), cs]
freq = np.bincount(draws, minlength=N_CLASSES) / S
closed = (a / a.sum()) @ q[:, 0, i, :]
print("sampled:", freq.round(4))
print("closed :", closed.round(4))
assert np.allclose(freq, closed, atol=4 / np.sqrt(S)), "closed form disagrees with sampling"

obs = outcome_classes(classes, months, cutoffs, 6)
score, floored = log_score(q, dc.sizes, obs, ring_weight, memory)
print(f"log score {score:.4f}, floored {floored}")
print("OK: closed-form class probabilities match the three-step draw")
