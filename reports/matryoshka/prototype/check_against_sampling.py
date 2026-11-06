"""Check the shells and the closed-form class probabilities against an independent construction.

Small synthetic grid with two countries, provinces inside them, one cell without a province and one
without a country. Shell membership is built here directly from grid distance, province and country,
independently of doll_counts. Checks:
  1. the six shells are disjoint and together cover every cell, for every cell;
  2. their sizes equal DollCounts.sizes;
  3. every doll's class shares equal a direct average over that doll's cells, for every cell;
  4. drawing doll → month → cell many times, as ADR-024 describes, reproduces the closed-form
     probabilities for a cell with a province and for one without.
Run: python check_against_sampling.py
"""

from __future__ import annotations

import numpy as np
from doll_counts import DOLLS, build_doll_counts
from forecast_probability import doll_shares, log_score, outcome_classes, time_weights
from grid_geometry import N_COLS, NO_PROVINCE, build_geometry
from size_classes import N_CLASSES, classify

rng = np.random.default_rng(0)
rows, cols = np.meshgrid(np.arange(100, 108), np.arange(200, 212), indexing="ij")
cells = np.sort((rows * N_COLS + cols + 1).ravel())
r, c = (cells - 1) // N_COLS, (cells - 1) % N_COLS
country = np.where(c < 206, 7, 9)
province = np.where(country == 7, np.where(r < 104, 71, 72), 91)      # two provinces in 7, one in 9
province[5] = NO_PROVINCE                                              # a cell without a province
country[3], province[3] = 0, NO_PROVINCE                               # a cell without a country
geo = build_geometry(cells, country, province)
months = np.arange(100, 160)
values = rng.choice([0, 0, 0, 0, 1, 3, 12, 40, 300, 2000], size=(months.size, cells.size))
classes = classify(values)
dc = build_doll_counts(classes, months, geo)

# 1–2. independent shells
cheb = np.maximum(abs(r[:, None] - r[None, :]), abs(c[:, None] - c[None, :]))
same_c = (country[:, None] == country[None, :]) & (country[:, None] > 0)
same_p = (province[:, None] == province[None, :]) & (province[:, None] != NO_PROVINCE)
outside = cheb > 2
shells = np.stack([cheb == 0, cheb == 1, cheb == 2, same_p & outside,
                   same_c & ~same_p & outside, ~same_c & outside])          # (6, N, N)
assert len(shells) == len(DOLLS)
assert (shells.sum(axis=0) == 1).all(), "shells must be disjoint and cover every cell"
assert (shells.sum(axis=2) == dc.sizes).all(), "shell sizes disagree with DollCounts.sizes"
assert (dc.sizes[3] > 0).any() and (dc.sizes[3] == 0).any(), "grid must exercise the province shell"

# 3. shares equal direct averages over each shell
cutoffs = np.array([150])
memory = np.array([3.0, 6.0, 12.0, 20.0, 40.0, 200.0])
W = np.stack([time_weights(months, cutoffs, memory[k]) for k in range(len(DOLLS))])   # (6, 1, M)
q = np.stack([doll_shares(dc, W[k], k) for k in range(len(DOLLS))])                  # (6, 1, N, K)
onehot = (classes[..., None] == np.arange(N_CLASSES)).astype(float)                   # (M, N, K)
for k in range(len(DOLLS)):
    size = shells[k].sum(axis=1)
    direct = np.einsum("m,ij,mjk->ik", W[k, 0], shells[k].astype(float), onehot)
    direct = np.where(size[:, None] > 0, direct / np.maximum(size, 1)[:, None], 0.0)
    assert np.allclose(q[k, 0], direct, atol=1e-5), f"doll {DOLLS[k]} shares disagree"
assert np.allclose(q.sum(axis=3)[:, 0][dc.sizes > 0], 1.0, atol=1e-5), "shares must sum to 1"

# 4. sampling, for a cell with a non-empty province shell and for the cell without a province
ring_weight = np.array([1.0, 0.6, 0.3, 0.4, 0.5, 0.2])
h, S = 5, 200_000
hist = months[months <= cutoffs[0]]
for i in (int(np.argmax(dc.sizes[3])), 5):
    a = np.array([ring_weight[k] * np.exp(-h / memory[k]) if shells[k][i].any() else 0.0
                  for k in range(len(DOLLS))])
    dolls = rng.choice(len(DOLLS), size=S, p=a / a.sum())
    draws = np.empty(S, np.int64)
    for k in range(len(DOLLS)):
        sel = dolls == k
        w = np.exp(-(cutoffs[0] - hist) / memory[k])
        ms = rng.choice(hist.size, size=sel.sum(), p=w / w.sum())
        members = np.nonzero(shells[k][i])[0]
        if sel.any():
            cs = members[rng.integers(0, members.size, sel.sum())]
            draws[sel] = classes[ms + (hist[0] - months[0]), cs]
    freq = np.bincount(draws, minlength=N_CLASSES) / S
    closed = (a / a.sum()) @ q[:, 0, i, :]
    print(f"cell {i} (province shell size {dc.sizes[3, i]}):")
    print("  sampled:", freq.round(4))
    print("  closed :", closed.round(4))
    assert np.allclose(freq, closed, atol=4 / np.sqrt(S)), "closed form disagrees with sampling"

obs = outcome_classes(classes, months, cutoffs, 6)
score, floored = log_score(q, dc.sizes, obs, ring_weight, memory)
print(f"log score {score:.4f}, floored {floored}")
print("OK: shells disjoint and complete, sizes and shares match a direct construction, "
      "and the closed form matches the three-step draw")
