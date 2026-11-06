"""Check the shells and the closed-form class probabilities against an independent construction.

Synthetic grid (16 × 22 cells, two countries, three provinces, one cell without a province, one
without a country, a few sea holes so that
neighbourhoods are incomplete). Shell membership and the neighbourhood kernel are built here directly
from pairwise distances (brute force over all cell pairs), independently of grid_geometry's offset
table and doll_counts' bands. Checks:
  1. the five shells are disjoint and together cover every cell, for every cell;
  2. their sizes equal DollCounts.sizes;
  3. every doll's class shares equal a direct (kernel-weighted for NEAR) average over that doll's
     cells, for every cell;
  4. drawing doll → month → cell many times, as ADR-024 describes (neighbour ∝ exp(−d/ℓ)),
     reproduces the closed-form probabilities, for a cell with a full neighbourhood and a corner cell.
Run: python check_against_sampling.py
"""

from __future__ import annotations

import numpy as np
from doll_counts import DOLLS, NEAR, build_doll_counts
from forecast_probability import doll_shares, log_score, outcome_classes, time_weights
from grid_geometry import N_COLS, NEAR_RADIUS, NO_PROVINCE, build_geometry
from size_classes import N_CLASSES, classify

rng = np.random.default_rng(0)
rows, cols = np.meshgrid(np.arange(100, 116), np.arange(200, 222), indexing="ij")
cells = (rows * N_COLS + cols + 1).ravel()
holes = np.isin(cells, cells[[40, 41, 63, 150, 151, 200]])               # sea: not land cells
cells = np.sort(cells[~holes])
r, c = (cells - 1) // N_COLS, (cells - 1) % N_COLS
country = np.where(c < 211, 7, 9)
province = np.where(country == 7, np.where(r < 108, 71, 72), 91)       # two provinces in 7, one in 9
province[5] = NO_PROVINCE                                               # a cell without a province
country[3], province[3] = 0, NO_PROVINCE                                # a cell without a country
geo = build_geometry(cells, country, province)
months = np.arange(100, 160)
values = rng.choice([0, 0, 0, 0, 1, 3, 12, 40, 300, 2000], size=(months.size, cells.size))
classes = classify(values)
dc = build_doll_counts(classes, months, geo)
length = 1.7

# 1–2. independent shells
dist = np.hypot(r[:, None] - r[None, :], c[:, None] - c[None, :])
near = (dist > 0) & (dist <= NEAR_RADIUS)
same_c = (country[:, None] == country[None, :]) & (country[:, None] > 0)
same_p = (province[:, None] == province[None, :]) & (province[:, None] != NO_PROVINCE)
inner = (dist == 0) | near
shells = np.stack([dist == 0, near, same_p & ~inner, same_c & ~same_p & ~inner,
                   ~same_c & ~inner])                                     # (5, N, N)
assert len(shells) == len(DOLLS)
assert (shells.sum(axis=0) == 1).all(), "shells must be disjoint and cover every cell"
assert (shells.sum(axis=2) == dc.sizes).all(), "shell sizes disagree with DollCounts.sizes"
assert (dc.sizes[2] > 0).any() and (dc.sizes[2] == 0).any(), "grid must exercise the province shell"
pick = np.stack([shells[0], np.where(near, np.exp(-dist / length), 0.0), *shells[2:]]).astype(float)
pick = pick / np.maximum(pick.sum(axis=2, keepdims=True), 1e-300)      # P(cell j | doll, cell i)

# 3. shares equal direct (kernel-)weighted averages over each shell
cutoffs = np.array([150])
memory = np.array([3.0, 8.0, 20.0, 40.0, 200.0])
W = np.stack([time_weights(months, cutoffs, memory[k]) for k in range(len(DOLLS))])   # (4, 1, M)
q = np.stack([doll_shares(dc, W[k], k, length) for k in range(len(DOLLS))])          # (4, 1, N, K)
onehot = (classes[..., None] == np.arange(N_CLASSES)).astype(float)                   # (M, N, K)
for k in range(len(DOLLS)):
    direct = np.einsum("m,ij,mjk->ik", W[k, 0], pick[k], onehot)
    assert np.allclose(q[k, 0], direct, atol=1e-10), f"doll {DOLLS[k]} shares disagree"
assert np.allclose(q.sum(axis=3)[:, 0][dc.sizes > 0], 1.0, atol=1e-10), "shares must sum to 1"

# 4. sampling
ring_weight = np.array([1.0, 0.8, 0.3, 0.4, 0.2])
h, S = 5, 200_000
hist = months[months <= cutoffs[0]]
for i in (int(np.argmax(dc.sizes[NEAR])), int(np.argmin(dc.sizes[NEAR]))):   # full, and corner
    a = np.array([ring_weight[k] * np.exp(-h / memory[k]) if shells[k][i].any() else 0.0
                  for k in range(len(DOLLS))])
    dolls = rng.choice(len(DOLLS), size=S, p=a / a.sum())
    draws = np.empty(S, np.int64)
    for k in range(len(DOLLS)):
        sel = dolls == k
        if not sel.any():
            continue
        w = np.exp(-(cutoffs[0] - hist) / memory[k])
        ms = rng.choice(hist.size, size=sel.sum(), p=w / w.sum())
        cs = rng.choice(cells.size, size=sel.sum(), p=pick[k][i])
        draws[sel] = classes[ms + (hist[0] - months[0]), cs]
    freq = np.bincount(draws, minlength=N_CLASSES) / S
    closed = (a / a.sum()) @ q[:, 0, i, :]
    print(f"cell {i} (neighbourhood size {dc.sizes[NEAR, i]}):")
    print("  sampled:", freq.round(4))
    print("  closed :", closed.round(4))
    assert np.allclose(freq, closed, atol=4 / np.sqrt(S)), "closed form disagrees with sampling"

obs = outcome_classes(classes, months, cutoffs, 6)
score, floored = log_score(q, dc.sizes, obs, ring_weight, memory)
print(f"log score {score:.4f}, floored {floored}")
print("OK: shells disjoint and complete, sizes and kernel-weighted shares match a direct "
      "construction, and the closed form matches the three-step draw")
