"""Numerical check of the ADR-024 weight formula on a small synthetic grid.

w(k, s; t) = ring_weight[k] * exp(-(t - c)/m_k) * [exp(-(c - s)/m_k) / Z_k] / n_k(s)
Z_k = sum_{s in history} exp(-(c - s)/m_k)
"""
import itertools

import numpy as np

rng = np.random.default_rng(0)
ROWS, COLS, H0 = 9, 9, 121           # small grid; history months H0..c
c = 200                               # history ends here
months = np.arange(H0, c + 1)
country = np.where(np.arange(COLS)[None, :] < 6, 1, 2) * np.ones((ROWS, 1), int)
country[0, 0] = 0                     # a cell with no country
Y = (rng.random((ROWS, COLS, len(months))) < 0.1) * rng.integers(1, 50, (ROWS, COLS, len(months)))
target = (4, 4)                       # the forecast cell


def shells(cell):
    """Disjoint shells: cell, ring 1, ring 2, rest of country, rest of map."""
    r0, c0 = cell
    cheb = np.maximum(abs(np.arange(ROWS)[:, None] - r0), abs(np.arange(COLS)[None, :] - c0))
    cell_m = cheb == 0
    ring1, ring2 = cheb == 1, cheb == 2
    inner = cheb <= 2
    ctry = (country == country[cell]) & ~inner if country[cell] > 0 else np.zeros_like(inner)
    world = ~inner & ~ctry
    return [cell_m, ring1, ring2, ctry, world]


def weights(cell, t, rw, mem, history=months):
    """Joint weight of every (shell k, month s, cell i) under the ADR formula."""
    S = shells(cell)
    out = {}
    for k, (mask, w_k, m_k) in enumerate(zip(S, rw, mem)):
        n = int(mask.sum())
        if n == 0 or w_k == 0:
            continue
        age_c = c - history
        kern = np.exp(-age_c / m_k) if np.isfinite(m_k) else np.ones_like(age_c, float)
        Z = kern.sum()
        lead = np.exp(-(t - c) / m_k) if np.isfinite(m_k) else 1.0
        for j, s in enumerate(history):
            for i in zip(*np.nonzero(mask)):
                out[(k, s, i)] = w_k * lead * kern[j] / Z / n
    return out, S


rw = [1.0, 0.6, 0.25, 0.3, 0.05]
mem = [6.0, 12.0, 24.0, 60.0, 120.0]

# 1. A doll's total say = ring_weight * exp(-(t-c)/m), independent of memory length and history length
for t in (c + 1, c + 12, c + 36):
    W, _ = weights(target, t, rw, mem)
    for k in range(5):
        tot = sum(v for (kk, _, _), v in W.items() if kk == k)
        assert np.isclose(tot, rw[k] * np.exp(-(t - c) / mem[k])), (t, k, tot)
for short_hist in (months[-36:], months[-120:]):          # change the history length
    W, _ = weights(target, c + 5, rw, mem, history=short_hist)
    tot0 = sum(v for (kk, _, _), v in W.items() if kk == 0)
    assert np.isclose(tot0, rw[0] * np.exp(-5 / mem[0]))
print("1 ok: doll total = ring_weight * exp(-(t-c)/memory), independent of memory/history length")

# 2. Opening outward: shares of inner dolls fall with lead when inner memory is shorter
shares = []
for L in (1, 12, 36, 60):
    W, _ = weights(target, c + L, rw, mem)
    tot = np.array([sum(v for (kk, _, _), v in W.items() if kk == k) for k in range(5)])
    shares.append(tot / tot.sum())
shares = np.array(shares)
assert np.all(np.diff(shares[:, 0]) < 0) and np.all(np.diff(shares[:, 4]) > 0)
print("2 ok: cell share", shares[:, 0].round(3), "| world share", shares[:, 4].round(3))

# 3. Shells are disjoint and cover the map
S = shells(target)
assert (sum(m.astype(int) for m in S) == 1).all()
S0 = shells((0, 0))                                      # no-country cell: country shell empty
assert S0[3].sum() == 0
print("3 ok: shells disjoint and cover the grid; no-country cell has an empty country shell")

# 4. Special case = conflictology: only the cell, no fading, 36-month history -> uniform over last 36 values
W, _ = weights(target, c + 7, [1, 0, 0, 0, 0], [np.inf] * 5, history=months[-36:])
p = np.array(list(W.values())) / sum(W.values())
assert np.allclose(p, 1 / 36) and len(p) == 36
vals = [Y[target][s - H0] for (_, s, _) in W]
assert sorted(vals) == sorted(Y[target][-36:])
print("4 ok: special case = uniform draw over the cell's last 36 values (conflictology)")

# 5. Sequential sampling (doll -> month -> cell) gives the same probabilities as the joint weights
t = c + 10
W, S = weights(target, t, rw, mem)
Ztot = sum(W.values())
for (k, s, i), v in itertools.islice(W.items(), 0, None, 997):
    m_k = mem[k]
    kern = np.exp(-(c - months) / m_k)
    p_doll = rw[k] * np.exp(-(t - c) / m_k) / sum(rw[j] * np.exp(-(t - c) / mem[j]) for j in range(5)
                                                  if S[j].sum() and rw[j])
    p_month = np.exp(-(c - s) / m_k) / kern.sum()
    p_cell = 1 / S[k].sum()
    assert np.isclose(v / Ztot, p_doll * p_month * p_cell)
print("5 ok: P(draw) = P(doll) * P(month | doll) * P(cell | doll, month)")

# 6. The old formula (no Z_k) confounds memory with weight: doll total ~ ring_weight * memory
def old_total(m, H):
    a = np.arange(H)
    return np.exp(-a / m).sum()
print("6 old formula, total say of a doll with weight 1 (H = 336 months):",
      {m: round(old_total(m, 336), 1) for m in (4, 12, 120)})
