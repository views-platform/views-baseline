"""From knobs to forecast probabilities and the fitting score (ADR-024), without sampling.

For history end c, forecast month t = c + h and cell i:

  P(doll k | h)       ∝ ring_weight[k] · exp(−h / memory[k])            (empty shells excluded)
  q_k(i, class)       = Σ_s exp(−(c − s)/memory[k]) / Z_k · share of doll k's cells in that class
  P(class | i, h)     = Σ_k P(doll k | h) · q_k(i, class)

The score is the mean log probability of the size class that actually happened (higher is better).
This is exactly the distribution the three-step draw samples from, computed in closed form.
"""

from __future__ import annotations

import numpy as np

from doll_counts import DollCounts
from size_classes import N_CLASSES

PROB_FLOOR = 1e-12   # only to keep the log finite; how often it bites is reported


def time_weights(months: np.ndarray, cutoffs: np.ndarray, memory: float,
                 window: int | None = None) -> np.ndarray:
    """(n_cutoffs, M) weights over history months, each row summing to 1 over months ≤ cutoff.

    ``memory = inf`` gives equal weights. ``window`` keeps only the last ``window`` months (used for
    the conflictology reference).
    """
    age = cutoffs[:, None] - months[None, :]
    keep = age >= 0
    if window is not None:
        keep &= age < window
    w = np.where(keep, 1.0 if np.isinf(memory) else np.exp(-np.maximum(age, 0) / memory), 0.0)
    return (w / w.sum(axis=1, keepdims=True)).astype(np.float32)


def _weigh(W: np.ndarray, X: np.ndarray, k: int) -> np.ndarray:
    """Σ_s W[c, s] · X[s, ..., k]  for one class k: (n_cutoffs, ...)."""
    xs = X[..., k].reshape(X.shape[0], -1).astype(np.float32)
    return (W @ xs).reshape((W.shape[0],) + X.shape[1:-1])


def doll_shares(dc: DollCounts, cutoffs: np.ndarray, doll: int, memory: float,
                window: int | None = None) -> np.ndarray:
    """q_doll: (n_cutoffs, N, K), the memory-weighted class shares of one doll for every cell."""
    W = time_weights(dc.months, cutoffs, memory, window)
    n_cut, N = W.shape[0], dc.own.shape[1]
    q = np.zeros((n_cut, N, N_CLASSES), np.float32)
    has_country = dc.country_index >= 0
    ci = np.maximum(dc.country_index, 0)
    for k in range(N_CLASSES):
        own = (W @ (dc.own == k).astype(np.float32))                       # (n_cut, N)
        if doll == 0:
            q[..., k] = own
            continue
        r1, r2 = _weigh(W, dc.ring1, k), _weigh(W, dc.ring2, k)
        if doll == 1:
            q[..., k] = r1
        elif doll == 2:
            q[..., k] = r2
        else:
            same = _weigh(W, dc.inner_same, k)
            ctry = np.where(has_country, _weigh(W, dc.country_tot, k)[:, ci] - same, 0.0)
            if doll == 3:
                q[..., k] = ctry
            else:
                glob = (W @ dc.global_tot[:, k].astype(np.float32))[:, None]
                q[..., k] = glob - (own + r1 + r2) - ctry
    size = dc.sizes[doll].astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(size[None, :, None] > 0, q / size[None, :, None], 0.0)
    return q


def outcome_classes(classes: np.ndarray, months: np.ndarray, cutoffs: np.ndarray,
                    horizon: int) -> np.ndarray:
    """(n_cutoffs, horizon, N): the observed size class at month c + h, h = 1..horizon."""
    pos = np.searchsorted(months, cutoffs[:, None] + np.arange(1, horizon + 1)[None, :])
    return classes[pos]


def log_score(q: np.ndarray, sizes: np.ndarray, outcomes: np.ndarray, ring_weight: np.ndarray,
              memory: np.ndarray) -> tuple[float, int]:
    """Mean log P(observed class), and how many probabilities hit the floor.

    q: (5, n_cut, N, K) doll shares; sizes: (5, N); outcomes: (n_cut, H, N);
    ring_weight, memory: (5,) per doll.
    """
    n_cut, H, N = outcomes.shape
    h = np.arange(1, H + 1, dtype=np.float32)
    a = ring_weight[:, None] * np.exp(-h[None, :] / memory[:, None])     # (5, H)
    valid = (sizes > 0).astype(np.float32)                                # (5, N)
    num = np.zeros((n_cut, H, N), np.float32)
    den = np.zeros((H, N), np.float32)
    for k in range(5):
        if ring_weight[k] == 0:
            continue
        pk = np.take_along_axis(q[k][:, None, :, :], outcomes[..., None].astype(np.int64),
                                axis=3)[..., 0]                           # (n_cut, H, N)
        wk = a[k][:, None] * valid[k][None, :]                            # (H, N)
        num += wk[None] * pk
        den += wk
    p = num / den[None]
    floored = int((p < PROB_FLOOR).sum())
    return float(np.log(np.maximum(p, PROB_FLOOR)).mean()), floored
