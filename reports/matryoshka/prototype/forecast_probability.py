"""From knobs to forecast probabilities, the fitting score and its exact slope (ADR-024), no sampling.

For history end c, forecast month t = c + h and cell i:

  P(doll k | h)       ∝ ring_weight[k] · exp(−h / memory[k])            (empty shells excluded)
  q_k(i, class)       = Σ_s exp(−(c − s)/memory[k]) / Z_k · share of doll k's cells in that class
  P(class | i, h)     = Σ_k P(doll k | h) · q_k(i, class)

The score is the mean log probability of the size class that actually happened (higher is better).
This is exactly the distribution the three-step draw samples from, computed in closed form.

The slope (for the knob search) is taken with respect to log ring_weight[1..D-1] and log memory[0..D-1],
D = len(DOLLS).
With A_k = ring_weight[k] · exp(−h / memory[k]) for a non-empty shell, p = Σ A_k q_k / Σ A_k, and
w_s = exp(−a_s / m) / Z the time weight of a month a_s months before c:

  ∂ log p / ∂ log ring_weight[k] = A_k q_k / num − A_k / den                      (call it T_k)
  ∂ log p / ∂ log memory[k]      = T_k · h / m_k  +  A_k · ∂q_k/∂log m_k / num
  ∂ w_s / ∂ log m                = w_s · (a_s − ā) / m,    ā = Σ_s w_s a_s

so ∂q_k/∂log m_k is the same weighted sum as q_k, with w_s · (a_s − ā) / m in place of w_s.
`check_slope.py` compares this with small nudges.
"""

from __future__ import annotations

import numpy as np
from doll_counts import CELL, COUNTRY, DOLLS, PROVINCE, RING1, RING2, DollCounts
from size_classes import N_CLASSES

PROB_FLOOR = 1e-12   # only to keep the log finite; how often it bites is reported


def _ages(months: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    return (cutoffs[:, None] - months[None, :]).astype(np.float64)


def time_weights(months: np.ndarray, cutoffs: np.ndarray, memory: float,
                 window: int | None = None) -> np.ndarray:
    """(n_cutoffs, M) weights over history months, each row summing to 1 over months ≤ cutoff.

    ``memory = inf`` gives equal weights. ``window`` keeps only the last ``window`` months (used for
    the conflictology reference).
    """
    age = _ages(months, cutoffs)
    keep = age >= 0
    if window is not None:
        keep &= age < window
    w = np.where(keep, 1.0 if np.isinf(memory) else np.exp(-np.maximum(age, 0) / memory), 0.0)
    return (w / w.sum(axis=1, keepdims=True)).astype(np.float32)


def time_weight_slopes(months: np.ndarray, cutoffs: np.ndarray, memory: float) -> np.ndarray:
    """(n_cutoffs, M): ∂ time_weights / ∂ log memory. Each row sums to 0."""
    age = _ages(months, cutoffs)
    keep = age >= 0
    w = np.where(keep, np.exp(-np.maximum(age, 0) / memory), 0.0)
    w /= w.sum(axis=1, keepdims=True)
    mean_age = (w * np.where(keep, age, 0.0)).sum(axis=1, keepdims=True)
    return np.where(keep, w * (age - mean_age) / memory, 0.0).astype(np.float32)


def doll_shares(dc: DollCounts, W: np.ndarray, doll: int) -> np.ndarray:
    """(R, N, K): Σ_s W[r, s] · (share of doll ``doll``'s cells in each class in month s).

    ``W`` holds one row of month weights per output row: `time_weights` gives q_doll, and
    `time_weight_slopes` its slope; stacking both reads the counts once. Shell arithmetic as in
    doll_counts.py.
    """
    R, N = W.shape[0], dc.own.shape[2]
    q = np.zeros((R, N, N_CLASSES), np.float32)
    has_country, has_province = dc.country_index >= 0, dc.province_index >= 0
    ci, pi = np.maximum(dc.country_index, 0), np.maximum(dc.province_index, 0)
    for k in range(N_CLASSES):
        if doll == CELL:
            q[..., k] = W @ dc.own[k]
        elif doll == RING1:
            q[..., k] = W @ dc.ring1[k]
        elif doll == RING2:
            q[..., k] = W @ dc.ring2[k]
        else:
            province_rest = np.where(
                has_province, (W @ dc.province_tot[k])[:, pi] - W @ dc.inner_same_province[k], 0.0)
            if doll == PROVINCE:
                q[..., k] = province_rest
                continue
            country_outside_inner = np.where(
                has_country, (W @ dc.country_tot[k])[:, ci] - W @ dc.inner_same_country[k], 0.0)
            if doll == COUNTRY:
                q[..., k] = country_outside_inner - province_rest
            else:
                inner = W @ dc.own[k] + W @ dc.ring1[k] + W @ dc.ring2[k]
                q[..., k] = (W @ dc.global_tot[k])[:, None] - inner - country_outside_inner
    size = dc.sizes[doll].astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(size[None, :, None] > 0, q / size[None, :, None], 0.0)
    return q


def outcome_classes(classes: np.ndarray, months: np.ndarray, cutoffs: np.ndarray,
                    horizon: int) -> np.ndarray:
    """(n_cutoffs, horizon, N): the observed size class at month c + h, h = 1..horizon."""
    pos = np.searchsorted(months, cutoffs[:, None] + np.arange(1, horizon + 1)[None, :])
    return classes[pos]


def _doll_mix(sizes, H, ring_weight, memory):
    """(D, H, N) A_k: the unnormalised P(doll k | h) per cell, zero for empty shells."""
    h = np.arange(1, H + 1, dtype=np.float64)
    a = ring_weight[:, None] * np.exp(-h[None, :] / memory[:, None])         # (D, H)
    return a[:, :, None] * (sizes > 0)[:, None, :], h


def _observed(q_k: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """(n_cut, H, N): q_k at the class that actually happened."""
    return np.take_along_axis(q_k[:, None, :, :], outcomes[..., None].astype(np.int64),
                              axis=3)[..., 0].astype(np.float64)


def log_score(q: np.ndarray, sizes: np.ndarray, outcomes: np.ndarray, ring_weight: np.ndarray,
              memory: np.ndarray) -> tuple[float, int]:
    """Mean log P(observed class), and how many probabilities hit the floor.

    q: (D, n_cut, N, K) doll shares, D = len(DOLLS); sizes: (D, N); outcomes: (n_cut, H, N);
    ring_weight, memory: (D,) per doll.
    """
    A, _ = _doll_mix(sizes, outcomes.shape[1], ring_weight, memory)
    num = sum(A[k][None] * _observed(q[k], outcomes) for k in range(len(DOLLS)) if ring_weight[k] > 0)
    p = num / A.sum(axis=0)[None]
    return float(np.log(np.maximum(p, PROB_FLOOR)).mean()), int((p < PROB_FLOOR).sum())


def log_score_slope(q: np.ndarray, dq: np.ndarray, sizes: np.ndarray, outcomes: np.ndarray,
                    ring_weight: np.ndarray, memory: np.ndarray) -> tuple[float, int, np.ndarray]:
    """As `log_score`, plus the slope over (log ring_weight[1..D-1], log memory[0..D-1]): (2D − 1,).

    dq: (D, n_cut, N, K), each doll's shares differentiated by its log memory. Floored probabilities
    contribute no slope (the floor is flat).
    """
    A, h = _doll_mix(sizes, outcomes.shape[1], ring_weight, memory)
    D = len(DOLLS)
    P = [_observed(q[k], outcomes) for k in range(D)]
    num = sum(A[k][None] * P[k] for k in range(D))
    den = A.sum(axis=0)[None]
    p = num / den
    ok = p >= PROB_FLOOR
    safe_num = np.where(ok, num, 1.0)
    slope = np.zeros(2 * D - 1)
    for k in range(D):
        T = np.where(ok, A[k][None] * P[k] / safe_num - A[k][None] / den, 0.0)
        if k > 0:
            slope[k - 1] = T.mean()
        via_memory = np.where(ok, A[k][None] * _observed(dq[k], outcomes) / safe_num, 0.0)
        slope[D - 1 + k] = (T * (h / memory[k])[None, :, None] + via_memory).mean()
    return float(np.log(np.maximum(p, PROB_FLOOR)).mean()), int((~ok).sum()), slope
