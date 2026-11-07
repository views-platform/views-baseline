"""From knobs to forecast probabilities, the fitting score and its exact slope (ADR-024).

No sampling: everything here is exact.

For history end c, forecast month t = c + h and cell i:

  P(doll k | h)       ∝ weight[k] · exp(−h / memory[k])                 (empty shells excluded)
  q_k(i, class)       = Σ_s exp(−(c − s)/memory[k]) / Z_k · share of doll k's cells in that class
  P(class | i, h)     = Σ_k P(doll k | h) · q_k(i, class)

The score is the mean log probability of the size class that actually happened (higher is better).
This is exactly the distribution the three-step draw samples from, computed in closed form.

The neighbourhood doll draws neighbour j with probability ∝ exp(−d_ij / ℓ) instead of uniformly.
Neighbours are grouped in distance bands b (distance d_b, n_b(i) cells), so with
a_b = exp(−d_b/ℓ) and X_b the band's class counts:

  q_near(i, class) = Σ_b a_b · (time-weighted X_b)(i, class) / Σ_b a_b · n_b(i)

The slope (for the knob search) is taken with respect to log weight[1..D-1] and
log memory[0..D-1], D = len(DOLLS).
With A_k = weight[k] · exp(−h / memory[k]) for a non-empty shell, p = Σ A_k q_k / Σ A_k, and
w_s = exp(−a_s / m) / Z the time weight of a month a_s months before c:

  ∂ log p / ∂ log weight[k] = A_k q_k / num − A_k / den                           (call it T_k)
  ∂ log p / ∂ log memory[k]      = T_k · h / m_k  +  A_k · ∂q_k/∂log m_k / num
  ∂ w_s / ∂ log m                = w_s · (a_s − ā) / m,    ā = Σ_s w_s a_s

so ∂q_k/∂log m_k is the same weighted sum as q_k, with w_s · (a_s − ā) / m in place of w_s. For the
length scale, with a'_b = ∂a_b/∂log ℓ = a_b · d_b / ℓ:

  ∂q_near/∂log ℓ = Σ_b a'_b X_b / Σ_b a_b n_b − q_near · Σ_b a'_b n_b / Σ_b a_b n_b
  ∂ log p / ∂ log ℓ = A_near · ∂q_near/∂log ℓ / num

tests/test_matryoshka.py compares this with small nudges (the prototype's check_slope.py).
"""

from __future__ import annotations

import numpy as np

from views_baseline.model.models.distributional.matryoshka.doll_counts import (
    CELL,
    COUNTRY,
    DOLLS,
    NEAR,
    DollCounts,
)
from views_baseline.model.models.distributional.matryoshka.geometry import BAND_DISTANCE
from views_baseline.model.models.distributional.matryoshka.size_classes import N_CLASSES

PROB_FLOOR = 1e-12   # only to keep the log finite; how often it bites is reported


def _ages(months: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    return (cutoffs[:, None] - months[None, :]).astype(np.float64)


def time_weights(months: np.ndarray, cutoffs: np.ndarray, memory: float) -> np.ndarray:
    """(n_cutoffs, M) weights over history months, each row summing to 1 over months ≤ cutoff.

    ``memory = inf`` gives equal weights.
    """
    age = _ages(months, cutoffs)
    keep = age >= 0
    w = np.where(keep, 1.0 if np.isinf(memory) else np.exp(-np.maximum(age, 0) / memory), 0.0)
    return w / w.sum(axis=1, keepdims=True)


def time_weight_slopes(months: np.ndarray, cutoffs: np.ndarray, memory: float) -> np.ndarray:
    """(n_cutoffs, M): ∂ time_weights / ∂ log memory. Each row sums to 0."""
    age = _ages(months, cutoffs)
    keep = age >= 0
    w = np.where(keep, np.exp(-np.maximum(age, 0) / memory), 0.0)
    w /= w.sum(axis=1, keepdims=True)
    mean_age = (w * np.where(keep, age, 0.0)).sum(axis=1, keepdims=True)
    return np.where(keep, w * (age - mean_age) / memory, 0.0)


def _weigh(X, W: np.ndarray) -> np.ndarray:
    """Σ_s W[r, s] · X[i, s] for a sparse (N, M) X: (R, N)."""
    return np.asarray(X @ W.T).T


def _kernel(length: float) -> tuple[np.ndarray, np.ndarray]:
    """Band weights a_b = exp(−d_b / ℓ) and their slope a_b · d_b / ℓ (per log ℓ)."""
    a = np.exp(-BAND_DISTANCE / length)
    return a, a * BAND_DISTANCE / length


def doll_shares(dc: DollCounts, W: np.ndarray, doll: int,
                length: float | None = None) -> np.ndarray:
    """(R, N, K): Σ_s W[r, s] · (share of doll ``doll``'s cells in each class in month s).

    ``W`` holds one row of month weights per output row: `time_weights` gives q_doll, and
    `time_weight_slopes` its slope; stacking both reads the counts once. ``length`` is the
    neighbourhood length scale ℓ (cells), needed for doll NEAR only. Shell arithmetic as in
    doll_counts.py; class 0 is the rest (each row of shares sums to the row sum of W).
    """
    W = np.asarray(W, dtype=np.float64)
    R, N = W.shape[0], dc.sizes.shape[1]
    q = np.zeros((R, N, N_CLASSES))
    size = dc.sizes[doll].astype(np.float64)
    has_country = dc.country_index >= 0
    # Cells without a country index a zero column appended after the countries, so this also
    # works when no cell has a country (code review of 702f17d).
    ci = np.where(has_country, dc.country_index, dc.country_tot.shape[2])
    if doll == NEAR:
        a, _ = _kernel(length)
        z = a @ dc.band_sizes                                                    # (N,)
    for k in range(1, N_CLASSES):
        if doll == CELL:
            q[..., k] = _weigh(dc.own[k], W)
        elif doll == NEAR:
            num = sum(a[b] * _weigh(dc.band[k][b], W) for b in range(a.size))
            with np.errstate(invalid="ignore", divide="ignore"):
                q[..., k] = np.where(z > 0, num / z, 0.0)
        else:
            totals = np.concatenate([W @ dc.country_tot[k], np.zeros((R, 1))], axis=1)
            country_rest = np.where(has_country, totals[:, ci]
                                    - _weigh(dc.inner_same_country[k], W), 0.0)
            if doll == COUNTRY:
                q[..., k] = country_rest
            else:
                inner = _weigh(dc.own[k], W) + sum(_weigh(x, W) for x in dc.band[k])
                q[..., k] = (W @ dc.global_tot[k])[:, None] - inner - country_rest
            with np.errstate(invalid="ignore", divide="ignore"):
                q[..., k] = np.where(size > 0, q[..., k] / size, 0.0)
    q[..., 0] = np.where(size > 0, W.sum(axis=1)[:, None] - q[..., 1:].sum(axis=2), 0.0)
    return q


def near_length_slope(dc: DollCounts, W: np.ndarray, length: float) -> np.ndarray:
    """(R, N, K): ∂ doll_shares(NEAR) / ∂ log ℓ, for time-weight rows W."""
    W = np.asarray(W, dtype=np.float64)
    a, da = _kernel(length)
    z, dz = a @ dc.band_sizes, da @ dc.band_sizes
    q = doll_shares(dc, W, NEAR, length)
    out = np.zeros_like(q)
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in range(1, N_CLASSES):
            dnum = sum(da[b] * _weigh(dc.band[k][b], W) for b in range(a.size))
            out[..., k] = np.where(z > 0, dnum / z - q[..., k] * dz / z, 0.0)
    out[..., 0] = -out[..., 1:].sum(axis=2)
    return out


def outcome_classes(classes: np.ndarray, months: np.ndarray, cutoffs: np.ndarray,
                    horizon: int) -> np.ndarray:
    """(n_cutoffs, horizon, N): the observed size class at month c + h, h = 1..horizon."""
    pos = np.searchsorted(months, cutoffs[:, None] + np.arange(1, horizon + 1)[None, :])
    return classes[pos]


def _doll_mix(sizes, H, weight, memory):
    """(D, H, N) A_k: the unnormalised P(doll k | h) per cell, zero for empty shells."""
    h = np.arange(1, H + 1, dtype=np.float64)
    a = weight[:, None] * np.exp(-h[None, :] / memory[:, None])              # (D, H)
    return a[:, :, None] * (sizes > 0)[:, None, :], h


def _observed(q_k: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """(n_cut, H, N): q_k at the class that actually happened."""
    return np.take_along_axis(q_k[:, None, :, :], outcomes[..., None].astype(np.int64),
                              axis=3)[..., 0].astype(np.float64)


def log_score(q: np.ndarray, sizes: np.ndarray, outcomes: np.ndarray, weight: np.ndarray,
              memory: np.ndarray) -> tuple[float, int]:
    """Mean log P(observed class), and how many probabilities hit the floor.

    q: (D, n_cut, N, K) doll shares, D = len(DOLLS); sizes: (D, N); outcomes: (n_cut, H, N);
    weight, memory: (D,) per doll.
    """
    A, _ = _doll_mix(sizes, outcomes.shape[1], weight, memory)
    num = sum(A[k][None] * _observed(q[k], outcomes) for k in range(len(DOLLS)))
    p = num / A.sum(axis=0)[None]
    return float(np.log(np.maximum(p, PROB_FLOOR)).mean()), int((p < PROB_FLOOR).sum())


def log_score_slope(q: np.ndarray, dq: np.ndarray, dq_length: np.ndarray, sizes: np.ndarray,
                    outcomes: np.ndarray, weight: np.ndarray,
                    memory: np.ndarray) -> tuple[float, int, np.ndarray]:
    """As `log_score`, plus the slope (2D,) over
    (log weight[1..D-1], log memory[0..D-1], log ℓ).

    dq: (D, n_cut, N, K), each doll's shares differentiated by its log memory;
    dq_length: (n_cut, N, K), the neighbourhood's shares differentiated by log ℓ. Floored
    probabilities contribute no slope (the floor is flat).
    """
    A, h = _doll_mix(sizes, outcomes.shape[1], weight, memory)
    D = len(DOLLS)
    P = [_observed(q[k], outcomes) for k in range(D)]
    num = sum(A[k][None] * P[k] for k in range(D))
    den = A.sum(axis=0)[None]
    p = num / den
    ok = p >= PROB_FLOOR
    safe_num = np.where(ok, num, 1.0)
    slope = np.zeros(2 * D)
    for k in range(D):
        T = np.where(ok, A[k][None] * P[k] / safe_num - A[k][None] / den, 0.0)
        if k > 0:
            slope[k - 1] = T.mean()
        via_memory = np.where(ok, A[k][None] * _observed(dq[k], outcomes) / safe_num, 0.0)
        slope[D - 1 + k] = (T * (h / memory[k])[None, :, None] + via_memory).mean()
    slope[2 * D - 1] = np.where(ok, A[NEAR][None] * _observed(dq_length, outcomes) / safe_num,
                                0.0).mean()
    return float(np.log(np.maximum(p, PROB_FLOOR)).mean()), int((~ok).sum()), slope
