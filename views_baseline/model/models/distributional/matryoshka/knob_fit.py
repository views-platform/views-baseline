"""The knob search: fit Matryoshka's knobs (8 with all four dolls) per violence type by backtest
(ADR-024).

Data in:   one violence type's complete panel (M months × N cells) up to ``train_end``, its
           geometry, and the backtest design (cutoffs, horizon).
Transform: size classes → per-month doll counts → for each knob setting, the exact class
           probabilities at every cutoff and horizon → mean log score and its exact slope →
           L-BFGS-B over the logarithms of the knobs.
Config:    the constants below. ADR-024 fixes them (amendment 2026-11-06); they are not
           configuration.
Out:       a plain dict of knobs and fit diagnostics (`fit_knobs`), stored on the model.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from views_baseline.model.models.distributional.matryoshka.doll_counts import (
    DOLLS,
    NEAR,
    DollCounts,
    build_doll_counts,
    doll_names,
)
from views_baseline.model.models.distributional.matryoshka.forecast_probability import (
    doll_shares,
    log_score,
    log_score_slope,
    near_length_slope,
    outcome_classes,
    time_weight_slopes,
    time_weights,
)
from views_baseline.model.models.distributional.matryoshka.geometry import Geometry
from views_baseline.model.models.distributional.matryoshka.size_classes import (
    N_CLASSES,
    classify,
)

CUTOFF_SPACING = 12               # months between backtest cutoffs
MIN_HISTORY_MONTHS = 120          # history before the earliest cutoff (ADR-024: at least 10 years)
# In DOLLS order: cell, near, country, map. The cell's weight is the reference, fixed at 1.
START_WEIGHT = np.array([1.0, 0.6, 0.1, 0.01])
START_MEMORY = np.array([4.0, 12.0, 60.0, 122.0])          # months
START_LENGTH = 1.44                                         # cells (0.72°, Currents of Conflict)
WEIGHT_BOUNDS = (1e-6, 1e3)
MEMORY_BOUNDS = (1.0, 600.0)      # months; 600 is a deliberate ceiling: "the whole history" (#140)
LENGTH_BOUNDS = (0.25, 10.0)      # cells; the neighbourhood radius is 5
# The score and its slopes are small numbers (score ≈ −0.01, slopes ≈ 1e-4), so the default
# stopping thresholds stop too early; reports/matryoshka/LOG.md, 2026-11-06.
SEARCH_OPTIONS = {"maxiter": 200, "ftol": 1e-13, "gtol": 1e-9}


def backtest_cutoffs(train_end: int, n_cutoffs: int, horizon: int) -> np.ndarray:
    """The cutoffs, ``CUTOFF_SPACING`` months apart, the last ``horizon`` months before
    ``train_end`` so that every scored month is ≤ ``train_end``. Ascending."""
    if n_cutoffs < 1 or horizon < 1:
        raise ValueError(f"backtest_cutoffs and backtest_horizon must be ≥ 1, got {n_cutoffs} "
                         f"and {horizon}.")
    last = train_end - horizon
    return last - CUTOFF_SPACING * np.arange(n_cutoffs)[::-1]


def free_knobs(on: np.ndarray) -> np.ndarray:
    """(2D,) bool: which entries of the full knob vector (log weight[1..D−1], log memory[0..D−1],
    log length scale) the search fits for the on-mask. An off doll's knobs are left out, not
    pinned: its shell is empty, so it adds nothing to the score (#153)."""
    return np.concatenate([on[1:], on, [on[NEAR]]])


class Objective:
    """Minus the mean log score and minus its exact slope, as a function of the free knobs.

    The full knob vector is (log weight[1..D−1], log memory[0..D−1], log length scale),
    D = len(DOLLS): 2D values. theta holds only its free entries (`free_knobs(dc.on)`); with every
    doll on, theta is the full vector. An off doll has weight 0 and, inside the score, memory inf
    (so its terms stay finite); its shares are zeros and are never computed.
    Keeps each doll's latest shares (and their slopes), keyed by its memory (and, for NEAR, the
    length scale), so a weight-only change, or a doll whose knobs did not move, reuses them. One
    entry per doll bounds the memory: each is 3 arrays of n_cutoffs × N × 7 float64.
    ``first_score`` is the score at the first evaluation (the starting values).
    """

    def __init__(self, dc: DollCounts, cutoffs: np.ndarray, outcomes: np.ndarray):
        self.dc, self.cutoffs, self.outcomes = dc, cutoffs, outcomes
        self.on = dc.on
        self.free = free_knobs(dc.on)
        self._latest: dict[int, tuple[tuple, tuple]] = {}
        self._zeros = np.zeros((cutoffs.size, dc.sizes.shape[1], N_CLASSES))
        self.evaluations = 0
        self.first_score: float | None = None

    def shares(self, doll: int, memory: float, length: float):
        """(q, ∂q/∂log memory, ∂q/∂log ℓ or None), each (n_cut, N, K). Zeros for an off doll."""
        if not self.on[doll]:
            return self._zeros, self._zeros, (self._zeros if doll == NEAR else None)
        key = (round(float(memory), 9), round(float(length), 9) if doll == NEAR else None)
        if doll not in self._latest or self._latest[doll][0] != key:
            n = self.cutoffs.size
            Wt = time_weights(self.dc.months, self.cutoffs, memory)
            W = np.vstack([Wt, time_weight_slopes(self.dc.months, self.cutoffs, memory)])
            both = doll_shares(self.dc, W, doll, length)
            dlen = near_length_slope(self.dc, Wt, length) if doll == NEAR else None
            self._latest[doll] = (key, (both[:n], both[n:], dlen))
        return self._latest[doll][1]

    def unpack(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """(weight, memory, length) from the free knobs. An off doll: weight 0, memory inf; the
        length scale is NaN with the neighbourhood off."""
        D = len(DOLLS)
        full = np.zeros(2 * D)
        full[self.free] = theta
        weight = np.where(self.on, np.concatenate([[1.0], np.exp(full[:D - 1])]), 0.0)
        memory = np.where(self.on, np.exp(full[D - 1:2 * D - 1]), np.inf)
        length = float(np.exp(full[2 * D - 1])) if self.on[NEAR] else np.nan
        return weight, memory, length

    def score(self, weight: np.ndarray, memory: np.ndarray, length: float) -> tuple[float, int]:
        q = np.stack([self.shares(k, memory[k], length)[0] for k in range(len(DOLLS))])
        return log_score(q, self.dc.sizes, self.outcomes, weight, memory)

    def __call__(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        self.evaluations += 1
        weight, memory, length = self.unpack(theta)
        parts = [self.shares(k, memory[k], length) for k in range(len(DOLLS))]
        q, dq = np.stack([a for a, _, _ in parts]), np.stack([b for _, b, _ in parts])
        score, _, slope = log_score_slope(q, dq, parts[NEAR][2], self.dc.sizes, self.outcomes,
                                          weight, memory)
        if self.first_score is None:
            self.first_score = score
        return -score, -slope[self.free]


def start_theta(on: np.ndarray) -> np.ndarray:
    """The starting values of the free knobs (`free_knobs`)."""
    full = np.concatenate([np.log(START_WEIGHT[1:]), np.log(START_MEMORY),
                           [np.log(START_LENGTH)]])
    return full[free_knobs(on)]


def search(dc: DollCounts, cutoffs: np.ndarray, outcomes: np.ndarray) -> dict:
    """One L-BFGS-B search from the fixed starting values. Plain-Python result.

    ``seconds`` is wall-clock time: the one field that differs between identical fits.
    """
    obj = Objective(dc, cutoffs, outcomes)
    bounds = ([tuple(np.log(WEIGHT_BOUNDS))] * (len(DOLLS) - 1)
              + [tuple(np.log(MEMORY_BOUNDS))] * len(DOLLS) + [tuple(np.log(LENGTH_BOUNDS))])
    bounds = [b for b, free in zip(bounds, obj.free) if free]
    t0 = time.time()
    res = minimize(obj, start_theta(dc.on), jac=True, method="L-BFGS-B", bounds=bounds,
                   options=SEARCH_OPTIONS)
    weight, memory, length = obj.unpack(res.x)
    score, floored = obj.score(weight, memory, length)
    # Stored: an off doll's memory is NaN (it has none), not the inf used inside the score.
    return {"dolls": doll_names(dc.on), "weight": weight.tolist(),
            "memory": np.where(dc.on, memory, np.nan).tolist(), "length_scale": length,
            "score": score, "start_score": obj.first_score, "floored": floored,
            "converged": bool(res.success),
            "message": str(res.message), "evaluations": obj.evaluations, "steps": int(res.nit),
            "seconds": round(time.time() - t0, 1)}


def fit_knobs(values: np.ndarray, months: np.ndarray, geo: Geometry, cutoffs: np.ndarray,
              horizon: int, stability_refits: bool, on: np.ndarray) -> dict:
    """The knobs for one violence type. ``values`` is the (M, N) panel up to ``train_end``;
    ``on`` is the on-mask of the dolls in use (`doll_mask`).

    With ``stability_refits``, the search is repeated with each cutoff left out in turn
    (ADR-024: "show how stable each knob is"); needs at least two cutoffs.
    """
    if cutoffs[0] - months[0] + 1 < MIN_HISTORY_MONTHS:
        raise ValueError(
            f"MatryoshkaConflictology needs at least {MIN_HISTORY_MONTHS} months of history "
            f"before the earliest backtest cutoff (month {int(cutoffs[0])}), but the data starts "
            f"at month {int(months[0])} ({int(cutoffs[0] - months[0] + 1)} months)."
        )
    if stability_refits and cutoffs.size < 2:
        raise ValueError("stability_refits needs at least 2 backtest cutoffs.")
    classes = classify(values)
    dc = build_doll_counts(classes, months, geo, on)
    outcomes = outcome_classes(classes, months, cutoffs, horizon)
    result = search(dc, cutoffs, outcomes)
    if stability_refits:
        result["leave_one_cutoff_out"] = [
            {"left_out": int(c), **search(dc, cutoffs[cutoffs != c], outcomes[cutoffs != c])}
            for c in cutoffs
        ]
    return result
