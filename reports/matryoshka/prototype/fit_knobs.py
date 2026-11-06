"""Fit Matryoshka's knobs on real data by backtest (ADR-024, prototype for #127).

Data in:   the calibration input pull (parquet, index month_id × priogrid_id; lr_*_best, c_id),
           restricted to months ≤ TRAIN_END, the last training month.
Transform: size classes → per-month doll counts → for each knob setting, class probabilities at every
           backtest cutoff and horizon → mean log score and its exact slope (for L-BFGS-B).
Config:    the constants below (cutoffs, horizon, bounds, starting values), all from ADR-024 / the plan.
Out:       one JSON per violence type in ../results/ (fitted knobs, references, stability, timings).

Run: python fit_knobs.py <input.parquet> [--stability]
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from doll_counts import DOLLS, build_doll_counts
from forecast_probability import (doll_shares, log_score, log_score_slope, outcome_classes,
                                  time_weight_slopes, time_weights)
from grid_geometry import build_geometry
from size_classes import LABELS, classify

TARGETS = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
TRAIN_END = 456
HISTORY_START = 121
CUTOFFS = np.array([384, 396, 408, 420])
HORIZON = 36
# The score and its slopes are small numbers (score ≈ −0.01, slopes ≈ 1e-4), so the default stopping
# thresholds (ftol 2.2e-9 relative, gtol 1e-5) are tightened; LOG.md 2026-11-06.
MEMORY_BOUNDS = (1.0, 600.0)      # months; a fit at 600 means "longer than we can see"
WEIGHT_BOUNDS = (1e-6, 1e3)
START_MEMORY = np.array([4.0, 12.0, 24.0, 60.0, 122.0])        # Currents of Conflict as a guide
START_WEIGHT = np.array([1.0, 0.4, 0.16, 0.1, 0.01])            # cell fixed at 1; rho ~ 0.4
RESULTS = Path(__file__).resolve().parent.parent / "results"


def load(path: str, last_month: int = TRAIN_END):
    """Classes-ready panel: months, sorted cells, their country, and per-target (M, N) values.

    Only months HISTORY_START..last_month are read; the fit uses last_month = TRAIN_END.
    """
    df = pd.read_parquet(path, columns=TARGETS + ["c_id"]).reset_index()
    df = df[(df.month_id >= HISTORY_START) & (df.month_id <= last_month)]
    months = np.sort(df.month_id.unique())
    cells = np.sort(df.priogrid_id.unique())
    if len(df) != months.size * cells.size:
        raise ValueError(f"panel is not complete: {len(df)} rows for {months.size} × {cells.size}")
    df = df.sort_values(["month_id", "priogrid_id"])
    country = df[df.month_id == months[0]].c_id.to_numpy().astype(int)
    values = {t: df[t].to_numpy().reshape(months.size, cells.size) for t in TARGETS}
    return months, cells, country, values


class Objective:
    """Mean log score and its exact slope as a function of the 9 knobs.

    Caches each doll's shares (and their slope) per memory value: a weight-only change rescores
    without touching the counts.
    """

    def __init__(self, dc, cutoffs, outcomes):
        self.dc, self.cutoffs, self.outcomes = dc, cutoffs, outcomes
        self._cache: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}
        self.evaluations = 0

    def shares(self, doll: int, memory: float) -> tuple[np.ndarray, np.ndarray]:
        key = (doll, round(float(memory), 9))
        if key not in self._cache:
            if len(self._cache) > 40:
                self._cache.clear()
            n = self.cutoffs.size
            W = np.vstack([time_weights(self.dc.months, self.cutoffs, memory),
                           time_weight_slopes(self.dc.months, self.cutoffs, memory)])
            both = doll_shares(self.dc, W, doll)
            self._cache[key] = (both[:n], both[n:])
        return self._cache[key]

    @staticmethod
    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        weight = np.concatenate([[1.0], np.exp(theta[:4])])
        memory = np.exp(theta[4:])
        return weight, memory

    def score(self, weight: np.ndarray, memory: np.ndarray) -> tuple[float, int]:
        q = np.stack([self.shares(k, memory[k])[0] for k in range(5)])
        return log_score(q, self.dc.sizes, self.outcomes, weight, memory)

    def __call__(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """Minus the score and minus its slope (the search minimises)."""
        self.evaluations += 1
        weight, memory = self.unpack(theta)
        pairs = [self.shares(k, memory[k]) for k in range(5)]
        q, dq = np.stack([a for a, _ in pairs]), np.stack([b for _, b in pairs])
        score, _, slope = log_score_slope(q, dq, self.dc.sizes, self.outcomes, weight, memory)
        return -score, -slope


def fit(dc, cutoffs, outcomes) -> dict:
    obj = Objective(dc, cutoffs, outcomes)
    theta0 = np.concatenate([np.log(START_WEIGHT[1:]), np.log(START_MEMORY)])
    bounds = [tuple(np.log(WEIGHT_BOUNDS))] * 4 + [tuple(np.log(MEMORY_BOUNDS))] * 5
    t0 = time.time()
    res = minimize(obj, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 200, "ftol": 1e-13, "gtol": 1e-9})
    weight, memory = Objective.unpack(res.x)
    score, floored = obj.score(weight, memory)
    return {"weight": weight.tolist(), "memory": memory.tolist(), "score": score,
            "floored": floored, "converged": bool(res.success), "message": str(res.message),
            "evaluations": obj.evaluations, "steps": int(res.nit),
            "seconds": round(time.time() - t0, 1)}


def reference_scores(dc, cutoffs, outcomes) -> dict:
    """Conflictology-like references: only the cell, equal weights (36-month window / all history)."""
    out = {}
    for name, window in (("conflictology_36m", 36), ("cell_all_history", None)):
        q0 = doll_shares(dc, time_weights(dc.months, cutoffs, np.inf, window), 0)
        q = np.stack([q0] + [np.zeros_like(q0)] * 4)
        w = np.array([1.0, 0, 0, 0, 0])
        out[name] = dict(zip(("score", "floored"),
                             log_score(q, dc.sizes, outcomes, w, np.full(5, np.inf))))
    return out


def run(path: str, stability: bool) -> None:
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    months, cells, country, values = load(path)
    geo = build_geometry(cells, country)
    load_s = time.time() - t0
    for target in TARGETS:
        t1 = time.time()
        classes = classify(values[target])
        dc = build_doll_counts(classes, months, geo)
        counts_s = time.time() - t1
        outcomes = outcome_classes(classes, months, CUTOFFS, HORIZON)
        coverage = dict(zip(LABELS, np.bincount(classes.ravel(), minlength=7).tolist()))
        t2 = time.time()
        one = Objective(dc, CUTOFFS, outcomes)(np.concatenate([np.log(START_WEIGHT[1:]),
                                                               np.log(START_MEMORY)]))
        one_eval_s = time.time() - t2
        result = {
            "target": target, "dolls": DOLLS, "cutoffs": CUTOFFS.tolist(), "horizon": HORIZON,
            "class_counts_in_history": coverage,
            "references": reference_scores(dc, CUTOFFS, outcomes),
            "start_score": -one[0],
            "fit": fit(dc, CUTOFFS, outcomes),
            "timing_seconds": {"load_all_targets": round(load_s, 1),
                               "doll_counts": round(counts_s, 1),
                               "first_score_evaluation": round(one_eval_s, 1)},
        }
        if stability:
            result["leave_one_cutoff_out"] = [
                {"left_out": int(c), **fit(dc, CUTOFFS[CUTOFFS != c],
                                           outcomes[CUTOFFS != c])}
                for c in CUTOFFS
            ]
        result["peak_rss_gib"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
        (RESULTS / f"{target}.json").write_text(json.dumps(result, indent=2))
        f = result["fit"]
        print(f"{target}: score {f['score']:.5f} (36m conflictology "
              f"{result['references']['conflictology_36m']['score']:.5f}); "
              f"memory {np.round(f['memory'], 1).tolist()}; weight "
              f"{np.round(f['weight'], 4).tolist()}; {f['seconds']} s, "
              f"{f['evaluations']} evaluations", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--stability", action="store_true")
    args = ap.parse_args()
    run(args.input, args.stability)
