"""Fit Matryoshka's knobs on real data by backtest (ADR-024, prototype for #127).

Data in:   the calibration input pull (parquet, index month_id × priogrid_id; lr_*_best, c_id),
           restricted to months ≤ TRAIN_END, the last training month; GAUL admin-1 per cell.
Transform: size classes → per-month doll counts → for each knob setting, class probabilities at every
           backtest cutoff and horizon → mean log score and its exact slope (for L-BFGS-B).
Config:    the constants below (cutoffs, horizon, bounds, starting values), all from ADR-024 / the plan.
Out:       one JSON per violence type in ../results/v6_kernel_province/ (knobs, references, stability, timings).

Run: python fit_knobs.py <input.parquet> <gaul1_code.parquet> [--stability]
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from doll_counts import DOLLS, NEAR, build_doll_counts
from forecast_probability import (
    doll_shares,
    log_score,
    log_score_slope,
    near_length_slope,
    outcome_classes,
    time_weight_slopes,
    time_weights,
)
from grid_geometry import NO_PROVINCE, build_geometry
from scipy.optimize import minimize
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
# In DOLLS order: cell, near, province, country, map. Currents of Conflict as a guide: the near weight
# starts at the sum of the two former rings; the length scale at 0.72° = 1.44 cells (#139); the
# province sits between the neighbourhood and the country (#136 re-test).
START_MEMORY = np.array([4.0, 12.0, 36.0, 60.0, 122.0])
START_WEIGHT = np.array([1.0, 0.6, 0.05, 0.1, 0.01])           # cell fixed at 1
START_LENGTH = 1.44                                             # cells
LENGTH_BOUNDS = (0.25, 10.0)                                    # cells; NEAR_RADIUS is 5
RESULTS = Path(__file__).resolve().parent.parent / "results" / "v6_kernel_province"


def load(path: str, province_path: str, last_month: int = TRAIN_END):
    """Classes-ready panel: months, sorted cells, their country and province, per-target (M, N) values.

    Only months HISTORY_START..last_month are read; the fit uses last_month = TRAIN_END.
    ``province_path`` is the GAUL admin-1 file (views-datafactory data/raw/gaul_admin/gaul1_code.parquet,
    columns gid = priogrid_id and value = gaul1 code). A cell has a province iff its code is not
    NO_PROVINCE and it has a country.
    """
    df = pd.read_parquet(path, columns=TARGETS + ["c_id"]).reset_index()
    df = df[(df.month_id >= HISTORY_START) & (df.month_id <= last_month)]
    months = np.sort(df.month_id.unique())
    cells = np.sort(df.priogrid_id.unique())
    if len(df) != months.size * cells.size:
        raise ValueError(f"panel is not complete: {len(df)} rows for {months.size} × {cells.size}")
    df = df.sort_values(["month_id", "priogrid_id"])
    country = df[df.month_id == months[0]].c_id.to_numpy().astype(int)
    gaul1 = pd.read_parquet(province_path).set_index("gid")["value"]
    missing = np.setdiff1d(cells, gaul1.index.to_numpy())
    if missing.size:
        raise ValueError(f"{missing.size} cells have no row in {province_path}")
    province = gaul1.reindex(cells).to_numpy().astype(int)
    province = np.where(country > 0, province, NO_PROVINCE)
    print(f"cells without a province: {(province == NO_PROVINCE).sum()} "
          f"(of which without a country: {(country <= 0).sum()}); "
          f"negative real province codes in use: {sorted(set(province[province < -1]))}", flush=True)
    values = {t: df[t].to_numpy().reshape(months.size, cells.size) for t in TARGETS}
    return months, cells, country, province, values


class Objective:
    """Mean log score and its exact slope as a function of the knobs.

    theta = (log weight[1..D−1], log memory[0..D−1], log length scale), D = len(DOLLS): 2D values.
    Caches each doll's shares (and their slopes) per memory (and, for NEAR, length scale): a
    weight-only change rescores without touching the counts.
    """

    def __init__(self, dc, cutoffs, outcomes):
        self.dc, self.cutoffs, self.outcomes = dc, cutoffs, outcomes
        self._cache: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
        self.evaluations = 0

    def shares(self, doll: int, memory: float, length: float):
        """(q, ∂q/∂log memory, ∂q/∂log ℓ or None), each (n_cut, N, K)."""
        key = (doll, round(float(memory), 9), round(float(length), 9) if doll == NEAR else None)
        if key not in self._cache:
            if len(self._cache) > 40:
                self._cache.clear()
            n = self.cutoffs.size
            Wt = time_weights(self.dc.months, self.cutoffs, memory)
            W = np.vstack([Wt, time_weight_slopes(self.dc.months, self.cutoffs, memory)])
            both = doll_shares(self.dc, W, doll, length)
            dlen = near_length_slope(self.dc, Wt, length) if doll == NEAR else None
            self._cache[key] = (both[:n], both[n:], dlen)
        return self._cache[key]

    @staticmethod
    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        D = len(DOLLS)
        weight = np.concatenate([[1.0], np.exp(theta[:D - 1])])
        memory = np.exp(theta[D - 1:2 * D - 1])
        return weight, memory, float(np.exp(theta[2 * D - 1]))

    def score(self, weight: np.ndarray, memory: np.ndarray, length: float) -> tuple[float, int]:
        q = np.stack([self.shares(k, memory[k], length)[0] for k in range(len(DOLLS))])
        return log_score(q, self.dc.sizes, self.outcomes, weight, memory)

    def __call__(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """Minus the score and minus its slope (the search minimises)."""
        self.evaluations += 1
        weight, memory, length = self.unpack(theta)
        parts = [self.shares(k, memory[k], length) for k in range(len(DOLLS))]
        q, dq = np.stack([a for a, _, _ in parts]), np.stack([b for _, b, _ in parts])
        score, _, slope = log_score_slope(q, dq, parts[NEAR][2], self.dc.sizes, self.outcomes,
                                          weight, memory)
        return -score, -slope


def start_theta() -> np.ndarray:
    return np.concatenate([np.log(START_WEIGHT[1:]), np.log(START_MEMORY), [np.log(START_LENGTH)]])


def fit(dc, cutoffs, outcomes) -> dict:
    obj = Objective(dc, cutoffs, outcomes)
    bounds = ([tuple(np.log(WEIGHT_BOUNDS))] * (len(DOLLS) - 1)
              + [tuple(np.log(MEMORY_BOUNDS))] * len(DOLLS) + [tuple(np.log(LENGTH_BOUNDS))])
    t0 = time.time()
    res = minimize(obj, start_theta(), jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 200, "ftol": 1e-13, "gtol": 1e-9})
    weight, memory, length = Objective.unpack(res.x)
    score, floored = obj.score(weight, memory, length)
    return {"weight": weight.tolist(), "memory": memory.tolist(), "length_scale": length,
            "score": score, "floored": floored, "converged": bool(res.success),
            "message": str(res.message), "evaluations": obj.evaluations, "steps": int(res.nit),
            "seconds": round(time.time() - t0, 1)}


def reference_scores(dc, cutoffs, outcomes) -> dict:
    """Conflictology-like references: only the cell, equal weights (36-month window / all history)."""
    out = {}
    for name, window in (("conflictology_36m", 36), ("cell_all_history", None)):
        q0 = doll_shares(dc, time_weights(dc.months, cutoffs, np.inf, window), 0)
        q = np.stack([q0] + [np.zeros_like(q0)] * (len(DOLLS) - 1))
        w = np.eye(len(DOLLS))[0]
        out[name] = dict(zip(("score", "floored"),
                             log_score(q, dc.sizes, outcomes, w, np.full(len(DOLLS), np.inf))))
    return out


def run(path: str, province_path: str, stability: bool) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    months, cells, country, province, values = load(path, province_path)
    geo = build_geometry(cells, country, province)
    load_s = time.time() - t0
    for target in TARGETS:
        t1 = time.time()
        classes = classify(values[target])
        dc = build_doll_counts(classes, months, geo)
        counts_s = time.time() - t1
        outcomes = outcome_classes(classes, months, CUTOFFS, HORIZON)
        coverage = dict(zip(LABELS, np.bincount(classes.ravel(), minlength=7).tolist()))
        t2 = time.time()
        one = Objective(dc, CUTOFFS, outcomes)(start_theta())
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
              f"{np.round(f['weight'], 4).tolist()}; length {f['length_scale']:.2f} cells; "
              f"{f['seconds']} s, "
              f"{f['evaluations']} evaluations", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("provinces", help="GAUL admin-1 file: views-datafactory "
                                      "data/raw/gaul_admin/gaul1_code.parquet")
    ap.add_argument("--stability", action="store_true")
    args = ap.parse_args()
    run(args.input, args.provinces, args.stability)
