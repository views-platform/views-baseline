"""Closeness-to-conflictology measurement harness (ADR-022 / epic #33 S3).

Operationalises "I couldn't tell which model the 1000 draws came from" as a
distributional two-sample comparison between two ``dict[str, PredictionFrame]``:

- per (cell, target): 1-Wasserstein, energy distance, and a 1-NN **C2ST** accuracy
  (0.5 = indistinguishable);
- a **same-model null** (conflictology two seeds) as the finite-N noise floor;
- **stratification** by activity regime (zero / low / high), target, and level;
- an **equivalence** verdict against a pre-registered margin ``delta`` (not a p-value).

This module is *not* in the model path — it only consumes PredictionFrames.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import energy_distance, wasserstein_distance

# Activity strata by the reference (conflictology) per-cell mean, in raw counts.
_LOW_HIGH_CUT = 1.0  # mean < this (and not all-zero) = "low"; >= this = "high"


def c2st_1nn(a: np.ndarray, b: np.ndarray, rng: np.random.Generator | None = None) -> float:
    """Leave-one-out 1-NN classifier two-sample accuracy in 1-D.

    Pools the two balanced samples, labels them, and for each point asks whether its
    nearest neighbour shares its label. ``0.5`` ⇒ indistinguishable, ``1.0`` ⇒
    perfectly separable. Pure numpy, O(n log n). Ties (e.g. constant/degenerate
    cells) are broken by a random permutation before the stable sort, so identical
    samples score ~0.5 rather than spuriously ~1.0.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    x = np.concatenate([a, b])
    y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    n = len(x)
    if n < 2:
        return 0.5
    perm = rng.permutation(n)  # randomise tie order
    x, y = x[perm], y[perm]
    order = np.argsort(x, kind="mergesort")  # stable → ties keep the random order
    xs, ys = x[order], y[order]
    dl = np.empty(n)
    dl[0] = np.inf
    dl[1:] = xs[1:] - xs[:-1]
    dr = np.empty(n)
    dr[-1] = np.inf
    dr[:-1] = xs[1:] - xs[:-1]
    idx = np.arange(n)
    nn = np.where(dl <= dr, idx - 1, idx + 1)
    return float((ys[nn] == ys).mean())


def _cell_map(pf) -> dict:
    """Map (unit, time) -> row index for a PredictionFrame."""
    time = np.asarray(pf.index.time)
    unit = np.asarray(pf.index.unit)
    return {(int(u), int(t)): i for i, (u, t) in enumerate(zip(unit, time))}


def _stratum(ref_samples: np.ndarray) -> str:
    m = float(np.mean(ref_samples))
    if m == 0.0:
        return "zero"
    return "low" if m < _LOW_HIGH_CUT else "high"


def compare_models(model: dict, reference: dict, seed: int = 0) -> dict:
    """Per-cell distances between ``model`` and ``reference`` (e.g. conflictology).

    Both are ``dict[str, PredictionFrame]`` with ``.values`` shape ``(N, S)``. Cells
    are aligned by ``(unit, time)``; the stratum is assigned from the *reference*
    per-cell mean. ``seed`` makes the C2ST tie-breaking reproducible. Returns
    ``{target: {"wasserstein","energy","c2st","stratum","level"}}`` with numpy arrays
    over the common cells.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for target, ref_pf in reference.items():
        mod_pf = model[target]
        ref_map, mod_map = _cell_map(ref_pf), _cell_map(mod_pf)
        common = [k for k in ref_map if k in mod_map]
        w = np.empty(len(common))
        e = np.empty(len(common))
        c = np.empty(len(common))
        strat = np.empty(len(common), dtype=object)
        for j, key in enumerate(common):
            ra = np.asarray(ref_pf.values[ref_map[key]], dtype=np.float64).ravel()
            ma = np.asarray(mod_pf.values[mod_map[key]], dtype=np.float64).ravel()
            w[j] = wasserstein_distance(ma, ra)
            e[j] = energy_distance(ma, ra)
            c[j] = c2st_1nn(ma, ra, rng)
            strat[j] = _stratum(ra)
        out[target] = {
            "wasserstein": w,
            "energy": e,
            "c2st": c,
            "stratum": strat,
            "level": ref_pf.index.level.name,
        }
    return out


def summarize(metrics: dict) -> dict:
    """Stratified summary: per (target, stratum) → median / p90 / p99 of each metric."""
    rows = {}
    for target, m in metrics.items():
        for stratum in ("zero", "low", "high", "all"):
            mask = np.ones(len(m["c2st"]), bool) if stratum == "all" else (m["stratum"] == stratum)
            if not mask.any():
                continue
            rows[(target, stratum)] = {
                "n": int(mask.sum()),
                "wasserstein": _pcts(m["wasserstein"][mask]),
                "energy": _pcts(m["energy"][mask]),
                "c2st": _pcts(m["c2st"][mask]),
            }
    return rows


def _pcts(a: np.ndarray) -> dict:
    return {
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
    }


def equivalence(candidate: dict, null: dict, delta: dict) -> dict:
    """Per (target, stratum) equivalence verdict.

    A candidate is *equivalent to conflictology* in a stratum when its median C2ST is
    within ``0.5 + delta["c2st"]`` AND its median Wasserstein is within the same-model
    null median plus ``delta["wasserstein"]``.
    """
    verdict = {}
    for key, cand in candidate.items():
        null_row = null.get(key)
        c2st_ok = cand["c2st"]["median"] <= 0.5 + delta["c2st"]
        w_ref = null_row["wasserstein"]["median"] if null_row else 0.0
        w_ok = cand["wasserstein"]["median"] <= w_ref + delta["wasserstein"]
        verdict[key] = {
            "equivalent": bool(c2st_ok and w_ok),
            "c2st_median": cand["c2st"]["median"],
            "c2st_ok": bool(c2st_ok),
            "wasserstein_median": cand["wasserstein"]["median"],
            "wasserstein_null_median": w_ref,
            "wasserstein_ok": bool(w_ok),
        }
    return verdict
