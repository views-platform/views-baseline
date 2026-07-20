"""Distribution family strategies for the parametric climatology models (ADR-022).

Each family exposes `fit(values) -> params` (method-of-moments) and
`sample(params, size, rng) -> ndarray`. The native-zero families (`nb`, `zinb`) model
the full window; continuous families (`lognormal`, `gumbel`, `gamma`) are positive-part
families for the hurdle model. (Tweedie was evaluated in S9 and excluded — see ADR-022 /
`reports/closeness_experiment/FINDINGS.md`.)

Numeric edges **fail safe** (point-mass / Poisson fallback) with a WARN, never NaN.
Pure module: numpy only; no pandas, no views-frames, no model imports.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_EULER_GAMMA = 0.5772156649015329

# Native-zero families (no-hurdle); continuous positive-part families (hurdle).
NATIVE_ZERO_FAMILIES = frozenset({"nb", "zinb"})
CONTINUOUS_FAMILIES = frozenset({"lognormal", "gumbel", "gamma"})


# ---------------------------------------------------------------------------
# point-mass (the universal degenerate fallback — matches conflictology on
# all-zero / single-value windows)
# ---------------------------------------------------------------------------
def _point(value: float) -> dict:
    return {"kind": "point", "value": float(value)}


def _sample_point(params: dict, size, rng) -> np.ndarray:
    return np.full(size, params["value"], dtype=np.float64)


# ---------------------------------------------------------------------------
# nb — negative binomial via the Gamma-Poisson mixture (handles real n, rng-native)
# ---------------------------------------------------------------------------
def fit_nb(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64)
    m, var = float(v.mean()), float(v.var())
    if m == 0.0:
        return _point(0.0)
    if var <= m:  # not overdispersed -> NB moment-match undefined; fall back
        logger.warning("nb: var<=mean (%.4g<=%.4g) -> Poisson fallback", var, m)
        return {"kind": "poisson", "mu": m}
    p = m / var
    n = m * m / (var - m)
    return {"kind": "nb", "n": n, "p": p}


def sample_nb(params: dict, size, rng) -> np.ndarray:
    kind = params["kind"]
    if kind == "point":
        return _sample_point(params, size, rng)
    if kind == "poisson":
        return rng.poisson(params["mu"], size=size).astype(np.float64)
    # NB(n,p) == Poisson(lam), lam ~ Gamma(shape=n, scale=(1-p)/p)
    lam = rng.gamma(params["n"], (1.0 - params["p"]) / params["p"], size=size)
    return rng.poisson(lam).astype(np.float64)


# ---------------------------------------------------------------------------
# zinb — zero-inflated negative binomial (native-zero). A structural-zero spike
# `pi` on top of an NB: Y = 0 w.p. pi, else Y ~ NB. Fixes the NB dispersion `n`
# (from the window's overdispersion) and solves the inflated NB mean so BOTH the
# sample mean and the empirical zero-rate match exactly. Falls back to plain NB
# when the NB already explains the zeros (no excess) — no magic pi (ADR-021).
# ---------------------------------------------------------------------------
def fit_zinb(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64)
    m = float(v.mean())
    if m == 0.0:
        return _point(0.0)
    nb = fit_nb(v)
    if nb["kind"] != "nb":  # underdispersed/degenerate -> NB component undefined
        logger.warning("zinb: NB component undefined (%s) -> fallback", nb["kind"])
        return nb
    n = nb["n"]
    p0 = float(np.mean(v == 0.0))

    def mixture_zero_rate(m_nb: float) -> float:
        pi = 1.0 - m / m_nb            # structural-zero prob (m_nb >= m)
        p_nb0 = (n / (n + m_nb)) ** n  # NB(n, mean=m_nb) mass at 0
        return pi + (1.0 - pi) * p_nb0

    if p0 <= mixture_zero_rate(m) + 1e-12:
        return nb  # NB alone already has >= the observed zeros; no inflation needed
    # zero-rate is monotone increasing in m_nb on [m, inf) from P(NB0) up to 1 -> bisect
    lo, hi = m, m
    for _ in range(100):
        hi *= 2.0
        if mixture_zero_rate(hi) >= p0:
            break
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if mixture_zero_rate(mid) < p0:
            lo = mid
        else:
            hi = mid
    m_nb = 0.5 * (lo + hi)
    pi = 1.0 - m / m_nb
    return {"kind": "zinb", "pi": pi, "n": n, "p": n / (n + m_nb)}


def sample_zinb(params: dict, size, rng) -> np.ndarray:
    if params["kind"] != "zinb":  # point/poisson/nb fallbacks share the NB sampler
        return sample_nb(params, size, rng)
    struct = rng.random(size) < params["pi"]
    lam = rng.gamma(params["n"], (1.0 - params["p"]) / params["p"], size=size)
    w = rng.poisson(lam).astype(np.float64)
    w[struct] = 0.0
    return w


# ---------------------------------------------------------------------------
# lognormal / gumbel / gamma (continuous positive-part families)
# ---------------------------------------------------------------------------
def fit_lognormal(values: np.ndarray) -> dict:
    pos = np.asarray(values, dtype=np.float64)
    pos = pos[pos > 0]
    if pos.size == 0:
        return _point(0.0)
    logs = np.log(pos)
    mu, sigma = float(logs.mean()), float(logs.std())
    if sigma == 0.0:
        return _point(float(pos[0]))
    return {"kind": "lognormal", "mu": mu, "sigma": sigma}


def sample_lognormal(params: dict, size, rng) -> np.ndarray:
    if params["kind"] == "point":
        return _sample_point(params, size, rng)
    return np.exp(rng.normal(params["mu"], params["sigma"], size=size))


def fit_gumbel(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64)
    m, std = float(v.mean()), float(v.std())
    if std == 0.0:
        return _point(m)
    scale = std * np.sqrt(6.0) / np.pi
    loc = m - _EULER_GAMMA * scale
    return {"kind": "gumbel", "loc": loc, "scale": scale}


def sample_gumbel(params: dict, size, rng) -> np.ndarray:
    if params["kind"] == "point":
        return _sample_point(params, size, rng)
    return rng.gumbel(params["loc"], params["scale"], size=size)


def fit_gamma(values: np.ndarray) -> dict:
    pos = np.asarray(values, dtype=np.float64)
    pos = pos[pos > 0]
    if pos.size == 0:
        return _point(0.0)
    m, var = float(pos.mean()), float(pos.var())
    if var == 0.0:
        return _point(float(pos[0]))
    theta = var / m
    k = m / theta
    return {"kind": "gamma", "k": k, "theta": theta}


def sample_gamma(params: dict, size, rng) -> np.ndarray:
    if params["kind"] == "point":
        return _sample_point(params, size, rng)
    return rng.gamma(params["k"], params["theta"], size=size)


FAMILIES = {
    "nb": (fit_nb, sample_nb),
    "zinb": (fit_zinb, sample_zinb),
    "lognormal": (fit_lognormal, sample_lognormal),
    "gumbel": (fit_gumbel, sample_gumbel),
    "gamma": (fit_gamma, sample_gamma),
}
