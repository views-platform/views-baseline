"""Distribution family + transform registries for the parametric climatology models.

A **strategy registry** (ADR-022): each family exposes `fit(values) -> params`
(method-of-moments) and `sample(params, size, rng) -> ndarray`. The native-zero families
(`nb`, `zinb`) model the full window; continuous families (`lognormal`, `gumbel`, `gamma`)
are positive-part families for the hurdle model. (Tweedie was evaluated in S9 and excluded —
see ADR-022 / `reports/closeness_experiment/FINDINGS.md`.)

Numeric edges **fail safe** (point-mass / Poisson fallback) with a WARN, never NaN.
`TRANSFORMS` holds the `(forward, inverse)` pairs; `EMIT_LOG_CEIL` is the single-sourced
clamp applied before `expm1` (overflow guard — mirrors hydranet C-113).

Pure module: numpy + scipy only; no pandas, no views-frames, no model imports.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Single source of truth: cap log-space samples before expm1 so a heavy-tailed draw
# cannot expm1 into astronomical counts / float overflow (ADR-022; hydranet C-113).
EMIT_LOG_CEIL = 20.0  # expm1(20) ~ 4.85e8 — far above any real conflict count

# Emitted magnitudes are non-negative, but some positive-part families (`gumbel_r`)
# have support on all of ℝ — their lower tail can produce negative draws (directly, or
# via expm1(x)∈(-1,0) for x<0). Floor the emitted raw scale at zero so the models never
# emit a negative conflict magnitude (ADR-022; the ceiling's mirror-image guard).
EMIT_FLOOR = 0.0

_EULER_GAMMA = 0.5772156649015329

# Native-zero families (no-hurdle); continuous positive-part families (hurdle).
NATIVE_ZERO_FAMILIES = frozenset({"nb", "zinb"})
CONTINUOUS_FAMILIES = frozenset({"lognormal", "gumbel", "gamma"})

TRANSFORMS = {
    "none": (lambda x: np.asarray(x, dtype=np.float64), lambda x: np.asarray(x, dtype=np.float64)),
    "log1p": (np.log1p, np.expm1),
}


def clamp_log(x: np.ndarray) -> np.ndarray:
    """Cap log-space samples at `EMIT_LOG_CEIL` (WARN if any are clamped)."""
    x = np.asarray(x, dtype=np.float64)
    over = x > EMIT_LOG_CEIL
    if np.any(over):
        logger.warning(
            "parametric: clamped %d log-space sample(s) at EMIT_LOG_CEIL=%s",
            int(over.sum()), EMIT_LOG_CEIL,
        )
        return np.minimum(x, EMIT_LOG_CEIL)
    return x


def clamp_floor(x: np.ndarray) -> np.ndarray:
    """Floor emitted raw-scale samples at `EMIT_FLOOR` (WARN if any are floored).

    Mirror of `clamp_log`: applied on the *raw* emitted scale (after any `expm1`) so a
    left-tail draw from an ℝ-support positive-part family (`gumbel_r`) can never leave the
    model as a negative magnitude.
    """
    x = np.asarray(x, dtype=np.float64)
    under = x < EMIT_FLOOR
    if np.any(under):
        logger.warning(
            "parametric: floored %d sample(s) at EMIT_FLOOR=%s",
            int(under.sum()), EMIT_FLOOR,
        )
        return np.maximum(x, EMIT_FLOOR)
    return x


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


def fit_family(family: str, values: np.ndarray) -> dict:
    if family not in FAMILIES:
        raise ValueError(f"Unknown family {family!r}; available: {sorted(FAMILIES)}")
    return FAMILIES[family][0](values)


def sample_family(family: str, params: dict, size, rng) -> np.ndarray:
    return FAMILIES[family][1](params, size, rng)


def validate_family_transform(family: str, transform: str) -> None:
    """ADR-021/ADR-022: `log1p` is only valid for continuous positive-part families.

    A count/native-zero family with `log1p` is a contract violation (fails loud), not
    a silent no-op.
    """
    if transform not in TRANSFORMS:
        raise ValueError(f"Unknown transform {transform!r}; available: {sorted(TRANSFORMS)}")
    if transform == "log1p" and family in NATIVE_ZERO_FAMILIES:
        raise ValueError(
            f"transform='log1p' is invalid for count family {family!r} — log1p of a "
            f"count is not a count (ADR-021/ADR-022). Use transform='none'."
        )
