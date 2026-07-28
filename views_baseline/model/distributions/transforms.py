"""Log-space transforms and emission clamps for the parametric families.

`TRANSFORMS` holds the `(forward, inverse)` pairs applied to a positive part before
fitting and inverted per sampled draw (Jensen-safe). `EMIT_LOG_CEIL` / `EMIT_FLOOR` are
the single-sourced clamps around `expm1` (overflow guard + non-negativity guard —
mirrors hydranet C-113). Pure module: numpy only; no family or model knowledge.
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
