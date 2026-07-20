"""Distribution family + transform registries for the parametric climatology models.

A **strategy registry** (ADR-022) split across three cohesive leaves:

* :mod:`transforms` — `TRANSFORMS`, `clamp_log`/`clamp_floor`, `EMIT_*` (pure log math).
* :mod:`families` — per-family `fit_`/`sample_` (method-of-moments; point-mass fallback).
* :mod:`registry` — `fit_family`/`sample_family` dispatch + `validate_family_transform`.

This package re-exports the stable public surface so importers depend on
``views_baseline.model.distributions`` and are insulated from the internal split.
Pure package: numpy + scipy only; no pandas, no views-frames, no model imports.
"""

from views_baseline.model.distributions.families import (
    CONTINUOUS_FAMILIES,
    FAMILIES,
    NATIVE_ZERO_FAMILIES,
    fit_gamma,
    fit_gumbel,
    fit_lognormal,
    fit_nb,
    fit_zinb,
    sample_gamma,
    sample_gumbel,
    sample_lognormal,
    sample_nb,
    sample_zinb,
)
from views_baseline.model.distributions.registry import (
    fit_family,
    sample_family,
    validate_family_transform,
)
from views_baseline.model.distributions.transforms import (
    EMIT_FLOOR,
    EMIT_LOG_CEIL,
    TRANSFORMS,
    clamp_floor,
    clamp_log,
)

__all__ = [
    "CONTINUOUS_FAMILIES",
    "EMIT_FLOOR",
    "EMIT_LOG_CEIL",
    "FAMILIES",
    "NATIVE_ZERO_FAMILIES",
    "TRANSFORMS",
    "clamp_floor",
    "clamp_log",
    "fit_family",
    "fit_gamma",
    "fit_gumbel",
    "fit_lognormal",
    "fit_nb",
    "fit_zinb",
    "sample_family",
    "sample_gamma",
    "sample_gumbel",
    "sample_lognormal",
    "sample_nb",
    "sample_zinb",
    "validate_family_transform",
]
