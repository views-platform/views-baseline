"""Family dispatch + family×transform validation (the strategy registry, ADR-022).

Composition point: dispatches `fit`/`sample` over the family strategies in
:mod:`families` and validates a requested `family`×`transform` pairing against the
`TRANSFORMS` registry in :mod:`transforms`. Depends on both leaf modules; nothing
depends back on it (acyclic — ADR-002/ADP).
"""

from __future__ import annotations

import numpy as np

from views_baseline.model.distributions.families import FAMILIES, NATIVE_ZERO_FAMILIES
from views_baseline.model.distributions.transforms import TRANSFORMS


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
