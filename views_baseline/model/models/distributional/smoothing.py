"""Shared rules for the smoothed climatology baselines (ADR-023 §2).

B (`InterpolatedQuantileConflictology`), C and E all smooth only the **positive** part of a
cell's window; the zero mass stays the window's empirical zero rate, exactly as in
`ConflictologyModel`. This module holds the parts of that rule they share:

- `MIN_POSITIVES_FOR_SMOOTHING`, the single-sourced threshold below which a cell is not
  smoothed;
- `classify_window`, which decides per cell whether to smooth or fall back;
- `draw_fallback`, the fallback draw (A's own resampling, so the fallback *is* A);
- `round_positive`, the integer rule for positive draws;
- `warn_fallbacks`, the one aggregate WARN per `predict()` (ADR-008).

Pure module: numpy and logging only.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)

#: A cell with fewer positive values than this is resampled empirically, not smoothed.
#: Two is the least an interpolation or a kernel can spread (ADR-023 §2).
MIN_POSITIVES_FOR_SMOOTHING = 2

#: Per-cell outcome of `classify_window`. Only SMOOTH cells are smoothed.
SMOOTH = "smooth"
ALL_ZERO = "all_zero"
TOO_FEW_POSITIVES = "too_few_positives"
NO_SPREAD = "no_spread"


def classify_window(pool: np.ndarray) -> tuple[str, np.ndarray]:
    """Decide how one cell's window is drawn from.

    Returns ``(kind, positives)`` where ``positives`` are the window's positive values,
    sorted ascending. ``kind`` is:

    - ``ALL_ZERO``: no positive value. The cell is a point mass at 0 (what A gives too).
    - ``TOO_FEW_POSITIVES``: fewer than ``MIN_POSITIVES_FOR_SMOOTHING`` positive values.
    - ``NO_SPREAD``: every positive value is the same, so there is nothing to smooth.
    - ``SMOOTH``: smooth the positives.

    An empty window cannot reach here: `window_pool` only returns entities with at least
    one row in the window.
    """
    positives = np.sort(pool[pool > 0])
    if positives.size == 0:
        return ALL_ZERO, positives
    if positives.size < MIN_POSITIVES_FOR_SMOOTHING:
        return TOO_FEW_POSITIVES, positives
    if positives[0] == positives[-1]:
        return NO_SPREAD, positives
    return SMOOTH, positives


def draw_fallback(pool: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """The draw for a cell that is not smoothed: A's resampling of the whole window.

    Using A's own rule keeps the fallback's zero mass and positive values exactly A's,
    in distribution. An all-zero window gives a point mass at 0.
    """
    return rng.choice(pool, size=n_samples, replace=True)


def round_positive(x: np.ndarray) -> np.ndarray:
    """Round smoothed positive values to counts ≥ 1 (ADR-023 §2, integer output).

    Rounds half up (``floor(x + 0.5)``), not half-to-even: C's reflection puts values at
    exactly 0.5, which must land on 1, not 0. The floor at 1 keeps a positive draw positive
    even if the window held a fractional value below 0.5; otherwise smoothing would move
    mass onto 0 and change the zero mass.
    """
    return np.maximum(np.floor(x + 0.5), 1.0)


def warn_fallbacks(model_name: str, kinds: Counter) -> None:
    """Log one aggregate WARN for the cells that were not smoothed (ADR-008).

    ``kinds`` counts per-(entity, target) outcomes of `classify_window`. Silent when every
    cell was smoothed.
    """
    n_cells = sum(kinds.values())
    not_smoothed = n_cells - kinds[SMOOTH]
    if not_smoothed == 0:
        return
    detail = ", ".join(f"{k}={kinds[k]}" for k in sorted(kinds) if k != SMOOTH and kinds[k])
    logger.warning(
        "%s: %d of %d (entity, target) cells not smoothed, drawn as ConflictologyModel "
        "draws them (%s). MIN_POSITIVES_FOR_SMOOTHING=%d.",
        model_name, not_smoothed, n_cells, detail, MIN_POSITIVES_FOR_SMOOTHING,
    )
