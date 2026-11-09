"""The seven size classes the knobs are fitted on (ADR-024, "How the knobs are estimated").

0 · 1 · 2–5 · 6–25 · 26–100 · 101–1,000 · more than 1,000 fatalities.
"""

from __future__ import annotations

import numpy as np

# Lower edge of each class; a value belongs to the last class whose edge it reaches.
LOWER_EDGES = np.array([0, 1, 2, 6, 26, 101, 1001], dtype=np.float64)
LABELS = ["0", "1", "2-5", "6-25", "26-100", "101-1000", ">1000"]
N_CLASSES = len(LOWER_EDGES)


def classify(values: np.ndarray) -> np.ndarray:
    """Size class (0..6) of each value, as int8. Values are fatality counts (≥ 0, no NaN)."""
    values = np.asarray(values)
    if np.isnan(values).any():
        raise ValueError("classify: NaN in values.")
    return (np.searchsorted(LOWER_EDGES, values, side="right") - 1).astype(np.int8)
