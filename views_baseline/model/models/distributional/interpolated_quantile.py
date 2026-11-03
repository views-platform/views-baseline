from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from views_baseline.model.defaults import DEFAULT_SEED
from views_baseline.model.frames.input import to_feature_frame, to_level
from views_baseline.model.frames.output import sample_prediction_grid
from views_baseline.model.frames.pooling import window_pool
from views_baseline.model.grid import train_test_boundary
from views_baseline.model.models.distributional.smoothing import (
    SMOOTH,
    classify_window,
    draw_fallback,
    round_positive,
    warn_fallbacks,
)

if TYPE_CHECKING:
    import pandas as pd
    from views_frames import FeatureFrame


class InterpolatedQuantileConflictology:
    """Climatology with an interpolated quantile function for the positive part (ADR-023, B).

    Per entity and target, on the same window as `ConflictologyModel`:

    - a draw is 0 with probability equal to the window's empirical zero rate (as in A);
    - otherwise it is ``round(Q₊(U))``, ``U ~ Uniform(0, 1)``, where ``Q₊`` linearly
      interpolates the sorted positive values ``p(1) ≤ … ≤ p(m)`` at plotting positions
      ``(i − 1)/(m − 1)``.

    Draws are integers; positive draws are ≥ 1 and never leave ``[p(1), p(m)]``: there is
    no extrapolation (ADR-023 §3). Cells with fewer than `MIN_POSITIVES_FOR_SMOOTHING`
    positive values, or with all positive values equal, are drawn exactly as A draws
    them, and are counted in one WARN per `predict()`.
    """

    distributional = True

    def __init__(
        self,
        targets: list[str],
        window_months: int,
        partition_dict: dict,
        loa: str,
        n_samples: int,
        seed: int = DEFAULT_SEED,
    ):
        self.targets = targets
        self.window_months = window_months
        self.partition_dict = partition_dict
        self.loa = loa
        self.n_samples = n_samples
        self.seed = seed
        self.entity_ids = None
        # per cid/target: {"kind", "zero_rate", "positions", "positives"} or {"kind", "pool"}
        self.cells = None
        self.fallback_counts: Counter = Counter()

    def fit(self, df: pd.DataFrame | FeatureFrame) -> InterpolatedQuantileConflictology:
        _, train_end = train_test_boundary(self.partition_dict)
        ff = to_feature_frame(df, loa=self.loa, targets=self.targets)
        # Same pool as ConflictologyModel, byte for byte (ADR-023 §1).
        self.entity_ids, pools = window_pool(ff, self.targets, self.window_months, train_end)

        self.cells = {}
        self.fallback_counts = Counter()
        for cid in self.entity_ids:
            self.cells[cid] = {}
            for t in self.targets:
                pool = pools[cid][t]
                kind, positives = classify_window(pool)
                self.fallback_counts[kind] += 1
                if kind == SMOOTH:
                    m = positives.size
                    self.cells[cid][t] = {
                        "kind": kind,
                        "zero_rate": float(np.mean(pool == 0.0)),
                        "positions": np.arange(m) / (m - 1),
                        "positives": positives,
                    }
                else:
                    self.cells[cid][t] = {"kind": kind, "pool": pool}
        return self

    def predict(
        self, df: pd.DataFrame | FeatureFrame, sequence_number: int, output_length: int
    ) -> dict:
        def draw(cid, t, rng):
            cell = self.cells[cid][t]
            if cell["kind"] != SMOOTH:
                return draw_fallback(cell["pool"], self.n_samples, rng)
            draws = np.zeros(self.n_samples, dtype=np.float64)
            is_positive = rng.random(self.n_samples) >= cell["zero_rate"]
            u = rng.random(int(is_positive.sum()))
            draws[is_positive] = round_positive(
                np.interp(u, cell["positions"], cell["positives"])
            )
            return draws

        warn_fallbacks("InterpolatedQuantileConflictology", self.fallback_counts)
        level = to_level(df, loa=self.loa)
        test_start, _ = train_test_boundary(self.partition_dict)
        return sample_prediction_grid(
            entity_ids=self.entity_ids, fitted_state=self.cells,
            model_name="InterpolatedQuantileConflictology", targets=self.targets,
            n_samples=self.n_samples, level=level, test_start=test_start,
            sequence_number=sequence_number, output_length=output_length, seed=self.seed,
            draw_cell=draw,
        )
