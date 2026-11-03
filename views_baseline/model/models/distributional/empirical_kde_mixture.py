from __future__ import annotations

import math
import numbers
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from views_baseline.model.defaults import DEFAULT_SEED
from views_baseline.model.frames.input import to_feature_frame, to_level
from views_baseline.model.frames.output import sample_prediction_grid
from views_baseline.model.frames.pooling import window_pool
from views_baseline.model.grid import train_test_boundary
from views_baseline.model.models.distributional.kde import (
    draw_kde_cell,
    fit_kde_cell,
    validate_bandwidth,
)
from views_baseline.model.models.distributional.smoothing import draw_fallback, warn_fallbacks

if TYPE_CHECKING:
    import pandas as pd
    from views_frames import FeatureFrame


def validate_alpha(alpha) -> None:
    """Fail loud unless `alpha` is a real number in [0, 1] (not bool)."""
    if isinstance(alpha, bool) or not isinstance(alpha, numbers.Real):
        raise ValueError(f"alpha must be a number in [0, 1], got {alpha!r}.")
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha!r}.")


class EmpiricalKDEMixtureConflictology:
    """Mixture of conflictology (A) and its KDE smoothing (C): ``F = (1−α)·F_A + α·F_C``
    (ADR-023, E).

    A mixture of **distributions**, not an average of draws. Per cell and per call, E draws,
    in this fixed order:

    1. ``n_samples`` uniforms ``U``;
    2. ``n_samples`` A-draws (resampling the window, as `ConflictologyModel`);
    3. ``n_samples`` C-draws (as `KDEConflictology` with the same ``bandwidth``);

    and keeps the C-draw where ``U < α``, otherwise the A-draw.

    **Common random numbers.** Nothing E consumes depends on ``α``, so at a fixed seed
    raising ``α`` only switches draws from A to C: an α-sweep is a coupled (paired) design.
    At ``α = 0`` E equals A in distribution and at ``α = 1`` C in distribution; it is not
    byte-identical to either, because it consumes all three streams.

    Zero mass equals A's (both components have it). Mixing can only widen spread, never
    sharpen it (Gneiting & Ranjan 2013, Thm 3.1), so ``α`` is a spread dial between A and C.
    Fallback cells (C does not smooth them) are drawn from A in both slots, counted in one
    WARN per `predict()`.
    """

    distributional = True

    def __init__(
        self,
        targets: list[str],
        window_months: int,
        partition_dict: dict,
        loa: str,
        n_samples: int,
        alpha: float,
        bandwidth: float | str,
        seed: int = DEFAULT_SEED,
    ):
        validate_alpha(alpha)
        validate_bandwidth(bandwidth)
        self.targets = targets
        self.window_months = window_months
        self.partition_dict = partition_dict
        self.loa = loa
        self.n_samples = n_samples
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.seed = seed
        self.entity_ids = None
        self.cells = None  # per cid/target: kde.fit_kde_cell's dict
        self.fallback_counts: Counter = Counter()

    def fit(self, df: pd.DataFrame | FeatureFrame) -> EmpiricalKDEMixtureConflictology:
        _, train_end = train_test_boundary(self.partition_dict)
        ff = to_feature_frame(df, loa=self.loa, targets=self.targets)
        # Same pool as ConflictologyModel and KDEConflictology (ADR-023 §1).
        self.entity_ids, pools = window_pool(ff, self.targets, self.window_months, train_end)

        self.cells = {}
        self.fallback_counts = Counter()
        for cid in self.entity_ids:
            self.cells[cid] = {}
            for t in self.targets:
                cell = fit_kde_cell(pools[cid][t], self.bandwidth)
                self.fallback_counts[cell["kind"]] += 1
                self.cells[cid][t] = cell
        return self

    def predict(
        self, df: pd.DataFrame | FeatureFrame, sequence_number: int, output_length: int
    ) -> dict:
        def draw(cid, t, rng):
            cell = self.cells[cid][t]
            u = rng.random(self.n_samples)
            a_draws = draw_fallback(cell["pool"], self.n_samples, rng)
            c_draws = draw_kde_cell(cell, self.n_samples, rng)
            return np.where(u < self.alpha, c_draws, a_draws)

        warn_fallbacks("EmpiricalKDEMixtureConflictology", self.fallback_counts)
        level = to_level(df, loa=self.loa)
        test_start, _ = train_test_boundary(self.partition_dict)
        return sample_prediction_grid(
            entity_ids=self.entity_ids, fitted_state=self.cells,
            model_name="EmpiricalKDEMixtureConflictology", targets=self.targets,
            n_samples=self.n_samples, level=level, test_start=test_start,
            sequence_number=sequence_number, output_length=output_length, seed=self.seed,
            draw_cell=draw,
        )
