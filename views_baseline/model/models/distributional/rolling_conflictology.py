from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from views_baseline.model.defaults import DEFAULT_SEED
from views_baseline.model.frames.input import panel, to_feature_frame, to_level
from views_baseline.model.frames.output import sample_prediction_grid
from views_baseline.model.frames.pooling import window_pool, window_pool_arrays
from views_baseline.model.grid import train_test_boundary

if TYPE_CHECKING:
    import pandas as pd
    from views_frames import FeatureFrame


class RollingConflictology:
    """Conflictology whose window moves with each rolling origin (#126, #82; ADR-024's control).

    `ConflictologyModel` draws every origin's forecast from one window ending at ``train_end``.
    This model draws origin ``k``'s forecast from the ``window_months`` ending at
    ``train_end + k``: the month just before that origin's first forecast month
    (``test_start + k``). That is the information rule views-stepshifter uses for the same
    rolling evaluation, so no forecast month is ever inside its own window.

    Everything else is conflictology's: the same pooling (`window_pool`), the same draw
    (`rng.choice` over the pool), the same RNG seeding. At origin 0 the window ends at
    ``train_end``, so origin 0's output is byte-identical to `ConflictologyModel`'s.

    **Fails loud** if the input does not reach ``train_end + k`` for some cell. A silent fall
    back to an older window would turn this control back into conflictology without saying so.
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
        self.partition_dict = partition_dict
        self.loa = loa
        self.window_months = window_months
        self.n_samples = n_samples
        self.seed = seed
        self.entity_ids = None
        self.hist_per_entity = None  # the origin-0 pools (window ending at train_end)

    def fit(self, df: pd.DataFrame | FeatureFrame) -> RollingConflictology:
        """Fix the entity set and the origin-0 pools (identical to `ConflictologyModel.fit`)."""
        _, train_end = train_test_boundary(self.partition_dict)
        ff = to_feature_frame(df, loa=self.loa, targets=self.targets)
        self.entity_ids, self.hist_per_entity = window_pool(
            ff, self.targets, self.window_months, train_end
        )
        return self

    def _pools_for_origin(self, df, sequence_number: int) -> dict:
        """The pools for one origin: the ``window_months`` ending at ``train_end + k``."""
        if sequence_number == 0:
            return self.hist_per_entity
        _, train_end = train_test_boundary(self.partition_dict)
        anchor = train_end + sequence_number
        ff = to_feature_frame(df, loa=self.loa, targets=self.targets)
        time, unit, values = panel(ff, self.targets)
        entity_ids, pools = window_pool_arrays(
            time, unit, values, self.targets, self.window_months, anchor
        )
        reached = set(np.asarray(unit)[np.asarray(time) == anchor].tolist())
        missing = [cid for cid in self.entity_ids if cid not in reached]
        if missing:
            raise ValueError(
                f"RollingConflictology needs observed data up to month {anchor} "
                f"(train_end {train_end} + sequence_number {sequence_number}) for every cell, "
                f"but {len(missing)} of {len(self.entity_ids)} cells have no row at month "
                f"{anchor} (e.g. {missing[:3]}). Refusing to fall back to an older window."
            )
        return pools

    def predict(
        self, df: pd.DataFrame | FeatureFrame, sequence_number: int, output_length: int
    ) -> dict:
        """Dict[str, PredictionFrame], one per target, each (N, n_samples)."""
        pools = self._pools_for_origin(df, sequence_number)
        level = to_level(df, loa=self.loa)
        test_start, _ = train_test_boundary(self.partition_dict)
        return sample_prediction_grid(
            entity_ids=self.entity_ids, fitted_state=pools,
            model_name="RollingConflictology", targets=self.targets, n_samples=self.n_samples,
            level=level, test_start=test_start,
            sequence_number=sequence_number, output_length=output_length, seed=self.seed,
            draw_cell=lambda cid, t, rng: rng.choice(
                pools[cid][t], size=self.n_samples, replace=True
            ),
        )
