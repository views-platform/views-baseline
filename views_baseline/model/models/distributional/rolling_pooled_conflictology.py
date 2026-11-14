from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from views_baseline.model.frames.input import panel, to_feature_frame, to_level
from views_baseline.model.frames.output import sample_prediction_grid
from views_baseline.model.frames.pooling import window_pool_arrays
from views_baseline.model.grid import train_test_boundary
from views_baseline.model.models.distributional.pooled_conflictology import PooledConflictology
from views_baseline.model.models.distributional.rolling_conflictology import (
    check_months_after_train_end,
)

if TYPE_CHECKING:
    import pandas as pd
    from views_frames import FeatureFrame

MODEL_NAME = "RollingPooledConflictology"


class RollingPooledConflictology(PooledConflictology):
    """Pooled conflictology whose window moves with each rolling origin (#165).

    `PooledConflictology` draws every origin's forecast from the area windows ending at
    ``train_end``. This model draws origin ``k``'s forecast from the area windows ending at
    ``train_end + k``, the information rule of `RollingConflictology` and of Matryoshka
    (ADR-024), so a comparison with Matryoshka differs only in the level, not in how recent the
    data is.

    A cell's area is the one ``fit`` read up to ``train_end``; codes after ``train_end`` are never
    read. Origin 0 is byte-identical to `PooledConflictology`. The common-random-numbers caveat of
    `RollingConflictology` applies: the same seed picks the same pool positions at every origin
    only while every pool keeps its length, i.e. every forecast cell has a full window.

    **Fails loud** if, for any forecast cell, a month after ``train_end`` inside the window is
    missing or NaN (`check_months_after_train_end`): a gap would silently pull older months back
    into the window.
    """

    model_name = MODEL_NAME

    def _pools_for_origin(self, df, sequence_number: int) -> dict:
        """The pools for one origin: each area's windows, anchored at ``train_end + k``."""
        if sequence_number < 0:
            raise ValueError(f"sequence_number must be >= 0, got {sequence_number}.")
        if sequence_number == 0:
            return self.hist_per_entity
        _, train_end = train_test_boundary(self.partition_dict)
        anchor = train_end + sequence_number
        ff = to_feature_frame(df, loa=self.loa, targets=self.targets)
        time, unit, values = panel(ff, self.targets)
        time, unit = np.asarray(time), np.asarray(unit)

        # Only the forecast cells are pooled, so a cell that is never forecast cannot stop the run.
        fitted = np.isin(unit, np.asarray(self.entity_ids))
        time, unit = time[fitted], unit[fitted]
        values = {t: values[t][fitted] for t in self.targets}
        check_months_after_train_end(
            time, unit, values, targets=self.targets, entity_ids=self.entity_ids,
            window_months=self.window_months, train_end=train_end, anchor=anchor,
            sequence_number=sequence_number, model_name=MODEL_NAME,
        )
        # The check guarantees every fitted cell has the anchor month, so the cells are the same.
        _, own = window_pool_arrays(time, unit, values, self.targets, self.window_months, anchor)
        return self._pools(own, self.entity_ids, self.area)

    def predict(
        self, df: pd.DataFrame | FeatureFrame, sequence_number: int, output_length: int
    ) -> dict:
        """Dict[str, PredictionFrame], one per target, each (N · output_length, n_samples)."""
        pools = self._pools_for_origin(df, sequence_number)
        level = to_level(df, loa=self.loa)
        test_start, _ = train_test_boundary(self.partition_dict)
        return sample_prediction_grid(
            entity_ids=self.entity_ids, fitted_state=pools,
            model_name=MODEL_NAME, targets=self.targets, n_samples=self.n_samples,
            level=level, test_start=test_start,
            sequence_number=sequence_number, output_length=output_length, seed=self.seed,
            draw_cell=lambda cid, t, rng: rng.choice(
                pools[cid][t], size=self.n_samples, replace=True
            ),
        )
