from __future__ import annotations

from typing import TYPE_CHECKING

from views_baseline.model.frames.input import to_level
from views_baseline.model.frames.output import sample_prediction_grid
from views_baseline.model.frames.pooling import rolling_window_pools
from views_baseline.model.grid import train_test_boundary
from views_baseline.model.models.distributional.conflictology import ConflictologyModel

if TYPE_CHECKING:
    import pandas as pd
    from views_frames import FeatureFrame


class RollingConflictology(ConflictologyModel):
    """Conflictology whose window moves with each rolling origin (#126, #82; ADR-024's control).

    `ConflictologyModel` draws every origin's forecast from one window ending at ``train_end``.
    This model draws origin ``k``'s forecast from the window ending at ``train_end + k``: the month
    just before that origin's first forecast month (``test_start + k``). That is the information
    rule views-stepshifter uses for the same rolling evaluation, so no forecast month is ever
    inside its own window.

    The window is conflictology's own: the last ``window_months`` rows per cell, from the shared
    `window_pool_arrays`, only anchored later. It is a subclass, so ``__init__``, ``fit`` and the
    origin-0 pools *are* conflictology's: origin 0 is byte-identical to `ConflictologyModel`.
    At every origin the same seed picks the same pool positions (common random numbers),
    provided every forecast cell has a full ``window_months`` history at ``train_end``, so the
    two models' pools have the same length. With a shorter history the same random words map to
    different positions for that cell; a 1-value pool also consumes no randomness, which shifts
    the shared random stream for every cell after it.

    **Fails loud** if, for any forecast cell, a month after ``train_end`` that lies inside the
    window (``max(train_end + 1, train_end + k − window_months + 1) .. train_end + k``) is missing
    or NaN. Those are the months conflictology never sees; a gap there would silently pull older
    months back into the window and turn this control back into conflictology.
    """

    model_name = "RollingConflictology"

    def _pools_for_origin(self, df, sequence_number: int) -> dict:
        """The pools for one origin: conflictology's window, anchored at ``train_end + k``."""
        if sequence_number < 0:
            raise ValueError(f"{self.model_name}: sequence_number must be >= 0, "
                             f"got {sequence_number}.")
        if sequence_number == 0:
            return self.hist_per_entity
        _, train_end = train_test_boundary(self.partition_dict)
        return rolling_window_pools(
            df, loa=self.loa, targets=self.targets, entity_ids=self.entity_ids,
            window_months=self.window_months, train_end=train_end,
            sequence_number=sequence_number, model_name=self.model_name,
        )

    def predict(
        self, df: pd.DataFrame | FeatureFrame, sequence_number: int, output_length: int
    ) -> dict:
        """Dict[str, PredictionFrame], one per target, each (N, n_samples)."""
        pools = self._pools_for_origin(df, sequence_number)
        level = to_level(df, loa=self.loa)
        test_start, _ = train_test_boundary(self.partition_dict)
        return sample_prediction_grid(
            entity_ids=self.entity_ids, fitted_state=pools,
            model_name=self.model_name, targets=self.targets, n_samples=self.n_samples,
            level=level, test_start=test_start,
            sequence_number=sequence_number, output_length=output_length, seed=self.seed,
            draw_cell=lambda cid, t, rng: rng.choice(
                pools[cid][t], size=self.n_samples, replace=True
            ),
        )
