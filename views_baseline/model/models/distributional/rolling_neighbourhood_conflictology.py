from __future__ import annotations

from views_baseline.model.frames.pooling import rolling_window_pools
from views_baseline.model.grid import train_test_boundary
from views_baseline.model.models.distributional.neighbourhood_conflictology import (
    NeighbourhoodConflictology,
)

MODEL_NAME = "RollingNeighbourhoodConflictology"


class RollingNeighbourhoodConflictology(NeighbourhoodConflictology):
    """Neighbourhood conflictology whose window moves with each rolling origin (#178).

    `NeighbourhoodConflictology` draws every origin's forecast from the disc's windows ending at
    ``train_end``. This model draws origin ``k``'s forecast from the disc's windows ending at
    ``train_end + k``, the information rule of `RollingConflictology` and of Matryoshka
    (ADR-024), so a comparison with Matryoshka differs only in the level, not in how recent the
    data is.

    The disc is the one ``fit`` built from the cells present at ``train_end``. Origin 0 is
    byte-identical to `NeighbourhoodConflictology`. The common-random-numbers caveat of
    `RollingConflictology` applies: the same seed picks the same pool positions at every origin
    only while every disc keeps its total length, i.e. every forecast cell has a full window.

    **Fails loud** if, for any forecast cell, a month after ``train_end`` inside the window is
    missing or NaN (`rolling_window_pools`): a gap would silently pull older months back into
    the window.
    """

    model_name = MODEL_NAME

    def _windows_for_origin(self, df, sequence_number: int) -> dict:
        """The windows ending at ``train_end + k``, read with `rolling_window_pools`."""
        if sequence_number < 0:
            raise ValueError(f"{self.model_name}: sequence_number must be >= 0, "
                             f"got {sequence_number}.")
        if sequence_number == 0:
            return self.windows
        _, train_end = train_test_boundary(self.partition_dict)
        own = rolling_window_pools(
            df, loa=self.loa, targets=self.targets, entity_ids=self.entity_ids,
            window_months=self.window_months, train_end=train_end,
            sequence_number=sequence_number, model_name=self.model_name,
        )
        return self._windows(own)
