from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from views_baseline.model.frames.input import panel, to_feature_frame, to_level
from views_baseline.model.frames.output import sample_prediction_grid
from views_baseline.model.grid import train_test_boundary
from views_baseline.model.models.distributional.conflictology import ConflictologyModel

if TYPE_CHECKING:
    import pandas as pd
    from views_frames import FeatureFrame


class RollingConflictology(ConflictologyModel):
    """Conflictology whose window moves with each rolling origin (#126, #82; ADR-024's control).

    `ConflictologyModel` draws every origin's forecast from one window ending at ``train_end``.
    This model draws origin ``k``'s forecast from the ``window_months`` calendar months ending
    at ``train_end + k``: the month just before that origin's first forecast month
    (``test_start + k``). That is the information rule views-stepshifter uses for the same
    rolling evaluation, so no forecast month is ever inside its own window.

    It is a subclass, so ``__init__``, ``fit`` and the stored origin-0 pools *are*
    conflictology's. Origin 0 is therefore byte-identical to `ConflictologyModel`, and at every
    origin the same seed picks the same pool positions (common random numbers), which keeps the
    comparison with conflictology paired.

    **Fails loud** if, for any forecast cell, the window ``train_end + k − window_months + 1 ..
    train_end + k`` is not complete in the input (a missing month or a NaN). A silent fall back
    to an older window would turn this control back into conflictology without saying so.
    """

    def _pools_for_origin(self, df, sequence_number: int) -> dict:
        """The pools for one origin: the ``window_months`` months ending at ``train_end + k``.

        Reads the window directly by (cell, month) position for the fitted cells only: one pass
        over the panel, no per-cell scan (measured on the 24.9M-row pgm `land` panel: about
        0.5 s, against about 14 minutes for re-running `window_pool_arrays`). On a complete
        panel the result equals `window_pool_arrays` anchored at ``train_end + k``.
        """
        if sequence_number == 0:
            return self.hist_per_entity
        _, train_end = train_test_boundary(self.partition_dict)
        anchor = train_end + sequence_number
        first = anchor - self.window_months + 1
        ff = to_feature_frame(df, loa=self.loa, targets=self.targets)
        time, unit, values = panel(ff, self.targets)
        time, unit = np.asarray(time), np.asarray(unit)

        cells = np.asarray(self.entity_ids)          # ascending (np.unique at fit)
        in_window = (time >= first) & (time <= anchor)
        row = np.searchsorted(cells, unit[in_window])
        known = row < cells.size
        known[known] = cells[row[known]] == unit[in_window][known]
        row = row[known]
        col = (time[in_window] - first)[known]

        present = np.zeros((cells.size, self.window_months), dtype=bool)
        present[row, col] = True
        incomplete = ~present.all(axis=1)
        if incomplete.any():
            examples = [
                (int(cells[i]), [int(first + m) for m in np.nonzero(~present[i])[0][:3]])
                for i in np.nonzero(incomplete)[0][:3]
            ]
            raise ValueError(
                f"RollingConflictology needs every month {first}..{anchor} (the window for "
                f"sequence_number {sequence_number}, ending at train_end {train_end} + "
                f"{sequence_number}) for every forecast cell, but {int(incomplete.sum())} of "
                f"{cells.size} cells are missing months (cell, first missing months: "
                f"{examples}). Refusing to fall back to an older window."
            )

        dense = {}
        for t in self.targets:
            block = np.empty((cells.size, self.window_months), dtype=np.float64)
            block[row, col] = values[t][in_window][known]
            bad = np.isnan(block).any(axis=1)
            if bad.any():
                i = int(np.nonzero(bad)[0][0])
                months = [int(first + m) for m in np.nonzero(np.isnan(block[i]))[0][:3]]
                raise ValueError(
                    f"RollingConflictology: NaN in target {t!r} for cell {int(cells[i])} in "
                    f"month(s) {months}, inside the window {first}..{anchor} for "
                    f"sequence_number {sequence_number}. {int(bad.sum())} cell(s) affected."
                )
            dense[t] = block
        return {
            cid: {t: dense[t][i] for t in self.targets}
            for i, cid in enumerate(self.entity_ids)
        }

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
