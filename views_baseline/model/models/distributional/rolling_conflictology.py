from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from views_baseline.model.frames.input import panel, to_feature_frame, to_level
from views_baseline.model.frames.output import sample_prediction_grid
from views_baseline.model.frames.pooling import window_pool_arrays
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
    provided every forecast cell's pool has at least 2 values at both: a 1-value pool consumes
    no randomness, which shifts the shared random stream for the cells after it.

    **Fails loud** if, for any forecast cell, a month after ``train_end`` that lies inside the
    window (``max(train_end + 1, train_end + k − window_months + 1) .. train_end + k``) is missing
    or NaN. Those are the months conflictology never sees; a gap there would silently pull older
    months back into the window and turn this control back into conflictology.
    """

    def _pools_for_origin(self, df, sequence_number: int) -> dict:
        """The pools for one origin: conflictology's window, anchored at ``train_end + k``."""
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
        self._check_months_after_train_end(time, unit, values, train_end, anchor, sequence_number)

        _, pools = window_pool_arrays(
            time, unit, values, self.targets, self.window_months, anchor
        )
        return pools

    def _check_months_after_train_end(
        self, time, unit, values, train_end: int, anchor: int, sequence_number: int
    ) -> None:
        """Every forecast cell must have every month after train_end that lies inside the window,
        none NaN: months ``max(train_end + 1, anchor - window_months + 1) .. anchor``."""
        start = max(train_end + 1, anchor - self.window_months + 1)
        cells = np.asarray(self.entity_ids)
        after = (time >= start) & (time <= anchor)
        row = np.searchsorted(cells, unit[after])          # unit is already restricted to cells
        col = time[after] - start
        present = np.zeros((cells.size, anchor - start + 1), dtype=bool)
        present[row, col] = True
        incomplete = ~present.all(axis=1)
        if incomplete.any():
            examples = [
                (int(cells[i]), [int(start + m) for m in np.nonzero(~present[i])[0][:3]])
                for i in np.nonzero(incomplete)[0][:3]
            ]
            raise ValueError(
                f"RollingConflictology needs every month {start}..{anchor} (after "
                f"train_end {train_end} and inside the window for sequence_number "
                f"{sequence_number}) for every forecast cell, but {int(incomplete.sum())} of "
                f"{cells.size} cells are missing months (cell, first missing months: "
                f"{examples}). Refusing to fall back to an older window."
            )
        for t in self.targets:
            nan = np.isnan(values[t][after])
            if nan.any():
                cids = unit[after][nan]
                months = sorted({int(m) for m in time[after][nan][cids == cids[0]]})[:3]
                raise ValueError(
                    f"RollingConflictology: NaN in target {t!r} for cell {int(cids[0])} in "
                    f"month(s) {months}, after train_end {train_end} and inside the window for "
                    f"sequence_number {sequence_number}. {np.unique(cids).size} cell(s) affected."
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
            model_name="RollingConflictology", targets=self.targets, n_samples=self.n_samples,
            level=level, test_start=test_start,
            sequence_number=sequence_number, output_length=output_length, seed=self.seed,
            draw_cell=lambda cid, t, rng: rng.choice(
                pools[cid][t], size=self.n_samples, replace=True
            ),
        )
