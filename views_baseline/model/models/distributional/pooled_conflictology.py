from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from views_baseline.model.defaults import DEFAULT_SEED
from views_baseline.model.frames.input import panel, to_feature_frame, to_level
from views_baseline.model.frames.output import sample_prediction_grid
from views_baseline.model.frames.pooling import window_pool_arrays
from views_baseline.model.grid import train_test_boundary
from views_baseline.model.models.distributional.conflictology import ConflictologyModel

if TYPE_CHECKING:
    import pandas as pd
    from views_frames import FeatureFrame

logger = logging.getLogger(__name__)

MODEL_NAME = "PooledConflictology"

#: ``pool_by`` value for one area holding every cell.
POOL_MAP = "map"

#: The area code for "no area". The VIEWS data codes a cell without a country (``c_id``) and a
#: cell without an admin-1 unit (GAUL admin-1) as -1. Other negative codes are real areas: the
#: Azores admin-1 codes are -3727 to -3730 (views-datafactory ADR-043).
NO_AREA = -1


class PooledConflictology(ConflictologyModel):
    """Conflictology that pools the window over an area (#151).

    `ConflictologyModel` draws a cell's forecast from that cell's last ``window_months``. This
    model draws it from the last ``window_months`` of **every cell in the cell's area**: the
    cells with the same code in the ``pool_by`` column (e.g. country or admin 1), or every cell
    when ``pool_by == "map"``. It answers "does one level on its own do as well?" for ADR-024.

    The window is conflictology's own (`window_pool_arrays`), so each cell contributes exactly the
    rows `ConflictologyModel` would draw from. The cells pooled are the forecast cells, the ones
    present at ``train_end``. A cell coded `NO_AREA` draws from its own window, as
    `ConflictologyModel` does; ``fit`` logs one warning with their number.

    The area code must be a finite integer, the same in every month up to ``train_end``; a cell
    belongs to one area. An area's pool is one array per target, shared by its cells, not copied.
    """

    def __init__(
        self,
        targets: list[str],
        window_months: int,
        partition_dict: dict,
        loa: str,
        n_samples: int,
        pool_by: str,
        seed: int = DEFAULT_SEED,
    ):
        super().__init__(
            targets=targets, window_months=window_months, partition_dict=partition_dict,
            loa=loa, n_samples=n_samples, seed=seed,
        )
        if not isinstance(pool_by, str) or not pool_by:
            raise ValueError(f"{MODEL_NAME}: pool_by must be a column name or {POOL_MAP!r}, "
                             f"got {pool_by!r}.")
        if pool_by in self.targets:
            raise ValueError(f"{MODEL_NAME}: pool_by {pool_by!r} must not be a target.")
        self.pool_by = pool_by
        self.area = None              # {cell id: area code}, NO_AREA for a cell without one

    def fit(self, df: pd.DataFrame | FeatureFrame) -> PooledConflictology:
        """Pool the last ``window_months`` ≤ ``train_end`` of every cell in each area."""
        _, train_end = train_test_boundary(self.partition_dict)
        columns = self.targets if self.pool_by == POOL_MAP else self.targets + [self.pool_by]
        ff = to_feature_frame(df, loa=self.loa, targets=columns)
        time, unit, values = panel(ff, columns)
        codes = None if self.pool_by == POOL_MAP else values.pop(self.pool_by)

        cells, own = window_pool_arrays(time, unit, values, self.targets, self.window_months,
                                        train_end)
        if self.pool_by == POOL_MAP:
            area = {cid: 0 for cid in cells}
        else:
            area = self._area_per_cell(time, unit, codes, cells, train_end)

        members: dict[int, list] = {}
        for cid in cells:
            if area[cid] != NO_AREA:
                members.setdefault(area[cid], []).append(cid)
        area_pool = {
            a: {t: np.concatenate([own[cid][t] for cid in cs]) for t in self.targets}
            for a, cs in members.items()
        }
        self.entity_ids = cells
        self.area = area
        self.hist_per_entity = {
            cid: own[cid] if area[cid] == NO_AREA else area_pool[area[cid]] for cid in cells
        }

        alone = sum(1 for cid in cells if area[cid] == NO_AREA)
        logger.info("%s: pooled %d cells into %d areas by %r.", MODEL_NAME,
                    len(cells) - alone, len(area_pool), self.pool_by)
        if alone:
            logger.warning("%s: %d of %d cells have no area (%r = %d) and draw from their own "
                           "window, as ConflictologyModel does.", MODEL_NAME, alone, len(cells),
                           self.pool_by, NO_AREA)
        return self

    def _area_per_cell(self, time, unit, codes, cells, train_end: int) -> dict:
        """Each forecast cell's area code. Raises unless it is a finite integer, the same in
        every month up to train_end."""
        time, unit, codes = np.asarray(time), np.asarray(unit), np.asarray(codes)
        rows = (time <= train_end) & np.isin(unit, cells)
        unit, codes = unit[rows], codes[rows]
        bad = ~np.isfinite(codes) | (codes != np.round(codes))
        if bad.any():
            cid = int(unit[bad][0])
            raise ValueError(f"{MODEL_NAME}: {self.pool_by!r} must hold integer area codes up to "
                             f"train_end {train_end}; cell {cid} has {codes[bad][0]!r} "
                             f"({int(bad.sum())} rows affected).")
        index = np.searchsorted(cells, unit)
        low = np.full(cells.size, np.inf)
        high = np.full(cells.size, -np.inf)
        np.minimum.at(low, index, codes)
        np.maximum.at(high, index, codes)
        changes = low != high
        if changes.any():
            raise ValueError(f"{MODEL_NAME}: {self.pool_by!r} changes over time for "
                             f"{int(changes.sum())} cells (first: cell {int(cells[changes][0])}); "
                             f"a cell must belong to one area.")
        return {cid: int(code) for cid, code in zip(cells, low)}

    def predict(
        self, df: pd.DataFrame | FeatureFrame, sequence_number: int, output_length: int
    ) -> dict:
        """Dict[str, PredictionFrame], one per target, each (N · output_length, n_samples)."""
        level = to_level(df, loa=self.loa)
        test_start, _ = train_test_boundary(self.partition_dict)
        return sample_prediction_grid(
            entity_ids=self.entity_ids, fitted_state=self.hist_per_entity,
            model_name=MODEL_NAME, targets=self.targets, n_samples=self.n_samples,
            level=level, test_start=test_start,
            sequence_number=sequence_number, output_length=output_length, seed=self.seed,
            draw_cell=lambda cid, t, rng: rng.choice(
                self.hist_per_entity[cid][t], size=self.n_samples, replace=True
            ),
        )
