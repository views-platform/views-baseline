from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from views_baseline.model.defaults import DEFAULT_SEED
from views_baseline.model.frames.input import to_level
from views_baseline.model.frames.output import sample_prediction_grid
from views_baseline.model.grid import train_test_boundary
from views_baseline.model.models.distributional.conflictology import ConflictologyModel
from views_baseline.model.models.distributional.matryoshka.geometry import neighbour_table
from views_baseline.model.spatial import spatial_level

if TYPE_CHECKING:
    import pandas as pd
    from views_frames import FeatureFrame

MODEL_NAME = "NeighbourhoodConflictology"


class NeighbourhoodConflictology(ConflictologyModel):
    """Conflictology that pools the window over a disc of cells around each cell (#178).

    `ConflictologyModel` draws a cell's forecast from that cell's last ``window_months``. This
    model draws it from the last ``window_months`` of **every forecast cell within grid distance
    ``radius``** of the cell, itself included, every cell and month with the same weight. It is
    the single-level control for Matryoshka's ``near`` doll (ADR-024): with ``radius`` 5 the disc
    is that doll's neighbourhood plus the cell. The disc comes from Matryoshka's own
    `neighbour_table`, so the two cannot drift apart. ``radius`` 0 is the cell alone, which is
    `ConflictologyModel` byte for byte.

    The draws are those of ``rng.choice`` over the disc's windows concatenated in ascending cell
    order, as `PooledConflictology` draws from an area. The pools are not built: a radius-5 disc
    holds about 81 cells, and copying them per cell would take gigabytes. A draw picks positions
    ``0 .. total − 1`` with ``rng.choice(total)``, which uses the random stream exactly as
    ``rng.choice(pool)`` does, and maps each position to its cell and month.

    pgm only: the disc comes from the priogrid ids.
    """

    model_name = MODEL_NAME

    def __init__(
        self,
        targets: list[str],
        window_months: int,
        partition_dict: dict,
        loa: str,
        n_samples: int,
        radius: int,
        seed: int = DEFAULT_SEED,
    ):
        super().__init__(
            targets=targets, window_months=window_months, partition_dict=partition_dict,
            loa=loa, n_samples=n_samples, seed=seed,
        )
        if isinstance(radius, bool) or not isinstance(radius, (int, np.integer)) or radius < 0:
            raise ValueError(f"{self.model_name}: radius must be an integer >= 0 (grid cells), "
                             f"got {radius!r}.")
        self.radius = int(radius)
        self.disc = None      # (N, 1 + n_offsets) indices into entity_ids, ascending, -1 padded
        self.windows = None   # the fitted windows, see `_windows`

    def fit(self, df: pd.DataFrame | FeatureFrame) -> NeighbourhoodConflictology:
        """Each forecast cell's window (conflictology's) and its disc of forecast cells."""
        from views_frames import SpatialLevel

        if spatial_level(self.loa) != SpatialLevel.PGM:
            raise ValueError(f"{self.model_name} is defined at grid-month (pgm) only: the disc "
                             f"comes from the priogrid ids; got loa={self.loa!r}.")
        super().fit(df)
        self.disc = self._disc(np.asarray(self.entity_ids))
        self.windows = self._windows(self.hist_per_entity)
        return self

    def _disc(self, cells: np.ndarray) -> np.ndarray:
        """Each cell's disc: itself and its neighbours within ``radius``, as indices into
        ``cells`` in ascending order, padded with -1 at the end."""
        members = np.concatenate([np.arange(cells.size)[:, None],
                                  neighbour_table(cells, self.radius)], axis=1)
        members = np.where(members < 0, cells.size, members)     # absent sorts last
        members = np.sort(members, axis=1)
        return np.where(members == cells.size, -1, members)

    def _windows(self, own: dict) -> dict:
        """The windows as arrays: ``values[t]`` (N, window_months), each cell's window left
        aligned in time order, and ``length`` (N,), its number of months."""
        cells = self.entity_ids
        length = np.array([len(own[cid][self.targets[0]]) for cid in cells])
        values = {}
        for t in self.targets:
            dtype = own[cells[0]][t].dtype if len(cells) else float
            v = np.zeros((len(cells), self.window_months), dtype=dtype)
            for i, cid in enumerate(cells):
                v[i, :length[i]] = own[cid][t]
            values[t] = v
        return {"values": values, "length": length}

    def _windows_for_origin(self, df, sequence_number: int) -> dict:
        """The windows to draw from at this origin: the fitted ones (the window ends at
        ``train_end`` at every origin, as in `ConflictologyModel`)."""
        return self.windows

    def predict(
        self, df: pd.DataFrame | FeatureFrame, sequence_number: int, output_length: int
    ) -> dict:
        """Dict[str, PredictionFrame], one per target, each (N · output_length, n_samples)."""
        windows = self._windows_for_origin(df, sequence_number)
        values, length = windows["values"], windows["length"]
        # Per cell, once per origin: its disc members and where each member's months end in the
        # concatenated pool. Every (month, target) draw of the cell reuses them.
        disc = {}
        for i, cid in enumerate(self.entity_ids):
            members = self.disc[i][self.disc[i] >= 0]
            end = np.cumsum(length[members])
            disc[cid] = (members, end, end - length[members])

        def draw_cell(cid, t, rng):
            members, end, start = disc[cid]
            j = rng.choice(end[-1], size=self.n_samples, replace=True)
            k = np.searchsorted(end, j, side="right")
            return values[t][members[k], j - start[k]]

        level = to_level(df, loa=self.loa)
        test_start, _ = train_test_boundary(self.partition_dict)
        return sample_prediction_grid(
            entity_ids=self.entity_ids, fitted_state=self.hist_per_entity,
            model_name=self.model_name, targets=self.targets, n_samples=self.n_samples,
            level=level, test_start=test_start,
            sequence_number=sequence_number, output_length=output_length, seed=self.seed,
            draw_cell=draw_cell,
        )
