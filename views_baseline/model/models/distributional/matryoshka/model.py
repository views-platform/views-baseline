"""`MatryoshkaConflictology`: conflictology that also draws from a cell's surroundings (ADR-024).

Data in:   a complete grid-month panel (DataFrame or FeatureFrame) with the targets and the
           country column. fit reads months ≤ train_end; predict reads months ≤ train_end + k.
Transform: fit finds 8 knobs per target by backtest (`knob_fit`); predict draws past observed
           values in three steps (`draw`).
Config:    targets, partition_dict, loa (pgm only), n_samples, seed, country_column,
           backtest_cutoffs, backtest_horizon, stability_refits. Everything else is fixed by
           ADR-024.
Out:       dict[str, PredictionFrame], one per target, each (cells × months, n_samples).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from views_baseline.model.defaults import DEFAULT_SEED
from views_baseline.model.frames.input import panel, to_feature_frame, to_level
from views_baseline.model.frames.output import to_prediction_frames
from views_baseline.model.grid import (
    build_identifier_arrays,
    build_time_grid,
    require_entities,
    train_test_boundary,
)
from views_baseline.model.models.distributional.matryoshka.draw import (
    build_sampler,
    build_shell_tables,
)
from views_baseline.model.models.distributional.matryoshka.geometry import build_geometry
from views_baseline.model.models.distributional.matryoshka.knob_fit import (
    backtest_cutoffs,
    fit_knobs,
)
from views_baseline.model.spatial import spatial_level

if TYPE_CHECKING:
    import pandas as pd
    from views_frames import FeatureFrame

MODEL_NAME = "MatryoshkaConflictology"
BLOCK_CELLS = 256      # cells drawn at once in predict; memory/speed only


def _dense_panel(time: np.ndarray, unit: np.ndarray, values: dict, last_month: int):
    """Months ≤ ``last_month`` as a dense (M, N) panel per column. Fails loud unless every
    cell has every month from the first month on, with no NaN (ADR-024: the shells and their
    sizes assume a complete panel)."""
    keep = np.asarray(time) <= last_month
    time, unit = np.asarray(time)[keep], np.asarray(unit)[keep]
    if time.size == 0:
        raise ValueError(f"{MODEL_NAME}: no data at or before month {last_month}.")
    months, m_idx = np.unique(time, return_inverse=True)
    cells, c_idx = np.unique(unit, return_inverse=True)
    if np.any(np.diff(months) != 1):
        gaps = months[1:][np.diff(months) != 1][:3].tolist()
        raise ValueError(f"{MODEL_NAME}: months are not consecutive (first gaps before "
                         f"{gaps}).")
    count = np.zeros((months.size, cells.size), dtype=np.int64)
    np.add.at(count, (m_idx, c_idx), 1)
    if np.any(count != 1):
        missing = int((count == 0).sum())
        duplicate = int((count > 1).sum())
        raise ValueError(
            f"{MODEL_NAME} needs a complete panel: every cell in every month "
            f"{int(months[0])}..{last_month}, exactly once. Found {missing} missing and "
            f"{duplicate} duplicated cell-months.")
    dense = {}
    for name, v in values.items():
        x = np.empty((months.size, cells.size), dtype=np.float32)
        x[m_idx, c_idx] = np.asarray(v)[keep]
        if np.isnan(x).any():
            s, j = np.argwhere(np.isnan(x))[0]
            raise ValueError(f"{MODEL_NAME}: NaN in {name!r} (cell {int(cells[j])}, month "
                             f"{int(months[s])}); {int(np.isnan(x).sum())} cell-months "
                             f"affected.")
        dense[name] = x
    return months, cells, dense


class MatryoshkaConflictology:
    """See the module docstring, ADR-024 and docs/CICs/MatryoshkaConflictology.md."""

    distributional = True

    def __init__(
        self,
        targets: list[str],
        partition_dict: dict,
        loa: str,
        n_samples: int,
        country_column: str,
        backtest_cutoffs: int,
        backtest_horizon: int,
        stability_refits: bool,
        seed: int = DEFAULT_SEED,
    ):
        self.targets = list(targets)
        self.partition_dict = partition_dict
        self.loa = loa
        self.n_samples = n_samples
        self.country_column = country_column
        self.backtest_cutoffs = backtest_cutoffs
        self.backtest_horizon = backtest_horizon
        self.stability_refits = stability_refits
        self.seed = seed
        if country_column in self.targets:
            raise ValueError(f"country_column {country_column!r} must not be a target.")
        self.entity_ids = None        # (N,) priogrid ids, ascending
        self.country = None           # (N,) GAUL country code per cell
        self.knobs = None             # {target: fit result dict}

    def _check_level(self) -> None:
        from views_frames import SpatialLevel

        if spatial_level(self.loa) != SpatialLevel.PGM:
            raise ValueError(f"{MODEL_NAME} is defined at grid-month (pgm) only; got "
                             f"loa={self.loa!r} (ADR-024: no country-level version yet).")

    def fit(self, df: pd.DataFrame | FeatureFrame) -> MatryoshkaConflictology:
        """Fit the 8 knobs per target on data ≤ train_end."""
        self._check_level()
        _, train_end = train_test_boundary(self.partition_dict)
        columns = self.targets + [self.country_column]
        ff = to_feature_frame(df, loa=self.loa, targets=columns)
        time, unit, values = panel(ff, columns)
        months, cells, dense = _dense_panel(time, unit, values, train_end)
        if months[-1] != train_end:
            raise ValueError(f"{MODEL_NAME}: the data ends at month {int(months[-1])}, before "
                             f"train_end {train_end}.")
        country = dense.pop(self.country_column)
        if np.any(country != country[:1]):
            raise ValueError(f"{MODEL_NAME}: {self.country_column!r} changes over time for "
                             f"some cells; a cell must belong to one country.")
        if np.any(country[0] != np.round(country[0])):
            raise ValueError(f"{MODEL_NAME}: {self.country_column!r} must hold integer "
                             f"country codes.")
        geo = build_geometry(cells, country[0].astype(np.int64))
        cutoffs = backtest_cutoffs(train_end, self.backtest_cutoffs, self.backtest_horizon)
        self.knobs = {
            t: fit_knobs(dense[t].astype(np.float64), months, geo, cutoffs,
                         self.backtest_horizon, self.stability_refits)
            for t in self.targets
        }
        self.entity_ids = cells
        self.country = geo.country
        return self

    def predict(
        self, df: pd.DataFrame | FeatureFrame, sequence_number: int, output_length: int
    ) -> dict:
        """Dict[str, PredictionFrame], one per target, each (N · output_length, n_samples).

        The history ends at c = train_end + sequence_number; forecast month t is drawn with
        h = t − c. Draw order: entity → time → target, one fresh RNG per call (ADR-011).
        """
        if self.knobs is None:
            raise ValueError(f"{MODEL_NAME}: call fit() before predict().")
        if sequence_number < 0:
            raise ValueError(f"sequence_number must be >= 0, got {sequence_number}.")
        test_start, train_end = train_test_boundary(self.partition_dict)
        last = train_end + sequence_number
        ff = to_feature_frame(df, loa=self.loa, targets=self.targets)
        time, unit, values = panel(ff, self.targets)
        months, cells, dense = _dense_panel(time, unit, values, last)
        if months[-1] != last:
            raise ValueError(f"{MODEL_NAME}: sequence_number {sequence_number} needs data up to "
                             f"month {last}, but it ends at {int(months[-1])}.")
        if not np.array_equal(cells, self.entity_ids):
            raise ValueError(f"{MODEL_NAME}: predict data has {cells.size} cells, not the "
                             f"{self.entity_ids.size} cells it was fitted on.")
        geo = build_geometry(cells, self.country)
        shells = build_shell_tables(geo)
        samplers = {t: build_sampler(dense[t], months, last, self.knobs[t], geo, shells)
                    for t in self.targets}

        time_ids = build_time_grid(test_start, sequence_number, output_length)
        require_entities(list(cells), MODEL_NAME)
        time_arr, unit_arr = build_identifier_arrays(cells, time_ids)
        rng = np.random.default_rng(self.seed)
        S, T = self.n_samples, len(time_ids)
        h = np.asarray(time_ids) - last
        y_preds = {t: np.empty((time_arr.size, S), dtype=np.float32) for t in self.targets}
        # Uniforms are generated a block of cells at a time, in C order over
        # (cell, month, target, step, sample): the same stream, in the same entity → time →
        # target order, as one call per (cell, month, target) (ADR-011). BLOCK_CELLS changes
        # memory and speed only, never the output (tested).
        for b0 in range(0, cells.size, BLOCK_CELLS):
            block = np.arange(b0, min(cells.size, b0 + BLOCK_CELLS))
            u = rng.random((block.size, T, len(self.targets), 3, S))
            rows = slice(b0 * T, (b0 + block.size) * T)
            for ti, t in enumerate(self.targets):
                y_preds[t][rows] = samplers[t].draw(block, h, u[:, :, ti]).reshape(-1, S)
        return to_prediction_frames(y_preds, time=time_arr, unit=unit_arr,
                                    level=to_level(df, loa=self.loa))
