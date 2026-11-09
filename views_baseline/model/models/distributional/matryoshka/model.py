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

import logging
import numbers
from typing import TYPE_CHECKING

import numpy as np

from views_baseline.model.defaults import DEFAULT_SEED
from views_baseline.model.frames.input import panel, to_feature_frame, to_level
from views_baseline.model.frames.output import to_prediction_frames
from views_baseline.model.grid import (
    build_identifier_arrays,
    build_time_grid,
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

logger = logging.getLogger(__name__)

MODEL_NAME = "MatryoshkaConflictology"
BLOCK_UNIFORMS = 2 ** 24   # uniforms per predict block (128 MiB of float64); memory/speed only


def _require_int(name: str, value, minimum: int) -> None:
    """An integer (not a bool, not a float) ≥ ``minimum``; ADR-008: fail at construction."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{MODEL_NAME}: {name} must be an integer ≥ {minimum}, got {value!r}.")


def _dense_panel(time: np.ndarray, unit: np.ndarray, values: dict, last_month: int,
                 non_negative: list[str]):
    """Months ≤ ``last_month`` as a dense (M, N) panel per column. Fails loud unless every
    cell has every month from the first month on, with no NaN (ADR-024: the shells and their
    sizes assume a complete panel), and unless the ``non_negative`` columns (the targets:
    fatality counts) have no negative value."""
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
    count = np.bincount(m_idx * cells.size + c_idx, minlength=months.size * cells.size)
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
        if name in non_negative:
            for bad, what in ((~np.isfinite(x), "an infinite value"), (x < 0, "a negative value"),
                              (x != np.round(x), "a non-integer value")):
                if bad.any():
                    s, j = np.argwhere(bad)[0]
                    raise ValueError(f"{MODEL_NAME}: {what} in target {name!r} (cell "
                                     f"{int(cells[j])}, month {int(months[s])}); the targets "
                                     f"must be fatality counts.")
        dense[name] = x
    return months, cells, dense


def _log_fit(target: str, k: dict) -> None:
    """INFO the knobs; WARN when the search did not converge or probabilities hit the floor
    (ADR-008: the diagnostics must not stay on the model only)."""
    logger.info("%s %s: weight %s, memory %s months, length scale %.3g cells; score %.6g "
                "(start %.6g), %d steps, %.0f s.", MODEL_NAME, target,
                np.round(k["weight"], 4).tolist(), np.round(k["memory"], 1).tolist(),
                k["length_scale"], k["score"], k["start_score"], k["steps"], k["seconds"])
    searches = [("the knob search", k)] + [
        (f"the stability refit without cutoff {r['left_out']}", r)
        for r in k.get("leave_one_cutoff_out", [])]
    for what, r in searches:
        if not r["converged"]:
            logger.warning("%s %s: %s did not converge (%s); its knobs are the last step's.",
                           MODEL_NAME, target, what, r["message"])
        if r["floored"]:
            logger.warning("%s %s: in %s, %d backtest probabilities fell below the floor and "
                           "score as the floor; the knobs give some observed outcomes (almost) "
                           "no chance.", MODEL_NAME, target, what, r["floored"])


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
        if not self.targets:
            raise ValueError(f"{MODEL_NAME}: targets must not be empty.")
        _require_int("n_samples", n_samples, 1)
        _require_int("backtest_cutoffs", backtest_cutoffs, 1)
        _require_int("backtest_horizon", backtest_horizon, 1)
        _require_int("seed", seed, 0)
        if not isinstance(stability_refits, (bool, np.bool_)):
            raise ValueError(f"{MODEL_NAME}: stability_refits must be True or False, got "
                             f"{stability_refits!r}.")
        self.stability_refits = bool(stability_refits)
        if not isinstance(country_column, str):
            raise ValueError(f"{MODEL_NAME}: country_column must be a column name, got "
                             f"{country_column!r}.")
        if country_column in self.targets:
            raise ValueError(f"country_column {country_column!r} must not be a target.")
        self.entity_ids = None        # (N,) priogrid ids, ascending
        self.country = None           # (N,) GAUL country code per cell
        self.history_start = None     # first month of the fit history; predict must match it
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
        months, cells, dense = _dense_panel(time, unit, values, train_end, self.targets)
        if months[-1] != train_end:
            raise ValueError(f"{MODEL_NAME}: the data ends at month {int(months[-1])}, before "
                             f"train_end {train_end}.")
        country = dense.pop(self.country_column)
        if np.any(country != country[:1]):
            raise ValueError(f"{MODEL_NAME}: {self.country_column!r} changes over time for "
                             f"some cells; a cell must belong to one country.")
        if not np.all(np.isfinite(country[0])) or np.any(country[0] != np.round(country[0])):
            raise ValueError(f"{MODEL_NAME}: {self.country_column!r} must hold integer "
                             f"country codes.")
        geo = build_geometry(cells, country[0].astype(np.int64))
        cutoffs = backtest_cutoffs(train_end, self.backtest_cutoffs, self.backtest_horizon)
        self.knobs = {
            t: fit_knobs(dense[t].astype(np.float64), months, geo, cutoffs,
                         self.backtest_horizon, self.stability_refits)
            for t in self.targets
        }
        for t, k in self.knobs.items():
            _log_fit(t, k)
        self.entity_ids = cells
        self.country = geo.country
        self.history_start = int(months[0])
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
        if output_length < 1:
            raise ValueError(f"output_length must be >= 1, got {output_length}.")
        test_start, train_end = train_test_boundary(self.partition_dict)
        last = train_end + sequence_number
        ff = to_feature_frame(df, loa=self.loa, targets=self.targets)
        time, unit, values = panel(ff, self.targets)
        months, cells, dense = _dense_panel(time, unit, values, last, self.targets)
        if months[0] != self.history_start:
            raise ValueError(
                f"{MODEL_NAME}: the predict data starts at month {int(months[0])}, but the knobs "
                f"were fitted on history starting at month {self.history_start}. The history is "
                f"all the data passed, so a different start would draw from a different "
                f"distribution than the one fitted.")
        if months[-1] != last:
            raise ValueError(f"{MODEL_NAME}: sequence_number {sequence_number} needs data up to "
                             f"month {last}, but it ends at {int(months[-1])}.")
        if not np.array_equal(cells, self.entity_ids):
            extra = np.setdiff1d(cells, self.entity_ids)[:5].tolist()
            missing = np.setdiff1d(self.entity_ids, cells)[:5].tolist()
            raise ValueError(
                f"{MODEL_NAME}: the predict data must have the {self.entity_ids.size} cells it "
                f"was fitted on (it has {cells.size}). Cells not fitted on: {extra or 'none'}; "
                f"fitted cells missing: {missing or 'none'} (at most 5 of each shown). Every cell "
                f"needs a country and shells from the fit, so the cells cannot change after "
                f"train_end.")
        level = to_level(df, loa=self.loa)
        del ff, time, unit, values         # only the dense panel is needed from here (memory)
        geo = build_geometry(cells, self.country)
        shells = build_shell_tables(geo)
        samplers = {t: build_sampler(dense[t], months, last, self.knobs[t], geo, shells)
                    for t in self.targets}

        time_ids = build_time_grid(test_start, sequence_number, output_length)
        time_arr, unit_arr = build_identifier_arrays(cells, time_ids)
        rng = np.random.default_rng(self.seed)
        S, T = self.n_samples, len(time_ids)
        h = np.asarray(time_ids) - last
        y_preds = {t: np.empty((time_arr.size, S), dtype=np.float32) for t in self.targets}
        # Uniforms are generated a block of cells at a time, in C order over
        # (cell, month, target, step, sample): the same stream, in the same entity → time →
        # target order, as one call per (cell, month, target) (ADR-011). The block size changes
        # memory and speed only, never the output (tested).
        block_cells = max(1, BLOCK_UNIFORMS // (T * len(self.targets) * 3 * S))
        for b0 in range(0, cells.size, block_cells):
            block = np.arange(b0, min(cells.size, b0 + block_cells))
            u = rng.random((block.size, T, len(self.targets), 3, S))
            rows = slice(b0 * T, (b0 + block.size) * T)
            for ti, t in enumerate(self.targets):
                y_preds[t][rows] = samplers[t].draw(block, h, u[:, :, ti]).reshape(-1, S)
        return to_prediction_frames(y_preds, time=time_arr, unit=unit_arr, level=level)
