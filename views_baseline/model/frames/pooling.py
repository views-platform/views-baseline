"""Shared per-entity window pooling for climatology-style baselines (numpy-on-FeatureFrame).

`window_pool` is the single source of truth for "the last `window_months` observed values
per entity, before the test period." `ConflictologyModel` and the parametric climatology
models (ADR-022) all consume it, so their per-entity pools are **byte-identical** — the
precondition for a valid closeness comparison.

Since PR-2 (ADR-019) the windowing runs on a `views_frames.FeatureFrame`'s numpy arrays —
no pandas. `window_pool(ff, ...)` extracts the ``(time, unit, values)`` panel via
:func:`views_baseline.model.frames.input.panel` and delegates to the pure-numpy
:func:`window_pool_arrays`. The array core is dtype-agnostic; the FeatureFrame is float32
by design, so pooled magnitudes are float32-rounded (accepted trade-off — see risk C-32).
The **entity order, per-entity tail order, and RNG-advance order are preserved** exactly
(the byte-identity contract, ADR-011); only value precision follows the FeatureFrame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from views_baseline.model.frames.input import panel, to_feature_frame

if TYPE_CHECKING:
    from views_frames import FeatureFrame


def window_pool(
    ff: "FeatureFrame",
    targets: list[str],
    window_months: int,
    train_end: int,
) -> tuple:
    """Extract the last `window_months` values per entity, per target, up to `train_end`.

    Frame-facing entry: reads the ``(time, unit, values)`` numpy panel from ``ff`` and
    delegates to :func:`window_pool_arrays`. No pandas.

    Args:
        ff: Input ``FeatureFrame`` carrying the target features (observed data).
        targets: Target feature names.
        window_months: Number of most-recent rows per entity to keep.
        train_end: Last training time id (``test_start - 1``); rows with time > this
            are excluded.

    Returns:
        ``(entity_ids, pools)`` where ``entity_ids`` are the entities present at
        ``train_end`` (ascending) and ``pools[cid][target]`` is a ``float64`` array of that
        entity's last ``window_months`` values (entities with empty history skipped).
    """
    time, unit, values = panel(ff, targets)
    return window_pool_arrays(time, unit, values, targets, window_months, train_end)


def window_pool_arrays(
    time: np.ndarray,
    unit: np.ndarray,
    values: dict,
    targets: list[str],
    window_months: int,
    train_end: int,
) -> tuple:
    """Pure-numpy windowing core (the byte-identity-critical logic; dtype-agnostic).

    Reproduces the previous pandas ``sort_index(level=[entity, time]).groupby(entity).tail``
    semantics: restrict to ``time <= train_end``, sort stably by ``(entity, time)``, take the
    entities present at ``train_end`` in ascending order, and keep each such entity's last
    ``window_months`` rows in time order. Composed from :func:`sort_train_panel` +
    :func:`tail_pools` so the sort can be shared (see ``MixtureBaseline``).

    Returns:
        ``(entity_ids, pools)`` — see :func:`window_pool`.
    """
    entity_ids, s_unit, s_vals = sort_train_panel(time, unit, values, targets, train_end)
    return entity_ids, tail_pools(entity_ids, s_unit, s_vals, targets, window_months)


def sort_train_panel(
    time: np.ndarray,
    unit: np.ndarray,
    values: dict,
    targets: list[str],
    train_end: int,
) -> tuple:
    """Restrict to ``time <= train_end`` and stably sort by ``(entity, time)``.

    The single ``(entity, time)`` sort shared by :func:`window_pool_arrays` and
    ``MixtureBaseline`` (which derives both its local *and* global pools from the one sorted
    panel, instead of re-sorting). Returns ``(entity_ids, s_unit, s_vals)``: the entities
    present at ``train_end`` (ascending — matching the pre-PR pandas ``.unique()`` on an
    entity-sorted frame) and the sorted unit + per-target value arrays.
    """
    time = np.asarray(time)
    unit = np.asarray(unit)

    keep = time <= train_end
    k_time = time[keep]
    k_unit = unit[keep]
    k_vals = {t: np.asarray(values[t])[keep] for t in targets}

    # lexsort's last key is primary -> unit primary, time secondary. Panel rows are unique
    # per (time, entity), so there are no ties to break.
    order = np.lexsort((k_time, k_unit))
    s_time = k_time[order]
    s_unit = k_unit[order]
    s_vals = {t: k_vals[t][order] for t in targets}

    entity_ids = np.unique(s_unit[s_time == train_end])
    return entity_ids, s_unit, s_vals


def tail_pools(
    entity_ids: np.ndarray,
    s_unit: np.ndarray,
    s_vals: dict,
    targets: list[str],
    window_months: int,
) -> dict:
    """Per-entity last ``window_months`` values from a ``(entity, time)``-sorted panel.

    ``s_unit`` **must** be sorted ascending (entity primary, as :func:`sort_train_panel`
    returns it): each entity's rows are located by binary search, so an unsorted panel would
    pool the wrong rows. Entity order is checked (an unsorted ``s_unit`` raises); time order
    within an entity is the caller's responsibility (:func:`sort_train_panel` provides it).

    Fails loud on ``window_months <= 0`` (else every pool would be empty) and on a NaN in
    the pooled tail (C-33) — the pre-PR pandas path silently skipna'd (``last()`` /
    ``mean(skipna=True)``); the numpy port surfaces it instead of forward-filling / poisoning
    the forecast. This guards only the **tail** each pooling model
    uses; a model that consumes non-tail rows (``MixtureBaseline``'s global pool) applies its
    own NaN guard on the full panel.
    """
    if window_months <= 0:
        raise ValueError(f"window_months must be >= 1, got {window_months}.")
    if s_unit.size > 1 and np.any(s_unit[1:] < s_unit[:-1]):
        raise ValueError(
            "tail_pools: s_unit is not sorted ascending by entity; pass the panel from "
            "sort_train_panel."
        )

    # s_unit is sorted (entity primary, time secondary), so each entity's rows are one contiguous
    # block: find its bounds by binary search instead of scanning the whole panel per entity
    # (that scan made pooling O(entities x rows): ~14 min at pgm `land`). Same rows, same order.
    starts = np.searchsorted(s_unit, entity_ids, side="left")
    ends = np.searchsorted(s_unit, entity_ids, side="right")
    pools = {}
    for cid, start, end in zip(entity_ids, starts, ends):
        # entity_ids come from rows present at train_end, so every block is non-empty.
        tail = np.arange(max(start, end - window_months), end)  # last window_months, time order
        pool = {}
        for t in targets:
            vals = s_vals[t][tail].astype(np.float64)
            if np.isnan(vals).any():
                raise ValueError(
                    f"window_pool: NaN target value in the training window for entity {cid}, "
                    f"target {t!r}. Baseline models require complete target data in the "
                    f"training window; NaN is not silently forward-filled."
                )
            pool[t] = vals
        pools[cid] = pool
    return pools


def rolling_window_pools(
    df, *, loa: str, targets: list[str], entity_ids, window_months: int, train_end: int,
    sequence_number: int, model_name: str,
) -> dict:
    """The windows of the forecast cells ending at ``train_end + k`` (``k = sequence_number``).

    The information rule of the moving-window models (`RollingConflictology`,
    `RollingPooledConflictology`; ADR-024): origin ``k`` draws from the last ``window_months``
    rows up to ``train_end + k``. Only the forecast cells (``entity_ids``, fixed at fit) are read,
    so a cell that is never forecast cannot stop the run. Fails loud on a missing or NaN month
    after ``train_end`` inside the window (`_check_months_after_train_end`).

    Returns ``pools[cid][target]``, as `window_pool_arrays`.
    """
    anchor = train_end + sequence_number
    ff = to_feature_frame(df, loa=loa, targets=targets)
    time, unit, values = panel(ff, targets)
    time, unit = np.asarray(time), np.asarray(unit)
    fitted = np.isin(unit, np.asarray(entity_ids))
    time, unit = time[fitted], unit[fitted]
    values = {t: values[t][fitted] for t in targets}
    _check_months_after_train_end(
        time, unit, values, targets=targets, entity_ids=entity_ids, window_months=window_months,
        train_end=train_end, anchor=anchor, sequence_number=sequence_number,
        model_name=model_name,
    )
    _, pools = window_pool_arrays(time, unit, values, targets, window_months, anchor)
    return pools


def _check_months_after_train_end(
    time, unit, values, *, targets, entity_ids, window_months: int, train_end: int, anchor: int,
    sequence_number: int, model_name: str,
) -> None:
    """Every forecast cell must have every month after train_end that lies inside the window,
    none NaN: months ``max(train_end + 1, anchor - window_months + 1) .. anchor``. Those are the
    months a fixed-window model never sees; a gap there would silently pull older months back
    into the window. ``unit`` holds rows of ``entity_ids`` only (`rolling_window_pools` filters).
    """
    start = max(train_end + 1, anchor - window_months + 1)
    cells = np.sort(np.asarray(entity_ids))          # searchsorted needs ascending ids
    after = (time >= start) & (time <= anchor)
    row = np.searchsorted(cells, unit[after])
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
            f"{model_name} needs every month {start}..{anchor} (after "
            f"train_end {train_end} and inside the window for sequence_number "
            f"{sequence_number}) for every forecast cell, but {int(incomplete.sum())} of "
            f"{cells.size} cells are missing months (cell, first missing months: "
            f"{examples}). Refusing to fall back to an older window."
        )
    for t in targets:
        nan = np.isnan(values[t][after])
        if nan.any():
            cids = unit[after][nan]
            months = sorted({int(m) for m in time[after][nan][cids == cids[0]]})[:3]
            raise ValueError(
                f"{model_name}: NaN in target {t!r} for cell {int(cids[0])} in "
                f"month(s) {months}, after train_end {train_end} and inside the window for "
                f"sequence_number {sequence_number}. {np.unique(cids).size} cell(s) affected."
            )
