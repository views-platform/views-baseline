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

from views_baseline.model.frames.input import panel

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
    semantics on flat numpy arrays: restrict to ``time <= train_end``, sort stably by
    ``(entity, time)``, take the entities present at ``train_end`` in ascending order, and
    keep each such entity's last ``window_months`` rows in time order.

    Args:
        time: ``(N,)`` int time identifiers.
        unit: ``(N,)`` int entity identifiers.
        values: ``{target -> (N,)}`` observed series (aligned with ``time``/``unit``).
        targets: Target names to pool.
        window_months: Most-recent rows per entity to keep.
        train_end: Last training time id; rows with time > this are dropped.

    Returns:
        ``(entity_ids, pools)`` — see :func:`window_pool`.
    """
    if window_months <= 0:
        # Fail loud on a degenerate window. Guards the numpy slicing trap where
        # `block[-0:]` would silently return the *entire* history (the pre-PR-2 pandas
        # path failed here too, but only accidentally via an empty-group KeyError).
        raise ValueError(f"window_months must be >= 1, got {window_months}.")

    time = np.asarray(time)
    unit = np.asarray(unit)

    keep = time <= train_end
    k_time = time[keep]
    k_unit = unit[keep]
    k_vals = {t: np.asarray(values[t])[keep] for t in targets}

    # Sort stably by (entity, time): lexsort's last key is primary -> unit primary, time
    # secondary. Panel rows are unique per (time, entity), so there are no ties to break.
    order = np.lexsort((k_time, k_unit))
    s_time = k_time[order]
    s_unit = k_unit[order]
    s_vals = {t: k_vals[t][order] for t in targets}

    # Entities present exactly at train_end. On the entity-sorted array, order of appearance
    # is ascending, which is what the previous pandas `.unique()` produced; np.unique matches.
    entity_ids = np.unique(s_unit[s_time == train_end])

    pools = {}
    for cid in entity_ids:
        block = np.nonzero(s_unit == cid)[0]  # this entity's rows, already in time order
        if block.size == 0:
            continue
        tail = block[-window_months:]  # last window_months in time order
        pools[cid] = {t: s_vals[t][tail].astype(np.float64) for t in targets}

    return entity_ids, pools
