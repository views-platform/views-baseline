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

    Fails loud on ``window_months <= 0`` (else the numpy ``block[-0:]`` slice returns the
    whole history) and on a NaN in the pooled tail (C-33) — the pre-PR pandas path silently
    skipna'd (``last()`` / ``mean(skipna=True)``); the numpy port surfaces it instead of
    forward-filling / poisoning the forecast. This guards only the **tail** each pooling model
    uses; a model that consumes non-tail rows (``MixtureBaseline``'s global pool) applies its
    own NaN guard on the full panel.
    """
    if window_months <= 0:
        raise ValueError(f"window_months must be >= 1, got {window_months}.")

    pools = {}
    for cid in entity_ids:
        # entity_ids come from rows present at train_end, so every block is non-empty.
        block = np.nonzero(s_unit == cid)[0]  # this entity's rows, already in time order
        tail = block[-window_months:]  # last window_months in time order
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
