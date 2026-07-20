from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from views_baseline.model.grid import (
    build_identifier_arrays,
    build_time_grid,
    filter_entities,
    require_entities,
)

if TYPE_CHECKING:
    from views_frames import PredictionFrame, SpatialLevel


def to_prediction_frames(
    y_pred_by_target: dict[str, np.ndarray],
    time: np.ndarray,
    unit: np.ndarray,
    level: "SpatialLevel",
) -> dict[str, "PredictionFrame"]:
    """Build a ``dict[str, PredictionFrame]`` from precomputed arrays.

    This is the **single** ``views_frames`` construction site for the whole
    package (ADR-020). Both point models (via :func:`build_prediction_frame`)
    and distributional models route their output through here, so the platform
    leaf is imported and constructed in exactly one place. A single
    ``SpatioTemporalIndex`` (carrying the validated spatial ``level``) backs every
    per-target frame.

    Args:
        y_pred_by_target: One array per target, each shape ``(N, S)`` — ``S == 1``
            for point models, ``S == n_samples`` for distributional models.
        time: Row time identifiers, shape ``(N,)``.
        unit: Row unit identifiers, shape ``(N,)``.
        level: The validated :class:`views_frames.SpatialLevel` (see
            :func:`resolve_level`).

    Returns:
        One ``PredictionFrame`` per target, sharing one ``SpatioTemporalIndex``.
    """
    from views_frames import PredictionFrame, SpatioTemporalIndex

    for target, y_pred in y_pred_by_target.items():
        if y_pred.shape[1] == 0:
            raise ValueError(
                f"{target}: y_pred has 0 sample columns; a PredictionFrame must have "
                f"at least one sample column (got shape {y_pred.shape}). This guards a "
                f"degenerate n_samples=0 — views_frames no longer rejects it (ADR-020)."
            )

    index = SpatioTemporalIndex(
        time=time.astype(np.int64),
        unit=unit.astype(np.int64),
        level=level,
    )
    return {target: PredictionFrame(y_pred, index) for target, y_pred in y_pred_by_target.items()}


def build_prediction_frame(
    entity_ids,
    time_ids: list[int],
    targets: list[str],
    value_fn: Callable[[Any, str], Any],
    level: "SpatialLevel",
) -> dict[str, "PredictionFrame"]:
    """Build a dict[str, PredictionFrame] on the (entity, time) grid for point models.

    Returns one PredictionFrame per target with y_pred shape (N, 1). Builds the
    per-target value arrays and delegates construction to the single seam
    :func:`to_prediction_frames` (ADR-020).

    Iterates entity→time to match distributional model ordering (ADR-011).
    """
    time_arr, unit_arr = build_identifier_arrays(entity_ids, time_ids)
    n_rows = len(time_arr)

    y_pred_by_target = {}
    for target in targets:
        values = np.empty((n_rows, 1), dtype=np.float64)
        idx = 0
        for cid in entity_ids:
            # Point baselines are constant across the horizon: evaluate value_fn
            # once per (entity, target) and fill every horizon step. Preserves the
            # entity→time→target fill order (ADR-011); resolves C-23.
            value = value_fn(cid, target)
            for _ in time_ids:
                values[idx, 0] = value
                idx += 1
        y_pred_by_target[target] = values

    return to_prediction_frames(y_pred_by_target, time=time_arr, unit=unit_arr, level=level)


def sample_prediction_grid(
    *,
    entity_ids,
    fitted_state,
    model_name: str,
    targets: list[str],
    n_samples: int,
    level: "SpatialLevel",
    test_start: int,
    sequence_number: int,
    output_length: int,
    seed: int,
    draw_cell: Callable[..., np.ndarray],
) -> dict[str, "PredictionFrame"]:
    """Shared distributional-model predict scaffold (ADR-011/ADR-020).

    Every distributional baseline shares one shell — build the (entity, time) grid, drop
    entities without fitted state (``fitted_state``), then fill an ``(N, n_samples)`` array
    in **entity→time→target** order from a single seeded RNG — differing only in the per-cell
    draw. ``draw_cell(cid, target, rng)`` returns the ``(n_samples,)`` draw for one cell and
    is called once per (entity, time, target), so the RNG advances identically for every
    model (the reproducibility contract, ADR-011). The ``level`` is resolved once at the
    model's input boundary (``to_feature_frame``) and passed in. Output routes through the
    single ``to_prediction_frames`` seam (ADR-020). The distributional analogue of
    :func:`build_prediction_frame` (which fills constants for point models).
    """
    time_ids = build_time_grid(test_start, sequence_number, output_length)
    entities = filter_entities(entity_ids, fitted_state, model_name)
    require_entities(entities, model_name)

    time_arr, unit_arr = build_identifier_arrays(entities, time_ids)
    n_rows = len(time_arr)
    rng = np.random.default_rng(seed)

    y_preds = {t: np.empty((n_rows, n_samples), dtype=np.float64) for t in targets}
    idx = 0
    for cid in entities:
        for _ in time_ids:
            for t in targets:
                y_preds[t][idx] = draw_cell(cid, t, rng)
            idx += 1

    return to_prediction_frames(y_preds, time=time_arr, unit=unit_arr, level=level)
