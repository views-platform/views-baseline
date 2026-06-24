from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from views_frames import PredictionFrame, SpatialLevel

logger = logging.getLogger(__name__)


def build_time_grid(
    test_start: int, sequence_number: int, output_length: int
) -> list[int]:
    """Build the list of prediction time IDs starting from test_start + sequence_number."""
    prediction_start = test_start + sequence_number
    return list(range(prediction_start, prediction_start + output_length))


def resolve_level(loa: str, index_names) -> "SpatialLevel":
    """Resolve the declared level-of-analysis to a views-frames ``SpatialLevel``.

    The declared ``loa`` is authoritative (ADR-003); the DataFrame index is a
    cross-check that must agree. ``views_frames.SpatialLevel`` is the single
    source of truth for the ``(time, entity)`` index vocabulary per level, so we
    validate against ``level.index_names`` rather than hardcoding column names.
    This makes the required spatial ``level`` (carried into every output
    ``PredictionFrame``) a declared-and-validated quantity, not an inference from
    the positional entity index (ADR-020; closes risk C-18, mitigates C-05).

    Args:
        loa: Declared level of analysis — the ``SpatialLevel`` value (``"cm"`` or
            ``"pgm"``), as carried in ``config["level"]``.
        index_names: The DataFrame's ``index.names`` — expected ``(time, entity)``.

    Returns:
        The resolved ``SpatialLevel``.

    Raises:
        ValueError: if ``loa`` is not a known level, or the index names do not
            match the declared level's ``(time, entity)`` vocabulary.
    """
    from views_frames import SpatialLevel

    try:
        level = SpatialLevel(loa)
    except ValueError:
        valid = [lvl.value for lvl in SpatialLevel]
        raise ValueError(
            f"Unknown level of analysis loa={loa!r}; expected one of {valid} "
            f"(ADR-020 spatial-level contract)."
        ) from None

    expected = level.index_names
    actual = tuple(index_names)
    if actual != expected:
        raise ValueError(
            f"Index does not match declared level: loa={loa!r} -> {level.name} "
            f"expects index names {expected}, but the DataFrame index is {actual}. "
            f"Declarations are authoritative (ADR-003); the input does not match "
            f"the declared spatial level."
        )
    return level


def filter_entities(entity_ids, valid_set, model_name: str) -> list:
    """Filter entity IDs to those present in valid_set, logging a warning on drops."""
    n_before = len(entity_ids)
    filtered = [cid for cid in entity_ids if cid in valid_set]
    if len(filtered) < n_before:
        logger.warning(
            f"{model_name}: {n_before - len(filtered)} entities dropped"
        )
    return filtered


def require_entities(entities, model_name: str) -> None:
    """Fail loud when there are no entities to predict for.

    A prediction over zero entities cannot satisfy the pipeline's evaluation
    contract (it requires at least one (entity, time) cell per target) and would
    otherwise surface as an opaque error far from the cause — either
    ``ValueError: y_pred must have at least one row`` when constructing a
    PredictionFrame, or a ``StopIteration`` deep inside pipeline-core's
    streaming evaluation. Raising here names the actual cause at the model
    boundary.

    Reached when the data has no rows at the train/test boundary, or when every
    entity present at ``train_end`` was absent from the fitted state and dropped
    by :func:`filter_entities`.
    """
    if len(entities) == 0:
        raise ValueError(
            f"{model_name}: no entities to predict. The data has no rows at the "
            f"train/test boundary (train_end), or all entities present at "
            f"train_end were absent from the fitted state and were dropped. "
            f"A prediction cannot be constructed."
        )


def build_identifier_arrays(
    entities, time_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Build time and unit identifier arrays.

    Iterates entity→time to preserve ordering per ADR-011.
    """
    time_arr = []
    unit_arr = []
    for cid in entities:
        for tid in time_ids:
            time_arr.append(tid)
            unit_arr.append(cid)
    return np.array(time_arr), np.array(unit_arr)


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
            for _ in time_ids:
                values[idx, 0] = value_fn(cid, target)
                idx += 1
        y_pred_by_target[target] = values

    return to_prediction_frames(y_pred_by_target, time=time_arr, unit=unit_arr, level=level)
