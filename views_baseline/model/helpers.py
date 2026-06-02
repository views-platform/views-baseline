from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_prediction_grid(
    time_idx: str,
    entity_idx: str,
    entity_ids,
    time_ids: list[int],
    targets: list[str],
    value_fn: Callable[[Any, str], Any],
) -> pd.DataFrame:
    """
    Build a prediction DataFrame on the (time, entity) grid.

    Args:
        time_idx: Name of the time index level.
        entity_idx: Name of the entity index level.
        entity_ids: Entity IDs to include.
        time_ids: Time IDs to include.
        targets: Target variable names (without 'pred_' prefix).
        value_fn: Called as value_fn(entity_id, target) to get the prediction value.

    Returns:
        DataFrame with MultiIndex [time_idx, entity_idx] and columns pred_{target}.
    """
    records = []
    for cid in entity_ids:
        for tid in time_ids:
            row = {time_idx: tid, entity_idx: cid}
            row.update({f"pred_{target}": value_fn(cid, target) for target in targets})
            records.append(row)

    if not records:
        df = pd.DataFrame(columns=[f"pred_{target}" for target in targets])
        df.index = pd.MultiIndex.from_arrays([[] for _ in range(2)], names=[time_idx, entity_idx])
        return df

    df = pd.DataFrame(records)
    df = df.set_index([time_idx, entity_idx]).sort_index()
    return df[[f"pred_{target}" for target in targets]]


def build_time_grid(
    test_start: int, sequence_number: int, output_length: int
) -> list[int]:
    """Build the list of prediction time IDs starting from test_start + sequence_number."""
    prediction_start = test_start + sequence_number
    return list(range(prediction_start, prediction_start + output_length))


def filter_entities(entity_ids, valid_set, model_name: str) -> list:
    """Filter entity IDs to those present in valid_set, logging a warning on drops."""
    n_before = len(entity_ids)
    filtered = [cid for cid in entity_ids if cid in valid_set]
    if len(filtered) < n_before:
        logger.warning(
            f"{model_name}: {n_before - len(filtered)} entities dropped"
        )
    return filtered


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


def build_prediction_frame(
    entity_ids,
    time_ids: list[int],
    targets: list[str],
    value_fn: Callable[[Any, str], Any],
) -> dict:
    """Build a dict[str, PredictionFrame] on the (entity, time) grid.

    Point-model counterpart of build_prediction_grid(). Returns one
    PredictionFrame per target with y_pred shape (N, 1).

    Iterates entity→time to match distributional model ordering (ADR-011).
    """
    from views_pipeline_core.data.prediction_frame import PredictionFrame

    time_arr, unit_arr = build_identifier_arrays(entity_ids, time_ids)
    n_rows = len(time_arr)

    result = {}
    for target in targets:
        values = np.empty((n_rows, 1), dtype=np.float64)
        idx = 0
        for cid in entity_ids:
            for _ in time_ids:
                values[idx, 0] = value_fn(cid, target)
                idx += 1
        result[target] = PredictionFrame(
            y_pred=values,
            identifiers={"time": time_arr.copy(), "unit": unit_arr.copy()},
        )
    return result
