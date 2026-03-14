from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_prediction_grid(
    time_idx: str,
    entity_idx: str,
    loa_ids,
    time_ids: list[int],
    targets: list[str],
    value_fn: Callable[[Any, str], Any],
) -> pd.DataFrame:
    """
    Build a prediction DataFrame on the (time, entity) grid.

    Args:
        time_idx: Name of the time index level.
        entity_idx: Name of the entity index level.
        loa_ids: Entity IDs to include.
        time_ids: Time IDs to include.
        targets: Target variable names (without 'pred_' prefix).
        value_fn: Called as value_fn(entity_id, target) to get the prediction value.

    Returns:
        DataFrame with MultiIndex [time_idx, entity_idx] and columns pred_{target}.
    """
    records = []
    for cid in loa_ids:
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


def filter_entities(loa_ids, valid_set, model_name: str) -> list:
    """Filter entity IDs to those present in valid_set, logging a warning on drops."""
    n_before = len(loa_ids)
    filtered = [cid for cid in loa_ids if cid in valid_set]
    if len(filtered) < n_before:
        logger.warning(
            f"{model_name}: {n_before - len(filtered)} entities dropped"
        )
    return filtered


def build_identifier_arrays(
    entities, time_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Build time and unit identifier arrays for distributional models.

    Iterates entity→time to preserve ordering per ADR-011.
    """
    time_arr = []
    unit_arr = []
    for cid in entities:
        for tid in time_ids:
            time_arr.append(tid)
            unit_arr.append(cid)
    return np.array(time_arr), np.array(unit_arr)
