from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def train_test_boundary(partition_dict: dict) -> tuple[int, int]:
    """Return ``(test_start, train_end)`` from the partition dict — the single home of the
    train/test boundary convention (``train_end = test_start - 1``), so the ~ten fit/predict
    sites don't each re-encode it (C-37)."""
    test_start = partition_dict["test"][0]
    return test_start, test_start - 1


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


def entities_at(unit: np.ndarray, time: np.ndarray, at_time: int) -> np.ndarray:
    """Entities present at a given time id, in **first-appearance** order.

    Reproduces the pre-PR-2 point-model predict extraction
    ``df.loc[time == at].index.get_level_values(entity).unique()`` — appearance order on the
    input row order (the predict panel is not sorted). Distinct from
    :func:`window_pool_arrays`, which sorts by entity; point-model predict must preserve the
    order of the incoming rows.
    """
    unit = np.asarray(unit)
    time = np.asarray(time)
    sel = unit[time == at_time]
    _, first = np.unique(sel, return_index=True)
    return sel[np.sort(first)]


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
