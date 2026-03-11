import pandas as pd
from typing import List, Callable, Any


def build_prediction_grid(
    time_idx: str,
    entity_idx: str,
    loa_ids,
    time_ids: List[int],
    targets: List[str],
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
            row.update({f"pred_{t}": value_fn(cid, t) for t in targets})
            records.append(row)

    if not records:
        df = pd.DataFrame(columns=[f"pred_{t}" for t in targets])
        df.index = pd.MultiIndex.from_arrays(
            [[] for _ in range(2)], names=[time_idx, entity_idx]
        )
        return df

    df = pd.DataFrame(records)
    df = df.set_index([time_idx, entity_idx]).sort_index()
    return df[[f"pred_{t}" for t in targets]]
