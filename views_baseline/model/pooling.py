"""Shared per-entity window pooling for climatology-style baselines.

`window_pool` is the single source of truth for "the last `window_months` observed
values per entity, before the test period." `ConflictologyModel` and the parametric
climatology models (ADR-022) all consume it, so their per-entity pools are
**byte-identical** — the precondition for a valid closeness comparison. Extracted
verbatim from `ConflictologyModel.fit` (ADR-022 / epic #33 S2); behaviour unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def window_pool(
    df: pd.DataFrame,
    time_idx: str,
    entity_idx: str,
    targets: list[str],
    window_months: int,
    train_end: int,
) -> tuple:
    """Extract the last `window_months` values per entity, per target, up to `train_end`.

    Pandas is confined to this extraction seam; the returned pools are plain numpy
    arrays so all downstream fitting/sampling is array-native.

    Args:
        df: Panel indexed by ``(time_idx, entity_idx)`` with the target columns.
        time_idx: Name of the time index level.
        entity_idx: Name of the entity index level.
        targets: Target column names.
        window_months: Number of most-recent rows per entity to keep.
        train_end: Last training time id (``test_start - 1``); rows with time > this
            are excluded.

    Returns:
        ``(entity_ids, pools)`` where ``entity_ids`` are the entities present at
        ``train_end`` and ``pools[cid][target]`` is a ``float64`` array of that
        entity's last ``window_months`` values (entities with empty history skipped).
    """
    df = df[df.index.get_level_values(time_idx) <= train_end]
    df = df.sort_index(level=[entity_idx, time_idx])

    last_n_months = df.groupby(level=entity_idx, group_keys=False).apply(
        lambda g: g.tail(window_months)
    )

    entity_ids = (
        df.loc[df.index.get_level_values(time_idx) == train_end]
        .index.get_level_values(entity_idx)
        .unique()
    )

    pools = {}
    for cid in entity_ids:
        history = last_n_months.xs(cid, level=entity_idx, drop_level=False)
        if history.empty:
            continue
        pools[cid] = {
            t: np.array(history[t].tolist(), dtype=np.float64) for t in targets
        }

    return entity_ids, pools
