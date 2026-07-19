"""Characterization tests for the shared `window_pool` (ADR-022 / epic #33 S2).

Asserts the extracted helper reproduces `ConflictologyModel`'s per-entity pools
byte-for-byte — the precondition for a valid closeness comparison.
"""

import numpy as np
import pandas as pd

from views_baseline.model.baseline import ConflictologyModel
from views_baseline.model.pooling import window_pool

_PARTITION = {"test": (493, 540)}


def _df():
    idx = pd.MultiIndex.from_product(
        [range(440, 500), [1, 2]], names=["month_id", "priogrid_id"]
    )
    vals = [float(t % 5) for t in idx.get_level_values(0)]
    return pd.DataFrame({"y1": vals}, index=idx)


def test_window_pool_matches_conflictology_fit():
    df = _df()
    model = ConflictologyModel(
        targets=["y1"], window_months=6, partition_dict=_PARTITION, loa="pgm", n_samples=8
    )
    model.fit(df)

    entity_ids, pools = window_pool(df, "month_id", "priogrid_id", ["y1"], 6, 492)

    assert list(entity_ids) == list(model.entity_ids)
    assert set(pools.keys()) == set(model.hist_per_entity.keys())
    for cid in pools:
        np.testing.assert_array_equal(pools[cid]["y1"], model.hist_per_entity[cid]["y1"])


def test_window_pool_holds_last_window_months_up_to_train_end():
    df = _df()
    _, pools = window_pool(df, "month_id", "priogrid_id", ["y1"], 4, 492)
    # last 4 rows up to t=492 for entity 1 -> t in {489,490,491,492}, values t%5
    np.testing.assert_array_equal(
        pools[1]["y1"], [float(t % 5) for t in (489, 490, 491, 492)]
    )
