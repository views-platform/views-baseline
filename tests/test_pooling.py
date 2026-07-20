"""Characterization tests for the shared `window_pool` (ADR-022 / epic #33 S2; PR-2 S6).

Two guarantees:

* ``window_pool(ff, ...)`` reproduces ``ConflictologyModel``'s per-entity pools (the
  precondition for a valid closeness comparison).
* ``window_pool_arrays`` (the pure-numpy core) reproduces the previous pandas
  ``sort_index(...).groupby(...).tail`` semantics **byte-for-byte in float64** — pinning
  entity order, per-entity tail order, and values independent of the FeatureFrame's float32
  precision (the ordering guard for the PR-2 numpy port; see C-32).
"""

import numpy as np
import pandas as pd

from views_baseline.model.frames.input import to_feature_frame
from views_baseline.model.frames.pooling import window_pool, window_pool_arrays
from views_baseline.model.models.distributional import ConflictologyModel

_PARTITION = {"test": (493, 540)}


def _df(vals_fn=lambda t, u: float(t % 5)):
    idx = pd.MultiIndex.from_product(
        [range(440, 500), [1, 2]], names=["month_id", "priogrid_id"]
    )
    t = idx.get_level_values(0)
    u = idx.get_level_values(1)
    return pd.DataFrame({"y1": [vals_fn(a, b) for a, b in zip(t, u)]}, index=idx)


def _pandas_pool_reference(df, targets, window_months, train_end):
    """Replicate the pre-PR-2 pandas window_pool logic as an independent float64 reference."""
    time_name, entity_name = df.index.names
    d = df[df.index.get_level_values(time_name) <= train_end]
    d = d.sort_index(level=[entity_name, time_name])
    last = d.groupby(level=entity_name, group_keys=False).apply(lambda g: g.tail(window_months))
    entity_ids = (
        d.loc[d.index.get_level_values(time_name) == train_end]
        .index.get_level_values(entity_name)
        .unique()
    )
    pools = {}
    for cid in entity_ids:
        h = last.xs(cid, level=entity_name, drop_level=False)
        pools[cid] = {t: np.array(h[t].tolist(), dtype=np.float64) for t in targets}
    return list(entity_ids), pools


def test_window_pool_matches_conflictology_fit():
    df = _df()
    model = ConflictologyModel(
        targets=["y1"], window_months=6, partition_dict=_PARTITION, loa="pgm", n_samples=8
    )
    model.fit(df)

    ff = to_feature_frame(df, loa="pgm", targets=["y1"])
    entity_ids, pools = window_pool(ff, ["y1"], 6, 492)

    assert list(entity_ids) == list(model.entity_ids)
    assert set(pools.keys()) == set(model.hist_per_entity.keys())
    for cid in pools:
        np.testing.assert_array_equal(pools[cid]["y1"], model.hist_per_entity[cid]["y1"])


def test_window_pool_holds_last_window_months_up_to_train_end():
    ff = to_feature_frame(_df(), loa="pgm", targets=["y1"])
    _, pools = window_pool(ff, ["y1"], 4, 492)
    # last 4 rows up to t=492 for entity 1 -> t in {489,490,491,492}, values t%5
    np.testing.assert_array_equal(
        pools[1]["y1"], [float(t % 5) for t in (489, 490, 491, 492)]
    )


def test_window_pool_arrays_matches_pandas_reference_in_float64():
    """The numpy core reproduces the pandas ordering/logic byte-for-byte in float64.

    Uses non-float32-exact values so the equality genuinely exercises float64 precision —
    this pins entity/tail order and values independent of the FeatureFrame float32 path.
    """
    df = _df(vals_fn=lambda t, u: t * 0.1 + u * 0.017)  # non-float32-exact
    ref_ids, ref_pools = _pandas_pool_reference(df, ["y1"], 6, 492)

    time = df.index.get_level_values(0).to_numpy(np.int64)
    unit = df.index.get_level_values(1).to_numpy(np.int64)
    values = {"y1": df["y1"].to_numpy(np.float64)}
    entity_ids, pools = window_pool_arrays(time, unit, values, ["y1"], 6, 492)

    assert list(entity_ids) == ref_ids
    assert set(pools.keys()) == set(ref_pools.keys())
    for cid in pools:
        # byte-for-byte equal in float64 (not allclose) — the ordering/logic guard
        np.testing.assert_array_equal(pools[cid]["y1"], ref_pools[cid]["y1"])
