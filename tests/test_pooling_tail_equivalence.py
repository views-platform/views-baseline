"""tail_pools finds each entity's block by binary search (O(rows)) instead of scanning the
whole panel per entity (O(entities x rows)). The output must be exactly the old one."""

import numpy as np
import pytest

from views_baseline.model.frames.pooling import sort_train_panel, tail_pools


def _old_tail_pools(entity_ids, s_unit, s_vals, targets, window_months):
    """The previous implementation, kept here as the reference."""
    pools = {}
    for cid in entity_ids:
        block = np.nonzero(s_unit == cid)[0]
        tail = block[-window_months:]
        pools[cid] = {t: s_vals[t][tail].astype(np.float64) for t in targets}
    return pools


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("window", [1, 6, 36, 500])
def test_tail_pools_equals_the_per_entity_scan(seed, window):
    rng = np.random.default_rng(seed)
    n_units, n_months = 300, 120
    unit = np.repeat(rng.permutation(np.arange(1000, 1000 + n_units)), n_months)
    time = np.tile(np.arange(100, 100 + n_months), n_units)
    keep = rng.random(unit.size) > 0.2                    # gaps and short histories
    keep[time == 219] = True                              # every unit present at train_end
    unit, time = unit[keep], time[keep]
    order = rng.permutation(unit.size)                    # unsorted input
    unit, time = unit[order], time[order]
    vals = {"a": rng.integers(0, 50, unit.size).astype(float),
            "b": rng.integers(0, 9, unit.size).astype(float)}
    ents, s_unit, s_vals = sort_train_panel(time, unit, vals, ["a", "b"], 219)
    new = tail_pools(ents, s_unit, s_vals, ["a", "b"], window)
    old = _old_tail_pools(ents, s_unit, s_vals, ["a", "b"], window)
    assert list(new) == list(old)
    for cid in old:
        for t in ("a", "b"):
            np.testing.assert_array_equal(new[cid][t], old[cid][t])


def test_tail_pools_rejects_an_unsorted_panel():
    s_unit = np.array([2, 2, 1, 1])
    with pytest.raises(ValueError, match="not sorted"):
        tail_pools(np.array([1, 2]), s_unit, {"a": np.zeros(4)}, ["a"], 3)
