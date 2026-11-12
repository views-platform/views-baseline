"""PooledConflictology (#151): conflictology whose window is pooled over an area.

Every value encodes where it came from: ``y1 = month * 100 + cell``. So each draw can be read
back to the exact cell and month it was drawn from.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from views_baseline.model.frames.input import to_feature_frame
from views_baseline.model.models.distributional import ConflictologyModel, PooledConflictology

PARTITION = {"test": (500, 535)}           # train_end = 499
TRAIN_END = 499
WINDOW = 4
MONTHS = range(480, 540)
# cells 1-3 in area 10, cells 4-5 in area -3727 (negative, but a real area, like the Azores),
# cell 6 in no area (-1).
AREA = {1: 10, 2: 10, 3: 10, 4: -3727, 5: -3727, 6: -1}
KW = dict(targets=["y1"], window_months=WINDOW, partition_dict=PARTITION, loa="pgm",
          n_samples=4000, seed=3)


def _df(area=AREA, months=MONTHS):
    index = pd.MultiIndex.from_product([list(months), sorted(area)],
                                       names=["month_id", "priogrid_id"])
    df = pd.DataFrame(index=index)
    df["y1"] = [t * 100 + c for t, c in index]
    df["area"] = [float(area[c]) for _, c in index]
    return df


def _sources(pf):
    """(forecast cell, source cell, source month) for every draw."""
    vals = np.asarray(pf.values).astype(np.int64)
    cells = np.repeat(np.asarray(pf.index.unit), vals.shape[1])
    flat = vals.ravel()
    return cells, flat % 100, flat // 100


def _drawn_from(pf):
    """{forecast cell: set of (source cell, source month)}."""
    cells, src, month = _sources(pf)
    out = {}
    for c, s, m in zip(cells, src, month):
        out.setdefault(int(c), set()).add((int(s), int(m)))
    return out


def _window(cells):
    return {(c, m) for c in cells for m in range(TRAIN_END - WINDOW + 1, TRAIN_END + 1)}


def test_each_cell_draws_from_all_of_its_areas_window_and_nothing_else():
    pf = PooledConflictology(**KW, pool_by="area").fit(_df()).predict(
        _df(), sequence_number=0, output_length=2)["y1"]
    drawn = _drawn_from(pf)
    for cell, area in AREA.items():
        if area == -1:
            continue
        same_area = [c for c, a in AREA.items() if a == area]
        assert drawn[cell] == _window(same_area), cell


def test_a_cell_without_an_area_draws_from_its_own_window_and_is_warned_about(caplog):
    with caplog.at_level(logging.WARNING):
        m = PooledConflictology(**KW, pool_by="area").fit(_df())
    assert "1 of 6 cells have no area" in caplog.text
    pf = m.predict(_df(), sequence_number=0, output_length=1)["y1"]
    assert _drawn_from(pf)[6] == _window([6])


def test_map_pools_every_cell_and_needs_no_area_column():
    df = _df().drop(columns="area")
    pf = PooledConflictology(**KW, pool_by="map").fit(df).predict(
        df, sequence_number=0, output_length=1)["y1"]
    for cell, sources in _drawn_from(pf).items():
        assert sources == _window(AREA), cell


def test_window_stays_at_train_end_for_later_origins():
    """As ConflictologyModel: the pool is fixed at fit, whatever the origin."""
    pf = PooledConflictology(**KW, pool_by="area").fit(_df()).predict(
        _df(), sequence_number=12, output_length=3)["y1"]
    _, _, month = _sources(pf)
    assert month.max() == TRAIN_END and month.min() == TRAIN_END - WINDOW + 1


def test_cells_of_one_area_share_one_pool_not_copies():
    m = PooledConflictology(**KW, pool_by="area").fit(_df())
    assert m.hist_per_entity[1] is m.hist_per_entity[2] is m.hist_per_entity[3]
    assert m.hist_per_entity[4] is m.hist_per_entity[5]
    assert m.hist_per_entity[1] is not m.hist_per_entity[4]
    assert m.hist_per_entity[1]["y1"].size == 3 * WINDOW


def test_a_cell_absent_at_train_end_is_neither_forecast_nor_pooled():
    """As ConflictologyModel: only the cells present at train_end are forecast."""
    df = _df().drop(index=[(t, 3) for t in range(TRAIN_END, 540)])
    m = PooledConflictology(**KW, pool_by="area").fit(df)
    assert 3 not in m.hist_per_entity
    pf = m.predict(df, sequence_number=0, output_length=1)["y1"]
    assert _drawn_from(pf)[1] == _window([1, 2])


def test_a_one_cell_area_is_conflictology():
    """With every cell in its own area, the pools are conflictology's and so are the draws."""
    df = _df(area={c: c for c in AREA})
    a = ConflictologyModel(**KW).fit(df).predict(df, sequence_number=0, output_length=3)["y1"]
    p = PooledConflictology(**KW, pool_by="area").fit(df).predict(
        df, sequence_number=0, output_length=3)["y1"]
    np.testing.assert_array_equal(np.asarray(a.values), np.asarray(p.values))


def test_dataframe_and_featureframe_give_the_same_draws():
    df = _df()
    ff = to_feature_frame(df, loa="pgm", targets=["y1", "area"])
    a = PooledConflictology(**KW, pool_by="area").fit(df).predict(df, 0, 2)["y1"]
    b = PooledConflictology(**KW, pool_by="area").fit(ff).predict(ff, 0, 2)["y1"]
    np.testing.assert_array_equal(np.asarray(a.values), np.asarray(b.values))


@pytest.mark.parametrize("bad, match", [
    (np.nan, r"cell 2 has nan"),
    (10.5, r"cell 2 has 10\.5"),
])
def test_an_area_code_that_is_not_an_integer_raises_naming_the_cell(bad, match):
    df = _df()
    df.loc[(490, 2), "area"] = bad
    with pytest.raises(ValueError, match=match):
        PooledConflictology(**KW, pool_by="area").fit(df)


def test_a_cell_that_changes_area_raises():
    df = _df()
    df.loc[(470 + 20, 2), "area"] = 20.0
    with pytest.raises(ValueError, match=r"changes over time for 1 cells \(first: cell 2\)"):
        PooledConflictology(**KW, pool_by="area").fit(df)


def test_area_changes_after_train_end_are_ignored():
    df = _df()
    df.loc[(TRAIN_END + 1, 2), "area"] = 20.0
    PooledConflictology(**KW, pool_by="area").fit(df)


def test_a_missing_area_column_raises():
    with pytest.raises(ValueError, match="'area'"):
        PooledConflictology(**KW, pool_by="area").fit(_df().drop(columns="area"))


@pytest.mark.parametrize("pool_by", ["", None, 3, "y1"])
def test_pool_by_must_be_a_column_that_is_not_a_target(pool_by):
    with pytest.raises(ValueError, match="pool_by"):
        PooledConflictology(**KW, pool_by=pool_by)
