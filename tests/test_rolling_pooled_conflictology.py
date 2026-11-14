"""RollingPooledConflictology (#165, S2 #167): pooled conflictology with a moving window.

Every value encodes where it came from: ``y1 = month * 100 + cell``. So each draw can be read
back to the exact cell and month it was drawn from.
"""

import numpy as np
import pytest
from test_pooled_conflictology import AREA, KW, TRAIN_END, WINDOW, _df, _drawn_from, _sources

from views_baseline.model.frames.input import to_feature_frame
from views_baseline.model.models.distributional import PooledConflictology, RollingConflictology
from views_baseline.model.models.distributional.rolling_pooled_conflictology import (
    RollingPooledConflictology,
)


def _window(cells, k):
    """The windows of ``cells`` ending at train_end + k."""
    anchor = TRAIN_END + k
    return {(c, m) for c in cells for m in range(anchor - WINDOW + 1, anchor + 1)}


@pytest.mark.parametrize("k", [1, 3, 12])
def test_each_cell_draws_from_all_of_its_areas_window_at_the_origin_and_nothing_else(k):
    pf = RollingPooledConflictology(**KW, pool_by="area").fit(_df()).predict(
        _df(), sequence_number=k, output_length=2)["y1"]
    drawn = _drawn_from(pf)
    for cell, area in AREA.items():
        same_area = [cell] if area == -1 else [c for c, a in AREA.items() if a == area]
        assert drawn[cell] == _window(same_area, k), cell     # area -3727 is a real area


@pytest.mark.parametrize("k", [1, 12])
def test_map_pools_every_cells_window_at_the_origin(k):
    df = _df().drop(columns="area")
    pf = RollingPooledConflictology(**KW, pool_by="map").fit(df).predict(
        df, sequence_number=k, output_length=1)["y1"]
    for cell, sources in _drawn_from(pf).items():
        assert sources == _window(AREA, k), cell


def test_origin_zero_is_pooled_conflictology_byte_for_byte():
    a = PooledConflictology(**KW, pool_by="area").fit(_df()).predict(_df(), 0, 3)["y1"]
    b = RollingPooledConflictology(**KW, pool_by="area").fit(_df()).predict(_df(), 0, 3)["y1"]
    np.testing.assert_array_equal(np.asarray(a.values), np.asarray(b.values))


@pytest.mark.parametrize("k", [0, 1, 5, 12])
def test_one_cell_areas_are_rolling_conflictology_at_every_origin(k):
    df = _df(area={c: c for c in AREA})
    a = RollingConflictology(**KW).fit(df).predict(df, sequence_number=k, output_length=3)["y1"]
    b = RollingPooledConflictology(**KW, pool_by="area").fit(df).predict(
        df, sequence_number=k, output_length=3)["y1"]
    np.testing.assert_array_equal(np.asarray(a.values), np.asarray(b.values))


def test_areas_stay_as_fitted_at_train_end():
    """A cell that moves area after train_end keeps its train_end area at every origin."""
    df = _df()
    for t in range(TRAIN_END + 1, 540):
        df.loc[(t, 2), "area"] = -3727.0
    df.loc[(TRAIN_END + 3, 2), "area"] = np.nan
    pf = RollingPooledConflictology(**KW, pool_by="area").fit(df).predict(
        df, sequence_number=5, output_length=1)["y1"]
    assert _drawn_from(pf)[2] == _window([1, 2, 3], 5)


@pytest.mark.parametrize("bad", ["drop", "nan"])
def test_a_gap_after_train_end_inside_the_window_raises_naming_the_cell(bad):
    df = _df()
    if bad == "drop":
        df = df.drop(index=[(TRAIN_END + 2, 4)])
        match = r"RollingPooledConflictology needs every month .*\(4, \[501\]\)"
    else:
        df.loc[(TRAIN_END + 2, 4), "y1"] = np.nan
        match = r"RollingPooledConflictology: NaN in target 'y1' for cell 4 in month\(s\) \[501\]"
    m = RollingPooledConflictology(**KW, pool_by="area").fit(df)
    m.predict(df, sequence_number=1, output_length=1)          # month 501 not yet in the window
    with pytest.raises(ValueError, match=match):
        m.predict(df, sequence_number=2, output_length=1)


def test_negative_sequence_number_raises():
    m = RollingPooledConflictology(**KW, pool_by="area").fit(_df())
    with pytest.raises(ValueError, match="sequence_number must be >= 0"):
        m.predict(_df(), sequence_number=-1, output_length=1)


def test_dataframe_and_featureframe_give_the_same_draws():
    df = _df()
    ff = to_feature_frame(df, loa="pgm", targets=["y1", "area"])
    a = RollingPooledConflictology(**KW, pool_by="area").fit(df).predict(df, 4, 2)["y1"]
    b = RollingPooledConflictology(**KW, pool_by="area").fit(ff).predict(ff, 4, 2)["y1"]
    np.testing.assert_array_equal(np.asarray(a.values), np.asarray(b.values))


def test_the_window_really_moves():
    """Guards against a silent fallback to the train_end window: the newest month drawn is the
    origin's anchor, not train_end."""
    pf = RollingPooledConflictology(**KW, pool_by="area").fit(_df()).predict(
        _df(), sequence_number=7, output_length=1)["y1"]
    _, _, month = _sources(pf)
    assert month.max() == TRAIN_END + 7 and month.min() == TRAIN_END + 7 - WINDOW + 1


def test_the_catalog_requires_pool_by_and_forwards_every_key():
    """ADR-021: no silent default for the area column; every genome key reaches the model."""
    from test_pooled_conflictology import PARTITION

    from views_baseline.model.catalog import BaselineModelCatalog

    config = {"regression_targets": ["y1"], "window_months": WINDOW, "n_samples": 8, "seed": 1}
    with pytest.raises(ValueError, match="pool_by"):
        BaselineModelCatalog(config=config, partition_dict=PARTITION, loa="pgm").get_model(
            "RollingPooledConflictology")
    m = BaselineModelCatalog(config={**config, "pool_by": "area"}, partition_dict=PARTITION,
                             loa="pgm").get_model("RollingPooledConflictology")
    assert isinstance(m, RollingPooledConflictology)
    assert (m.window_months, m.n_samples, m.seed, m.pool_by) == (WINDOW, 8, 1, "area")


def test_inherited_errors_and_warnings_name_this_model(caplog):
    """A failure must name the model that ran, not its parent (two families run side by side)."""
    import logging

    with pytest.raises(ValueError, match=r"^RollingPooledConflictology: pool_by 'y1'"):
        RollingPooledConflictology(**KW, pool_by="y1")
    df = _df()
    df.loc[(490, 2), "area"] = 10.5
    with pytest.raises(ValueError, match=r"^RollingPooledConflictology: 'area' must hold integer"):
        RollingPooledConflictology(**KW, pool_by="area").fit(df)
    with pytest.raises(ValueError, match=r"^RollingPooledConflictology: pool_by must be a column"):
        RollingPooledConflictology(**KW, pool_by="")
    df = _df()
    df.loc[(490, 2), "area"] = 20.0
    with pytest.raises(ValueError, match=r"^RollingPooledConflictology: 'area' changes over time"):
        RollingPooledConflictology(**KW, pool_by="area").fit(df)
    with caplog.at_level(logging.INFO):
        RollingPooledConflictology(**KW, pool_by="area").fit(_df())
    assert "RollingPooledConflictology: 2 of 7 cells have no area" in caplog.text
    assert "RollingPooledConflictology: pooled 5 cells into 2 areas" in caplog.text
    with pytest.raises(ValueError, match=r"^RollingPooledConflictology: sequence_number"):
        RollingPooledConflictology(**KW, pool_by="area").fit(_df()).predict(_df(), -1, 1)


@pytest.mark.parametrize("k", [3, 12])
def test_forecasts_are_labelled_with_the_origins_months(k):
    """Origin k forecasts months test_start + k .. test_start + k + output_length - 1."""
    pf = RollingPooledConflictology(**KW, pool_by="area").fit(_df()).predict(
        _df(), sequence_number=k, output_length=3)["y1"]
    months = sorted({int(t) for t in np.asarray(pf.index.time)})
    assert months == [TRAIN_END + 1 + k, TRAIN_END + 2 + k, TRAIN_END + 3 + k]


def test_a_cell_that_first_appears_after_train_end_is_neither_forecast_nor_pooled():
    """As RollingConflictology: the forecast cells are fixed at fit, at every origin."""
    df = _df(area={**AREA, 9: 10}).drop(index=[(t, 9) for t in range(480, TRAIN_END + 1)])
    m = RollingPooledConflictology(**KW, pool_by="area").fit(df)
    pf = m.predict(df, sequence_number=4, output_length=1)["y1"]
    drawn = _drawn_from(pf)
    assert 9 not in drawn
    assert drawn[1] == _window([1, 2, 3], 4)


def test_the_area_pool_is_its_cells_windows_in_ascending_cell_order():
    """The exact array, not only its set of values: the order fixes which value a seed draws."""
    m = RollingPooledConflictology(**KW, pool_by="area").fit(_df())
    for k in [0, 2]:
        pools = m._pools_for_origin(_df(), k)
        anchor = TRAIN_END + k
        expected = [t * 100 + c for c in (1, 2, 3) for t in range(anchor - WINDOW + 1, anchor + 1)]
        np.testing.assert_array_equal(pools[2]["y1"], expected)


def test_a_nan_after_train_end_names_the_first_cell_its_months_and_the_cell_count():
    df = _df()
    df.loc[(TRAIN_END + 1, 2), "y1"] = np.nan          # cell 2: two NaN months
    df.loc[(TRAIN_END + 2, 2), "y1"] = np.nan
    df.loc[(TRAIN_END + 3, 5), "y1"] = np.nan          # cell 5: another month
    with pytest.raises(ValueError,
                       match=r"for cell 2 in month\(s\) \[500, 501\],.* 2 cell\(s\) affected"):
        RollingPooledConflictology(**KW, pool_by="area").fit(df).predict(df, 3, 1)


def test_a_gap_earlier_in_the_window_raises():
    """The whole stretch after train_end inside the window is checked, not only the anchor."""
    df = _df().drop(index=[(TRAIN_END + 1, 4)])
    m = RollingPooledConflictology(**KW, pool_by="area").fit(df)
    with pytest.raises(ValueError, match=r"needs every month 500\.\.502 .*\(4, \[500\]\)"):
        m.predict(df, sequence_number=3, output_length=1)


def test_the_shared_window_read_finds_a_gap_even_with_unsorted_entity_ids():
    """`rolling_window_pools` maps rows to cells by sorted lookup; ids given in another order
    must not move one cell's months onto another and hide a gap."""
    from views_baseline.model.frames.pooling import rolling_window_pools

    df = _df().drop(index=[(TRAIN_END + 1, 2)])
    with pytest.raises(ValueError, match=r"X needs every month .*\(2, \[500\]\)"):
        rolling_window_pools(df, loa="pgm", targets=["y1"], entity_ids=[7, 2, 1],
                             window_months=WINDOW, train_end=TRAIN_END, sequence_number=1,
                             model_name="X")
