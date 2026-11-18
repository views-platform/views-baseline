"""NeighbourhoodConflictology and RollingNeighbourhoodConflictology (#178): conflictology whose
window is pooled over a disc of cells around each cell.

The cells are real priogrid ids (the disc comes from them): a block at the date line, a block at
the south-pole row, and two cells far from everything. Every value encodes where it came from:
``y1 = month * 100 + i``, with ``i`` the cell's position in ``CELLS``.
"""

import numpy as np
import pandas as pd
import pytest

from views_baseline.model.frames.input import to_feature_frame
from views_baseline.model.models.distributional import ConflictologyModel, RollingConflictology
from views_baseline.model.models.distributional.matryoshka.geometry import (
    N_COLS,
    NEAR_RADIUS,
    build_geometry,
)
from views_baseline.model.models.distributional.neighbourhood_conflictology import (
    NeighbourhoodConflictology,
)
from views_baseline.model.models.distributional.rolling_neighbourhood_conflictology import (
    RollingNeighbourhoodConflictology,
)

PARTITION = {"test": (500, 535)}           # train_end = 499
TRAIN_END = 499
WINDOW = 4
MONTHS = range(480, 540)


def _id(row, col):
    return row * N_COLS + col + 1


# rows 100-101 around the date line (cols 718, 719, 0, 1); row 0 (the pole row) cols 300-302;
# two lone cells far from everything; and a cell 3 columns from the date-line block.
CELLS = sorted([_id(r, c) for r in (100, 101) for c in (718, 719, 0, 1)]
               + [_id(0, c) for c in (300, 301, 302)]
               + [_id(200, 400), _id(250, 100), _id(100, 4)])
INDEX = {cid: i for i, cid in enumerate(CELLS)}
KW = dict(targets=["y1"], window_months=WINDOW, partition_dict=PARTITION, loa="pgm",
          n_samples=4000, seed=3)


def _df(cells=CELLS, months=MONTHS):
    index = pd.MultiIndex.from_product([list(months), sorted(cells)],
                                       names=["month_id", "priogrid_id"])
    df = pd.DataFrame(index=index)
    df["y1"] = [float(t * 100 + INDEX[c]) for t, c in index]
    return df


def _drawn_from(pf):
    """{forecast cell: set of (source cell, source month)}."""
    vals = np.asarray(pf.values).astype(np.int64)
    cells = np.repeat(np.asarray(pf.index.unit), vals.shape[1])
    out = {}
    for c, v in zip(cells, vals.ravel()):
        out.setdefault(int(c), set()).add((CELLS[v % 100], int(v // 100)))
    return out


def _distance(a, b):
    ra, ca = divmod(a - 1, N_COLS)
    rb, cb = divmod(b - 1, N_COLS)
    dc = (cb - ca + N_COLS // 2) % N_COLS - N_COLS // 2
    return np.hypot(rb - ra, dc)


def _window(cells, end=TRAIN_END):
    return {(c, m) for c in cells for m in range(end - WINDOW + 1, end + 1)}


@pytest.mark.parametrize("radius", [1, 2, 5])
def test_each_cell_draws_from_all_of_its_discs_window_and_nothing_else(radius):
    pf = NeighbourhoodConflictology(**KW, radius=radius).fit(_df()).predict(
        _df(), sequence_number=0, output_length=1)["y1"]
    drawn = _drawn_from(pf)
    for cell in CELLS:
        disc = [c for c in CELLS if _distance(cell, c) <= radius]
        assert drawn[cell] == _window(disc), (cell, radius)


def test_the_draws_are_rng_choice_over_the_discs_windows_in_ascending_cell_order():
    """The exact draws: the reference builds every disc's pool by concatenation, as
    `PooledConflictology` builds an area's, and draws with the same random stream. One cell has
    only 2 months, so the windows have different lengths."""
    df = _df()
    young = _id(100, 0)
    df = df[~((df.index.get_level_values("priogrid_id") == young)
              & (df.index.get_level_values("month_id") < TRAIN_END - 1))]
    m = NeighbourhoodConflictology(**{**KW, "n_samples": 7}, radius=2).fit(df)
    assert len(m.hist_per_entity[young]["y1"]) == 2
    got = np.asarray(m.predict(df, sequence_number=0, output_length=3)["y1"].values)

    own = m.hist_per_entity
    rng = np.random.default_rng(KW["seed"])
    want = []
    for cell in m.entity_ids:
        disc = [c for c in m.entity_ids if _distance(cell, c) <= 2]
        pool = np.concatenate([own[c]["y1"] for c in disc])
        for _ in range(3):
            want.append(rng.choice(pool, size=7, replace=True))
    assert np.array_equal(got, np.asarray(want, dtype=np.float32))


def test_radius_zero_is_conflictology_byte_for_byte():
    df = _df()
    df["y1"] = np.random.default_rng(0).poisson(2.0, len(df)).astype(float)
    got = NeighbourhoodConflictology(**KW, radius=0).fit(df).predict(
        df, sequence_number=3, output_length=5)["y1"]
    want = ConflictologyModel(**KW).fit(df).predict(df, sequence_number=3, output_length=5)["y1"]
    assert np.array_equal(np.asarray(got.values), np.asarray(want.values))
    assert np.array_equal(np.asarray(got.index.unit), np.asarray(want.index.unit))
    assert np.array_equal(np.asarray(got.index.time), np.asarray(want.index.time))


def test_the_radius_5_disc_is_matryoshkas_neighbourhood_plus_the_cell():
    """The control must pool over the same cells as Matryoshka's `near` doll (#178): checked cell
    for cell against `build_geometry`, including date-line and pole-row cells."""
    m = NeighbourhoodConflictology(**KW, radius=NEAR_RADIUS).fit(_df())
    geo = build_geometry(np.asarray(CELLS), np.ones(len(CELLS), dtype=int))
    for i in range(len(CELLS)):
        want = sorted([i] + [int(j) for j in geo.neighbours[i] if j >= 0])
        got = [int(j) for j in m.disc[i] if j >= 0]
        assert got == want, CELLS[i]
    date_line = INDEX[_id(100, 719)]
    assert INDEX[_id(100, 0)] in m.disc[date_line]
    assert INDEX[_id(100, 4)] in m.disc[date_line]          # 5 columns away across the date line


def test_a_cell_with_no_neighbours_draws_from_its_own_window():
    lone = _id(200, 400)
    drawn = _drawn_from(NeighbourhoodConflictology(**KW, radius=NEAR_RADIUS).fit(_df()).predict(
        _df(), sequence_number=0, output_length=1)["y1"])
    assert drawn[lone] == _window([lone])


@pytest.mark.parametrize("bad", [dict(loa="cm"), dict(radius=-1), dict(radius=1.5),
                                 dict(radius=True), dict(radius="5")])
def test_bad_settings_raise_naming_the_model(bad):
    kw = {**KW, "radius": 2, **bad}
    with pytest.raises(ValueError, match="NeighbourhoodConflictology"):
        NeighbourhoodConflictology(**kw)


def test_dataframe_and_featureframe_give_the_same_draws():
    df = _df()
    ff = to_feature_frame(df, loa="pgm", targets=["y1"])
    a = NeighbourhoodConflictology(**KW, radius=2).fit(df).predict(df, 0, 2)["y1"]
    b = NeighbourhoodConflictology(**KW, radius=2).fit(ff).predict(ff, 0, 2)["y1"]
    assert np.array_equal(np.asarray(a.values), np.asarray(b.values))


def test_a_short_history_cell_is_pooled_with_only_the_months_it_has():
    """A cell present for 2 months only contributes 2 months to its neighbours' pools."""
    df = _df()
    young = _id(100, 0)
    df = df[~((df.index.get_level_values("priogrid_id") == young)
              & (df.index.get_level_values("month_id") < TRAIN_END - 1))]
    drawn = _drawn_from(NeighbourhoodConflictology(**KW, radius=1).fit(df).predict(
        df, sequence_number=0, output_length=1)["y1"])
    neighbour = _id(100, 1)
    assert {(young, TRAIN_END - 1), (young, TRAIN_END)} <= drawn[neighbour]
    assert not any(c == young and m < TRAIN_END - 1 for c, m in drawn[neighbour])


# --- RollingNeighbourhoodConflictology (#181) -------------------------------------------------

@pytest.mark.parametrize("k", [1, 3, 12])
def test_rolling_each_cell_draws_from_its_discs_window_ending_at_the_origin(k):
    pf = RollingNeighbourhoodConflictology(**KW, radius=2).fit(_df()).predict(
        _df(), sequence_number=k, output_length=1)["y1"]
    drawn = _drawn_from(pf)
    for cell in CELLS:
        disc = [c for c in CELLS if _distance(cell, c) <= 2]
        assert drawn[cell] == _window(disc, end=TRAIN_END + k), (cell, k)


def test_rolling_origin_zero_is_the_fixed_window_model_byte_for_byte():
    a = RollingNeighbourhoodConflictology(**KW, radius=2).fit(_df()).predict(_df(), 0, 4)["y1"]
    b = NeighbourhoodConflictology(**KW, radius=2).fit(_df()).predict(_df(), 0, 4)["y1"]
    assert np.array_equal(np.asarray(a.values), np.asarray(b.values))


@pytest.mark.parametrize("k", [0, 1, 5, 12])
def test_rolling_radius_zero_is_rolling_conflictology_byte_for_byte(k):
    df = _df()
    df["y1"] = np.random.default_rng(1).poisson(2.0, len(df)).astype(float)
    got = RollingNeighbourhoodConflictology(**KW, radius=0).fit(df).predict(df, k, 3)["y1"]
    want = RollingConflictology(**KW).fit(df).predict(df, k, 3)["y1"]
    assert np.array_equal(np.asarray(got.values), np.asarray(want.values))


@pytest.mark.parametrize("bad", ["missing", "nan"])
def test_rolling_a_gap_inside_the_window_raises_naming_the_model(bad):
    df = _df()
    hole = (df.index.get_level_values("priogrid_id") == _id(100, 1)) & (
        df.index.get_level_values("month_id") == TRAIN_END + 2)
    df = df[~hole] if bad == "missing" else df.assign(y1=np.where(hole, np.nan, df["y1"]))
    m = RollingNeighbourhoodConflictology(**KW, radius=2).fit(df)
    m.predict(df, sequence_number=1, output_length=1)              # the gap is not in this window
    with pytest.raises(ValueError, match="RollingNeighbourhoodConflictology"):
        m.predict(df, sequence_number=2, output_length=1)


def test_rolling_negative_sequence_number_raises():
    m = RollingNeighbourhoodConflictology(**KW, radius=1).fit(_df())
    with pytest.raises(ValueError, match="RollingNeighbourhoodConflictology"):
        m.predict(_df(), sequence_number=-1, output_length=1)


@pytest.mark.parametrize("k", [0, 4])
def test_rolling_forecasts_are_labelled_with_the_origins_months(k):
    pf = RollingNeighbourhoodConflictology(**KW, radius=1).fit(_df()).predict(_df(), k, 3)["y1"]
    assert set(np.asarray(pf.index.time)) == {500 + k, 501 + k, 502 + k}


def test_rolling_the_disc_stays_as_fitted():
    """A cell that first appears after train_end is neither forecast nor pooled."""
    df = _df()
    late = pd.DataFrame({"y1": [float(m * 100 + 99) for m in range(TRAIN_END + 1, 540)]},
                        index=pd.MultiIndex.from_product([range(TRAIN_END + 1, 540),
                                                          [_id(100, 2)]],
                                                         names=["month_id", "priogrid_id"]))
    df = pd.concat([df, late]).sort_index()
    pf = RollingNeighbourhoodConflictology(**KW, radius=2).fit(df).predict(df, 5, 1)["y1"]
    assert _id(100, 2) not in set(np.asarray(pf.index.unit))
    assert not (np.asarray(pf.values).astype(np.int64) % 100 == 99).any()
