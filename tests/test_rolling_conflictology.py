"""RollingConflictology (#126, ADR-024's control): conflictology whose window ends at each
rolling origin, ``train_end + sequence_number``.

The input from `make_dummy_df` encodes its month in every value (``y1 = month * 10 + entity``),
so the month each draw came from can be read back exactly.
"""

import numpy as np
import pytest
from conftest import make_dummy_df

from views_baseline.model.models.distributional import ConflictologyModel, RollingConflictology

PARTITION = {"test": (500, 535)}           # train_end = 499
TRAIN_END = 499
WINDOW = 6
KW = dict(targets=["y1"], window_months=WINDOW, partition_dict=PARTITION, loa="pgm",
          n_samples=200, seed=7)


def _months_drawn(pf, entity_ids):
    """Months each draw came from, per row (y1 = month * 10 + entity)."""
    vals = np.asarray(pf.values)
    ents = np.asarray(entity_ids)
    return (vals - ents[:, None]) / 10


def _row_entities(pf):
    return np.asarray(pf.index.unit)


@pytest.fixture
def df():
    return make_dummy_df(time_range=range(440, 540))


def test_origin_zero_is_byte_identical_to_conflictology(df):
    a = ConflictologyModel(**KW).fit(df).predict(df, sequence_number=0, output_length=12)["y1"]
    r = RollingConflictology(**KW).fit(df).predict(df, sequence_number=0, output_length=12)["y1"]
    np.testing.assert_array_equal(np.asarray(a.values), np.asarray(r.values))
    np.testing.assert_array_equal(np.asarray(a.index.time), np.asarray(r.index.time))
    np.testing.assert_array_equal(np.asarray(a.index.unit), np.asarray(r.index.unit))


@pytest.mark.parametrize("k", [0, 1, 5, 12, 30])
def test_window_ends_at_train_end_plus_k(df, k):
    """Origin k draws only from months train_end + k - WINDOW + 1 .. train_end + k, and from all of
    them: never from a month it forecasts (the first is test_start + k = train_end + k + 1)."""
    pf = RollingConflictology(**KW).fit(df).predict(df, sequence_number=k, output_length=6)["y1"]
    months = _months_drawn(pf, _row_entities(pf))
    anchor = TRAIN_END + k
    assert months.min() == anchor - WINDOW + 1
    assert months.max() == anchor
    first_forecast = int(np.asarray(pf.index.time).min())
    assert first_forecast == anchor + 1 and months.max() < first_forecast


def test_conflictology_does_not_move_its_window(df):
    """The contrast this control exists for: A's window stays at train_end for every origin."""
    pf = ConflictologyModel(**KW).fit(df).predict(df, sequence_number=12, output_length=6)["y1"]
    assert _months_drawn(pf, _row_entities(pf)).max() == TRAIN_END


def test_fails_loud_when_the_data_does_not_reach_the_origin(df):
    short = make_dummy_df(time_range=range(440, 505))       # data ends at month 504
    m = RollingConflictology(**KW).fit(short)
    m.predict(short, sequence_number=5, output_length=6)     # anchor 504: present
    with pytest.raises(ValueError, match="month 505"):
        m.predict(short, sequence_number=6, output_length=6)  # anchor 505: absent


def test_fails_loud_when_one_cell_is_missing_at_the_origin(df):
    holed = df.drop(index=(503, 2))                          # entity 2 has no row at month 503
    m = RollingConflictology(**KW).fit(holed)
    with pytest.raises(ValueError, match="1 of 2 cells"):
        m.predict(holed, sequence_number=4, output_length=6)


def test_reproducible_and_seed_sensitive(df):
    def run(seed):
        m = RollingConflictology(**{**KW, "seed": seed}).fit(df)
        return np.asarray(m.predict(df, sequence_number=3, output_length=6)["y1"].values)
    np.testing.assert_array_equal(run(7), run(7))
    assert not np.array_equal(run(7), run(8))


# --- added after the independent guard audit of ca78cdf (survivors K, F, I) ---------------

KW2 = {**KW, "targets": ["y1", "y2"]}


def test_each_target_draws_from_its_own_history_at_every_origin(df):
    """Guards against mixing targets when re-pooling at k > 0 (y1 = month*10 + e < 10,000;
    y2 = month*100 + e >= 10,000)."""
    m = RollingConflictology(**KW2).fit(df)
    for k in (0, 3):
        out = m.predict(df, sequence_number=k, output_length=6)
        e = _row_entities(out["y1"])[:, None]
        y1, y2 = np.asarray(out["y1"].values), np.asarray(out["y2"].values)
        assert np.all((y1 - e) % 10 == 0) and np.all(y1 < 10_000)
        assert np.all(y2 >= 10_000)


def test_entity_set_is_fixed_at_fit_even_if_a_cell_appears_later(df):
    """A cell that first appears after train_end never enters the forecast, at any origin,
    and an earlier predict() call cannot change a later one."""
    import pandas as pd

    months = range(501, 540)
    extra = pd.DataFrame(
        {"y1": [t * 10 + 3 for t in months], "y2": [t * 100 + 3 for t in months]},
        index=pd.MultiIndex.from_product([months, [3]], names=df.index.names),
    )
    df2 = pd.concat([df, extra]).sort_index()
    m = RollingConflictology(**KW2).fit(df2)
    for k in (5, 0):
        pf = m.predict(df2, sequence_number=k, output_length=6)["y1"]
        assert set(_row_entities(pf).tolist()) == {1, 2}


@pytest.mark.parametrize("k", [1, 12])
def test_same_draw_indices_as_conflictology_at_every_origin(df, k):
    """Common random numbers at every origin, not only origin 0: the same seed picks the same
    positions in the pool, so each draw is conflictology's shifted forward by exactly k months.
    That keeps the control-vs-conflictology comparison paired."""
    a = ConflictologyModel(**KW).fit(df).predict(df, sequence_number=k, output_length=6)["y1"]
    r = RollingConflictology(**KW).fit(df).predict(df, sequence_number=k, output_length=6)["y1"]
    np.testing.assert_array_equal(
        (np.asarray(r.values) - np.asarray(a.values)) / 10, k
    )
