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
