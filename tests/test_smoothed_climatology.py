"""Tests for the smoothed climatology baselines (ADR-023, epic #104).

Each test maps to a line of ADR-023 "Validation & Monitoring". B is
`InterpolatedQuantileConflictology`; A is `ConflictologyModel`.
"""

import logging
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from views_frames import PredictionFrame

from views_baseline.model.models.distributional import (
    ConflictologyModel,
    InterpolatedQuantileConflictology,
)
from views_baseline.model.models.distributional.smoothing import (
    ALL_ZERO,
    MIN_POSITIVES_FOR_SMOOTHING,
    NO_SPREAD,
    SMOOTH,
    TOO_FEW_POSITIVES,
    classify_window,
    round_positive,
    warn_fallbacks,
)

_PARTITION = {"test": (493, 540)}
_WINDOW = 12


def _df_from_windows(windows: dict[int, list[float]]) -> pd.DataFrame:
    """One entity per key; its values are the last len(values) months before the test."""
    months = range(493 - _WINDOW, 493)
    rows, vals = [], []
    for cid, w in windows.items():
        assert len(w) == _WINDOW
        for m, v in zip(months, w):
            rows.append((m, cid))
            vals.append(float(v))
    idx = pd.MultiIndex.from_tuples(rows, names=["month_id", "priogrid_id"])
    return pd.DataFrame({"y1": vals}, index=idx).sort_index()


# Entity 1: steep decay, many zeros — the shape the evaluation cares about.
# Entity 2: all zero. Entity 3: one positive. Entity 4: positives with no spread.
_WINDOWS = {
    1: [0, 0, 0, 0, 0, 1, 1, 2, 3, 0, 40, 0],
    2: [0] * 12,
    3: [0] * 11 + [7],
    4: [0, 5, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0],
}


# Pinned draws for entity 1 alone, S=8, seed=42 (test_golden_b_with_zeros). Regenerate
# deliberately, never blindly — see the note in test_golden.py.
GOLDEN_B_WITH_ZEROS = [[1.0, 0.0, 2.0, 1.0, 0.0, 29.0, 3.0, 14.0]]


def _b(n_samples=64, seed=42):
    return InterpolatedQuantileConflictology(
        targets=["y1"], window_months=_WINDOW, partition_dict=_PARTITION, loa="pgm",
        n_samples=n_samples, seed=seed,
    )


def _a(n_samples=64, seed=42):
    return ConflictologyModel(
        targets=["y1"], window_months=_WINDOW, partition_dict=_PARTITION, loa="pgm",
        n_samples=n_samples, seed=seed,
    )


def _draws_by_entity(model, df, output_length=1):
    pf = model.fit(df).predict(df=df, sequence_number=0, output_length=output_length)["y1"]
    unit = np.asarray(pf.index.unit)
    return {cid: pf.values[unit == cid].ravel() for cid in np.unique(unit)}


# --------------------------------------------------------------------------------------
# Shared rules (smoothing.py)
# --------------------------------------------------------------------------------------


def test_min_positives_is_two():
    """ADR-023 §2 fixes the threshold; changing it is a decision, not a refactor."""
    assert MIN_POSITIVES_FOR_SMOOTHING == 2


@pytest.mark.parametrize(
    "pool, kind",
    [
        ([0, 0, 0], ALL_ZERO),
        ([0, 4, 0], TOO_FEW_POSITIVES),
        ([3, 0, 3], NO_SPREAD),
        ([1, 0, 9], SMOOTH),
    ],
)
def test_classify_window(pool, kind):
    got, positives = classify_window(np.asarray(pool, dtype=float))
    assert got == kind
    assert (positives > 0).all()
    assert np.all(np.diff(positives) >= 0)


def test_round_positive_rounds_half_up_and_stays_positive():
    x = np.array([0.5, 0.2, 1.49, 1.5, 2.5, 9.99])
    np.testing.assert_array_equal(round_positive(x), [1, 1, 1, 2, 3, 10])


def test_warn_fallbacks_is_one_aggregate_warning(caplog):
    with caplog.at_level(logging.WARNING):
        warn_fallbacks("M", Counter({SMOOTH: 3, ALL_ZERO: 5, NO_SPREAD: 1}))
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 1
    assert "6 of 9" in msgs[0] and "all_zero=5" in msgs[0] and "no_spread=1" in msgs[0]


def test_warn_fallbacks_silent_when_all_smoothed(caplog):
    with caplog.at_level(logging.WARNING):
        warn_fallbacks("M", Counter({SMOOTH: 4}))
    assert not caplog.records


# --------------------------------------------------------------------------------------
# B — InterpolatedQuantileConflictology
# --------------------------------------------------------------------------------------


def test_b_output_shape_and_type():
    df = _df_from_windows(_WINDOWS)
    out = _b().fit(df).predict(df=df, sequence_number=0, output_length=3)
    assert set(out) == {"y1"}
    assert isinstance(out["y1"], PredictionFrame)
    assert out["y1"].values.shape == (4 * 3, 64)


def test_b_pool_is_as():
    """Every estimator: the pool is A's (ADR-023 §1). B keeps A's positives, sorted."""
    df = _df_from_windows(_WINDOWS)
    a, b = _a().fit(df), _b().fit(df)
    np.testing.assert_array_equal(a.entity_ids, b.entity_ids)
    for cid in a.entity_ids:
        pool = a.hist_per_entity[cid]["y1"]
        cell = b.cells[cid]["y1"]
        if cell["kind"] == SMOOTH:
            np.testing.assert_array_equal(cell["positives"], np.sort(pool[pool > 0]))
        else:
            np.testing.assert_array_equal(cell["pool"], pool)


def test_b_draws_are_integers_and_positive_draws_at_least_one():
    df = _df_from_windows(_WINDOWS)
    v = _b(n_samples=2000).fit(df).predict(df=df, sequence_number=0, output_length=2)["y1"].values
    assert (v >= 0).all()
    np.testing.assert_array_equal(v, np.round(v))
    assert (v[v > 0] >= 1).all()


def test_b_zero_mass_equals_as():
    """Every estimator: zero mass equals A's within Monte-Carlo error (ADR-023 §2).

    With S = 20,000 the binomial SE of a zero fraction is ≤ 0.0036; 5 SE is the bound.
    """
    df = _df_from_windows(_WINDOWS)
    s = 20_000
    a = _draws_by_entity(_a(n_samples=s, seed=1), df)
    b = _draws_by_entity(_b(n_samples=s, seed=2), df)
    for cid, w in _WINDOWS.items():
        expected = np.mean(np.asarray(w) == 0)
        for draws in (a[cid], b[cid]):
            assert abs(np.mean(draws == 0) - expected) < 5 * np.sqrt(0.25 / s)


def test_b_never_leaves_the_observed_positive_range():
    """B: no extrapolation (ADR-023 §3)."""
    df = _df_from_windows(_WINDOWS)
    draws = _draws_by_entity(_b(n_samples=5000), df)
    for cid, w in _WINDOWS.items():
        pos = draws[cid][draws[cid] > 0]
        obs = [x for x in w if x > 0]
        if obs:
            assert pos.min() >= min(obs) and pos.max() <= max(obs)


def test_b_fills_the_gaps_a_cannot():
    """The point of B: between 3 and 40, A can only draw {3, 40}; B draws the counts between."""
    df = _df_from_windows(_WINDOWS)
    a = _draws_by_entity(_a(n_samples=5000), df)[1]
    b = _draws_by_entity(_b(n_samples=5000), df)[1]
    gap = np.arange(4, 40)
    assert not np.isin(a, gap).any()
    assert np.isin(b, gap).sum() > 100


def test_b_interpolates_on_plotting_positions():
    """Q₊(u) at u = (i−1)/(m−1) is p(i); between, it is linear (ADR-023 §3)."""
    df = _df_from_windows(_WINDOWS)
    cell = _b().fit(df).cells[1]["y1"]
    np.testing.assert_array_equal(cell["positives"], [1, 1, 2, 3, 40])
    np.testing.assert_allclose(cell["positions"], [0, 0.25, 0.5, 0.75, 1.0])
    # u = 0.875 is halfway from 3 to 40.
    assert np.interp(0.875, cell["positions"], cell["positives"]) == pytest.approx(21.5)


def test_b_fallback_cells_are_drawn_exactly_as_a_draws_them():
    """Fallback cells use A's rule, so with nothing else consuming the RNG they match A."""
    df = _df_from_windows({k: v for k, v in _WINDOWS.items() if k != 1})
    a = _a(seed=9).fit(df).predict(df=df, sequence_number=0, output_length=2)["y1"].values
    b = _b(seed=9).fit(df).predict(df=df, sequence_number=0, output_length=2)["y1"].values
    np.testing.assert_array_equal(a, b)


def test_b_counts_each_fallback_and_warns_once(caplog):
    df = _df_from_windows(_WINDOWS)
    m = _b().fit(df)
    assert m.fallback_counts == Counter(
        {SMOOTH: 1, ALL_ZERO: 1, TOO_FEW_POSITIVES: 1, NO_SPREAD: 1}
    )
    with caplog.at_level(logging.WARNING):
        m.predict(df=df, sequence_number=0, output_length=5)
    warns = [r for r in caplog.records if "not smoothed" in r.getMessage()]
    assert len(warns) == 1
    assert "3 of 4" in warns[0].getMessage()


def test_b_reproducible_and_seed_sensitive():
    df = _df_from_windows(_WINDOWS)
    o1 = _b(seed=7).fit(df).predict(df=df, sequence_number=0, output_length=2)["y1"].values
    o2 = _b(seed=7).fit(df).predict(df=df, sequence_number=0, output_length=2)["y1"].values
    o3 = _b(seed=8).fit(df).predict(df=df, sequence_number=0, output_length=2)["y1"].values
    np.testing.assert_array_equal(o1, o2)
    assert not np.array_equal(o1, o3)


def test_b_refit_resets_fallback_counts():
    df = _df_from_windows(_WINDOWS)
    m = _b().fit(df).fit(df)
    assert sum(m.fallback_counts.values()) == 4


def test_b_zero_rate_is_the_windows_exactly():
    """The statistical zero-mass test tolerates ±1.7 points; this pins the stored rate exactly.

    Guard audit of 95717ab: a ±0.015 bias in `zero_rate` survived every other test.
    """
    df = _df_from_windows(_WINDOWS)
    assert _b().fit(df).cells[1]["y1"]["zero_rate"] == 7 / 12


def test_golden_b_with_zeros():
    """Exact draws on a window that has zeros, so the zero gate's RNG path is pinned too."""
    df = _df_from_windows({1: _WINDOWS[1]})
    out = _b(n_samples=8, seed=42).fit(df).predict(df=df, sequence_number=0, output_length=1)
    np.testing.assert_array_equal(out["y1"].values, GOLDEN_B_WITH_ZEROS)
