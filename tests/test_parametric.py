"""Tests for the parametric climatology baselines (ADR-022 / epic #33 S5, S6)."""

import logging

import numpy as np
import pandas as pd
import pytest
from views_frames import PredictionFrame

from views_baseline.model.baseline import (
    ParametricConflictology,
    ParametricHurdleConflictology,
)

_PARTITION = {"test": (493, 540)}


def _df(entities=(1, 2, 3)):
    idx = pd.MultiIndex.from_product(
        [range(440, 500), list(entities)], names=["month_id", "priogrid_id"]
    )
    vals = np.random.default_rng(0).integers(0, 8, size=len(idx)).astype(float)
    return pd.DataFrame({"y1": vals}, index=idx)


def _model(seed=42, family="nb", transform="none"):
    return ParametricConflictology(
        targets=["y1"], window_months=12, partition_dict=_PARTITION,
        loa="pgm", n_samples=64, family=family, transform=transform, seed=seed,
    )


def test_output_shape_and_type():
    df = _df()
    m = _model()
    m.fit(df)
    out = m.predict(df=df, sequence_number=0, output_length=3)
    assert set(out) == {"y1"}
    assert isinstance(out["y1"], PredictionFrame)
    assert out["y1"].values.shape == (3 * 3, 64)  # 3 entities × 3 steps
    assert m.distributional is True


def test_reproducible_under_seed():
    df = _df()
    a = _model(seed=7)
    a.fit(df)
    o1 = a.predict(df=df, sequence_number=0, output_length=2)["y1"].values
    b = _model(seed=7)
    b.fit(df)
    o2 = b.predict(df=df, sequence_number=0, output_length=2)["y1"].values
    np.testing.assert_array_equal(o1, o2)


def test_nb_samples_are_nonnegative():
    df = _df()
    m = _model()
    m.fit(df)
    out = m.predict(df=df, sequence_number=0, output_length=2)
    assert (out["y1"].values >= 0).all()


def test_all_zero_entity_is_point_mass_zero():
    idx = pd.MultiIndex.from_product(
        [range(440, 500), [1, 2, 3]], names=["month_id", "priogrid_id"]
    )
    ent = idx.get_level_values(1).to_numpy()
    base = np.random.default_rng(0).integers(1, 8, size=len(idx)).astype(float)
    vals = np.where(ent == 3, 0.0, base)
    df = pd.DataFrame({"y1": vals}, index=idx)
    m = _model()
    m.fit(df)
    pf = m.predict(df=df, sequence_number=0, output_length=2)["y1"]
    ent3 = pf.values[np.asarray(pf.index.unit) == 3]
    assert (ent3 == 0.0).all()


def test_nb_log1p_is_illegal():
    with pytest.raises(ValueError, match="invalid for count family"):
        _model(family="nb", transform="log1p")


def test_continuous_family_rejected_by_no_hurdle_model():
    with pytest.raises(ValueError, match="native-zero families"):
        _model(family="lognormal")


def test_no_hurdle_accepts_zinb_and_is_nonnegative():
    df = _df()
    m = _model(family="zinb")  # native-zero, no-hurdle (ZINB)
    m.fit(df)
    out = m.predict(df=df, sequence_number=0, output_length=2)
    assert isinstance(out["y1"], PredictionFrame)
    assert (out["y1"].values >= 0).all()


def test_no_hurdle_zinb_reproducible():
    df = _df()
    a = _model(family="zinb", seed=5)
    a.fit(df)
    o1 = a.predict(df=df, sequence_number=0, output_length=2)["y1"].values
    b = _model(family="zinb", seed=5)
    b.fit(df)
    o2 = b.predict(df=df, sequence_number=0, output_length=2)["y1"].values
    np.testing.assert_array_equal(o1, o2)


def test_zinb_log1p_is_illegal():
    with pytest.raises(ValueError, match="invalid for count family"):
        _model(family="zinb", transform="log1p")


def test_hurdle_rejects_zinb():
    with pytest.raises(ValueError, match="continuous positive-part families"):
        _hurdle(family="zinb")


# ----------------------------------------------------------------------------
# ParametricHurdleConflictology
# ----------------------------------------------------------------------------


def _hurdle(seed=42, family="lognormal", transform="none"):
    return ParametricHurdleConflictology(
        targets=["y1"], window_months=12, partition_dict=_PARTITION,
        loa="pgm", n_samples=2000, family=family, transform=transform, seed=seed,
    )


def _half_zero_df():
    # entity 1: alternating 0 / 5 over time -> ~50% zero-rate in the window
    idx = pd.MultiIndex.from_product(
        [range(440, 500), [1, 2]], names=["month_id", "priogrid_id"]
    )
    t = idx.get_level_values(0).to_numpy()
    vals = np.where(t % 2 == 0, 0.0, 5.0)
    return pd.DataFrame({"y1": vals}, index=idx)


def test_hurdle_output_shape_and_type():
    df = _df()
    m = _hurdle()
    m.fit(df)
    out = m.predict(df=df, sequence_number=0, output_length=2)
    assert isinstance(out["y1"], PredictionFrame)
    assert out["y1"].values.shape == (3 * 2, 2000)
    assert m.distributional is True


def test_hurdle_reproducible_under_seed():
    df = _df()
    a = _hurdle(seed=9)
    a.fit(df)
    o1 = a.predict(df=df, sequence_number=0, output_length=2)["y1"].values
    b = _hurdle(seed=9)
    b.fit(df)
    o2 = b.predict(df=df, sequence_number=0, output_length=2)["y1"].values
    np.testing.assert_array_equal(o1, o2)


def test_hurdle_sampled_zero_rate_matches_empirical():
    df = _half_zero_df()  # window zero-rate = 0.5 for entity 1
    m = _hurdle(family="gamma")
    m.fit(df)
    pf = m.predict(df=df, sequence_number=0, output_length=1)["y1"]
    ent1 = pf.values[np.asarray(pf.index.unit) == 1]
    assert abs((ent1 == 0.0).mean() - 0.5) < 0.05


def test_hurdle_log1p_round_trips_to_raw_scale():
    df = _df()
    m = _hurdle(family="gumbel", transform="log1p")  # Vesco path
    m.fit(df)
    out = m.predict(df=df, sequence_number=0, output_length=2)["y1"].values
    assert np.isfinite(out).all()
    assert (out >= 0).all()  # expm1 of non-negative log-space -> non-negative raw counts


def test_hurdle_rejects_count_family():
    with pytest.raises(ValueError, match="continuous positive-part families"):
        _hurdle(family="nb")


def _spread_positive_df():
    # one entity, all-positive window with a heavy outlier -> the gumbel fit's loc dips
    # below 0, so a large fraction of positive-part draws are negative pre-floor.
    months = list(range(481, 493))  # 12 months <= train_end (492)
    vals = [1.0] * 11 + [50.0]
    idx = pd.MultiIndex.from_arrays([months, [1] * 12], names=["month_id", "priogrid_id"])
    return pd.DataFrame({"y1": vals}, index=idx)


def test_hurdle_gumbel_floors_negative_tail_to_nonnegative():
    # gumbel_r has ℝ support; without the floor ~40% of these draws are negative.
    df = _spread_positive_df()
    m = _hurdle(family="gumbel", transform="none")
    m.fit(df)
    out = m.predict(df=df, sequence_number=0, output_length=1)["y1"].values
    assert (out >= 0).all()  # guaranteed by clamp_floor, not by seed luck
    assert (out == 0.0).any()  # floor actually engaged (left tail was present)


def test_no_hurdle_no_entities_fails_loud():
    # a df with no rows at train_end (492) -> no fitted entities -> predict fails loud
    idx = pd.MultiIndex.from_product(
        [range(481, 491), [1, 2]], names=["month_id", "priogrid_id"]
    )
    df = pd.DataFrame({"y1": np.ones(len(idx))}, index=idx)
    m = _model()
    m.fit(df)
    with pytest.raises(ValueError, match="no entities to predict"):
        m.predict(df=df, sequence_number=0, output_length=1)


def test_hurdle_log1p_clamps_overflow_in_predict(caplog):
    # an astronomically large positive -> log1p ≈ 23 > EMIT_LOG_CEIL(20): clamp_log must
    # engage on the positive-part draws in predict() (overflow ceiling on the model path).
    months = list(range(481, 493))
    vals = [0.0] * 11 + [1e10]
    idx = pd.MultiIndex.from_arrays([months, [1] * 12], names=["month_id", "priogrid_id"])
    df = pd.DataFrame({"y1": vals}, index=idx)
    m = _hurdle(family="lognormal", transform="log1p")
    m.fit(df)
    with caplog.at_level(logging.WARNING):
        out = m.predict(df=df, sequence_number=0, output_length=1)["y1"].values
    assert "clamped" in caplog.text
    assert np.isfinite(out).all()
    assert out.max() <= np.expm1(20.0) + 1.0  # emitted positives capped by the ceiling
