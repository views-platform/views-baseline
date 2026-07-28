"""Unit tests for the distribution + transform registry (ADR-022 / epic #33 S4)."""

import logging

import numpy as np
import pytest

from views_baseline.model.distributions import (
    EMIT_FLOOR,
    EMIT_LOG_CEIL,
    TRANSFORMS,
    clamp_floor,
    clamp_log,
    fit_family,
    sample_family,
    validate_family_transform,
)

_RNG = lambda: np.random.default_rng(0)  # noqa: E731


@pytest.mark.parametrize("family", ["nb", "zinb", "lognormal", "gumbel", "gamma"])
def test_fit_sample_recovers_mean(family):
    # data with a clear positive mean and overdispersion (nb needs var>mean)
    data = _RNG().gamma(2.0, 3.0, size=5000)  # mean 6, var 18
    params = fit_family(family, data)
    draws = sample_family(family, params, 20000, _RNG())
    assert draws.shape == (20000,)
    assert abs(draws.mean() - data.mean()) < 0.5 * data.mean()  # generous


def test_nb_recovers_mean_and_overdispersion():
    data = _RNG().gamma(2.0, 3.0, size=5000)
    params = fit_family("nb", data)
    assert params["kind"] == "nb"
    draws = sample_family("nb", params, 50000, _RNG())
    assert abs(draws.mean() - data.mean()) < 0.15 * data.mean()
    assert draws.var() > draws.mean()  # overdispersed, as fit


def test_nb_underdispersed_falls_back_to_poisson(caplog):
    data = np.array([3.0, 3.0, 3.0, 4.0, 3.0, 3.0])  # var < mean
    with caplog.at_level(logging.WARNING):
        params = fit_family("nb", data)
    assert params["kind"] == "poisson"
    assert "Poisson fallback" in caplog.text


def test_all_zero_window_is_point_mass_at_zero():
    for family in ("nb", "zinb", "lognormal", "gumbel", "gamma"):
        params = fit_family(family, np.zeros(10))
        draws = sample_family(family, params, 100, _RNG())
        assert np.all(draws == 0.0)


def test_zinb_matches_mean_and_zero_rate():
    # zero-inflated overdispersed window: ~92% zeros + a heavy positive remainder.
    rng = _RNG()
    pos = rng.negative_binomial(3, 0.25, size=400).astype(float) + 1.0  # positive, overdispersed
    data = np.concatenate([np.zeros(4600), pos])  # ~92% zeros
    params = fit_family("zinb", data)
    assert params["kind"] in ("zinb", "nb")
    draws = sample_family("zinb", params, 60000, np.random.default_rng(3))
    # both the zero mass and the mean are matched by construction
    assert abs((draws == 0.0).mean() - (data == 0.0).mean()) < 0.02
    assert abs(draws.mean() - data.mean()) < 0.15 * data.mean()
    assert (draws >= 0).all()


def test_zinb_inflates_when_nb_underpredicts_zeros():
    # large, tightly-clustered positives (NB alone would emit few zeros) + many zeros
    # -> structural inflation must kick in (pi > 0).
    rng = _RNG()
    pos = rng.integers(40, 60, size=200).astype(float)   # far from 0; NB(mean~2) ~never 0 here
    data = np.concatenate([np.zeros(1800), pos])
    params = fit_family("zinb", data)
    assert params["kind"] == "zinb"
    assert params["pi"] > 0.0


def test_zinb_no_excess_zeros_gives_negligible_inflation():
    # data generated as a plain NB -> zeros are NB-consistent -> pi ~ 0 (no spurious spike).
    # (MoM fit is inexact, so pi may be a hair above 0 rather than an exact nb fallback.)
    data = _RNG().negative_binomial(5, 0.5, size=8000).astype(float)
    params = fit_family("zinb", data)
    assert params["kind"] == "nb" or params["pi"] < 0.05


def test_zinb_reproducible_under_same_rng():
    data = np.concatenate([np.zeros(500), _RNG().negative_binomial(3, 0.3, 100).astype(float) + 1])
    params = fit_family("zinb", data)
    a = sample_family("zinb", params, 500, np.random.default_rng(9))
    b = sample_family("zinb", params, 500, np.random.default_rng(9))
    np.testing.assert_array_equal(a, b)


def test_single_value_window_is_point_mass():
    params = fit_family("gamma", np.full(8, 5.0))
    draws = sample_family("gamma", params, 100, _RNG())
    assert np.all(draws == 5.0)


def test_zinb_underdispersed_falls_back_to_nb_component(caplog):
    # var <= mean -> NB component undefined -> zinb falls back (Poisson), WARN, no crash.
    data = np.array([3.0, 3.0, 3.0, 4.0, 3.0, 3.0])  # underdispersed, no zeros
    with caplog.at_level(logging.WARNING):
        params = fit_family("zinb", data)
    assert params["kind"] in ("poisson", "point")
    assert "NB component undefined" in caplog.text
    draws = sample_family("zinb", params, 200, _RNG())
    assert draws.shape == (200,) and (draws >= 0).all()


def test_lognormal_single_positive_is_point_mass():
    # one positive value -> log-space std 0 -> point mass at that value (no NaN)
    params = fit_family("lognormal", np.array([0.0, 0.0, 7.0, 0.0]))
    draws = sample_family("lognormal", params, 100, _RNG())
    assert np.all(draws == 7.0)


def test_transforms_round_trip():
    fwd, inv = TRANSFORMS["log1p"]
    x = np.array([0.0, 1.0, 7.0, 123.0])
    np.testing.assert_allclose(inv(fwd(x)), x, rtol=1e-12)


def test_clamp_log_caps_at_ceiling(caplog):
    with caplog.at_level(logging.WARNING):
        out = clamp_log(np.array([1.0, EMIT_LOG_CEIL + 5, 50.0]))
    assert out.max() == EMIT_LOG_CEIL
    assert "clamped" in caplog.text


def test_clamp_floor_caps_at_floor(caplog):
    with caplog.at_level(logging.WARNING):
        out = clamp_floor(np.array([-5.0, -0.3, 0.0, 7.0]))
    assert out.min() == EMIT_FLOOR
    np.testing.assert_array_equal(out, np.array([0.0, 0.0, 0.0, 7.0]))
    assert "floored" in caplog.text


def test_clamp_floor_leaves_nonnegative_untouched(caplog):
    x = np.array([0.0, 1.5, 900.0])
    with caplog.at_level(logging.WARNING):
        out = clamp_floor(x)
    np.testing.assert_array_equal(out, x)
    assert "floored" not in caplog.text


def test_validate_family_transform():
    validate_family_transform("lognormal", "log1p")  # ok
    validate_family_transform("nb", "none")  # ok
    validate_family_transform("zinb", "none")  # ok
    with pytest.raises(ValueError, match="invalid for count family"):
        validate_family_transform("nb", "log1p")
    with pytest.raises(ValueError, match="invalid for count family"):
        validate_family_transform("zinb", "log1p")
    with pytest.raises(ValueError, match="Unknown transform"):
        validate_family_transform("gamma", "sqrt")


def test_reproducible_under_same_rng():
    data = _RNG().gamma(2.0, 3.0, size=1000)
    params = fit_family("gamma", data)
    a = sample_family("gamma", params, 500, np.random.default_rng(7))
    b = sample_family("gamma", params, 500, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_unknown_family_raises():
    with pytest.raises(ValueError, match="Unknown family"):
        fit_family("weibull", np.arange(5.0))
