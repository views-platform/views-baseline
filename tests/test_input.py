"""Unit tests for the FeatureFrame input boundary (ADR-019 / PR-2 S5).

Covers ``to_feature_frame`` (df -> FeatureFrame lift, FeatureFrame passthrough +
loa<->level validation) and ``panel`` (numpy view extraction). No model caller yet —
the adapter is exercised in isolation.
"""

import numpy as np
import pytest
from conftest import make_dummy_df

from views_baseline.model.frames.input import panel, to_feature_frame

pytest.importorskip("views_frames")
from views_frames import FeatureFrame, SpatialLevel  # noqa: E402


def _df():
    return make_dummy_df(entity_id="priogrid_id", time_range=range(490, 500))


# ---------------------------------------------------------------------------
# to_feature_frame — DataFrame lift
# ---------------------------------------------------------------------------
def test_to_feature_frame_lifts_dataframe():
    df = _df()
    ff = to_feature_frame(df, loa="pgm", targets=["y1", "y2"])

    assert isinstance(ff, FeatureFrame)
    assert list(ff.feature_names) == ["y1", "y2"]
    assert ff.values.shape == (len(df), 2, 1)
    # FeatureFrame is a float32 container by design (views_frames contract). This is the
    # crux of the byte-identity limit for the FeatureFrame input path (see C-32).
    assert ff.values.dtype == np.float32
    assert ff.index.level is SpatialLevel.PGM
    np.testing.assert_array_equal(ff.index.time, df.index.get_level_values(0).to_numpy())
    np.testing.assert_array_equal(ff.index.unit, df.index.get_level_values(1).to_numpy())


def test_to_feature_frame_lift_preserves_values_and_order():
    df = _df()
    ff = to_feature_frame(df, loa="pgm", targets=["y1", "y2"])
    np.testing.assert_array_equal(ff.values[:, 0, 0], df["y1"].to_numpy(dtype=np.float64))
    np.testing.assert_array_equal(ff.values[:, 1, 0], df["y2"].to_numpy(dtype=np.float64))


def test_to_feature_frame_lift_only_carries_requested_targets():
    df = _df()
    ff = to_feature_frame(df, loa="pgm", targets=["y2"])
    assert list(ff.feature_names) == ["y2"]
    assert ff.values.shape == (len(df), 1, 1)


def test_from_dataframe_rejects_index_name_mismatch():
    # country_id index but declared pgm -> resolve_level disagreement (ADR-003)
    df = make_dummy_df(entity_id="country_id", time_range=range(490, 500))
    with pytest.raises(ValueError, match="does not match the declared spatial level"):
        to_feature_frame(df, loa="pgm", targets=["y1"])


def test_from_dataframe_rejects_missing_target_column():
    df = _df()
    with pytest.raises(ValueError, match="missing required target column"):
        to_feature_frame(df, loa="pgm", targets=["y3"])


def test_unknown_loa_raises():
    df = _df()
    with pytest.raises(ValueError, match="Unknown level of analysis"):
        to_feature_frame(df, loa="bogus", targets=["y1"])


# ---------------------------------------------------------------------------
# to_feature_frame — FeatureFrame passthrough
# ---------------------------------------------------------------------------
def test_feature_frame_passthrough_is_identity():
    ff = to_feature_frame(_df(), loa="pgm", targets=["y1", "y2"])
    out = to_feature_frame(ff, loa="pgm", targets=["y1"])
    assert out is ff  # no copy, no pandas


def test_passthrough_rejects_loa_level_mismatch():
    ff = to_feature_frame(_df(), loa="pgm", targets=["y1"])  # PGM
    with pytest.raises(ValueError, match="disagrees with declared loa"):
        to_feature_frame(ff, loa="cm", targets=["y1"])


def test_passthrough_rejects_missing_target_feature():
    ff = to_feature_frame(_df(), loa="pgm", targets=["y1", "y2"])
    with pytest.raises(ValueError, match="missing required target feature"):
        to_feature_frame(ff, loa="pgm", targets=["y3"])


def test_rejects_non_frame_non_dataframe():
    with pytest.raises(TypeError, match="pandas.DataFrame or a views_frames.FeatureFrame"):
        to_feature_frame([1, 2, 3], loa="pgm", targets=["y1"])


# ---------------------------------------------------------------------------
# panel — numpy view
# ---------------------------------------------------------------------------
def test_panel_returns_time_unit_and_per_target_series():
    df = _df()
    ff = to_feature_frame(df, loa="pgm", targets=["y1", "y2"])
    time, unit, values = panel(ff, ["y1", "y2"])

    np.testing.assert_array_equal(time, df.index.get_level_values(0).to_numpy())
    np.testing.assert_array_equal(unit, df.index.get_level_values(1).to_numpy())
    assert set(values) == {"y1", "y2"}
    for t in ("y1", "y2"):
        assert values[t].shape == (len(df),)
        np.testing.assert_array_equal(values[t], df[t].to_numpy(dtype=np.float64))


def test_panel_rejects_unknown_target():
    ff = to_feature_frame(_df(), loa="pgm", targets=["y1"])
    with pytest.raises(ValueError, match="not a feature of the FeatureFrame"):
        panel(ff, ["y3"])
