"""Dual-input tests (PR-2 S10, ADR-019): every model accepts a pandas DataFrame OR a
views_frames FeatureFrame, and both routes produce identical output.

Equivalence is **exact** (not merely close): a DataFrame and the FeatureFrame built from it
via ``to_feature_frame`` converge to the *same* float32 FeatureFrame inside every model's
entry boundary, so the downstream windowing, fitting, and seeded sampling are byte-identical.
"""

import numpy as np
import pytest
from conftest import make_dummy_df, make_dummy_ff

from views_baseline.model.models.distributional import (
    ConflictologyModel,
    MixtureBaseline,
    ParametricConflictology,
    ParametricHurdleConflictology,
)
from views_baseline.model.models.point import AverageModel, LocfModel, ZeroModel

pytest.importorskip("views_frames")

_P = {"test": (493, 540)}
_BASE = dict(targets=["y1", "y2"], partition_dict=_P, loa="pgm")

# (name, zero-arg factory) — factories so each fit() gets a fresh instance.
_MODELS = [
    ("ZeroModel", lambda: ZeroModel(**_BASE)),
    ("LocfModel", lambda: LocfModel(**_BASE)),
    ("AverageModel", lambda: AverageModel(**_BASE, window_months=3)),
    ("ConflictologyModel",
     lambda: ConflictologyModel(**_BASE, window_months=3, n_samples=8, seed=42)),
    ("MixtureBaseline",
     lambda: MixtureBaseline(**_BASE, window_months=3, lambda_mix=0.1, n_samples=8, seed=42)),
    ("ParametricConflictology",
     lambda: ParametricConflictology(**_BASE, window_months=3, n_samples=8,
                                     family="nb", transform="none", seed=42)),
    ("ParametricHurdleConflictology",
     lambda: ParametricHurdleConflictology(**_BASE, window_months=3, n_samples=8,
                                           family="gamma", transform="none", seed=42)),
]
_IDS = [n for n, _ in _MODELS]


@pytest.mark.parametrize("factory", [f for _, f in _MODELS], ids=_IDS)
def test_df_and_featureframe_inputs_are_equivalent(factory):
    df = make_dummy_df()
    ff = make_dummy_ff()  # == to_feature_frame(df, loa="pgm", targets=["y1", "y2"])

    out_df = factory().fit(df).predict(df=df, sequence_number=0, output_length=2)
    out_ff = factory().fit(ff).predict(df=ff, sequence_number=0, output_length=2)

    assert set(out_df) == set(out_ff)
    for t in out_df:
        np.testing.assert_array_equal(out_df[t].values, out_ff[t].values)
        np.testing.assert_array_equal(
            np.asarray(out_df[t].index.unit), np.asarray(out_ff[t].index.unit)
        )
        np.testing.assert_array_equal(
            np.asarray(out_df[t].index.time), np.asarray(out_ff[t].index.time)
        )


@pytest.mark.parametrize("factory", [f for _, f in _MODELS], ids=_IDS)
def test_featureframe_input_produces_valid_output(factory):
    """A FeatureFrame alone (no DataFrame anywhere) drives a full fit/predict."""
    ff = make_dummy_ff()
    out = factory().fit(ff).predict(df=ff, sequence_number=0, output_length=3)

    assert set(out) == {"y1", "y2"}
    for t in out:
        assert out[t].values.shape[0] > 0
        assert not np.isnan(out[t].values).any()
