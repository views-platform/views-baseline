"""Golden / characterization tests for the distributional sampling path (C-29).

Unlike the reproducibility tests (which assert same-seed *self-consistency* — determinism,
not regression), these pin the **exact** ``y_pred`` of each distributional model against a
captured reference on a fixed seed + fixed tiny window. A draw-path change (RNG order or
per-cell logic in ``sample_prediction_grid`` or a ``draw_cell`` closure) that preserves shape
and marginal distribution — the exact class of change the ``sample_prediction_grid`` refactor
was — now fails loudly here. Regenerate the constants **deliberately** (never blindly) only
when the sampling contract intentionally changes.
"""

import numpy as np
import pandas as pd

from views_baseline.model.baseline import (
    ConflictologyModel,
    MixtureBaseline,
    ParametricConflictology,
    ParametricHurdleConflictology,
)

_PARTITION = {"test": (493, 540)}


def _golden_df():
    idx = pd.MultiIndex.from_product(
        [range(481, 493), [1, 2]], names=["month_id", "priogrid_id"]
    )
    t = idx.get_level_values(0).to_numpy()
    u = idx.get_level_values(1).to_numpy()
    return pd.DataFrame({"y1": ((t * 3 + u * 7) % 9).astype(float)}, index=idx)


def _run(model):
    df = _golden_df()
    model.fit(df)
    return model.predict(df=df, sequence_number=0, output_length=1)["y1"].values


def test_golden_conflictology():
    out = _run(ConflictologyModel(
        targets=["y1"], window_months=6, partition_dict=_PARTITION, loa="pgm",
        n_samples=4, seed=42))
    np.testing.assert_array_equal(out, [[1.0, 4.0, 1.0, 7.0], [5.0, 5.0, 8.0, 2.0]])


def test_golden_mixture():
    out = _run(MixtureBaseline(
        targets=["y1"], window_months=6, lambda_mix=0.3, n_samples=4,
        partition_dict=_PARTITION, loa="pgm", seed=42))
    np.testing.assert_array_equal(out, [[4.0, 1.0, 1.0, 7.0], [5.0, 2.0, 8.0, 5.0]])


def test_golden_parametric_nb():
    out = _run(ParametricConflictology(
        targets=["y1"], window_months=6, partition_dict=_PARTITION, loa="pgm",
        n_samples=4, family="nb", transform="none", seed=42))
    np.testing.assert_array_equal(out, [[4.0, 4.0, 3.0, 5.0], [4.0, 9.0, 0.0, 7.0]])


def test_golden_parametric_hurdle_gamma():
    out = _run(ParametricHurdleConflictology(
        targets=["y1"], window_months=6, partition_dict=_PARTITION, loa="pgm",
        n_samples=4, family="gamma", transform="none", seed=42))
    np.testing.assert_allclose(out, [
        [3.8011667728424072, 3.4616448879241943, 5.926358222961426, 3.6534857749938965],
        [6.987948894500732, 4.179197788238525, 8.111642837524414, 3.665259599685669],
    ], rtol=1e-6)
