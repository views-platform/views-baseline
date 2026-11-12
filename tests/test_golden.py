"""Golden / characterization tests for the distributional sampling path (C-29).

Unlike the reproducibility tests (which assert same-seed *self-consistency* — determinism,
not regression), these pin the **exact** ``y_pred`` of each distributional model against a
captured reference on a fixed seed + fixed tiny window. A draw-path change (RNG order or
per-cell logic in ``sample_prediction_grid`` or a ``draw_cell`` closure) that preserves shape
and marginal distribution — the exact class of change the ``sample_prediction_grid`` refactor
was — now fails loudly here. Regenerate the constants **deliberately** (never blindly) only
when the sampling contract intentionally changes.

Note (C-30): the constants are pinned to the current numpy ``Generator`` streams
(``gamma``/``poisson``/``choice``). A numpy upgrade that changes those algorithms will fail
these tests even with unchanged model code — that is a maintenance regeneration, not a defect.
If ``pyproject.toml`` grows a numpy pin, keep it consistent with the version captured here.
"""

import numpy as np
import pandas as pd

from views_baseline.model.models.distributional import (
    ConflictologyModel,
    InterpolatedQuantileConflictology,
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


def test_golden_interpolated_quantile():
    """B (ADR-023). The windows have no zeros; 6.0 is an interpolated value A cannot draw."""
    out = _run(InterpolatedQuantileConflictology(
        targets=["y1"], window_months=6, partition_dict=_PARTITION, loa="pgm",
        n_samples=4, seed=42))
    np.testing.assert_array_equal(out, [[1.0, 7.0, 6.0, 7.0], [6.0, 8.0, 5.0, 2.0]])


_MATRYOSHKA_CELLS = np.array([1, 2, 20, 40, 721, 722])


def _golden_matryoshka():
    """Matryoshka's draw path (ADR-024) with set knobs: the fit needs ten years of history and
    has its own tests. Each value is month · 1000 + priogrid id, so it shows where it came from.
    Cell 1's four dolls are all non-empty: 2 and 721 are neighbours at distance 1 and 722 at
    √2 (so the kernel's shape matters); 20 shares its country (7) outside the neighbourhood,
    and country 7's block does not start at position 0, because 40 (country 6) comes first."""
    from views_baseline.model.models.distributional.matryoshka import MatryoshkaConflictology

    cells = _MATRYOSHKA_CELLS
    idx = pd.MultiIndex.from_product([range(481, 493), cells],
                                     names=["month_id", "priogrid_id"])
    t = idx.get_level_values(0).to_numpy()
    u = idx.get_level_values(1).to_numpy()
    df = pd.DataFrame({"y1": (t * 1000 + u).astype(float),
                       "c_id": np.where(u == 40, 6.0, 7.0)}, index=idx)
    model = MatryoshkaConflictology(
        targets=["y1"], partition_dict=_PARTITION, loa="pgm", n_samples=16,
        country_column="c_id", backtest_cutoffs=1, backtest_horizon=1,
        stability_refits=False, dolls=["cell", "near", "country", "map"], seed=42)
    model.knobs = {"y1": {"weight": [1.0, 0.8, 0.6, 0.6], "memory": [3.0, 6.0, 12.0, 24.0],
                          "length_scale": 1.5}}
    model.entity_ids, model.history_start = cells, 481
    model.country = np.where(cells == 40, 6, 7)
    return model.predict(df=df, sequence_number=0, output_length=1)["y1"].values


GOLDEN_MATRYOSHKA = [
    [488040, 483721, 491040, 489020, 492001, 485040, 492020, 491040,
     492001, 485722, 489721, 481040, 483020, 489040, 491721, 492001],
    [489020, 491002, 491002, 489002, 481040, 487020, 484020, 486040,
     492001, 485020, 485002, 489002, 485020, 490722, 489020, 491020],
    [489020, 491020, 488020, 491040, 490040, 490040, 490020, 489040,
     488040, 489040, 482722, 490020, 484020, 487040, 486722, 487020],
    [486040, 492040, 488040, 484040, 491040, 490040, 491020, 491002,
     489040, 492040, 489040, 487001, 489040, 492002, 483722, 481002],
    [487721, 487040, 491721, 491721, 484020, 491040, 492721, 491722,
     482040, 482040, 492002, 490721, 489721, 490721, 491721, 492020],
    [491040, 482040, 485722, 490721, 483020, 491040, 486020, 484001,
     492002, 485002, 485020, 483040, 484721, 483722, 485722, 484020],
]


def test_golden_matryoshka():
    out = _golden_matryoshka()
    # The seed reaches every doll of cell 1 (row 0): its own value, a neighbour, its country
    # (20) and the map (40).
    sources = set((out[0].astype(np.int64) % 1000).tolist())
    assert {1, 20, 40} <= sources and sources & {2, 721, 722}
    np.testing.assert_array_equal(out, GOLDEN_MATRYOSHKA)
