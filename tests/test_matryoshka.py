"""MatryoshkaConflictology (ADR-024, #127).

The first four tests port the prototype's checks (reports/matryoshka/prototype/,
check_against_sampling.py and check_slope.py). They build shells and kernel weights independently,
from pairwise distances on a small synthetic grid, rather than from the package's offset table.
The rest test the package: the sampler against the exact probabilities, the conflictology special
case, the fit, the information rule, loud failures, and the shared contracts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from views_baseline.model.models.distributional.matryoshka import MatryoshkaConflictology
from views_baseline.model.models.distributional.matryoshka.doll_counts import (
    COUNTRY,
    DOLLS,
    MAP,
    NEAR,
    build_doll_counts,
    shell_sizes,
)
from views_baseline.model.models.distributional.matryoshka.draw import (
    build_sampler,
    build_shell_tables,
    kept_positions,
)
from views_baseline.model.models.distributional.matryoshka.forecast_probability import (
    doll_shares,
    outcome_classes,
    time_weights,
)
from views_baseline.model.models.distributional.matryoshka.geometry import (
    N_COLS,
    NEAR_RADIUS,
    build_geometry,
)
from views_baseline.model.models.distributional.matryoshka.knob_fit import (
    Objective,
    backtest_cutoffs,
)
from views_baseline.model.models.distributional.matryoshka.size_classes import (
    N_CLASSES,
    classify,
)

pytest.importorskip("views_frames")

VALUES = [0, 0, 0, 0, 1, 3, 12, 40, 300, 2000]


def synthetic_grid(seed=0, n_rows=16, n_cols=22, holes=(40, 41, 63, 150, 151, 200)):
    """Cells with sea holes, two countries split by column, and one cell without a country."""
    rows, cols = np.meshgrid(np.arange(100, 100 + n_rows), np.arange(200, 200 + n_cols),
                             indexing="ij")
    cells = (rows * N_COLS + cols + 1).ravel()
    cells = np.sort(cells[~np.isin(cells, cells[list(holes)])])
    c = (cells - 1) % N_COLS
    country = np.where(c < 200 + n_cols // 2, 7, 9)
    country[3] = 0
    return cells, country


def brute_force_shells(cells, country, length):
    """(4, N, N) shell membership and (4, N, N) P(cell j | doll, cell i), from pairwise
    distances, independently of the package's offset table."""
    r, c = (cells - 1) // N_COLS, (cells - 1) % N_COLS
    dist = np.hypot(r[:, None] - r[None, :], c[:, None] - c[None, :])
    near = (dist > 0) & (dist <= NEAR_RADIUS)
    same_c = (country[:, None] == country[None, :]) & (country[:, None] > 0)
    inner = (dist == 0) | near
    shells = np.stack([dist == 0, near, same_c & ~inner, ~same_c & ~inner])
    pick = np.stack([shells[0], np.where(near, np.exp(-dist / length), 0.0), shells[2],
                     shells[3]]).astype(float)
    pick = pick / np.maximum(pick.sum(axis=2, keepdims=True), 1e-300)
    return shells, pick


@pytest.fixture(scope="module")
def grid():
    cells, country = synthetic_grid()
    geo = build_geometry(cells, country)
    months = np.arange(100, 160)
    values = np.random.default_rng(0).choice(VALUES, size=(months.size, cells.size))
    return cells, country, geo, months, values


# ── ported prototype checks ─────────────────────────────────────────────────────────────


def test_shells_are_disjoint_cover_every_cell_and_have_the_counted_sizes(grid):
    cells, country, geo, _, _ = grid
    shells, _ = brute_force_shells(cells, country, 1.0)
    assert (shells.sum(axis=0) == 1).all(), "shells must be disjoint and cover every cell"
    _, sizes = shell_sizes(geo)
    np.testing.assert_array_equal(shells.sum(axis=2), sizes)


def test_doll_shares_equal_a_direct_kernel_weighted_average(grid):
    cells, country, geo, months, values = grid
    length = 1.7
    _, pick = brute_force_shells(cells, country, length)
    classes = classify(values)
    dc = build_doll_counts(classes, months, geo)
    W = np.stack([time_weights(months, np.array([150]), m) for m in (3.0, 8.0, 40.0, 200.0)])
    onehot = (classes[..., None] == np.arange(N_CLASSES)).astype(float)
    for k in range(len(DOLLS)):
        q = doll_shares(dc, W[k], k, length)[0]
        direct = np.einsum("m,ij,mjk->ik", W[k, 0], pick[k], onehot)
        np.testing.assert_allclose(q, direct, atol=1e-10, err_msg=DOLLS[k])


def test_exact_probabilities_match_an_independent_three_step_draw(grid):
    """The closed form against ADR-024's draw, sampled from the brute-force shells."""
    cells, country, geo, months, values = grid
    length, h, S = 1.7, 5, 200_000
    weight, memory = np.array([1.0, 0.8, 0.4, 0.2]), np.array([3.0, 8.0, 40.0, 200.0])
    shells, pick = brute_force_shells(cells, country, length)
    classes = classify(values)
    dc = build_doll_counts(classes, months, geo)
    cutoff = 150
    q = np.stack([doll_shares(dc, time_weights(months, np.array([cutoff]), memory[k]), k,
                              length)[0] for k in range(len(DOLLS))])        # (4, N, K)
    rng = np.random.default_rng(1)
    hist = months[months <= cutoff]
    for i in (int(np.argmax(dc.sizes[NEAR])), int(np.argmin(dc.sizes[NEAR]))):
        a = np.array([weight[k] * np.exp(-h / memory[k]) if shells[k][i].any() else 0.0
                      for k in range(len(DOLLS))])
        dolls = rng.choice(len(DOLLS), size=S, p=a / a.sum())
        draws = np.empty(S, np.int64)
        for k in range(len(DOLLS)):
            sel = dolls == k
            w = np.exp(-(cutoff - hist) / memory[k])
            ms = rng.choice(hist.size, size=sel.sum(), p=w / w.sum())
            cs = rng.choice(cells.size, size=sel.sum(), p=pick[k][i])
            draws[sel] = classes[ms + (hist[0] - months[0]), cs]
        freq = np.bincount(draws, minlength=N_CLASSES) / S
        np.testing.assert_allclose(freq, (a / a.sum()) @ q[:, i, :], atol=4 / np.sqrt(S))


def test_exact_slope_matches_nudged_differences():
    """Strict: relative tolerance 1e-6, no absolute slack (the prototype's check_slope.py)."""
    cells, country = synthetic_grid(n_rows=14, n_cols=20, holes=())
    geo = build_geometry(cells, country)
    months = np.arange(100, 190)
    values = np.random.default_rng(1).choice(VALUES, size=(months.size, cells.size))
    classes = classify(values)
    dc = build_doll_counts(classes, months, geo)
    cutoffs = np.array([150, 162])
    obj = Objective(dc, cutoffs, outcome_classes(classes, months, cutoffs, 24))
    theta = np.log([0.8, 0.5, 0.2, 3.0, 8.0, 40.0, 200.0, 1.7])
    exact = -obj(theta)[1]
    nudge = 1e-4
    numeric = np.array([
        -(obj(theta + nudge * e)[0] - obj(theta - nudge * e)[0]) / (2 * nudge)
        for e in np.eye(theta.size)
    ])
    np.testing.assert_allclose(exact, numeric, rtol=1e-6, atol=0)


# ── the package's sampler ───────────────────────────────────────────────────────────────


def test_kept_positions_skip_excluded_stretches():
    # row 0: positions 0..9, excluded 2, 3 and 6 → kept 0 1 4 5 7 8 9
    # row 1: excluded 0..4 (one stretch of length 5) → kept 5 6 7 ...
    pad = 99
    kept_before = np.array([[2, 2, 4], [0, pad, pad]])
    skip = np.array([[1, 2, 3], [5, 5, 5]])
    got = kept_positions(np.tile(np.arange(7), (2, 1)), np.array([0, 1]), kept_before, skip)
    np.testing.assert_array_equal(got, [[0, 1, 4, 5, 7, 8, 9], [5, 6, 7, 8, 9, 10, 11]])


def test_sampler_shells_are_exactly_the_brute_force_shells(grid):
    """Every uniform pick j = 0..size−1 of the country and map shells lands on a distinct cell
    of that shell, and together they are the whole shell."""
    cells, country, geo, _, _ = grid
    shells, _ = brute_force_shells(cells, country, 1.0)
    tables = build_shell_tables(geo)
    for i in range(cells.size):
        for k in (COUNTRY, MAP):
            n = int(tables.sizes[k, i])
            j, row = np.arange(n)[None, :], np.array([i])
            if k == COUNTRY:
                p = tables.block_start[i] + kept_positions(
                    j, row, tables.country_kept_before, tables.country_skip)[0]
            else:
                p = kept_positions(j, row, tables.map_kept_before, tables.map_skip)[0]
            got = np.sort(tables.order[p])
            np.testing.assert_array_equal(got, np.nonzero(shells[k][i])[0],
                                          err_msg=f"cell {i}, {DOLLS[k]}")


@pytest.mark.parametrize("h", [1, 12])
def test_sampler_matches_the_exact_probabilities(grid, h):
    """What predict draws equals what the fit scores: class frequencies of many draws against
    the closed form, for a cell with a full neighbourhood and a corner cell."""
    cells, country, geo, months, values = grid
    knobs = {"weight": [1.0, 0.8, 0.4, 0.2], "memory": [3.0, 8.0, 40.0, 200.0],
             "length_scale": 1.7}
    cutoff = 150
    hist = months <= cutoff
    sampler = build_sampler(values[hist].astype(np.float32), months[hist], cutoff, knobs, geo,
                            build_shell_tables(geo))
    classes = classify(values)
    dc = build_doll_counts(classes[hist], months[hist], geo)
    memory = np.asarray(knobs["memory"])
    q = np.stack([doll_shares(dc, time_weights(months[hist], np.array([cutoff]), memory[k]),
                              k, knobs["length_scale"])[0] for k in range(len(DOLLS))])
    rng = np.random.default_rng(2)
    S = 200_000
    for i in (int(np.argmax(dc.sizes[NEAR])), int(np.argmin(dc.sizes[NEAR]))):
        p_doll = np.diff(np.r_[0.0, sampler.doll_cdf(np.array([i]), np.array([h]))[0, 0]])
        draws = sampler.draw(np.array([i]), np.array([h]), rng.random((1, 1, 3, S)))[0, 0]
        freq = np.bincount(classify(draws), minlength=N_CLASSES) / S
        np.testing.assert_allclose(freq, p_doll @ q[:, i, :], atol=4 / np.sqrt(S))


def test_forecast_opens_outward_with_distance(grid):
    _, _, geo, months, values = grid
    knobs = {"weight": [1.0, 0.8, 0.4, 0.2], "memory": [3.0, 8.0, 40.0, 200.0],
             "length_scale": 1.7}
    s = build_sampler(values.astype(np.float32), months, months[-1], knobs, geo,
                      build_shell_tables(geo))
    cdf = s.doll_cdf(np.array([100]), np.array([1, 6, 12, 36]))[0]
    p = np.diff(np.concatenate([np.zeros((4, 1)), cdf], axis=1), axis=1)
    assert np.all(np.diff(p[:, 0]) < 0), "the cell's share must shrink with distance"
    assert np.all(np.diff(p[:, MAP]) > 0), "the map's share must grow with distance"


# ── the model ───────────────────────────────────────────────────────────────────────────

PARTITION = {"test": (493, 540)}
TRAIN_END = 492
CONFIG = dict(n_samples=32, country_column="c_id", backtest_cutoffs=2, backtest_horizon=6,
              stability_refits=False, seed=7)


def planted_panel(first=300, last=540, seed=3):
    """A small grid where violence spreads to neighbours: a few moving hotspots."""
    cells, country = synthetic_grid(n_rows=10, n_cols=12, holes=(17, 50))
    r, c = (cells - 1) // N_COLS, (cells - 1) % N_COLS
    rng = np.random.default_rng(seed)
    months = np.arange(first, last + 1)
    centre = np.array([[104.0, 203.0], [107.0, 209.0]])
    values = np.zeros((months.size, cells.size))
    for s in range(months.size):
        centre += rng.normal(0, 0.4, centre.shape)
        d = np.min(np.hypot(r[None, :] - centre[:, :1], c[None, :] - centre[:, 1:]), axis=0)
        values[s] = rng.poisson(3.0 * np.exp(-d / 1.2) * (rng.random(cells.size) < 0.5))
    return cells, country, months, values


def to_df(cells, country, months, values, targets=("y",)):
    index = pd.MultiIndex.from_product([months, cells], names=["month_id", "priogrid_id"])
    df = pd.DataFrame(index=index)
    for t in targets:
        df[t] = values.ravel()
    df["c_id"] = np.tile(country, months.size).astype(float)
    return df


@pytest.fixture(scope="module")
def planted():
    cells, country, months, values = planted_panel()
    return cells, country, months, values, to_df(cells, country, months, values)


@pytest.fixture(scope="module")
def fitted(planted):
    *_, df = planted
    return MatryoshkaConflictology(targets=["y"], partition_dict=PARTITION, loa="pgm",
                                   **CONFIG).fit(df)


def test_backtest_cutoffs_are_spaced_and_end_a_horizon_before_train_end():
    np.testing.assert_array_equal(backtest_cutoffs(456, 4, 36), [384, 396, 408, 420])


def test_fit_beats_the_cell_only_special_case_on_a_planted_neighbour_effect(planted, fitted):
    cells, country, months, values, _ = planted
    k = fitted.knobs["y"]
    assert k["converged"], k["message"]
    assert len(k["weight"]) == len(k["memory"]) == len(DOLLS)
    hist = months <= TRAIN_END
    classes = classify(values[hist])
    dc = build_doll_counts(classes, months[hist], build_geometry(cells, country))
    cutoffs = backtest_cutoffs(TRAIN_END, 2, 6)
    obj = Objective(dc, cutoffs, outcome_classes(classes, months[hist], cutoffs, 6))
    cell_only = obj.score(np.array([1.0, 1e-6, 1e-6, 1e-6]), np.array([600.0] * 4), 1.0)[0]
    assert k["score"] > cell_only + 0.01
    assert k["weight"][NEAR] > 0.1, "the planted neighbour effect must be picked up"


def test_special_case_is_conflictology_with_a_moving_window(planted):
    """Only the cell, nothing fades, 36 months of history: each draw is uniform over the
    cell's last 36 values (RollingConflictology's pool at that origin)."""
    cells, country, months, values, df = planted
    m = MatryoshkaConflictology(targets=["y"], partition_dict=PARTITION, loa="pgm", **CONFIG)
    m.knobs = {"y": {"weight": [1.0, 0.0, 0.0, 0.0], "memory": [np.inf] * 4,
                     "length_scale": 1.0}}
    m.entity_ids, m.country = cells, country
    m.n_samples = 3600
    k = 4
    last = TRAIN_END + k
    window = (months > last - 36) & (months <= last)
    out = m.predict(df[df.index.get_level_values("month_id") > last - 36], k, 2)["y"]
    draws = out.values.reshape(cells.size, 2, -1)
    rng = np.random.default_rng(0)
    for i in rng.choice(cells.size, 5, replace=False):
        pool = values[window, i]
        got = np.sort(draws[i].ravel())
        assert set(np.unique(got)) <= set(np.unique(pool))
        expected = np.array([np.mean(pool == v) for v in np.unique(pool)])
        observed = np.array([np.mean(got == v) for v in np.unique(pool)])
        np.testing.assert_allclose(observed, expected, atol=4 / np.sqrt(got.size))


def test_predict_never_draws_a_month_after_the_history_end(planted, fitted):
    cells, country, months, values, _ = planted
    future = values.copy()
    future[months > TRAIN_END + 2] = 999_999.0
    df = to_df(cells, country, months, future)
    out = fitted.predict(df, 2, 6)["y"].values
    assert not np.any(out == 999_999.0)


def test_predict_output_grid_and_shape(planted, fitted):
    cells, *_, df = planted
    out = fitted.predict(df, 3, 4)["y"]
    assert out.values.shape == (cells.size * 4, CONFIG["n_samples"])
    np.testing.assert_array_equal(np.asarray(out.index.time)[:4], [496, 497, 498, 499])
    np.testing.assert_array_equal(np.asarray(out.index.unit)[:4], [cells[0]] * 4)


def test_same_seed_same_output_and_seed_matters(planted, fitted):
    *_, df = planted
    a = fitted.predict(df, 0, 3)["y"].values
    np.testing.assert_array_equal(a, fitted.predict(df, 0, 3)["y"].values)
    fitted.seed = 8
    try:
        assert not np.array_equal(a, fitted.predict(df, 0, 3)["y"].values)
    finally:
        fitted.seed = CONFIG["seed"]


def test_block_size_changes_nothing(planted, fitted, monkeypatch):
    """Drawing one cell at a time gives the same output as drawing blocks (ADR-011 order)."""
    from views_baseline.model.models.distributional.matryoshka import model as model_module

    *_, df = planted
    blocked = fitted.predict(df, 1, 3)["y"].values
    monkeypatch.setattr(model_module, "BLOCK_CELLS", 1)
    np.testing.assert_array_equal(fitted.predict(df, 1, 3)["y"].values, blocked)


def test_dataframe_and_featureframe_inputs_give_identical_output(planted, fitted):
    from views_baseline.model.frames.input import to_feature_frame

    *_, df = planted
    ff = to_feature_frame(df, loa="pgm", targets=["y", "c_id"])
    m = MatryoshkaConflictology(targets=["y"], partition_dict=PARTITION, loa="pgm", **CONFIG)
    m.fit(ff)
    assert m.knobs["y"]["weight"] == fitted.knobs["y"]["weight"]
    np.testing.assert_array_equal(m.predict(ff, 1, 2)["y"].values,
                                  fitted.predict(df, 1, 2)["y"].values)


def test_stability_refits_leave_each_cutoff_out(planted):
    *_, df = planted
    m = MatryoshkaConflictology(targets=["y"], partition_dict=PARTITION, loa="pgm",
                                **{**CONFIG, "stability_refits": True}).fit(df)
    refits = m.knobs["y"]["leave_one_cutoff_out"]
    assert [r["left_out"] for r in refits] == [474, 486]


# ── loud failures ───────────────────────────────────────────────────────────────────────


def _model(**overrides):
    return MatryoshkaConflictology(targets=["y"], partition_dict=PARTITION, loa="pgm",
                                   **{**CONFIG, **overrides})


def test_fails_loud_on_a_missing_cell_month(planted):
    *_, df = planted
    with pytest.raises(ValueError, match="complete panel"):
        _model().fit(df.drop(df.index[5]))


def test_fails_loud_on_nan(planted):
    *_, df = planted
    bad = df.copy()
    bad.iloc[7, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        _model().fit(bad)


def test_fails_loud_when_a_cell_changes_country(planted):
    *_, df = planted
    bad = df.copy()
    bad.iloc[0, bad.columns.get_loc("c_id")] = 12345.0
    with pytest.raises(ValueError, match="changes over time"):
        _model().fit(bad)


def test_fails_loud_on_too_little_history(planted):
    *_, df = planted
    short = df[df.index.get_level_values("month_id") >= 400]
    with pytest.raises(ValueError, match="at least 120 months"):
        _model().fit(short)


def test_fails_loud_at_country_level():
    with pytest.raises(ValueError, match="pgm"):
        MatryoshkaConflictology(targets=["y"], partition_dict=PARTITION, loa="cm",
                                **CONFIG).fit(pd.DataFrame())


def test_predict_fails_loud(planted, fitted):
    *_, df = planted
    with pytest.raises(ValueError, match="sequence_number"):
        fitted.predict(df, -1, 3)
    with pytest.raises(ValueError, match="fit"):
        _model().predict(df, 0, 3)
    with pytest.raises(ValueError, match="needs data up to"):
        fitted.predict(df[df.index.get_level_values("month_id") <= TRAIN_END], 2, 3)
    first = df.index.get_level_values("priogrid_id")[0]
    with pytest.raises(ValueError, match="cells"):
        fitted.predict(df[df.index.get_level_values("priogrid_id") != first], 0, 3)


def test_stability_refits_need_two_cutoffs(planted):
    *_, df = planted
    with pytest.raises(ValueError, match="at least 2"):
        _model(backtest_cutoffs=1, stability_refits=True).fit(df)


def test_country_column_must_not_be_a_target():
    with pytest.raises(ValueError, match="must not be a target"):
        MatryoshkaConflictology(targets=["c_id"], partition_dict=PARTITION, loa="pgm",
                                **CONFIG)
