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
    """Cells with sea holes, two countries split by column, and four cells without a country:
    two coded -1 (as in the real data) and two coded 0, each pair far apart, so that a bug
    treating either code as a country gives that "country" a non-empty shell and shows."""
    rows, cols = np.meshgrid(np.arange(100, 100 + n_rows), np.arange(200, 200 + n_cols),
                             indexing="ij")
    cells = (rows * N_COLS + cols + 1).ravel()
    cells = np.sort(cells[~np.isin(cells, cells[list(holes)])])
    c = (cells - 1) % N_COLS
    country = np.where(c < 200 + n_cols // 2, 7, 9)
    country[[3, cells.size - 4]] = -1
    country[[20, cells.size // 2]] = 0
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


def test_row_lookup_never_overflows_onto_a_zero_probability_entry():
    """Code review of 702f17d: with rows shifted apart by 2·row, u + shift can round up onto a
    row's final 1.0 for the largest u that rng.random returns. The last offset here has weight
    0, so the search must stop at index 3, in every row of a 256-row block."""
    from views_baseline.model.models.distributional.matryoshka.draw import (
        _cdf,
        _cdf_on_grid,
        _row_searchsorted,
        _uniforms_on_grid,
    )

    table = _cdf_on_grid(_cdf(np.tile([1.0, 1.0, 1.0, 1.0, 0.0], (256, 1))))
    u = _uniforms_on_grid(np.full(256, np.nextafter(1.0, 0.0)))
    got = _row_searchsorted(table, np.arange(256), u, np.full(256, 4))
    np.testing.assert_array_equal(got, 3)


@pytest.mark.parametrize("rows", [[0, 600, 1200, 1999], [1999, 0, 1200, 600]],
                         ids=["ascending", "unordered"])
def test_row_search_answers_a_chunk_with_a_single_lookup(rows):
    """Code review of 84fd824: with one lookup per chunk of rows (CDF tables are searched about
    500 rows at a time), every lookup must still be answered, sorted or not."""
    from views_baseline.model.models.distributional.matryoshka.draw import (
        _cdf,
        _cdf_on_grid,
        _row_searchsorted,
        _uniforms_on_grid,
    )

    cdf = _cdf(np.random.default_rng(3).random((2000, 6)))
    rows = np.array(rows)
    u = np.array([0.1, 0.5, 0.7, 0.95])
    expected = [np.searchsorted(cdf[r], x, side="right") for r, x in zip(rows, u)]
    got = _row_searchsorted(_cdf_on_grid(cdf), rows, _uniforms_on_grid(u), np.full(2000, 5))
    np.testing.assert_array_equal(got, expected)


def test_row_search_is_exact_and_ignores_where_a_row_sits():
    """Code review #147, finding 4: the pick for a row must equal a plain per-row search of
    u ≥ cdf, for uniforms on rng.random's grid, including u exactly on a CDF entry, whatever
    the row's position and however many rows are searched together."""
    from views_baseline.model.models.distributional.matryoshka.draw import (
        UNIFORM_STEPS,
        _cdf,
        _cdf_on_grid,
        _row_searchsorted,
        _uniforms_on_grid,
    )

    rng = np.random.default_rng(11)
    R, W = 70_000, 9
    cdf = _cdf(rng.random((R, W)) * (rng.random((R, W)) < 0.7))
    rows = rng.integers(0, R, 50_000)
    u = rng.random(rows.size)
    on_entry = rng.random(rows.size) < 0.3       # some u exactly on a CDF entry
    entry = cdf[rows, rng.integers(0, W, rows.size)]
    u[on_entry] = np.floor(entry[on_entry] * UNIFORM_STEPS) / UNIFORM_STEPS
    u = np.minimum(u, np.nextafter(1.0, 0.0))
    expected = np.array([np.searchsorted(cdf[r], x, side="right") for r, x in zip(rows, u)])
    last = W - 1 - np.argmax((cdf > 0)[:, ::-1], axis=1)
    expected = np.minimum(expected, last[rows])
    table = _cdf_on_grid(cdf)
    np.testing.assert_array_equal(
        _row_searchsorted(table, rows, _uniforms_on_grid(u), last), expected)
    # The same rows moved to the far end of a much larger table give the same answers.
    big = np.vstack([np.zeros((3 * R, W), dtype=np.int64), table])
    np.testing.assert_array_equal(
        _row_searchsorted(big, rows + 3 * R, _uniforms_on_grid(u), np.r_[np.zeros(3 * R, int),
                                                                         last]), expected)


def test_a_panel_where_no_cell_has_a_country_still_works():
    cells, _ = synthetic_grid(n_rows=6, n_cols=6, holes=())
    country = np.zeros(cells.size, dtype=int)
    geo = build_geometry(cells, country)
    months = np.arange(100, 130)
    values = np.random.default_rng(4).choice(VALUES, size=(months.size, cells.size))
    _, sizes = shell_sizes(geo)
    assert (sizes[COUNTRY] == 0).all()
    dc = build_doll_counts(classify(values), months, geo)
    q = doll_shares(dc, time_weights(months, np.array([120]), 10.0), MAP)
    # On a 6 × 6 grid the central cells have every other cell within 5 cells: empty map shell.
    np.testing.assert_allclose(q.sum(axis=2)[0], (sizes[MAP] > 0).astype(float))
    assert (sizes[MAP] > 0).any()
    knobs = {"weight": [1.0, 0.8, 0.4, 0.2], "memory": [3.0, 8.0, 40.0, 200.0],
             "length_scale": 1.7}
    s = build_sampler(values.astype(np.float32), months, months[-1], knobs, geo,
                      build_shell_tables(geo))
    draws = s.draw(np.arange(cells.size), np.array([1, 5]),
                   np.random.default_rng(5).random((cells.size, 2, 3, 50)))
    assert set(np.unique(draws)) <= set(VALUES)


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
    m.history_start = TRAIN_END + 4 - 35            # the 36-month history passed below
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
    monkeypatch.setattr(model_module, "BLOCK_UNIFORMS", 1)           # one cell per block
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


def test_fails_loud_on_a_negative_value(planted):
    *_, df = planted
    bad = df.copy()
    bad.iloc[9, 0] = -1.0
    with pytest.raises(ValueError, match="negative"):
        _model().fit(bad)


def test_classify_rejects_negative_values():
    with pytest.raises(ValueError, match="negative"):
        classify(np.array([0.0, -0.5]))


def test_predict_fails_loud_when_the_history_starts_elsewhere(planted, fitted):
    *_, df = planted
    later = df[df.index.get_level_values("month_id") > 310]
    with pytest.raises(ValueError, match="starts at month 311"):
        fitted.predict(later, 0, 3)


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


# ── guard-audit follow-ups (falsify guard audit of 702f17d) ─────────────────────────────


def _encoded(months, n_cells):
    """Values that say where they came from: month index × n_cells + cell index (float32-exact)."""
    return (np.arange(months.size)[:, None] * n_cells + np.arange(n_cells)[None, :]).astype(
        np.float32)


@pytest.mark.parametrize("doll", range(len(DOLLS)))
def test_each_doll_draws_exactly_from_its_shell_with_the_right_cell_weights(grid, doll):
    """With all weight on one doll and values that encode their cell, every draw comes from
    that doll's brute-force shell, in the right proportions (kernel for the neighbourhood,
    uniform for country and map). Checked for a full cell, a corner and a no-country cell."""
    cells, country, geo, months, _ = grid
    length = 1.7
    shells, pick = brute_force_shells(cells, country, length)
    weight = np.full(len(DOLLS), 1e-300)
    weight[doll] = 1.0
    knobs = {"weight": weight, "memory": [10.0] * 4, "length_scale": length}
    values = _encoded(months, cells.size)
    s = build_sampler(values, months, months[-1], knobs, geo, build_shell_tables(geo))
    S = 40_000
    no_country = int(np.nonzero(country <= 0)[0][0])
    second_country = int(np.nonzero(country == 9)[0][-1])     # its block does not start at 0
    for i in {int(np.argmax(shells[NEAR].sum(1))), 0, no_country, second_country}:
        if not shells[doll][i].any():
            continue
        got = s.draw(np.array([i]), np.array([1]), np.random.default_rng(i).random(
            (1, 1, 3, S)))[0, 0].astype(np.int64) % cells.size
        assert shells[doll][i][got].all(), f"{DOLLS[doll]}: a draw outside the shell (cell {i})"
        freq = np.bincount(got, minlength=cells.size) / S
        np.testing.assert_allclose(freq, pick[doll][i], atol=5 / np.sqrt(S))


def test_a_cell_without_a_country_never_picks_the_country_doll(grid):
    cells, country, geo, months, values = grid
    knobs = {"weight": [1.0, 0.8, 0.4, 0.2], "memory": [3.0, 8.0, 40.0, 200.0],
             "length_scale": 1.7}
    s = build_sampler(values.astype(np.float32), months, months[-1], knobs, geo,
                      build_shell_tables(geo))
    for i in np.nonzero(country <= 0)[0]:
        p = np.diff(np.r_[0.0, s.doll_cdf(np.array([i]), np.array([1]))[0, 0]])
        assert p[COUNTRY] == 0.0


def test_cdf_ends_exactly_at_one_from_the_last_positive_entry():
    from views_baseline.model.models.distributional.matryoshka.draw import _cdf

    c = _cdf(np.array([[0.1, 0.2, 0.3, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]))
    np.testing.assert_array_equal(c[0, 2:], 1.0)
    np.testing.assert_allclose(c[0, :2], [1 / 6, 0.5])
    np.testing.assert_array_equal(c[1], 0.0)


def test_predict_uses_history_end_c_and_months_ahead_h(planted):
    """Values encode month and cell. Per forecast step j: the share of draws from the cell
    itself equals P(cell | h = j + 1), and the cell doll's months follow exp(−(c − s)/m) up to
    and including c = train_end + k."""
    cells, country, months, _, _ = planted
    k, steps = 5, 3
    last = TRAIN_END + k
    hist = months <= last
    values = np.zeros((months.size, cells.size), dtype=np.float32)
    values[hist] = _encoded(months[hist], cells.size)
    df = to_df(cells, country, months, values)
    m = _model(n_samples=20_000)
    memory = [4.0, 30.0, 30.0, 30.0]
    m.knobs = {"y": {"weight": [1.0, 2.0, 0.0, 0.0], "memory": memory, "length_scale": 1.0}}
    m.entity_ids, m.country, m.history_start = cells, country, int(months[0])
    out = m.predict(df, k, steps)["y"].values.reshape(cells.size, steps, -1).astype(np.int64)
    i = 40
    month_of, cell_of = out[i] // cells.size, out[i] % cells.size
    for j in range(steps):
        h = j + 1
        a = np.array([1.0, 2.0]) * np.exp(-h / np.array(memory[:2]))
        own = cell_of[j] == i
        np.testing.assert_allclose(own.mean(), a[0] / a.sum(), atol=0.015)
        age = np.arange(hist.sum())[::-1]                       # c − s for s = first..c
        w = np.exp(-age / memory[0])
        expected = w / w.sum()
        freq = np.bincount(month_of[j][own], minlength=hist.sum()) / own.sum()
        np.testing.assert_allclose(freq, expected, atol=0.02)
        assert month_of[j].max() == hist.sum() - 1              # month c itself is drawn


def test_rng_order_is_entity_then_time_then_target(planted):
    """predict equals one sampler call per (cell, month, target), in that order, each taking
    3 × n_samples uniforms from one fresh generator (ADR-011)."""
    cells, country, months, values, _ = planted
    df = to_df(cells, country, months, values, targets=("y", "z"))
    df["z"] = df["z"] * 2 + 1
    m = _model(n_samples=8)
    m.targets = ["y", "z"]
    m.knobs = {t: {"weight": [1.0, 0.8, 0.4, 0.2], "memory": [3.0, 8.0, 40.0, 200.0],
                   "length_scale": 1.7} for t in m.targets}
    m.entity_ids, m.country, m.history_start = cells, country, int(months[0])
    k, steps = 2, 3
    out = m.predict(df, k, steps)
    last = TRAIN_END + k
    hist = months <= last
    geo = build_geometry(cells, country)
    tables = build_shell_tables(geo)
    panel = {"y": values[hist], "z": values[hist] * 2 + 1}
    samplers = {t: build_sampler(panel[t].astype(np.float32), months[hist], last, m.knobs[t],
                                 geo, tables) for t in m.targets}
    rng = np.random.default_rng(m.seed)
    for i in range(cells.size):
        for j in range(steps):
            for t in m.targets:
                expected = samplers[t].draw(np.array([i]), np.array([j + 1]),
                                            rng.random((1, 1, 3, 8)))[0, 0]
                np.testing.assert_array_equal(out[t].values[i * steps + j], expected)


@pytest.mark.parametrize("start, ok", [(355, True), (356, False)])
def test_the_120_month_history_boundary(planted, start, ok):
    """Cutoffs 474 and 486: history from 355 is exactly 120 months before 474."""
    *_, df = planted
    part = df[df.index.get_level_values("month_id") >= start]
    if ok:
        assert _model().fit(part).knobs["y"]["converged"] is not None
    else:
        with pytest.raises(ValueError, match="at least 120 months"):
            _model().fit(part)


def test_stability_refits_are_searches_without_the_left_out_cutoff(planted):
    from views_baseline.model.models.distributional.matryoshka.knob_fit import search

    cells, country, months, values, df = planted
    m = _model(stability_refits=True).fit(df)
    hist = months <= TRAIN_END
    classes = classify(values[hist])
    dc = build_doll_counts(classes, months[hist], build_geometry(cells, country))
    cutoffs = backtest_cutoffs(TRAIN_END, 2, 6)
    outcomes = outcome_classes(classes, months[hist], cutoffs, 6)
    for refit, c in zip(m.knobs["y"]["leave_one_cutoff_out"], cutoffs):
        keep = cutoffs != c
        direct = search(dc, cutoffs[keep], outcomes[keep])
        np.testing.assert_allclose(refit["weight"], direct["weight"])
        assert refit["score"] == pytest.approx(direct["score"])
        assert refit["score"] != pytest.approx(m.knobs["y"]["score"])


def test_outcomes_are_the_months_after_each_cutoff():
    months = np.arange(10, 30)
    classes = np.repeat(months[:, None], 2, axis=1)          # class array encodes the month
    out = outcome_classes(classes, months, np.array([15, 20]), 3)
    np.testing.assert_array_equal(out[:, :, 0], [[16, 17, 18], [21, 22, 23]])


def test_log_score_equals_the_samplers_mix_on_a_grid_with_empty_shells():
    """The fitting score uses exactly the doll probabilities the sampler draws with, including
    excluded empty shells (a 6 × 6 grid: central cells have no map shell)."""
    from views_baseline.model.models.distributional.matryoshka.forecast_probability import (
        log_score,
    )

    cells, country = synthetic_grid(n_rows=6, n_cols=6, holes=())
    geo = build_geometry(cells, country)
    months = np.arange(100, 140)
    values = np.random.default_rng(6).choice(VALUES, size=(months.size, cells.size))
    classes = classify(values)
    dc = build_doll_counts(classes, months, geo)
    assert (dc.sizes[MAP] == 0).any() and (dc.sizes[MAP] > 0).any()
    weight, memory, length = np.array([1.0, 0.8, 0.4, 0.2]), np.array([3., 8., 40., 200.]), 1.7
    cutoff, H = 130, 4
    q = np.stack([doll_shares(dc, time_weights(months, np.array([cutoff]), memory[k]), k,
                              length) for k in range(len(DOLLS))])           # (4, 1, N, K)
    obs = outcome_classes(classes, months, np.array([cutoff]), H)
    score, _ = log_score(q, dc.sizes, obs, weight, memory)
    s = build_sampler(values[months <= cutoff].astype(np.float32), months[months <= cutoff],
                      cutoff, {"weight": weight, "memory": memory, "length_scale": length},
                      geo, build_shell_tables(geo))
    p_doll = np.diff(np.concatenate(
        [np.zeros((cells.size, H, 1)), s.doll_cdf(np.arange(cells.size), np.arange(1, H + 1))],
        axis=2), axis=2)                                                     # (N, H, 4)
    p = np.array([[p_doll[n, h] @ q[:, 0, n, obs[0, h, n]] for h in range(H)]
                  for n in range(cells.size)])
    assert score == pytest.approx(np.log(p).mean(), rel=1e-12)


def test_geometry_wraps_columns_at_the_date_line():
    """Col 719 is col 0's neighbour at offset (0, −1), and col 0 is col 719's at (0, +1).
    Checked by offset: without the wrap, col 0's offset (+1, −1) aliases onto the col-719 id."""
    from views_baseline.model.models.distributional.matryoshka.geometry import (
        OFFSET_DCOL,
        OFFSET_DROW,
    )

    cells = np.array([100 * N_COLS + 1, 100 * N_COLS + N_COLS])      # cols 0 and 719, row 100
    geo = build_geometry(cells, np.array([1, 1]))
    left = np.nonzero((OFFSET_DROW == 0) & (OFFSET_DCOL == -1))[0][0]
    right = np.nonzero((OFFSET_DROW == 0) & (OFFSET_DCOL == 1))[0][0]
    assert geo.neighbours[0, left] == 1 and geo.neighbours[1, right] == 0
    assert (geo.neighbours >= 0).sum() == 2


def test_geometry_rejects_unsorted_cells():
    with pytest.raises(ValueError, match="ascending"):
        build_geometry(np.array([5, 3]), np.array([1, 1]))


def test_fails_loud_on_duplicate_cell_months(planted):
    *_, df = planted
    with pytest.raises(ValueError, match="1 duplicated"):
        _model().fit(pd.concat([df, df.iloc[[0]]]))


def test_fails_loud_on_non_consecutive_months(planted):
    *_, df = planted
    gap = df[df.index.get_level_values("month_id") != 400]
    with pytest.raises(ValueError, match="not consecutive"):
        _model().fit(gap)


def test_fails_loud_on_a_non_integer_country_code(planted):
    *_, df = planted
    bad = df.copy()
    bad["c_id"] = bad["c_id"] + 0.5
    with pytest.raises(ValueError, match="integer country codes"):
        _model().fit(bad)


def test_fails_loud_when_the_data_ends_before_train_end(planted):
    *_, df = planted
    with pytest.raises(ValueError, match="before train_end"):
        _model().fit(df[df.index.get_level_values("month_id") <= TRAIN_END - 1])


def test_nan_in_the_predict_history_is_named(planted, fitted):
    *_, df = planted
    bad = df.copy()
    row = np.nonzero(bad.index.get_level_values("month_id") == TRAIN_END + 1)[0][0]
    bad.iloc[row, 0] = np.nan
    with pytest.raises(ValueError, match="NaN in 'y'"):
        fitted.predict(bad, 2, 3)


def test_fit_names_nan_before_classifying(planted):
    *_, df = planted
    bad = df.copy()
    bad.iloc[7, 0] = np.nan
    with pytest.raises(ValueError, match="NaN in 'y'"):
        _model().fit(bad)


def test_predict_fails_loud_on_same_count_but_different_cells(planted, fitted):
    cells, *_, df = planted
    ids = df.index.get_level_values("priogrid_id").to_numpy().copy()
    ids[ids == cells[-1]] = cells[-1] + 1                 # a cell not in the fit, same count
    moved = df.copy()
    moved.index = pd.MultiIndex.from_arrays(
        [df.index.get_level_values("month_id"), ids], names=df.index.names)
    with pytest.raises(ValueError, match="cells it was fitted on"):
        fitted.predict(moved.sort_index(), 0, 3)


def test_near_last_is_each_cells_farthest_present_neighbour(grid):
    cells, country, geo, months, values = grid
    knobs = {"weight": [1.0, 0.8, 0.4, 0.2], "memory": [3.0, 8.0, 40.0, 200.0],
             "length_scale": 1.7}
    s = build_sampler(values.astype(np.float32), months, months[-1], knobs, geo,
                      build_shell_tables(geo))
    present = geo.neighbours >= 0
    expected = np.array([np.nonzero(r)[0][-1] if r.any() else 0 for r in present])
    np.testing.assert_array_equal(s.near_last, expected)


def test_predict_rejects_a_negative_value_in_its_history(planted, fitted):
    *_, df = planted
    bad = df.copy()
    row = np.nonzero(bad.index.get_level_values("month_id") == TRAIN_END + 1)[0][0]
    bad.iloc[row, 0] = -5.0
    with pytest.raises(ValueError, match="negative value in target 'y'"):
        fitted.predict(bad, 2, 3)


def test_start_score_is_the_score_at_the_starting_values(planted, fitted):
    from views_baseline.model.models.distributional.matryoshka.knob_fit import (
        START_LENGTH,
        START_MEMORY,
        START_WEIGHT,
    )

    cells, country, months, values, _ = planted
    hist = months <= TRAIN_END
    classes = classify(values[hist])
    dc = build_doll_counts(classes, months[hist], build_geometry(cells, country))
    cutoffs = backtest_cutoffs(TRAIN_END, 2, 6)
    obj = Objective(dc, cutoffs, outcome_classes(classes, months[hist], cutoffs, 6))
    expected = obj.score(START_WEIGHT, START_MEMORY, START_LENGTH)[0]
    assert fitted.knobs["y"]["start_score"] == pytest.approx(expected, rel=1e-12)
    assert fitted.knobs["y"]["start_score"] < fitted.knobs["y"]["score"]


# ── code review #147 ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("setting, value", [
    ("n_samples", 0), ("n_samples", True), ("n_samples", 2.0),
    ("backtest_cutoffs", 0), ("backtest_cutoffs", 2.0), ("backtest_cutoffs", True),
    ("backtest_horizon", 6.0), ("seed", -1), ("seed", 1.5),
    ("stability_refits", "False"), ("stability_refits", 0), ("country_column", 3),
])
def test_settings_are_type_checked_at_construction(setting, value):
    with pytest.raises(ValueError, match=setting):
        _model(**{setting: value})


def test_targets_must_not_be_empty():
    with pytest.raises(ValueError, match="targets must not be empty"):
        MatryoshkaConflictology(targets=[], partition_dict=PARTITION, loa="pgm", **CONFIG)


def test_predict_rejects_an_empty_forecast(planted, fitted):
    *_, df = planted
    with pytest.raises(ValueError, match="output_length must be >= 1"):
        fitted.predict(df, 0, 0)


@pytest.mark.parametrize("bad, message", [(1.5, "non-integer value"),
                                          (np.inf, "infinite value")])
def test_fails_loud_on_a_target_that_is_not_a_count(planted, bad, message):
    *_, df = planted
    broken = df.copy()
    broken.iloc[7, 0] = bad
    with pytest.raises(ValueError, match=f"{message} in target 'y'"):
        _model().fit(broken)


@pytest.mark.parametrize("bad, message", [(np.inf, "infinite"), (-np.inf, "infinite"),
                                          (2.5, "non-integer")])
def test_classify_rejects_values_that_are_not_counts(bad, message):
    with pytest.raises(ValueError, match=message):
        classify(np.array([0.0, 3.0, bad]))


def test_fit_logs_the_knobs_and_warns_when_the_search_stops_early(planted, monkeypatch, caplog):
    import logging

    from views_baseline.model.models.distributional.matryoshka import knob_fit

    *_, df = planted
    monkeypatch.setitem(knob_fit.SEARCH_OPTIONS, "maxiter", 1)
    with caplog.at_level(logging.INFO):
        m = _model().fit(df)
    assert not m.knobs["y"]["converged"]
    messages = [(r.levelno, r.getMessage()) for r in caplog.records]
    assert any(lv == logging.INFO and "MatryoshkaConflictology y: weight" in s
               for lv, s in messages)
    assert any(lv == logging.WARNING and "did not converge" in s for lv, s in messages)


def test_fit_warns_when_probabilities_hit_the_floor(caplog):
    import logging

    from views_baseline.model.models.distributional.matryoshka.model import _log_fit

    knobs = {"weight": [1.0] * 4, "memory": [1.0] * 4, "length_scale": 1.0, "score": -1.0,
             "start_score": -2.0, "steps": 3, "seconds": 1.0, "converged": True,
             "message": "ok", "floored": 5}
    with caplog.at_level(logging.INFO):
        _log_fit("y", knobs)
    assert any(r.levelno == logging.WARNING and "5 backtest probabilities fell below the floor"
               in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        _log_fit("y", {**knobs, "floored": 1})
    assert any("1 backtest probabilities fell below" in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        _log_fit("y", {**knobs, "floored": 0})
    assert not any(r.levelno == logging.WARNING for r in caplog.records)
    caplog.clear()
    refits = [{**knobs, "left_out": 474, "floored": 2},
              {**knobs, "left_out": 486, "floored": 0, "converged": False, "message": "limit"}]
    with caplog.at_level(logging.INFO):
        _log_fit("y", {**knobs, "floored": 0, "leave_one_cutoff_out": refits})
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert "stability refit without cutoff 474, 2 backtest probabilities fell below" in \
        warnings[0]
    assert "stability refit without cutoff 486 did not converge" in warnings[1]


Z_OFFSET = 1_000_000          # float32-exact with the encoded values added


def test_each_target_draws_with_its_own_knobs(planted):
    """Code review #147, finding 7. Target z has cell-only knobs, so its draws for a cell come
    from that cell's own history; target y puts its weight on the neighbourhood. Values encode
    their cell, so a draw shows which cell it came from. Using one target's knobs for the other
    breaks one of the two assertions."""
    cells, country, months, _, _ = planted
    values = _encoded(months, cells.size)
    df = to_df(cells, country, months, values, targets=("y", "z"))
    df["z"] += Z_OFFSET
    m = MatryoshkaConflictology(targets=["y", "z"], partition_dict=PARTITION, loa="pgm",
                                **CONFIG)
    m.knobs = {"y": {"weight": [1e-6, 1.0, 1e-6, 1e-6], "memory": [12.0] * 4,
                     "length_scale": 2.0},
               "z": {"weight": [1.0, 0.0, 0.0, 0.0], "memory": [12.0] * 4,
                     "length_scale": 2.0}}
    m.entity_ids, m.country, m.history_start = cells, country, int(months[0])
    out = m.predict(df[df.index.get_level_values("month_id") <= TRAIN_END], 0, 2)
    source = {t: (out[t].values.reshape(cells.size, -1).astype(np.int64) % Z_OFFSET)
              % cells.size for t in ("y", "z")}
    own = np.arange(cells.size)[:, None]
    assert (source["z"] == own).all(), "z (cell only) drew from another cell"
    assert (source["y"] != own).mean() > 0.9, "y (neighbourhood) drew mostly from its own cell"
    # Each target draws from its own values: z's are offset, y's are not.
    assert (out["z"].values >= Z_OFFSET).all() and (out["y"].values < Z_OFFSET).all()


def test_predict_names_the_cells_that_were_not_fitted(planted, fitted):
    cells, country, months, values, _ = planted
    extra = np.r_[cells, cells[-1] + 1]
    df = to_df(extra, np.r_[country, country[-1]], months,
               np.hstack([values, np.zeros((months.size, 1))]))
    with pytest.raises(ValueError, match=rf"Cells not fitted on: \[{extra[-1]}\]"):
        fitted.predict(df, 0, 2)


# ── review of 62bd614 ───────────────────────────────────────────────────────────────────


GOOD_KNOBS = {"weight": [1.0, 0.8, 0.4, 0.2], "memory": [3.0, 8.0, 40.0, 200.0],
              "length_scale": 1.7}


def _with(name, doll, value):
    out = list(GOOD_KNOBS[name])
    out[doll] = value
    return {name: out}


@pytest.mark.parametrize("change", [
    *[_with("weight", d, -0.1) for d in range(4)],
    *[_with("weight", d, np.inf) for d in range(4)],
    *[_with("memory", d, 0.0) for d in range(4)],
    _with("memory", 2, np.nan),
    {"weight": [1.0]}, {"memory": [3.0, 8.0, 40.0]},
    {"length_scale": 0.0}, {"length_scale": np.nan},
], ids=lambda c: str(c))
def test_sampler_rejects_knobs_out_of_range(grid, change):
    """Hand-set knobs: every doll's weight and memory is checked, and their number."""
    cells, country, geo, months, values = grid
    with pytest.raises(ValueError, match="knobs out of range"):
        build_sampler(values.astype(np.float32), months, months[-1], {**GOOD_KNOBS, **change},
                      geo, build_shell_tables(geo))


def test_sampler_accepts_an_infinite_memory_and_a_zero_cell_weight(grid):
    cells, country, geo, months, values = grid
    knobs = {**GOOD_KNOBS, "weight": [0.0, 0.8, 0.4, 0.2], "memory": [np.inf] * 4}
    s = build_sampler(values.astype(np.float32), months, months[-1], knobs, geo,
                      build_shell_tables(geo))
    s.draw(np.arange(5), np.array([1, 3]), np.random.default_rng(0).random((5, 2, 3, 8)))


@pytest.mark.parametrize("change, h", [
    ({"memory": [0.01] * 4}, 100),                         # every weight fades to exactly 0
    ({"weight": [0.0] * 4}, 1),
    ({"weight": [1.0, 1e308, 1e308, 1e308]}, 1),           # the sum overflows
])
def test_draw_fails_loud_when_no_doll_can_be_drawn(grid, change, h):
    """Code review of 84fd824: an all-zero or undefined doll row used to fall through to the
    map silently (or crash on an empty map shell)."""
    cells, country, geo, months, values = grid
    s = build_sampler(values.astype(np.float32), months, months[-1], {**GOOD_KNOBS, **change},
                      geo, build_shell_tables(geo))
    with pytest.raises(ValueError, match="no doll can be drawn for cell index 0"):
        s.draw(np.arange(3), np.array([h]), np.random.default_rng(0).random((3, 1, 3, 4)))


def test_fails_loud_on_an_infinite_country_code(planted):
    *_, df = planted
    broken = df.copy()
    broken["c_id"] = broken["c_id"].where(broken.index.get_level_values("priogrid_id")
                                          != broken.index.get_level_values("priogrid_id")[0],
                                          np.inf)
    with pytest.raises(ValueError, match="integer country codes"):
        _model().fit(broken)


def test_a_numpy_bool_is_accepted_for_stability_refits():
    assert _model(stability_refits=np.False_).stability_refits is False
