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
    ALL_ON,
    COUNTRY,
    DOLLS,
    MAP,
    NEAR,
    build_doll_counts,
    doll_mask,
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
    START_LENGTH,
    START_MEMORY,
    START_WEIGHT,
    Objective,
    backtest_cutoffs,
    free_knobs,
    search,
    start_theta,
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


def brute_force_shells(cells, country, length, on=ALL_ON):
    """(4, N, N) shell membership and (4, N, N) P(cell j | doll, cell i), from pairwise
    distances, independently of the package's offset table.

    An off doll's shell is empty and its cells move outward to the next doll that is on (#153):
    without the neighbourhood the inner set is the cell alone; without the country, the map holds
    it; without the map, its cells are in no shell."""
    r, c = (cells - 1) // N_COLS, (cells - 1) % N_COLS
    dist = np.hypot(r[:, None] - r[None, :], c[:, None] - c[None, :])
    near = (dist > 0) & (dist <= NEAR_RADIUS) & on[NEAR]
    same_c = (country[:, None] == country[None, :]) & (country[:, None] > 0)
    inner = (dist == 0) | near
    country_shell = same_c & ~inner & on[COUNTRY]
    map_shell = ~inner & ~(same_c & on[COUNTRY]) & on[MAP]
    shells = np.stack([dist == 0, near, country_shell, map_shell])
    pick = np.stack([shells[0], np.where(near, np.exp(-dist / length), 0.0), shells[2],
                     shells[3]]).astype(float)
    pick = pick / np.maximum(pick.sum(axis=2, keepdims=True), 1e-300)
    return shells, pick


#: Every on-mask with the cell on: the subsets of dolls a model can use (#153).
SUBSETS = [np.array([True, n, c, m]) for n in (False, True) for c in (False, True)
           for m in (False, True)]
SUBSET_IDS = ["+".join(d for d, o in zip(DOLLS, on) if o) for on in SUBSETS]


@pytest.fixture(scope="module")
def grid():
    cells, country = synthetic_grid()
    geo = build_geometry(cells, country)
    months = np.arange(100, 160)
    values = np.random.default_rng(0).choice(VALUES, size=(months.size, cells.size))
    return cells, country, geo, months, values


# ── ported prototype checks ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("on", SUBSETS, ids=SUBSET_IDS)
def test_shells_are_disjoint_cover_every_cell_and_have_the_counted_sizes(grid, on):
    """With the map on, the shells cover every cell; with it off, exactly the cells the map
    would hold are in no shell."""
    cells, country, geo, _, _ = grid
    shells, _ = brute_force_shells(cells, country, 1.0, on)
    covered = shells.sum(axis=0)
    assert (covered <= 1).all(), "shells must be disjoint"
    with_map, _ = brute_force_shells(cells, country, 1.0, on | np.array([0, 0, 0, 1], bool))
    np.testing.assert_array_equal(covered == 0, ~on[MAP] & with_map[MAP])
    assert (covered == 1).all() or not on[MAP]
    _, sizes = shell_sizes(geo, on)
    np.testing.assert_array_equal(shells.sum(axis=2), sizes)


def test_the_brute_force_shells_move_an_off_dolls_cells_outward(grid):
    """Pins the reference itself: the cells of an off doll join the next doll that is on."""
    cells, country, _, _, _ = grid
    full, _ = brute_force_shells(cells, country, 1.0)
    no_near, _ = brute_force_shells(cells, country, 1.0, np.array([1, 0, 1, 1], bool))
    no_country, _ = brute_force_shells(cells, country, 1.0, np.array([1, 1, 0, 1], bool))
    same_c = (country[:, None] == country[None, :]) & (country[:, None] > 0)
    np.testing.assert_array_equal(no_near[COUNTRY], full[COUNTRY] | (full[NEAR] & same_c))
    np.testing.assert_array_equal(no_near[MAP], full[MAP] | (full[NEAR] & ~same_c))
    np.testing.assert_array_equal(no_country[MAP], full[MAP] | full[COUNTRY])
    assert full[NEAR].any() and (full[NEAR] & same_c).any() and (full[NEAR] & ~same_c).any()


@pytest.mark.parametrize("on", SUBSETS, ids=SUBSET_IDS)
def test_doll_shares_equal_a_direct_kernel_weighted_average(grid, on):
    """An off doll's shares are all 0 (its shell is empty)."""
    cells, country, geo, months, values = grid
    length = 1.7
    _, pick = brute_force_shells(cells, country, length, on)
    classes = classify(values)
    dc = build_doll_counts(classes, months, geo, on)
    W = np.stack([time_weights(months, np.array([150]), m) for m in (3.0, 8.0, 40.0, 200.0)])
    onehot = (classes[..., None] == np.arange(N_CLASSES)).astype(float)
    for k in range(len(DOLLS)):
        q = doll_shares(dc, W[k], k, length)[0]
        direct = np.einsum("m,ij,mjk->ik", W[k, 0], pick[k], onehot)
        direct[:, 0] = np.where(pick[k].sum(axis=1) > 0, direct[:, 0], 0.0)
        np.testing.assert_allclose(q, direct, atol=1e-10, err_msg=DOLLS[k])


@pytest.mark.parametrize("on", [np.array([0, 1, 1, 1], bool), np.ones(3, bool),
                                np.array([1, 1, 1, 1])])
def test_the_doll_mask_needs_four_booleans_with_the_cell_on(on):
    with pytest.raises(ValueError, match="doll"):
        doll_mask(on)


@pytest.mark.parametrize("on", SUBSETS, ids=SUBSET_IDS)
def test_exact_probabilities_match_an_independent_three_step_draw(grid, on):
    """The closed form against ADR-024's draw, sampled from the brute-force shells."""
    cells, country, geo, months, values = grid
    length, h, S = 1.7, 5, 200_000
    weight, memory = np.array([1.0, 0.8, 0.4, 0.2]), np.array([3.0, 8.0, 40.0, 200.0])
    shells, pick = brute_force_shells(cells, country, length, on)
    classes = classify(values)
    dc = build_doll_counts(classes, months, geo, on)
    cutoff = 150
    q = np.stack([doll_shares(dc, time_weights(months, np.array([cutoff]), memory[k]), k,
                              length)[0] for k in range(len(DOLLS))])        # (4, N, K)
    rng = np.random.default_rng(1)
    hist = months[months <= cutoff]
    near_sizes = shell_sizes(geo, ALL_ON)[1][NEAR]          # cells picked by geometry, not mask
    for i in (int(np.argmax(near_sizes)), int(np.argmin(near_sizes))):
        a = np.array([weight[k] * np.exp(-h / memory[k]) if shells[k][i].any() else 0.0
                      for k in range(len(DOLLS))])
        dolls = rng.choice(len(DOLLS), size=S, p=a / a.sum())
        draws = np.empty(S, np.int64)
        for k in range(len(DOLLS)):
            sel = dolls == k
            if not sel.any():                                  # an off or empty doll
                continue
            w = np.exp(-(cutoff - hist) / memory[k])
            ms = rng.choice(hist.size, size=sel.sum(), p=w / w.sum())
            cs = rng.choice(cells.size, size=sel.sum(), p=pick[k][i])
            draws[sel] = classes[ms + (hist[0] - months[0]), cs]
        freq = np.bincount(draws, minlength=N_CLASSES) / S
        np.testing.assert_allclose(freq, (a / a.sum()) @ q[:, i, :], atol=4 / np.sqrt(S))


@pytest.mark.parametrize("on", SUBSETS, ids=SUBSET_IDS)
def test_exact_slope_matches_nudged_differences(on):
    """Strict: relative tolerance 1e-6, no absolute slack (the prototype's check_slope.py).
    The search fits only the free knobs: 2 per on doll, minus the cell's fixed weight, plus ℓ
    with the neighbourhood on."""
    cells, country = synthetic_grid(n_rows=14, n_cols=20, holes=())
    geo = build_geometry(cells, country)
    months = np.arange(100, 190)
    values = np.random.default_rng(1).choice(VALUES, size=(months.size, cells.size))
    classes = classify(values)
    dc = build_doll_counts(classes, months, geo, on)
    cutoffs = np.array([150, 162])
    obj = Objective(dc, cutoffs, outcome_classes(classes, months, cutoffs, 24))
    theta = np.log([0.8, 0.5, 0.2, 3.0, 8.0, 40.0, 200.0, 1.7])[free_knobs(on)]
    assert theta.size == 2 * on.sum() - 1 + on[NEAR]
    exact = -obj(theta)[1]
    nudge = 1e-4
    numeric = np.array([
        -(obj(theta + nudge * e)[0] - obj(theta - nudge * e)[0]) / (2 * nudge)
        for e in np.eye(theta.size)
    ])
    np.testing.assert_allclose(exact, numeric, rtol=1e-6, atol=0)


# ── the package's sampler ───────────────────────────────────────────────────────────────


def test_with_every_doll_on_the_search_fits_all_eight_knobs_from_the_same_start():
    """All four dolls: the free knobs are the full vector, in the order the search always had."""
    assert free_knobs(ALL_ON).all()
    np.testing.assert_array_equal(start_theta(ALL_ON), np.concatenate(
        [np.log(START_WEIGHT[1:]), np.log(START_MEMORY), [np.log(START_LENGTH)]]))


@pytest.mark.parametrize("on", SUBSETS, ids=SUBSET_IDS)
def test_each_free_knob_gets_its_own_start_value_and_bounds(on, monkeypatch):
    """Built knob by knob from the names, not from the code's own mask: a weight for each on doll
    after the cell, a memory for each on doll, then ℓ if near is on."""
    from views_baseline.model.models.distributional.matryoshka import knob_fit

    expected = ([(f"weight {d}", START_WEIGHT[i], knob_fit.WEIGHT_BOUNDS)
                 for i, d in enumerate(DOLLS) if i > 0 and on[i]]
                + [(f"memory {d}", START_MEMORY[i], knob_fit.MEMORY_BOUNDS)
                   for i, d in enumerate(DOLLS) if on[i]]
                + ([("length", START_LENGTH, knob_fit.LENGTH_BOUNDS)] if on[NEAR] else []))
    np.testing.assert_array_equal(start_theta(on), np.log([v for _, v, _ in expected]))
    seen = {}

    def fake_minimize(fun, x0, jac, method, bounds, options):
        seen["x0"], seen["bounds"] = x0, bounds
        raise StopIteration

    monkeypatch.setattr(knob_fit, "minimize", fake_minimize)
    cells, country = synthetic_grid(n_rows=6, n_cols=8, holes=())
    months = np.arange(100, 140)
    classes = classify(np.random.default_rng(4).choice(VALUES, size=(months.size, cells.size)))
    dc = build_doll_counts(classes, months, build_geometry(cells, country), on)
    with pytest.raises(StopIteration):
        search(dc, np.array([125]), outcome_classes(classes, months, np.array([125]), 6))
    np.testing.assert_array_equal(seen["x0"], np.log([v for _, v, _ in expected]))
    assert seen["bounds"] == [tuple(np.log(b)) for _, _, b in expected]


@pytest.mark.parametrize("on", SUBSETS, ids=SUBSET_IDS)
def test_a_search_stores_an_off_dolls_knobs_as_weight_zero_and_memory_nan(on):
    """And the length scale as NaN when the neighbourhood is off; on dolls get fitted values."""
    cells, country = synthetic_grid(n_rows=10, n_cols=12, holes=())
    geo = build_geometry(cells, country)
    months = np.arange(100, 190)
    classes = classify(np.random.default_rng(4).choice(VALUES, size=(months.size, cells.size)))
    dc = build_doll_counts(classes, months, geo, on)
    cutoffs = np.array([160])
    r = search(dc, cutoffs, outcome_classes(classes, months, cutoffs, 12))
    assert r["dolls"] == [d for d, o in zip(DOLLS, on) if o]
    weight, memory = np.array(r["weight"]), np.array(r["memory"])
    assert np.all(weight[~on] == 0) and np.all(np.isnan(memory[~on]))
    assert np.all(weight[on] > 0) and np.all(np.isfinite(memory[on])) and weight[0] == 1.0
    assert np.isnan(r["length_scale"]) != on[NEAR]
    assert np.isfinite(r["score"]) and r["score"] >= r["start_score"]


def test_the_search_never_computes_an_off_dolls_shares(monkeypatch):
    from views_baseline.model.models.distributional.matryoshka import knob_fit

    seen = []
    real = knob_fit.doll_shares
    monkeypatch.setattr(knob_fit, "doll_shares",
                        lambda dc, W, doll, length=None: seen.append(doll) or real(dc, W, doll,
                                                                                   length))
    cells, country = synthetic_grid(n_rows=10, n_cols=12, holes=())
    months = np.arange(100, 190)
    classes = classify(np.random.default_rng(4).choice(VALUES, size=(months.size, cells.size)))
    on = np.array([True, False, True, False])
    dc = build_doll_counts(classes, months, build_geometry(cells, country), on)
    search(dc, np.array([160]), outcome_classes(classes, months, np.array([160]), 12))
    assert set(seen) == {0, COUNTRY}


def test_kept_positions_skip_excluded_stretches():
    # row 0: positions 0..9, excluded 2, 3 and 6 → kept 0 1 4 5 7 8 9
    # row 1: excluded 0..4 (one stretch of length 5) → kept 5 6 7 ...
    pad = 99
    kept_before = np.array([[2, 2, 4], [0, pad, pad]])
    skip = np.array([[1, 2, 3], [5, 5, 5]])
    got = kept_positions(np.tile(np.arange(7), (2, 1)), np.array([0, 1]), kept_before, skip)
    np.testing.assert_array_equal(got, [[0, 1, 4, 5, 7, 8, 9], [5, 6, 7, 8, 9, 10, 11]])


@pytest.mark.parametrize("on", SUBSETS, ids=SUBSET_IDS)
def test_sampler_shells_are_exactly_the_brute_force_shells(grid, on):
    """Every uniform pick j = 0..size−1 of the country and map shells lands on a distinct cell
    of that shell, and together they are the whole shell."""
    cells, country, geo, _, _ = grid
    shells, _ = brute_force_shells(cells, country, 1.0, on)
    tables = build_shell_tables(geo, on)
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


def _knobs_as_fitted(on):
    """Knobs as the fit stores them for the on-mask: an off doll has weight 0 and memory NaN,
    and without the neighbourhood the length scale is NaN."""
    return {"weight": np.where(on, [1.0, 0.8, 0.4, 0.2], 0.0).tolist(),
            "memory": np.where(on, [3.0, 8.0, 40.0, 200.0], np.nan).tolist(),
            "length_scale": 1.7 if on[NEAR] else np.nan}


@pytest.mark.parametrize("h", [1, 12])
@pytest.mark.parametrize("on", SUBSETS, ids=SUBSET_IDS)
def test_sampler_matches_the_exact_probabilities(grid, h, on):
    """What predict draws equals what the fit scores: class frequencies of many draws against
    the closed form, for a cell with a full neighbourhood and a corner cell."""
    cells, country, geo, months, values = grid
    knobs = _knobs_as_fitted(on)
    cutoff = 150
    hist = months <= cutoff
    sampler = build_sampler(values[hist].astype(np.float32), months[hist], cutoff, knobs, geo,
                            build_shell_tables(geo, on))
    classes = classify(values)
    dc = build_doll_counts(classes[hist], months[hist], geo, on)
    memory = np.asarray(knobs["memory"])
    q = np.stack([doll_shares(dc, time_weights(months[hist], np.array([cutoff]), memory[k]),
                              k, knobs["length_scale"])[0] for k in range(len(DOLLS))])
    rng = np.random.default_rng(2)
    S = 200_000
    near_sizes = shell_sizes(geo, ALL_ON)[1][NEAR]          # cells picked by geometry, not mask
    for i in (int(np.argmax(near_sizes)), int(np.argmin(near_sizes))):
        p_doll = np.diff(np.r_[0.0, sampler.doll_cdf(np.array([i]), np.array([h]))[0, 0]])
        assert np.all(p_doll[~on] == 0.0), "an off doll must never be picked"
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
    _, sizes = shell_sizes(geo, ALL_ON)
    assert (sizes[COUNTRY] == 0).all()
    dc = build_doll_counts(classify(values), months, geo, ALL_ON)
    q = doll_shares(dc, time_weights(months, np.array([120]), 10.0), MAP)
    # On a 6 × 6 grid the central cells have every other cell within 5 cells: empty map shell.
    np.testing.assert_allclose(q.sum(axis=2)[0], (sizes[MAP] > 0).astype(float))
    assert (sizes[MAP] > 0).any()
    knobs = {"weight": [1.0, 0.8, 0.4, 0.2], "memory": [3.0, 8.0, 40.0, 200.0],
             "length_scale": 1.7}
    s = build_sampler(values.astype(np.float32), months, months[-1], knobs, geo,
                      build_shell_tables(geo, ALL_ON))
    draws = s.draw(np.arange(cells.size), np.array([1, 5]),
                   np.random.default_rng(5).random((cells.size, 2, 3, 50)))
    assert set(np.unique(draws)) <= set(VALUES)


def test_forecast_opens_outward_with_distance(grid):
    _, _, geo, months, values = grid
    knobs = {"weight": [1.0, 0.8, 0.4, 0.2], "memory": [3.0, 8.0, 40.0, 200.0],
             "length_scale": 1.7}
    s = build_sampler(values.astype(np.float32), months, months[-1], knobs, geo,
                      build_shell_tables(geo, ALL_ON))
    cdf = s.doll_cdf(np.array([100]), np.array([1, 6, 12, 36]))[0]
    p = np.diff(np.concatenate([np.zeros((4, 1)), cdf], axis=1), axis=1)
    assert np.all(np.diff(p[:, 0]) < 0), "the cell's share must shrink with distance"
    assert np.all(np.diff(p[:, MAP]) > 0), "the map's share must grow with distance"


# ── the model ───────────────────────────────────────────────────────────────────────────

PARTITION = {"test": (493, 540)}
TRAIN_END = 492
CONFIG = dict(n_samples=32, country_column="c_id", backtest_cutoffs=2, backtest_horizon=6,
              stability_refits=False, dolls=list(DOLLS), seed=7)


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
    dc = build_doll_counts(classes, months[hist], build_geometry(cells, country), ALL_ON)
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


# ── the dolls setting (#153) ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("on", SUBSETS, ids=SUBSET_IDS)
def test_every_subset_of_dolls_fits_and_predicts(planted, on):
    """The fitted knobs say which dolls were used, an off doll is stored as weight 0 and memory
    NaN, and predict draws (finite, from the history) with the off dolls never picked."""
    cells, country, months, values, df = planted
    dolls = [d for d, o in zip(DOLLS, on) if o]
    m = _model(dolls=dolls).fit(df)
    k = m.knobs["y"]
    assert k["converged"], k["message"]
    assert k["dolls"] == dolls
    assert np.all(np.array(k["weight"])[~on] == 0) and np.all(np.isnan(np.array(k["memory"])[~on]))
    out = m.predict(df, 0, 3)["y"]
    assert np.isin(out.values, values[months <= TRAIN_END]).all()


def test_with_the_country_off_the_country_codes_change_nothing(planted):
    """Its cells moved out to the map, so relabelling the countries changes neither the knobs
    nor the draws."""
    cells, country, months, values, df = planted
    relabelled = to_df(cells, np.where(country > 0, 50 - country, country), months, values)
    dolls = ["cell", "near", "map"]
    a = _model(dolls=dolls).fit(df)
    b = _model(dolls=dolls).fit(relabelled)
    for key in ("weight", "memory", "length_scale", "score"):
        np.testing.assert_array_equal(a.knobs["y"][key], b.knobs["y"][key])
    np.testing.assert_array_equal(a.predict(df, 2, 3)["y"].values,
                                  b.predict(relabelled, 2, 3)["y"].values)
    full_a = _model().fit(df).knobs["y"]["score"]
    full_b = _model().fit(relabelled).knobs["y"]["score"]
    assert full_a == full_b, "relabelling must not matter to the full model either"


def test_with_the_country_on_merging_two_countries_changes_the_fit(planted):
    """The contrast: the invariance above holds because the country doll is off."""
    cells, country, months, values, df = planted
    merged = to_df(cells, np.where(country > 0, 7, country), months, values)
    assert _model().fit(df).knobs["y"]["score"] != _model().fit(merged).knobs["y"]["score"]
    off = ["cell", "near", "map"]
    assert (_model(dolls=off).fit(df).knobs["y"]["score"]
            == _model(dolls=off).fit(merged).knobs["y"]["score"])


def test_stability_refits_use_the_same_dolls(planted):
    *_, df = planted
    k = _model(dolls=["cell", "country"], stability_refits=True).fit(df).knobs["y"]
    assert len(k["leave_one_cutoff_out"]) == CONFIG["backtest_cutoffs"]
    for r in k["leave_one_cutoff_out"]:
        assert r["dolls"] == ["cell", "country"]
        assert r["weight"][NEAR] == 0 and np.isnan(r["memory"][NEAR])
        assert np.isnan(r["length_scale"])


def test_predict_refuses_knobs_fitted_with_other_dolls(planted):
    *_, df = planted
    m = _model(dolls=["cell", "near"]).fit(df)
    m.dolls = list(DOLLS)
    with pytest.raises(ValueError, match=r"fitted with dolls \['cell', 'near'\]"):
        m.predict(df, 0, 1)


@pytest.mark.parametrize("dolls", [
    ["near", "country", "map"], ["cell", "cell"], ["cell", "map", "near"], ["cell", "ring"],
    [], "cell", ("cell", 1), None,
])
def test_the_dolls_setting_is_checked(dolls):
    with pytest.raises(ValueError, match="dolls must be"):
        _model(dolls=dolls)


def test_the_catalog_requires_dolls():
    """ADR-021: no silent default for which dolls are used."""
    from views_baseline.model.catalog import BaselineModelCatalog

    config = {"regression_targets": ["y"], "n_samples": 8, "seed": 1, "country_column": "c_id",
              "backtest_cutoffs": 1, "backtest_horizon": 6, "stability_refits": False}
    catalog = BaselineModelCatalog(config=config, partition_dict=PARTITION, loa="pgm")
    with pytest.raises(ValueError, match="dolls"):
        catalog.get_model("MatryoshkaConflictology")


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


@pytest.mark.parametrize("on, doll", [(on, d) for on in SUBSETS for d in range(len(DOLLS))
                                      if on[d]],
                         ids=[f"{i}-{DOLLS[d]}" for on, i in zip(SUBSETS, SUBSET_IDS)
                              for d in range(len(DOLLS)) if on[d]])
def test_each_doll_draws_exactly_from_its_shell_with_the_right_cell_weights(grid, on, doll):
    """With all weight on one doll and values that encode their cell, every draw comes from
    that doll's brute-force shell, in the right proportions (kernel for the neighbourhood,
    uniform for country and map). Checked for a full cell, a corner and a no-country cell."""
    cells, country, geo, months, _ = grid
    length = 1.7
    shells, pick = brute_force_shells(cells, country, length, on)
    weight = np.where(on, 1e-300, 0.0)
    weight[doll] = 1.0
    knobs = {"weight": weight, "memory": [10.0] * 4, "length_scale": length}
    values = _encoded(months, cells.size)
    s = build_sampler(values, months, months[-1], knobs, geo, build_shell_tables(geo, on))
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


def test_a_non_zero_weight_on_an_off_doll_raises(grid):
    cells, country, geo, months, values = grid
    on = np.array([True, True, False, True])
    knobs = {**_knobs_as_fitted(on), "weight": [1.0, 0.8, 0.4, 0.2]}
    with pytest.raises(ValueError, match="off must have weight 0"):
        build_sampler(values.astype(np.float32), months, months[-1], knobs, geo,
                      build_shell_tables(geo, on))


@pytest.mark.parametrize("on", SUBSETS, ids=SUBSET_IDS)
def test_an_off_doll_is_never_picked_for_any_cell(grid, on):
    """Its step-1 probability is exactly 0 for every cell, with NaN knobs as the fit stores."""
    cells, country, geo, months, values = grid
    s = build_sampler(values.astype(np.float32), months, months[-1], _knobs_as_fitted(on), geo,
                      build_shell_tables(geo, on))
    s.check_drawable(np.array([1, 12, 36]))
    cdf = s.doll_cdf(np.arange(cells.size), np.array([1, 36]))
    p = np.diff(np.concatenate([np.zeros(cdf.shape[:2] + (1,)), cdf], axis=2), axis=2)
    assert np.all(p[..., ~on] == 0.0)
    assert np.all(np.isfinite(cdf))


def test_a_cell_without_a_country_never_picks_the_country_doll(grid):
    cells, country, geo, months, values = grid
    knobs = {"weight": [1.0, 0.8, 0.4, 0.2], "memory": [3.0, 8.0, 40.0, 200.0],
             "length_scale": 1.7}
    s = build_sampler(values.astype(np.float32), months, months[-1], knobs, geo,
                      build_shell_tables(geo, ALL_ON))
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
    tables = build_shell_tables(geo, ALL_ON)
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
    dc = build_doll_counts(classes, months[hist], build_geometry(cells, country), ALL_ON)
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
    dc = build_doll_counts(classes, months, geo, ALL_ON)
    assert (dc.sizes[MAP] == 0).any() and (dc.sizes[MAP] > 0).any()
    weight, memory, length = np.array([1.0, 0.8, 0.4, 0.2]), np.array([3., 8., 40., 200.]), 1.7
    cutoff, H = 130, 4
    q = np.stack([doll_shares(dc, time_weights(months, np.array([cutoff]), memory[k]), k,
                              length) for k in range(len(DOLLS))])           # (4, 1, N, K)
    obs = outcome_classes(classes, months, np.array([cutoff]), H)
    score, _ = log_score(q, dc.sizes, obs, weight, memory)
    s = build_sampler(values[months <= cutoff].astype(np.float32), months[months <= cutoff],
                      cutoff, {"weight": weight, "memory": memory, "length_scale": length},
                      geo, build_shell_tables(geo, ALL_ON))
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
                      build_shell_tables(geo, ALL_ON))
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
    dc = build_doll_counts(classes, months[hist], build_geometry(cells, country), ALL_ON)
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
    assert any(lv == logging.INFO and "MatryoshkaConflictology y: dolls ['cell', 'near', "
               "'country', 'map']; weight" in s for lv, s in messages)
    assert any(lv == logging.WARNING and "did not converge" in s for lv, s in messages)


def test_fit_warns_when_probabilities_hit_the_floor(caplog):
    import logging

    from views_baseline.model.models.distributional.matryoshka.model import _log_fit

    knobs = {"dolls": DOLLS, "weight": [1.0] * 4, "memory": [1.0] * 4, "length_scale": 1.0,
             "score": -1.0,
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
    with caplog.at_level(logging.INFO):
        _log_fit("y", {**knobs, "floored": 3, "converged": False, "message": "limit"})
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert "did not converge" in warned[0] and "3 backtest probabilities" in warned[1]
    caplog.clear()
    refits = [{**knobs, "left_out": 474, "floored": 2},
              {**knobs, "left_out": 486, "floored": 0, "converged": False, "message": "limit"}]
    with caplog.at_level(logging.INFO):
        _log_fit("y", {**knobs, "floored": 0, "leave_one_cutoff_out": refits})
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 2
    assert "stability refit without cutoff 474, 2 backtest probabilities fell below" in \
        warned[0]
    assert "stability refit without cutoff 486 did not converge" in warned[1]


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
    {"length_scale": 0.0}, {"length_scale": np.nan}, {"length_scale": np.inf},
], ids=lambda c: str(c))
def test_sampler_rejects_knobs_out_of_range(grid, change):
    """Hand-set knobs: every doll's weight and memory is checked, and their number."""
    cells, country, geo, months, values = grid
    with pytest.raises(ValueError, match="knobs out of range"):
        build_sampler(values.astype(np.float32), months, months[-1], {**GOOD_KNOBS, **change},
                      geo, build_shell_tables(geo, ALL_ON))


def test_sampler_accepts_an_infinite_memory_and_a_zero_cell_weight(grid):
    """Allowed, and honoured: with the cell's weight 0 no draw comes from the cell itself, and
    with every memory infinite the months are drawn uniformly over the history."""
    cells, country, geo, months, _ = grid
    knobs = {**GOOD_KNOBS, "weight": [0.0, 0.8, 0.4, 0.2], "memory": [np.inf] * 4}
    values = _encoded(months, cells.size)
    s = build_sampler(values, months, months[-1], knobs, geo, build_shell_tables(geo, ALL_ON))
    S = 20_000
    got = s.draw(np.arange(5), np.array([1, 3]),
                 np.random.default_rng(0).random((5, 2, 3, S))).astype(np.int64)
    assert ((got % cells.size) != np.arange(5)[:, None, None]).all()
    month_share = np.bincount((got // cells.size).ravel(), minlength=months.size) / got.size
    p = 1 / months.size
    np.testing.assert_allclose(month_share, p, atol=5 * np.sqrt(p / got.size))


@pytest.mark.parametrize("change, h, total", [
    ({"memory": [0.01] * 4}, 100, "0.0"),                  # every weight fades to exactly 0
    ({"weight": [0.0] * 4}, 1, "0.0"),
    ({"weight": [1.0, 1e308, 1e308, 1e308]}, [1, 400], "inf"),  # overflows at 1, not at 400
])
def test_draw_fails_loud_when_no_doll_can_be_drawn(grid, change, h, total):
    """Code review of 84fd824: an all-zero or undefined doll row used to fall through to the
    map silently (or crash on an empty map shell)."""
    cells, country, geo, months, values = grid
    s = build_sampler(values.astype(np.float32), months, months[-1], {**GOOD_KNOBS, **change},
                      geo, build_shell_tables(geo, ALL_ON))
    with pytest.raises(ValueError, match=rf"no doll can be drawn.* sums to {total} "):
        h = np.atleast_1d(h)
        s.draw(np.array([5, 7]), h, np.random.default_rng(0).random((2, h.size, 3, 4)))


# Only the cell (fading within a month or so) and the country (never fading): a cell with a
# country can always be drawn; a cell without one cannot, once the cell's weight has faded.
FADES_WITHOUT_A_COUNTRY = {"weight": [1.0, 0.0, 1.0, 0.0], "memory": [0.05, 8.0, np.inf, 200.0],
                           "length_scale": 1.7}


def test_the_no_doll_message_names_the_first_bad_cell_and_months_ahead(grid):
    """Block: a cell with a country (always fine), then two without (bad far ahead). Months
    ahead 50, 1, 60: bad at 50 and 60. Four bad (cell, month) pairs, off the diagonal: the
    message must name the first, the second cell at 50."""
    cells, country, geo, months, values = grid
    shells = build_shell_tables(geo, ALL_ON)
    s = build_sampler(values.astype(np.float32), months, months[-1], FADES_WITHOUT_A_COUNTRY,
                      geo, shells)
    with_country = int(np.nonzero(country > 0)[0][0])
    first, second = np.nonzero(country <= 0)[0][:2]
    assert shells.sizes[COUNTRY, with_country] > 0
    with pytest.raises(ValueError, match=rf"no doll can be drawn for cell {cells[first]} at 50 "
                                         rf"months ahead: .* sums to 0.0 "):
        s.draw(np.array([with_country, first, second]), np.array([50, 1, 60]),
               np.random.default_rng(0).random((3, 3, 3, 4)))


def _block_grid():
    """A 6 × 6 block, one country, two cells without one: every combination of an empty and a
    non-empty country and map shell (centre cells have neither)."""
    rr, cc = np.meshgrid(np.arange(100, 106), np.arange(200, 206), indexing="ij")
    cells = np.sort((rr * N_COLS + cc + 1).ravel())
    country = np.full(cells.size, 7)
    country[[0, 35]] = -1
    return cells, country


SPARSE_CELL = 103 * N_COLS + 210          # nearest neighbour 4 cells away


def _sparse_grid():
    """A 6 × 6 block in two countries, a cell 4 cells off its east edge (in the east country),
    and two isolated cells far apart in country 8, so each has a country shell of exactly one
    cell and an empty neighbourhood."""
    rr, cc = np.meshgrid(np.arange(100, 106), np.arange(200, 206), indexing="ij")
    cells = np.sort(np.r_[(rr * N_COLS + cc + 1).ravel(), SPARSE_CELL, 150 * N_COLS + 401,
                          150 * N_COLS + 421])
    country = np.where((cells - 1) % N_COLS < 203, 7, 9)
    country[[0, 20]] = -1
    country[(cells - 1) // N_COLS == 150] = 8
    return cells, country


@pytest.mark.parametrize("make_grid, n_patterns", [(_block_grid, 4), (_sparse_grid, 3)])
def test_check_drawable_covers_every_pattern_and_every_months_ahead(make_grid, n_patterns,
                                                                    monkeypatch):
    """One cell per pattern of drawable dolls is checked, at every months ahead given."""
    from views_baseline.model.models.distributional.matryoshka.draw import Sampler

    cells, country = make_grid()
    geo = build_geometry(cells, country)
    months = np.arange(100, 130)
    values = np.random.default_rng(4).choice(VALUES, size=(months.size, cells.size))
    s = build_sampler(values.astype(np.float32), months, months[-1], GOOD_KNOBS, geo,
                      build_shell_tables(geo, ALL_ON))
    seen = []
    real = Sampler.doll_cdf

    def spy(self, c, h):
        seen.append((np.asarray(c), np.asarray(h)))
        return real(self, c, h)

    monkeypatch.setattr(Sampler, "doll_cdf", spy)
    h = np.array([1, 7, 36])
    s.check_drawable(h)
    checked = np.concatenate([c for c, _ in seen])
    pattern = s.drawable.T
    assert {tuple(p) for p in pattern[checked]} == {tuple(p) for p in pattern}
    assert len({tuple(p) for p in pattern}) == n_patterns
    assert all(np.array_equal(hh, h) for _, hh in seen)


def test_the_two_grids_have_the_shell_cases_the_coverage_test_needs():
    block = build_shell_tables(build_geometry(*_block_grid()), ALL_ON).sizes
    sparse = build_shell_tables(build_geometry(*_sparse_grid()), ALL_ON).sizes
    assert (block[MAP] == 0).any() and (block[MAP] > 0).any()
    assert (block[COUNTRY] == 0).any() and (block[COUNTRY] > 0).any()
    assert (sparse[COUNTRY] == 1).any() and (sparse[NEAR] == 0).any()


def test_a_neighbourhood_whose_weights_all_underflow_is_never_drawn(monkeypatch):
    """Code review of 8035122 and df349b4: at ℓ = 0.002, exp(−1/ℓ) > 0 but exp(−4/ℓ) = 0, so
    only the sparse cell (nearest neighbour 4 cells away) loses its neighbourhood. It used to
    be picked anyway, and drew another cell's value. Now that cell draws only from its other
    dolls; the block's cells still draw neighbours; and with weight on the neighbourhood alone,
    the sparse cell is named as having no doll to draw."""
    cells, country = _sparse_grid()
    geo = build_geometry(cells, country)
    months = np.arange(100, 130)
    values = _encoded(months, cells.size)
    shells = build_shell_tables(geo, ALL_ON)
    sparse = int(np.nonzero(cells == SPARSE_CELL)[0][0])
    assert shells.sizes[NEAR, sparse] > 0, "the sparse cell does have a neighbour"
    knobs = {"weight": [1.0, 1.0, 0.0, 0.0], "memory": [10.0] * 4, "length_scale": 0.002}
    s = build_sampler(values, months, months[-1], knobs, geo, shells)
    assert not s.drawable[NEAR, sparse] and s.drawable[NEAR, sparse - 1]
    block_cell = int(np.nonzero(cells == 102 * N_COLS + 204)[0][0])     # inside the block
    got = s.draw(np.array([block_cell, sparse]), np.array([1]),
                 np.random.default_rng(0).random((2, 1, 3, 4000)))[:, 0].astype(np.int64)
    source = got % cells.size
    assert (source[1] == sparse).all(), "the sparse cell drew from another cell"
    assert (source[0] != block_cell).any(), "the block cell never drew a neighbour"
    near_only = {**knobs, "weight": [0.0, 1.0, 0.0, 0.0]}
    s = build_sampler(values, months, months[-1], near_only, geo, shells)
    with pytest.raises(ValueError, match=rf"no doll can be drawn for cell {SPARSE_CELL} .* sums "
                                         rf"to 0.0 "):
        s.check_drawable(np.array([1]))


def test_predict_checks_the_knobs_for_every_cell_before_drawing(planted, fitted, monkeypatch):
    """Code review of 49364fb: knobs under which no doll can be drawn fail before the first
    block is drawn, not minutes into predict."""
    from views_baseline.model.models.distributional.matryoshka import draw as draw_module

    _, country, *_, df = planted
    assert country[0] > 0, "the first cell must be fine, so checking it alone is not enough"
    m = _model()
    m.knobs = {"y": FADES_WITHOUT_A_COUNTRY}         # bad only for the cells without a country
    m.entity_ids, m.country, m.history_start = fitted.entity_ids, fitted.country, \
        fitted.history_start

    drawn = []

    def no_draw(self, cells, h, u):
        drawn.append(len(cells))
        return np.zeros((len(cells), len(h), u.shape[-1]))

    monkeypatch.setattr(draw_module.Sampler, "draw", no_draw)
    with pytest.raises(ValueError, match="no doll can be drawn"):
        m.predict(df, 0, 48)
    assert not drawn, "predict drew before it checked the knobs"


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
