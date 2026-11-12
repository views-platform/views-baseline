"""The three-step draw of ADR-024: pick a doll, then a month, then a cell; return its value.

For cell i, forecast month t = c + h (c = the last observed month) and doll k:

  1. doll k     ∝ weight[k] · exp(−h / memory[k])           (dolls with nothing to draw excluded)
  2. month s    ∝ exp(−(c − s) / memory[k])                 (s = the history months ≤ c)
  3. cell j     the cell itself; a neighbour ∝ exp(−d_ij / ℓ); or uniform within the rest of the
                country / the rest of the map

Within the fit's knob bounds this is exactly the distribution `forecast_probability` scores in
closed form (below ℓ ≈ 0.003 a cell's neighbour weights can all round to 0; the draw then drops
that neighbourhood, the score does not); the tests check
the two against each other.

**Uniform draws from a shell without rejection.** The cells are put in country-major order, so
each country is one contiguous block. A shell is then "a range of positions minus a few excluded
stretches" (the country: its block minus the cell and its same-country neighbours; the map:
everything minus the country block and the other inner cells). The shells follow the on-mask
(`doll_counts.doll_mask`, #153): without the neighbourhood, the inner set is the cell alone;
without the country, the map excludes the inner cells only; an off doll's shell is empty.
The j-th kept position is ``j + (total length of the excluded stretches that come before it)``,
found with one sorted search per cell. The tables of excluded stretches depend only on the
geometry and the on-mask, not on the knobs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from views_baseline.model.models.distributional.matryoshka.doll_counts import (
    COUNTRY,
    DOLLS,
    MAP,
    NEAR,
    doll_mask,
    doll_names,
    group_index,
    shell_sizes,
)
from views_baseline.model.models.distributional.matryoshka.forecast_probability import (
    time_weights,
)
from views_baseline.model.models.distributional.matryoshka.geometry import (
    OFFSET_DISTANCE,
    Geometry,
    in_a_country,
)


@dataclass(frozen=True)
class ShellTables:
    """Geometry-only tables for uniform draws from the country and map shells.

    ``order`` (N,): cell index at each country-major position. ``block_start`` (N,): position of
    the first cell of each cell's country block (0 for a cell without a country).
    ``country_kept_before``/``country_skip`` (N, E): for the country shell, in block-relative
    positions, per excluded stretch r: kept positions before it, and total excluded length up to
    and including it (padding: kept_before = a large number). ``map_kept_before``/``map_skip``:
    the same for the map shell in absolute positions. ``sizes`` (4, N): shell sizes, DOLLS order.
    ``on`` (D,): the on-mask the shells were built for.
    """

    order: np.ndarray
    block_start: np.ndarray
    country_kept_before: np.ndarray
    country_skip: np.ndarray
    map_kept_before: np.ndarray
    map_skip: np.ndarray
    sizes: np.ndarray
    on: np.ndarray


def _stretch_table(starts: np.ndarray, lengths: np.ndarray, pad: int):
    """Per row, sorted excluded stretches → (kept positions before each, cumulative skip).

    ``starts``/``lengths`` (N, E); unused entries have ``starts == pad`` (sorted last) and
    length 0.
    """
    order = np.argsort(starts, axis=1, kind="stable")
    starts = np.take_along_axis(starts, order, axis=1)
    lengths = np.take_along_axis(lengths, order, axis=1)
    skip = np.cumsum(lengths, axis=1)
    kept_before = starts - (skip - lengths)
    kept_before[starts == pad] = pad              # padding never counts
    return kept_before, skip


def build_shell_tables(geo: Geometry, on: np.ndarray) -> ShellTables:
    """The tables for the on-mask ``on`` (`doll_mask`)."""
    on = doll_mask(on)
    N = geo.cells.size
    # With near off, the inner set is the cell alone: its neighbours move outward.
    nbr = geo.neighbours if on[NEAR] else np.full_like(geo.neighbours, -1)
    has_country = in_a_country(geo.country)
    country_index, _ = group_index(geo.country, has_country)
    # Country-major order: cells with a country, grouped by country; then the cells without one.
    key = np.where(has_country, country_index, np.iinfo(np.int64).max)
    order = np.lexsort((np.arange(N), key))
    position = np.empty(N, dtype=np.int64)
    position[order] = np.arange(N)
    first = np.full(N, -1, dtype=np.int64)        # block start per country index
    block_len = np.zeros(N, dtype=np.int64)
    if has_country.any():
        ci = country_index[order][has_country[order]]
        pos = np.arange(ci.size)
        first_pos = np.r_[0, np.nonzero(np.diff(ci))[0] + 1]
        first[ci[first_pos]] = pos[first_pos]
        block_len[: ci.max() + 1] = np.bincount(ci)
    start = np.where(has_country, first[np.maximum(country_index, 0)], 0)
    length = np.where(has_country, block_len[np.maximum(country_index, 0)], 0)

    inner = np.concatenate([np.arange(N)[:, None], nbr], axis=1)          # (N, 1 + n_offsets)
    present = inner >= 0
    inner_pos = np.where(present, position[np.where(present, inner, 0)], 0)
    same_country = present & has_country[:, None] & (
        country_index[np.where(present, inner, 0)] == country_index[:, None])

    pad = 4 * N + 1
    # Country shell: block-relative positions of the same-country inner cells, length 1 each.
    c_starts = np.where(same_country, inner_pos - start[:, None], pad)
    c_lengths = same_country.astype(np.int64)
    country_kept_before, country_skip = _stretch_table(c_starts, c_lengths, pad)
    # Map shell: the other inner cells (length 1 each) plus the whole country block. With the
    # country off, the map keeps the country: it excludes every inner cell, one by one.
    block = has_country & on[COUNTRY]
    other = present & ~(same_country & on[COUNTRY])
    m_starts = np.concatenate([np.where(other, inner_pos, pad),
                               np.where(block, start, pad)[:, None]], axis=1)
    m_lengths = np.concatenate([other.astype(np.int64), np.where(block, length, 0)[:, None]],
                               axis=1)
    map_kept_before, map_skip = _stretch_table(m_starts, m_lengths, pad)
    _, sizes = shell_sizes(geo, on)
    return ShellTables(order=order, block_start=start,
                       country_kept_before=country_kept_before, country_skip=country_skip,
                       map_kept_before=map_kept_before, map_skip=map_skip, sizes=sizes, on=on)


# rng.random() returns k / 2**53, so u · 2**53 = k is an exact integer, and for any float c in
# [0, 1]: c ≤ u exactly when ceil(c · 2**53) ≤ k (scaling by a power of two is exact). The
# searches below compare these integers, so a draw depends only on its own uniform and its own
# row: never on rounding, on the block it is drawn in, or on its position there (code review
# #147, finding 4).
UNIFORM_STEPS = 2 ** 53


def _uniforms_on_grid(u: np.ndarray) -> np.ndarray:
    """k = u · 2**53 as int64 (exact for rng.random's uniforms; rounded down otherwise)."""
    return np.floor(np.asarray(u, dtype=np.float64) * UNIFORM_STEPS).astype(np.int64)


def _cdf_on_grid(c: np.ndarray) -> np.ndarray:
    """ceil(c · 2**53) as int64, for CDF entries c in [0, 1]: entry ≤ k exactly when c ≤ u."""
    return np.ceil(np.asarray(c, dtype=np.float64) * UNIFORM_STEPS).astype(np.int64)


def _row_searchsorted(table: np.ndarray, rows: np.ndarray, x: np.ndarray,
                      last: np.ndarray) -> np.ndarray:
    """``min(searchsorted(table[rows[b]], x[b], side="right"), last[rows[b]])`` for every b.

    ``table`` (R, W): int64, each row ascending and ≥ 0; ``rows``, ``x``: int64, same shape;
    ``last`` (R,). Each row is shifted into its own integer range, so one search answers every
    element, exactly. Rows are taken in chunks small enough for the shifted values to stay in
    int64 (about 500 rows for a CDF table); each chunk's elements are found once (a slice when
    the rows are already in order, as in the draw; otherwise by sorting on the row), so the cost
    does not grow with the number of chunks. The chunking cannot change the
    answer.
    """
    shape = np.shape(x)
    rows = np.asarray(rows, dtype=np.int64).ravel()
    x = np.asarray(x, dtype=np.int64).ravel()
    out = np.empty(x.size, dtype=np.int64)
    if x.size == 0:
        return out.reshape(shape)
    R, W = table.shape
    span = int(max(table.max(initial=0), x.max())) + 1
    chunk = max(1, (2 ** 62) // span)

    def search(r0: int, index) -> None:
        local = rows[index] - r0
        part = table[r0:r0 + chunk]
        flat = (part + span * np.arange(part.shape[0], dtype=np.int64)[:, None]).ravel()
        i = np.searchsorted(flat, x[index] + span * local, side="right") - W * local
        out[index] = np.minimum(i, last[r0:r0 + chunk][local])

    if R <= chunk:
        search(0, slice(None))
    else:
        ascending = bool(np.all(rows[1:] >= rows[:-1]))     # as in the draw: no sort needed
        order = None if ascending else np.argsort(rows, kind="stable")
        bounds = np.searchsorted(rows if ascending else rows[order],
                                 np.arange(0, R + chunk, chunk))
        for c, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
            if hi > lo:
                search(c * chunk, slice(lo, hi) if ascending else order[lo:hi])
    return out.reshape(shape)


def kept_positions(pick: np.ndarray, rows: np.ndarray, kept_before: np.ndarray,
                   skip: np.ndarray) -> np.ndarray:
    """Position of the ``pick``-th kept element of row ``rows[b]``'s shell, for every b.

    The j-th kept position is j plus the total length of the excluded stretches before it:
    with r = the number of stretches whose ``kept_before`` ≤ j, that is ``skip[r − 1]``.
    ``pick`` is (B, ...); ``rows`` (B, ...) broadcastable to it from the left;
    ``kept_before``/``skip`` (R, E) as `ShellTables`.
    """
    pick = np.asarray(pick, dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    rows = np.broadcast_to(rows.reshape(rows.shape + (1,) * (pick.ndim - rows.ndim)), pick.shape)
    E = kept_before.shape[1]
    r = _row_searchsorted(kept_before, rows, pick, np.full(kept_before.shape[0], E))
    return pick + np.where(r > 0, skip[rows, np.maximum(r - 1, 0)], 0)


def _cdf(p: np.ndarray) -> np.ndarray:
    """Cumulative distribution along the last axis, exactly 1 from the last positive entry on,
    so that a uniform u in [0, 1) never picks a zero-probability entry. Rows of all zeros stay
    zero."""
    c = np.cumsum(p, axis=-1)
    total = c[..., -1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(total > 0, c / total, 0.0)
    last = p.shape[-1] - 1 - np.argmax((p > 0)[..., ::-1], axis=-1)
    idx = np.arange(p.shape[-1])
    return np.where((idx >= last[..., None]) & (total > 0), 1.0, c)


@dataclass(frozen=True)
class Sampler:
    """Everything one violence type needs to draw, for one history end c.

    ``values`` (M, N): the history panel, months ascending, ending at c. ``month_grid`` (D, M):
    step 2's CDF per doll, and ``near_grid`` (N, n_offsets) step 3's for the neighbourhood (zero
    row for an empty neighbourhood), both as `_cdf_on_grid` integers. ``weight``, ``memory``
    (D,): step 1. ``drawable`` (D, N): whether each doll of each cell has a cell to draw.

    ``drawable`` is the shells' non-emptiness, except that a neighbourhood counts only if some
    neighbour has a positive kernel weight: step 1 must never pick a doll that step 3 cannot
    draw from (code review of df349b4). At fitted knobs (ℓ ≥ 0.25) every present neighbour has
    a positive weight, so this equals the shells the fit scores.
    """

    values: np.ndarray
    weight: np.ndarray
    memory: np.ndarray
    month_grid: np.ndarray
    near_grid: np.ndarray
    near_last: np.ndarray       # (N,) last neighbour offset with positive weight (0 if none)
    neighbours: np.ndarray
    shells: ShellTables
    length_scale: float
    drawable: np.ndarray        # (D, N) bool
    cell_ids: np.ndarray        # (N,) priogrid ids, for messages

    def doll_cdf(self, cells: np.ndarray, h: np.ndarray) -> np.ndarray:
        """(B, T, D): step 1's cumulative probabilities per cell and months ahead."""
        a = self.weight[None, :] * np.exp(-np.asarray(h, dtype=np.float64)[:, None] / self.memory)
        p = a[None, :, :] * self.drawable[:, cells].T[:, None, :]
        with np.errstate(over="ignore"):              # an overflow is reported just below
            total = p.sum(axis=-1)
        # Fitted knobs always give a positive, finite total (the cell's weight is 1, memories
        # ≥ 1 month). Hand-set ones may not: all weights fading to 0, or overflowing.
        bad = ~(np.isfinite(total) & (total > 0))
        if bad.any():
            b, t = np.argwhere(bad)[0]
            raise ValueError(f"no doll can be drawn for cell {int(self.cell_ids[cells[b]])} at "
                             f"{int(np.asarray(h)[t])} months ahead: over the dolls it can "
                             f"draw from, weight · exp(−h / memory) sums to {total[b, t]} (weight "
                             f"{self.weight.tolist()}, memory {self.memory.tolist()}; its "
                             f"neighbourhood counts only if a neighbour's weight exp(−d / ℓ) is "
                             f"positive, ℓ = {self.length_scale}).")
        return _cdf(p)

    def check_drawable(self, h: np.ndarray) -> None:
        """Raise (as `doll_cdf`) unless every cell has a doll to draw at every months ahead h.

        A cell's doll mix depends only on h and on which of its dolls are drawable, so one cell
        per pattern of drawable dolls checks them all: at most 2^3 patterns, since a cell's own
        doll is always drawable. Predict calls this before drawing anything.
        """
        _, one_per_pattern = np.unique(self.drawable.T, axis=0, return_index=True)
        self.doll_cdf(np.sort(one_per_pattern), h)

    def draw(self, cells: np.ndarray, h: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Forecast draws (B, T, S) for cells (B,) at months ahead h (T,), from uniforms
        u (B, T, 3, S): u[..., 0, :] picks the doll, u[..., 1, :] the month, u[..., 2, :] the
        cell. Every sample uses its own three uniforms, whatever the knobs. Each sample's cell is
        worked out only for the doll it picked."""
        cells = np.asarray(cells, dtype=np.int64)
        sh = self.shells
        B, T, _, S = u.shape
        doll = (u[:, :, 0, :, None] >= self.doll_cdf(cells, h)[:, :, None, :]).sum(axis=-1)
        doll = np.minimum(doll, len(DOLLS) - 1).ravel()           # u < 1 ≤ the last cdf entry
        u_month = _uniforms_on_grid(u[:, :, 1, :]).ravel()
        u_cell = u[:, :, 2, :].ravel()
        month = _row_searchsorted(self.month_grid, doll, u_month,
                                  np.full(len(DOLLS), self.month_grid.shape[1] - 1))
        b = np.repeat(np.arange(B), T * S)                 # each sample's cell, in the block
        cell = cells[b]
        j = cell.copy()                                            # the cell doll
        m = doll == NEAR
        o = _row_searchsorted(self.near_grid[cells], b[m], _uniforms_on_grid(u_cell[m]),
                              self.near_last[cells])
        j[m] = self.neighbours[cell[m], o]
        for k, kept_before, skip in ((COUNTRY, sh.country_kept_before, sh.country_skip),
                                     (MAP, sh.map_kept_before, sh.map_skip)):
            m = doll == k
            size = sh.sizes[k, cell[m]].astype(np.int64)
            pick = np.minimum((u_cell[m] * size).astype(np.int64), np.maximum(size - 1, 0))
            pos = kept_positions(pick, b[m], kept_before[cells], skip[cells])
            if k == COUNTRY:                     # country positions are relative to its block
                pos += sh.block_start[cell[m]]
            j[m] = sh.order[pos]
        return self.values[month, j].reshape(B, T, S)


def _kernel_weights(length: float) -> np.ndarray:
    with np.errstate(over="ignore"):              # a tiny ℓ only rounds weights to 0
        return np.exp(-OFFSET_DISTANCE / length)


def build_sampler(values: np.ndarray, months: np.ndarray, last_month: int, knobs: dict,
                  geo: Geometry, shells: ShellTables) -> Sampler:
    """The sampler for history ending at ``last_month`` (``months`` ascending, the last one is
    ``last_month``). ``knobs`` holds ``weight``, ``memory`` (DOLLS order) and ``length_scale``.

    The dolls in use are ``shells.on``. An off doll must have weight 0; its memory (and, with the
    neighbourhood off, the length scale) is not used and may be NaN, as the fit stores it."""
    if months[-1] != last_month:
        raise ValueError(f"the history must end at month {last_month}, got {months[-1]}.")
    on = shells.on
    weight = np.asarray(knobs["weight"], dtype=np.float64)
    memory = np.asarray(knobs["memory"], dtype=np.float64)
    length = float(knobs["length_scale"])
    # The fit keeps the knobs in range; these checks are for knobs set by hand. A memory of inf
    # means no fading and is allowed. Whether some doll can be drawn at all is checked per
    # cell and month ahead, in `Sampler.doll_cdf`.
    if not (weight.shape == memory.shape == (len(DOLLS),) and np.all(np.isfinite(weight[on]))
            and np.all(weight[on] >= 0) and np.all(memory[on] > 0)
            and (0 < length < np.inf or not on[NEAR])):
        raise ValueError(f"knobs out of range: {len(DOLLS)} weights, finite and ≥ 0, "
                         f"{len(DOLLS)} memories > 0 (inf allowed), and a finite length scale "
                         f"> 0, for the dolls in use ({doll_names(on)}); got weight "
                         f"{weight.tolist()}, memory {memory.tolist()}, length scale {length}.")
    if np.any(weight[~on] != 0):
        raise ValueError(f"a doll that is off must have weight 0; the dolls in use are "
                         f"{doll_names(on)}, got weight {weight.tolist()}.")
    # An off doll is never drawn (its shell is empty); inf keeps its unused terms finite.
    weight, memory = np.where(on, weight, 0.0), np.where(on, memory, np.inf)
    # Step 2 is the fit's time weighting at the one cutoff c: the same function, so the draw and
    # the score cannot drift apart (code review #147, finding 9).
    month_w = np.stack([time_weights(months, np.array([last_month]), m)[0] for m in memory])
    near_w = np.where(geo.neighbours >= 0,
                      _kernel_weights(length)[None, :] if on[NEAR] else 0.0, 0.0)
    has_near = near_w > 0
    drawable = shells.sizes > 0
    drawable[NEAR] = has_near.any(axis=1)
    near_last = np.where(has_near.any(axis=1),
                         near_w.shape[1] - 1 - np.argmax(has_near[:, ::-1], axis=1), 0)
    return Sampler(values=values, weight=weight, memory=memory,
                   month_grid=_cdf_on_grid(_cdf(month_w)),
                   near_grid=_cdf_on_grid(_cdf(near_w)), near_last=near_last,
                   neighbours=geo.neighbours, shells=shells, length_scale=length,
                   drawable=drawable, cell_ids=geo.cells)
