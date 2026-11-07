"""The three-step draw of ADR-024: pick a doll, then a month, then a cell; return its value.

For cell i, forecast month t = c + h (c = the last observed month) and doll k:

  1. doll k     ∝ weight[k] · exp(−h / memory[k])           (empty shells excluded)
  2. month s    ∝ exp(−(c − s) / memory[k])                 (s = the history months ≤ c)
  3. cell j     the cell itself; a neighbour ∝ exp(−d_ij / ℓ); or uniform within the rest of the
                country / the rest of the map

This is exactly the distribution `forecast_probability` scores in closed form; the tests check
the two against each other.

**Uniform draws from a shell without rejection.** The cells are put in country-major order, so
each country is one contiguous block. A shell is then "a range of positions minus a few excluded
stretches" (the country: its block minus the cell and its same-country neighbours; the map:
everything minus the country block and the other inner cells). The j-th kept position is
``j + (total length of the excluded stretches that come before it)``, found with one sorted
search per cell. The tables of excluded stretches depend only on the geometry, not on the knobs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from views_baseline.model.models.distributional.matryoshka.doll_counts import (
    COUNTRY,
    DOLLS,
    MAP,
    group_index,
    shell_sizes,
)
from views_baseline.model.models.distributional.matryoshka.geometry import (
    OFFSET_DISTANCE,
    Geometry,
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
    """

    order: np.ndarray
    block_start: np.ndarray
    country_kept_before: np.ndarray
    country_skip: np.ndarray
    map_kept_before: np.ndarray
    map_skip: np.ndarray
    sizes: np.ndarray


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


def build_shell_tables(geo: Geometry) -> ShellTables:
    N = geo.cells.size
    nbr = geo.neighbours
    has_country = geo.country > 0
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
    # Map shell: the other inner cells (length 1 each) plus the whole country block.
    other = present & ~same_country
    m_starts = np.concatenate([np.where(other, inner_pos, pad),
                               np.where(has_country, start, pad)[:, None]], axis=1)
    m_lengths = np.concatenate([other.astype(np.int64), length[:, None]], axis=1)
    map_kept_before, map_skip = _stretch_table(m_starts, m_lengths, pad)
    _, sizes = shell_sizes(geo)
    return ShellTables(order=order, block_start=start,
                       country_kept_before=country_kept_before, country_skip=country_skip,
                       map_kept_before=map_kept_before, map_skip=map_skip, sizes=sizes)


def _row_searchsorted(table: np.ndarray, rows: np.ndarray, x: np.ndarray, span: float):
    """``searchsorted(table[rows[b]], x[b, ...], side="right")`` for every b at once.

    Each row of ``table`` is ascending with values in [0, span); rows are shifted apart by
    ``span`` so that one search over the flattened table answers every row.
    """
    width = table.shape[1]
    flat = (table + span * np.arange(table.shape[0])[:, None]).ravel()
    shift = (span * rows).reshape(rows.shape + (1,) * (x.ndim - rows.ndim))
    return np.searchsorted(flat, x + shift, side="right") - width * rows.reshape(shift.shape)


def kept_positions(pick: np.ndarray, rows: np.ndarray, kept_before: np.ndarray,
                   skip: np.ndarray) -> np.ndarray:
    """Position of the ``pick``-th kept element of row ``rows[b]``'s shell, for every b.

    The j-th kept position is j plus the total length of the excluded stretches before it:
    with r = the number of stretches whose ``kept_before`` ≤ j, that is ``skip[r − 1]``.
    ``pick`` is (B, ...); ``rows`` (B,); ``kept_before``/``skip`` (N, E) as `ShellTables`.
    """
    span = float(kept_before.max()) + 1.0
    r = _row_searchsorted(kept_before[rows], np.arange(rows.size), pick, span)
    where = rows.reshape(rows.shape + (1,) * (pick.ndim - 1))
    return pick + np.where(r > 0, skip[where, np.maximum(r - 1, 0)], 0)


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

    ``values`` (M, N): the history panel, months ascending, ending at c. ``month_cdf`` (D, M):
    step 2 per doll. ``near_cdf`` (N, n_offsets): step 3 for the neighbourhood (zero row for an
    empty neighbourhood). ``weight``, ``memory`` (D,): step 1.
    """

    values: np.ndarray
    weight: np.ndarray
    memory: np.ndarray
    month_cdf: np.ndarray
    near_cdf: np.ndarray
    neighbours: np.ndarray
    shells: ShellTables

    def doll_cdf(self, cells: np.ndarray, h: np.ndarray) -> np.ndarray:
        """(B, T, D): step 1's cumulative probabilities per cell and months ahead."""
        a = self.weight[None, :] * np.exp(-np.asarray(h, dtype=np.float64)[:, None] / self.memory)
        nonempty = (self.shells.sizes[:, cells] > 0).T                       # (B, D)
        return _cdf(a[None, :, :] * nonempty[:, None, :])

    def draw(self, cells: np.ndarray, h: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Forecast draws (B, T, S) for cells (B,) at months ahead h (T,), from uniforms
        u (B, T, 3, S): u[..., 0, :] picks the doll, u[..., 1, :] the month, u[..., 2, :] the
        cell. Every sample uses its own three uniforms, whatever the knobs."""
        cells = np.asarray(cells, dtype=np.int64)
        sh = self.shells
        B = cells.size
        doll = (u[:, :, 0, :, None] >= self.doll_cdf(cells, h)[:, :, None, :]).sum(axis=-1)
        doll = np.minimum(doll, len(DOLLS) - 1)                    # u < 1 ≤ the last cdf entry
        month = _row_searchsorted(self.month_cdf, doll, u[:, :, 1, :], 2.0)
        rows = np.broadcast_to(cells[:, None, None], doll.shape)
        o = _row_searchsorted(self.near_cdf[cells], np.arange(B), u[:, :, 2, :], 2.0)
        near = self.neighbours[rows, np.minimum(o, self.neighbours.shape[1] - 1)]

        def uniform_in(k, kept_before, skip):
            """Position of a uniform pick from shell k, for every sample."""
            size = sh.sizes[k, cells].astype(np.int64)[:, None, None]
            pick = np.minimum((u[:, :, 2, :] * size).astype(np.int64), np.maximum(size - 1, 0))
            return kept_positions(pick, cells, kept_before, skip)

        country = sh.order[np.minimum(
            sh.block_start[cells][:, None, None]
            + uniform_in(COUNTRY, sh.country_kept_before, sh.country_skip), sh.order.size - 1)]
        world = sh.order[np.minimum(uniform_in(MAP, sh.map_kept_before, sh.map_skip),
                                    sh.order.size - 1)]
        j = np.choose(doll, [rows, near, country, world])
        return self.values[month, j]


def build_sampler(values: np.ndarray, months: np.ndarray, last_month: int, knobs: dict,
                  geo: Geometry, shells: ShellTables) -> Sampler:
    """The sampler for history ending at ``last_month`` (``months`` ascending, the last one is
    ``last_month``). ``knobs`` holds ``weight``, ``memory`` (DOLLS order) and ``length_scale``."""
    if months[-1] != last_month:
        raise ValueError(f"the history must end at month {last_month}, got {months[-1]}.")
    weight = np.asarray(knobs["weight"], dtype=np.float64)
    memory = np.asarray(knobs["memory"], dtype=np.float64)
    age = (last_month - months).astype(np.float64)
    month_w = np.stack([np.ones_like(age) if np.isinf(m) else np.exp(-age / m) for m in memory])
    near_w = np.where(geo.neighbours >= 0,
                      np.exp(-OFFSET_DISTANCE / float(knobs["length_scale"]))[None, :], 0.0)
    return Sampler(values=values, weight=weight, memory=memory, month_cdf=_cdf(month_w),
                   near_cdf=_cdf(near_w), neighbours=geo.neighbours, shells=shells)
