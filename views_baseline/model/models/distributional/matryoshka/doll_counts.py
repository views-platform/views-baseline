"""Per month, how many cells of each doll fall in each size class: the data stage of the prototype.

For cell i in month s, the four dolls are disjoint shells (ADR-024, amendment 2026-11-06):
  cell     {i}
  near     its neighbours in the data at grid distance 0 < d ≤ NEAR_RADIUS, grouped in distance
           bands; at draw time a neighbour j is picked with probability ∝ exp(−d_ij / ℓ)
  country  the other cells of its country, outside the cell and its neighbourhood
  map      every other cell (outside the cell, its neighbourhood and its country)

With "inner" = cell + neighbourhood:
  country rest = country total − inner cells in the same country
  map rest     = global total − inner cells − country rest

Almost every cell-month has no deaths (class 0), so only the non-zero classes are stored, as
sparse (N, M) matrices. A doll's class-0 share is 1 minus its other shares (the shares of a
non-empty shell sum to 1 every month). Everything the scoring needs can be summed over months
later, so this stage stores counts, not shares. The panel is complete, so each shell's size does
not change over time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from views_baseline.model.models.distributional.matryoshka.geometry import (
    BAND_DISTANCE,
    OFFSET_BAND,
    Geometry,
    in_a_country,
)
from views_baseline.model.models.distributional.matryoshka.size_classes import N_CLASSES

DOLLS = ["cell", "near", "country", "map"]
CELL, NEAR, COUNTRY, MAP = range(len(DOLLS))


@dataclass(frozen=True)
class DollCounts:
    """Counts per month for every cell's dolls. K = 7 classes, M months, N cells, B distance bands.

    Lists indexed by class hold sparse (N, M) CSR matrices for classes 1..K−1 (index 0 is None):
      own[k]                 1 where the cell is in class k that month
      band[k][b]             number of the cell's neighbours in band b that are in class k
      inner_same_country[k]  cell + neighbourhood, own country only, in class k
    Dense, classes 1..K−1 (row 0 unused, zero):
      country_tot (K, M, C), global_tot (K, M)
    band_sizes (B, N): land neighbours per band; sizes (4, N): shell sizes in DOLLS order (for the
    neighbourhood: its number of land cells; whether it is empty is what matters for the mix).
    """

    months: np.ndarray
    own: list
    band: list
    inner_same_country: list
    country_tot: np.ndarray
    global_tot: np.ndarray
    country_index: np.ndarray   # (N,) row into country_tot, or -1 for no country
    band_sizes: np.ndarray
    sizes: np.ndarray


def group_index(codes: np.ndarray, member: np.ndarray) -> tuple[np.ndarray, int]:
    """(N,) row index per cell into a per-group table (−1 for non-members), and the group count."""
    groups, index = np.unique(codes[member], return_inverse=True)
    out = np.full(codes.size, -1)
    out[member] = index
    return out, groups.size


def _sparse(rows: np.ndarray, cols: np.ndarray, data: np.ndarray, shape) -> sparse.csr_matrix:
    return sparse.csr_matrix((data, (rows, cols)), shape=shape)   # duplicates are summed


def build_doll_counts(classes: np.ndarray, months: np.ndarray, geo: Geometry) -> DollCounts:
    """``classes`` is (M, N): the size class of every cell in every month (complete panel)."""
    M, N = classes.shape
    B = BAND_DISTANCE.size
    if classes.shape != (months.size, geo.cells.size):
        raise ValueError(f"classes {classes.shape} do not match months × cells "
                         f"({months.size}, {geo.cells.size}).")
    nbr = geo.neighbours
    has_country = in_a_country(geo.country)
    country_index, n_countries = group_index(geo.country, has_country)

    s, j = np.nonzero(classes > 0)                       # the non-zero cell-months
    k = classes[s, j]
    # Cell j is cell i's neighbour at offset o exactly when i is j's neighbour at offset −o; the
    # offsets are symmetric and so are their distances, so j's event counts for every i = nbr[j, o]
    # in band OFFSET_BAND[o].
    i_all = nbr[j]                                       # (E, n_offsets)
    present = i_all >= 0
    e_idx, o_idx = np.nonzero(present)
    i_hit, s_hit, k_hit = i_all[e_idx, o_idx], s[e_idx], k[e_idx]
    b_hit = OFFSET_BAND[o_idx]
    same_c = has_country[j[e_idx]] & (country_index[i_hit] == country_index[j[e_idx]])

    own, band, inner_c = [None], [None], [None]
    for cls in range(1, N_CLASSES):
        sel = k == cls
        own.append(_sparse(j[sel], s[sel], np.ones(sel.sum()), (N, M)))
        hit = k_hit == cls
        band.append([_sparse(i_hit[hit & (b_hit == b)], s_hit[hit & (b_hit == b)],
                             np.ones((hit & (b_hit == b)).sum()), (N, M)) for b in range(B)])
        own_in_country = sel & has_country[j]
        rows = np.concatenate([i_hit[hit & same_c], j[own_in_country]])
        cols = np.concatenate([s_hit[hit & same_c], s[own_in_country]])
        inner_c.append(_sparse(rows, cols, np.ones(rows.size), (N, M)))

    country_tot = np.zeros((N_CLASSES, M, n_countries))
    in_c = has_country[j]
    np.add.at(country_tot, (k[in_c], s[in_c], country_index[j[in_c]]), 1.0)
    global_tot = np.zeros((N_CLASSES, M))
    np.add.at(global_tot, (k, s), 1.0)

    band_sizes, sizes = shell_sizes(geo)
    return DollCounts(months=months, own=own, band=band, inner_same_country=inner_c,
                      country_tot=country_tot, global_tot=global_tot, country_index=country_index,
                      band_sizes=band_sizes, sizes=sizes)


def shell_sizes(geo: Geometry) -> tuple[np.ndarray, np.ndarray]:
    """``band_sizes`` (B, N): neighbours per distance band; ``sizes`` (4, N): shell sizes in DOLLS
    order. The panel is complete, so neither changes over time."""
    nbr = geo.neighbours
    N = nbr.shape[0]
    has_country = in_a_country(geo.country)
    country_index, n_countries = group_index(geo.country, has_country)
    band_sizes = np.stack([((nbr >= 0) & (OFFSET_BAND[None, :] == b)).sum(axis=1)
                           for b in range(BAND_DISTANCE.size)])
    n_near = (nbr >= 0).sum(axis=1)
    same_country_nbr = (nbr >= 0) & has_country[:, None] & (
        country_index[np.where(nbr >= 0, nbr, 0)] == country_index[:, None])
    # One extra zero entry for the cells without a country (also when no cell has one).
    country_size = np.r_[np.bincount(country_index[has_country], minlength=n_countries), 0]
    n_country = np.where(has_country,
                         country_size[np.where(has_country, country_index, n_countries)]
                         - 1 - same_country_nbr.sum(axis=1), 0)
    n_map = N - 1 - n_near - n_country
    return band_sizes, np.stack([np.ones(N, int), n_near, n_country, n_map])
