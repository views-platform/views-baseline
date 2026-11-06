"""Per month, how many cells of each doll fall in each size class: the data stage of the prototype.

For cell i in month s, the six dolls are disjoint shells (ADR-024; province added for #136):
  cell      {i}
  ring1     its land neighbours at distance 1
  ring2     its land neighbours at distance 2
  province  the other cells of its province (GAUL admin-1), outside the cell and its rings
  country   the other cells of its country, outside the cell, its rings and its province
  map       every other cell (outside the inner 25 and outside its country)

Provinces nest in countries, so with "inner" = cell + rings:
  province rest = province total − inner cells in the same province
  country rest  = country total − inner cells in the same country − province rest
  map rest      = global total − inner cells − (country total − inner cells in the same country)

Everything the scoring needs can be summed over months later, so this stage stores counts, not shares.
The panel is complete (every cell, every month), so each shell's size does not change over time.

The counts are stored once as float32, class first, so that the scoring can multiply them by the time
weights directly: converting them on every scoring cost about 0.7 s per ring (LOG.md, 2026-11-06).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from grid_geometry import NO_PROVINCE, Geometry
from size_classes import N_CLASSES

DOLLS = ["cell", "ring1", "ring2", "province", "country", "map"]
CELL, RING1, RING2, PROVINCE, COUNTRY, MAP = range(len(DOLLS))


@dataclass(frozen=True)
class DollCounts:
    """Class counts per month for every cell's dolls. K = 7 classes, M months, N cells; all float32.

    own                  (K, M, N)  1 where the cell is in that class that month, else 0
    ring1, ring2         (K, M, N)  class counts within each ring
    inner_same_province  (K, M, N)  counts over cell + rings, restricted to the cell's own province
    inner_same_country   (K, M, N)  counts over cell + rings, restricted to the cell's own country
    province_tot         (K, M, P)  counts per province (index from `province_index`)
    country_tot          (K, M, C)  counts per country (index from `country_index`)
    global_tot           (K, M)     counts over all cells
    sizes                (6, N) int shell sizes in DOLLS order
    """

    months: np.ndarray
    own: np.ndarray
    ring1: np.ndarray
    ring2: np.ndarray
    inner_same_province: np.ndarray
    inner_same_country: np.ndarray
    province_tot: np.ndarray
    country_tot: np.ndarray
    global_tot: np.ndarray
    province_index: np.ndarray  # (N,) row into province_tot, or -1 for no province
    country_index: np.ndarray   # (N,) row into country_tot, or -1 for no country
    sizes: np.ndarray


def _ring_counts(classes_s: np.ndarray, nbr: np.ndarray) -> np.ndarray:
    """(N, K) class counts over the given neighbour columns for one month."""
    present = nbr >= 0
    cls = np.where(present, classes_s[np.where(present, nbr, 0)], -1)
    out = np.zeros((classes_s.size, N_CLASSES), dtype=np.uint8)
    for k in range(N_CLASSES):
        out[:, k] = (cls == k).sum(axis=1)
    return out


def group_index(codes: np.ndarray, member: np.ndarray) -> tuple[np.ndarray, int]:
    """(N,) row index per cell into a per-group table (−1 for non-members), and the number of groups."""
    groups, index = np.unique(codes[member], return_inverse=True)
    out = np.full(codes.size, -1)
    out[member] = index
    return out, groups.size


def same_group(nbr: np.ndarray, index: np.ndarray) -> np.ndarray:
    """(N, n_nbr) True where the neighbour exists and is in the cell's own group."""
    return (nbr >= 0) & (index[:, None] >= 0) & (index[np.where(nbr >= 0, nbr, 0)] == index[:, None])


def build_doll_counts(classes: np.ndarray, months: np.ndarray, geo: Geometry) -> DollCounts:
    """``classes`` is (M, N): the size class of every cell in every month (complete panel)."""
    M, N = classes.shape
    nbr1, nbr2 = geo.neighbours[:, Geometry.RING1], geo.neighbours[:, Geometry.RING2]
    has_country = geo.country > 0
    has_province = geo.province != NO_PROVINCE
    country_index, n_countries = group_index(geo.country, has_country)
    province_index, n_provinces = group_index(geo.province, has_province)
    same1_c, same2_c = same_group(nbr1, country_index), same_group(nbr2, country_index)
    same1_p, same2_p = same_group(nbr1, province_index), same_group(nbr2, province_index)

    ring1 = np.empty((M, N, N_CLASSES), np.uint8)
    ring2 = np.empty((M, N, N_CLASSES), np.uint8)
    inner_same_province = np.empty((M, N, N_CLASSES), np.uint8)
    inner_same_country = np.empty((M, N, N_CLASSES), np.uint8)
    province_tot = np.zeros((M, n_provinces, N_CLASSES), np.int32)
    country_tot = np.zeros((M, n_countries, N_CLASSES), np.int32)
    global_tot = np.zeros((M, N_CLASSES), np.int32)
    rows = np.arange(N)
    for s in range(M):
        c = classes[s]
        ring1[s] = _ring_counts(c, nbr1)
        ring2[s] = _ring_counts(c, nbr2)
        own_c = np.zeros((N, N_CLASSES), np.uint8)
        own_c[rows, c] = has_country                 # the cell counts toward its own country
        own_p = np.zeros((N, N_CLASSES), np.uint8)
        own_p[rows, c] = has_province                # ... and toward its own province
        inner_same_country[s] = (_ring_counts(c, np.where(same1_c, nbr1, -1))
                                 + _ring_counts(c, np.where(same2_c, nbr2, -1)) + own_c)
        inner_same_province[s] = (_ring_counts(c, np.where(same1_p, nbr1, -1))
                                  + _ring_counts(c, np.where(same2_p, nbr2, -1)) + own_p)
        np.add.at(country_tot[s], (country_index[has_country], c[has_country]), 1)
        np.add.at(province_tot[s], (province_index[has_province], c[has_province]), 1)
        global_tot[s] = np.bincount(c, minlength=N_CLASSES)

    n1, n2 = (nbr1 >= 0).sum(axis=1), (nbr2 >= 0).sum(axis=1)
    country_size = np.bincount(country_index[has_country], minlength=n_countries)
    province_size = np.bincount(province_index[has_province], minlength=n_provinces)
    n_province = np.where(has_province, province_size[np.maximum(province_index, 0)]
                          - (has_province + same1_p.sum(axis=1) + same2_p.sum(axis=1)), 0)
    n_country_outside_inner = np.where(has_country, country_size[np.maximum(country_index, 0)]
                                       - (has_country + same1_c.sum(axis=1) + same2_c.sum(axis=1)), 0)
    n_country = n_country_outside_inner - n_province
    n_map = N - (1 + n1 + n2) - n_country_outside_inner
    sizes = np.stack([np.ones(N, int), n1, n2, n_province, n_country, n_map])
    own = (np.arange(N_CLASSES)[:, None, None] == classes[None]).astype(np.float32)
    return DollCounts(months=months, own=own, ring1=_class_first(ring1), ring2=_class_first(ring2),
                      inner_same_province=_class_first(inner_same_province),
                      inner_same_country=_class_first(inner_same_country),
                      province_tot=_class_first(province_tot), country_tot=_class_first(country_tot),
                      global_tot=np.ascontiguousarray(global_tot.T, dtype=np.float32),
                      province_index=province_index, country_index=country_index, sizes=sizes)


def _class_first(counts: np.ndarray) -> np.ndarray:
    """(M, X, K) counts → (K, M, X) float32, contiguous per class."""
    return np.ascontiguousarray(np.moveaxis(counts, 2, 0), dtype=np.float32)
