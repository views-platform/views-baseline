"""Per month, how many cells of each doll fall in each size class: the data stage of the prototype.

For cell i in month s, the five dolls are disjoint shells (ADR-024):
  cell     {i}
  ring 1   its land neighbours at distance 1
  ring 2   its land neighbours at distance 2
  country  the other cells of its country, outside the cell and its rings
  map      every other cell (outside the inner 25 and outside its country)

Everything the scoring needs can be summed over months later, so this stage stores counts, not shares.
The panel is complete (every cell, every month), so each shell's size does not change over time.

The counts are stored once as float32, class first, so that the scoring can multiply them by the time
weights directly: converting them on every scoring cost about 0.7 s per ring (LOG.md, 2026-11-06).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grid_geometry import Geometry
from size_classes import N_CLASSES

DOLLS = ["cell", "ring1", "ring2", "country", "map"]


@dataclass(frozen=True)
class DollCounts:
    """Class counts per month for every cell's dolls. K = 7 classes, M months, N cells; all float32.

    own          (K, M, N)  1 where the cell is in that class that month, else 0
    ring1, ring2 (K, M, N)  class counts within each ring
    inner_same   (K, M, N)  counts over cell + rings, restricted to the cell's own country
    country_tot  (K, M, C)  counts per country (index from `country_index`)
    global_tot   (K, M)     counts over all cells
    sizes        (5, N) int shell sizes: cell, ring1, ring2, rest of country, rest of map
    """

    months: np.ndarray
    own: np.ndarray
    ring1: np.ndarray
    ring2: np.ndarray
    inner_same: np.ndarray
    country_tot: np.ndarray
    global_tot: np.ndarray
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


def build_doll_counts(classes: np.ndarray, months: np.ndarray, geo: Geometry) -> DollCounts:
    """``classes`` is (M, N): the size class of every cell in every month (complete panel)."""
    M, N = classes.shape
    nbr1, nbr2 = geo.neighbours[:, Geometry.RING1], geo.neighbours[:, Geometry.RING2]
    countries, country_index = np.unique(geo.country, return_inverse=True)
    has_country = geo.country > 0
    country_index = np.where(has_country, country_index, -1)
    n_countries = countries.size

    same1 = (nbr1 >= 0) & has_country[:, None] & (
        geo.country[np.where(nbr1 >= 0, nbr1, 0)] == geo.country[:, None])
    same2 = (nbr2 >= 0) & has_country[:, None] & (
        geo.country[np.where(nbr2 >= 0, nbr2, 0)] == geo.country[:, None])

    ring1 = np.empty((M, N, N_CLASSES), np.uint8)
    ring2 = np.empty((M, N, N_CLASSES), np.uint8)
    inner_same = np.empty((M, N, N_CLASSES), np.uint8)
    country_tot = np.zeros((M, n_countries, N_CLASSES), np.int32)
    global_tot = np.zeros((M, N_CLASSES), np.int32)
    onehot_rows = np.arange(N)
    for s in range(M):
        c = classes[s]
        ring1[s] = _ring_counts(c, nbr1)
        ring2[s] = _ring_counts(c, nbr2)
        same = (_ring_counts(c, np.where(same1, nbr1, -1))
                + _ring_counts(c, np.where(same2, nbr2, -1)))
        own = np.zeros((N, N_CLASSES), np.uint8)
        own[onehot_rows, c] = has_country            # the cell counts toward its own country
        inner_same[s] = same + own
        np.add.at(country_tot[s], (country_index[has_country], c[has_country]), 1)
        global_tot[s] = np.bincount(c, minlength=N_CLASSES)

    n1 = (nbr1 >= 0).sum(axis=1)
    n2 = (nbr2 >= 0).sum(axis=1)
    country_size = np.bincount(country_index[has_country], minlength=n_countries)
    n_inner_same = has_country + same1.sum(axis=1) + same2.sum(axis=1)
    n_country = np.where(has_country, country_size[np.maximum(country_index, 0)] - n_inner_same, 0)
    n_map = N - (1 + n1 + n2) - np.where(has_country, country_size[np.maximum(country_index, 0)]
                                         - n_inner_same, 0)
    sizes = np.stack([np.ones(N, int), n1, n2, n_country, n_map])
    own = (np.arange(N_CLASSES)[:, None, None] == classes[None]).astype(np.float32)
    return DollCounts(months=months, own=own, ring1=_class_first(ring1), ring2=_class_first(ring2),
                      inner_same=_class_first(inner_same), country_tot=_class_first(country_tot),
                      global_tot=np.ascontiguousarray(global_tot.T, dtype=np.float32),
                      country_index=country_index, sizes=sizes)


def _class_first(counts: np.ndarray) -> np.ndarray:
    """(M, X, K) counts → (K, M, X) float32, contiguous per class."""
    return np.ascontiguousarray(np.moveaxis(counts, 2, 0), dtype=np.float32)
