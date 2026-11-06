"""Per month, the summed fatalities over each doll's cells: the value counterpart of doll_counts.py.

Used for the expected value of a draw. The draw returns an observed value from a cell in the shell,
so E[value] = Σ_k P(doll k) · Σ_s w_k(s) · (mean value over doll k's cells in month s), exactly as the
class shares are built, with values in place of class indicators. Values are ≥ 0, so the sum of all
values equals the sum of the non-zero ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grid_geometry import Geometry


@dataclass(frozen=True)
class DollValueSums:
    """Summed values per month (float64). M months, N cells, C countries.

    own (M, N); ring1, ring2 (M, N); inner_same (M, N): cell + rings, same country only;
    country_tot (M, C); global_tot (M,); country_index (N,) row into country_tot or -1;
    sizes (5, N): the shell sizes, as in DollCounts.
    """

    months: np.ndarray
    own: np.ndarray
    ring1: np.ndarray
    ring2: np.ndarray
    inner_same: np.ndarray
    country_tot: np.ndarray
    global_tot: np.ndarray
    country_index: np.ndarray
    sizes: np.ndarray


def build_doll_value_sums(values: np.ndarray, months: np.ndarray, geo: Geometry,
                          sizes: np.ndarray) -> DollValueSums:
    """``values`` is (M, N), complete panel. ``sizes`` comes from the matching DollCounts."""
    M, N = values.shape
    nbr1, nbr2 = geo.neighbours[:, Geometry.RING1], geo.neighbours[:, Geometry.RING2]
    countries, country_index = np.unique(geo.country, return_inverse=True)
    has_country = geo.country > 0
    country_index = np.where(has_country, country_index, -1)
    same1 = (nbr1 >= 0) & has_country[:, None] & (
        geo.country[np.where(nbr1 >= 0, nbr1, 0)] == geo.country[:, None])
    same2 = (nbr2 >= 0) & has_country[:, None] & (
        geo.country[np.where(nbr2 >= 0, nbr2, 0)] == geo.country[:, None])

    def ring_sum(v: np.ndarray, nbr: np.ndarray, keep: np.ndarray) -> np.ndarray:
        return np.where(keep, v[np.where(nbr >= 0, nbr, 0)], 0.0).sum(axis=1)

    ring1, ring2, inner_same = (np.empty((M, N)) for _ in range(3))
    country_tot = np.zeros((M, countries.size))
    for s in range(M):
        v = values[s].astype(np.float64)
        ring1[s] = ring_sum(v, nbr1, nbr1 >= 0)
        ring2[s] = ring_sum(v, nbr2, nbr2 >= 0)
        inner_same[s] = ring_sum(v, nbr1, same1) + ring_sum(v, nbr2, same2) + v * has_country
        np.add.at(country_tot[s], country_index[has_country], v[has_country])
    return DollValueSums(months=months, own=values.astype(np.float64), ring1=ring1, ring2=ring2,
                         inner_same=inner_same, country_tot=country_tot,
                         global_tot=values.sum(axis=1, dtype=np.float64),
                         country_index=country_index, sizes=sizes)


def doll_mean_values(dv: DollValueSums, W: np.ndarray, doll: int) -> np.ndarray:
    """(R, N): Σ_s W[r, s] · (mean value over doll ``doll``'s cells in month s)."""
    W = W.astype(np.float64)
    own = W @ dv.own
    if doll == 0:
        total = own
    elif doll == 1:
        total = W @ dv.ring1
    elif doll == 2:
        total = W @ dv.ring2
    else:
        has_country = dv.country_index >= 0
        ctry = np.where(has_country, (W @ dv.country_tot)[:, np.maximum(dv.country_index, 0)]
                        - W @ dv.inner_same, 0.0)
        if doll == 3:
            total = ctry
        else:
            total = (W @ dv.global_tot)[:, None] - (own + W @ dv.ring1 + W @ dv.ring2) - ctry
    size = dv.sizes[doll].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(size[None, :] > 0, total / size[None, :], 0.0)
