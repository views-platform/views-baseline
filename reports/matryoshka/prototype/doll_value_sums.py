"""Per month, the summed fatalities over each doll's cells: the value counterpart of doll_counts.py.

Used for the expected value of a draw. The draw returns an observed value from a cell in the shell,
so E[value] = Σ_k P(doll k) · Σ_s w_k(s) · (mean value over doll k's cells in month s), exactly as the
class shares are built, with values in place of class indicators. Values are ≥ 0, so the sum of all
values equals the sum of the non-zero ones. Shell arithmetic as in doll_counts.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from doll_counts import CELL, COUNTRY, PROVINCE, RING1, RING2, group_index, same_group
from grid_geometry import NO_PROVINCE, Geometry


@dataclass(frozen=True)
class DollValueSums:
    """Summed values per month (float64). M months, N cells, P provinces, C countries.

    own, ring1, ring2 (M, N); inner_same_province, inner_same_country (M, N): cell + rings, own
    province / country only; province_tot (M, P); country_tot (M, C); global_tot (M,);
    province_index, country_index (N,) row into the totals or -1; sizes (6, N) as in DollCounts.
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
    province_index: np.ndarray
    country_index: np.ndarray
    sizes: np.ndarray


def _ring_sum(v: np.ndarray, nbr: np.ndarray, keep: np.ndarray) -> np.ndarray:
    return np.where(keep, v[np.where(nbr >= 0, nbr, 0)], 0.0).sum(axis=1)


def build_doll_value_sums(values: np.ndarray, months: np.ndarray, geo: Geometry,
                          sizes: np.ndarray) -> DollValueSums:
    """``values`` is (M, N), complete panel. ``sizes`` comes from the matching DollCounts."""
    M, N = values.shape
    nbr1, nbr2 = geo.neighbours[:, Geometry.RING1], geo.neighbours[:, Geometry.RING2]
    has_country = geo.country > 0
    has_province = geo.province != NO_PROVINCE
    country_index, n_countries = group_index(geo.country, has_country)
    province_index, n_provinces = group_index(geo.province, has_province)
    same1_c, same2_c = same_group(nbr1, country_index), same_group(nbr2, country_index)
    same1_p, same2_p = same_group(nbr1, province_index), same_group(nbr2, province_index)

    ring1, ring2, inner_p, inner_c = (np.empty((M, N)) for _ in range(4))
    province_tot = np.zeros((M, n_provinces))
    country_tot = np.zeros((M, n_countries))
    for s in range(M):
        v = values[s].astype(np.float64)
        ring1[s] = _ring_sum(v, nbr1, nbr1 >= 0)
        ring2[s] = _ring_sum(v, nbr2, nbr2 >= 0)
        inner_c[s] = _ring_sum(v, nbr1, same1_c) + _ring_sum(v, nbr2, same2_c) + v * has_country
        inner_p[s] = _ring_sum(v, nbr1, same1_p) + _ring_sum(v, nbr2, same2_p) + v * has_province
        np.add.at(country_tot[s], country_index[has_country], v[has_country])
        np.add.at(province_tot[s], province_index[has_province], v[has_province])
    return DollValueSums(months=months, own=values.astype(np.float64), ring1=ring1, ring2=ring2,
                         inner_same_province=inner_p, inner_same_country=inner_c,
                         province_tot=province_tot, country_tot=country_tot,
                         global_tot=values.sum(axis=1, dtype=np.float64),
                         province_index=province_index, country_index=country_index, sizes=sizes)


def doll_mean_values(dv: DollValueSums, W: np.ndarray, doll: int) -> np.ndarray:
    """(R, N): Σ_s W[r, s] · (mean value over doll ``doll``'s cells in month s)."""
    W = W.astype(np.float64)
    if doll == CELL:
        total = W @ dv.own
    elif doll == RING1:
        total = W @ dv.ring1
    elif doll == RING2:
        total = W @ dv.ring2
    else:
        has_country, has_province = dv.country_index >= 0, dv.province_index >= 0
        province_rest = np.where(has_province, (W @ dv.province_tot)[:, np.maximum(dv.province_index, 0)]
                                 - W @ dv.inner_same_province, 0.0)
        country_outside_inner = np.where(
            has_country, (W @ dv.country_tot)[:, np.maximum(dv.country_index, 0)]
            - W @ dv.inner_same_country, 0.0)
        if doll == PROVINCE:
            total = province_rest
        elif doll == COUNTRY:
            total = country_outside_inner - province_rest
        else:
            inner = W @ dv.own + W @ dv.ring1 + W @ dv.ring2
            total = (W @ dv.global_tot)[:, None] - inner - country_outside_inner
    size = dv.sizes[doll].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(size[None, :] > 0, total / size[None, :], 0.0)
