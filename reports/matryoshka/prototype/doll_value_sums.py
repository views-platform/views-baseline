"""Per month, the summed fatalities over each doll's cells: the value counterpart of doll_counts.py.

Used for the expected value of a draw. The draw returns an observed value from a cell in the shell,
so E[value] = Σ_k P(doll k) · Σ_s w_k(s) · (kernel- or size-weighted mean value over doll k's cells in
month s), exactly as the class shares are built, with values in place of class indicators. Values
are ≥ 0, so the sum of all values equals the sum of the non-zero ones; only non-zero cell-months are
stored (sparse). Shell arithmetic as in doll_counts.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from doll_counts import CELL, COUNTRY, NEAR, PROVINCE, group_index
from grid_geometry import BAND_DISTANCE, NO_PROVINCE, OFFSET_BAND, Geometry
from scipy import sparse


@dataclass(frozen=True)
class DollValueSums:
    """Summed values per month. Sparse (N, M) CSR: own, band[b], inner_same_province,
    inner_same_country. Dense: province_tot (M, P), country_tot (M, C), global_tot (M,).
    band_sizes (B, N), sizes (5, N) as in DollCounts."""

    months: np.ndarray
    own: sparse.csr_matrix
    band: list
    inner_same_province: sparse.csr_matrix
    inner_same_country: sparse.csr_matrix
    province_tot: np.ndarray
    country_tot: np.ndarray
    global_tot: np.ndarray
    province_index: np.ndarray
    country_index: np.ndarray
    band_sizes: np.ndarray
    sizes: np.ndarray


def build_doll_value_sums(values: np.ndarray, months: np.ndarray, geo: Geometry,
                          band_sizes: np.ndarray, sizes: np.ndarray) -> DollValueSums:
    """``values`` is (M, N), complete panel. ``band_sizes`` and ``sizes`` come from the DollCounts."""
    M, N = values.shape
    has_country = geo.country > 0
    country_index, n_countries = group_index(geo.country, has_country)
    has_province = geo.province != NO_PROVINCE
    province_index, n_provinces = group_index(geo.province, has_province)
    s, j = np.nonzero(values != 0)
    v = values[s, j].astype(np.float64)
    i_all = geo.neighbours[j]
    e_idx, o_idx = np.nonzero(i_all >= 0)
    i_hit, s_hit, v_hit, b_hit = i_all[e_idx, o_idx], s[e_idx], v[e_idx], OFFSET_BAND[o_idx]
    same_c = has_country[j[e_idx]] & (country_index[i_hit] == country_index[j[e_idx]])
    same_p = has_province[j[e_idx]] & (province_index[i_hit] == province_index[j[e_idx]])

    def csr(rows, cols, data):
        return sparse.csr_matrix((data, (rows, cols)), shape=(N, M))

    own = csr(j, s, v)
    band = [csr(i_hit[b_hit == b], s_hit[b_hit == b], v_hit[b_hit == b])
            for b in range(BAND_DISTANCE.size)]
    in_c = has_country[j]
    inner_c = csr(np.concatenate([i_hit[same_c], j[in_c]]),
                  np.concatenate([s_hit[same_c], s[in_c]]),
                  np.concatenate([v_hit[same_c], v[in_c]]))
    in_p = has_province[j]
    inner_p = csr(np.concatenate([i_hit[same_p], j[in_p]]),
                  np.concatenate([s_hit[same_p], s[in_p]]),
                  np.concatenate([v_hit[same_p], v[in_p]]))
    country_tot = np.zeros((M, n_countries))
    np.add.at(country_tot, (s[in_c], country_index[j[in_c]]), v[in_c])
    province_tot = np.zeros((M, n_provinces))
    np.add.at(province_tot, (s[in_p], province_index[j[in_p]]), v[in_p])
    global_tot = np.bincount(s, weights=v, minlength=M).astype(np.float64)
    return DollValueSums(months=months, own=own, band=band, inner_same_province=inner_p,
                         inner_same_country=inner_c, province_tot=province_tot,
                         country_tot=country_tot, global_tot=global_tot,
                         province_index=province_index, country_index=country_index,
                         band_sizes=band_sizes, sizes=sizes)


def doll_mean_values(dv: DollValueSums, W: np.ndarray, doll: int,
                     length: float | None = None) -> np.ndarray:
    """(R, N): Σ_s W[r, s] · (mean value over doll ``doll``'s cells in month s; kernel-weighted for
    NEAR with length scale ``length``)."""
    W = np.asarray(W, dtype=np.float64)

    def weigh(X):
        return np.asarray(X @ W.T).T

    size = dv.sizes[doll].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        if doll == CELL:
            return weigh(dv.own)
        if doll == NEAR:
            a = np.exp(-BAND_DISTANCE / length)
            z = a @ dv.band_sizes
            return np.where(z > 0, sum(a[b] * weigh(dv.band[b]) for b in range(a.size)) / z, 0.0)
        has_country, has_province = dv.country_index >= 0, dv.province_index >= 0
        province_rest = np.where(has_province,
                                 (W @ dv.province_tot)[:, np.maximum(dv.province_index, 0)]
                                 - weigh(dv.inner_same_province), 0.0)
        country_outside_inner = np.where(has_country,
                                         (W @ dv.country_tot)[:, np.maximum(dv.country_index, 0)]
                                         - weigh(dv.inner_same_country), 0.0)
        if doll == PROVINCE:
            total = province_rest
        elif doll == COUNTRY:
            total = country_outside_inner - province_rest
        else:
            inner = weigh(dv.own) + sum(weigh(x) for x in dv.band)
            total = (W @ dv.global_tot)[:, None] - inner - country_outside_inner
        return np.where(size[None, :] > 0, total / size[None, :], 0.0)
