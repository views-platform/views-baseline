"""Where each cell's dolls are: its neighbourhood (within NEAR_RADIUS), its country and province.

PRIO-GRID cells are 0.5° squares on a 360 × 720 grid, identified by
``priogrid_id = (row − 1) × 720 + col`` (verified on all 64,818 `land` cells). The neighbourhood of a
cell is every land cell whose centre lies at straight-line grid distance 0 < d ≤ NEAR_RADIUS (in
cells; no latitude correction, #139). Neighbours are grouped into distance bands (all offsets with
the same d), so that a smooth distance kernel can be applied band by band. Only land cells present
in the data count; columns wrap around the date line, rows do not.

The province is carried for the parked admin-1 group (#136), validated here, and not used by the
current dolls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_ROWS, N_COLS = 360, 720
NEAR_RADIUS = 5    # cells (≈ 2.5°); #139
NO_PROVINCE = -1   # GAUL admin-1 "unassigned"; other negative codes are real (Azores, views-datafactory ADR-043)


def _offsets(radius: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(drow, dcol, distance) of every offset with 0 < distance ≤ radius, sorted by distance."""
    dr, dc = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1),
                         indexing="ij")
    dr, dc = dr.ravel(), dc.ravel()
    d = np.sqrt(dr ** 2 + dc ** 2)
    keep = (d > 0) & (d <= radius)
    order = np.lexsort((dc[keep], dr[keep], d[keep]))
    return dr[keep][order], dc[keep][order], d[keep][order]


OFFSET_DROW, OFFSET_DCOL, OFFSET_DISTANCE = _offsets(NEAR_RADIUS)
BAND_DISTANCE, OFFSET_BAND = np.unique(np.round(OFFSET_DISTANCE, 9), return_inverse=True)


@dataclass(frozen=True)
class Geometry:
    """The cells (ascending ``priogrid_id``) and, per cell, its neighbours, country and province.

    ``neighbours[i, o]`` is the index (into ``cells``) of the cell at offset o (OFFSET_DROW/DCOL,
    distance OFFSET_DISTANCE, band OFFSET_BAND) from cell i, or -1 if that is not a land cell in the
    data. ``country[i]`` is the GAUL ``c_id`` (≤ 0 means no country); ``province[i]`` the GAUL
    admin-1 code or NO_PROVINCE.
    """

    cells: np.ndarray        # (N,) priogrid_id, ascending
    country: np.ndarray      # (N,) c_id
    province: np.ndarray     # (N,) gaul1 code or NO_PROVINCE
    neighbours: np.ndarray   # (N, n_offsets) cell index or -1


def build_geometry(cells: np.ndarray, country: np.ndarray, province: np.ndarray) -> Geometry:
    """Neighbour table for the given cells. ``cells`` must be ascending and unique.

    A province must lie inside one country, and a cell without a country has no province.
    """
    cells, country, province = np.asarray(cells), np.asarray(country), np.asarray(province)
    if np.any(np.diff(cells) <= 0):
        raise ValueError("cells must be strictly ascending priogrid ids.")
    has_province = province != NO_PROVINCE
    if np.any(has_province & (country <= 0)):
        raise ValueError("a cell with a province must have a country.")
    pairs = np.unique(np.stack([province[has_province], country[has_province]]), axis=1)
    if np.unique(pairs[0]).size != pairs.shape[1]:
        raise ValueError("a province spans more than one country.")
    row = (cells - 1) // N_COLS          # 0-based
    col = (cells - 1) % N_COLS
    nr = row[:, None] + OFFSET_DROW[None, :]
    nc = (col[:, None] + OFFSET_DCOL[None, :]) % N_COLS
    on_grid = (nr >= 0) & (nr < N_ROWS)
    nid = nr * N_COLS + nc + 1
    pos = np.searchsorted(cells, nid)
    found = on_grid & (pos < cells.size)
    found[found] = cells[pos[found]] == nid[found]
    neighbours = np.where(found, pos, -1)
    return Geometry(cells=cells, country=country, province=province, neighbours=neighbours)
