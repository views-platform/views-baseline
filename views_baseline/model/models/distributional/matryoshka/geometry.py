"""Where each cell's dolls are: its neighbourhood (within NEAR_RADIUS) and its country (ADR-024).

PRIO-GRID cells are 0.5° squares on a 360 × 720 grid, identified by
``priogrid_id = (row − 1) × 720 + col`` (verified on all 64,818 `land` cells). The neighbourhood of
a cell is every cell in the data whose centre lies at straight-line grid distance
0 < d ≤ NEAR_RADIUS (in cells; no latitude correction, #141). Neighbours are grouped into distance
bands (all offsets with the same d), so that the smooth distance kernel can be applied band by
band. Columns wrap around the date line, rows do not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_ROWS, N_COLS = 360, 720
NEAR_RADIUS = 5    # cells (≈ 2.5°); ADR-024, amendment 2026-11-06


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
    """The cells (ascending ``priogrid_id``) and, per cell, its neighbours and country.

    ``neighbours[i, o]`` is the index (into ``cells``) of the cell at offset o (OFFSET_DROW/DCOL,
    distance OFFSET_DISTANCE, band OFFSET_BAND) from cell i, or -1 if no such cell is in the data.
    ``country[i]`` is the GAUL country code (``c_id``); see `in_a_country`.
    """

    cells: np.ndarray        # (N,) priogrid_id, ascending
    country: np.ndarray      # (N,) GAUL country code
    neighbours: np.ndarray   # (N, n_offsets) cell index or -1


def in_a_country(country: np.ndarray) -> np.ndarray:
    """Whether each cell belongs to a country: a positive GAUL code. The real data codes "no
    country" as -1; 0 and other non-positive codes are treated the same."""
    return np.asarray(country) > 0


def build_geometry(cells: np.ndarray, country: np.ndarray) -> Geometry:
    """Neighbour table for the given cells. ``cells`` must be ascending, unique priogrid ids."""
    cells, country = np.asarray(cells, dtype=np.int64), np.asarray(country, dtype=np.int64)
    if cells.shape != country.shape:
        raise ValueError(f"cells {cells.shape} and country {country.shape} differ in shape.")
    if np.any(np.diff(cells) <= 0):
        raise ValueError("cells must be strictly ascending priogrid ids.")
    if cells.size and (cells[0] < 1 or cells[-1] > N_ROWS * N_COLS):
        raise ValueError(f"priogrid ids must lie in 1..{N_ROWS * N_COLS}.")
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
    return Geometry(cells=cells, country=country, neighbours=neighbours)
