"""Where each cell's dolls are: its two neighbour rings and its country (ADR-024).

PRIO-GRID cells are 0.5° squares on a 360 × 720 grid, identified by
``priogrid_id = (row − 1) × 720 + col`` (verified on all 64,818 `land` cells). Ring 1 is the 8 cells
touching a cell (including diagonally), ring 2 the 16 cells around those: the queen-contiguity rings at
Chebyshev distance 1 and 2. Only land cells present in the data count; columns wrap around the
date line, rows do not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_ROWS, N_COLS = 360, 720
NO_PROVINCE = -1   # GAUL admin-1 "unassigned"; other negative codes are real (Azores, views-datafactory ADR-043)


@dataclass(frozen=True)
class Geometry:
    """The cells (ascending ``priogrid_id``) and, per cell, its ring neighbours and country.

    ``neighbours[i, j]`` is the index (into ``cells``) of the j-th neighbour of cell i, or -1 if that
    position is not a land cell in the data. Columns 0–7 are ring 1, columns 8–23 ring 2.
    ``country[i]`` is the GAUL ``c_id`` of cell i (≤ 0 means no country).
    ``province[i]`` is the GAUL admin-1 code of cell i, or NO_PROVINCE. Provinces nest in countries.
    """

    cells: np.ndarray        # (N,) priogrid_id, ascending
    country: np.ndarray      # (N,) c_id
    province: np.ndarray     # (N,) gaul1 code or NO_PROVINCE
    neighbours: np.ndarray   # (N, 24) cell index or -1

    RING1 = slice(0, 8)
    RING2 = slice(8, 24)


def _offsets() -> tuple[np.ndarray, np.ndarray]:
    """(drow, dcol) for ring 1 then ring 2, in a fixed order."""
    ring1, ring2 = [], []
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            d = max(abs(dr), abs(dc))
            if d == 1:
                ring1.append((dr, dc))
            elif d == 2:
                ring2.append((dr, dc))
    both = np.array(ring1 + ring2)
    return both[:, 0], both[:, 1]


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
    dr, dc = _offsets()
    nr = row[:, None] + dr[None, :]
    nc = (col[:, None] + dc[None, :]) % N_COLS
    on_grid = (nr >= 0) & (nr < N_ROWS)
    nid = nr * N_COLS + nc + 1
    pos = np.searchsorted(cells, nid)
    found = on_grid & (pos < cells.size)
    found[found] = cells[pos[found]] == nid[found]
    neighbours = np.where(found, pos, -1)
    return Geometry(cells=cells, country=country, province=province, neighbours=neighbours)
