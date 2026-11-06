"""Maps: what happened vs what conflictology and Matryoshka forecast, per violence type.

Data in:   the calibration input pull; GAUL admin-1 per cell; fitted knobs in <knobs_dir>/{target}.json.
Transform: history up to CUTOFF = TRAIN_END (456), the last month the fit could see; forecast months
           457–492 were never used in fitting (out of sample); for each cell the chance of at least one death, averaged over the
           36 forecast months (Matryoshka: the closed-form mix of dolls; conflictology: the cell's own
           last 36 months, equal weights). Observed: share of those 36 months with at least one death.
Out:       <out_dir>/forecast_maps_world.png and <out_dir>/forecast_maps_africa_middle_east.png.

Run: python map_forecasts.py <input.parquet> <gaul1_code.parquet> <knobs_dir> <out_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from doll_counts import DOLLS, build_doll_counts  # noqa: E402
from fit_knobs import TARGETS, TRAIN_END, load  # noqa: E402
from forecast_probability import doll_shares, time_weights  # noqa: E402
from grid_geometry import N_COLS, N_ROWS, build_geometry  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from size_classes import classify  # noqa: E402

CUTOFF = TRAIN_END
HORIZON = 36
FLOOR = 1e-6   # colour floor; below it (including exact zeros) a land cell is grey. Was 1e-4, which hid
               # real low levels (Saudi Arabia one-sided ≈ 6e-5; LOG.md 2026-11-06)
HERE = Path(__file__).resolve().parent
LABEL = {"lr_sb_best": "state-based", "lr_ns_best": "non-state", "lr_os_best": "one-sided"}
AFRICA_ME = {"rows": (120, 260), "cols": (320, 480)}   # roughly 20°W–60°E, 40°S–40°N


def matryoshka_any_death(dc, knobs: dict) -> np.ndarray:
    """(N,) chance of ≥ 1 death, averaged over forecast months 1..HORIZON after CUTOFF."""
    weight, memory = np.array(knobs["weight"]), np.array(knobs["memory"])
    cut = np.array([CUTOFF])
    not_zero = np.stack([1.0 - doll_shares(dc, time_weights(dc.months, cut, memory[k]), k,
                                           knobs["length_scale"])[0, :, 0]
                         for k in range(len(DOLLS))])                            # (D, N)
    h = np.arange(1, HORIZON + 1)
    A = (weight[:, None] * np.exp(-h[None, :] / memory[:, None]))[:, :, None] \
        * (dc.sizes > 0)[:, None, :]                                             # (D, H, N)
    return ((A * not_zero[:, None, :]).sum(0) / A.sum(0)).mean(0)


def conflictology_any_death(dc) -> np.ndarray:
    W = time_weights(dc.months, np.array([CUTOFF]), np.inf, window=36)
    return 1.0 - doll_shares(dc, W, 0)[0, :, 0]


def to_grid(cells: np.ndarray, values: np.ndarray) -> np.ndarray:
    grid = np.full((N_ROWS, N_COLS), np.nan)
    grid[(cells - 1) // N_COLS, (cells - 1) % N_COLS] = values
    return grid


def main(path: str, province_path: str, knobs_dir: Path, out: Path) -> None:
    months, cells, country, province, values = load(path, province_path, last_month=CUTOFF + HORIZON)
    geo = build_geometry(cells, country, province)
    land = to_grid(cells, np.zeros(cells.size))
    after = (months > CUTOFF) & (months <= CUTOFF + HORIZON)
    panels = {}
    for target in TARGETS:
        classes = classify(values[target])
        dc = build_doll_counts(classes, months, geo)
        knobs = json.loads((knobs_dir / f"{target}.json").read_text())["fit"]
        panels[target] = [
            ("what happened (share of months with ≥1 death)", (classes[after] > 0).mean(0)),
            ("conflictology: chance of ≥1 death", conflictology_any_death(dc)),
            ("Matryoshka: chance of ≥1 death", matryoshka_any_death(dc, knobs)),
        ]
        del dc
    out.mkdir(parents=True, exist_ok=True)
    for name, window in (("world", None), ("africa_middle_east", AFRICA_ME)):
        fig, axes = plt.subplots(3, 3, figsize=(18, 11) if window is None else (15, 14),
                                 constrained_layout=True)
        norm = LogNorm(FLOOR, 1)
        for r, target in enumerate(TARGETS):
            for c, (title, v) in enumerate(panels[target]):
                ax = axes[r, c]
                grid, base = to_grid(cells, np.maximum(v, FLOOR)), land
                if window is not None:
                    (r0, r1), (c0, c1) = window["rows"], window["cols"]
                    grid, base = grid[r0:r1, c0:c1], land[r0:r1, c0:c1]
                ax.imshow(np.where(np.isnan(base), np.nan, 0.0), cmap="Greys", vmin=-1, vmax=1,
                          origin="lower", interpolation="nearest")
                im = ax.imshow(np.where(grid > FLOOR, grid, np.nan), cmap="magma_r", norm=norm,
                               origin="lower", interpolation="nearest")
                ax.set_title(f"{LABEL[target]}: {title}", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
        fig.colorbar(im, ax=axes, shrink=0.6,
                     label="probability / share (log scale; grey = exactly 0 or below 1e-6)")
        fig.suptitle(f"History up to month {CUTOFF}; forecast months {CUTOFF + 1}–{CUTOFF + HORIZON}"
                     f" (out of sample; knobs from {knobs_dir.name}, fitted on months ≤ {TRAIN_END})", fontsize=12)
        fig.savefig(out / f"forecast_maps_{name}.png", dpi=110)
        plt.close(fig)
        print("wrote", out / f"forecast_maps_{name}.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))
