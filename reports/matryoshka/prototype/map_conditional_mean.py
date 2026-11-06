"""Maps of the conditional mean: expected fatalities in a cell-month, given at least one.

Data in:   the calibration input pull; GAUL admin-1 per cell; fitted knobs in <knobs_dir>/{target}.json.
Transform: history up to CUTOFF = TRAIN_END (456); forecast months 457–492 (out of sample). Pooled over
           the 36 forecast months:
             Matryoshka     Σ_h E[value | h] / Σ_h P(value > 0 | h), both exact from the doll mix
             conflictology  sum of the cell's values over its last 36 months / number of non-zero months
             what happened  sum over months 457–492 / number of non-zero months
           Cells where the quantity is undefined (never any deaths) are grey.
Out:       <out_dir>/conditional_mean_{world,africa_middle_east}.png

Run: python map_conditional_mean.py <input.parquet> <gaul1_code.parquet> <knobs_dir> <out_dir>
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
from doll_value_sums import build_doll_value_sums, doll_mean_values  # noqa: E402
from fit_knobs import TARGETS, TRAIN_END, load  # noqa: E402
from forecast_probability import doll_shares, time_weights  # noqa: E402
from grid_geometry import N_COLS, N_ROWS, build_geometry  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from size_classes import classify  # noqa: E402

CUTOFF = TRAIN_END
HORIZON = 36
HERE = Path(__file__).resolve().parent
LABEL = {"lr_sb_best": "state-based", "lr_ns_best": "non-state", "lr_os_best": "one-sided"}
AFRICA_ME = {"rows": (120, 260), "cols": (320, 480)}   # roughly 20°W–60°E, 40°S–40°N


def matryoshka_conditional_mean(dc, dv, knobs: dict) -> np.ndarray:
    weight, memory = np.array(knobs["weight"]), np.array(knobs["memory"])
    cut = np.array([CUTOFF])
    nonzero, mean = [], []
    for k in range(len(DOLLS)):
        W = time_weights(dc.months, cut, memory[k])
        nonzero.append(1.0 - doll_shares(dc, W, k)[0, :, 0])
        mean.append(doll_mean_values(dv, W, k)[0])
    nonzero, mean = np.stack(nonzero), np.stack(mean)                            # (D, N)
    h = np.arange(1, HORIZON + 1)
    A = (weight[:, None] * np.exp(-h[None, :] / memory[:, None]))[:, :, None] \
        * (dc.sizes > 0)[:, None, :]                                             # (D, H, N)
    pi = A / A.sum(0)
    expected = (pi * mean[:, None, :]).sum(0).sum(0)
    p_nonzero = (pi * nonzero[:, None, :]).sum(0).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(p_nonzero > 0, expected / p_nonzero, np.nan)


def pooled_conditional_mean(values: np.ndarray) -> np.ndarray:
    """(M, N) → (N,): sum of values / number of non-zero months; NaN where there are none."""
    n = (values > 0).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, values.sum(0) / n, np.nan)


def to_grid(cells: np.ndarray, values: np.ndarray) -> np.ndarray:
    grid = np.full((N_ROWS, N_COLS), np.nan)
    grid[(cells - 1) // N_COLS, (cells - 1) % N_COLS] = values
    return grid


def main(path: str, province_path: str, knobs_dir: Path, out: Path) -> None:
    months, cells, country, province, values = load(path, province_path, last_month=CUTOFF + HORIZON)
    geo = build_geometry(cells, country, province)
    land = to_grid(cells, np.zeros(cells.size))
    last36 = (months > CUTOFF - 36) & (months <= CUTOFF)
    after = (months > CUTOFF) & (months <= CUTOFF + HORIZON)
    panels = {}
    for target in TARGETS:
        v = values[target]
        dc = build_doll_counts(classify(v), months, geo)
        dv = build_doll_value_sums(v, months, geo, dc.sizes)
        knobs = json.loads((knobs_dir / f"{target}.json").read_text())["fit"]
        panels[target] = [
            ("what happened", pooled_conditional_mean(v[after])),
            ("conflictology", pooled_conditional_mean(v[last36])),
            ("Matryoshka", matryoshka_conditional_mean(dc, dv, knobs)),
        ]
        del dc, dv
    out.mkdir(parents=True, exist_ok=True)
    for name, window in (("world", None), ("africa_middle_east", AFRICA_ME)):
        fig, axes = plt.subplots(3, 3, figsize=(18, 11) if window is None else (15, 14),
                                 constrained_layout=True)
        norm = LogNorm(1, 1000)
        for r, target in enumerate(TARGETS):
            for c, (title, v) in enumerate(panels[target]):
                ax = axes[r, c]
                grid, base = to_grid(cells, v), land
                if window is not None:
                    (r0, r1), (c0, c1) = window["rows"], window["cols"]
                    grid, base = grid[r0:r1, c0:c1], land[r0:r1, c0:c1]
                ax.imshow(np.where(np.isnan(base), np.nan, 0.0), cmap="Greys", vmin=-1, vmax=1,
                          origin="lower", interpolation="nearest")
                im = ax.imshow(grid, cmap="viridis", norm=norm, origin="lower",
                               interpolation="nearest")
                ax.set_title(f"{LABEL[target]}: {title}", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
        fig.colorbar(im, ax=axes, shrink=0.6, extend="both",
                     label="mean fatalities in a cell-month, given at least one (log scale; "
                           "grey = never any deaths)")
        fig.suptitle(f"Conditional mean. History up to month {CUTOFF}; forecast months "
                     f"{CUTOFF + 1}–{CUTOFF + HORIZON} (out of sample)", fontsize=12)
        fig.savefig(out / f"conditional_mean_{name}.png", dpi=110)
        plt.close(fig)
        print("wrote", out / f"conditional_mean_{name}.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))
