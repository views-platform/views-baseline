"""Plot: how strongly a single neighbouring cell is drawn from, by its distance (#139).

Per-cell pull, relative to the cell itself (= 1), for a cell with a complete neighbourhood, from the
fitted knobs (doll weights only; memories ignored, i.e. the mix at a short forecast distance):
  square rings (5 groups):   ring-1 cell w_ring1 / 8, ring-2 cell w_ring2 / 16, by Chebyshev distance
  smooth kernel:             w_near · exp(−d/ℓ) / Σ_offsets exp(−d_o/ℓ), by straight-line distance
Data in:   <rings_dir>/{target}.json (5-group fit), <kernel_dir>/{target}.json (kernel fit).
Out:       <out.png>
Run: python plot_neighbour_pull.py <rings_dir> <kernel_dir> <out.png>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from grid_geometry import NEAR_RADIUS, OFFSET_DISTANCE  # noqa: E402

TARGETS = {"lr_sb_best": "state-based", "lr_ns_best": "non-state", "lr_os_best": "one-sided"}


def main(rings_dir: Path, kernel_dir: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True, constrained_layout=True)
    d = np.linspace(0.5, NEAR_RADIUS, 200)
    offsets = np.unique(np.round(OFFSET_DISTANCE, 6))
    for ax, (t, label) in zip(axes, TARGETS.items()):
        rings = json.loads((rings_dir / f"{t}.json").read_text())["fit"]
        kern = json.loads((kernel_dir / f"{t}.json").read_text())["fit"]
        w1, w2 = rings["weight"][1] / 8, rings["weight"][2] / 16
        ax.step([0.5, 1.5, 2.5, NEAR_RADIUS], [w1, w2, 0, 0], where="post", color="#b2182b",
                label="square rings (by ring: 1, 2)")
        ell = kern["length_scale"]
        z = np.exp(-OFFSET_DISTANCE / ell).sum()
        pull = kern["weight"][1] * np.exp(-d / ell) / z
        ax.plot(d, pull, color="#2166ac", lw=2, label=f"smooth kernel (ℓ = {ell:.2f} cells "
                                                       f"= {ell * 0.5:.2f}°)")
        ax.plot(offsets, kern["weight"][1] * np.exp(-offsets / ell) / z, "o", color="#2166ac",
                ms=4)
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1)
        ax.set_title(label)
        ax.set_xlabel("distance to the neighbouring cell (grid cells of 0.5°)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("pull of one neighbouring cell, relative to the cell itself")
    fig.suptitle("How strongly one neighbouring cell is drawn from, by distance "
                 "(dots: the distances that exist on the grid)", fontsize=12)
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
