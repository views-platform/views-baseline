"""Side-by-side maps of two prototype versions, and their ratio (made for #136: 5 vs 6 groups).

Data in:   two .npz files from dump_version.py (cells, and per target the chance of ≥1 death `_p` and
           the conditional mean `_m`, months 457–492, history up to 456).
Out:       <out_dir>/compare_{chance,conditional_mean}_{world,africa_middle_east}.png; prints ratio summaries.
Run: python compare_maps.py <a.npz> <label a> <b.npz> <label b> <out_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

N_ROWS, N_COLS = 360, 720
TARGETS = {"lr_sb_best": "state-based", "lr_ns_best": "non-state", "lr_os_best": "one-sided"}
AFRICA_ME = (slice(120, 260), slice(320, 480))
WHAT = {"p": ("chance", "chance of ≥1 death", LogNorm(1e-6, 1), "magma_r"),
        "m": ("conditional_mean", "deaths given ≥1", LogNorm(1, 1000), "viridis")}


def grid(cells, v):
    g = np.full((N_ROWS, N_COLS), np.nan)
    g[(cells - 1) // N_COLS, (cells - 1) % N_COLS] = v
    return g


def main(a_path, a_label, b_path, b_label, out):
    a, b = np.load(a_path), np.load(b_path)
    assert (a["cells"] == b["cells"]).all()
    cells = a["cells"]
    out.mkdir(parents=True, exist_ok=True)
    for key, (fname, title, norm, cmap) in WHAT.items():
        for t, label in TARGETS.items():
            r = b[f"{t}_{key}"] / a[f"{t}_{key}"]
            r = r[np.isfinite(r)]
            print(f"{title:20s} {label:12s} ratio {b_label}/{a_label}: median {np.median(r):.3f}, "
                  f"1–99% {np.percentile(r, 1):.3f}–{np.percentile(r, 99):.3f}, "
                  f"share of cells changed by >10%: {(np.abs(np.log(r)) > np.log(1.1)).mean():.1%}")
        for region, window in (("world", None), ("africa_middle_east", AFRICA_ME)):
            fig, axes = plt.subplots(3, 3, figsize=(18, 11) if window is None else (15, 14),
                                     constrained_layout=True)
            for row, (t, label) in enumerate(TARGETS.items()):
                va, vb = grid(cells, a[f"{t}_{key}"]), grid(cells, b[f"{t}_{key}"])
                panels = [(va, f"{a_label}", norm, cmap), (vb, f"{b_label}", norm, cmap),
                          (vb / va, f"ratio {b_label} / {a_label}", LogNorm(0.5, 2), "RdBu_r")]
                for col, (g, sub, nrm, cm) in enumerate(panels):
                    if window is not None:
                        g = g[window]
                    im = axes[row, col].imshow(g, origin="lower", norm=nrm, cmap=cm,
                                               interpolation="nearest")
                    axes[row, col].set_title(f"{label}: {sub}", fontsize=10)
                    axes[row, col].set_xticks([])
                    axes[row, col].set_yticks([])
                    if row == 2:
                        fig.colorbar(im, ax=axes[:, col], shrink=0.5, location="bottom",
                                     label=title if col < 2 else "ratio (red = higher in "
                                     f"{b_label}, blue = lower)")
            fig.suptitle(f"Matryoshka {title}: {a_label} vs {b_label} (history to 456; months "
                         "457–492, out of sample)", fontsize=12)
            fig.savefig(out / f"compare_{fname}_{region}.png", dpi=100)
            plt.close(fig)
            print("wrote", out / f"compare_{fname}_{region}.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], Path(sys.argv[5]))
