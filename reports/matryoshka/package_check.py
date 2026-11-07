"""Real-data check of the package build (#127 stage 1–2 gate): not part of CI.

1. Fit `MatryoshkaConflictology` on the pinned calibration input (train_end 456; 4 cutoffs, 36
   months), and compare its knobs with the prototype's results/v5_kernel/*.json.
2. With --predict: time predict for one origin (36 months) at pgm `land`, all cells, for one
   target and S = 64 (about 0.6 GB of output; the full 3 targets × S = 256 is about 7 GB and
   belongs on the views-models pod). Draw time scales with targets × S.

Run: python package_check.py <input.parquet> <out.json> [--predict]
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from views_baseline.model.models.distributional.matryoshka import MatryoshkaConflictology

TARGETS = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
PROTOTYPE = Path(__file__).resolve().parent / "results" / "v5_kernel"


def main(path: str, out: str, predict: bool) -> None:
    df = pd.read_parquet(path, columns=TARGETS + ["c_id"])
    model = MatryoshkaConflictology(
        targets=TARGETS, partition_dict={"test": (457, 492)}, loa="pgm", n_samples=256,
        country_column="c_id", backtest_cutoffs=4, backtest_horizon=36, stability_refits=False,
        seed=42)
    t0 = time.time()
    model.fit(df)
    report = {"fit_seconds": round(time.time() - t0, 1), "targets": {}}
    for t in TARGETS:
        proto = json.loads((PROTOTYPE / f"{t}.json").read_text())["fit"]
        k = model.knobs[t]
        report["targets"][t] = {
            "package": {n: k[n] for n in ("weight", "memory", "length_scale", "score")},
            "prototype": {n: proto[n] for n in ("weight", "memory", "length_scale", "score")},
            "max_relative_knob_gap": float(np.max(np.abs(
                np.r_[k["weight"], k["memory"], k["length_scale"]]
                / np.r_[proto["weight"], proto["memory"], proto["length_scale"]] - 1))),
            "score_gap": k["score"] - proto["score"],
            "converged": k["converged"], "steps": k["steps"],
        }
    Path(out).write_text(json.dumps(report, indent=2))     # kept even if predict is stopped
    if predict:
        model.targets, model.n_samples = TARGETS[:1], 64
        t1 = time.time()
        frames = model.predict(df[TARGETS[:1]], 0, 36)
        report["predict_one_origin"] = {
            "targets": TARGETS[:1], "n_samples": 64, "seconds": round(time.time() - t1, 1),
            "shape": list(frames[TARGETS[0]].values.shape)}
    report["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    Path(out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--predict" in sys.argv)
