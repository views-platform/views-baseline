"""Predict-only timing on real data, old code against new (#147): not part of CI.

Times one origin (36 months, all `land` cells) for lr_sb_best at S = 64, with the knobs from
`results/package_check.json` set directly (no fit). Run it with the code under test first on
the path, alternating versions in one session so that machine load affects both alike:

  python predict_timing.py <code dir> <input.parquet> results/package_check.json <label>

Results of 2026-11-09 (old = b6dda80, new = 84fd824): `results/predict_timing.json`.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

TARGET = "lr_sb_best"


def main(code_dir: str, input_path: str, knobs_path: str, label: str) -> None:
    sys.path.insert(0, code_dir)
    import views_baseline
    from views_baseline.model.models.distributional.matryoshka import MatryoshkaConflictology

    print(label, "times the code in", views_baseline.__file__)
    with open(knobs_path) as f:
        knobs = json.load(f)["targets"][TARGET]["package"]
    df = pd.read_parquet(input_path, columns=[TARGET, "c_id"])
    months = df.index.get_level_values("month_id")
    cells = np.sort(df.index.get_level_values("priogrid_id").unique().to_numpy())
    country = df.xs(months.min(), level="month_id")["c_id"].reindex(cells).to_numpy()
    if pd.isna(country).any() or not np.isfinite(country.astype(float)).all():
        raise ValueError("c_id is missing or infinite for some cells in the first month.")
    model = MatryoshkaConflictology(
        targets=[TARGET], partition_dict={"test": (457, 492)}, loa="pgm", n_samples=64,
        country_column="c_id", backtest_cutoffs=4, backtest_horizon=36, stability_refits=False,
        dolls=["cell", "near", "country", "map"], seed=42)
    model.knobs = {TARGET: {k: knobs[k] for k in ("weight", "memory", "length_scale")}}
    model.entity_ids, model.country = cells, country.astype(np.int64)
    model.history_start = int(months.min())
    t0 = time.time()
    model.predict(df[[TARGET]], 0, 36)
    print(label, f"predict {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main(*sys.argv[1:5])
