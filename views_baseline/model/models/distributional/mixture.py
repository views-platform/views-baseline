from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from views_baseline.model.defaults import DEFAULT_SEED
from views_baseline.model.frames.input import panel, to_feature_frame
from views_baseline.model.frames.output import sample_prediction_grid
from views_baseline.model.frames.pooling import window_pool_arrays

if TYPE_CHECKING:
    import pandas as pd
    from views_frames import FeatureFrame


class MixtureBaseline:
    distributional = True

    def __init__(
        self,
        targets: list[str],
        window_months: int,
        lambda_mix: float,
        n_samples: int,
        partition_dict: dict,
        loa: str,
        seed: int = DEFAULT_SEED,
    ):
        """
        Mixture empirical baseline that combines local history with a global
        positive pool to avoid the zero-probability trap.
        """
        self.targets = targets
        self.window_months = window_months
        self.lambda_mix = lambda_mix
        self.n_samples = n_samples
        self.partition_dict = partition_dict
        self.loa = loa
        self.seed = seed
        self.time_idx = None
        self.entity_idx = None
        self.local_pool = None
        self.global_pool = None
        self.entity_ids = None

    def fit(self, df: pd.DataFrame | FeatureFrame) -> MixtureBaseline:
        test_start = self.partition_dict["test"][0]
        train_end = test_start - 1

        ff = to_feature_frame(df, loa=self.loa, targets=self.targets)
        self.time_idx, self.entity_idx = ff.index.level.index_names
        time, unit, values = panel(ff, self.targets)

        # Local pool: last window_months values per entity (the shared windowing).
        self.entity_ids, self.local_pool = window_pool_arrays(
            time, unit, values, self.targets, self.window_months, train_end
        )

        # Global pool: all positive values per target across the whole train panel, in
        # (entity, time)-sorted order — matching the pre-PR-2 sorted `train_df[t].values`
        # order so the global `rng.choice` draws stay byte-identical (ADR-011).
        keep = time <= train_end
        order = np.lexsort((time[keep], unit[keep]))
        self.global_pool = {}
        for t in self.targets:
            v = values[t][keep][order]
            self.global_pool[t] = v[v > 0].astype(np.float64)

        return self

    def _sample(self, cid: int, target: str, rng: np.random.Generator) -> np.ndarray:
        """Generate n_samples by mixing local and global pools."""
        local = self.local_pool[cid][target]
        gpool = self.global_pool[target]

        if len(gpool) == 0:
            # No positive values in training data — local only
            return rng.choice(local, size=self.n_samples)

        use_global = rng.random(self.n_samples) < self.lambda_mix
        n_global = int(np.sum(use_global))
        n_local = self.n_samples - n_global

        samples = np.empty(self.n_samples, dtype=np.float64)
        samples[use_global] = rng.choice(gpool, size=n_global)
        samples[~use_global] = rng.choice(local, size=n_local)
        return samples

    def predict(
        self, df: pd.DataFrame | FeatureFrame, sequence_number: int, output_length: int
    ) -> dict:
        ff = to_feature_frame(df, loa=self.loa, targets=self.targets)
        return sample_prediction_grid(
            entity_ids=self.entity_ids, fitted_state=self.local_pool, model_name="MixtureBaseline",
            targets=self.targets, n_samples=self.n_samples, level=ff.index.level,
            test_start=self.partition_dict["test"][0],
            sequence_number=sequence_number, output_length=output_length, seed=self.seed,
            draw_cell=self._sample,
        )
