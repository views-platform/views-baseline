from __future__ import annotations

from typing import TYPE_CHECKING

from views_baseline.model.defaults import DEFAULT_SEED
from views_baseline.model.frames.input import to_feature_frame
from views_baseline.model.frames.output import sample_prediction_grid
from views_baseline.model.frames.pooling import window_pool

if TYPE_CHECKING:
    import pandas as pd
    from views_frames import FeatureFrame


class ConflictologyModel:
    distributional = True

    def __init__(
        self,
        targets: list[str],
        window_months: int,
        partition_dict: dict,
        loa: str,
        n_samples: int,
        seed: int = DEFAULT_SEED,
    ):
        """
        Climatology baseline that resamples with replacement from the last window_months
        of data for a given cm/pgm, producing n_samples i.i.d. draws per cell.
        """
        self.targets = targets
        self.partition_dict = partition_dict
        self.loa = loa
        self.window_months = window_months
        self.n_samples = n_samples
        self.seed = seed
        self.time_idx = None
        self.entity_idx = None
        self.hist_per_entity = None
        self.entity_ids = None

    def fit(self, df: pd.DataFrame | FeatureFrame) -> ConflictologyModel:
        """
        Extract and store the last `window_months` of history per entity before the test period.
        """
        test_start = self.partition_dict["test"][0]
        train_end = test_start - 1

        # Normalize to a FeatureFrame at the boundary (ADR-019); windowing runs on its
        # numpy panel, not pandas. Shared with the parametric climatology models so the
        # per-entity pools are byte-identical in order (ADR-022; value precision follows
        # the FeatureFrame's float32 — see C-32).
        ff = to_feature_frame(df, loa=self.loa, targets=self.targets)
        self.time_idx, self.entity_idx = ff.index.level.index_names
        self.entity_ids, self.hist_per_entity = window_pool(
            ff, self.targets, self.window_months, train_end
        )

        return self

    def predict(
        self, df: pd.DataFrame | FeatureFrame, sequence_number: int, output_length: int
    ) -> dict:
        """
        Return predictions as Dict[str, PredictionFrame] — one PF per target.
        Each PF has y_pred shape (N, n_samples) with resampled draws.
        """
        ff = to_feature_frame(df, loa=self.loa, targets=self.targets)
        return sample_prediction_grid(
            entity_ids=self.entity_ids, fitted_state=self.hist_per_entity,
            model_name="ConflictologyModel", targets=self.targets, n_samples=self.n_samples,
            level=ff.index.level, test_start=self.partition_dict["test"][0],
            sequence_number=sequence_number, output_length=output_length, seed=self.seed,
            draw_cell=lambda cid, t, rng: rng.choice(
                self.hist_per_entity[cid][t], size=self.n_samples, replace=True
            ),
        )
