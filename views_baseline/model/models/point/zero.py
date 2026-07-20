from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from views_baseline.model.frames.input import panel, to_feature_frame
from views_baseline.model.frames.output import build_prediction_frame
from views_baseline.model.grid import build_time_grid, entities_at, require_entities

if TYPE_CHECKING:
    import pandas as pd
    from views_frames import FeatureFrame

logger = logging.getLogger(__name__)


class ZeroModel:
    def __init__(self, targets: list[str], partition_dict: dict, loa: str):
        """
        Baseline model that predicts 0 for all targets.
        """
        self.targets = targets
        self.partition_dict = partition_dict
        self.loa = loa
        self.time_idx = None
        self.entity_idx = None

    def fit(self, df: "pd.DataFrame | FeatureFrame") -> ZeroModel:
        # Normalize at the boundary (ADR-019); record the declared (time, entity) names.
        ff = to_feature_frame(df, loa=self.loa, targets=self.targets)
        self.time_idx, self.entity_idx = ff.index.level.index_names
        return self

    def predict(
        self,
        df: "pd.DataFrame | FeatureFrame",
        sequence_number: int,
        output_length: int,
    ) -> dict:
        """
        Predicts zero for each target variable over output_length time steps
        starting from test_start + sequence_number.
        """
        test_start = self.partition_dict["test"][0]
        train_end = test_start - 1

        logger.info(f"Generating ZeroModel predictions on level: {self.entity_idx}")

        ff = to_feature_frame(df, loa=self.loa, targets=self.targets)
        time, unit, _ = panel(ff, self.targets)
        entity_ids = entities_at(unit, time, train_end)
        require_entities(entity_ids, "ZeroModel")
        time_ids = build_time_grid(test_start, sequence_number, output_length)

        return build_prediction_frame(
            entity_ids=entity_ids,
            time_ids=time_ids,
            targets=self.targets,
            value_fn=lambda cid, target: 0.0,
            level=ff.index.level,
        )
