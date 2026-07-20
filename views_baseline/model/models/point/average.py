from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from views_baseline.model.frames.input import panel, to_feature_frame
from views_baseline.model.frames.output import build_prediction_frame
from views_baseline.model.frames.pooling import window_pool
from views_baseline.model.grid import (
    build_time_grid,
    entities_at,
    filter_entities,
    require_entities,
)

if TYPE_CHECKING:
    import pandas as pd
    from views_frames import FeatureFrame

logger = logging.getLogger(__name__)


class AverageModel:
    def __init__(self, targets: list[str], window_months: int, partition_dict: dict, loa: str):
        """
        Baseline model that carries forward the average of the last window_months months
        specified in the config file.
        """
        self.targets = targets
        self.partition_dict = partition_dict
        self.loa = loa
        self.mean = None  # {entity -> {target -> trailing-window mean}}
        self.window_months = window_months
        self.time_idx = None
        self.entity_idx = None

    def fit(self, df: "pd.DataFrame | FeatureFrame") -> AverageModel:
        """
        Get the average of the last m observations before the test partition for each entity.
        """
        test_start = self.partition_dict["test"][0]
        train_end = test_start - 1

        logger.info(f"Fitting AverageModel on level: {self.entity_idx}")

        ff = to_feature_frame(df, loa=self.loa, targets=self.targets)
        self.time_idx, self.entity_idx = ff.index.level.index_names
        # Trailing-window mean per entity == the mean of the shared window_pool.
        entity_ids, pools = window_pool(ff, self.targets, self.window_months, train_end)
        self.mean = {
            cid: {t: pools[cid][t].mean() for t in self.targets} for cid in entity_ids
        }
        return self

    def predict(
        self,
        df: "pd.DataFrame | FeatureFrame",
        sequence_number: int,
        output_length: int,
    ) -> dict:
        """
        Repeats the average over the last m months for each target
        and entity over the forecast horizon.
        """
        test_start = self.partition_dict["test"][0]
        train_end = test_start - 1

        logger.info(f"Generating average predictions on level: {self.entity_idx}")

        ff = to_feature_frame(df, loa=self.loa, targets=self.targets)
        time, unit, _ = panel(ff, self.targets)
        entity_ids = entities_at(unit, time, train_end)
        entity_ids = filter_entities(entity_ids, self.mean, "AverageModel")
        require_entities(entity_ids, "AverageModel")
        time_ids = build_time_grid(test_start, sequence_number, output_length)

        return build_prediction_frame(
            entity_ids=entity_ids,
            time_ids=time_ids,
            targets=self.targets,
            value_fn=lambda cid, target: self.mean[cid][target],
            level=ff.index.level,
        )
