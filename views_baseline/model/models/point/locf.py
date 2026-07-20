from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from views_baseline.model.frames.input import to_feature_frame, to_index
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


class LocfModel:
    def __init__(self, targets: list[str], partition_dict: dict, loa: str):
        """
        Baseline model that carries forward the last observation for each entity and target.
        """
        self.targets = targets
        self.partition_dict = partition_dict
        self.loa = loa
        self.last_observations = None  # {entity -> {target -> last observed value}}

    def fit(self, df: "pd.DataFrame | FeatureFrame") -> LocfModel:
        """
        Store the last available observation before the test period for each entity.
        """
        test_start = self.partition_dict["test"][0]
        train_end = test_start - 1

        logger.info(f"Fitting LocfModel on level: {self.loa}")

        ff = to_feature_frame(df, loa=self.loa, targets=self.targets)
        # Last observation per entity == the last-window-of-1 pool up to train_end.
        entity_ids, pools = window_pool(ff, self.targets, 1, train_end)
        self.last_observations = {
            cid: {t: pools[cid][t][0] for t in self.targets} for cid in entity_ids
        }
        return self

    def predict(
        self,
        df: "pd.DataFrame | FeatureFrame",
        sequence_number: int,
        output_length: int,
    ) -> dict:
        """
        Repeats the last observed value for each target and entity over the forecast horizon.
        """
        test_start = self.partition_dict["test"][0]
        train_end = test_start - 1

        logger.info(f"Generating LOCF predictions on level: {self.loa}")

        level, time, unit = to_index(df, loa=self.loa)
        entity_ids = entities_at(unit, time, train_end)
        entity_ids = filter_entities(entity_ids, self.last_observations, "LocfModel")
        require_entities(entity_ids, "LocfModel")
        time_ids = build_time_grid(test_start, sequence_number, output_length)

        return build_prediction_frame(
            entity_ids=entity_ids,
            time_ids=time_ids,
            targets=self.targets,
            value_fn=lambda cid, target: self.last_observations[cid][target],
            level=level,
        )
