from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from views_baseline.model.frames.input import to_index
from views_baseline.model.frames.output import build_prediction_frame
from views_baseline.model.grid import (
    build_time_grid,
    entities_at,
    require_entities,
    train_test_boundary,
)

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

    def fit(self, df: "pd.DataFrame | FeatureFrame") -> ZeroModel:
        # ZeroModel is data-independent — it learns nothing. Validate the declared level and
        # index at the boundary (fail loud on a bad frame) and return; predict derives
        # everything from the input and requires no target columns (C-34).
        to_index(df, loa=self.loa)
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
        test_start, train_end = train_test_boundary(self.partition_dict)

        logger.info(f"Generating ZeroModel predictions on level: {self.loa}")

        level, time, unit = to_index(df, loa=self.loa)
        entity_ids = entities_at(unit, time, train_end)
        require_entities(entity_ids, "ZeroModel")
        time_ids = build_time_grid(test_start, sequence_number, output_length)

        return build_prediction_frame(
            entity_ids=entity_ids,
            time_ids=time_ids,
            targets=self.targets,
            value_fn=lambda cid, target: 0.0,
            level=level,
        )
