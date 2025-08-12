import pandas as pd
import numpy as np
from typing import List, Optional

import logging
logger = logging.getLogger(__name__)



class ZeroModel:
    def __init__(self, targets: List[str], partition_dict:dict, loa:str):
        """
        Baseline model that predicts 0 for all targets.
        """
        self.targets = targets
        self.partition_dict = partition_dict
        self.loa=loa

    def fit(self, df: pd.DataFrame):
        # No training needed
        return self

    def predict(
        self,
        df: pd.DataFrame,
        sequence_number: int,
        output_length: int = 36,
    ) -> pd.DataFrame:
        """
        Predicts zero for each target variable over output_length time steps
        starting from test_start + sequence_number.
        """
        test_start, _ = self.partition_dict["test"]
        time_idx = df.index.names[0]
        entity_idx = df.index.names[1]
        prediction_start = test_start + sequence_number
        logger.debug(f'prediction_start: {prediction_start}')
        prediction_end = prediction_start + output_length
        logger.debug(f'prediction_end: {prediction_end}')
        
        logger.info(f"Currently running a {entity_idx} model")

        train_end = test_start - 1
        loa_ids = df.loc[df.index.get_level_values(time_idx) == train_end].index.get_level_values(entity_idx).unique()

        time_ids = list(range(prediction_start, prediction_end))

        records = []
        for cid in loa_ids:
            for tid in time_ids:
                row = {
                    time_idx: tid,
                    entity_idx: cid,
                }
                row.update({f"pred_{t}": 0.0 for t in self.targets})
                records.append(row)

        df_preds = pd.DataFrame(records)
        df_preds = df_preds.set_index([time_idx, entity_idx]).sort_index()
        pred_cols = [f"pred_{t}" for t in self.targets]
        
        return df_preds[pred_cols]
    


class LocfModel:
    def __init__(self, targets: List[str], partition_dict: dict, loa: str):
        """
        Baseline model that carries forward the last observation for each entity and target.
        """
        self.targets = targets
        self.partition_dict = partition_dict
        self.loa = loa
        self.last_observations = None
        self.time_idx = None,
        self.entity_idx = None

    def fit(self, df: pd.DataFrame):
        """
        Store the last available observation before the test period for each entity.
        """
        test_start, _ = self.partition_dict["test"]
        self.time_idx = df.index.names[0]
        self.entity_idx = df.index.names[1]
        df = df[df.index.get_level_values(self.time_idx) < test_start]

        logger.info(f"Fitting LastObservationModel on level: {self.entity_idx}")
        self.last_observations = df.groupby(self.entity_idx)[self.targets].last()
        return self

    def predict(
        self,
        df: pd.DataFrame,
        sequence_number: int,
        output_length: int = 36,
    ) -> pd.DataFrame:
        """
        Repeats the last observed value for each target and entity over the forecast horizon.
        """
        test_start, _ = self.partition_dict["test"]
        prediction_start = test_start + sequence_number
        prediction_end = prediction_start + output_length
        
        logger.info(f"Generating LOCF predictions on level: {self.entity_idx}")

        #unique ids at test_start -1
        train_end = test_start - 1
        loa_ids = df.loc[df.index.get_level_values(self.time_idx) == train_end].index.get_level_values(self.entity_idx).unique()
        time_ids = list(range(prediction_start, prediction_end))

        records = []
        for cid in loa_ids:
            if cid not in self.last_observations.index:
                logger.warning(f"No last observation found for {self.entity_idx} = {cid}")
                continue

            last_vals = self.last_observations.loc[cid]
            for tid in time_ids:
                row = {
                    self.time_idx: tid,
                    self.entity_idx: cid,
                }
                row.update({f"pred_{t}": last_vals[t] for t in self.targets})
                records.append(row)

        df_preds = pd.DataFrame(records)
        df_preds = df_preds.set_index([self.time_idx, self.entity_idx]).sort_index()
        pred_cols = [f"pred_{t}" for t in self.targets]

        return df_preds[pred_cols]


class AverageModel:
    def __init__(self, targets: List[str], months: int, partition_dict: dict, loa: str):
        """
        Baseline model that carries forward the last observation for each entity and target.
        """
        self.targets = targets
        self.partition_dict = partition_dict
        self.loa = loa
        self.mean = None
        self.months = months
        self.time_idx = None
        self.entity_idx = None

    def fit(self, df: pd.DataFrame):
        """
        Store the last available observation before the test period for each entity.
        """
        test_start, _ = self.partition_dict["test"]
        self.time_idx = df.index.names[0]
        self.entity_idx = df.index.names[1]
        df = df[df.index.get_level_values(self.time_idx) < test_start]

        logger.info(f"Fitting AverageModel on level: {self.entity_idx}")

        df = df.sort_index(level=[self.entity_idx, self.time_idx])

        # Group by loa_index and take mean of last 6 rows
        last_6_rows_mean = (
            df.groupby(level=self.entity_idx, group_keys=False)
            .apply(lambda g: g.tail(self.months)[self.targets].mean())
        )
        self.mean = last_6_rows_mean
        return self

    def predict(
        self,
        df: pd.DataFrame,
        sequence_number: int,
        output_length: int = 36,
    ) -> pd.DataFrame:
        """
        Repeats the last observed value for each target and entity over the forecast horizon.
        """
        test_start, _ = self.partition_dict["test"]
        prediction_start = test_start + sequence_number
        prediction_end = prediction_start + output_length

        logger.info(f"Generating LOCF predictions on level: {self.entity_idx}")

        train_end = test_start - 1
        loa_ids = df.loc[df.index.get_level_values(self.time_idx) == train_end].index.get_level_values(self.entity_idx).unique()

        time_ids = list(range(prediction_start, prediction_end))

        records = []
        for cid in loa_ids:
            if cid not in self.mean.index:
                logger.warning(f"No last observation found for {self.entity_idx} = {cid}")
                continue

            last_vals = self.mean.loc[cid]
            for tid in time_ids:
                row = {
                    self.time_idx: tid,
                    self.entity_idx: cid,
                }
                row.update({f"pred_{t}": last_vals[t] for t in self.targets})
                records.append(row)

        df_preds = pd.DataFrame(records)
        df_preds = df_preds.set_index([self.time_idx, self.entity_idx]).sort_index()
        pred_cols = [f"pred_{t}" for t in self.targets]

        return df_preds[pred_cols]

