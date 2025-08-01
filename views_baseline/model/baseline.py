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
        prediction_start = test_start + sequence_number
        logger.debug(f'prediction_start: {prediction_start}')
        prediction_end = prediction_start + output_length
        logger.debug(f'prediction_end: {prediction_end}')

        if self.loa == 'cm':
            loa_index = "country_id"
        elif self.loa == 'pgm':
            loa_index = "priogrid_id"
        else:
            logger.warning(f"Unknown level of analysis: {self.loa}")
            raise ValueError(f"Unknown level of analysis: {self.loa}")
        
        logger.info(f"Currently running a {loa_index} model")

        ### Temporary fix until views-eval filters out the right countries 
        loa_ids = df.index.get_level_values(loa_index).unique()

        list_to_ignore = [59,185,186,187,188,189,191,192,196,197,208,227,230,236,239,240,247,248,250,252,253,254]

        loa_ids = [cid for cid in loa_ids if cid not in list_to_ignore]

        ### Temporary fix over 

        time_ids = list(range(prediction_start, prediction_end))

        records = []
        for cid in loa_ids:
            for tid in time_ids:
                row = {
                    "month_id": tid,
                    loa_index: cid,
                }
                row.update({f"pred_{t}": 0.0 for t in self.targets})
                records.append(row)

        df_preds = pd.DataFrame(records)
        df_preds = df_preds.set_index(["month_id", loa_index]).sort_index()
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

    def fit(self, df: pd.DataFrame):
        """
        Store the last available observation before the test period for each entity.
        """
        test_start, _ = self.partition_dict["test"]
        df = df[df.index.get_level_values("month_id") < test_start]

        if self.loa == "cm":
            loa_index = "country_id"
        elif self.loa == "pgm":
            loa_index = "priogrid_id"
        else:
            logger.warning(f"Unknown level of analysis: {self.loa}")
            raise ValueError(f"Unknown level of analysis: {self.loa}")

        logger.info(f"Fitting LastObservationModel on level: {loa_index}")
        self.last_observations = df.groupby(loa_index)[self.targets].last()
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
        if self.loa == "cm":
            loa_index = "country_id"
        elif self.loa == "pgm":
            loa_index = "priogrid_id"
        else:
            raise ValueError(f"Unknown level of analysis: {self.loa}")
        
        logger.info(f"Generating LOCF predictions on level: {loa_index}")

        loa_ids = df.index.get_level_values(loa_index).unique()
        list_to_ignore = [59, 185, 186, 187, 188, 189, 191, 192, 196, 197,
                        208, 227, 230, 236, 239, 240, 247, 248, 250, 252, 253, 254]
        loa_ids = [cid for cid in loa_ids if cid not in list_to_ignore]

        time_ids = list(range(prediction_start, prediction_end))

        records = []
        for cid in loa_ids:
            if cid not in self.last_observations.index:
                logger.warning(f"No last observation found for {loa_index} = {cid}")
                continue

            last_vals = self.last_observations.loc[cid]
            for tid in time_ids:
                row = {
                    "month_id": tid,
                    loa_index: cid,
                }
                row.update({f"pred_{t}": last_vals[t] for t in self.targets})
                records.append(row)

        df_preds = pd.DataFrame(records)
        df_preds = df_preds.set_index(["month_id", loa_index]).sort_index()
        pred_cols = [f"pred_{t}" for t in self.targets]

        return df_preds[pred_cols]

