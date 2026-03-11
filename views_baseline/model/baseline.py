import pandas as pd
import numpy as np
from typing import List

import logging

logger = logging.getLogger(__name__)


class ZeroModel:
    def __init__(self, targets: List[str], partition_dict: dict, loa: str):
        """
        Baseline model that predicts 0 for all targets.
        """
        self.targets = targets
        self.partition_dict = partition_dict
        self.loa = loa

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
        logger.debug(f"prediction_start: {prediction_start}")
        prediction_end = prediction_start + output_length
        logger.debug(f"prediction_end: {prediction_end}")

        logger.info(f"Currently running a {entity_idx} model")

        train_end = test_start - 1
        loa_ids = (
            df.loc[df.index.get_level_values(time_idx) == train_end]
            .index.get_level_values(entity_idx)
            .unique()
        )

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
        self.time_idx = (None,)
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

        # unique ids at test_start -1
        train_end = test_start - 1
        loa_ids = (
            df.loc[df.index.get_level_values(self.time_idx) == train_end]
            .index.get_level_values(self.entity_idx)
            .unique()
        )
        time_ids = list(range(prediction_start, prediction_end))

        records = []
        for cid in loa_ids:
            if cid not in self.last_observations.index:
                logger.warning(
                    f"No last observation found for {self.entity_idx} = {cid}"
                )
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
        Baseline model that carries forward the average of the last m months specified in the config file.
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
        Get the average of the last m observations before the test partition for each entity.
        """
        test_start, _ = self.partition_dict["test"]
        self.time_idx = df.index.names[0]
        self.entity_idx = df.index.names[1]
        df = df[df.index.get_level_values(self.time_idx) < test_start]

        logger.info(f"Fitting AverageModel on level: {self.entity_idx}")

        df = df.sort_index(level=[self.entity_idx, self.time_idx])

        # Group by loa_index and take mean of last 6 rows
        last_6_rows_mean = df.groupby(level=self.entity_idx, group_keys=False).apply(
            lambda g: g.tail(self.months)[self.targets].mean()
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
        Repeats the average over the last m months for each target and entity over the forecast horizon.
        """
        test_start, _ = self.partition_dict["test"]
        prediction_start = test_start + sequence_number
        prediction_end = prediction_start + output_length

        logger.info(f"Generating AverAGE predictions on level: {self.entity_idx}")

        train_end = test_start - 1
        loa_ids = (
            df.loc[df.index.get_level_values(self.time_idx) == train_end]
            .index.get_level_values(self.entity_idx)
            .unique()
        )

        time_ids = list(range(prediction_start, prediction_end))

        records = []
        for cid in loa_ids:
            if cid not in self.mean.index:
                logger.warning(
                    f"No last observation found for {self.entity_idx} = {cid}"
                )
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


class ConflictologyModel:
    def __init__(self, targets: List[str], months: int, partition_dict: dict, loa: str):
        """
        Baseline model that takes the set of the last w=12 months of data for a given cm/pgm (so for a sequence
        starting at month m, the 12 months from m-12 to m-1).
        """
        self.targets = targets
        self.partition_dict = partition_dict
        self.loa = loa
        self.months = months
        self.time_idx = None
        self.entity_idx = None
        self.hist_per_entity = None
        self.loa_ids = None

    def fit(self, df: pd.DataFrame):
        """
        Extract and store the last `months` of history per entity before the test period.
        """
        test_start, _ = self.partition_dict["test"]
        self.time_idx = df.index.names[0]
        self.entity_idx = df.index.names[1]

        train_end = test_start - 1
        history_start = train_end - (self.months - 1)

        df = df.sort_index(level=[self.time_idx, self.entity_idx])

        df_hist = df[
            (df.index.get_level_values(self.time_idx) >= history_start)
            & (df.index.get_level_values(self.time_idx) <= train_end)
        ]

        last_n_months = df_hist.groupby(level=self.entity_idx, group_keys=False).apply(
            lambda g: g.tail(self.months)
        )

        self.loa_ids = (
            df_hist.loc[df_hist.index.get_level_values(self.time_idx) == train_end]
            .index.get_level_values(self.entity_idx)
            .unique()
        )

        self.hist_per_entity = {}
        for cid in self.loa_ids:
            history = last_n_months.xs(cid, level=self.entity_idx, drop_level=False)
            if history.empty:
                continue
            self.hist_per_entity[cid] = {
                t: history[t].tolist() for t in self.targets
            }

        return self

    def predict(
        self, df: pd.DataFrame, sequence_number: int, output_length: int = 36
    ) -> pd.DataFrame:
        test_start, _ = self.partition_dict["test"]
        time_idx = self.time_idx
        entity_idx = self.entity_idx

        prediction_start = test_start + sequence_number
        prediction_end = prediction_start + output_length
        time_ids = list(range(prediction_start, prediction_end))

        records = []
        for cid in self.loa_ids:
            if cid not in self.hist_per_entity:
                continue

            hist_lists = self.hist_per_entity[cid]
            for tid in time_ids:
                row = {time_idx: tid, entity_idx: cid}
                for t in self.targets:
                    row[f"pred_{t}"] = hist_lists[t]
                records.append(row)

        df_preds = pd.DataFrame(records)
        if not df_preds.empty:
            df_preds = df_preds.set_index([time_idx, entity_idx]).sort_index()
        else:
            df_preds = pd.DataFrame(columns=[f"pred_{t}" for t in self.targets])
            df_preds.index = pd.MultiIndex.from_arrays(
                [[] for _ in range(2)], names=[time_idx, entity_idx]
            )

        return df_preds

    def predict_prediction_frame(
        self, df: pd.DataFrame, sequence_number: int, output_length: int = 36
    ) -> dict:
        """
        Return predictions as Dict[str, PredictionFrame] — one PF per target.
        Each PF has y_pred shape (N, S) where S = self.months.
        """
        from views_pipeline_core.data.prediction_frame import PredictionFrame

        test_start, _ = self.partition_dict["test"]
        prediction_start = test_start + sequence_number
        prediction_end = prediction_start + output_length
        time_ids = list(range(prediction_start, prediction_end))

        entities_with_history = [
            cid for cid in self.loa_ids if cid in self.hist_per_entity
        ]

        if not entities_with_history:
            return {}

        time_arr = []
        unit_arr = []
        for cid in entities_with_history:
            for tid in time_ids:
                time_arr.append(tid)
                unit_arr.append(cid)

        time_arr = np.array(time_arr)
        unit_arr = np.array(unit_arr)
        n_rows = len(time_arr)

        result = {}
        for t in self.targets:
            y_pred = np.empty((n_rows, self.months), dtype=np.float32)
            idx = 0
            for cid in entities_with_history:
                hist = self.hist_per_entity[cid][t]
                for _ in time_ids:
                    y_pred[idx] = hist
                    idx += 1

            result[t] = PredictionFrame(
                y_pred=y_pred,
                identifiers={"time": time_arr.copy(), "unit": unit_arr.copy()},
            )

        return result
