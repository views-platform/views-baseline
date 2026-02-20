import pandas as pd
import numpy as np
from typing import List, Optional

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
        self.last_n_months = None

    def fit(self, df: pd.DataFrame):
        """
        Store the index names.
        """

        self.time_idx = df.index.names[0]
        self.entity_idx = df.index.names[1]

        return self

    def predict(
        self, df: pd.DataFrame, sequence_number: int, output_length: int = 36
    ) -> pd.DataFrame:
        test_start, _ = self.partition_dict["test"]

        # --- 1. Compute sliding history window for this sequence ---
        # train_end is "last month with data" for this sequence
        train_end = test_start - 1
        history_start = train_end - (self.months - 1)

        time_idx = self.time_idx
        entity_idx = self.entity_idx

        df = df.sort_index(level=[time_idx, entity_idx])

        # restrict to history window
        df_hist = df[
            (df.index.get_level_values(time_idx) >= history_start)
            & (df.index.get_level_values(time_idx) <= train_end)
        ]

        # last `months` per entity
        last_n_months = df_hist.groupby(level=entity_idx, group_keys=False).apply(
            lambda g: g.tail(self.months)
        )

        # entities present at train_end (i.e. have data at the last month with data)
        loa_ids = (
            df_hist.loc[df_hist.index.get_level_values(time_idx) == train_end]
            .index.get_level_values(entity_idx)
            .unique()
        )

        # --- 2. Forecast window for this sequence ---
        prediction_start = test_start + sequence_number
        prediction_end = prediction_start + output_length
        time_ids = list(range(prediction_start, prediction_end))

        records = []
        for cid in loa_ids:
            # history for this entity
            history = last_n_months.xs(cid, level=entity_idx, drop_level=False)

            if history.empty:
                # skip entities without enough history
                continue

            hist_lists = {}
            for t in self.targets:
                hist_lists[t] = history[t].tolist()

            for tid in time_ids:
                row = {time_idx: tid, entity_idx: cid}
                for t in self.targets:
                    row[f"pred_{t}"] = hist_lists[t]
                records.append(row)

        df_preds = pd.DataFrame(records)
        if not df_preds.empty:
            df_preds = df_preds.set_index([time_idx, entity_idx]).sort_index()
        else:
            # return an empty frame with the right columns if nothing could be computed
            df_preds = pd.DataFrame(columns=[f"pred_{t}" for t in self.targets])
            df_preds.index = pd.MultiIndex.from_arrays(
                [[] for _ in range(2)], names=[time_idx, entity_idx]
            )

        return df_preds


class MirrorModel:
    def __init__(self, targets: List[str], partition_dict: dict, loa: str):
        """
        Baseline model that mirrors (reverses) the historical data to create forecasts.
        
        The forecast is the reverse of the history before the test period.
        For example, if the last 36 months before test are [v1, v2, ..., v35, v36],
        the forecast will be [v36, v35, ..., v2, v1].
        """
        self.targets = targets
        self.partition_dict = partition_dict
        self.loa = loa
        self.mirrored_history = None
        self.time_idx = None
        self.entity_idx = None

    def fit(self, df: pd.DataFrame, output_length: int = 36):
        """
        Extract and reverse the last output_length months of history for each entity.
        
        Args:
            df: DataFrame with MultiIndex (time, entity)
            output_length: Number of historical months to mirror (default 36)
        """
        test_start, _ = self.partition_dict["test"]
        self.time_idx = df.index.names[0]
        self.entity_idx = df.index.names[1]
        
        # Filter to training data only
        df_train = df[df.index.get_level_values(self.time_idx) < test_start]
        
        logger.info(f"Fitting MirrorModel on level: {self.entity_idx}")
        
        # Sort by entity and time
        df_train = df_train.sort_index(level=[self.entity_idx, self.time_idx])
        
        # Get last output_length rows per entity and reverse them
        self.mirrored_history = {}
        for entity_id in df_train.index.get_level_values(self.entity_idx).unique():
            entity_data = df_train.xs(entity_id, level=self.entity_idx)
            # Take last output_length months and reverse
            last_n = entity_data[self.targets].tail(output_length)
            # Reverse the order (mirror)
            mirrored = last_n.iloc[::-1].reset_index(drop=True)
            self.mirrored_history[entity_id] = mirrored
        
        return self

    def predict(
        self,
        df: pd.DataFrame,
        sequence_number: int,
        output_length: int = 36,
    ) -> pd.DataFrame:
        """
        Generates forecasts by mirroring the historical data.
        
        The prediction at step k is the value from (output_length - k) months before test_start.
        """
        test_start, _ = self.partition_dict["test"]
        prediction_start = test_start + sequence_number
        prediction_end = prediction_start + output_length

        logger.info(f"Generating Mirror predictions on level: {self.entity_idx}")

        # Get entities available at train_end
        train_end = test_start - 1
        loa_ids = (
            df.loc[df.index.get_level_values(self.time_idx) == train_end]
            .index.get_level_values(self.entity_idx)
            .unique()
        )

        time_ids = list(range(prediction_start, prediction_end))

        records = []
        for cid in loa_ids:
            if cid not in self.mirrored_history:
                logger.warning(
                    f"No mirrored history found for {self.entity_idx} = {cid}"
                )
                continue

            mirrored_vals = self.mirrored_history[cid]
            n_available = len(mirrored_vals)
            
            for i, tid in enumerate(time_ids):
                row = {
                    self.time_idx: tid,
                    self.entity_idx: cid,
                }
                # Use mirrored value if available, otherwise use last available
                idx = min(i, n_available - 1)
                for t in self.targets:
                    row[f"pred_{t}"] = mirrored_vals[t].iloc[idx] if idx < n_available else 0.0
                records.append(row)

        df_preds = pd.DataFrame(records)
        if not df_preds.empty:
            df_preds = df_preds.set_index([self.time_idx, self.entity_idx]).sort_index()
        else:
            df_preds = pd.DataFrame(columns=[f"pred_{t}" for t in self.targets])
            df_preds.index = pd.MultiIndex.from_arrays(
                [[] for _ in range(2)], names=[self.time_idx, self.entity_idx]
            )
        
        pred_cols = [f"pred_{t}" for t in self.targets]
        return df_preds[pred_cols]


class DriftModel:
    def __init__(self, targets: List[str], months: int, partition_dict: dict, loa: str):
        """
        Baseline model that extrapolates a linear trend from the last N months.
        
        The model fits a linear regression on the last N months for each entity
        and extrapolates that trend into the forecast horizon.
        """
        self.targets = targets
        self.months = months
        self.partition_dict = partition_dict
        self.loa = loa
        self.slopes = None
        self.intercepts = None
        self.time_idx = None
        self.entity_idx = None

    def fit(self, df: pd.DataFrame):
        """
        Compute linear trend (slope and intercept) from last N months for each entity.
        """
        test_start, _ = self.partition_dict["test"]
        self.time_idx = df.index.names[0]
        self.entity_idx = df.index.names[1]
        
        df_train = df[df.index.get_level_values(self.time_idx) < test_start]
        df_train = df_train.sort_index(level=[self.entity_idx, self.time_idx])
        
        logger.info(f"Fitting DriftModel on level: {self.entity_idx}")
        
        self.slopes = {}
        self.intercepts = {}
        self.last_time = {}
        
        for entity_id in df_train.index.get_level_values(self.entity_idx).unique():
            entity_data = df_train.xs(entity_id, level=self.entity_idx)
            last_n = entity_data[self.targets].tail(self.months)
            
            if len(last_n) < 2:
                # Not enough data for trend, use zero slope
                self.slopes[entity_id] = {t: 0.0 for t in self.targets}
                self.intercepts[entity_id] = {t: last_n[t].iloc[-1] if len(last_n) > 0 else 0.0 for t in self.targets}
                self.last_time[entity_id] = entity_data.index[-1] if len(entity_data) > 0 else test_start - 1
                continue
            
            # Time indices relative to first point in window
            x = np.arange(len(last_n))
            self.last_time[entity_id] = entity_data.index[-1]
            
            slopes = {}
            intercepts = {}
            for t in self.targets:
                y = last_n[t].values
                # Simple linear regression: y = mx + b
                if np.std(x) > 0:
                    slope = np.cov(x, y)[0, 1] / np.var(x)
                    intercept = np.mean(y) - slope * np.mean(x)
                    # Extrapolate from end of training data
                    intercepts[t] = intercept + slope * (len(last_n) - 1)
                else:
                    slope = 0.0
                    intercepts[t] = y[-1]
                slopes[t] = slope
            
            self.slopes[entity_id] = slopes
            self.intercepts[entity_id] = intercepts
        
        return self

    def predict(
        self,
        df: pd.DataFrame,
        sequence_number: int,
        output_length: int = 36,
    ) -> pd.DataFrame:
        """
        Extrapolate the linear trend into the forecast horizon.
        """
        test_start, _ = self.partition_dict["test"]
        prediction_start = test_start + sequence_number
        prediction_end = prediction_start + output_length

        logger.info(f"Generating Drift predictions on level: {self.entity_idx}")

        train_end = test_start - 1
        loa_ids = (
            df.loc[df.index.get_level_values(self.time_idx) == train_end]
            .index.get_level_values(self.entity_idx)
            .unique()
        )

        time_ids = list(range(prediction_start, prediction_end))

        records = []
        for cid in loa_ids:
            if cid not in self.slopes:
                logger.warning(f"No trend found for {self.entity_idx} = {cid}")
                continue
            
            slopes = self.slopes[cid]
            intercepts = self.intercepts[cid]
            
            for i, tid in enumerate(time_ids):
                row = {
                    self.time_idx: tid,
                    self.entity_idx: cid,
                }
                # Extrapolate: value = intercept + slope * steps_ahead
                # Clamp to 0 to avoid negative predictions (breaks RMSLE)
                for t in self.targets:
                    pred_val = intercepts[t] + slopes[t] * (i + 1)
                    row[f"pred_{t}"] = max(0.0, pred_val)
                records.append(row)

        df_preds = pd.DataFrame(records)
        if not df_preds.empty:
            df_preds = df_preds.set_index([self.time_idx, self.entity_idx]).sort_index()
        else:
            df_preds = pd.DataFrame(columns=[f"pred_{t}" for t in self.targets])
            df_preds.index = pd.MultiIndex.from_arrays(
                [[] for _ in range(2)], names=[self.time_idx, self.entity_idx]
            )

        pred_cols = [f"pred_{t}" for t in self.targets]
        return df_preds[pred_cols]


class HistoricalAverageModel:
    def __init__(self, targets: List[str], partition_dict: dict, loa: str):
        """
        Baseline model that predicts the all-time historical average for each entity.
        
        Tests whether a model beats simply knowing each entity's baseline conflict level.
        """
        self.targets = targets
        self.partition_dict = partition_dict
        self.loa = loa
        self.historical_means = None
        self.time_idx = None
        self.entity_idx = None

    def fit(self, df: pd.DataFrame):
        """
        Compute the all-time mean for each entity across all training data.
        """
        test_start, _ = self.partition_dict["test"]
        self.time_idx = df.index.names[0]
        self.entity_idx = df.index.names[1]
        
        df_train = df[df.index.get_level_values(self.time_idx) < test_start]
        
        logger.info(f"Fitting HistoricalAverageModel on level: {self.entity_idx}")
        
        self.historical_means = df_train.groupby(self.entity_idx)[self.targets].mean()
        
        return self

    def predict(
        self,
        df: pd.DataFrame,
        sequence_number: int,
        output_length: int = 36,
    ) -> pd.DataFrame:
        """
        Predict the historical average for each entity over the forecast horizon.
        """
        test_start, _ = self.partition_dict["test"]
        prediction_start = test_start + sequence_number
        prediction_end = prediction_start + output_length

        logger.info(f"Generating HistoricalAverage predictions on level: {self.entity_idx}")

        train_end = test_start - 1
        loa_ids = (
            df.loc[df.index.get_level_values(self.time_idx) == train_end]
            .index.get_level_values(self.entity_idx)
            .unique()
        )

        time_ids = list(range(prediction_start, prediction_end))

        records = []
        for cid in loa_ids:
            if cid not in self.historical_means.index:
                logger.warning(f"No historical data for {self.entity_idx} = {cid}")
                continue
            
            means = self.historical_means.loc[cid]
            
            for tid in time_ids:
                row = {
                    self.time_idx: tid,
                    self.entity_idx: cid,
                }
                row.update({f"pred_{t}": means[t] for t in self.targets})
                records.append(row)

        df_preds = pd.DataFrame(records)
        if not df_preds.empty:
            df_preds = df_preds.set_index([self.time_idx, self.entity_idx]).sort_index()
        else:
            df_preds = pd.DataFrame(columns=[f"pred_{t}" for t in self.targets])
            df_preds.index = pd.MultiIndex.from_arrays(
                [[] for _ in range(2)], names=[self.time_idx, self.entity_idx]
            )

        pred_cols = [f"pred_{t}" for t in self.targets]
        return df_preds[pred_cols]


class ClimatologyModel:
    def __init__(self, targets: List[str], partition_dict: dict, loa: str):
        """
        Baseline model that predicts the historical average for each (entity, month-of-year).
        
        Captures both entity baseline AND seasonality patterns. Month-of-year is computed
        from month_id using: month_of_year = ((month_id - 1) % 12) + 1
        """
        self.targets = targets
        self.partition_dict = partition_dict
        self.loa = loa
        self.climatology = None
        self.entity_fallback = None  # Fallback to entity mean if month not seen
        self.time_idx = None
        self.entity_idx = None

    def _month_of_year(self, month_id: int) -> int:
        """Convert month_id to month-of-year (1-12)."""
        return ((month_id - 1) % 12) + 1

    def fit(self, df: pd.DataFrame):
        """
        Compute the mean for each (entity, month-of-year) combination.
        """
        test_start, _ = self.partition_dict["test"]
        self.time_idx = df.index.names[0]
        self.entity_idx = df.index.names[1]
        
        df_train = df[df.index.get_level_values(self.time_idx) < test_start].copy()
        df_train = df_train.reset_index()
        
        # Add month-of-year column
        df_train["month_of_year"] = df_train[self.time_idx].apply(self._month_of_year)
        
        logger.info(f"Fitting ClimatologyModel on level: {self.entity_idx}")
        
        # Compute climatology: mean by (entity, month_of_year)
        self.climatology = df_train.groupby([self.entity_idx, "month_of_year"])[self.targets].mean()
        
        # Fallback: entity mean (for months not seen in training)
        self.entity_fallback = df_train.groupby(self.entity_idx)[self.targets].mean()
        
        return self

    def predict(
        self,
        df: pd.DataFrame,
        sequence_number: int,
        output_length: int = 36,
    ) -> pd.DataFrame:
        """
        Predict the climatological average for each (entity, month-of-year).
        """
        test_start, _ = self.partition_dict["test"]
        prediction_start = test_start + sequence_number
        prediction_end = prediction_start + output_length

        logger.info(f"Generating Climatology predictions on level: {self.entity_idx}")

        train_end = test_start - 1
        loa_ids = (
            df.loc[df.index.get_level_values(self.time_idx) == train_end]
            .index.get_level_values(self.entity_idx)
            .unique()
        )

        time_ids = list(range(prediction_start, prediction_end))

        records = []
        for cid in loa_ids:
            for tid in time_ids:
                moy = self._month_of_year(tid)
                
                row = {
                    self.time_idx: tid,
                    self.entity_idx: cid,
                }
                
                # Try climatology first, fall back to entity mean
                if (cid, moy) in self.climatology.index:
                    vals = self.climatology.loc[(cid, moy)]
                    row.update({f"pred_{t}": vals[t] for t in self.targets})
                elif cid in self.entity_fallback.index:
                    vals = self.entity_fallback.loc[cid]
                    row.update({f"pred_{t}": vals[t] for t in self.targets})
                else:
                    logger.warning(f"No climatology data for {self.entity_idx} = {cid}")
                    row.update({f"pred_{t}": 0.0 for t in self.targets})
                
                records.append(row)

        df_preds = pd.DataFrame(records)
        if not df_preds.empty:
            df_preds = df_preds.set_index([self.time_idx, self.entity_idx]).sort_index()
        else:
            df_preds = pd.DataFrame(columns=[f"pred_{t}" for t in self.targets])
            df_preds.index = pd.MultiIndex.from_arrays(
                [[] for _ in range(2)], names=[self.time_idx, self.entity_idx]
            )

        pred_cols = [f"pred_{t}" for t in self.targets]
        return df_preds[pred_cols]