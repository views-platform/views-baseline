import logging
from typing import List

import numpy as np
import pandas as pd

from views_baseline.model.helpers import build_prediction_grid

logger = logging.getLogger(__name__)


class ZeroModel:
    def __init__(self, targets: List[str], partition_dict: dict, loa: str):
        """
        Baseline model that predicts 0 for all targets.
        """
        self.targets = targets
        self.partition_dict = partition_dict
        self.loa = loa
        self.time_idx = None
        self.entity_idx = None

    def fit(self, df: pd.DataFrame):
        self.time_idx = df.index.names[0]
        self.entity_idx = df.index.names[1]
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
        test_start = self.partition_dict["test"][0]
        train_end = test_start - 1

        logger.info(f"Currently running a {self.entity_idx} model")

        loa_ids = (
            df.loc[df.index.get_level_values(self.time_idx) == train_end]
            .index.get_level_values(self.entity_idx)
            .unique()
        )
        prediction_start = test_start + sequence_number
        time_ids = list(range(prediction_start, prediction_start + output_length))

        return build_prediction_grid(
            time_idx=self.time_idx,
            entity_idx=self.entity_idx,
            loa_ids=loa_ids,
            time_ids=time_ids,
            targets=self.targets,
            value_fn=lambda cid, target: 0.0,
        )


class LocfModel:
    def __init__(self, targets: List[str], partition_dict: dict, loa: str):
        """
        Baseline model that carries forward the last observation for each entity and target.
        """
        self.targets = targets
        self.partition_dict = partition_dict
        self.loa = loa
        self.last_observations = None
        self.time_idx = None
        self.entity_idx = None

    def fit(self, df: pd.DataFrame):
        """
        Store the last available observation before the test period for each entity.
        """
        test_start = self.partition_dict["test"][0]
        self.time_idx = df.index.names[0]
        self.entity_idx = df.index.names[1]
        df = df[df.index.get_level_values(self.time_idx) < test_start]

        logger.info(f"Fitting LastObservationModel on level: {self.entity_idx}")
        df = df.sort_index(level=[self.entity_idx, self.time_idx])
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
        test_start = self.partition_dict["test"][0]
        train_end = test_start - 1

        logger.info(f"Generating LOCF predictions on level: {self.entity_idx}")

        loa_ids = (
            df.loc[df.index.get_level_values(self.time_idx) == train_end]
            .index.get_level_values(self.entity_idx)
            .unique()
        )
        # Filter to entities that have stored observations
        n_before = len(loa_ids)
        loa_ids = [cid for cid in loa_ids if cid in self.last_observations.index]
        if len(loa_ids) < n_before:
            logger.warning(
                f"LocfModel: {n_before - len(loa_ids)} entities dropped"
                " (missing from last_observations)"
            )

        prediction_start = test_start + sequence_number
        time_ids = list(range(prediction_start, prediction_start + output_length))

        return build_prediction_grid(
            time_idx=self.time_idx,
            entity_idx=self.entity_idx,
            loa_ids=loa_ids,
            time_ids=time_ids,
            targets=self.targets,
            value_fn=lambda cid, target: self.last_observations.loc[cid, target],
        )


class AverageModel:
    def __init__(self, targets: List[str], window_months: int, partition_dict: dict, loa: str):
        """
        Baseline model that carries forward the average of the last window_months months
        specified in the config file.
        """
        self.targets = targets
        self.partition_dict = partition_dict
        self.loa = loa
        self.mean = None
        self.window_months = window_months
        self.time_idx = None
        self.entity_idx = None

    def fit(self, df: pd.DataFrame):
        """
        Get the average of the last m observations before the test partition for each entity.
        """
        test_start = self.partition_dict["test"][0]
        self.time_idx = df.index.names[0]
        self.entity_idx = df.index.names[1]
        df = df[df.index.get_level_values(self.time_idx) < test_start]

        logger.info(f"Fitting AverageModel on level: {self.entity_idx}")

        df = df.sort_index(level=[self.entity_idx, self.time_idx])

        # Group by entity and take mean of last `months` rows
        entity_means = df.groupby(level=self.entity_idx, group_keys=False).apply(
            lambda g: g.tail(self.window_months)[self.targets].mean()
        )
        self.mean = entity_means
        return self

    def predict(
        self,
        df: pd.DataFrame,
        sequence_number: int,
        output_length: int = 36,
    ) -> pd.DataFrame:
        """
        Repeats the average over the last m months for each target
        and entity over the forecast horizon.
        """
        test_start = self.partition_dict["test"][0]
        train_end = test_start - 1

        logger.info(f"Generating average predictions on level: {self.entity_idx}")

        loa_ids = (
            df.loc[df.index.get_level_values(self.time_idx) == train_end]
            .index.get_level_values(self.entity_idx)
            .unique()
        )
        n_before = len(loa_ids)
        loa_ids = [cid for cid in loa_ids if cid in self.mean.index]
        if len(loa_ids) < n_before:
            logger.warning(
                f"AverageModel: {n_before - len(loa_ids)} entities dropped (missing from mean)"
            )

        prediction_start = test_start + sequence_number
        time_ids = list(range(prediction_start, prediction_start + output_length))

        return build_prediction_grid(
            time_idx=self.time_idx,
            entity_idx=self.entity_idx,
            loa_ids=loa_ids,
            time_ids=time_ids,
            targets=self.targets,
            value_fn=lambda cid, target: self.mean.loc[cid, target],
        )


class ConflictologyModel:
    distributional = True

    def __init__(
        self,
        targets: List[str],
        window_months: int,
        partition_dict: dict,
        loa: str,
        n_samples: int = 256,
        seed: int = 42,
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
        self.loa_ids = None

    def fit(self, df: pd.DataFrame):
        """
        Extract and store the last `window_months` of history per entity before the test period.
        """
        test_start = self.partition_dict["test"][0]
        self.time_idx = df.index.names[0]
        self.entity_idx = df.index.names[1]

        train_end = test_start - 1

        df = df[df.index.get_level_values(self.time_idx) <= train_end]
        df = df.sort_index(level=[self.entity_idx, self.time_idx])

        last_n_months = df.groupby(level=self.entity_idx, group_keys=False).apply(
            lambda g: g.tail(self.window_months)
        )

        self.loa_ids = (
            df.loc[df.index.get_level_values(self.time_idx) == train_end]
            .index.get_level_values(self.entity_idx)
            .unique()
        )

        self.hist_per_entity = {}
        for cid in self.loa_ids:
            history = last_n_months.xs(cid, level=self.entity_idx, drop_level=False)
            if history.empty:
                continue
            self.hist_per_entity[cid] = {
                t: np.array(history[t].tolist(), dtype=np.float64) for t in self.targets
            }

        return self

    def predict(
        self, df: pd.DataFrame, sequence_number: int, output_length: int = 36
    ) -> dict:
        """
        Return predictions as Dict[str, PredictionFrame] — one PF per target.
        Each PF has y_pred shape (N, n_samples) with resampled draws.
        """
        from views_pipeline_core.data.prediction_frame import PredictionFrame

        test_start = self.partition_dict["test"][0]
        prediction_start = test_start + sequence_number
        time_ids = list(range(prediction_start, prediction_start + output_length))

        entities_with_history = [cid for cid in self.loa_ids if cid in self.hist_per_entity]
        if len(entities_with_history) < len(self.loa_ids):
            logger.warning(
                f"ConflictologyModel: {len(self.loa_ids) - len(entities_with_history)} "
                "entities dropped (missing from hist_per_entity)"
            )

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

        rng = np.random.default_rng(self.seed)

        # Iterate entity→time→target for consistent RNG ordering
        y_preds = {t: np.empty((n_rows, self.n_samples), dtype=np.float64) for t in self.targets}
        idx = 0
        for cid in entities_with_history:
            for _ in time_ids:
                for t in self.targets:
                    y_preds[t][idx] = rng.choice(
                        self.hist_per_entity[cid][t], size=self.n_samples, replace=True
                    )
                idx += 1

        result = {}
        for t in self.targets:
            result[t] = PredictionFrame(
                y_pred=y_preds[t],
                identifiers={"time": time_arr.copy(), "unit": unit_arr.copy()},
            )

        return result


class MixtureBaseline:
    distributional = True

    def __init__(
        self,
        targets: List[str],
        window_months: int,
        lambda_mix: float,
        n_samples: int,
        partition_dict: dict,
        loa: str,
        seed: int = 42,
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
        self.loa_ids = None

    def fit(self, df: pd.DataFrame):
        test_start = self.partition_dict["test"][0]
        self.time_idx = df.index.names[0]
        self.entity_idx = df.index.names[1]

        train_df = df[df.index.get_level_values(self.time_idx) < test_start]
        train_df = train_df.sort_index(level=[self.entity_idx, self.time_idx])

        train_end = test_start - 1
        self.loa_ids = (
            train_df.loc[train_df.index.get_level_values(self.time_idx) == train_end]
            .index.get_level_values(self.entity_idx)
            .unique()
        )

        # Local pool: last window_months values per entity per target
        self.local_pool = {}
        for cid in self.loa_ids:
            entity_data = train_df.xs(cid, level=self.entity_idx, drop_level=False)
            tail = entity_data.tail(self.window_months)
            self.local_pool[cid] = {
                t: np.array(tail[t].tolist(), dtype=np.float64) for t in self.targets
            }

        # Global pool: all positive values per target across all entities
        self.global_pool = {}
        for t in self.targets:
            vals = train_df[t].values
            self.global_pool[t] = vals[vals > 0].astype(np.float64)

        return self

    def _sample(self, cid, target, rng):
        """Generate n_samples by mixing local and global pools."""
        local = self.local_pool[cid][target]
        glob = self.global_pool[target]

        if len(glob) == 0:
            # No positive values in training data — local only
            return rng.choice(local, size=self.n_samples)

        use_global = rng.random(self.n_samples) < self.lambda_mix
        n_global = int(np.sum(use_global))
        n_local = self.n_samples - n_global

        samples = np.empty(self.n_samples, dtype=np.float64)
        samples[use_global] = rng.choice(glob, size=n_global)
        samples[~use_global] = rng.choice(local, size=n_local)
        return samples

    def predict(
        self, df: pd.DataFrame, sequence_number: int, output_length: int = 36
    ) -> dict:
        from views_pipeline_core.data.prediction_frame import PredictionFrame

        test_start = self.partition_dict["test"][0]
        prediction_start = test_start + sequence_number
        time_ids = list(range(prediction_start, prediction_start + output_length))

        entities_with_pool = [cid for cid in self.loa_ids if cid in self.local_pool]
        if len(entities_with_pool) < len(self.loa_ids):
            logger.warning(
                f"MixtureBaseline: {len(self.loa_ids) - len(entities_with_pool)} "
                "entities dropped (missing from local_pool)"
            )

        if not entities_with_pool:
            return {}

        time_arr = []
        unit_arr = []
        for cid in entities_with_pool:
            for tid in time_ids:
                time_arr.append(tid)
                unit_arr.append(cid)

        time_arr = np.array(time_arr)
        unit_arr = np.array(unit_arr)
        n_rows = len(time_arr)

        rng = np.random.default_rng(self.seed)

        # Iterate entity→time→target for consistent RNG ordering
        y_preds = {t: np.empty((n_rows, self.n_samples), dtype=np.float64) for t in self.targets}
        idx = 0
        for cid in entities_with_pool:
            for _ in time_ids:
                for t in self.targets:
                    y_preds[t][idx] = self._sample(cid, t, rng)
                idx += 1

        result = {}
        for t in self.targets:
            result[t] = PredictionFrame(
                y_pred=y_preds[t],
                identifiers={"time": time_arr.copy(), "unit": unit_arr.copy()},
            )

        return result
