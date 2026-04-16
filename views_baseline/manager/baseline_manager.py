from views_pipeline_core.managers.model import ModelPathManager, ForecastingModelManager
from views_pipeline_core.files.utils import read_dataframe
from views_pipeline_core.configs.pipeline import PipelineConfig
import logging
import pickle
import re
from pathlib import Path
from views_baseline.model.baseline import ZeroModel
from views_baseline.model.baseline import LocfModel
from views_pipeline_core.files.utils import read_dataframe, generate_model_file_name
import pandas as pd
from datetime import datetime
from views_baseline.model.catalog import BaselineModelCatalog


logger = logging.getLogger(__name__)


class BaselineForecastingModelManager(ForecastingModelManager):
    """
    Baseline Forecasting Model Manager

    """

    def __init__(
        self,
        model_path: ModelPathManager,
        wandb_notifications: bool = False,
        use_prediction_store: bool = False,
    ) -> None:
        """
        Initialize Baseline Forecasting Model Manager.
        """
        super().__init__(model_path, wandb_notifications, use_prediction_store)

        # Add your custom initialization below
        logger.info("Initializing BaselineModelManager")

        # YOUR CODE HERE

    def _timestamp_from_artifact_name(self, artifact_name: str) -> str:
        # accepts "calibration_model_20251211_101925.pt" or full path
        stem = Path(artifact_name).stem  # calibration_model_20251211_101925
        ts = stem[-15:]  # 20251211_101925
        if not re.match(r"^\d{8}_\d{6}$", ts):
            raise ValueError(f"Cannot parse timestamp from artifact '{artifact_name}'")
        return ts

    def _resolve_artifact_path(self, run_type: str, artifact_name: str | None) -> Path:
        path_artifacts = self._model_path.artifacts
        if artifact_name:
            logger.info(f"Using (non-default) artifact: {artifact_name}")
            return path_artifacts / artifact_name
        logger.info(f"Using latest (default) run type ({run_type}) specific artifact")
        return self._model_path.get_latest_model_artifact_path(run_type)

    def _train_model_artifact(self) -> any:
        """
        Train and save your model artifact.

        """

        # Common paths and data loading (provided)
        path_raw = self._model_path.data_raw
        path_artifacts = self._model_path.artifacts
        logger.info(f"The Path artifacts is: {path_artifacts}")
        run_type = self.configs[
            "run_type"
        ]  # "calibration", "validation", "forecasting"
        loa = self.configs["level"]
        logger.info(f"Level of Analysis {loa}")

        # 2. Model initialization
        logger.warning("Baseline Models does not require training - skipping training")

        # Optional: instantiate the baseline model anyway (useful for consistency/debugging)
        # self.model = ZeroModel(
        #     targets=self.config["targets"],
        #     partition_dict=partition_dict,
        #     loa=loa,
        # )

        # 4. Save artifact (if not in sweep)
        if not self.configs["sweep"]:
            # IMPORTANT: match the pipeline naming convention used elsewhere:
            # e.g. calibration_model_20251211_101925.pt
            model_filename = generate_model_file_name(run_type, file_extension=".pkl")
            logger.info(f"Saving baseline artifact as {model_filename}")
            path_artifact = path_artifacts / model_filename
            logger.info(f"Saving baseline artifact to {path_artifact}")
            logger.info(f"Baseline artifact path stem: {path_artifact.stem[-15:]}")
            logger.info(
                f"CONFIGS obj id BEFORE set: {id(self.configs)} ts_before={self.configs.get('timestamp')}"
            )

            # self.configs["timestamp"] = path_artifact.stem[-15:]
            self.configs = {
                "timestamp": path_artifact.stem[-15:],
                "artifact_name": path_artifact.name,
            }
            logger.info(
                f"CONFIGS obj id AFTER  set: {id(self.configs)} ts_after={self.configs.get('timestamp')}"
            )
            self.configs["artifact_name"] = path_artifact.name
            logger.info(f"Using timestamp={self.configs['timestamp']} for artifact")

            payload = {
                "artifact_type": "baseline_marker",
                "algorithm": self.configs.get("algorithm"),
                "targets": self.configs.get("targets"),
                "level": self.configs.get("level"),
                "run_type": run_type,
                "timestamp": path_artifact.stem[-15:],
            }

            with open(path_artifact, "wb") as f:
                pickle.dump(payload, f)

            logger.info(f"Saved baseline artifact: {path_artifact.name}")

        return None  # Return trained model for sweep evaluation
        # --- USER IMPLEMENTATION ENDS HERE ---

    def _evaluate_model_artifact(
        self, eval_type: str, artifact_name: str = None
    ) -> list:
        """
        Evaluate trained model artifact.

        """
        logger.info("Evaluating baseline model artifact")
        logger.info(f"Evaluation type: {eval_type}, Artifact name: {artifact_name}")
        # Common setup (provided)
        path_raw = self._model_path.data_raw
        path_artifacts = self._model_path.artifacts
        run_type = self.configs["run_type"]
        loa = self.configs["level"]
        partition_dict = self._data_loader.partition_dict
        catalog = BaselineModelCatalog(
            config=self.configs, partition_dict=partition_dict, loa=loa
        )
        model_name = self.configs["algorithm"]
        self.model = catalog.get_model(model_name)

        logger.info(f"Model type is {model_name}")

        if artifact_name:
            if not artifact_name.endswith(".pt"):
                artifact_name += ".pt"
            path_artifact = path_artifacts / artifact_name
        else:
            path_artifact = self._model_path.get_latest_model_artifact_path(run_type)

        ts = path_artifact.stem[-15:]
        self.configs = {
            "timestamp": ts,
            "artifact_name": path_artifact.name,
        }
        logger.info(f"Artifact used: {path_artifact.name}")
        logger.info(f"Using timestamp={ts}")
        df_viewser = read_dataframe(
            path_raw / f"{run_type}_viewser_df{PipelineConfig.dataframe_format}"
        )

        self.model.fit(df_viewser)

        logger.info(f"Generating predictions for {eval_type} evaluation")
        predictions = []

        # Determine evaluation length
        sequence_numbers = self._resolve_evaluation_sequence_number(eval_type)
        for seq_num in range(sequence_numbers):
            # YOUR PREDICTION CODE HERE
            preds = self.model.predict(df=df_viewser, sequence_number=seq_num)
            # preds = preds.clip(lower=1e-4)
            predictions.append(preds)  # Append predictions for each sequence

        return predictions

    def _forecast_model_artifact(self, artifact_name: str = None) -> pd.DataFrame:
        """
        Generate forecasts using trained model artifact.

        """
        # Common setup (provided)
        path_raw = self._model_path.data_raw
        path_artifacts = self._model_path.artifacts
        run_type = self.configs["run_type"]
        loa = self.configs["level"]
        partition_dict = self._data_loader.partition_dict
        catalog = BaselineModelCatalog(
            config=self.configs, partition_dict=partition_dict, loa=loa
        )
        model_name = self.configs["algorithm"]  # e.g., "ZeroModel" or "LocfModel"
        self.model = catalog.get_model(model_name)
        logger.info(f"Model type is {model_name}")

        if artifact_name:
            if not artifact_name.endswith(".pt"):
                artifact_name += ".pt"
            path_artifact = path_artifacts / artifact_name
        else:
            path_artifact = self._model_path.get_latest_model_artifact_path(run_type)

        ts = path_artifact.stem[-15:]
        self.configs = {
            "timestamp": ts,
            "artifact_name": path_artifact.name,
        }
        logger.info(f"Artifact used: {path_artifact.name}")
        logger.info(f"Using timestamp={ts}")
        df_viewser = read_dataframe(
            path_raw / f"{run_type}_viewser_df{PipelineConfig.dataframe_format}"
        )

        logger.info("Generating forecasts")

        self.model.fit(df_viewser)

        forecasts = self.model.predict(sequence_number=0, df=df_viewser)

        return forecasts

    def _evaluate_sweep(self, eval_type: str, model: any) -> list:

        logger.info(
            f"Baseline Models does not support sweep evaluation - skipping evaluation"
        )
        raise NotImplementedError(
            "Baseline Models does not support sweep evaluation - skipping evaluation"
        )


    def _evaluate_prediction_dataframe(
        self, df_predictions, eval_type, ensemble=False
    ) -> None:

        import pandas as pd
        from views_evaluation.evaluation.evaluation_manager import EvaluationManager
        from views_pipeline_core.files.utils import read_dataframe

        evaluation_manager = EvaluationManager(self.config["metrics"])

        df_path = self._model_path._get_raw_data_file_paths(
            run_type=self.args.run_type
        )[0]

        df_viewser = read_dataframe(df_path)
        df_actual = df_viewser[self.config["targets"]]

        for target in self.config["targets"]:
            logger.info(f"Calculating evaluation metrics for {target}")

            eval_result_dict = evaluation_manager.evaluate(
                df_actual,
                df_predictions,
                target,
                self.config,
            )

            step_eval, df_step = eval_result_dict["step"]
            ts_eval, df_ts = eval_result_dict["time_series"]
            month_eval, df_month = eval_result_dict["month"]

            self._wandb_module.log_evaluation_results(
                step_eval,
                month_eval,
                ts_eval,
                "",   
            )