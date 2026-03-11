from views_pipeline_core.managers.model import ModelPathManager, ForecastingModelManager
from views_pipeline_core.files.utils import read_dataframe
from views_pipeline_core.configs.pipeline import PipelineConfig
import logging
import pandas as pd
from datetime import datetime
from views_baseline.model.catalog import BaselineModelCatalog
from views_baseline.model.protocol import DistributionalBaselineModel


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

        logger.info("Initializing BaselineModelManager")

    def _train_model_artifact(self):
        """
        Train and save your model artifact.

        """

        logger.warning("Baseline Models does not require training - skipping training")

    def _setup_model_and_data(self):
        """
        Instantiate the baseline model via the catalog, load data, fit, and return both.
        """
        path_raw = self._model_path.data_raw
        run_type = self.config["run_type"]
        loa = self.config["level"]
        partition_dict = self._data_loader.partition_dict
        catalog = BaselineModelCatalog(
            config=self.config, partition_dict=partition_dict, loa=loa
        )
        model = catalog.get_model(self.config["algorithm"])
        logger.info(f"Model type is {self.config['algorithm']}")
        self.config["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        df = read_dataframe(
            path_raw / f"{run_type}_viewser_df{PipelineConfig.dataframe_format}"
        )
        model.fit(df)
        return model, df

    def _evaluate_model_artifact(
        self, eval_type: str, artifact_name: str = None
    ) -> list:
        """
        Evaluate trained model artifact.

        """
        logger.info("Evaluating baseline model artifact")

        self.model, df_viewser = self._setup_model_and_data()

        logger.info(f"Generating predictions for {eval_type} evaluation")

        sequence_numbers = self._resolve_evaluation_sequence_number(eval_type)

        if self._prediction_format == "prediction_frame" and isinstance(self.model, DistributionalBaselineModel):
            predictions = {}
            for seq_num in range(sequence_numbers):
                pf_dict = self.model.predict_prediction_frame(df=df_viewser, sequence_number=seq_num)
                for target, pf in pf_dict.items():
                    predictions.setdefault(target, []).append(pf)
            return predictions

        predictions = []
        for seq_num in range(sequence_numbers):
            preds = self.model.predict(df=df_viewser, sequence_number=seq_num)
            predictions.append(preds)

        return predictions

    def _forecast_model_artifact(self, artifact_name: str = None) -> pd.DataFrame:
        """
        Generate forecasts using trained model artifact.

        """
        logger.info("Generating forecasts")

        self.model, df_viewser = self._setup_model_and_data()

        if self._prediction_format == "prediction_frame" and isinstance(self.model, DistributionalBaselineModel):
            return self.model.predict_prediction_frame(df=df_viewser, sequence_number=0)

        return self.model.predict(sequence_number=0, df=df_viewser)

    def _evaluate_sweep(self, eval_type: str, model: any) -> list:

        logger.info(
            "Baseline Models does not support sweep evaluation - skipping evaluation"
        )
        raise NotImplementedError(
            "Baseline Models does not support sweep evaluation - skipping evaluation"
        )
