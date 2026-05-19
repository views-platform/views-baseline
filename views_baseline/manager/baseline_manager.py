import logging
import pickle

from views_pipeline_core.files.utils import generate_model_file_name, read_dataframe
from views_pipeline_core.managers.model import ForecastingModelManager, ModelPathManager

from views_baseline.infrastructure.reproducibility_gate import ReproducibilityGate
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
        Fit the baseline model and save it as a pickle artifact.

        Although baselines are stateless and deterministic, the downstream
        ensemble manager requires an artifact file to exist so it can
        resolve timestamps via get_latest_model_artifact_path().
        """
        self.model, _ = self._setup_model_and_data()
        path_artifacts = self._model_path.artifacts
        run_type = self.config["run_type"]
        model_filename = generate_model_file_name(run_type, file_extension=".pkl")
        with open(path_artifacts / model_filename, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Saved baseline artifact: {model_filename}")
        return self.model

    def _setup_model_and_data(self):
        """
        Instantiate the baseline model via the catalog, load data, fit, and return both.
        """
        ReproducibilityGate.Config.audit_manifest(self.config)
        loa = self.config["level"]
        partition_dict = self._data_loader.partition_dict
        catalog = BaselineModelCatalog(
            config=self.config, partition_dict=partition_dict, loa=loa
        )
        model = catalog.get_model(self.config["algorithm"])
        logger.info(f"Model type is {self.config['algorithm']}")
        df_source = read_dataframe(self._get_cached_data_path())
        model.fit(df_source)
        return model, df_source

    def _generate_predictions(self, model, df, eval_type):
        """
        Generate predictions for all sequence numbers in an evaluation.

        Dispatches to the appropriate predict method based on model type:
        distributional models return Dict[str, list[PredictionFrame]],
        point models return list[DataFrame].
        """
        sequence_numbers = self._resolve_evaluation_sequence_number(eval_type)
        output_length = self.config["time_steps"]

        if isinstance(model, DistributionalBaselineModel):
            predictions = {}
            for seq_num in range(sequence_numbers):
                pf_dict = model.predict(
                    df=df, sequence_number=seq_num, output_length=output_length
                )
                for target, pf in pf_dict.items():
                    predictions.setdefault(target, []).append(pf)
            return predictions

        predictions = []
        for seq_num in range(sequence_numbers):
            preds = model.predict(
                df=df, sequence_number=seq_num, output_length=output_length
            )
            predictions.append(preds)
        return predictions

    def _evaluate_model_artifact(self, eval_type: str, artifact_name: str = None):
        """
        Evaluate trained model artifact.
        """
        logger.info("Evaluating baseline model artifact")
        if artifact_name:
            path_artifact = self._model_path.artifacts / artifact_name
        else:
            path_artifact = self._model_path.get_latest_model_artifact_path(
                run_type=self.config["run_type"]
            )
        self._config_manager.add_config({"timestamp": path_artifact.stem[-15:]})
        self.model, df_source = self._setup_model_and_data()
        logger.info(f"Generating predictions for {eval_type} evaluation")
        return self._generate_predictions(self.model, df_source, eval_type)

    def _forecast_model_artifact(self, artifact_name: str = None):
        """
        Generate forecasts using trained model artifact.
        """
        logger.info("Generating forecasts")
        if artifact_name:
            path_artifact = self._model_path.artifacts / artifact_name
        else:
            path_artifact = self._model_path.get_latest_model_artifact_path(
                run_type=self.config["run_type"]
            )
        self._config_manager.add_config({"timestamp": path_artifact.stem[-15:]})
        self.model, df_source = self._setup_model_and_data()
        output_length = self.config["time_steps"]

        if isinstance(self.model, DistributionalBaselineModel):
            return self.model.predict(
                df=df_source, sequence_number=0, output_length=output_length
            )

        return self.model.predict(
            df=df_source, sequence_number=0, output_length=output_length
        )

    def _evaluate_sweep(self, eval_type: str, model):
        """
        Evaluate a baseline model during a WandB sweep iteration.

        The model has already been fitted by _train_model_artifact().
        We load the data and generate predictions using it.
        """
        df_source = read_dataframe(self._get_cached_data_path())
        return self._generate_predictions(model, df_source, eval_type)
