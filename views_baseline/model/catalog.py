from views_baseline.model.baseline import (
    ZeroModel,
    LocfModel,
    AverageModel,
    ConflictologyModel,
    MirrorModel,
    DriftModel,
    HistoricalAverageModel,
    ClimatologyModel,
)


class BaselineModelCatalog:
    def __init__(self, config: dict, partition_dict: dict, loa: str):
        """
        Catalog of available baseline models.
        """
        self.config = config
        self.partition_dict = partition_dict
        self.loa = loa

        self.models = {
            "ZeroModel": self._get_zero_model,
            "LocfModel": self._get_locf_model,
            "AverageModel":self._get_average_model,
            "ConflictologyModel":self._get_conflictology_model,
            "MirrorModel": self._get_mirror_model,
            "DriftModel": self._get_drift_model,
            "HistoricalAverageModel": self._get_historical_average_model,
            "ClimatologyModel": self._get_climatology_model,
        }

    def get_model(self, model_name: str):
        """
        Returns an initialized model instance.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' is not in the catalog. Available: {self.list_models()}")
        return self.models[model_name]()

    def list_models(self):
        return list(self.models.keys())

    def _get_zero_model(self):
        return ZeroModel(
            targets=self.config["targets"],
            partition_dict=self.partition_dict,
            loa=self.loa
        )
    
    def _get_locf_model(self):
        return LocfModel( 
            targets=self.config["targets"],
            partition_dict=self.partition_dict,
            loa=self.loa
        )
    
    def _get_average_model(self):
        return AverageModel( 
            targets=self.config["targets"],
            months = self.config["months"],
            partition_dict=self.partition_dict,
            loa=self.loa
        )

    def _get_conflictology_model(self):
        return ConflictologyModel( 
            targets=self.config["targets"],
            months = self.config["months"],
            partition_dict=self.partition_dict,
            loa=self.loa
        )

    def _get_mirror_model(self):
        return MirrorModel(
            targets=self.config["targets"],
            partition_dict=self.partition_dict,
            loa=self.loa
        )

    def _get_drift_model(self):
        return DriftModel(
            targets=self.config["targets"],
            months=self.config["months"],
            partition_dict=self.partition_dict,
            loa=self.loa
        )

    def _get_historical_average_model(self):
        return HistoricalAverageModel(
            targets=self.config["targets"],
            partition_dict=self.partition_dict,
            loa=self.loa
        )

    def _get_climatology_model(self):
        return ClimatologyModel(
            targets=self.config["targets"],
            partition_dict=self.partition_dict,
            loa=self.loa
        )

