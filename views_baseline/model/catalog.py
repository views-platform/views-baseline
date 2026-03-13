from views_baseline.model.baseline import (
    AverageModel,
    ConflictologyModel,
    LocfModel,
    MixtureBaseline,
    ZeroModel,
)


class BaselineModelCatalog:
    # Required config keys per model (beyond the universal "targets").
    # Keys with .get() defaults in _get_* methods are optional and not listed here.
    MODEL_GENOMES = {
        "ZeroModel": [],
        "LocfModel": [],
        "AverageModel": ["months"],
        "ConflictologyModel": ["months"],
        "MixtureBaseline": [],
    }

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
            "AverageModel": self._get_average_model,
            "ConflictologyModel": self._get_conflictology_model,
            "MixtureBaseline": self._get_mixture_model,
        }

    def get_model(self, model_name: str):
        """
        Returns an initialized model instance.
        Validates that required config keys for the model are present.
        """
        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' is not in the catalog. "
                f"Available: {self.list_models()}"
            )
        missing = [k for k in self.MODEL_GENOMES[model_name] if k not in self.config]
        if missing:
            raise ValueError(
                f"Model '{model_name}' requires config keys {missing} "
                f"but they are missing"
            )
        return self.models[model_name]()

    def list_models(self):
        return list(self.models.keys())

    def _get_zero_model(self):
        return ZeroModel(
            targets=self.config["targets"], partition_dict=self.partition_dict, loa=self.loa
        )

    def _get_locf_model(self):
        return LocfModel(
            targets=self.config["targets"], partition_dict=self.partition_dict, loa=self.loa
        )

    def _get_average_model(self):
        return AverageModel(
            targets=self.config["targets"],
            window_months=self.config["months"],
            partition_dict=self.partition_dict,
            loa=self.loa,
        )

    def _get_conflictology_model(self):
        return ConflictologyModel(
            targets=self.config["targets"],
            window_months=self.config["months"],
            partition_dict=self.partition_dict,
            loa=self.loa,
            n_samples=self.config.get("n_samples", 256),
        )

    def _get_mixture_model(self):
        return MixtureBaseline(
            targets=self.config["targets"],
            window_months=self.config.get("window_months", 18),
            lambda_mix=self.config.get("lambda_mix", 0.05),
            n_samples=self.config.get("n_samples", 256),
            partition_dict=self.partition_dict,
            loa=self.loa,
        )
