from views_baseline.infrastructure.reproducibility_gate import ReproducibilityGate
from views_baseline.model.baseline import (
    AverageModel,
    ConflictologyModel,
    LocfModel,
    MixtureBaseline,
    ParametricConflictology,
    ParametricHurdleConflictology,
    ZeroModel,
)


class BaselineModelCatalog:
    # Single source of truth lives in ReproducibilityGate.Config.ALGORITHM_GENOMES.
    # This alias keeps the catalog's get_model() validation unchanged.
    MODEL_GENOMES = ReproducibilityGate.Config.ALGORITHM_GENOMES

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
            "ParametricConflictology": self._get_parametric_conflictology,
            "ParametricHurdleConflictology": self._get_parametric_hurdle,
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
            window_months=self.config["window_months"],
            partition_dict=self.partition_dict,
            loa=self.loa,
        )

    def _get_conflictology_model(self):
        return ConflictologyModel(
            targets=self.config["targets"],
            window_months=self.config["window_months"],
            partition_dict=self.partition_dict,
            loa=self.loa,
            n_samples=self.config["n_samples"],
            seed=self.config["seed"],  # required + audited genome key (ADR-021); no silent default
        )

    def _get_mixture_model(self):
        return MixtureBaseline(
            targets=self.config["targets"],
            window_months=self.config["window_months"],
            lambda_mix=self.config["lambda_mix"],
            n_samples=self.config["n_samples"],
            partition_dict=self.partition_dict,
            loa=self.loa,
            seed=self.config["seed"],  # required + audited genome key (ADR-021); no silent default
        )

    def _get_parametric_conflictology(self):
        # family/transform/seed are required, audited genome keys (ADR-021/ADR-022); the
        # constructor fails loud on an unsupported family or an illegal family×transform.
        return ParametricConflictology(
            targets=self.config["targets"],
            window_months=self.config["window_months"],
            partition_dict=self.partition_dict,
            loa=self.loa,
            n_samples=self.config["n_samples"],
            family=self.config["family"],
            transform=self.config["transform"],
            seed=self.config["seed"],
        )

    def _get_parametric_hurdle(self):
        # family/transform/seed are required, audited genome keys (ADR-021/ADR-022); the
        # constructor fails loud on a non-continuous family or an illegal family×transform.
        return ParametricHurdleConflictology(
            targets=self.config["targets"],
            window_months=self.config["window_months"],
            partition_dict=self.partition_dict,
            loa=self.loa,
            n_samples=self.config["n_samples"],
            family=self.config["family"],
            transform=self.config["transform"],
            seed=self.config["seed"],
        )
