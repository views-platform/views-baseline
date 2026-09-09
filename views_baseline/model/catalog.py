from views_baseline.infrastructure.reproducibility_gate import ReproducibilityGate
from views_baseline.model.models.distributional import (
    ConflictologyModel,
    MixtureBaseline,
    ParametricConflictology,
    ParametricHurdleConflictology,
)
from views_baseline.model.models.point import AverageModel, LocfModel, ZeroModel


class BaselineModelCatalog:
    # Single source of truth lives in ReproducibilityGate.Config.ALGORITHM_GENOMES.
    # This alias keeps the catalog's get_model() validation unchanged.
    MODEL_GENOMES = ReproducibilityGate.Config.ALGORITHM_GENOMES

    def __init__(self, config: dict, partition_dict: dict, loa: str):
        """
        Catalog of available baseline models.

        Reads the views-pipeline-core config vocabulary: ``regression_targets``.
        (``targets`` was a synthesised backward-compatibility key, retired in
        pipeline-core #380; ``combined_targets()`` now *raises* on a config that
        still carries it.) Derived once here rather than at each of the seven
        factory sites, so a future rename of the platform's target vocabulary is
        one edit, not seven.

        ``classification_targets`` is deliberately not read: no baseline is wired
        for a classification target today, and no baseline config declares one.
        See ADR-012 when that changes — the decision is per-model (Zero/LOCF carry
        over cleanly; the magnitude-fitting families do not), not catalog-wide.
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

    @property
    def targets(self) -> list:
        """The target names, read from the one config key that carries them.

        Derived on access rather than stored in ``__init__`` so that
        :meth:`list_models` stays a pure accessor with no config precondition — it is
        declared Stable in ADR-004 and "no side effects" in this class's CIC, and
        enumerating the catalog must not require a config that carries targets.

        Rejects a bare string: ``regression_targets="lr_ged_sb"`` would otherwise
        ``list()`` into nine single-character targets, which both the gate (presence and
        non-emptiness only) and the models accept without complaint.
        """
        raw = self.config["regression_targets"]
        if isinstance(raw, str):
            raise ValueError(
                f"regression_targets must be a sequence of target names, got the string "
                f"{raw!r}. A bare string would be split into one target per character."
            )
        return list(raw)

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
            targets=self.targets, partition_dict=self.partition_dict, loa=self.loa
        )

    def _get_locf_model(self):
        return LocfModel(
            targets=self.targets, partition_dict=self.partition_dict, loa=self.loa
        )

    def _get_average_model(self):
        return AverageModel(
            targets=self.targets,
            window_months=self.config["window_months"],
            partition_dict=self.partition_dict,
            loa=self.loa,
        )

    def _get_conflictology_model(self):
        return ConflictologyModel(
            targets=self.targets,
            window_months=self.config["window_months"],
            partition_dict=self.partition_dict,
            loa=self.loa,
            n_samples=self.config["n_samples"],
            seed=self.config["seed"],  # required + audited genome key (ADR-021); no silent default
        )

    def _get_mixture_model(self):
        return MixtureBaseline(
            targets=self.targets,
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
            targets=self.targets,
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
            targets=self.targets,
            window_months=self.config["window_months"],
            partition_dict=self.partition_dict,
            loa=self.loa,
            n_samples=self.config["n_samples"],
            family=self.config["family"],
            transform=self.config["transform"],
            seed=self.config["seed"],
        )
