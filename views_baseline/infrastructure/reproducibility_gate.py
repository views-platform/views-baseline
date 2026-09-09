import logging

from .exceptions import MissingHyperparameterError

logger = logging.getLogger(__name__)


class ReproducibilityGate:
    """
    Canonical hyperparameter contract for views-baseline models.

    Defines which configuration keys are required by all baseline models
    (CORE_GENOME) and which are required per algorithm (ALGORITHM_GENOMES).
    The audit_manifest() method enforces these contracts at runtime.

    This class is importable by downstream packages (e.g. views-models)
    so they can validate their config_hyperparameters.py files statically.
    """

    class Config:
        """Gates related to configuration and hyperparameter integrity."""

        # Core keys required by ALL baseline models regardless of algorithm.
        #
        # `regression_targets` and `level` were promoted here after the pipeline-core
        # #380 outage (issue #85): both are dereferenced unconditionally on every run —
        # `level` at `manager/baseline_manager.py` one line after this audit returns, and
        # `regression_targets` by all seven catalog factories — but neither was declared,
        # so a config missing one cleared both validation layers and died on a bare
        # KeyError deep in a factory. ADR-009 recorded that gap as accepted debt in March
        # 2026; it is what the outage was made of. Verified against all 29 shipped
        # baseline configs before promotion (29/29 declare both).
        CORE_GENOME = [
            "steps",
            "time_steps",
            "prediction_format",
            "regression_targets",
            "level",
        ]

        # Algorithm-specific keys (audited only when the algorithm matches).
        ALGORITHM_GENOMES = {
            "ZeroModel": [],
            "LocfModel": [],
            "AverageModel": ["window_months"],
            "ConflictologyModel": ["window_months", "n_samples", "seed"],
            "MixtureBaseline": ["window_months", "lambda_mix", "n_samples", "seed"],
            # Parametric climatology (ADR-022): `family` and `transform` join `seed` as
            # required, audited reproducibility keys — no magic defaults (ADR-021).
            "ParametricConflictology": [
                "window_months", "n_samples", "seed", "family", "transform",
            ],
            "ParametricHurdleConflictology": [
                "window_months", "n_samples", "seed", "family", "transform",
            ],
        }

        @staticmethod
        def audit_manifest(config: dict) -> None:
            """
            Verify that all mandatory hyperparameters are present and non-None.

            Checks in order:
            1. All CORE_GENOME keys are present.
            2. The algorithm is registered in ALGORITHM_GENOMES.
            3. All algorithm-specific keys are present.
            4. No required key has a value of None.

            Raises MissingHyperparameterError on any violation.
            """
            # 1. Audit Core Genome
            missing_core = [
                k
                for k in ReproducibilityGate.Config.CORE_GENOME
                if k not in config
            ]
            if missing_core:
                msg = (
                    "REPRODUCIBILITY CONTRACT VIOLATED: "
                    f"Missing core parameters: {missing_core}"
                )
                logger.error(msg)
                raise MissingHyperparameterError(msg)

            # 2. Identify algorithm and check it is registered
            if "algorithm" not in config:
                msg = (
                    "REPRODUCIBILITY CONTRACT VIOLATED: "
                    "Missing required key: 'algorithm'"
                )
                logger.error(msg)
                raise MissingHyperparameterError(msg)

            algo = config["algorithm"]
            algo_genomes = ReproducibilityGate.Config.ALGORITHM_GENOMES
            if algo not in algo_genomes:
                available = list(algo_genomes.keys())
                msg = (
                    "REPRODUCIBILITY CONTRACT VIOLATED: "
                    f"Unknown algorithm '{algo}'. "
                    f"Available: {available}"
                )
                logger.error(msg)
                raise MissingHyperparameterError(msg)

            # 3. Audit algorithm-specific keys
            algo_keys = algo_genomes[algo]
            missing_algo = [k for k in algo_keys if k not in config]
            if missing_algo:
                msg = (
                    "REPRODUCIBILITY CONTRACT VIOLATED: "
                    f"Algorithm '{algo}' requires missing parameters: {missing_algo}"
                )
                logger.error(msg)
                raise MissingHyperparameterError(msg)

            # 4. Reject None values for all required keys
            all_required = list(ReproducibilityGate.Config.CORE_GENOME) + algo_keys
            explicit_nones = [k for k in all_required if config.get(k) is None]
            if explicit_nones:
                msg = (
                    "REPRODUCIBILITY CONTRACT VIOLATED: "
                    f"Mandatory parameters set to None: {explicit_nones}. "
                    "Implicit defaults are forbidden."
                )
                logger.error(msg)
                raise MissingHyperparameterError(msg)
