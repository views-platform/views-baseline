class ReproducibilityError(Exception):
    """Base class for all reproducibility gate failures."""

    pass


class MissingHyperparameterError(ReproducibilityError):
    """Raised when a mandatory hyperparameter is missing from the config."""

    pass
