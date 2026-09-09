import pytest

from views_baseline.model.catalog import BaselineModelCatalog
from views_baseline.model.models.distributional import (
    ConflictologyModel,
    MixtureBaseline,
    ParametricConflictology,
    ParametricHurdleConflictology,
)
from views_baseline.model.models.point import AverageModel, LocfModel, ZeroModel


def test_catalog_lists_all_models():
    config = {"regression_targets": ["y1"], "window_months": 3}
    partition_dict = {"test": (445, 492)}
    loa = "pg_id"

    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa=loa)
    models = set(catalog.list_models())

    assert {
        "ZeroModel",
        "LocfModel",
        "AverageModel",
        "ConflictologyModel",
        "MixtureBaseline",
    }.issubset(models)


def test_catalog_returns_zero_model():
    config = {"regression_targets": ["y1"]}
    partition_dict = {"test": (493, 540)}
    loa = "pg_id"

    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa=loa)
    model = catalog.get_model("ZeroModel")

    assert isinstance(model, ZeroModel)
    assert model.targets == config["regression_targets"]
    assert model.partition_dict is partition_dict
    assert model.loa == loa


def test_catalog_returns_locf_model():
    config = {"regression_targets": ["y1", "y2"]}
    partition_dict = {"test": (445, 492)}
    loa = "pg_id"

    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa=loa)
    model = catalog.get_model("LocfModel")

    assert isinstance(model, LocfModel)
    assert model.targets == config["regression_targets"]
    assert model.partition_dict is partition_dict
    assert model.loa == loa


def test_catalog_returns_average_model():
    config = {"regression_targets": ["y1"], "window_months": 6}
    partition_dict = {"test": (445, 492)}
    loa = "pg_id"

    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa=loa)
    model = catalog.get_model("AverageModel")

    assert isinstance(model, AverageModel)
    assert model.targets == config["regression_targets"]
    assert model.partition_dict is partition_dict
    assert model.loa == loa
    assert model.window_months == config["window_months"]


def test_catalog_returns_conflictology_model():
    config = {"regression_targets": ["y1", "y2"], "window_months": 5, "n_samples": 128, "seed": 7}
    partition_dict = {"test": (445, 492)}
    loa = "pg_id"

    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa=loa)
    model = catalog.get_model("ConflictologyModel")

    assert isinstance(model, ConflictologyModel)
    assert model.targets == config["regression_targets"]
    assert model.partition_dict is partition_dict
    assert model.loa == loa
    assert model.window_months == config["window_months"]
    assert model.n_samples == 128
    assert model.seed == 7  # forwarded from config, not the default (ADR-021, C-10)


def test_catalog_missing_n_samples_for_conflictology_raises():
    config = {"regression_targets": ["y1"], "window_months": 5}
    partition_dict = {"test": (445, 492)}
    loa = "pg_id"

    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa=loa)
    with pytest.raises(ValueError, match="n_samples"):
        catalog.get_model("ConflictologyModel")


def test_catalog_missing_keys_for_mixture_raises():
    config = {"regression_targets": ["y1"]}
    partition_dict = {"test": (445, 492)}
    loa = "pg_id"

    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa=loa)
    with pytest.raises(ValueError, match="window_months"):
        catalog.get_model("MixtureBaseline")


def test_catalog_raises_for_unknown_model():
    config = {"regression_targets": ["y1"]}
    partition_dict = {"test": (493, 540)}
    loa = "pg_id"

    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa=loa)

    with pytest.raises(ValueError, match="Model 'NonExistingModel'"):
        catalog.get_model("NonExistingModel")


def test_catalog_returns_mixture_model():
    config = {
        "regression_targets": ["y1"], "window_months": 18, "lambda_mix": 0.05,
        "n_samples": 256, "seed": 11,
    }
    partition_dict = {"test": (493, 540)}
    loa = "pg_id"

    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa=loa)
    model = catalog.get_model("MixtureBaseline")

    assert isinstance(model, MixtureBaseline)
    assert model.targets == config["regression_targets"]
    assert model.partition_dict is partition_dict
    assert model.loa == loa
    assert model.window_months == 18
    assert model.lambda_mix == 0.05
    assert model.n_samples == 256
    assert model.seed == 11  # forwarded from config, not the default (ADR-021, C-10)


def test_catalog_returns_parametric_conflictology():
    config = {
        "regression_targets": ["y1"], "window_months": 5, "n_samples": 128,
        "seed": 7, "family": "nb", "transform": "none",
    }
    catalog = BaselineModelCatalog(
        config=config, partition_dict={"test": (445, 492)}, loa="pg_id"
    )
    model = catalog.get_model("ParametricConflictology")

    assert isinstance(model, ParametricConflictology)
    assert model.window_months == 5
    assert model.n_samples == 128
    assert model.seed == 7  # forwarded, not the DEFAULT_SEED sentinel (ADR-021)
    assert model.family == "nb"
    assert model.transform == "none"


def test_catalog_returns_parametric_hurdle():
    config = {
        "regression_targets": ["y1", "y2"], "window_months": 9, "n_samples": 256,
        "seed": 11, "family": "gumbel", "transform": "log1p",
    }
    catalog = BaselineModelCatalog(
        config=config, partition_dict={"test": (493, 540)}, loa="pg_id"
    )
    model = catalog.get_model("ParametricHurdleConflictology")

    assert isinstance(model, ParametricHurdleConflictology)
    assert model.n_samples == 256
    assert model.seed == 11
    assert model.family == "gumbel"
    assert model.transform == "log1p"


def test_catalog_missing_family_for_parametric_raises():
    config = {
        "regression_targets": ["y1"], "window_months": 5, "n_samples": 128, "seed": 7,
        "transform": "none",
    }
    catalog = BaselineModelCatalog(
        config=config, partition_dict={"test": (445, 492)}, loa="pg_id"
    )
    with pytest.raises(ValueError, match="family"):
        catalog.get_model("ParametricConflictology")


def test_catalog_illegal_family_transform_fails_loud():
    """ADR-021/ADR-022: nb + log1p is a contract violation, surfaced at construction."""
    config = {
        "regression_targets": ["y1"], "window_months": 5, "n_samples": 128,
        "seed": 7, "family": "nb", "transform": "log1p",
    }
    catalog = BaselineModelCatalog(
        config=config, partition_dict={"test": (445, 492)}, loa="pg_id"
    )
    with pytest.raises(ValueError, match="invalid for count family"):
        catalog.get_model("ParametricConflictology")


def test_catalog_continuous_family_rejected_by_no_hurdle_via_catalog():
    config = {
        "regression_targets": ["y1"], "window_months": 5, "n_samples": 128,
        "seed": 7, "family": "lognormal", "transform": "none",
    }
    catalog = BaselineModelCatalog(
        config=config, partition_dict={"test": (445, 492)}, loa="pg_id"
    )
    with pytest.raises(ValueError, match="native-zero families"):
        catalog.get_model("ParametricConflictology")


def test_catalog_count_family_rejected_by_hurdle_via_catalog():
    config = {
        "regression_targets": ["y1"], "window_months": 5, "n_samples": 128,
        "seed": 7, "family": "nb", "transform": "none",
    }
    catalog = BaselineModelCatalog(
        config=config, partition_dict={"test": (445, 492)}, loa="pg_id"
    )
    with pytest.raises(ValueError, match="continuous positive-part families"):
        catalog.get_model("ParametricHurdleConflictology")


# -----------------------------------------------------------------------
# Param-completeness (ADR-021 / SOLID-OCP): the factory must forward EVERY
# config-supplied constructor parameter. Generic guard so the next added
# parameter cannot be silently dropped the way `seed` was (C-10, C-19/FM-3).
# -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "algo, cls",
    [("ConflictologyModel", ConflictologyModel), ("MixtureBaseline", MixtureBaseline)],
)
def test_catalog_forwards_all_config_params(algo, cls):
    import inspect

    # Distinct, non-default values for every possible constructor param.
    config = {
        "regression_targets": ["y1"],
        "window_months": 7,
        "n_samples": 13,
        "lambda_mix": 0.3,
        "seed": 99,
    }
    partition_dict = {"test": (445, 492)}
    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa="pgm")
    model = catalog.get_model(algo)

    # partition_dict and loa are injected by the catalog itself, not from config.
    injected = {"self", "partition_dict", "loa"}
    for pname in inspect.signature(cls.__init__).parameters:
        if pname in injected or pname not in config:
            continue
        assert getattr(model, pname) == config[pname], (
            f"{algo}: constructor param '{pname}' was not forwarded from config "
            f"(got {getattr(model, pname)!r}, expected {config[pname]!r}) — a silently "
            f"dropped parameter (C-10 class)."
        )


@pytest.mark.parametrize(
    "algo, cls, family",
    [
        ("ParametricConflictology", ParametricConflictology, "nb"),
        ("ParametricHurdleConflictology", ParametricHurdleConflictology, "gumbel"),
    ],
)
def test_catalog_forwards_all_config_params_parametric(algo, cls, family):
    """Same C-10-class guard for the parametric factories — `family`/`transform`/`seed`
    (the newly added params, exactly the class that was dropped before) must all forward."""
    import inspect

    config = {
        "regression_targets": ["y1"],
        "window_months": 7,
        "n_samples": 13,
        "seed": 99,
        "family": family,
        "transform": "log1p" if family == "gumbel" else "none",
    }
    catalog = BaselineModelCatalog(
        config=config, partition_dict={"test": (445, 492)}, loa="pgm"
    )
    model = catalog.get_model(algo)

    injected = {"self", "partition_dict", "loa"}
    for pname in inspect.signature(cls.__init__).parameters:
        if pname in injected or pname not in config:
            continue
        assert getattr(model, pname) == config[pname], (
            f"{algo}: constructor param '{pname}' was not forwarded from config "
            f"(got {getattr(model, pname)!r}, expected {config[pname]!r}) — a silently "
            f"dropped parameter (C-10 class)."
        )
