import pytest

from views_baseline.model.baseline import (
    AverageModel,
    ConflictologyModel,
    LocfModel,
    MixtureBaseline,
    ZeroModel,
)
from views_baseline.model.catalog import BaselineModelCatalog


def test_catalog_lists_all_models():
    config = {"targets": ["y1"], "window_months": 3}
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
    config = {"targets": ["y1"]}
    partition_dict = {"test": (493, 540)}
    loa = "pg_id"

    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa=loa)
    model = catalog.get_model("ZeroModel")

    assert isinstance(model, ZeroModel)
    assert model.targets == config["targets"]
    assert model.partition_dict is partition_dict
    assert model.loa == loa


def test_catalog_returns_locf_model():
    config = {"targets": ["y1", "y2"]}
    partition_dict = {"test": (445, 492)}
    loa = "pg_id"

    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa=loa)
    model = catalog.get_model("LocfModel")

    assert isinstance(model, LocfModel)
    assert model.targets == config["targets"]
    assert model.partition_dict is partition_dict
    assert model.loa == loa


def test_catalog_returns_average_model():
    config = {"targets": ["y1"], "window_months": 6}
    partition_dict = {"test": (445, 492)}
    loa = "pg_id"

    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa=loa)
    model = catalog.get_model("AverageModel")

    assert isinstance(model, AverageModel)
    assert model.targets == config["targets"]
    assert model.partition_dict is partition_dict
    assert model.loa == loa
    assert model.window_months == config["window_months"]


def test_catalog_returns_conflictology_model():
    config = {"targets": ["y1", "y2"], "window_months": 5, "n_samples": 128}
    partition_dict = {"test": (445, 492)}
    loa = "pg_id"

    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa=loa)
    model = catalog.get_model("ConflictologyModel")

    assert isinstance(model, ConflictologyModel)
    assert model.targets == config["targets"]
    assert model.partition_dict is partition_dict
    assert model.loa == loa
    assert model.window_months == config["window_months"]
    assert model.n_samples == 128


def test_catalog_missing_n_samples_for_conflictology_raises():
    config = {"targets": ["y1"], "window_months": 5}
    partition_dict = {"test": (445, 492)}
    loa = "pg_id"

    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa=loa)
    with pytest.raises(ValueError, match="n_samples"):
        catalog.get_model("ConflictologyModel")


def test_catalog_missing_keys_for_mixture_raises():
    config = {"targets": ["y1"]}
    partition_dict = {"test": (445, 492)}
    loa = "pg_id"

    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa=loa)
    with pytest.raises(ValueError, match="window_months"):
        catalog.get_model("MixtureBaseline")


def test_catalog_raises_for_unknown_model():
    config = {"targets": ["y1"]}
    partition_dict = {"test": (493, 540)}
    loa = "pg_id"

    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa=loa)

    with pytest.raises(ValueError, match="Model 'NonExistingModel'"):
        catalog.get_model("NonExistingModel")


def test_catalog_returns_mixture_model():
    config = {"targets": ["y1"], "window_months": 18, "lambda_mix": 0.05, "n_samples": 256}
    partition_dict = {"test": (493, 540)}
    loa = "pg_id"

    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa=loa)
    model = catalog.get_model("MixtureBaseline")

    assert isinstance(model, MixtureBaseline)
    assert model.targets == config["targets"]
    assert model.partition_dict is partition_dict
    assert model.loa == loa
    assert model.window_months == 18
    assert model.lambda_mix == 0.05
    assert model.n_samples == 256
