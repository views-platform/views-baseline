import pytest

from views_baseline.model.catalog import BaselineModelCatalog
from views_baseline.model.baseline import (
    ZeroModel,
    LocfModel,
    AverageModel,
    ConflictologyModel,
)

def test_catalog_lists_all_models():
    config = {"targets": ["y1"], "months": 3}
    partition_dict = {"test": (445, 492)}
    loa = "pg_id"

    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa=loa)
    models = set(catalog.list_models())

    assert {"ZeroModel", "LocfModel", "AverageModel", "ConflictologyModel"}.issubset(models)


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
    config = {"targets": ["y1"], "months": 6}
    partition_dict = {"test": (445, 492)}
    loa = "pg_id"

    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa=loa)
    model = catalog.get_model("AverageModel")

    assert isinstance(model, AverageModel)
    assert model.targets == config["targets"]
    assert model.partition_dict is partition_dict
    assert model.loa == loa
    assert model.months == config["months"]


def test_catalog_returns_conflictology_model():
    config = {"targets": ["y1", "y2"], "months": 5}
    partition_dict = {"test": (445, 492)}
    loa = "pg_id"

    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa=loa)
    model = catalog.get_model("ConflictologyModel")

    assert isinstance(model, ConflictologyModel)
    assert model.targets == config["targets"]
    assert model.partition_dict is partition_dict
    assert model.loa == loa
    assert model.months == config["months"]


def test_catalog_raises_for_unknown_model():
    config = {"targets": ["y1"]}
    partition_dict = {"test": (493, 540)}
    loa = "pg_id"

    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa=loa)

    with pytest.raises(ValueError, match="Model 'NonExistingModel'"):
        catalog.get_model("NonExistingModel")
