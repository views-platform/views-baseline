import pytest

from views_baseline.model.baseline import (
    AverageModel,
    ConflictologyModel,
    LocfModel,
    MixtureBaseline,
    ParametricConflictology,
    ParametricHurdleConflictology,
    ZeroModel,
)
from views_baseline.model.protocol import BaselineModel, DistributionalBaselineModel

_PARTITION = {"test": (493, 540)}
_BASE = {"targets": ["y"], "partition_dict": _PARTITION, "loa": "pg_id"}

_POINT_MODELS = [
    (ZeroModel, _BASE),
    (LocfModel, _BASE),
    (AverageModel, {**_BASE, "window_months": 3}),
]

_DISTRIBUTIONAL_MODELS = [
    (ConflictologyModel, {**_BASE, "window_months": 3, "n_samples": 10}),
    (MixtureBaseline, {**_BASE, "window_months": 3, "lambda_mix": 0.05, "n_samples": 10}),
    (
        ParametricConflictology,
        {**_BASE, "window_months": 3, "n_samples": 10, "family": "nb", "transform": "none"},
    ),
    (
        ParametricHurdleConflictology,
        {**_BASE, "window_months": 3, "n_samples": 10, "family": "gumbel", "transform": "log1p"},
    ),
]

_ALL_MODELS = _POINT_MODELS + _DISTRIBUTIONAL_MODELS


@pytest.mark.parametrize("Model,kwargs", _ALL_MODELS)
def test_model_satisfies_baseline_protocol(Model, kwargs):
    instance = Model(**kwargs)
    assert isinstance(instance, BaselineModel)


@pytest.mark.parametrize("Model,kwargs", _DISTRIBUTIONAL_MODELS)
def test_distributional_models_satisfy_distributional_protocol(Model, kwargs):
    assert isinstance(Model(**kwargs), DistributionalBaselineModel)


@pytest.mark.parametrize("Model,kwargs", _POINT_MODELS)
def test_non_distributional_models_reject_distributional_protocol(Model, kwargs):
    assert not isinstance(Model(**kwargs), DistributionalBaselineModel)
