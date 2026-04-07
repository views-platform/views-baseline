import pickle
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from conftest import make_manager

import views_baseline.manager.baseline_manager as bm
from views_baseline.model.baseline import LocfModel, ZeroModel

# ---------------------------------------------------------------------
# Tests: _evaluate_model_artifact
# ---------------------------------------------------------------------


def test_manager_evaluate_uses_zero_model(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    """
    _evaluate_model_artifact should:
    - Use BaselineModelCatalog to get the correct baseline class (ZeroModel here)
    - Call .predict() for each sequence_number
    - Return a list of prediction DataFrames
    """
    config = {
        "run_type": "eval",
        "level": "pg_id",
        "algorithm": "ZeroModel",
        "targets": targets,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "sequence_numbers": 2,
    }

    manager = make_manager(config, manager_partition_dict)
    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)

    preds_list = manager._evaluate_model_artifact(eval_type="temporal")

    assert isinstance(preds_list, list)
    assert len(preds_list) == 2

    zero_model = ZeroModel(
        targets=targets, partition_dict=manager_partition_dict, loa="pg_id"
    )
    zero_model.fit(manager_df)

    expected0 = zero_model.predict(df=manager_df, sequence_number=0, output_length=36)
    expected1 = zero_model.predict(df=manager_df, sequence_number=1, output_length=36)

    pd.testing.assert_frame_equal(preds_list[0], expected0)
    pd.testing.assert_frame_equal(preds_list[1], expected1)


def test_manager_evaluate_uses_locf_model(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    """
    Same as above but for LocfModel, just to make sure the manager is respecting
    config['algorithm'] and not hard-coding ZeroModel.
    """
    config = {
        "run_type": "eval",
        "level": "pg_id",
        "algorithm": "LocfModel",
        "targets": targets,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "sequence_numbers": 1,
    }

    manager = make_manager(config, manager_partition_dict)
    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)

    preds_list = manager._evaluate_model_artifact(eval_type="temporal")

    assert isinstance(preds_list, list)
    assert len(preds_list) == 1

    locf = LocfModel(
        targets=targets, partition_dict=manager_partition_dict, loa="pg_id"
    )
    locf.fit(manager_df)
    expected = locf.predict(df=manager_df, sequence_number=0, output_length=36)

    pd.testing.assert_frame_equal(preds_list[0], expected)


# ---------------------------------------------------------------------
# Tests: _forecast_model_artifact
# ---------------------------------------------------------------------


def test_manager_forecast_uses_baseline_model(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    """
    _forecast_model_artifact should:
    - Load the viewser df (here via monkeypatched read_dataframe)
    - Instantiate the correct baseline model via the catalog
    - Call .fit() and then .predict(sequence_number=0)
    - Return that prediction DataFrame
    """
    config = {
        "run_type": "forecast",
        "level": "pg_id",
        "algorithm": "LocfModel",
        "targets": targets,
        "steps": [*range(1, 37)],
        "time_steps": 36,
    }

    manager = make_manager(config, manager_partition_dict)
    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)

    forecasts = manager._forecast_model_artifact()

    locf = LocfModel(
        targets=targets, partition_dict=manager_partition_dict, loa="pg_id"
    )
    locf.fit(manager_df)
    expected = locf.predict(df=manager_df, sequence_number=0, output_length=36)

    pd.testing.assert_frame_equal(forecasts, expected)


def test_manager_forecast_respects_algorithm_choice(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    """
    Smoke test: switching algorithm in config should change the forecast
    (ZeroModel vs LocfModel should not match unless the data are degenerate).
    """
    config_zero = {
        "run_type": "forecast",
        "level": "pg_id",
        "algorithm": "ZeroModel",
        "targets": targets,
        "steps": [*range(1, 37)],
        "time_steps": 36,
    }
    manager_zero = make_manager(config_zero, manager_partition_dict)
    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)
    forecasts_zero = manager_zero._forecast_model_artifact()

    config_locf = {
        "run_type": "forecast",
        "level": "pg_id",
        "algorithm": "LocfModel",
        "targets": targets,
        "steps": [*range(1, 37)],
        "time_steps": 36,
    }
    manager_locf = make_manager(config_locf, manager_partition_dict)
    forecasts_locf = manager_locf._forecast_model_artifact()

    assert forecasts_zero.shape == forecasts_locf.shape
    assert (forecasts_zero.values != forecasts_locf.values).any()


# ---------------------------------------------------------------------
# Tests: _setup_model_and_data
# ---------------------------------------------------------------------


def test_manager_setup_returns_model_and_data(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    """
    _setup_model_and_data should instantiate the correct model via the catalog,
    fit it on the data, and return (model, df).
    """
    config = {
        "run_type": "eval",
        "level": "pg_id",
        "algorithm": "ZeroModel",
        "targets": targets,
        "steps": [*range(1, 37)],
        "time_steps": 36,
    }
    manager = make_manager(config, manager_partition_dict)
    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)

    model, df = manager._setup_model_and_data()

    assert isinstance(model, ZeroModel)
    assert df.shape == manager_df.shape


# ---------------------------------------------------------------------
# Tests: _train_model_artifact
# ---------------------------------------------------------------------


def test_manager_train_saves_artifact(
    monkeypatch, manager_df, manager_partition_dict, targets, tmp_path
):
    """
    _train_model_artifact should pickle the fitted model to artifacts/.
    """
    config = {
        "run_type": "calibration",
        "level": "pg_id",
        "algorithm": "ZeroModel",
        "targets": targets,
        "steps": [*range(1, 37)],
        "time_steps": 36,
    }
    manager = make_manager(config, manager_partition_dict)
    manager._model_path = SimpleNamespace(
        data_raw=Path("dummy_raw_path"),
        artifacts=tmp_path,
    )
    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)

    model = manager._train_model_artifact()

    assert isinstance(model, ZeroModel)

    pkl_files = list(tmp_path.glob("calibration_model_*.pkl"))
    assert len(pkl_files) == 1

    assert re.match(r"calibration_model_\d{8}_\d{6}\.pkl", pkl_files[0].name)

    with open(pkl_files[0], "rb") as f:
        loaded = pickle.load(f)
    assert isinstance(loaded, ZeroModel)
    assert loaded.targets == targets


# ---------------------------------------------------------------------
# Tests: distributional model paths
# ---------------------------------------------------------------------


def test_manager_evaluate_distributional_model(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    """
    _evaluate_model_artifact for a distributional model (ConflictologyModel)
    should return dict[str, list] where each value is a list of PredictionFrames.
    """
    config = {
        "run_type": "eval",
        "level": "pg_id",
        "algorithm": "ConflictologyModel",
        "targets": targets,
        "window_months": 6,
        "n_samples": 64,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "sequence_numbers": 2,
    }

    manager = make_manager(config, manager_partition_dict)
    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)

    result = manager._evaluate_model_artifact(eval_type="temporal")

    assert isinstance(result, dict)
    assert set(result.keys()) == set(targets)
    for target_key in targets:
        assert isinstance(result[target_key], list)
        assert len(result[target_key]) == 2


def test_manager_forecast_distributional_model(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    """
    _forecast_model_artifact for a distributional model (ConflictologyModel)
    should return dict[str, PredictionFrame].
    """
    from views_pipeline_core.data.prediction_frame import PredictionFrame

    config = {
        "run_type": "forecast",
        "level": "pg_id",
        "algorithm": "ConflictologyModel",
        "targets": targets,
        "window_months": 6,
        "n_samples": 64,
        "steps": [*range(1, 37)],
        "time_steps": 36,
    }

    manager = make_manager(config, manager_partition_dict)
    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)

    result = manager._forecast_model_artifact()

    assert isinstance(result, dict)
    assert set(result.keys()) == set(targets)
    for target_key in targets:
        assert isinstance(result[target_key], PredictionFrame)
