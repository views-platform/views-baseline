import pickle
import re
from types import SimpleNamespace

import numpy as np
from conftest import make_manager

import views_baseline.manager.baseline_manager as bm
from views_baseline.model.models.point import ZeroModel

# ---------------------------------------------------------------------
# Tests: _evaluate_model_artifact
# ---------------------------------------------------------------------


def test_manager_evaluate_uses_zero_model(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    config = {
        "run_type": "eval",
        "level": "pgm",
        "algorithm": "ZeroModel",
        "targets": targets,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
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


def test_manager_evaluate_uses_locf_model(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    config = {
        "run_type": "eval",
        "level": "pgm",
        "algorithm": "LocfModel",
        "targets": targets,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
        "sequence_numbers": 1,
    }

    manager = make_manager(config, manager_partition_dict)
    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)

    result = manager._evaluate_model_artifact(eval_type="temporal")

    assert isinstance(result, dict)
    assert set(result.keys()) == set(targets)
    for target_key in targets:
        assert len(result[target_key]) == 1


# ---------------------------------------------------------------------
# Tests: _forecast_model_artifact
# ---------------------------------------------------------------------


def test_manager_forecast_uses_baseline_model(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    from views_frames import PredictionFrame

    config = {
        "run_type": "forecast",
        "level": "pgm",
        "algorithm": "LocfModel",
        "targets": targets,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
    }

    manager = make_manager(config, manager_partition_dict)
    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)

    result = manager._forecast_model_artifact()

    assert isinstance(result, dict)
    assert set(result.keys()) == set(targets)
    for target_key in targets:
        assert isinstance(result[target_key], PredictionFrame)


def test_manager_forecast_respects_algorithm_choice(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    config_zero = {
        "run_type": "forecast",
        "level": "pgm",
        "algorithm": "ZeroModel",
        "targets": targets,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
    }
    manager_zero = make_manager(config_zero, manager_partition_dict)
    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)
    forecasts_zero = manager_zero._forecast_model_artifact()

    config_locf = {
        "run_type": "forecast",
        "level": "pgm",
        "algorithm": "LocfModel",
        "targets": targets,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
    }
    manager_locf = make_manager(config_locf, manager_partition_dict)
    forecasts_locf = manager_locf._forecast_model_artifact()

    target = targets[0]
    assert forecasts_zero[target].values.shape == forecasts_locf[target].values.shape
    assert not np.array_equal(forecasts_zero[target].values, forecasts_locf[target].values)


# ---------------------------------------------------------------------
# Tests: _setup_model_and_data
# ---------------------------------------------------------------------


def test_manager_setup_returns_model_and_data(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    config = {
        "run_type": "eval",
        "level": "pgm",
        "algorithm": "ZeroModel",
        "targets": targets,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
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
    config = {
        "run_type": "calibration",
        "level": "pgm",
        "algorithm": "ZeroModel",
        "targets": targets,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
    }
    manager = make_manager(config, manager_partition_dict)
    manager._model_path = SimpleNamespace(
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
# Tests: distributional model paths (unchanged — already returned dict)
# ---------------------------------------------------------------------


def test_manager_evaluate_distributional_model(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    config = {
        "run_type": "eval",
        "level": "pgm",
        "algorithm": "ConflictologyModel",
        "targets": targets,
        "window_months": 6,
        "n_samples": 64,
        "seed": 42,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
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


def test_manager_evaluate_sweep(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    config = {
        "run_type": "eval",
        "level": "pgm",
        "algorithm": "LocfModel",
        "targets": targets,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
        "sequence_numbers": 2,
    }

    manager = make_manager(config, manager_partition_dict)
    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)

    model, _ = manager._setup_model_and_data()
    result = manager._evaluate_sweep(eval_type="temporal", model=model)

    assert isinstance(result, dict)
    assert set(result.keys()) == set(targets)
    for target_key in targets:
        assert isinstance(result[target_key], list)
        assert len(result[target_key]) == 2


def test_manager_forecast_distributional_model(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    from views_frames import PredictionFrame

    config = {
        "run_type": "forecast",
        "level": "pgm",
        "algorithm": "ConflictologyModel",
        "targets": targets,
        "window_months": 6,
        "n_samples": 64,
        "seed": 42,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
    }

    manager = make_manager(config, manager_partition_dict)
    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)

    result = manager._forecast_model_artifact()

    assert isinstance(result, dict)
    assert set(result.keys()) == set(targets)
    for target_key in targets:
        assert isinstance(result[target_key], PredictionFrame)
