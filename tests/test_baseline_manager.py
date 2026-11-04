import pickle
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import make_dummy_ff, make_manager
from views_pipeline_core.modules.dataloaders.datafactory_contract import (
    DATA_FORMAT_DATAFRAME,
    DATA_FORMAT_FEATURE_FRAME,
)

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
        "regression_targets": targets,
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
        "regression_targets": targets,
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
        "regression_targets": targets,
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


# ---------------------------------------------------------------------
# Tests: frame-native manager seam (issue #64)
#
# A model declaring data_format: feature_frame is served the pipeline-core
# FeatureFrame directory cache directly — no pandas in the manager. Every
# other model keeps the byte-identical read_dataframe path. These tests pin
# both read-sites (_setup_model_and_data at :60, _evaluate_sweep at :125),
# and assert the frame path NEVER touches read_dataframe (and vice versa).
# ---------------------------------------------------------------------


def _frame_config(algorithm="LocfModel", run_type="forecast", targets=("y1", "y2"), **extra):
    config = {
        "run_type": run_type,
        "level": "pgm",
        "algorithm": algorithm,
        "regression_targets": list(targets),
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
    }
    config.update(extra)
    return config


def _no_pandas(path):
    raise AssertionError("read_dataframe was called on a frame-declared model")


def test_manager_setup_flows_feature_frame_from_cache(
    monkeypatch, manager_partition_dict, targets
):
    """Frame-declared model: _setup_model_and_data loads the frame cache and a
    FeatureFrame — not a DataFrame — flows into fit()."""
    from views_frames import FeatureFrame

    manager = make_manager(_frame_config(), manager_partition_dict)
    manager._data_format = DATA_FORMAT_FEATURE_FRAME
    manager._cached_frame_path = Path("dummy_frame_cache")

    seen = {}

    def fake_load_frame_cache(path):
        seen["path"] = path
        return make_dummy_ff(time_range=range(110, 126))

    monkeypatch.setattr(bm, "read_dataframe", _no_pandas)
    monkeypatch.setattr(bm, "load_frame_cache", fake_load_frame_cache)

    model, source = manager._setup_model_and_data()

    assert isinstance(source, FeatureFrame)
    assert seen["path"] == Path("dummy_frame_cache")


def test_manager_forecast_via_frame_cache(
    monkeypatch, manager_partition_dict, targets
):
    """End-to-end forecast from the frame cache yields PredictionFrames with no pandas."""
    from views_frames import PredictionFrame

    manager = make_manager(_frame_config(run_type="forecast"), manager_partition_dict)
    manager._data_format = DATA_FORMAT_FEATURE_FRAME
    manager._cached_frame_path = Path("dummy_frame_cache")

    monkeypatch.setattr(bm, "read_dataframe", _no_pandas)
    monkeypatch.setattr(
        bm, "load_frame_cache", lambda path: make_dummy_ff(time_range=range(110, 126))
    )

    result = manager._forecast_model_artifact()

    assert set(result.keys()) == set(targets)
    for target_key in targets:
        assert isinstance(result[target_key], PredictionFrame)


def test_manager_evaluate_sweep_via_frame_cache(
    monkeypatch, manager_partition_dict, targets
):
    """The sweep read-site (:125) also serves the frame cache, no pandas."""
    manager = make_manager(
        _frame_config(run_type="eval", sequence_numbers=2), manager_partition_dict
    )
    manager._data_format = DATA_FORMAT_FEATURE_FRAME
    manager._cached_frame_path = Path("dummy_frame_cache")

    monkeypatch.setattr(bm, "read_dataframe", _no_pandas)
    monkeypatch.setattr(
        bm, "load_frame_cache", lambda path: make_dummy_ff(time_range=range(110, 126))
    )

    model, _ = manager._setup_model_and_data()
    result = manager._evaluate_sweep(eval_type="temporal", model=model)

    assert set(result.keys()) == set(targets)
    for target_key in targets:
        assert len(result[target_key]) == 2


def test_manager_frame_declared_missing_cache_fails_loud(
    monkeypatch, manager_partition_dict, targets
):
    """A frames-declared model whose cache was never populated must fail loud —
    never silently fall back to the pandas path (pipeline-core register C-214)."""
    manager = make_manager(_frame_config(), manager_partition_dict)
    manager._data_format = DATA_FORMAT_FEATURE_FRAME
    manager._cached_frame_path = None  # declared frame, but cache absent

    monkeypatch.setattr(
        bm, "read_dataframe", lambda path: pytest.fail("silent pandas fallback on frame model")
    )
    monkeypatch.setattr(
        bm, "load_frame_cache", lambda path: pytest.fail("loader reached with no cache path")
    )

    with pytest.raises(RuntimeError):
        manager._load_source()


def test_manager_legacy_format_never_calls_frame_loader(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    """An explicitly dataframe-declared model uses read_dataframe and never the
    frame loader — the branch does not degrade legacy behavior."""
    manager = make_manager(
        _frame_config(algorithm="ZeroModel", run_type="eval"), manager_partition_dict
    )
    manager._data_format = DATA_FORMAT_DATAFRAME

    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)
    monkeypatch.setattr(
        bm, "load_frame_cache", lambda path: pytest.fail("frame loader called on dataframe model")
    )

    model, source = manager._setup_model_and_data()

    assert isinstance(model, ZeroModel)
    assert source.shape == manager_df.shape


def test_manager_default_format_is_dataframe(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    """With no _data_format attribute set at all, the manager defaults to the
    pandas path (byte-identical legacy behavior)."""
    manager = make_manager(
        _frame_config(algorithm="ZeroModel", run_type="eval"), manager_partition_dict
    )
    # deliberately do NOT set manager._data_format

    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)
    monkeypatch.setattr(
        bm, "load_frame_cache", lambda path: pytest.fail("frame loader called by default")
    )

    _, source = manager._setup_model_and_data()

    assert source.shape == manager_df.shape


def test_manager_forecast_respects_algorithm_choice(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    config_zero = {
        "run_type": "forecast",
        "level": "pgm",
        "algorithm": "ZeroModel",
        "regression_targets": targets,
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
        "regression_targets": targets,
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
        "regression_targets": targets,
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
        "regression_targets": targets,
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
        "regression_targets": targets,
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
        "regression_targets": targets,
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
        "regression_targets": targets,
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


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Tests: streaming evaluation, one origin at a time (views-baseline#120)
# ---------------------------------------------------------------------

_STREAM_CONFIG = {
    "run_type": "eval",
    "level": "pgm",
    "algorithm": "ConflictologyModel",
    "window_months": 6,
    "n_samples": 16,
    "seed": 42,
    "steps": [*range(1, 37)],
    "time_steps": 36,
    "prediction_format": "prediction_frame",
    "sequence_numbers": 3,
}


def _stream_manager(monkeypatch, manager_df, manager_partition_dict, targets):
    manager = make_manager({**_STREAM_CONFIG, "regression_targets": targets},
                           manager_partition_dict)
    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)
    return manager


def _spy_on_predict(manager, before_predict):
    """Make the fitted model call ``before_predict(sequence_number)`` before each predict."""
    real_setup = manager._setup_model_and_data

    def setup_with_spy():
        model, df = real_setup()
        real_predict = model.predict

        def predict(**kw):
            before_predict(kw["sequence_number"])
            return real_predict(**kw)

        model.predict = predict
        return model, df

    manager._setup_model_and_data = setup_with_spy


def test_streaming_override_matches_pipeline_cores_hook():
    """pipeline-core calls the hook by name with these parameters. If it renames or
    re-signs it, the override silently stops being called and the batch-then-emit default
    runs again (the 93 GB OOM, C-54), while every other streaming test, which calls the
    override directly, stays green (C-44)."""
    import inspect

    from views_pipeline_core.managers.model import ForecastingModelManager

    name = "_evaluate_model_artifact_streaming"
    assert name in vars(ForecastingModelManager), "pipeline-core no longer defines the hook"
    base = list(inspect.signature(getattr(ForecastingModelManager, name)).parameters)
    ours = list(inspect.signature(getattr(bm.BaselineForecastingModelManager, name)).parameters)
    assert base == ours == ["self", "eval_type", "artifact_name", "origin_sink"]
    assert name in vars(bm.BaselineForecastingModelManager)


def test_streaming_hands_each_origin_to_the_sink_before_predicting_the_next(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    """Origin i reaches the sink before origin i+1 is predicted: nothing is accumulated
    and emitted afterwards (which is what pipeline-core's default does, #120)."""
    manager = _stream_manager(monkeypatch, manager_df, manager_partition_dict, targets)
    events = []
    _spy_on_predict(manager, lambda i: events.append(("predict", i)))

    def sink(i, pf_dict):
        assert set(pf_dict) == set(targets)
        events.append(("sink", i))

    manager._evaluate_model_artifact_streaming("temporal", None, origin_sink=sink)
    n = _STREAM_CONFIG["sequence_numbers"]
    assert events == [e for i in range(n) for e in (("predict", i), ("sink", i))]


def test_streaming_never_calls_the_batch_path(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    manager = _stream_manager(monkeypatch, manager_df, manager_partition_dict, targets)

    def no_batch(*a, **k):
        raise AssertionError("streaming evaluation must not build the batch dict")

    manager._evaluate_model_artifact = no_batch
    manager._generate_predictions = no_batch
    calls = []
    manager._evaluate_model_artifact_streaming(
        "temporal", None, origin_sink=lambda i, d: calls.append(i)
    )
    assert calls == list(range(_STREAM_CONFIG["sequence_numbers"]))


def test_streaming_frees_each_origin_before_predicting_the_next(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    """Peak memory is ONE origin: by the time origin i+1 is predicted, origin i is
    unreachable. The sink leaves its dict intact and keeps only weak references, so this
    tests the manager's own release, not the sink's."""
    import gc
    import weakref

    manager = _stream_manager(monkeypatch, manager_df, manager_partition_dict, targets)
    refs, alive_at_next_predict = [], []

    def check(_seq):
        gc.collect()
        alive_at_next_predict.extend(r for r in refs if r() is not None)

    _spy_on_predict(manager, check)

    def sink(i, pf_dict):  # saves nothing, drops nothing
        refs.extend(weakref.ref(pf) for pf in pf_dict.values())

    manager._evaluate_model_artifact_streaming("temporal", None, origin_sink=sink)
    assert len(refs) == _STREAM_CONFIG["sequence_numbers"] * len(targets)
    assert alive_at_next_predict == []


def test_streaming_output_equals_batch_output(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    """Values, index and identifiers equal the batch path's, origin by origin, so results
    are comparable across the change."""
    manager = _stream_manager(monkeypatch, manager_df, manager_partition_dict, targets)
    batch = manager._evaluate_model_artifact(eval_type="temporal")
    streamed = {}
    manager._evaluate_model_artifact_streaming(
        "temporal", None,
        origin_sink=lambda i, d: [streamed.setdefault(t, []).append(pf) for t, pf in d.items()],
    )
    assert set(streamed) == set(batch)
    for t in batch:
        assert len(streamed[t]) == len(batch[t]) == _STREAM_CONFIG["sequence_numbers"]
        for a, b in zip(streamed[t], batch[t]):
            np.testing.assert_array_equal(np.asarray(a.values), np.asarray(b.values))
            np.testing.assert_array_equal(np.asarray(a.index.time), np.asarray(b.index.time))
            np.testing.assert_array_equal(np.asarray(a.index.unit), np.asarray(b.index.unit))


def test_streaming_predictions_have_the_configured_horizon(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    """Absolute check: equality with the batch path cannot catch a defect both share."""
    manager = _stream_manager(monkeypatch, manager_df, manager_partition_dict, targets)
    seen = []
    manager._evaluate_model_artifact_streaming(
        "temporal", None, origin_sink=lambda i, d: seen.extend(d.values())
    )
    assert seen
    for pf in seen:
        assert len(set(np.asarray(pf.index.time).tolist())) == _STREAM_CONFIG["time_steps"]


def test_streaming_sets_the_timestamp_from_the_named_artifact(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    manager = _stream_manager(monkeypatch, manager_df, manager_partition_dict, targets)
    manager._evaluate_model_artifact_streaming(
        "temporal", "calibration_model_20250505_090909.pkl", origin_sink=lambda i, d: None
    )
    assert manager._config_manager.get_combined_config()["timestamp"] == "20250505_090909"


def test_streaming_propagates_sink_errors(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    """In production the sink writes to disk; a failed write must stop the run, never be
    skipped."""
    manager = _stream_manager(monkeypatch, manager_df, manager_partition_dict, targets)

    def sink(i, d):
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError, match="disk full"):
        manager._evaluate_model_artifact_streaming("temporal", None, origin_sink=sink)


def test_streaming_rejects_a_predict_that_does_not_return_a_dict(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    manager = _stream_manager(monkeypatch, manager_df, manager_partition_dict, targets)
    real_setup = manager._setup_model_and_data

    def setup_returning_list():
        model, df = real_setup()
        model.predict = lambda **kw: ["not", "a", "dict"]
        return model, df

    manager._setup_model_and_data = setup_returning_list
    sink_calls = []
    with pytest.raises(TypeError, match="expected dict"):
        manager._evaluate_model_artifact_streaming(
            "temporal", None, origin_sink=lambda i, d: sink_calls.append(i)
        )
    assert sink_calls == []


def test_streaming_logs_the_process_peak_rss_in_gib(
    monkeypatch, manager_df, manager_partition_dict, targets, caplog
):
    """Logged after every origin and once at the end, in GiB, from RUSAGE_SELF."""
    import logging
    import resource as real_resource

    manager = _stream_manager(monkeypatch, manager_df, manager_partition_dict, targets)
    monkeypatch.setattr(bm.sys, "platform", "linux")
    monkeypatch.setattr(
        bm.resource, "getrusage",
        lambda who: SimpleNamespace(
            ru_maxrss=3 * 2**20 if who == real_resource.RUSAGE_SELF else 0
        ),
    )
    with caplog.at_level(logging.INFO, logger=bm.__name__):
        manager._evaluate_model_artifact_streaming(
            "temporal", None, origin_sink=lambda i, d: None
        )
    msgs = [r.getMessage() for r in caplog.records if "peak RSS" in r.getMessage()]
    n = _STREAM_CONFIG["sequence_numbers"]
    assert len(msgs) == n + 1
    assert all("peak RSS so far 3.00 GiB" in m for m in msgs)
    assert msgs[-1].startswith(f"Streamed {n} origins;")
