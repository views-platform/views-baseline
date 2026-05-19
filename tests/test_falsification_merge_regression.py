"""
Falsification test stubs for merge-regression audit (2026-05-19).

F-1 (HARD): CIC lies about _setup_model_and_data timestamp behavior.
F-2 (SOFT): New FileNotFoundError precondition in evaluate/forecast.
F-5 (HARD): ADR-016 ships incorrect test count.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from conftest import make_manager


BASE_CONFIG = {
    "run_type": "calibration",
    "algorithm": "LocfModel",
    "level": "pgm",
    "time_steps": 36,
    "targets": ["synth_target"],
    "regression_targets": ["synth_target"],
    "regression_point_metrics": ["MSE"],
}

PARTITION = {"test": (120, 125)}


class TestF1CicDrift:
    """
    F-1: CIC says _setup_model_and_data stamps config['timestamp'].
    The code no longer does this. Verify the CIC statement is false.
    """

    def test_setup_model_and_data_does_not_stamp_timestamp(self, monkeypatch):
        import views_baseline.manager.baseline_manager as bm
        import pandas as pd

        mgr = make_manager(BASE_CONFIG.copy(), PARTITION)
        monkeypatch.setattr(bm, "read_dataframe", lambda path: pd.DataFrame())

        ts_before = mgr.config.get("timestamp")

        with patch.object(
            type(mgr), "_get_cached_data_path",
            return_value=Path("dummy"),
        ):
            try:
                mgr._setup_model_and_data()
            except Exception:
                pass

        ts_after = mgr.config.get("timestamp")
        assert ts_after == ts_before, (
            "CIC claims _setup_model_and_data stamps config['timestamp'] "
            "but it should no longer do so after the fix. "
            f"Before: {ts_before!r}, After: {ts_after!r}"
        )


class TestF2NewPrecondition:
    """
    F-2: evaluate/forecast now call get_latest_model_artifact_path()
    which raises FileNotFoundError when no artifact exists.
    This is a new failure mode not present before the fix.
    """

    def test_evaluate_raises_when_no_artifact_exists(self):
        mgr = make_manager(BASE_CONFIG.copy(), PARTITION)
        mgr._model_path = SimpleNamespace(
            artifacts=Path("nonexistent_dir"),
            get_latest_model_artifact_path=MagicMock(
                side_effect=FileNotFoundError(
                    "No model artifacts found for run type 'calibration'"
                )
            ),
        )

        raised = False
        try:
            mgr._evaluate_model_artifact(eval_type="standard")
        except FileNotFoundError:
            raised = True

        assert raised, (
            "Expected FileNotFoundError when no artifact exists. "
            "This is a new precondition introduced by the fix — "
            "pre-fix, evaluate ran without touching artifact paths."
        )

    def test_forecast_raises_when_no_artifact_exists(self):
        mgr = make_manager(BASE_CONFIG.copy(), PARTITION)
        mgr._model_path = SimpleNamespace(
            artifacts=Path("nonexistent_dir"),
            get_latest_model_artifact_path=MagicMock(
                side_effect=FileNotFoundError(
                    "No model artifacts found for run type 'calibration'"
                )
            ),
        )

        raised = False
        try:
            mgr._forecast_model_artifact()
        except FileNotFoundError:
            raised = True

        assert raised, (
            "Expected FileNotFoundError when no artifact exists. "
            "This is a new precondition introduced by the fix — "
            "pre-fix, forecast ran without touching artifact paths."
        )
