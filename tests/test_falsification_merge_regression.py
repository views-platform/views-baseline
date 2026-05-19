"""
Falsification test stubs for merge-regression audit (2026-05-19).

F-1 (HARD): CIC drift — _setup_model_and_data no longer stamps timestamp.
F-2 (SOFT): New FileNotFoundError precondition in evaluate/forecast.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from conftest import make_manager, MANAGER_BASE_CONFIG, MANAGER_PARTITION


class TestCicDrift:

    def test_setup_model_and_data_does_not_stamp_timestamp(self, monkeypatch):
        import views_baseline.manager.baseline_manager as bm
        import pandas as pd

        mgr = make_manager(MANAGER_BASE_CONFIG.copy(), MANAGER_PARTITION)
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

        assert mgr.config.get("timestamp") == ts_before


class TestFileNotFoundPrecondition:

    @pytest.mark.parametrize("method,kwargs", [
        ("_evaluate_model_artifact", {"eval_type": "standard"}),
        ("_forecast_model_artifact", {}),
    ])
    def test_raises_when_no_artifact_exists(self, method, kwargs):
        mgr = make_manager(MANAGER_BASE_CONFIG.copy(), MANAGER_PARTITION)
        mgr._model_path = SimpleNamespace(
            artifacts=Path("nonexistent_dir"),
            get_latest_model_artifact_path=MagicMock(
                side_effect=FileNotFoundError(
                    "No model artifacts found for run type 'calibration'"
                )
            ),
        )

        with pytest.raises(FileNotFoundError):
            getattr(mgr, method)(**kwargs)
