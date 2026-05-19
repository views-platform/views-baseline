"""
Falsification test stubs for ship-readiness audit (2026-05-19).

F-1 (SOFT): artifact_name parameter silently ignored — wrong timestamp
            when user specifies a non-latest artifact.
F-4 (SOFT): PR body claims 85 tests, ADR says 75/75, actual is 75+6.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from conftest import MANAGER_BASE_CONFIG, MANAGER_PARTITION, make_manager

LATEST_TS = "20260201_120000"
SPECIFIED_TS = "20260101_090000"
LATEST_PATH = Path(f"dummy_artifacts_path/calibration_model_{LATEST_TS}.pkl")
SPECIFIED_ARTIFACT = f"calibration_model_{SPECIFIED_TS}.pkl"


class TestF1ArtifactNameHonored:
    """
    F-1: When artifact_name is provided, the baseline manager should
    extract the timestamp from the specified artifact, not the latest.
    """

    @pytest.mark.parametrize("method,kwargs", [
        ("_evaluate_model_artifact", {"eval_type": "standard"}),
        ("_forecast_model_artifact", {}),
    ])
    def test_uses_specified_artifact_timestamp(self, method, kwargs):
        mgr = make_manager(MANAGER_BASE_CONFIG.copy(), MANAGER_PARTITION)
        mgr._model_path.get_latest_model_artifact_path = lambda run_type: LATEST_PATH
        mgr._model_path.artifacts = Path("dummy_artifacts_path")

        if method == "_forecast_model_artifact":
            with patch.object(mgr, "_setup_model_and_data") as mock_setup:
                mock_model = MagicMock()
                mock_model.predict.return_value = MagicMock()
                mock_setup.return_value = (mock_model, MagicMock())
                getattr(mgr, method)(artifact_name=SPECIFIED_ARTIFACT, **kwargs)
        else:
            with patch.object(mgr, "_setup_model_and_data") as mock_setup, \
                 patch.object(mgr, "_generate_predictions") as mock_preds:
                mock_setup.return_value = (MagicMock(), MagicMock())
                mock_preds.return_value = []
                getattr(mgr, method)(artifact_name=SPECIFIED_ARTIFACT, **kwargs)

        ts = mgr.config["timestamp"]
        assert ts == SPECIFIED_TS, (
            f"artifact_name='{SPECIFIED_ARTIFACT}' was provided but timestamp "
            f"is '{ts}' (from latest artifact), not '{SPECIFIED_TS}' "
            f"(from specified artifact). artifact_name is silently ignored."
        )
