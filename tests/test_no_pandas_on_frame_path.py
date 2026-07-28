"""Permanent tripwire (issue #73): a FRAME-FED views-baseline run must import **no pandas**.

Since epic #47 the model layer is numpy-on-FeatureFrame, and since views-pipeline-core #320
(lazy pandas in the base `ForecastingModelManager` / `read_dataframe`) the framework floor no
longer drags pandas in at import. Together that makes a FeatureFrame-fed fit→predict genuinely
pandas-free end to end. This test locks that in and fails loudly if either side regresses
(e.g. pipeline-core re-adds a module-level `import pandas`, or a baseline model starts touching
pandas on the frame path).

**Why a subprocess:** the assertion is on the *global* `sys.modules`, so it is only meaningful
in a clean interpreter. Inside the normal pytest process pandas is already loaded — the shared
`conftest.make_dummy_ff` builds its FeatureFrame *from* a pandas DataFrame, and several manager
tests import `read_dataframe`. So the check runs in a fresh `python -c` subprocess that builds
its FeatureFrame from pure numpy (no pandas fixture) and asserts pandas was never imported.

Feeding a `pd.DataFrame` instead (the ADR-019 dual-input escape hatch) *does* import pandas
lazily at the boundary — by design; this test deliberately exercises only the frame path.
"""

import subprocess
import sys

import pytest

# The platform leaves must be importable for the subprocess to run; skip cleanly where absent.
pytest.importorskip("views_frames")
pytest.importorskip("views_pipeline_core")

_FRAME_FED_RUN = r"""
import sys
import numpy as np
from views_frames import FeatureFrame, SpatioTemporalIndex, SpatialLevel

# Build a FeatureFrame from pure numpy — NO pandas anywhere in the fixture.
entities = [1, 2]
times = list(range(110, 160))
time = np.array([t for t in times for _ in entities], dtype=np.int64)
unit = np.array([e for _ in times for e in entities], dtype=np.int64)
block = np.empty((len(time), 2), dtype=np.float64)
block[:, 0] = time * 10 + unit
block[:, 1] = time * 100 + unit
ff = FeatureFrame.from_2d(
    block, SpatioTemporalIndex(time=time, unit=unit, level=SpatialLevel.PGM), ["y1", "y2"]
)

# Importing the manager used to pull pandas via the pipeline-core base class (pre #320).
from views_baseline.manager.baseline_manager import BaselineForecastingModelManager  # noqa: F401
from views_baseline.model.models.point import ZeroModel, LocfModel, AverageModel
from views_baseline.model.models.distributional.conflictology import ConflictologyModel

partition = {"test": (150, 155)}
runs = [
    (ZeroModel, {}),
    (LocfModel, {}),
    (AverageModel, {"window_months": 12}),
    (ConflictologyModel, {"window_months": 12, "n_samples": 8, "seed": 42}),
]
for model_cls, kw in runs:
    model = model_cls(targets=["y1", "y2"], partition_dict=partition, loa="pgm", **kw).fit(ff)
    model.predict(df=ff, sequence_number=0, output_length=3)

leaked = [m for m in sys.modules if m == "pandas" or m.startswith("pandas.")]
assert not leaked, f"pandas was imported during a frame-fed run: {leaked}"
print("PANDAS_FREE_OK")
"""


def test_frame_fed_run_imports_no_pandas():
    """A full FeatureFrame-fed fit→predict across all model kinds imports no pandas.

    Runs in a fresh interpreter so the `sys.modules` assertion is uncontaminated by the rest
    of the suite. Guards both regressions: a baseline model touching pandas on the frame path,
    or the pipeline-core floor re-introducing a module-level pandas import (views-frames #320).
    """
    result = subprocess.run(
        [sys.executable, "-c", _FRAME_FED_RUN],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "frame-fed run failed or imported pandas.\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert "PANDAS_FREE_OK" in result.stdout, result.stdout
