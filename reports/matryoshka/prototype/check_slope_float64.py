"""Run check_slope.py in float64, where the nudges are precise enough for a tight tolerance.

The prototype works in float32 (doll_counts.py and forecast_probability.py). This copies the
prototype into a temporary folder, replaces float32 with float64 in those two files, and runs
check_slope.py there with rtol 1e-6 and no absolute slack. The copy is the only thing changed; the
files in this folder are untouched. (LOG.md 2026-11-06: in float32 the two smallest slopes could not
meaningfully fail the loose check.)
Run: python check_slope_float64.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

with tempfile.TemporaryDirectory() as tmp:
    for f in HERE.glob("*.py"):
        shutil.copy(f, tmp)
    for name in ("doll_counts.py", "forecast_probability.py"):
        path = Path(tmp) / name
        path.write_text(path.read_text().replace("float32", "float64"))
    result = subprocess.run([sys.executable, "check_slope.py", "1e-6", "0", "1e-4"], cwd=tmp)
sys.exit(result.returncode)
