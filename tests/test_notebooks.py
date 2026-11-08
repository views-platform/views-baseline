"""Notebooks are committed without outputs: the plots are drawn when a notebook is run, never
stored (no committed graphics; #144)."""

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_notebook_has_no_outputs(path):
    cells = json.loads(path.read_text())["cells"]
    stored = [i for i, c in enumerate(cells)
              if c.get("outputs") or c.get("execution_count") is not None]
    assert not stored, f"{path.name}: clear the outputs of cells {stored} before committing."


def test_there_is_a_notebook_to_check():
    assert NOTEBOOKS
