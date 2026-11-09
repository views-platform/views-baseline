"""Notebooks are committed without outputs: the plots are drawn when a notebook is run, never
stored (no committed graphics; #144)."""

import json
import re
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_notebook_has_no_outputs(path):
    cells = json.loads(path.read_text())["cells"]
    stored = [i for i, c in enumerate(cells)
              if c.get("outputs") or c.get("execution_count") is not None]
    assert not stored, f"{path.name}: clear the outputs of cells {stored} before committing."


IMAGE = re.compile(r"!\[[^\]]*\]\(|<img\b|<svg\b|data:image", re.IGNORECASE)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_notebook_carries_no_images(path):
    """No graphics in the cells either: no attachments, no markdown or HTML image (to a file
    or inline), no inline SVG."""
    cells = json.loads(path.read_text())["cells"]
    found = [i for i, c in enumerate(cells)
             if c.get("attachments") or IMAGE.search("".join(c.get("source", "")))]
    assert not found, f"{path.name}: cells {found} carry an image."


@pytest.mark.parametrize("source", ["![fig](figures/plot.png)", '<img src="plot.png">',
                                    "<svg width='1'></svg>", "![](data:image/png;base64,AA)"])
def test_the_image_pattern_catches_each_form(source):
    assert IMAGE.search(source)


def test_there_is_a_notebook_to_check():
    assert NOTEBOOKS
