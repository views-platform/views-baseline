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


# Broad on purpose: any image file name, any HTML element that embeds content, a CSS url(),
# a data: URI, or a markdown image of either style. Notebook markdown here is plain text.
IMAGE = re.compile(
    r"\.(png|jpe?g|gif|svg|webp|bmp|tiff?|pdf)\b"
    r"|<(img|svg|image|object|iframe|embed|picture|source|video|canvas)\b"
    r"|url\(|data:image|!\[[^\]]*\][\[(]",
    re.IGNORECASE)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_notebook_carries_no_images(path):
    """No graphics in the cells either: no attachments, no markdown or HTML image (to a file
    or inline), no inline SVG."""
    cells = json.loads(path.read_text())["cells"]
    found = [i for i, c in enumerate(cells)
             if c.get("attachments") or IMAGE.search("".join(c.get("source", "")))]
    assert not found, f"{path.name}: cells {found} carry an image."


@pytest.mark.parametrize("source", [
    "![fig](figures/plot.png)", "![fig][p]\n\n[p]: figures/plot", '<IMG\nsrc="x">',
    "<svg width='1'></svg>", "![](data:image/png;base64,AA)", '<object data="p">',
    '<iframe src="p">', '<embed src="p">', '<picture><source srcset="p">',
    "<div style='background-image:url(p)'>", "see results/plot.png",
])
def test_the_image_pattern_catches_each_form(source):
    assert IMAGE.search(source)


def test_the_image_pattern_leaves_plain_text_alone():
    assert not IMAGE.search("The map shows P(any) [ADR-024](../docs/ADRs/024.md), x < 1.")


def test_no_image_files_are_tracked():
    """No committed graphics anywhere in the repository (#144)."""
    import subprocess

    root = Path(__file__).resolve().parents[1]
    try:
        files = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True,
                               check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    images = [f for f in files if re.search(r"\.(png|jpe?g|gif|svg|webp|bmp|tiff?)$", f, re.I)]
    assert not images, f"tracked image files: {images}"


def test_there_is_a_notebook_to_check():
    assert NOTEBOOKS
