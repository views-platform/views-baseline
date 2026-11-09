"""No committed graphics (#144). Notebooks are committed without outputs: the plots are drawn when
a notebook is run, never stored. Their markdown carries no image, and the repository tracks no
image file.

Broad on purpose: one list of graphic file extensions and one of HTML elements that embed
content serve both the markdown scan and the tracked-file check, and the tests below try every
entry of both lists.
"""

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
GRAPHIC_EXTENSIONS = ["png", "jpg", "jpeg", "gif", "svg", "webp", "bmp", "tif", "tiff", "pdf",
                      "eps", "ico", "avif", "heic"]
EMBEDDING_TAGS = ["img", "svg", "image", "object", "iframe", "embed", "picture", "source",
                  "video", "canvas"]
GRAPHIC_FILE = re.compile(r"\.(" + "|".join(GRAPHIC_EXTENSIONS) + r")\b", re.IGNORECASE)
IMAGE = re.compile(
    GRAPHIC_FILE.pattern
    + r"|<(" + "|".join(EMBEDDING_TAGS) + r")\b"
    + r"|url\(|data:image|!\[[^\]]*\][\[(]",
    re.IGNORECASE)


def is_graphic_file(path: str) -> bool:
    return Path(path).suffix.lower().lstrip(".") in GRAPHIC_EXTENSIONS


def cells_with_images(cells: list) -> list[int]:
    """Indices of cells that carry an image: an attachment, or a markdown or raw cell whose text
    matches IMAGE. Code cells are not scanned (code like ``stats.norm.pdf(x)`` is fine; what
    code draws is an output, and outputs are refused separately)."""
    return [i for i, c in enumerate(cells)
            if c.get("attachments")
            or (c.get("cell_type") != "code" and IMAGE.search("".join(c.get("source", ""))))]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_notebook_has_no_outputs(path):
    cells = json.loads(path.read_text())["cells"]
    stored = [i for i, c in enumerate(cells)
              if c.get("outputs") or c.get("execution_count") is not None]
    assert not stored, f"{path.name}: clear the outputs of cells {stored} before committing."


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_notebook_carries_no_images(path):
    found = cells_with_images(json.loads(path.read_text())["cells"])
    assert not found, f"{path.name}: cells {found} carry an image."


def test_there_is_a_notebook_to_check():
    assert NOTEBOOKS


@pytest.mark.parametrize("source", [
    *[f"see figures/plot.{e}" for e in GRAPHIC_EXTENSIONS],
    *[f'<{t} src="p">' for t in EMBEDDING_TAGS],
    "<IMG\nsrc='x'>", "![fig](figures/plot)", "![fig][p]\n\n[p]: figures/plot",
    "![](data:image/png;base64,AA)", "<div style='background-image:url(p)'>",
])
def test_markdown_with_an_image_is_caught(source):
    assert cells_with_images([{"cell_type": "markdown", "source": source}]) == [0]


def test_attachments_are_caught_in_any_cell():
    assert cells_with_images([{"cell_type": "markdown", "source": "x",
                               "attachments": {"a.png": {}}}]) == [0]


@pytest.mark.parametrize("cell", [
    {"cell_type": "markdown", "source": "P(any) [ADR-024](../docs/ADRs/024.md), x < 1."},
    {"cell_type": "code", "source": "p = stats.norm.pdf(x)\nplt.savefig('fig.png')"},
])
def test_plain_text_and_code_are_left_alone(cell):
    assert cells_with_images([cell]) == []


@pytest.mark.parametrize("extension", GRAPHIC_EXTENSIONS)
def test_the_file_check_knows_each_extension(extension):
    assert is_graphic_file(f"reports/x/plot.{extension.upper()}")
    assert not is_graphic_file(f"reports/x/plot.{extension}.txt")


def test_no_image_files_are_tracked():
    """Skipped only outside a git checkout (e.g. an exported copy); any git error fails."""
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                           check=True).stdout.splitlines()
    graphics = [f for f in files if is_graphic_file(f)]
    assert not graphics, f"tracked graphic files: {graphics}"
