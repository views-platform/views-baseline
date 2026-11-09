"""No committed graphics (#144): plots are drawn when a notebook is run, never stored.

Over everything git tracks:
- no tracked file is a graphic (by extension);
- no notebook has outputs or attachments, and nothing in it embeds an image;
- no markdown or HTML file embeds an image.

"Embeds" means markup that makes a viewer render an image: a markdown image, an HTML element
that shows content, a CSS url reference, or a base64 image URI. The scan is strict: it reads
every character, code spans included, so there is no markdown parsing to get wrong. Prose that
describes these forms must name them in words, as this docstring does. Mentioning a file name
embeds nothing (a committed graphic file is caught by extension), so prose like "poisson.pdf"
and code like ``stats.norm.pdf(x)`` pass. Code cells are checked for HTML elements and base64
images only, the ways code source itself can carry a picture; what code draws is an output,
and outputs are refused.

Outside a git checkout (an exported copy) only the notebooks in ``notebooks/`` and the policy
tests run; the checks that need the git file list are reported as skipped.
"""

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GRAPHIC_EXTENSIONS = ["png", "apng", "jpg", "jpeg", "gif", "svg", "svgz", "webp", "bmp", "tif",
                      "tiff", "pdf", "eps", "ico", "avif", "heic", "jxl", "psd"]
EMBEDDING_TAGS = ["img", "svg", "image", "object", "iframe", "embed", "picture", "source",
                  "video", "canvas", "input"]
TEXT_EXTENSIONS = ["md", "markdown", "html", "htm"]
# A tag must be followed by a space, ">" or "/", so code like `idx<image.shape` is not a tag.
_TAGS = r"<(" + "|".join(EMBEDDING_TAGS) + r")(?=[\s>/])"
_BASE64_IMAGE = "data" + r":image"
MARKUP_EMBED = re.compile(_TAGS + r"|url\(|" + _BASE64_IMAGE + r"|!\[", re.IGNORECASE)
CODE_EMBED = re.compile(_TAGS + "|" + _BASE64_IMAGE, re.IGNORECASE)


def _git_files() -> tuple[list[str] | None, str | None]:
    """(tracked paths, error). Paths are exact (-z). Outside a checkout: (None, None)."""
    if not (ROOT / ".git").exists():
        return None, None
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        return None, f"git ls-files failed: {e}"
    return [p for p in out.split("\0") if p and (ROOT / p).exists()], None


TRACKED, GIT_ERROR = _git_files()


def _need_git() -> list[str]:
    if GIT_ERROR:
        pytest.fail(GIT_ERROR)
    if TRACKED is None:
        pytest.skip("not a git checkout")
    return TRACKED


def _suffix(path: str) -> str:
    return Path(path).suffix.lower().lstrip(".")


NOTEBOOKS = sorted({p for p in TRACKED or [] if _suffix(p) == "ipynb"}
                   | {str(p.relative_to(ROOT)) for p in (ROOT / "notebooks").glob("*.ipynb")})


def is_graphic_file(path: str) -> bool:
    return _suffix(path) in GRAPHIC_EXTENSIONS


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def notebook_images(nb: dict) -> list[str]:
    """Where a notebook carries an image: cells (by index) and its top-level metadata."""
    found = []
    for i, c in enumerate(nb.get("cells", [])):
        source = "".join(c.get("source", ""))
        pattern = CODE_EMBED if c.get("cell_type") == "code" else MARKUP_EMBED
        if c.get("attachments") or pattern.search(source):
            found.append(f"cell {i}")
    if CODE_EMBED.search(json.dumps(nb.get("metadata", {}))):
        found.append("metadata")
    return found


def test_the_git_file_list_is_readable():
    if GIT_ERROR:
        pytest.fail(GIT_ERROR)


def test_no_graphic_files_are_tracked():
    graphics = [f for f in _need_git() if is_graphic_file(f)]
    assert not graphics, f"tracked graphic files: {graphics}"


def test_there_is_a_notebook_to_check():
    assert NOTEBOOKS


@pytest.mark.parametrize("path", NOTEBOOKS)
def test_notebook_has_no_outputs(path):
    cells = json.loads(_read(path))["cells"]
    stored = [i for i, c in enumerate(cells)
              if c.get("outputs") or c.get("execution_count") is not None]
    assert not stored, f"{path}: clear the outputs of cells {stored} before committing."


@pytest.mark.parametrize("path", NOTEBOOKS)
def test_notebook_carries_no_images(path):
    found = notebook_images(json.loads(_read(path)))
    assert not found, f"{path}: {found} carry an image."


def test_markdown_and_html_files_embed_no_images():
    found = [p for p in _need_git()
             if _suffix(p) in TEXT_EXTENSIONS and MARKUP_EMBED.search(_read(p))]
    assert not found, f"files that embed an image: {found}"


# ── the policy itself ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("extension", GRAPHIC_EXTENSIONS)
def test_the_file_check_knows_each_extension(extension):
    assert is_graphic_file(f"reports/x/plot.{extension.upper()}")
    assert not is_graphic_file(f"reports/x/plot.{extension}.txt")


def test_the_lists_hold_at_least_the_common_forms():
    """The generated tests try every entry, so they cannot notice an entry going missing."""
    assert {"png", "jpg", "jpeg", "gif", "svg", "webp", "tif", "tiff", "pdf", "eps"} <= set(
        GRAPHIC_EXTENSIONS)
    assert {"img", "svg", "image", "object", "iframe", "embed", "picture", "video", "canvas",
            "input"} <= set(EMBEDDING_TAGS)
    assert {"md", "markdown", "html", "htm"} <= set(TEXT_EXTENSIONS)


EMBEDDED = [
    *[f'<{t} src="p">' for t in EMBEDDING_TAGS], *[f"<{t}>" for t in EMBEDDING_TAGS],
    *[f"<{t}/>" for t in EMBEDDING_TAGS], '<input type="image" src="p">', "<IMG\nsrc='x'>",
    "![fig](figures/plot)", "![fig][p]\n\n[p]: figures/plot", "![fig]\n\n[fig]: figures/plot",
    "![](" + _BASE64_IMAGE + "/png;base64,AA)", "<div style='background-image:url(p)'>",
    "Use ``` for fences.\n![plot](p)\nlater\n```", "``a ` b`` ![x](figures/plot.png) `",
    "````\n```\n````\n![x](p)\n```",
]


@pytest.mark.parametrize("source", EMBEDDED)
def test_markdown_embedding_is_caught(source):
    assert MARKUP_EMBED.search(source)
    for kind in ("markdown", "raw"):
        assert notebook_images({"cells": [{"cell_type": kind, "source": source}]}) == ["cell 0"]


@pytest.mark.parametrize("source", [
    "%%html\n<img src='p'>", "display(SVG('<svg width=\"1\"/>'))", "HTML('<iframe src=\"p\">')",
    "Image(url='" + _BASE64_IMAGE + "/png;base64,AA')",
])
def test_code_that_embeds_an_image_is_caught(source):
    assert notebook_images({"cells": [{"cell_type": "code", "source": source}]}) == ["cell 0"]


def test_attachments_and_metadata_images_are_caught():
    for kind in ("markdown", "raw", "code"):
        nb = {"cells": [{"cell_type": kind, "source": "x", "attachments": {"a.png": {}}}]}
        assert notebook_images(nb) == ["cell 0"]
    nb = {"cells": [], "metadata": {"widgets": {"state": "<img src='p'>"}}}
    assert notebook_images(nb) == ["metadata"]


@pytest.mark.parametrize("cell", [
    {"cell_type": "markdown", "source": "P(any), see [ADR-024](../docs/ADRs/024.md), x < 1."},
    {"cell_type": "markdown", "source": "scipy.stats.poisson.pdf, np.finfo(f).eps, out.png"},
    {"cell_type": "code", "source": "p = stats.norm.pdf(x)\nplt.savefig('fig.png')\nurl(x)"},
    {"cell_type": "code", "source": "mask = idx<image.shape[0]\nok = n<source.size"},
])
def test_names_and_code_are_left_alone(cell):
    assert notebook_images({"cells": [cell]}) == []
