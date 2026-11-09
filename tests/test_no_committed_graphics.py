"""No committed graphics (#144): plots are drawn when a notebook is run, never stored.

Three checks over everything git tracks:
- no tracked file is a graphic (by extension);
- no tracked notebook has outputs or attachments, and no cell embeds an image;
- no tracked markdown file embeds an image.

"Embeds" means markup that makes a viewer render an image: a markdown image (``![…]``), an HTML
element that shows content, a CSS ``url(``, or a ``data:image`` URI. Mentioning a file name
does not embed anything (a committed graphic file is caught by the first check), so prose like
``poisson.pdf`` and code like ``stats.norm.pdf(x)`` pass. Code cells are checked only for HTML
elements and ``data:`` images, the ways code source itself can carry a picture; what code
draws is an output, and outputs are refused.

The lists below are the whole policy; the tests try every entry.
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
_TAGS = r"<(" + "|".join(EMBEDDING_TAGS) + r")\b"
MARKUP_EMBED = re.compile(_TAGS + r"|url\(|data:image|!\[", re.IGNORECASE)
CODE_EMBED = re.compile(_TAGS + r"|data:image", re.IGNORECASE)


def _tracked() -> list[str] | None:
    """Every tracked path (exact, via -z), or None outside a git checkout."""
    if not (ROOT / ".git").exists():
        return None
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout
    return [p for p in out.split("\0") if p]


TRACKED = _tracked()
NOTEBOOKS = ([p for p in TRACKED if p.endswith(".ipynb")] if TRACKED is not None
             else [str(p.relative_to(ROOT)) for p in (ROOT / "notebooks").glob("*.ipynb")])
MARKDOWN = [p for p in TRACKED if p.endswith(".md")] if TRACKED is not None else []


CODE_SPANS = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)


def without_code(text: str) -> str:
    """Markdown without its code blocks and inline code: those render as text, never as an
    image, so prose can name `url(` or `data:image` when describing this very policy."""
    return CODE_SPANS.sub("", text)


def is_graphic_file(path: str) -> bool:
    return Path(path).suffix.lower().lstrip(".") in GRAPHIC_EXTENSIONS


def cells_with_images(cells: list) -> list[int]:
    """Indices of cells that carry an image: an attachment, or embedding markup in the source
    (all markup forms for markdown and raw cells; HTML and data: images for code cells)."""
    found = []
    for i, c in enumerate(cells):
        source = "".join(c.get("source", ""))
        if c.get("cell_type") == "code":
            embeds = CODE_EMBED.search(source)
        else:
            embeds = MARKUP_EMBED.search(without_code(source))
        if c.get("attachments") or embeds:
            found.append(i)
    return found


def test_no_graphic_files_are_tracked():
    if TRACKED is None:
        pytest.skip("not a git checkout")
    graphics = [f for f in TRACKED if is_graphic_file(f)]
    assert not graphics, f"tracked graphic files: {graphics}"


def test_there_is_a_notebook_to_check():
    assert NOTEBOOKS


@pytest.mark.parametrize("path", NOTEBOOKS)
def test_notebook_has_no_outputs(path):
    cells = json.loads((ROOT / path).read_text())["cells"]
    stored = [i for i, c in enumerate(cells)
              if c.get("outputs") or c.get("execution_count") is not None]
    assert not stored, f"{path}: clear the outputs of cells {stored} before committing."


@pytest.mark.parametrize("path", NOTEBOOKS)
def test_notebook_carries_no_images(path):
    found = cells_with_images(json.loads((ROOT / path).read_text())["cells"])
    assert not found, f"{path}: cells {found} carry an image."


def test_markdown_files_embed_no_images():
    if TRACKED is None:
        pytest.skip("not a git checkout")
    found = [p for p in MARKDOWN if MARKUP_EMBED.search(without_code((ROOT / p).read_text()))]
    assert not found, f"markdown files that embed an image: {found}"


# ── the policy itself ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("extension", GRAPHIC_EXTENSIONS)
def test_the_file_check_knows_each_extension(extension):
    assert is_graphic_file(f"reports/x/plot.{extension.upper()}")
    assert not is_graphic_file(f"reports/x/plot.{extension}.txt")


@pytest.mark.parametrize("source", [
    *[f'<{t} src="p">' for t in EMBEDDING_TAGS], '<input type="image" src="p">',
    "<IMG\nsrc='x'>", "![fig](figures/plot)", "![fig][p]\n\n[p]: figures/plot",
    "![fig]\n\n[fig]: figures/plot", "![](data:image/png;base64,AA)",
    "<div style='background-image:url(p)'>",
])
def test_markdown_embedding_is_caught(source):
    assert MARKUP_EMBED.search(without_code(f"Some prose. {source} More prose `code`."))
    assert cells_with_images([{"cell_type": "markdown", "source": source}]) == [0]
    assert cells_with_images([{"cell_type": "raw", "source": source}]) == [0]


@pytest.mark.parametrize("source", [
    "%%html\n<img src='data:image/png;base64,AA'>", "display(SVG('<svg width=\"1\"/>'))",
    "HTML('<iframe src=\"p\">')",
])
def test_code_that_embeds_an_image_is_caught(source):
    assert cells_with_images([{"cell_type": "code", "source": source}]) == [0]


def test_attachments_are_caught_in_any_cell():
    for kind in ("markdown", "raw", "code"):
        assert cells_with_images([{"cell_type": kind, "source": "x",
                                   "attachments": {"a.png": {}}}]) == [0]


@pytest.mark.parametrize("cell", [
    {"cell_type": "markdown", "source": "P(any), see [ADR-024](../docs/ADRs/024.md), x < 1."},
    {"cell_type": "markdown", "source": "`scipy.stats.poisson.pdf`, `np.finfo(f).eps`, out.png"},
    {"cell_type": "code", "source": "p = stats.norm.pdf(x)\nplt.savefig('fig.png')\nurl(x)"},
    {"cell_type": "markdown", "source": "Refused: `url(`, `data:image`, and\n```\n![x](p)\n```"},
])
def test_names_and_code_are_left_alone(cell):
    assert cells_with_images([cell]) == []
