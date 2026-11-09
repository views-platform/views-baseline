"""No committed graphics (#144): plots are drawn when a notebook is run, never stored.

`problems(files, read)` is the whole policy, applied to every file git tracks:
- a graphic file (by extension) is a problem, whether or not it is on disk;
- a notebook with outputs, execution counts or attachments, or that embeds an image in a cell
  or in its metadata, is a problem;
- a markdown, reStructuredText or HTML file that embeds an image is a problem.

"Embeds" means markup that makes a viewer render an image: a markdown image, an RST image or
figure directive, an HTML element that shows content, a CSS url reference, or a base64 image
URI. The scan is strict: it reads every character, code spans included, so there is no markup
parsing to get wrong; prose that describes these forms names them in words, as this docstring
does. Naming a file embeds nothing (a committed graphic is caught by its extension), so prose
like "poisson.pdf" and code like ``stats.norm.pdf(x)`` pass. Code cells are checked for HTML
elements and base64 images only; what code draws is an output, and outputs are refused.

The tests run `problems` on a made-up repository holding one example of each case, so they
check the scan itself, not only what the repository happens to contain today. Outside a git
checkout (an exported copy) the notebooks in ``notebooks/`` are still checked.
"""

import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GRAPHIC_EXTENSIONS = ["png", "apng", "jpg", "jpeg", "jfif", "gif", "svg", "svgz", "webp", "bmp",
                      "tif", "tiff", "pdf", "ps", "eps", "ico", "avif", "heic", "heif", "jxl",
                      "jp2", "psd"]
EMBEDDING_TAGS = ["img", "svg", "image", "object", "iframe", "embed", "picture", "source",
                  "video", "canvas", "input"]
MARKDOWN_EXTENSIONS = ["md", "markdown", "mdx", "qmd", "rmd"]
HTML_EXTENSIONS = ["html", "htm"]
RST_EXTENSIONS = ["rst"]
# A tag must be followed by a space, ">" or "/", so code like `idx<image.shape` is not a tag.
_TAGS = r"<(" + "|".join(EMBEDDING_TAGS) + r")(?=[\s>/])"
_BASE64_IMAGE = "data" + r":image"
_URL = r"\burl\("                                  # `base_url(` is a name, not CSS
HTML_EMBED = re.compile(_TAGS + "|" + _URL + "|" + _BASE64_IMAGE + r"|\bbackground\s*=",
                        re.IGNORECASE)
MARKUP_EMBED = re.compile(HTML_EMBED.pattern + r"|!\[", re.IGNORECASE)
RST_EMBED = re.compile(HTML_EMBED.pattern + r"|\.\.\s+(image|figure)::", re.IGNORECASE)
CODE_EMBED = re.compile(_TAGS + "|" + _BASE64_IMAGE, re.IGNORECASE)


def suffix(path: str) -> str:
    return Path(path).suffix.lower().lstrip(".")


def is_graphic_file(path: str) -> bool:
    return suffix(path) in GRAPHIC_EXTENSIONS


def _strings(value) -> list[str]:
    """Every string inside a JSON value (searched as written, not as JSON-escaped)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def notebook_problems(nb: dict) -> list[str]:
    found = []
    for i, c in enumerate(nb.get("cells", [])):
        if c.get("outputs") or c.get("execution_count") is not None:
            found.append(f"cell {i} has outputs")
        source = "".join(c.get("source", ""))
        pattern = CODE_EMBED if c.get("cell_type") == "code" else MARKUP_EMBED
        if c.get("attachments") or pattern.search(source):
            found.append(f"cell {i} carries an image")
    if any(CODE_EMBED.search(s) for s in _strings(nb.get("metadata", {}))):
        found.append("metadata carries an image")
    return found


def problems(files: list[str], read: Callable[[str], str | None]) -> list[str]:
    """Every policy problem in ``files``. ``read`` returns a file's text, or None if it is not
    on disk (a graphic is still a problem; its content cannot be checked)."""
    found = []
    for f in files:
        kind = suffix(f)
        if is_graphic_file(f):
            found.append(f"{f}: a graphic file")
        if kind not in ["ipynb", *MARKDOWN_EXTENSIONS, *HTML_EXTENSIONS, *RST_EXTENSIONS]:
            continue
        text = read(f)
        if text is None:
            continue
        if kind == "ipynb":
            found += [f"{f}: {p}" for p in notebook_problems(json.loads(text))]
            continue
        pattern = (HTML_EMBED if kind in HTML_EXTENSIONS
                   else RST_EMBED if kind in RST_EXTENSIONS else MARKUP_EMBED)
        if pattern.search(text):
            found.append(f"{f}: embeds an image")
    return found


def read_from_disk(path: str) -> str | None:
    file = ROOT / path
    return file.read_text(encoding="utf-8") if file.exists() else None


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout


IN_A_CHECKOUT = (ROOT / ".git").exists()


# ── the repository ───────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not IN_A_CHECKOUT, reason="not a git checkout")
def test_every_tracked_file_follows_the_policy():
    tracked = [p for p in _git("ls-files", "-z").split("\0") if p]
    assert len(tracked) == len(_git("ls-files").splitlines()), "the file list is incomplete"
    assert "tests/test_no_committed_graphics.py" in tracked
    assert problems(tracked, read_from_disk) == []


def test_the_notebooks_folder_follows_the_policy():
    """Also outside a checkout. There must be a notebook to check."""
    notebooks = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "notebooks").glob("*.ipynb"))
    assert notebooks
    assert problems(notebooks, read_from_disk) == []


# ── the scan, on a made-up repository ────────────────────────────────────────────────────


def _nb(cells=(), metadata=None) -> str:
    return json.dumps({"cells": list(cells), "metadata": metadata or {}})


FAKE = {
    "reports/plot.PNG": None,                                    # tracked, not on disk
    "docs/figure.jp2": "binary",
    "docs/page.html": "<p>x</p><img src='p.png'>",
    "docs/page.htm": "<body background='p.png'>",
    "docs/app.html": "<script>if (![a].length) {}</script>",     # JavaScript, not an image
    "docs/NOTES.MD": "Text ![x](p)",
    "docs/a.markdown": "![x][r]\n\n[r]: p",
    "docs/a.qmd": "![x](p)",
    "docs/a.rst": ".. image:: p.png",
    "docs/b.rst": ".. figure:: p.png",
    "docs/clean.md": "See poisson.pdf and `base_url(x)`; x<image.shape.",
    "elsewhere/counted.ipynb": _nb([{"cell_type": "code", "source": "1",
                                     "execution_count": 3, "outputs": []}]),
    "elsewhere/output.ipynb": _nb([{"cell_type": "code", "source": "1",
                                    "outputs": [{"data": {"image/png": "AA"}}]}]),
    "elsewhere/meta.ipynb": _nb(metadata={"widgets": {"state": "<img\nsrc='p'>"}}),
    "elsewhere/clean.ipynb": _nb([{"cell_type": "code", "source": "stats.norm.pdf(x)",
                                   "execution_count": None, "outputs": []}]),
    "src/module.py": "<img src='p'>",                           # not a scanned kind
}
EXPECTED = {
    "reports/plot.PNG": ["a graphic file"], "docs/figure.jp2": ["a graphic file"],
    "docs/page.html": ["embeds an image"], "docs/page.htm": ["embeds an image"],
    "docs/NOTES.MD": ["embeds an image"], "docs/a.markdown": ["embeds an image"],
    "docs/a.qmd": ["embeds an image"], "docs/a.rst": ["embeds an image"],
    "docs/b.rst": ["embeds an image"],
    "elsewhere/counted.ipynb": ["cell 0 has outputs"],
    "elsewhere/output.ipynb": ["cell 0 has outputs"],
    "elsewhere/meta.ipynb": ["metadata carries an image"],
}


def test_the_scan_finds_each_case_in_a_made_up_repository():
    found = problems(list(FAKE), FAKE.get)
    expected = [f"{f}: {p}" for f in FAKE for p in EXPECTED.get(f, [])]
    assert found == expected


def test_the_made_up_repository_covers_every_scanned_kind():
    kinds = {suffix(f) for f in EXPECTED}
    assert {"ipynb", "html", "htm", "md", "markdown", "qmd", "rst"} <= kinds


# ── the rules ────────────────────────────────────────────────────────────────────────────


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
    for kind in ("markdown", "raw"):
        assert notebook_problems({"cells": [{"cell_type": kind, "source": source}]}) == [
            "cell 0 carries an image"]


@pytest.mark.parametrize("source", [
    "%%html\n<img src='p'>", "display(SVG('<svg width=\"1\"/>'))", "HTML('<iframe src=\"p\">')",
    "Image(url='" + _BASE64_IMAGE + "/png;base64,AA')",
])
def test_code_that_embeds_an_image_is_caught(source):
    assert notebook_problems({"cells": [{"cell_type": "code", "source": source}]}) == [
        "cell 0 carries an image"]


def test_attachments_are_caught_in_any_cell():
    for kind in ("markdown", "raw", "code"):
        nb = {"cells": [{"cell_type": kind, "source": "x", "attachments": {"a.png": {}}}]}
        assert notebook_problems(nb) == ["cell 0 carries an image"]


@pytest.mark.parametrize("cell", [
    {"cell_type": "markdown", "source": "P(any), see [ADR-024](../docs/ADRs/024.md), x < 1."},
    {"cell_type": "markdown", "source": "scipy.stats.poisson.pdf, np.finfo(f).eps, out.png"},
    {"cell_type": "markdown", "source": "call `client.base_url(x)`"},
    {"cell_type": "code", "source": "p = stats.norm.pdf(x)\nplt.savefig('fig.png')\nurl(x)"},
    {"cell_type": "code", "source": "mask = idx<image.shape[0]\nok = n<source.size"},
])
def test_names_and_code_are_left_alone(cell):
    assert notebook_problems({"cells": [cell]}) == []
