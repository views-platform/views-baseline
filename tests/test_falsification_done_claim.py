"""Register self-consistency guards (audit 2026-09-09, after v1.0.2).

These exist because the risk register misreported its own state three times at once: C-38's
body declared itself RESOLVED while filed under Open Concerns; C-27 had been withdrawn in
July with its status hidden in the Tier cell; C-42's stated discharge condition had been
met. The header arithmetic balanced throughout, because a separate Withdrawn count absorbed
the discrepancy.

**These guards were themselves audited, and the first version failed.** An independent
`/falsify guard` pass applied 18 mutations to the register: **13 survived**, and the verdict
on both guards was WEAK. Everything below is the hardened version. The lesson is recorded
here rather than in a commit message, because the next person tempted to fix this by adding
one more word to a list needs to read it:

* The marker vocabulary was incomplete *against the register's own text*. The guard matched
  `**RESOLVED**` / `**DISCHARGED**` / `**WITHDRAWN**`, while the register already retires
  entries with `**SUPERSEDED**` (C-17, and cluster K7). An entry retired the way C-17 was
  retired was invisible to the guard written to catch retired entries.
* Every other survivor attacked the *parser*, not the vocabulary: a duplicate `### C-45:`
  heading silently overwrote its twin in a dict; `### C-09 —` (dash, not colon) vanished
  from the count; `### C-99:` inside a code fence counted as real; the literal string
  `## Disagreements` inside an entry truncated the section scan; and an entry placed first
  in a section landed in the discarded `parts[0]`.

So the fix is structural, not lexical: parse strictly, and **fail loudly on anything that
looks like an entry but isn't**, rather than skipping it. A guard that silently ignores what
it cannot parse is worse than no guard, because the silence reads as a pass.

What these still cannot do is decide whether an entry's claim is *true of the world*. C-14
asserted a downstream block that had already been cleared; that was caught by running
views-models' suite, not by any text match. Do not add a guard that pretends otherwise.
"""

import pathlib
import re

REGISTER = (
    pathlib.Path(__file__).resolve().parent.parent / "reports" / "technical_risk_register.md"
)

# Status-declaration vocabulary, in the register's own convention: ALL CAPS, bold optional.
# Case-insensitive matching is deliberately NOT used — title-case "Resolved" appears
# throughout legitimate prose in *open* entries ("Moves to Resolved when #92 publishes";
# the `| Resolved | 2026-09-09 |` field row), so matching it would make the guard unusable.
# The trade-off is explicit: this catches the register's convention, not arbitrary prose.
_TERMINAL_MARKERS = (
    "RESOLVED", "DISCHARGED", "WITHDRAWN", "SUPERSEDED", "CLOSED", "OBSOLETE", "MOOT",
)
_MARKER_RE = re.compile(r"\*{0,2}\b(" + "|".join(_TERMINAL_MARKERS) + r")\b")

_ENTRY_RE = re.compile(r"^### (C-\d+): ", re.M)
_NEAR_MISS_RE = re.compile(r"^#{2,4}\s*[CD]-\d+", re.M)
_FENCE_RE = re.compile(r"^```.*?^```", re.M | re.S)


def _text() -> str:
    assert REGISTER.exists(), (
        f"{REGISTER} is missing. It is force-added despite `reports/` being gitignored; if it "
        f"is ever untracked these guards would otherwise skip silently and read as passing."
    )
    return _FENCE_RE.sub("", REGISTER.read_text())


def _sections() -> dict[str, str]:
    """Split on line-anchored ``^## `` headings.

    Not ``text.index("## Open Concerns")``: that matches the substring anywhere, so a
    cross-reference to ``## Disagreements`` inside an entry body truncated the scan and
    everything after it stopped being checked.
    """
    text = _text()
    heads = [
        (m.group(1).strip(), m.start(), m.end())
        for m in re.finditer(r"^## (.+)$", text, re.M)
    ]
    names = [h[0] for h in heads]
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, (
        f"Duplicate '## ' section heading(s): {dupes}. A second copy — a heading quoted at "
        f"the start of a line inside an entry body, say — truncates the first section there, "
        f"so every entry after it silently stops being checked. This was the one mutation "
        f"that still survived the hardened guards until this assertion was added."
    )
    out = {}
    for i, (name, _, end) in enumerate(heads):
        stop = heads[i + 1][1] if i + 1 < len(heads) else len(text)
        out[name] = text[end:stop]
    return out


def _entry_bodies(section: str) -> dict[str, str]:
    """Parse ``C-xx`` entries, refusing to silently drop anything entry-shaped."""
    strays = [
        m.group(0).strip()
        for m in _NEAR_MISS_RE.finditer(section)
        if not _ENTRY_RE.match(section[m.start():])
    ]
    assert not strays, (
        f"Heading(s) that look like register entries but do not match `### C-xx: `: {strays}. "
        f"A near-miss heading is not merely skipped — its body is merged into the previous "
        f"entry or into the discarded section preamble, so it stops being checked at all."
    )
    matches = list(_ENTRY_RE.finditer(section))
    bodies: dict[str, str] = {}
    for i, m in enumerate(matches):
        stop = matches[i + 1].start() if i + 1 < len(matches) else len(section)
        cid = m.group(1)
        assert cid not in bodies, (
            f"{cid} appears twice in one section. A dict-based parse keeps only the second, "
            f"hiding whatever the first said."
        )
        bodies[cid] = section[m.end():stop]
    return bodies


def _open() -> dict[str, str]:
    return _entry_bodies(_sections()["Open Concerns"])


def test_no_open_entry_declares_itself_terminal():
    """An entry cannot be both open and finished."""
    offenders = {
        cid: sorted({m.group(1) for m in _MARKER_RE.finditer(body)})
        for cid, body in _open().items()
    }
    offenders = {c: m for c, m in offenders.items() if m}
    assert not offenders, (
        f"Entries under '## Open Concerns' declaring themselves finished: {offenders}. "
        f"Move each to '## Resolved Concerns' and correct the header, or drop the marker."
    )


def test_tier_cells_are_a_bare_integer():
    """Status must not hide in the Tier cell.

    C-27 sat withdrawn under Open Concerns for weeks with
    ``| Tier | 3 — **WITHDRAWN 2026-07-18** |``, and the marker guard above sees that only
    if whoever wrote it happened to bold and capitalise — the unbolded form
    (``2 (withdrawn — no longer applies)``) survived the audit. Constraining the cell's
    *shape* closes the hole regardless of wording.

    Rationale for a tier change belongs in a status note under the narrative, not in the
    cell. This repository has now made that mistake twice: once in C-27, and once in the
    commit that criticised C-27 for it.
    """
    bad = {}
    for cid, body in _open().items():
        m = re.search(r"^\| Tier \| (.+?) \|$", body, re.M)
        if m and m.group(1).strip() not in {"1", "2", "3", "4"}:
            bad[cid] = m.group(1).strip()
    assert not bad, (
        f"Tier cells that are not a bare 1-4: {bad}. The schema specifies an integer; prose "
        f"here breaks every parser that reads the field, and is where retired entries hide."
    )


def test_every_entry_lives_in_a_concern_section():
    """No entry may sit outside Open or Resolved.

    A concern written between '## Disagreements' and '## Resolved Concerns' — a plausible
    slip, since that is where the D-entries live — is counted by neither section and by no
    header row.
    """
    sections = _sections()
    homed = set(_entry_bodies(sections["Open Concerns"])) | set(
        _entry_bodies(sections["Resolved Concerns"])
    )
    all_ids = set(_ENTRY_RE.findall(_text()))
    assert all_ids == homed, (
        f"Entries outside the two concern sections: {sorted(all_ids - homed)}. "
        f"They are invisible to both these guards and the header counts."
    )


def test_no_entry_appears_in_both_sections():
    """One concern, one home — a duplicate double-counts in a self-consistent way."""
    sections = _sections()
    both = set(_entry_bodies(sections["Open Concerns"])) & set(
        _entry_bodies(sections["Resolved Concerns"])
    )
    assert not both, f"Entries filed in both Open and Resolved: {sorted(both)}."


def test_header_counts_match_the_entries_actually_present():
    """The header is the register's index; a wrong count misdirects triage.

    Schema changed 2026-09-09: the separate ``Withdrawn`` row was folded into ``Resolved``,
    because it was how C-27 could be counted correctly while filed incorrectly.
    """
    text = _text()
    sections = _sections()

    def declared(label: str) -> int:
        # `\s+`, not a hard-coded single space: realigning the header table is what any
        # markdown formatter does, and the earlier version raised AttributeError from a
        # regex miss rather than reporting a count problem.
        m = re.search(rf"\| {label}\s+\| (\d+)", text)
        assert m, f"header row '{label}' not found or not numeric"
        return int(m.group(1))

    actual_open = len(_entry_bodies(sections["Open Concerns"]))
    actual_resolved = len(_entry_bodies(sections["Resolved Concerns"]))

    assert actual_open == declared("Open Concerns"), (
        f"header says {declared('Open Concerns')} open; the section holds {actual_open}."
    )
    assert actual_resolved == declared("Resolved Concerns"), (
        f"header says {declared('Resolved Concerns')} resolved; "
        f"the section holds {actual_resolved}."
    )
    assert declared("Total Concerns") == actual_open + actual_resolved, (
        f"{actual_open} open + {actual_resolved} resolved != "
        f"{declared('Total Concerns')} total."
    )
