"""Falsification stubs — the claim "we are done here; nothing up or downstream is blocked
by anything in this repo" (audit 2026-09-09, after v1.0.2).

Seven probes were run against the **published** 1.0.2 wheel and the real views-models
tree. Six passed outright: all 7 algorithms construct from real merged configs (P1); the
views-models smoke seam works under the #459 fix (P2); our ``views-frames<2`` pin is
byte-identical to what pipeline-core 3.2.0, views-evaluation and views-datafactory all
declare, so it blocks no adoption (P3); all 29 shipped configs pass the newly-tightened
gate and build (P4); the package co-resolves with views-datafactory (P5); and
``evaluation.closeness`` imports in a clean env, confirming scipy arrives transitively
(P6).

The adequacy probe (P7) is the one that found something, and it found it in the artifact
a maintainer actually reads to decide what is outstanding: **the risk register misreports
its own state.** Nothing here blocks a consumer — these are soft falsifications of "done",
not of "unblocked" — but a register that says a downstream repo is blocked when it is not
is worse than no register, because it is trusted.

**Status: written red, now green.** They encoded the corrected contract rather than the
current one, and went green when `review-rr strategic` reconciled the register on 2026-09-09
(C-03, C-05, C-14, C-38, C-42 resolved; C-27 refiled). They are kept as permanent guards,
because the drift they caught was introduced twice in one day by the epic that was cleaning
up drift.

**Two guards, not four.** The first draft had two more, asserting "C-14 must not be open" and
"C-42 must not contain this phrase" — tombstones that could only fail if those exact entries
returned, which they will not. `/review-diff` caught them, and they were exactly the K8
pattern this commit adds to the register. What survives is general.

What these cannot do is judge whether an entry is *true* — only whether the register
contradicts itself. C-14 was caught by running views-models' suite, not by a test.
"""

import pathlib
import re

import pytest

REGISTER = (
    pathlib.Path(__file__).resolve().parent.parent / "reports" / "technical_risk_register.md"
)

pytestmark = pytest.mark.skipif(
    not REGISTER.exists(),
    reason="register absent from this checkout (reports/ is gitignored; the file is force-added)",
)


def _open_section() -> str:
    text = REGISTER.read_text()
    start = text.index("## Open Concerns")
    end = text.index("## Disagreements")
    return text[start:end]


def _entry_bodies(section: str) -> dict[str, str]:
    """Split an Open/Resolved section into {concern id: body text}."""
    parts = re.split(r"^### (C-\d+):", section, flags=re.MULTILINE)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}


# Any marker by which an entry declares itself finished. An entry carrying one of these
# while filed under Open Concerns is self-contradicting, whatever the counts say.
_TERMINAL_MARKERS = ("RESOLVED", "DISCHARGED", "WITHDRAWN")


def test_no_open_entry_declares_itself_terminal():
    """An entry cannot be both open and finished (drift found 2026-09-09).

    Three instances existed simultaneously: C-38's body said "RESOLVED (2026-09-09,
    v1.0.2)" while filed under Open; C-27 had been withdrawn in July with its status
    encoded in the Tier cell; C-42's stated discharge condition had been met. The header
    arithmetic balanced throughout, because a separate Withdrawn count absorbed the
    discrepancy — the structure and the prose disagreed and nothing noticed.

    Deliberately general. The first draft of this file hardcoded "C-14 must not be open"
    and "C-42 must not contain this phrase" — assertions that could only fail if those
    exact entries came back, i.e. never. That is the K8 pattern this same commit adds to
    the register: a guard written by the author of the thing guarded, encoding the
    author's model of failure. What is *not* mechanically decidable — whether an entry's
    claim is still true of the world — is not faked here. C-14 was caught by running
    views-models' suite, not by a test.
    """
    offenders = {
        cid: [m for m in _TERMINAL_MARKERS if re.search(rf"\*\*{m}\b", body)]
        for cid, body in _entry_bodies(_open_section()).items()
    }
    offenders = {c: m for c, m in offenders.items() if m}
    assert not offenders, (
        f"Entries under '## Open Concerns' that declare themselves finished: {offenders}. "
        f"Move each to '## Resolved Concerns' and correct the header, or drop the marker."
    )


def test_header_counts_match_the_entries_actually_present():
    """The header is the register's index; a wrong count misdirects triage.

    The schema changed on 2026-09-09 (`review-rr` strategic): the separate ``Withdrawn``
    row was folded into ``Resolved`` because it created the ambiguity it was meant to
    resolve — C-27 was counted as withdrawn while being *filed* under Open Concerns, so
    the arithmetic came out right while the filing was wrong. The invariant is now simply
    that every entry lives in the section matching its status, and the two section counts
    sum to the total.
    """
    text = REGISTER.read_text()
    declared_open = int(re.search(r"\| Open Concerns\s+\| (\d+)", text).group(1))
    declared_total = int(re.search(r"\| Total Concerns\s+\| (\d+)", text).group(1))
    declared_resolved = int(re.search(r"\| Resolved Concerns \| (\d+)", text).group(1))

    actual_open = len(_entry_bodies(_open_section()))
    resolved_sec = text[text.index("## Resolved Concerns"):text.index("## Register Conventions")]
    actual_resolved = len(_entry_bodies(resolved_sec))

    assert actual_open == declared_open, (
        f"header says {declared_open} open concerns; '## Open Concerns' holds {actual_open}."
    )
    assert actual_resolved == declared_resolved, (
        f"header says {declared_resolved} resolved; the section holds {actual_resolved}."
    )
    assert declared_total == declared_open + declared_resolved, (
        f"header totals do not add up: {declared_open} open + {declared_resolved} resolved "
        f"!= {declared_total} total."
    )
