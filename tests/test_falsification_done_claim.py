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

**Status: all four now pass.** They were written red — as falsification stubs encoding the
corrected contract rather than the current one — and went green when `review-rr strategic`
reconciled the register on 2026-09-09 (C-03, C-05, C-14, C-38, C-42 resolved; C-27 refiled).
They are kept as permanent guards, because the drift they caught was introduced twice in one
day by the epic that was cleaning up drift.

They are cheap: four assertions over a markdown file, no imports from the package. What they
cannot do is judge whether an entry is *true* — only whether the register contradicts itself.
C-14 was caught because it declared a block while sitting in the open section, not because a
test knew views-models had moved on; that took running the downstream suite.
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


def test_no_open_entry_declares_itself_resolved():
    """An entry cannot be both open and resolved (#94-era drift, found 2026-09-09).

    C-38's body was updated to "RESOLVED (2026-09-09, v1.0.2)" with the clean-room install
    as evidence, but the entry was never moved out of Open Concerns and is still counted
    among the open 26. The register's structure and its prose disagree, and the structure
    is what a reader skims.
    """
    contradictory = [
        cid
        for cid, body in _entry_bodies(_open_section()).items()
        if re.search(r"\*\*RESOLVED\b", body)
    ]
    assert not contradictory, (
        f"Entries in '## Open Concerns' whose bodies declare themselves RESOLVED: "
        f"{sorted(contradictory)}. Move them to '## Resolved Concerns' and correct the "
        f"header counts, or remove the RESOLVED marker."
    )


def test_no_open_entry_has_a_met_discharge_condition():
    """C-42 says it "Stays Open until merged and released". Both have happened.

    Merged as 2b3d3b4 (PR #94) and released as v1.0.2 on 2026-09-09. An entry whose own
    stated discharge condition is satisfied but which remains open is indistinguishable,
    to a reader, from live work.
    """
    raw = _entry_bodies(_open_section()).get("C-42", "")
    # The phrase spans a blockquote line break ("> "), so normalise before matching —
    # a naive literal search silently passes and reports no finding.
    body = re.sub(r"\s*\n>?\s*", " ", raw)
    assert "Stays **Open** until merged and released" not in body, (
        "C-42 still carries the discharge condition 'Stays Open until merged and "
        "released'. views-baseline 1.0.2 was merged (2b3d3b4) and released on 2026-09-09, "
        "so the condition is met: resolve it, or state a new condition."
    )


def test_c14_does_not_claim_a_downstream_block_that_no_longer_exists():
    """C-14 asserts views-models is blocked by this repo. Verified false, 2026-09-09.

    Its trigger says enabling a point baseline with ``prediction_format:
    "prediction_frame"`` fails ``TestPFModelConfigReadiness`` because a deterministic model
    has no posterior samples. All 29 shipped configs now declare that format, the 9 point
    baselines correctly omit ``n_posterior_samples``, and views-models'
    ``tests/test_pfe_production_readiness.py`` passes **291 tests, 0 failures** — the test
    was rewritten to handle exactly this case ("point models omit n_posterior_samples and
    emit (N, 1)").

    This is the finding that matters most in this audit: a register entry claiming an
    outbound block that does not exist is the one kind of staleness that can stop someone
    else's work for no reason.
    """
    open_ids = set(_entry_bodies(_open_section()))
    assert "C-14" not in open_ids, (
        "C-14 is still open and still asserts that point baselines cannot satisfy "
        "views-models' PF readiness contract. views-models resolved this on their side; "
        "the contract now explicitly accommodates point models. Resolve C-14 with that as "
        "evidence, or restate what remains true."
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
