# ADR 000: Use of Architecture Decision Records

**Status:** Accepted
**Date:** 2026-03-13
**Deciders:** Project maintainers

---

## Context

views-baseline is a focused library of interpretable baseline forecasting models for the VIEWS conflict prediction pipeline. Although small in scope today (five model classes, one manager), the project sits at a junction of several pressures that make undocumented decisions costly:

- **Evolving research baseline expectations.** The VIEWS pipeline treats these models as reference points against which more sophisticated models are evaluated. What counts as a valid baseline — a point estimate, a distributional sample, a mixture — has changed (e.g., the shift from DataFrame to PredictionFrame output for distributional models), and will change again. Decisions made during those transitions are lost without a record.

- **Multiple contributors including AI-assisted development.** Code in this repository has been produced collaboratively, with human reviewers and AI-assisted authoring. This raises the stakes for capturing the *reasoning* behind structural choices, not just the choices themselves. A future contributor (human or AI) reading only the code cannot reconstruct why `model/` was kept free of pipeline-core imports, or why a class attribute was chosen over return-type inspection for protocol dispatch.

- **Pipeline integration decisions have downstream consequences.** `BaselineForecastingModelManager` extends `ForecastingModelManager` from views-pipeline-core. Choices about which methods to override, how prediction output is shaped, and when artifacts are written interact with ensemble managers and evaluation tooling outside this repository. These coupling points need to be legible to maintainers who did not make the original decisions.

- **Small test-and-iterate cycles obscure architecture drift.** With 51 tests across 4 files covering model behaviour, catalog validation, and protocol conformance, the mechanical behaviour of the code is well-protected. Architecture decisions — the shape of the dependency graph, the authority of explicit declarations, stability classifications — are not tested and can silently erode.

ADRs provide a lightweight, durable, human-readable record that fills this gap.

---

## Decision

This project will use Architecture Decision Records (ADRs) to document significant architectural and design decisions.

ADR files will be stored in `docs/ADRs/` within the repository, numbered sequentially starting at `000`. Each file is named `NNN_short_description.md`.

The minimum structure for each ADR is:

- **Status** — one of: Proposed, Accepted, Deprecated, Superseded by ADR-NNN
- **Date** — ISO 8601
- **Deciders** — who was involved
- **Context** — the forces, constraints, and situation that made a decision necessary
- **Decision** — the choice made, stated plainly
- **Rationale** — why this choice was made over alternatives
- **Considered Alternatives** — other options evaluated, with reasons for rejection
- **Consequences** — what becomes easier, harder, or newly required
- **Implementation Notes** — where in the codebase the decision is enacted
- **Validation & Monitoring** — how to tell if the decision is holding
- **Open Questions** — known uncertainties at time of writing
- **References** — related ADRs, external documents, or code pointers

ADRs are append-only. An existing ADR is never edited to change its decision. When a decision changes, a new ADR is written and the old one's status is updated to "Superseded by ADR-NNN".

---

## Rationale

ADRs were chosen over alternatives (inline comments, a wiki, a design document) because:

- They live in the repository alongside the code, so they are versioned, searchable, and visible in code review.
- They are immutable once accepted, which preserves the reasoning behind decisions that are later reversed — a common occurrence in research-adjacent codebases.
- They are cheap to write for a project of this size: one file per decision, no tooling required.

The sequential numbering scheme is intentional. It gives every ADR a stable identifier (`ADR-002`) that can be cited in code comments, PR descriptions, and other ADRs without the reference decaying.

---

## Considered Alternatives

**Inline code comments.** Pros: zero friction, co-located with code. Cons: comments explain *what*, rarely *why*; they are edited in place and lose historical context; they are invisible to someone reading the repository structure without opening every file.

**Confluence/wiki page.** Pros: rich formatting, easy to search across projects. Cons: lives outside the repository, decouples from code history, tends to drift out of date without clear ownership.

**Single `ARCHITECTURE.md` document.** Pros: one file, easy to find. Cons: grows unwieldy, conflates stable and evolving sections, and does not preserve the chronology of decisions.

---

## Consequences

**Positive:**
- New contributors (human or AI-assisted) have a structured starting point for understanding why the code is shaped as it is.
- Decisions about pipeline coupling, dependency rules, and stability classification are on record and can be cited when they come under pressure.
- The cost of reversing a decision becomes visible: it requires writing a new ADR, which creates a natural forcing function for deliberation.

**Negative:**
- Writing an ADR takes time. For a small project with fast iteration cycles, the overhead can feel disproportionate.
- ADRs can become stale if maintainers forget to update status fields when decisions are superseded.
- The value is proportional to discipline: ADRs that are written but not read or cited provide no benefit.

---

## Implementation Notes

ADR files reside at `docs/ADRs/NNN_description.md`. This ADR (000) is itself the first entry.

The template used in this repository is the one defined here — all subsequent ADRs follow the section structure above.

---

## Validation & Monitoring

- Every PR that introduces a significant structural change (new model class, dependency on a new external package, change to the manager interface) should reference or create an ADR.
- During code review, reviewers should check whether the change is consistent with existing ADRs.
- ADR status fields should be reviewed when views-pipeline-core releases a breaking change.

---

## Open Questions

- Should there be a lightweight ADR template file in `docs/ADRs/` to make the structure explicit for new contributors?
- As the project grows, should ADRs be categorised (e.g., model design vs. infrastructure vs. testing) or kept in a single flat list?

---

## References

- Michael Nygard, "Documenting Architecture Decisions" (2011) — original ADR proposal
- ADR 001: Ontology of the Repository
- ADR 002: Topology and Dependency Rules
- ADR 003: Authority of Declarations over Inference
- ADR 004: Rules for Evaluation and Stability
