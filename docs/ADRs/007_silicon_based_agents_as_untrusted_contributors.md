# ADR-007: Silicon-Based Agents as Untrusted Contributors

- **Status:** Accepted
- **Date:** 2026-03-13
- **Deciders:** Project maintainers

---

## Context

AI coding assistants — specifically Claude Code — have been actively used during the development of views-baseline. The scope of use includes refactoring model classes, writing and extending test suites, and code review of pull requests. Evidence of this is embedded in the git history: commits contain `Co-Authored-By: Claude` headers.

This creates a category of contributor that is qualitatively different from a human contributor. AI-generated code:

- Can be syntactically and structurally plausible while being semantically incorrect.
- Will not catch its own errors through the kind of second-order reasoning a senior developer applies when reviewing work they produced themselves.
- Has no persistent understanding of the project's history, design decisions, or the downstream consequences of changes in this specific codebase.
- Cannot be held accountable for the consequences of incorrect output.
- Does not know what it does not know: it will produce confident, well-formatted code for inputs it has misunderstood.

At the same time, AI tools provide genuine productivity value: they accelerate boilerplate-heavy tasks (test fixtures, repetitive model variants), surface alternative implementations for review, and can identify inconsistencies across a codebase faster than a single developer can.

The question is not whether to use AI tools — they are already in use — but how to use them without inadvertently lowering the quality bar or introducing undetected defects.

---

## Decision

AI coding assistants are treated as untrusted contributors. Their output is accepted into the codebase only after passing the same automated gates as any other contribution, and their involvement is tracked explicitly in the git record. No AI-generated change may bypass lint or tests.

The full protocol is documented in `docs/contributor_protocols/silicon_based_agents.md`. This ADR records the reasoning behind that protocol.

---

## What "Untrusted" Means in Practice

"Untrusted" does not mean "prohibited" or "suspected of malice." It means that AI output is treated the same way a contribution from an unfamiliar external contributor would be treated: with the assumption that it may contain errors, may not conform to project conventions, and has not been verified against the project's history of decisions.

Specifically:

- AI-generated code is not assumed to be correct because it looks correct.
- AI-generated tests are not assumed to be meaningful because they pass.
- AI-generated refactors are not assumed to preserve behaviour because the diff is clean.

---

## Current Usage in views-baseline

Claude Code is used for three categories of work in this project:

**Refactoring.** Examples include consolidating the `_generate_predictions` dispatch logic in `BaselineForecastingModelManager`, extracting `build_prediction_grid` into `helpers.py`, and simplifying distributional model implementations. All refactors are verified by running the full test suite before merge.

**Test writing.** Claude Code has contributed test cases to `test_baseline.py`, `test_catalog.py`, `test_protocol.py`, and `test_baseline_manager.py`. Every test produced by AI is reviewed for: correct fixture usage, whether the assertion is testing what it claims to test, and whether it is a tautology (asserting that the output of the code under test equals the output of the code under test with no independent ground truth).

**Code review.** Pull requests are reviewed in part by asking Claude Code to identify inconsistencies, missing edge cases, and deviations from project conventions. This is advisory input; it does not substitute for human review.

---

## Enforcement Gates

Two automated gates apply to every change, including AI-assisted changes:

**Lint (ruff).** The ruff linter is run in GitHub Actions CI on every push and pull request. AI-generated code that introduces style violations, unused imports, or flagged patterns fails CI and is not merged.

**Tests (pytest).** The full test suite (51 tests across 4 files) is run in GitHub Actions CI on every push and pull request. AI-generated code that breaks existing tests is not merged. AI-generated tests that fail on the current codebase are not accepted.

These gates are necessary but not sufficient. Passing lint and tests confirms that the code is syntactically valid, style-conformant, and consistent with what the existing tests check. It does not confirm that the code is correct for cases not covered by tests, that the AI understood the intent of a change correctly, or that an AI-generated test is actually testing the right thing.

---

## Attribution

`Co-Authored-By` git trailers record AI involvement at the commit level. This serves two purposes:

1. **Traceability.** If a defect is later traced to a specific commit, the `Co-Authored-By` header indicates that AI tooling was involved and the change should be scrutinised accordingly.
2. **Normalisation.** Recording AI involvement as a standard part of the commit record, rather than something to hide or omit, keeps the practice visible and subject to project-level discussion.

The format used is:

```
Co-Authored-By: Claude <model-identifier> <noreply@anthropic.com>
```

---

## What This Protocol Does Not Cover

- **AI-generated documentation** is not subject to automated verification and relies entirely on human review for accuracy.
- **AI-assisted design discussions** (e.g., asking an AI to evaluate architectural options) are not recorded in git and are treated as equivalent to consulting a reference document: useful input, not a decision.
- **Future AI tools.** This protocol was written with Claude Code in mind. Other AI tools (GitHub Copilot, Cursor, etc.) are not currently used in this project but would fall under the same untrusted-contributor classification if introduced.

---

## Consequences

**Positive:**

- AI productivity gains are captured without reducing the quality bar that applies to human-authored code.
- AI involvement is visible in the git record and therefore auditable.
- The untrusted-contributor framing prevents the gradual normalisation of accepting AI output without review.

**Negative / Risks:**

- The gate system catches mechanical errors (syntax, style, broken tests) but not semantic errors in AI-generated code that happens to pass existing tests. The test coverage gaps documented in ADR-005 are therefore also gaps in AI-output verification.
- `Co-Authored-By` attribution is a convention, not an enforcement mechanism. It can be omitted without CI failing.
- A human reviewer who is less familiar with the codebase than the AI that generated a change may fail to catch a subtle error. This is an inherent risk of using AI tools that cannot be fully mitigated by process alone.
