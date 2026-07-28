# ADR-015: Technical Risk Register

**Date:** 2026-04-27
**Status:** Active
**Deciders:** Project maintainers
**Relates to:** All ADRs (the register tracks risks across the codebase)

---

## Context

As the views-baseline codebase evolves — particularly during the viewser-to-datafactory migration — audit skills (falsification, code review, tech-debt cleanup) produce findings that need persistent tracking. Without a register, findings are lost between conversations and review cycles.

## Decision

Maintain a technical risk register at `reports/technical_risk_register.md` as a first-class governance artifact. All audit findings that warrant tracking are registered through the `/register-risk` skill, which enforces deduplication, tier consistency, and trigger quality.

## Tier Definitions

| Tier | Severity | Criteria |
|------|----------|----------|
| 1 | Critical | Silent data corruption or model output correctness risk |
| 2 | High | Structural fragility causing failures under realistic change scenarios |
| 3 | Medium | Maintainability or coupling issues increasing cost of change |
| 4 | Low | Code quality concerns with no correctness or reliability impact |

## Consequences

- Audit findings are tracked across conversations
- Risks have explicit triggers and tiers, not vague "we should fix this" notes
- The register is the single source of truth for known technical debt
- Cross-repo risks (e.g., C-59 filename coupling) are tracked in the views-pipeline-core register; this register covers views-baseline-specific concerns only
