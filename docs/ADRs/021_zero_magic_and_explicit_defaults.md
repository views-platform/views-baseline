# ADR-021: Zero-Magic and Explicit Defaults

**Status:** Accepted
**Date:** 2026-06-25
**Deciders:** Simon, VIEWS platform team
**Consulted:** model-review of white_ranger + falsification audit (2026-06-25); risk register C-05, C-10, C-25
**Informed:** views-models maintainers (downstream config impact)

---

## Context

views-baseline had no explicit rule against *magic values* — hardcoded literals, positional assumptions, and silently-applied defaults for parameters that affect model output or reproducibility. Its sister repos formalized one and views-baseline did not:

- **views-hydranet ADR-003 ("Philosophy of Engineering")** — **Law 1 (Fail Loud)** prohibits *"magic default values (e.g., assuming a spatial resolution if not provided)"*; **Law 2 (Zero-Magic)** prohibits *"hardcoded strings, magic indices, or positional assumptions."*
- **views-r2darts2 ADR-010** — prohibits hardcoded semantic thresholds, deferring such values to declared config genes.

The absence bit us concretely. `BaselineModelCatalog` silently applies `seed=42` when a config omits `seed` — a textbook "magic default (assuming a seed if not provided)" — which made every distributional model run at seed 42 regardless of config and rendered ~9 downstream seed sweeps inert (risk **C-10**). The same class of issue underlies the positional `df.index.names[0]/[1]` reads (**C-05**) and the default `42` literal duplicated across constructors (**C-25**).

This is also **internally inconsistent** in our own governance: **ADR-014 / the ReproducibilityGate** already declares *"Implicit defaults are forbidden"* for mandatory parameters, and **ADR-003** makes declarations authoritative over inference — yet **ADR-011 §2** blessed `seed` *with a silent default of 42*. A parameter cannot be both "reproducibility-critical" and "silently defaulted." This ADR resolves that contradiction by adopting the platform's Zero-Magic philosophy for views-baseline and stating where defaults are and are not permitted.

---

## Decision

**views-baseline adopts the Zero-Magic and Explicit-Defaults rules below. They govern every file under `views_baseline/`.**

### Law 1 — No magic default values (fail loud)

A parameter that affects model **output** or **reproducibility** MUST be **declared** in config and **validated** by the `ReproducibilityGate`. The code MUST NOT silently substitute a value when it is absent. A missing such parameter is a contract violation and MUST fail loud (`MissingHyperparameterError`) at config-audit time, not degrade to a default.

- Concretely: `seed` is reproducibility-critical. It MUST be a declared, required, **audited genome key** (`ALGORITHM_GENOMES`), and the catalog MUST forward `config["seed"]`. It may no longer be silently defaulted. This **amends ADR-011 §2** (which permitted a silent default of 42).

### Law 2 — Zero-Magic (no hardcoded strings, indices, or positional assumptions)

Domain-significant strings, column names, spatial levels, and index positions MUST be derived from a **declared source** (config) or an **authoritative vocabulary**, and validated at the boundary — never hardcoded or inferred positionally without a check.

- Concretely: the spatial `level` and `(time, entity)` index names are sourced from `views_frames.SpatialLevel` and validated by `resolve_level` (**ADR-020**) rather than hardcoded — this is the pattern to follow. The remaining positional `df.index.names[0]/[1]` reads (**C-05**) are the target for continued hardening under this Law.

### Where defaults ARE permitted

A default is permitted only for a parameter that is **not** reproducibility- or output-significant (e.g., a purely operational or cosmetic knob). When permitted, it MUST be a **single named constant** (e.g., `DEFAULT_SEED`), never an inline literal duplicated across call sites (**C-25**). Even then, a named constant used as a last-resort sentinel is defense-in-depth only — it MUST NOT be the production source of truth for a significant parameter; the genome is.

### Out of scope

Trivial, non-domain constants with no output/reproducibility impact (loop bounds, well-known mathematical constants, formatting widths) are out of scope. This ADR is about *semantic* magic, not stylistic literal-elimination.

---

## Rationale

- **Reproducibility to intent, not just to run.** A run that is internally deterministic but ignores the declared `seed` is reproducible to the wrong experiment. Zero-Magic makes the declared configuration the single authority (aligns ADR-003, ADR-014).
- **Loud beats silent.** A missing significant parameter should halt at the config boundary with a named error, not produce plausible-but-unintended output — the exact class the ReproducibilityGate exists to catch.
- **Platform consistency.** hydranet and r2darts2 already committed to this; adopting it removes a governance gap and makes cross-repo review uniform.

---

## Considered Alternatives

### Alternative A: Keep the silent `seed=42` default (optional-with-default)
- **Pros:** No downstream config changes; smallest diff.
- **Cons:** Perpetuates a magic default for a reproducibility-critical parameter; leaves the ADR-011↔ADR-014 contradiction; a misspelled `seed` still silently defaults.
- **Reason for rejection:** Directly violates the Law this ADR adopts.

### Alternative B: Eliminate the constructor default entirely (seed required at the class)
- **Pros:** Purest Zero-Magic — no default exists anywhere.
- **Cons:** Every direct construction (incl. `test_protocol.py`, which omits `seed`) must pass it; larger, lower-value churn.
- **Reason for rejection (partial):** The genome requirement already guarantees production never relies on a default. A single named `DEFAULT_SEED` sentinel for direct/test construction is acceptable defense-in-depth; full removal is optional and deferred.

---

## Consequences

### Positive
- `seed` becomes declared, required, and audited — seed sweeps become meaningful and reproducibility-to-intent is restored (closes C-10).
- One authority for defaults (named constants) removes the duplicated-`42` drift risk (closes C-25).
- Resolves the ADR-011 (silent default) ↔ ADR-014 (implicit defaults forbidden) contradiction in favour of the gate.

### Negative / costs
- Making `seed` a required genome key raises `audit_manifest` for the ~9 views-models distributional configs that omit it — a **coordinated cross-repo change** is required so they declare `seed` before (or with) the genome change (risk **C-24**). Under Law 1 this loud failure is *desired*, but it must be sequenced, not shipped blind.
- Slightly more verbose configs (every distributional config must state `seed`).

---

## Implementation Notes

1. Add `seed` to `ALGORITHM_GENOMES` for `ConflictologyModel` and `MixtureBaseline` (`infrastructure/reproducibility_gate.py`).
2. Forward `seed=self.config["seed"]` in `catalog.py` `_get_conflictology_model` / `_get_mixture_model`.
3. Introduce a single `DEFAULT_SEED = 42` constant (module-level in `model/baseline.py`) referenced by the constructor defaults; remove the duplicated inline `42` literals (C-25).
4. **Coordinated rollout (C-24) — there is no Law-1-clean baseline-only interim.** Because the catalog now reads `config["seed"]` *strictly* (a `config.get("seed", 42)` fallback would itself be the magic default this ADR retires), the ~9 distributional configs that omit `seed` must declare it first. Land as one coordinated views-baseline + views-models change: **(a)** add `seed` to the ~9 views-models configs that omit it (lucid_dream + the 8 MixtureBaseline models); **(b)** in views-baseline, add `seed` to `ALGORITHM_GENOMES` (so a missing seed fails as a clean `MissingHyperparameterError` at audit time, not a deep `KeyError`), forward `config["seed"]`, add the `DEFAULT_SEED` constructor sentinel, and the tests. The `DEFAULT_SEED` constant is the sentinel for direct/test construction only — the catalog does **not** use it.
5. Tests: the failing stubs in `tests/test_falsification_seed_wiring.py` enforce forwarding; add a genome test (missing `seed` → `MissingHyperparameterError`) and a param-completeness test (factory forwards ⊇ constructor's non-defaulted params).
6. Amend **ADR-011 §2** with a status note pointing here; keep C-05 index-position hardening as ongoing work under Law 2.

If nothing else is required for a given change, that is fine — this ADR is a standing rule, not a one-off migration.

---

## Validation & Monitoring

- `ReproducibilityGate.audit_manifest` raises `MissingHyperparameterError` when a distributional config omits `seed`.
- `grep -rn` for inline domain literals (`42`, hardcoded level/index strings) in `model/` returns only named-constant definitions.
- `tests/test_falsification_seed_wiring.py` (forwarding) + new genome/param-completeness tests are green.
- Failure mode that would reopen this: a new model parameter added with a silent default and no genome entry.

---

## Open Questions

- Should the constructor `seed` default be removed entirely (Alternative B) once all configs declare `seed`, or retained as a named sentinel? Deferred.
- Should a lightweight CI grep enforce "no un-named domain literal in `model/`" (mechanising Law 2), analogous to the import-linter gap noted in ADR-002/C-09?

---

## References

- views-hydranet ADR-003 (Laws 1–2, the source philosophy); views-r2darts2 ADR-010 (semantic thresholds).
- ADR-003 (declarations over inference), ADR-011 (RNG determinism — §2 amended here), ADR-014 (reproducibility gate / "implicit defaults forbidden"), ADR-020 (SpatialLevel vocabulary as the Zero-Magic pattern).
- Risk register: C-05 (positional index), C-10 (seed not forwarded), C-24 (rollout sequencing), C-25 (duplicated default); disagreements D-07, D-09.
- `fix/distributional-seed-not-forwarded` branch; falsification stubs `tests/test_falsification_seed_wiring.py`.
