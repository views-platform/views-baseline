# Documentation Instantiation Checklist — views-baseline

**Completed:** 2026-03-13
**Completed by:** Project maintainers

This checklist records the instantiation of the documentation framework for views-baseline.
It tracks which template items have been completed, which are deferred, and why.

---

## Phase Selection

- [x] Decide adoption phase

  **Selected:** Full adoption. All constitutional ADRs (000–009) are instantiated for
  views-baseline. The project is small but sits at a pipeline junction where undocumented
  decisions are costly.

---

## Ontological Preparation

- [x] Identify ontological categories

  **Six categories identified:** Point Forecast Models, Distributional Forecast Models,
  Model Protocols, Model Factory, Pipeline Integration, Prediction Builders. See ADR-001.

---

## ADR Status Fields

- [x] Update Status from `--template--` to Accepted

  All nine ADRs carry `Status: Accepted` and `Date: 2026-03-13`.

- [x] Fill in Date, Deciders

  All ADRs: `Date: 2026-03-13`, `Deciders: Project maintainers`.

---

## Constitutional ADR Instantiation

- [x] ADR-000: Updated path

  File path confirmed: `docs/ADRs/000_use_of_adrs.md`. Self-referential path is correct.

- [x] ADR-001: Defined categories and stability

  Six categories defined. Stability annotations: ZeroModel/LocfModel/AverageModel/protocols/
  catalog/helpers → Stable; ConflictologyModel/MixtureBaseline/manager → Evolving.

- [x] ADR-002: Defined layering and forbidden patterns

  Dependency graph: helpers → baseline → catalog → manager. Four forbidden import directions
  documented. Lazy `PredictionFrame` import rule documented.

- [x] ADR-003: Adapted forbidden behavior examples

  Five forbidden inference patterns listed (model type from output, targets from DataFrame,
  window_size from len(df), train/test boundary from index.max(), dispatch from predict()).
  Entity-drop warning noted as known deviation from fail-loud.

- [x] ADR-005: No domain adaptation needed

  Three-team model (Green/Beige/Red) applies directly. Comprehensive suite documented. Coverage gaps
  recorded honestly: manager distributional path, _evaluate_sweep, entity-drop warnings,
  non-default output_length, single-target distributional edge case.

- [x] ADR-006: No domain adaptation needed

  Seven CICs required and written. Two protocol classes explicitly excluded.

- [x] ADR-007: Verified protocol paths

  `docs/contributor_protocols/silicon_based_agents.md` written. `Co-Authored-By` format
  confirmed. Ruff + pytest gates confirmed.

- [x] ADR-009: Adapted boundary examples

  Five boundaries mapped: Config→Catalog, Catalog→Model, Model→Pipeline, Data→Model,
  Manager→core. Known gaps at each boundary documented.

---

## CIC Framework

- [x] Replace CIC placeholder list

  `docs/CICs/README.md` lists all 7 active CICs with one-line descriptions and governance
  cross-references to ADR-006, ADR-003, ADR-005.

- [x] Create intent contracts (7 CICs)

  All seven CICs written and present in `docs/CICs/`:
  - ZeroModel.md
  - LocfModel.md
  - AverageModel.md
  - ConflictologyModel.md
  - MixtureBaseline.md
  - BaselineModelCatalog.md
  - BaselineForecastingModelManager.md

---

## Contributor Protocols

- [x] Review silicon protocol

  `docs/contributor_protocols/silicon_based_agents.md` written. Covers: untrusted-contributor
  classification, mandatory gates (ruff + pytest), Anti-Truncation Rule, RNG ordering
  sensitivity, protocol dispatch sensitivity, lazy import fragility, entity-drop warnings,
  catalog consistency, and list of forbidden operations.

- [x] Review carbon protocol

  `docs/contributor_protocols/carbon_based_agents.md` written. Covers: ADR reading order,
  dependency rules, declarations-over-inference, model addition checklist, testing obligations,
  code review checklist, commit message conventions, and known limitations to preserve.

- [x] Adapt hardened protocol

  `docs/contributor_protocols/hardened_protocol_template.md` written. Covers: RNG
  determinism contract, MODEL_GENOMES validation, 1-Class-1-File known deviation (documented
  as intentional), testing taxonomy map to Green/Beige/Red, point vs distributional model
  checklists, no numerical airlock (no deep learning), logging conventions.

  Removed: PyTorch Lightning, Darts, gradients, float32 downcasting, CUDA references.

---

## Standards

- [x] Review logging standard

  `docs/standards/logging_and_observability_standard.md` written. Covers: logger instantiation,
  INFO messages (model fit/predict stages), WARNING messages (entity-drop in all four
  affected models), absence of ERROR/CRITICAL (documented as gap), no structured logging
  (documented as accepted debt), no run-level context, logging test gap, rules for new code.

- [ ] Review physical architecture standard (not included — multi-class files)

  Deferred. The physical architecture standard assumes 1-Class-1-File. views-baseline
  intentionally deviates: all seven model classes live in `baseline.py`, a
  decision recorded in ADR-001. A physical architecture standard that documents this
  deviation as intentional and provides guidance for navigating multi-class files would
  be a valuable addition but is not part of this instantiation.

---

## Quality Checks

- [x] No `--template--` status files

  All ADRs carry `Accepted`. All CICs carry `Active`. No placeholder status values remain.

- [x] No phantom references

  All cross-references between ADRs, CICs, and contributor protocols resolve to files that
  exist in the repository. Verified manually.

- [x] All cross-ADR references resolve

  Each ADR's References section cites other ADRs by number and title. All cited ADR numbers
  (000–009) have corresponding files in `docs/ADRs/`. Future ADRs (010–013) are listed only
  as candidates in `docs/ADRs/README.md`.

- [ ] Run validate_docs.sh (pending)

  `docs/validate_docs.sh` is present and executable. It has not yet been run against the
  completed documentation set. Run it to catch any remaining phantom references or
  structural issues:

  ```bash
  bash docs/validate_docs.sh
  ```

---

## Notes

- ADR-004 open question on `sequence_number` and RNG is tracked as candidate ADR-011.
- ADR-003 open question on entity-drop errors vs warnings is not yet resolved.
- The manager distributional dispatch path remains untested (known gap, ADR-005).
- The `validate_docs.sh` run should be done after any future documentation additions.
