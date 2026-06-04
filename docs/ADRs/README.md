# Architecture Decision Records — views-baseline

This directory contains the constitutional ADRs for the views-baseline project. ADRs are
append-only records of significant architectural and design decisions. They are numbered
sequentially and each has a stable identifier that can be cited in code, PRs, and other ADRs.

See [ADR-000](000_use_of_adrs.md) for the policy governing this directory.

---

## Constitutional ADRs (000–009)

| ADR | Title | One-line description |
|-----|-------|----------------------|
| [ADR-000](000_use_of_adrs.md) | Use of Architecture Decision Records | Establishes that significant decisions are recorded as numbered, append-only ADR files in `docs/ADRs/`. |
| [ADR-001](001_ontology_of_the_repository.md) | Ontology of the Repository | Defines the six authoritative categories (point models, distributional models, protocols, factory, pipeline integration, prediction builders) and their membership. |
| [ADR-002](002_topology_and_dependency_rules.md) | Topology and Dependency Rules | Specifies the allowed import directions (helpers → models → catalog → manager) and mandates the lazy-import pattern for `PredictionFrame` inside `model/`. |
| [ADR-003](003_authority_of_declarations_over_inference.md) | Authority of Declarations over Inference | Forbids inferring model type from output shape, target columns from DataFrame content, or train/test boundaries from index range; all must be declared explicitly. |
| [ADR-004](004_rules_for_evaluation_and_stability.md) | Rules for Evaluation and Stability | Classifies each component as Stable or Evolving, defines the RNG reproducibility contract (`default_rng(seed)`), and lists the five trigger conditions for an ADR update. |
| [ADR-005](005_testing_as_mandatory_critical_infrastructure.md) | Testing as Mandatory Critical Infrastructure | Mandates automated test coverage via the three-team model (Green/Beige/Red), documents the 51-test suite, and records known gaps including the untested manager distributional path. |
| [ADR-006](006_intent_contracts_for_non_trivial_classes.md) | Intent Contracts for Non-Trivial Classes | Requires a Class Intent Contract (CIC) in `docs/CICs/` for every non-trivial class; excludes the two protocol classes as self-documenting structural contracts. |
| [ADR-007](007_silicon_based_agents_as_untrusted_contributors.md) | Silicon-Based Agents as Untrusted Contributors | Classifies AI coding assistants (Claude Code) as untrusted contributors, requires `Co-Authored-By` attribution, and enforces the same lint + test gates as for human-authored code. |
| [ADR-008](008_observability_and_explicit_failure.md) | Observability and Explicit Failure | Defines the two-tier observability approach: fail loud on configuration errors (ValueError), log at WARNING on data-driven entity-drop events; documents the absence of ERROR-level signalling as accepted debt. |
| [ADR-009](009_boundary_contracts_and_configuration_validation.md) | Boundary Contracts and Configuration Validation | Maps the five module boundaries (Config→Catalog, Catalog→Model, Model→Pipeline, Data→Model, Manager→core), documents what is validated at each, and records the known gaps as accepted debt. |

---

## Governance Structure

**Writing a new ADR.** Use the [adr_template.md](adr_template.md) in this directory. Number it
sequentially after the last existing ADR. Any contributor may propose an ADR; it is accepted
when the project maintainers agree.

**When is an ADR required?** ADR-004 lists five trigger conditions:

1. A `views-pipeline-core` API change that requires code changes in `manager/` or distributional `predict()`.
2. Addition of a new model class (requires updating `MODEL_GENOMES`, the stability map, and ADR-001).
3. A `PredictionFrame` schema change upstream.
4. A protocol change in `BaselineModel` or `DistributionalBaselineModel`.
5. A stability reclassification of any component.

Any change that introduces a new dependency, alters the module topology, or changes a
declared quantity (targets, partition dict usage, seed arithmetic) also warrants an ADR.

**Superseding an ADR.** Never edit the decision of an existing ADR. Write a new ADR and
update the old ADR's status to `Superseded by ADR-NNN`.

**Recommended adoption order for new contributors.** See the section below.

---

## Project-Specific ADRs (010+)

| ADR | Title | One-line description |
|-----|-------|----------------------|
| [ADR-010](010_prediction_frame_adoption.md) | PredictionFrame as Universal Output Format | All baseline models return `dict[str, PredictionFrame]`; point models are `(N, 1)`. No DataFrame output path. |
| [ADR-011](011_rng_determinism_contract.md) | RNG Determinism Contract and Seed Arithmetic | Documents the fresh-RNG-per-predict, fixed-seed, entity→time→target iteration contract for reproducibility. |
| [ADR-012](012_model_addition_protocol.md) | Model Addition Protocol and Catalog Registration | Authoritative 7-step checklist for adding a new model (define, register, test, document). |
| [ADR-013](013_pipeline_core_coupling.md) | views-pipeline-core Coupling Management | Zero module-level imports in `model/`, lazy `PredictionFrame` imports (incl. `build_prediction_frame`), free imports in `manager/`. |
| [ADR-014](014_reproducibility_gate.md) | Reproducibility Gate | `CORE_GENOME` / `ALGORITHM_GENOMES` hyperparameter contract enforced via `audit_manifest()`. |
| [ADR-015](015_technical_risk_register.md) | Technical Risk Register | Establishes the register as a governance artifact tracking risks across the codebase. |
| [ADR-016](016_artifact_prediction_timestamp_contract.md) | Artifact-Prediction Timestamp Contract | Prediction filenames carry the trained-artifact timestamp, not wall-clock time. |
| [ADR-017](017_frames_replace_dataframes.md) | Frames Replace DataFrames as Operational Containers | PredictionFrame (output, in core) + FeatureFrame (input, in datafactory, pending); allows `n_samples == 1`. |
| [ADR-018](018_predictionframe_use_in_views_baseline.md) | views-baseline's Use of PredictionFrame (Output) | **Implemented.** All models return `dict[str, PredictionFrame]`; output side done in this repo. |
| [ADR-019](019_featureframe_use_in_views_baseline.md) | views-baseline's Use of FeatureFrame (Input) | **Proposed / NOT implemented.** Baseline still takes DataFrame input; blocked on pipeline-core FeatureFrame integration. |

---

## Recommended Adoption Order

New contributors to views-baseline should read the ADRs in the following order:

1. **ADR-000** — Understand why ADRs exist and how to use this directory.
2. **ADR-001** — Learn the six ontological categories and which files/classes belong to each.
3. **ADR-002** — Understand the dependency topology and forbidden import patterns.
4. **ADR-003** — Understand the declarations-over-inference principle before reading any model code.
5. **ADR-005** — Understand the testing obligations and the three-team model before writing any tests.
6. **ADR-006** — Read CICs alongside code when investigating a specific class.
7. **ADR-004** — Consult when planning a change to understand stability obligations and trigger conditions.
8. **ADR-007** — Read before using AI coding assistance in this project.
9. **ADR-008** — Consult when adding logging or failure signalling to any module.
10. **ADR-009** — Consult when working at any module boundary.
