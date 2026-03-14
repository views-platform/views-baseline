# Class Intent Contracts — views-baseline

This directory contains the Class Intent Contracts (CICs) for every non-trivial class in
views-baseline. CICs are required by [ADR-006](../ADRs/006_intent_contracts_for_non_trivial_classes.md).

A CIC is not API documentation. It records the intent, responsibilities, invariants, and known
limitations of a class — the reasoning that cannot be inferred from the code alone. Use CICs
alongside the code when reviewing a PR, onboarding, or investigating unexpected behaviour.

The two protocol classes (`BaselineModel` and `DistributionalBaselineModel`) are explicitly
excluded from this requirement; their `@runtime_checkable` Protocol bodies are
self-documenting. See ADR-006 for the rationale.

Use the [cic_template.md](cic_template.md) when writing a new CIC.

---

## Active Contracts (7)

| CIC | Class | One-line description |
|-----|-------|----------------------|
| [ZeroModel.md](ZeroModel.md) | `ZeroModel` | Trivial all-zeros baseline that validates the prediction pipeline end-to-end without any learned behaviour. |
| [LocfModel.md](LocfModel.md) | `LocfModel` | Last-observation-carried-forward baseline that stores the temporally last training value per entity and replicates it across the forecast horizon. |
| [AverageModel.md](AverageModel.md) | `AverageModel` | Windowed mean baseline that stores per-entity trailing means over a declared `window_months` window and returns them as a flat constant prediction. |
| [ConflictologyModel.md](ConflictologyModel.md) | `ConflictologyModel` | Distributional climatology resampler that draws `n_samples` bootstrap samples from each entity's historical window and returns a `dict[str, PredictionFrame]`. |
| [MixtureBaseline.md](MixtureBaseline.md) | `MixtureBaseline` | Distributional local/global mixture baseline that addresses the zero-history trap by blending each entity's local empirical pool with a global positive pool at weight `lambda_mix`. |
| [BaselineModelCatalog.md](BaselineModelCatalog.md) | `BaselineModelCatalog` | Config-validated model factory that maps algorithm names and config dicts to fully constructed model instances, enforcing required key presence via `MODEL_GENOMES`. |
| [BaselineForecastingModelManager.md](BaselineForecastingModelManager.md) | `BaselineForecastingModelManager` | Pipeline integration orchestrator that extends `ForecastingModelManager` from views-pipeline-core, routing point-forecast and distributional models through separate prediction and artifact paths. |

---

## Governance Cross-References

The CIC framework and its governance are grounded in the following ADRs:

- **[ADR-006](../ADRs/006_intent_contracts_for_non_trivial_classes.md)** — Mandates CICs for
  non-trivial classes; defines the seven covered classes; specifies what CICs are and are not.

- **[ADR-003](../ADRs/003_authority_of_declarations_over_inference.md)** — The principle that
  governs what invariants CICs must document (declared quantities, not inferred ones). CIC
  sections on Responsibilities and Failure Modes should be consistent with the declarations
  principle: if a class infers where it should declare, that is a known deviation to record.

- **[ADR-005](../ADRs/005_testing_as_mandatory_critical_infrastructure.md)** — The
  three-team testing model (Green/Beige/Red) is the reference for the Test Alignment section
  of every CIC. Each CIC's Test Alignment section should map its coverage to this taxonomy.

---

## Maintenance Obligation

A CIC must be updated when the intent of the class changes. A class that is refactored without
updating its CIC becomes actively misleading. If a behaviour change is intentional and
permanent, update the CIC as part of the same PR.

The enforcement mechanism is code review convention, not automation.
