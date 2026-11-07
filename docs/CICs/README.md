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

## Active Contracts (10)

| CIC | Class | One-line description |
|-----|-------|----------------------|
| [ZeroModel.md](ZeroModel.md) | `ZeroModel` | Trivial all-zeros baseline that validates the prediction pipeline end-to-end without any learned behaviour. |
| [LocfModel.md](LocfModel.md) | `LocfModel` | Last-observation-carried-forward baseline that stores the temporally last training value per entity and replicates it across the forecast horizon. |
| [AverageModel.md](AverageModel.md) | `AverageModel` | Windowed mean baseline that stores per-entity trailing means over a declared `window_months` window and returns them as a flat constant prediction. |
| [ConflictologyModel.md](ConflictologyModel.md) | `ConflictologyModel` | Distributional climatology resampler that draws `n_samples` bootstrap samples from each entity's historical window and returns a `dict[str, PredictionFrame]`. |
| [KDEConflictology.md](KDEConflictology.md) | `KDEConflictology` | Smoothed climatology C (ADR-023): A's zero mass, plus positives from a Gaussian KDE of the window's positive values, reflected at 0.5 and rounded; `bandwidth` is a number or `sj`/`silverman`. |
| [MixtureBaseline.md](MixtureBaseline.md) | `MixtureBaseline` | Distributional local/global mixture baseline that addresses the zero-history trap by blending each entity's local empirical pool with a global positive pool at weight `lambda_mix`. |
| [ParametricConflictology.md](ParametricConflictology.md) | `ParametricConflictology` | No-hurdle parametric climatology (ADR-022) that fits a native-zero family (`nb`) to conflictology's per-entity window and samples from it; `family`/`transform`/`seed` are audited genome keys. |
| [EmpiricalKDEMixtureConflictology.md](EmpiricalKDEMixtureConflictology.md) | `EmpiricalKDEMixtureConflictology` | Smoothed climatology E (ADR-023): per draw, A with probability 1−α, C with probability α; common random numbers make an α-sweep a paired design. |
| [InterpolatedQuantileConflictology.md](InterpolatedQuantileConflictology.md) | `InterpolatedQuantileConflictology` | Smoothed climatology B (ADR-023): A's zero mass, plus positives drawn from the linearly interpolated quantile function of the window's positive values; integer output, no extrapolation. |
| [RollingConflictology.md](RollingConflictology.md) | `RollingConflictology` | Conflictology whose window ends at each rolling origin (`train_end + k`), the information rule views-stepshifter uses; identical to `ConflictologyModel` at origin 0; the control for ADR-024 (#126). |
| [MatryoshkaConflictology.md](MatryoshkaConflictology.md) | `MatryoshkaConflictology` | Conflictology that draws from the cell, its smooth neighbourhood, the rest of its country and the rest of the map, each with a fitted weight and memory, so the forecast opens outward as it looks further ahead (ADR-024); 8 knobs per target, fitted by backtest. |
| [ParametricHurdleConflictology.md](ParametricHurdleConflictology.md) | `ParametricHurdleConflictology` | Hurdle parametric climatology (ADR-022): empirical zero-spike + continuous positive-part family (`lognormal`/`gumbel`/`gamma`), per-sample detransform, non-negativity floor. |
| [BaselineModelCatalog.md](BaselineModelCatalog.md) | `BaselineModelCatalog` | Config-validated model factory that maps algorithm names and config dicts to fully constructed model instances, enforcing required key presence via `MODEL_GENOMES`. |
| [ReproducibilityGate.md](ReproducibilityGate.md) | `ReproducibilityGate` | Canonical hyperparameter contract that defines `CORE_GENOME` and `ALGORITHM_GENOMES` for all baseline models, enforced via `audit_manifest()` before model instantiation. |
| [BaselineForecastingModelManager.md](BaselineForecastingModelManager.md) | `BaselineForecastingModelManager` | Pipeline integration orchestrator that extends `ForecastingModelManager` from views-pipeline-core, routing point-forecast and distributional models through separate prediction and artifact paths. |

---

## Governance Cross-References

The CIC framework and its governance are grounded in the following ADRs:

- **[ADR-006](../ADRs/006_intent_contracts_for_non_trivial_classes.md)** — Mandates CICs for
  non-trivial classes; defines the ten covered classes; specifies what CICs are and are not.

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
