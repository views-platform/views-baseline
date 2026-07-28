# ADR-006: Intent Contracts for Non-Trivial Classes

- **Status:** Accepted
- **Date:** 2026-03-13
- **Deciders:** Project maintainers

---

## Context

views-baseline is a small package, but it contains classes that carry non-obvious responsibilities. The seven model classes implement distinct forecasting strategies with shared surface area — the same `fit` / `predict` signature is used for both point-forecast models that return a `pd.DataFrame` and distributional models that return a `dict[str, PredictionFrame]`. The catalog is an architectural boundary that maps config keys to concrete model instances. The manager is an orchestrator that coordinates data loading, model construction, prediction generation, and artifact persistence across evaluation modes.

These classes are small enough that their docstrings describe *what* they do, but not *why* they exist, what they must never do, what assumptions they rely on, or what downstream components depend on their behaviour. That information lives in developers' heads or in commit messages, neither of which is findable during a code review or an onboarding session.

A Class Intent Contract (CIC) is a short, structured document that captures the intent, responsibilities, invariants, and known limitations of a class, written alongside the code and versioned with it. CICs are not API documentation — they do not restate the method signatures. They document the reasoning that cannot be inferred from the code alone.

---

## Decision

Every non-trivial class in views-baseline has a corresponding CIC in `docs/CICs/`. A class is considered non-trivial if it has more than one distinct responsibility, carries state that influences future behaviour, or sits at an architectural boundary where misuse has downstream consequences.

The two protocol classes (`BaselineModel` and `DistributionalBaselineModel` in `views_baseline/model/protocol.py`) are explicitly excluded from this requirement. Their intent is fully captured in their docstrings and their structural definition: a protocol's contract is its interface. Adding a CIC for a protocol would restate what the `Protocol` class body already says.

---

## Classes Covered

Ten classes have CICs. All CIC documents live in `docs/CICs/`.

### Core Domain Models (7 classes)

These seven classes implement the forecasting strategies. Each CIC describes the strategy's statistical meaning, the state retained after `fit()`, the assumptions made about the input DataFrame's index structure, and the entity-drop behaviour.

- **ZeroModel** — stateless constant-zero baseline. Minimal state (index names only). The CIC records why it still calls `fit()` despite retaining nothing from the data.
- **LocfModel** — last-observation-carried-forward. Stores `last_observations` per entity from the pre-test period. The CIC records the sort-before-groupby invariant that ensures temporal rather than positional last values.
- **AverageModel** — trailing window mean. Stores per-entity means over `window_months` of training data. The CIC records the undefined behaviour when `window_months` exceeds the available history for an entity, and the silent NaN propagation that results.
- **ConflictologyModel** — distributional climatology baseline. Stores the raw `window_months`-length history arrays per entity in `hist_per_entity`. Returns `dict[str, PredictionFrame]`, not `pd.DataFrame`. The CIC records the resampling strategy, the seed semantics, and the entity-iteration ordering that makes samples reproducible.
- **MixtureBaseline** — mixture empirical baseline combining local history with a global positive pool. The CIC records the zero-probability trap problem it solves, the `lambda_mix` parameter's role in controlling the local/global balance, and the degenerate case where the global pool is empty (all training values are zero).
- **ParametricConflictology** — no-hurdle parametric climatology (ADR-022). Fits a native-zero family (`nb`) to conflictology's shared `window_pool` and samples from it. The CIC records the `family`/`transform`/`seed` genome, the fail-loud validation (unsupported family, illegal `family×transform`), the degenerate/underdispersed fallbacks, and the Tweedie exclusion.
- **ParametricHurdleConflictology** — hurdle parametric climatology (ADR-022). Empirical zero-spike + continuous positive-part family (`lognormal`/`gumbel`/`gamma`). The CIC records the per-sample detransform (Jensen), the `EMIT_FLOOR` non-negativity guarantee that closes `gumbel`'s ℝ-support tail, and the `EMIT_LOG_CEIL` overflow guard.

### Architectural Boundary (1 class)

- **BaselineModelCatalog** — factory that maps model names and config dicts to instantiated model objects. The CIC records the `MODEL_GENOMES` structure and its role in config validation, the distinction between required and optional config keys, and the responsibility boundary: the catalog validates config completeness but does not validate config value ranges (e.g., it does not check that `months > 0`).

### Orchestrator (1 class)

- **BaselineForecastingModelManager** — extends `ForecastingModelManager` from `views-pipeline-core`. The CIC records which six symbols are imported from the base package, the decision to always call `_setup_model_and_data()` at the start of both evaluate and forecast operations rather than loading a pickled artifact, the distributional dispatch logic in `_generate_predictions()`, and the known gap that `_evaluate_sweep()` has no test coverage.

### Infrastructure (1 class)

- **ReproducibilityGate** — canonical hyperparameter contract for all baseline models. The CIC records the `CORE_GENOME` and `ALGORITHM_GENOMES` definitions, the four-step `audit_manifest()` validation sequence, the `MissingHyperparameterError` exception, and the single-source-of-truth relationship with `BaselineModelCatalog.MODEL_GENOMES`.

---

## CIC Format

Each CIC contains the following sections:

1. **One-line purpose** — what this class is for, in a single sentence.
2. **Responsibilities** — an explicit list of what this class is responsible for. Anything not on this list is out of scope.
3. **Invariants** — conditions that must be true at all times (e.g., "after `fit()`, `self.last_observations` is never `None`").
4. **Assumptions** — things the class takes for granted about its inputs and environment that are not validated in code.
5. **Known limitations** — documented deviations from ideal behaviour that are accepted as-is.
6. **Downstream dependents** — which other classes or pipeline components rely on this class's output contract.

---

## What CICs Are Not

CICs are not:

- **API documentation.** Method signatures, parameter types, and return types belong in docstrings or type annotations, not CICs.
- **Test specifications.** Test files document expected behaviour through executable assertions. CICs document intent that informs what assertions to write.
- **Design proposals.** CICs describe the class as it exists, not as it should be refactored. Future design decisions belong in ADRs.

---

## Consequences

**Positive:**

- New contributors can understand the intent of a class before reading its implementation, reducing the risk of misuse at module boundaries.
- Code reviewers have a reference for whether a change is consistent with the stated intent of the class being modified.
- Known limitations are documented in a findable location rather than being implicit knowledge.

**Negative / Risks:**

- CICs require maintenance. A class that is refactored without updating its CIC becomes actively misleading. The current enforcement mechanism is a code review convention, not an automated check.
- Ten CICs is a modest ongoing maintenance surface. If the model count grows significantly, the CIC burden grows linearly.

**Explicitly excluded:**

`BaselineModel` and `DistributionalBaselineModel` are structural protocols. Their `@runtime_checkable` decorator and typed method stubs are self-documenting. They have no state, no invariants beyond their structural definition, and no non-obvious responsibilities. A CIC would be redundant.
