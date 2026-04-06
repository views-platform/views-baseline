# Story: ReproducibilityGate for views-baseline

**Status:** Proposed
**Date:** 2026-04-06
**Origin:** views-models risk register C-05
**Reference implementation:** `views_r2darts2.infrastructure.reproducibility_gate.ReproducibilityGate`

---

## Problem

views-baseline has no canonical definition of which hyperparameters each baseline algorithm requires. A missing hyperparameter in a model's `config_hyperparameters.py` surfaces only at training time.

In views-models, 6 baseline models each define their own HP configs. There is no way to validate these configs statically because the package doesn't expose what it expects.

views-r2darts2 solved this with a `ReproducibilityGate` class that defines `CORE_GENOME` (params required by all models) and `ALGORITHM_GENOMES` (params required per architecture). This contract is enforced at runtime and importable by downstream tests.

views-baseline already has strong governance (14 ADRs, 8 CICs) but lacks this specific reproducibility contract. This story proposes adding one, consistent with the existing governance framework.

---

## Scope

### CORE_GENOME (required by all baseline models)

Based on empirical analysis of all 6 baseline model configs in views-models:

```python
CORE_GENOME = ["steps", "time_steps"]
```

Baseline models are intentionally simple — most have no tunable estimator parameters.

### ALGORITHM_GENOMES (per-algorithm requirements)

| Algorithm | Additional required keys | Notes |
|-----------|------------------------|-------|
| `AverageModel` | `window_months` | Rolling window size (60 for CM, 18 for PGM) |
| `ZeroModel` | — | No additional params (always predicts zero) |
| `LocfModel` | — | No additional params (last observation carried forward) |
| `MixtureBaseline` | TBD | Depends on mixture configuration |

### Runtime enforcement

The gate should audit hyperparameters when `BaselineForecastingModelManager` initializes a model run. Integration point: the existing `BaselineForecastingModelManager` CIC should be updated to document this gate as a precondition.

### Importable contract

The gate must be importable by views-models tests:

```python
from views_baseline.infrastructure.reproducibility_gate import ReproducibilityGate

CORE_PARAMS = set(ReproducibilityGate.CORE_GENOME)
ALGO_PARAMS = ReproducibilityGate.ALGORITHM_GENOMES
```

---

## Acceptance Criteria

1. `ReproducibilityGate` class exists with `CORE_GENOME` and `ALGORITHM_GENOMES`
2. Runtime audit runs before training in `BaselineForecastingModelManager`
3. `MissingHyperparameterError` raised for missing params
4. views-models can import the gate and validate all baseline models
5. Tests in views-baseline cover the gate itself
6. `BaselineForecastingModelManager` CIC updated to document the gate

---

## Notes

- Baseline's gate will be much simpler than DARTS (4 algorithms, minimal params). This is a feature, not a limitation — baseline models are intentionally simple.
- The existing ADR and CIC infrastructure in views-baseline means this work can follow established patterns. Consider whether this warrants a new ADR (e.g., ADR-014: Reproducibility Gate) or fits within existing ADR-011 (RNG Determinism).
- `MixtureBaseline` param requirements need investigation — it may have more complex configuration than the other three algorithms.
