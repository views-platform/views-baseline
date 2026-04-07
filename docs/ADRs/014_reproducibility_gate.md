# ADR-014: Reproducibility Gate

- **Status:** Accepted
- **Date:** 2026-04-07
- **Deciders:** Project maintainers

---

## Context

views-baseline has no canonical, importable definition of which hyperparameters each baseline algorithm requires. The `BaselineModelCatalog` validates algorithm-specific keys (`window_months`, `n_samples`, `lambda_mix`) at model instantiation time, but pipeline-level keys (`steps`, `time_steps`) are consumed silently by the manager without any presence check. A missing `time_steps` key would surface as a `KeyError` deep inside `_generate_predictions`, far from the configuration boundary where it belongs.

Downstream in views-models, 11 baseline model configs each declare their own `config_hyperparameters.py` dicts. There is no way for views-models tests to validate these configs statically against what views-baseline actually expects, because the contract is implicit — spread across the catalog's `MODEL_GENOMES`, the manager's `self.config["time_steps"]` access, and individual model `__init__` signatures.

views-r2darts2 solved this problem with a `ReproducibilityGate` class that defines `CORE_GENOME` (params required by all models) and `ALGORITHM_GENOMES` (params required per architecture). This contract is enforced at runtime via `audit_manifest()` and is importable by downstream tests.

---

## Decision

Add a `ReproducibilityGate` class to `views_baseline.infrastructure.reproducibility_gate` that serves as the single source of truth for baseline hyperparameter requirements.

### Structure

The gate defines:

- **`CORE_GENOME`**: `["steps", "time_steps"]` — required by all baseline models regardless of algorithm.
- **`ALGORITHM_GENOMES`**: per-algorithm required keys, identical to the former `BaselineModelCatalog.MODEL_GENOMES`.

The gate's `audit_manifest(config)` static method validates:
1. All core keys are present.
2. The algorithm is registered.
3. All algorithm-specific keys are present.
4. No required key is `None`.

### Integration

- `BaselineModelCatalog.MODEL_GENOMES` becomes an alias for `ReproducibilityGate.Config.ALGORITHM_GENOMES`.
- `BaselineForecastingModelManager._setup_model_and_data()` calls `ReproducibilityGate.Config.audit_manifest(self.config)` as its first operation, before catalog construction.
- Violations raise `MissingHyperparameterError` (a subclass of `ReproducibilityError`).

### Importable contract

Downstream packages can validate configs statically:

```python
from views_baseline.infrastructure.reproducibility_gate import ReproducibilityGate

CORE_PARAMS = set(ReproducibilityGate.Config.CORE_GENOME)
ALGO_PARAMS = ReproducibilityGate.Config.ALGORITHM_GENOMES
```

---

## Consequences

**Positive:**

- Missing pipeline-level keys (`steps`, `time_steps`) are caught at the configuration boundary, not deep inside prediction logic.
- views-models can validate all 11 baseline configs against the gate in CI, catching mismatches before runtime.
- The catalog's algorithm-specific validation remains as defense-in-depth, but the authoritative genome definitions live in one place.
- The pattern matches views-r2darts2's `ReproducibilityGate`, providing cross-project consistency.

**Negative:**

- Existing manager test configs that omitted `steps` required updating (8 configs across `test_baseline_manager.py`). This is a one-time cost that reflects the configs' prior incompleteness.

**Scope limitation:**

- The baseline gate implements only `Config` validation (no `Temporal` or `Data` gates). Baseline models use pandas DataFrames, not Darts TimeSeries, and have no optimizer/loss configuration. These gates can be added later if the need arises.

---

## Related

- **Story:** `docs/stories/reproducibility_gate.md`
- **Origin:** views-models risk register C-05
- **Reference implementation:** `views_r2darts2.infrastructure.reproducibility_gate.ReproducibilityGate`
- **CIC:** `docs/CICs/ReproducibilityGate.md`
