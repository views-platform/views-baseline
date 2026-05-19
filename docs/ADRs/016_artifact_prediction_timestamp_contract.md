# ADR-016: Artifact-Prediction Timestamp Contract

**Status:** Accepted
**Date:** 2026-05-19
**Deciders:** Simon, VIEWS platform team
**Consulted:** views-pipeline-core ADR-052 (central contract)

---

## Context

`BaselineForecastingModelManager._setup_model_and_data()` contained a line that overwrote the run configuration timestamp with `datetime.now()` on every call:

```python
self.config["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
```

This violated the pipeline-wide timestamp contract (views-pipeline-core ADR-052, ADR-013): prediction filenames must carry the timestamp of the **trained model artifact**, not the current wall-clock time. The violation had two compounding effects:

1. **Silent item-assignment no-op.** `self.config` is a property on `ForecastingModelManager` that returns a **new dictionary** from `get_combined_config()` on every call. Assigning to the returned dict modifies a transient copy — the value is silently lost. So the overwrite was both wrong in intent *and* ineffective in practice. The timestamp actually used came from `ConfigurationManager.__init__`, which happened to work for single-model runs but failed when the ensemble manager tried to match prediction files against artifact timestamps.

2. **Ensemble integration failure.** The ensemble manager resolves constituent model predictions by matching filenames against artifact timestamps. When baseline models produced predictions with non-artifact timestamps, the ensemble fell back to expensive subprocess re-execution via `run.sh`, coupling the ensemble environment to the views-baseline package.

The bug was discovered during integration testing of synthetic ensemble models in May 2026.

---

## Decision

`BaselineForecastingModelManager` now follows the canonical timestamp extraction pattern established in views-pipeline-core ADR-052:

- `_evaluate_model_artifact()` and `_forecast_model_artifact()` each resolve the artifact path via `get_latest_model_artifact_path()`, extract the 15-character timestamp from the filename stem, and persist it via `self._config_manager.add_config({"timestamp": ...})`.
- `_setup_model_and_data()` no longer touches timestamps. Its responsibility is model instantiation and data loading only.
- The `datetime` import was removed as it is no longer needed.

### Scope

- **In scope:** `BaselineForecastingModelManager` in this repo.
- **Out of scope:** All other baseline infrastructure (catalog, reproducibility gate, protocols).

---

## Rationale

- **Contract compliance.** Three of four model-specific repos (views-hydranet, views-stepshifter, views-r2darts2) already implemented this correctly. Baseline was the sole outlier.
- **Correct API usage.** `_config_manager.add_config()` is the only way to persist configuration changes through the `ForecastingModelManager` property boundary. Direct assignment to `self.config[...]` is a known no-op.
- **Decoupling.** Correct timestamps eliminate the ensemble's need to fall back to subprocess re-execution of constituent models.

---

## Considered Alternatives

### Alternative A: Keep `datetime.now()` but use `add_config` to persist it

- **Pros:** Minimal change; fixes the persistence bug.
- **Cons:** Still violates ADR-013/052. Predictions would have fresh timestamps, breaking artifact-to-prediction traceability.
- **Reason for rejection:** Fixes the symptom, not the root cause.

### Alternative B: Extract timestamp in `_setup_model_and_data()` instead of evaluate/forecast

- **Pros:** Single extraction point.
- **Cons:** `_setup_model_and_data()` doesn't have access to the artifact path — it's a model+data setup method, not an artifact resolution method. `_train_model_artifact()` also calls it, and during training there is no prior artifact to extract from.
- **Reason for rejection:** Wrong responsibility boundary.

---

## Consequences

### Positive

- Baseline predictions now carry artifact-derived timestamps, matching all other model repos.
- Ensemble manager can resolve baseline predictions without subprocess fallback.
- 75/75 existing tests pass after the fix.

### Negative

- Any existing prediction files generated before this fix have timestamps from `ConfigurationManager.__init__` (which happened to be close to the artifact timestamp for single-model runs). No migration needed — these files are functionally correct by coincidence.

---

## Implementation Notes

- **Fix commit:** On branch `bugfix/artifact-timestamp-extraction`.
- **Changed file:** `views_baseline/manager/baseline_manager.py`.
- **Test update:** `tests/conftest.py` mock updated to include `get_latest_model_artifact_path` on the `SimpleNamespace` mock.
- **Pattern to follow** (from `_evaluate_model_artifact`):
  ```python
  path_artifact = self._model_path.get_latest_model_artifact_path(
      run_type=self.config["run_type"]
  )
  self._config_manager.add_config({"timestamp": path_artifact.stem[-15:]})
  ```

---

## Validation & Monitoring

- **Test coverage:** Existing `test_evaluate_model_artifact_*` and `test_forecast_model_artifact_*` tests verify the flow. The mock now provides a path with a parseable timestamp stem.
- **Integration signal:** Synthetic ensemble (`synthetic_chorus`) exercises three baseline models end-to-end and validates timestamp-matched prediction resolution.
- **Failure signal:** Ensemble evaluation failing with "prediction file not found" for baseline constituents.

---

## Open Questions

- None. The fix aligns baseline with the established pattern in all other model repos.

---

## References

- views-pipeline-core ADR-052: Artifact-Prediction Timestamp Contract (central)
- views-pipeline-core ADR-013: Prediction Naming Convention
- views-hydranet ADR-026: Model Artifact Fetcher Specification
- `views_pipeline_core/templates/package/template_example_manager.py:217`: Canonical pattern
