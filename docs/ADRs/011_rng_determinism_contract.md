# ADR-011: RNG Determinism Contract and Seed Arithmetic

**Status:** Accepted
**Date:** 2026-03-13
**Deciders:** Project maintainers

---

## Context

Both distributional models (`ConflictologyModel` and `MixtureBaseline`) use random sampling to generate predictions. Reproducibility is critical for scientific credibility and for debugging: given the same inputs, the same model configuration must produce identical outputs.

The RNG strategy needs to be documented as a contract because:

1. Changing iteration order silently changes which random numbers are assigned to which cells, breaking reproducibility without any visible code change in the sampling logic.
2. The relationship between `seed` and `sequence_number` must be explicit — if the seed incorporates `sequence_number`, different evaluation sequences produce different samples (desirable for some use cases but not the current one).
3. Future contributors adding new distributional models must follow the same contract to maintain consistency.

---

## Decision

**The RNG determinism contract for distributional models is as follows:**

### 1. Fresh RNG per `predict()` call

Both distributional models create a fresh RNG at the start of each `predict()` call:

```python
rng = np.random.default_rng(self.seed)
```

This means: calling `predict()` twice with the same arguments produces identical output, regardless of prior calls. The RNG state does not accumulate across calls.

### 2. Fixed seed, not sequence-dependent

The seed is a constructor parameter (default `42`), passed at model instantiation:

```python
ConflictologyModel(targets=..., window_months=..., partition_dict=..., loa=..., seed=42)
```

The seed is **not** modified by `sequence_number`. The `sequence_number` parameter only shifts the prediction time window (`prediction_start = test_start + sequence_number`). Two different sequence numbers with the same seed produce the same random draws mapped to different time indices.

### 3. Iteration order: entity → time → target

Both models iterate in the same order to consume RNG values:

```python
for cid in entities:          # entity (outer)
    for _ in time_ids:        # time (middle)
        for t in self.targets:  # target (inner)
            y_preds[t][idx] = rng.choice(...)  # or self._sample(...)
```

This order is load-bearing. Changing it reassigns random draws to different cells, producing different predictions for the same seed. Any change to iteration order is a **breaking change** requiring a new ADR.

### 4. Scope of the reproducibility guarantee

**Same seed + same input data + same model parameters = identical output.**

The guarantee covers:
- Identical `y_pred` arrays in every `PredictionFrame`
- Identical `identifiers` arrays (time, unit)
- Identical dict keys

The guarantee does **not** cover:
- Cross-platform reproducibility (NumPy RNG may differ across architectures, though `default_rng` is designed to be portable)
- Reproducibility across NumPy major versions (the `default_rng` contract is versioned by NumPy)

---

## Rationale

- **Fresh RNG per call** is the simplest strategy that guarantees idempotent `predict()`. An alternative (persistent RNG state across calls) would make output depend on call history, which is fragile in a pipeline that may call `predict()` in different orders.
- **Fixed seed (not sequence-dependent)** was chosen because the current use case does not require independent samples across sequence numbers. The `sequence_number` shifts the time window, not the sampling distribution. If independent samples per sequence are needed in the future, this should be a deliberate change recorded in a new ADR.
- **entity → time → target** order matches both models. It was not chosen for mathematical reasons but must be preserved for reproducibility. Documenting it here prevents accidental reordering.

---

## Considered Alternatives

### Alternative A: Seed arithmetic incorporating sequence_number

`rng = np.random.default_rng(self.seed + sequence_number)`

- **Pros:** Different sequences produce different samples, which may be desirable for ensemble diversity.
- **Cons:** Changes the current contract; existing tests would break; the meaning of "reproducible" changes.
- **Reason for rejection:** Not needed for the current use case. Can be adopted later via a new ADR if ensemble diversity across sequences is required.

### Alternative B: Persistent RNG state across predict() calls

Create `self.rng` in `__init__` or `fit()`, reuse across `predict()` calls without resetting.

- **Pros:** Different calls produce different samples (useful if calling predict() multiple times for bootstrap).
- **Cons:** Output depends on call order and history. Breaks idempotency. Harder to debug.
- **Reason for rejection:** Idempotent `predict()` is more valuable than call-order-dependent diversity.

### Alternative C: Per-cell seeding

`rng_cell = np.random.default_rng(self.seed + hash((cid, tid, target)))`

- **Pros:** Fully parallelizable; adding/removing entities does not affect other cells.
- **Cons:** Expensive (one RNG per cell); `hash()` is not stable across Python sessions without `PYTHONHASHSEED`.
- **Reason for rejection:** Over-engineered for the current scale.

---

## Consequences

### Positive

- `predict()` is idempotent — safe to retry, cache, or call speculatively.
- Reproducibility is verified by `test_mixture_predict_reproducible` (and analogous ConflictologyModel tests).
- The contract is simple enough that new distributional models can follow it mechanically.

### Negative

- Different sequence numbers produce the same random draws (mapped to different time windows). This means predictions for sequence 0 and sequence 5 are not statistically independent.
- Changing iteration order (e.g., for performance) is a breaking change that silently alters outputs if the seed is unchanged.
- The contract is only as stable as NumPy's `default_rng`. A NumPy major version upgrade could change the stream.

---

## Implementation Notes

The contract is implemented in two locations:

**`ConflictologyModel.predict()`** (`views_baseline/model/baseline.py`):
```python
rng = np.random.default_rng(self.seed)
# entity → time → target loop
for cid in entities_with_history:
    for _ in time_ids:
        for t in self.targets:
            y_preds[t][idx] = rng.choice(...)
```

**`MixtureBaseline.predict()`** (`views_baseline/model/baseline.py`):
```python
rng = np.random.default_rng(self.seed)
# entity → time → target loop
for cid in entities_with_pool:
    for _ in time_ids:
        for t in self.targets:
            y_preds[t][idx] = self._sample(cid, t, rng)
```

New distributional models must follow this pattern: fresh `default_rng(self.seed)` at the top of `predict()`, entity → time → target iteration order.

---

## Validation & Monitoring

- `tests/test_baseline.py::test_mixture_predict_reproducible` — calls `predict()` twice with the same seed and asserts bitwise equality of `y_pred` arrays.
- `tests/test_baseline.py::test_conflictology_predict_reproducible` — same pattern for ConflictologyModel.
- Failure mode: if a refactor reorders the iteration loop, these tests will fail because the RNG stream is consumed in a different order.

---

## Open Questions

- Should different sequence numbers produce independent samples? This would require seed arithmetic (`seed + sequence_number` or similar). Currently not needed, but would enable richer ensemble diagnostics.
- Should the iteration order be formalized as a protocol requirement (e.g., a method `_iter_cells()` that enforces the order)?
- Should the NumPy version be pinned to protect the RNG stream? Currently no — the risk is low and pinning creates maintenance burden.

---

## References

- ADR-004: Rules for Evaluation and Stability (RNG reproducibility open question)
- `views_baseline/model/baseline.py` — `ConflictologyModel.predict()`, `MixtureBaseline.predict()`
- `tests/test_baseline.py` — reproducibility tests
- NumPy documentation: `numpy.random.default_rng` guarantees
