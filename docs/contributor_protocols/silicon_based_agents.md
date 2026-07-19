# Contributor Protocol: Silicon-Based Agents

**Applies to:** AI coding assistants operating on the views-baseline codebase.
**Primary tool in use:** Claude Code
**Last reviewed:** 2026-03-13
**Governing ADR:** [ADR-007](../ADRs/007_silicon_based_agents_as_untrusted_contributors.md)

---

## Classification

AI coding assistants are **untrusted contributors**. This does not mean prohibited or
suspected of malice. It means that AI output is treated the same way a contribution from an
unfamiliar external contributor would be treated: with the assumption that it may contain
errors, may not conform to project conventions, and has not been verified against the
project's history of decisions.

AI-generated code is not assumed correct because it looks correct.
AI-generated tests are not assumed meaningful because they pass.
AI-generated refactors are not assumed to preserve behaviour because the diff is clean.

---

## Mandatory Gates

Every change — human or AI — must pass both gates before merging:

```bash
ruff check .
pytest
```

These run automatically in GitHub Actions CI on every push and PR. There are no exceptions.
No AI-generated change bypasses lint or tests.

Passing these gates confirms: syntactic validity, style conformance, and consistency with
what the existing tests check. It does **not** confirm:

- Semantic correctness for cases not covered by existing tests
- That the AI understood the intent of the change
- That an AI-generated test is testing the right thing (not a tautology)

The test coverage gaps documented in ADR-005 are therefore also gaps in AI-output
verification. The manager distributional path, Red team tests, and entity-drop warning
assertions are all uncovered by CI.

---

## Attribution

Every commit that includes AI-generated code must include a `Co-Authored-By` git trailer:

```
Co-Authored-By: Claude <model-identifier> <noreply@anthropic.com>
```

This serves two purposes: traceability (if a defect is traced to a commit, the header flags
AI involvement) and normalisation (AI use is a visible, auditable part of the project record,
not something omitted to obscure provenance).

---

## The Anti-Truncation Rule

`views_baseline/model/baseline.py` contains seven closely related model
classes. When asked to modify one class, do not truncate the file. Do not emit `# ... rest
of file unchanged ...` or equivalent. The full file must be preserved. Use the Edit tool
(targeted string replacement) rather than a full-file rewrite wherever possible.

This rule exists because truncation has previously caused class definitions to be silently
dropped in large files. Any AI output that shortens `baseline.py` significantly without a
corresponding deletion of code is suspect and must be rejected.

---

## Sensitive Areas: Handle With Care

The following areas in the codebase have known sensitivity to the order of operations,
attribute presence, or import structure. Changes to them require explicit human review and
may not be merged on AI recommendation alone.

### RNG ordering sensitivity

`ConflictologyModel.predict()` and `MixtureBaseline.predict()` generate samples in a fixed
order: entity iteration → time step iteration → target iteration. This order is the
reproducibility contract. The test `test_mixture_predict_reproducible` validates it.

**Forbidden without updating seed contracts and tests:**
- Changing the order of entity, time step, or target iteration in distributional `predict()`
- Adding or removing `.shuffle()`, `.permutation()`, or equivalent calls before sample generation
- Splitting the entity loop into parallel or batched calls that change RNG draw order
- Changing `np.random.default_rng(self.seed)` to any other RNG initialisation pattern

### Protocol marker sensitivity

> **Status note (ADR-017, ADR-020):** Since PR #15 the manager has a **single, type-uniform
> prediction path** — it no longer dispatches via `isinstance(model, DistributionalBaselineModel)`.
> The `distributional = True` attribute is now a **semantic marker** (it still satisfies the
> `DistributionalBaselineModel` protocol and documents intent per ADR-012), not a dispatch
> discriminator. Earlier revisions of this protocol described an `isinstance` dispatch in the
> manager that no longer exists.

**Forbidden:**
- Removing the `distributional = True` class attribute from `ConflictologyModel` or
  `MixtureBaseline` (even as a "cleanup" — it is the semantic marker and protocol discriminator)
- Adding a `distributional` attribute to any point-forecast model class

### Lazy import fragility

> **Status note (ADR-020):** `PredictionFrame` now comes from the `views_frames` leaf
> (`from views_frames import PredictionFrame, SpatioTemporalIndex`), re-exported by
> `views-pipeline-core` ≥3.0.0 (#188). Per ADR-020 the construction is **consolidated into a
> single function-scoped import site**, `to_prediction_frames` in `model/helpers.py`. The
> distributional models no longer construct `PredictionFrame` inline.

The `views_frames` import lives inside `to_prediction_frames`, not at module top-level. This is
intentional (ADR-002, as amended by ADR-020).

**Forbidden:**
- Moving this import to the module top-level of `baseline.py` or `helpers.py`
- Adding any other `views_frames` / `views_pipeline_core` import to `model/` at module level
- Re-introducing a second `PredictionFrame` construction site outside `to_prediction_frames`

### Entity-drop warnings

`LocfModel`, `AverageModel`, `ConflictologyModel`, and `MixtureBaseline` all emit a
`logger.warning` when entities are dropped. These warnings are the primary operational
signal that predictions are incomplete.

**Forbidden:**
- Removing entity-drop warning calls
- Downgrading entity-drop warnings to `logger.info` or `logger.debug`
- Replacing the count-based message with a less informative one

### Model catalog consistency

`MODEL_GENOMES` in `BaselineModelCatalog` must be kept in sync with the actual required
constructor parameters of each model. A model whose constructor requires `window_months`
but whose `MODEL_GENOMES` entry is `[]` will fail at runtime, not at validation time.

**Forbidden:**
- Adding a required parameter to a model constructor without updating `MODEL_GENOMES`
- Removing a key from `MODEL_GENOMES` without verifying the constructor no longer requires it

---

## Forbidden Operations

Beyond the sensitive areas above, the following operations are absolutely forbidden for
AI-generated changes without an explicit human decision recorded in an ADR or PR:

1. Introducing any import from `manager/` into any file under `model/`
2. Adding a module-level `views_frames` / `views_pipeline_core` import to any `model/` file
3. Changing the output type of **any** model's `predict()` from `dict[str, PredictionFrame]`
   (every baseline model returns this since PR #15 / ADR-017 — point models `(N, 1)`, distributional
   `(N, n_samples)`)
4. Constructing a `PredictionFrame` anywhere other than the single seam `to_prediction_frames`
   (`model/helpers.py`), or reintroducing the `identifiers=` constructor (ADR-020)
5. Modifying the `MODEL_GENOMES` dict without a corresponding change to a model constructor
6. Adding state-mutation side effects to `predict()` methods (predict must be side-effect-free
   beyond logging)
7. Deriving the spatial `level` by inference alone instead of from the declared `loa` validated
   against `entity_idx` (ADR-003, ADR-020)
8. Replacing `partition_dict["test"][0]` lookups with index-derived train/test boundary inference

---

## What AI Assistance Is Appropriate For

The following categories of work are well-suited to AI assistance in this project, with the
standard human review gate applied:

- **Boilerplate-heavy tasks:** test fixtures, repetitive model variants, helper functions
- **Consistency checks:** identifying mismatches between CICs and code, between `MODEL_GENOMES`
  and constructor signatures, between ADR references and actual file paths
- **Refactoring within a single class:** extracting a method, renaming a variable, improving
  a docstring — provided the change does not touch any of the sensitive areas above
- **Draft ADRs and CICs:** AI can produce a well-structured draft that a human then reviews
  for accuracy and completeness. Documentation produced by AI is not subject to automated
  verification and relies entirely on human review.
- **Identifying test gaps:** AI can enumerate code paths not covered by the existing test suite
  and suggest test cases; the test cases must be reviewed for tautology before merging

---

## Interaction with the Three-Team Testing Model

From ADR-005:

- **Green team tests** that an AI writes must have an independently computed expected value.
  Do not accept tests of the form `assert result == model.predict(df)` — these are tautologies.
- **Beige team tests** that cross the `model/`–`manager/` boundary must use the monkeypatching
  pattern established in `test_baseline_manager.py`, not new mocking strategies.
- **Red team tests** are adversarial. AI can enumerate adversarial cases but the human reviewer
  must confirm that the expected failure mode is correct.

---

## Escalation

If an AI-generated change touches a sensitive area listed above and the human reviewer is
uncertain, the correct action is to block the merge and open a discussion in the PR, not to
merge and observe consequences in production.
