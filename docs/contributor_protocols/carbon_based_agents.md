# Contributor Protocol: Carbon-Based Agents

**Applies to:** Human contributors to the views-baseline project.
**Last reviewed:** 2026-03-13

---

## Context

views-baseline is a focused library of interpretable forecasting baselines for the VIEWS
conflict prediction pipeline. It is maintained by a small team of researchers and developers.
The codebase is intentionally small (seven model classes, one manager) and changes
to it have downstream consequences in the pipeline ecosystem.

This protocol describes how human contributors are expected to work in this project.

---

## Before You Start

1. **Read the ADRs.** The nine constitutional ADRs in `docs/ADRs/` record why the code is
   shaped as it is. Start with ADR-000 (overview), ADR-001 (ontology), ADR-002 (topology),
   and ADR-003 (declarations over inference). Do not skip this step — decisions that seem
   surprising in the code are usually explained by an ADR.

2. **Read the relevant CIC.** If you are modifying a specific class, read its Class Intent
   Contract in `docs/CICs/` before reading the implementation. CICs describe the class's
   intent, invariants, and known limitations in a form that is faster to understand than the
   code alone.

3. **Understand the stability classification.** ADR-004 classifies each component as Stable
   or Evolving. Changes to Stable components (`ZeroModel`, `LocfModel`, `AverageModel`,
   both protocols, `BaselineModelCatalog`, `build_prediction_grid`) require a migration path.
   Check the stability map before proposing a breaking change.

---

## Working in the Codebase

### Dependency rules

The import topology is defined in ADR-002 and must not be violated:

```
frames/, distributions/, grid.py, spatial.py  →  models/{point,distributional}/  →  catalog.py  →  manager/
                                                                                     protocol.py  ↗
```

The model classes (one-per-file under `models/point/` and `models/distributional/`) depend on
the shared support modules — `frames/` (output seam + pooling), `distributions/`, `grid.py`,
and `spatial.py` — and the catalog imports the model sub-packages. The `model/` layer must not
import from `manager/`. `PredictionFrame` (the `views_frames` leaf, re-exported by
views-pipeline-core ≥3.0.0) is lazy-imported inside the single `to_prediction_frames` seam in
`frames/output.py` (ADR-020) — not at the top of any module.

### Declarations, not inference

ADR-003 is the most important principle for day-to-day coding. Never infer:
- Model type from output shape (use `isinstance(model, DistributionalBaselineModel)`)
- Target columns by scanning DataFrame dtypes
- Train/test boundary from `df.index.max()`
- Required config keys by inspecting constructor signatures

Declare explicitly and validate against the declaration.

### Adding a new model

Adding a model class is a coordinated change. All of the following must be updated in the
same PR:

1. `views_baseline/model/models/point/` or `views_baseline/model/models/distributional/` — implement the class in its own one-class-per-file module and export it from the sub-package `__init__`
2. `views_baseline/model/catalog.py` — add to `MODEL_GENOMES` and write a factory method
3. `docs/CICs/` — write a new Class Intent Contract
4. `docs/ADRs/001_ontology_of_the_repository.md` — add to the ontology (or write ADR-012)
5. `docs/ADRs/004_rules_for_evaluation_and_stability.md` — add to the stability map
6. Tests — Green team tests covering `fit()` and `predict()` contracts are mandatory

If the new model is distributional, also add it to `docs/ADRs/002_topology_and_dependency_rules.md`
(lazy import) and `docs/ADRs/003_authority_of_declarations_over_inference.md` (protocol dispatch).

---

## Testing

The test suite must pass before any PR is merged. Run it locally before pushing:

```bash
ruff check .
pytest
```

GitHub Actions CI runs the same two commands on every push and pull request. A red CI build
blocks the merge.

The three-team testing model (from ADR-005) defines what must be tested:

- **Green team:** Every model class needs tests for `fit()` and `predict()` correctness under
  normal inputs. This is the CI backbone. All existing tests are Green or Beige.
- **Beige team:** Integration tests that cross module boundaries with realistic mocks. The
  six `test_baseline_manager.py` tests are Beige.
- **Red team:** Adversarial inputs (degenerate parameters, missing columns, wrong index
  structure). Currently absent — adding Red team tests is valued work.

Do not write tautological tests. A test that asserts `model.predict(df) == model.predict(df)`
is not a test; it is a tautology. Use an independently computed expected value or a known
invariant (e.g., all-zeros for `ZeroModel`, LOCF value matches a direct `groupby().last()`
call).

---

## Code Review

Changes are submitted as GitHub Pull Requests. Every PR requires at least one human review
before merging.

During code review, check:

1. **ADR consistency.** Does the change conform to the existing ADRs? If not, does it come
   with a new ADR or a supersession?
2. **CIC consistency.** If a class's behaviour changes, is the CIC updated?
3. **Stability obligations.** If a Stable component's interface changes, is there a migration
   path?
4. **Test coverage.** Does the change introduce new logic without new tests?
5. **Import topology.** Does the change violate any of the forbidden import directions in
   ADR-002?
6. **Entity-drop warnings.** If a model can drop entities at predict time, does it emit a
   `logger.warning` with the count?

If you are reviewing AI-assisted code (commits with `Co-Authored-By: Claude`), apply
additional scrutiny per ADR-007: passing lint and tests is necessary but not sufficient for
AI-generated changes.

---

## Commit Messages

Write commit messages that explain *why*, not just *what*. The diff explains what changed;
the commit message explains why the change was made.

Good: `Fix LocfModel to sort by time before groupby, not by position`
Not good: `Update baseline.py`

Reference ADRs in commit messages when a change relates to or is justified by an ADR:
`Per ADR-003: replace isinstance(result, dict) dispatch with protocol check`

If a commit was produced with AI assistance, include the `Co-Authored-By` trailer:
```
Co-Authored-By: Claude <model-identifier> <noreply@anthropic.com>
```

---

## When to Write an ADR

ADR-004 defines five trigger conditions that require an ADR. In addition, write an ADR when:

- You are making a decision that future contributors will wonder about
- You are consciously accepting technical debt (name the debt)
- You are overriding a decision recorded in an existing ADR

ADRs do not need to be long. A focused two-page ADR is better than no ADR and better than an
eight-page document that nobody reads.

---

## Known Limitations to Preserve

These known limitations are documented in the ADRs and CICs. Do not inadvertently fix them
in a way that breaks the documented contract:

- **Entity-drop warnings.** `LocfModel`, `AverageModel`, `ConflictologyModel`, and
  `MixtureBaseline` warn but do not raise when entities are dropped. This is documented as
  a known deviation in ADR-003 and ADR-008. Changing this behaviour (warn → raise) is a
  deliberate decision that requires a PR discussion, not a cleanup.

- **RNG seeding without sequence_number.** Both distributional models re-initialise the RNG
  from `self.seed` on every `predict()` call without incorporating `sequence_number`. This
  is the reproducibility contract (ADR-004). Do not "fix" this without first resolving the
  open question in ADR-004 and writing ADR-011.

- **`PredictionFrame` lazy import.** The deferred `from views_frames import ...` lives in the
  single `to_prediction_frames` seam in `frames/output.py` (ADR-002 as amended by ADR-020). Do not move
  it to module level, and do not add a second `PredictionFrame` construction site.
