"""Conformance guard (issue #85 / #84): every catalog builder must construct from a
**real merged views-models config**, not from a hand-rolled test fixture.

This is the test that did not exist on 2026-08-02, and whose absence let every baseline
model stay dead for five weeks.

**The mechanism it guards.** views-pipeline-core retired the synthesised ``targets`` config
key (``507ae11``, refs #380/#381, shipped in 3.0.0) and migrated its own 13 read sites.
views-baseline was never swept, so ``catalog.py`` kept reading ``config["targets"]`` and
every ``get_model()`` raised ``KeyError: 'targets'``. The suite stayed green throughout
because ``conftest.MANAGER_BASE_CONFIG`` manufactured the retired key — the fixture, not
the dependency, defined the contract.

**Why the check belongs here and not upstream.** pipeline-core has registered the blind
spot itself (its C-289: a repo "cannot see a **consumer**, because being imported by
someone leaves no trace in your own source") and its conformance suite explicitly exempts
views-baseline, pushing the check to our side.

**Fixture provenance — read this before editing a fixture.** ``MERGED_CONFIGS`` holds the
seven real merged configs from views-models at commit ``7743011d`` (2026-09-17), produced
by the command below and pasted in verbatim. They are **copied, never imported**: nothing
in this repo may read across the views-models boundary at test time.

The 2026-09-09 version of this file claimed the same provenance and was not a copy. It was
typed from a summary, and every one of the seven drifted: missing ``evaluation_mode``,
``evaluation_profile``, ``n_posterior_samples`` and ``regression_point_baselines``;
metric lists truncated to one entry (three of them naming a metric the real config does
not carry); ``deployment_status`` hard-coded to ``shadow`` for models that ship
``baseline`` and ``deprecated``. The cost was concrete: on 2026-09-17 pipeline-core's
sniffer rejected all three point baselines (``evaluation_mode='point' requires
aggregate_method``, views-models#477) while
``test_fixture_is_accepted_by_pipeline_cores_own_sniffer`` **passed** for the same three —
because the hand-roll omitted ``evaluation_mode``, so the check never fired. A guard that
passes on a fixture the real system rejects is worse than no guard.

Refresh (from the repo root, with views-models checked out as a sibling):

.. code-block:: console

    conda run -n views_pipeline python -c "
    import importlib.util, pathlib, pprint
    root = pathlib.Path('../views-models/models')
    def load(p, fn):
        s = importlib.util.spec_from_file_location('m', p); m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m); return getattr(m, fn)()
    for model in ['zero_pgmbaseline','locf_pgmbaseline','average_pgmbaseline',
                  'light_strider','black_ranger','doctorish_dwarf','sleepy_dwarf']:
        d = root / model / 'configs'
        cfg = {**load(d/'config_hyperparameters.py','get_hp_config'),
               **load(d/'config_deployment.py','get_deployment_config'),
               **load(d/'config_meta.py','get_meta_config')}
        print(f'# {model}'); pprint.pprint(cfg, width=95, sort_dicts=True)
    "

Two deliberate departures from the byte-faithful copy, and only two:

* ``steps`` is written ``[*range(1, 37)]`` rather than the 36-element literal.
* The three point-model fixtures carry ``aggregate_method: "arithmetic_mean"``, which the
  shipped configs **lack** as of ``7743011d`` — that omission is views-models#477, and
  without the key pipeline-core rejects the config. The fixture's purpose is "a config
  pipeline-core would accept", so it carries the post-#477 shape and says so here. Remove
  this note when #477 lands and the refresh command produces the key itself.

**What this does not cover.** It constructs models; it does not fit or predict them. A
copy can still go *stale* — it cannot go *wrong* the way a hand-roll can, but it will
lag the next views-models change until the command above is re-run. The live cross-repo
check belongs in views-models' ``tests/test_runtime_smoke.py``, which builds this catalog
from the real configs and should also run the sniffer.
"""

import pytest

from views_baseline.infrastructure.reproducibility_gate import ReproducibilityGate
from views_baseline.model.catalog import BaselineModelCatalog

# Keys views-evaluation retired in 0.4.0; pipeline-core's `combined_targets()` RAISES on a
# config carrying any of them. A fixture containing one is not a realistic merged config —
# which is exactly how the old conftest fixture drifted out of contact with production.
_RETIRED_EVALUATION_KEYS = (
    "targets",
    "metrics",
    "regression_uncertainty_metrics",
    "classification_uncertainty_metrics",
)

_PARTITION = {"test": (493, 540)}

# Verbatim from the refresh command in the module docstring (views-models 7743011d).
# Do not "tidy" these — every key that looks irrelevant to views-baseline is what makes
# the sniffer test mean something.
MERGED_CONFIGS = {
    # zero_pgmbaseline
    "ZeroModel": {
        "algorithm": "ZeroModel",
        "creator": "Sonja",
        "deployment_status": "shadow",
        "evaluation_mode": "point",
        "aggregate_method": "arithmetic_mean",  # post-#477 shape; see module docstring
        "level": "pgm",
        "name": "zero_pgmbaseline",
        "prediction_format": "prediction_frame",
        "regression_point_baselines": ["average_cmbaseline", "zero_cmbaseline", "locf_cmbaseline"],
        "regression_point_metrics": ["RMSLE", "MSE", "MSLE", "y_hat_bar"],
        "regression_targets": ["lr_ged_sb"],
        "rolling_origin_stride": 1,
        "skip_predictions_delivery": True,
        "steps": [*range(1, 37)],
        "time_steps": 36,
    },
    # locf_pgmbaseline
    "LocfModel": {
        "algorithm": "LocfModel",
        "creator": "Sonja",
        "deployment_status": "shadow",
        "evaluation_mode": "point",
        "aggregate_method": "arithmetic_mean",  # post-#477 shape; see module docstring
        "level": "pgm",
        "name": "locf_pgmbaseline",
        "prediction_format": "prediction_frame",
        "regression_point_baselines": ["average_cmbaseline", "zero_cmbaseline", "locf_cmbaseline"],
        "regression_point_metrics": ["RMSLE", "MSE", "MSLE", "y_hat_bar"],
        "regression_targets": ["lr_ged_sb"],
        "rolling_origin_stride": 1,
        "skip_predictions_delivery": True,
        "steps": [*range(1, 37)],
        "time_steps": 36,
    },
    # average_pgmbaseline
    "AverageModel": {
        "algorithm": "AverageModel",
        "creator": "Sonja",
        "deployment_status": "shadow",
        "evaluation_mode": "point",
        "aggregate_method": "arithmetic_mean",  # post-#477 shape; see module docstring
        "level": "pgm",
        "name": "average_pgmbaseline",
        "prediction_format": "prediction_frame",
        "regression_point_baselines": ["average_cmbaseline", "zero_cmbaseline", "locf_cmbaseline"],
        "regression_point_metrics": ["RMSLE", "MSE", "MSLE", "y_hat_bar"],
        "regression_targets": ["lr_ged_sb"],
        "rolling_origin_stride": 1,
        "skip_predictions_delivery": True,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "window_months": 18,
    },
    # light_strider
    "ConflictologyModel": {
        "algorithm": "ConflictologyModel",
        "creator": "Simon",
        "deployment_status": "shadow",
        "evaluation_profile": "hydranet_ucdp",
        "level": "pgm",
        "n_posterior_samples": 64,
        "n_samples": 64,
        "name": "light_strider",
        "prediction_format": "prediction_frame",
        "regression_sample_metrics": ["CRPS", "QS_sample", "MCR_sample", "Brier_rgs_sample"],
        "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        "rolling_origin_stride": 1,
        "seed": 42,
        "skip_predictions_delivery": True,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "window_months": 36,
    },
    # black_ranger
    "MixtureBaseline": {
        "algorithm": "MixtureBaseline",
        "creator": "Simon",
        "deployment_status": "shadow",
        "lambda_mix": 0.05,
        "level": "pgm",
        "n_posterior_samples": 256,
        "n_samples": 256,
        "name": "black_ranger",
        "prediction_format": "prediction_frame",
        "regression_sample_metrics": ["twCRPS", "QIS", "MIS", "MCR_sample"],
        "regression_targets": ["lr_os_best"],
        "rolling_origin_stride": 1,
        "seed": 42,
        "skip_predictions_delivery": True,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "window_months": 18,
    },
    # doctorish_dwarf
    "ParametricConflictology": {
        "algorithm": "ParametricConflictology",
        "creator": "Simon",
        "deployment_status": "baseline",
        "evaluation_profile": "hydranet_ucdp",
        "family": "nb",
        "level": "pgm",
        "n_posterior_samples": 64,
        "n_samples": 64,
        "name": "doctorish_dwarf",
        "prediction_format": "prediction_frame",
        "regression_sample_metrics": ["CRPS", "QS_sample", "MCR_sample", "Brier_rgs_sample"],
        "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        "rolling_origin_stride": 1,
        "seed": 42,
        "skip_predictions_delivery": True,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "transform": "none",
        "window_months": 36,
    },
    # sleepy_dwarf — NOT bashful_dwarf, which the 2026-09-09 version cited: bashful is
    # `deployment_status: deprecated`, and pipeline-core refuses to run a deprecated model.
    # The old fixture hard-coded `shadow` over the top and so never noticed it had chosen a
    # source that cannot run. None of the three runnable hurdle dwarves uses `log1p`.
    "ParametricHurdleConflictology": {
        "algorithm": "ParametricHurdleConflictology",
        "creator": "Simon",
        "deployment_status": "baseline",
        "evaluation_profile": "hydranet_ucdp",
        "family": "gamma",
        "level": "pgm",
        "n_posterior_samples": 64,
        "n_samples": 64,
        "name": "sleepy_dwarf",
        "prediction_format": "prediction_frame",
        "regression_sample_metrics": ["CRPS", "QS_sample", "MCR_sample", "Brier_rgs_sample"],
        "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        "rolling_origin_stride": 1,
        "seed": 42,
        "skip_predictions_delivery": True,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "transform": "none",
        "window_months": 36,
    },
}


def test_every_catalogued_algorithm_has_a_conformance_fixture():
    """A new algorithm must arrive with a real config, or this guard silently shrinks."""
    catalogued = set(
        BaselineModelCatalog(
            config=MERGED_CONFIGS["ZeroModel"], partition_dict=_PARTITION, loa="pgm"
        ).list_models()
    )
    assert catalogued == set(MERGED_CONFIGS), (
        f"catalog and conformance fixtures disagree: "
        f"only in catalog={sorted(catalogued - set(MERGED_CONFIGS))}, "
        f"only in fixtures={sorted(set(MERGED_CONFIGS) - catalogued)}. "
        f"Add the new algorithm's real merged config here (ADR-012 step)."
    )


@pytest.mark.parametrize("algorithm", sorted(MERGED_CONFIGS))
def test_fixture_carries_no_retired_evaluation_key(algorithm):
    """The fixtures must be configs pipeline-core would accept (#380).

    `combined_targets()` raises on any retired key, so a fixture carrying one could never
    reach the catalog in production — and a guard built on it proves nothing. This is the
    property the old `conftest.MANAGER_BASE_CONFIG` violated.
    """
    present = [k for k in _RETIRED_EVALUATION_KEYS if k in MERGED_CONFIGS[algorithm]]
    assert not present, (
        f"{algorithm}: conformance fixture carries retired evaluation key(s) {present}. "
        f"pipeline-core's combined_targets() raises on these, so this is not a config any "
        f"real run could produce."
    )


def test_shared_test_fixtures_carry_no_retired_evaluation_key():
    """The incident fixture itself must stay clean (#94 review, finding 9).

    `MANAGER_BASE_CONFIG` in conftest is the dict that manufactured `targets` for five
    weeks. This PR deletes the line; nothing stopped it coming back. Re-adding a retired
    key there to make some legacy manager test pass — the same one-line convenience that
    created #84 — would otherwise leave every guard in this repo green.
    """
    from conftest import MANAGER_BASE_CONFIG

    present = [k for k in _RETIRED_EVALUATION_KEYS if k in MANAGER_BASE_CONFIG]
    assert not present, (
        f"conftest.MANAGER_BASE_CONFIG carries retired evaluation key(s) {present}. "
        f"pipeline-core's combined_targets() raises on these — this is the exact fixture "
        f"drift that caused #84."
    )


@pytest.mark.parametrize("algorithm", sorted(MERGED_CONFIGS))
def test_real_config_passes_the_reproducibility_gate(algorithm):
    """A shipped config must satisfy CORE_GENOME + its ALGORITHM_GENOME (#87)."""
    ReproducibilityGate.Config.audit_manifest(MERGED_CONFIGS[algorithm])


@pytest.mark.parametrize("algorithm", sorted(MERGED_CONFIGS))
def test_real_config_constructs_its_model(algorithm):
    """The regression this file exists for: `KeyError: 'targets'` at catalog.py:75.

    Reverting the `regression_targets` migration in `catalog.py` must fail here.
    """
    config = MERGED_CONFIGS[algorithm]
    catalog = BaselineModelCatalog(
        config=config, partition_dict=_PARTITION, loa=config["level"]
    )
    model = catalog.get_model(algorithm)

    assert type(model).__name__ == algorithm
    assert model.targets == config["regression_targets"], (
        f"{algorithm}: model targets {model.targets} do not match the config's "
        f"regression_targets {config['regression_targets']}."
    )
    assert model.loa == config["level"]


@pytest.mark.parametrize("algorithm", sorted(MERGED_CONFIGS))
def test_catalog_does_not_alias_the_config_target_list(algorithm):
    """The model's target list must not be the config's list object.

    A shared list would let a model mutate the config that other models are still being
    built from — the kind of implicit coupling that is invisible until two models disagree.
    """
    config = MERGED_CONFIGS[algorithm]
    catalog = BaselineModelCatalog(
        config=config, partition_dict=_PARTITION, loa=config["level"]
    )
    model = catalog.get_model(algorithm)
    assert model.targets is not config["regression_targets"]


@pytest.mark.parametrize("algorithm", sorted(MERGED_CONFIGS))
def test_fixture_is_accepted_by_pipeline_cores_own_sniffer(algorithm):
    """The fixtures must be configs pipeline-core would actually run (#94 review, #7).

    Hands each fixture to `CoreConfigSniffer`, which pipeline-core runs as the first
    statement of `ModelManager.execute_single_run`.

    **What this can and cannot catch.** It catches a fixture that pipeline-core would
    reject. It cannot catch a fixture that pipeline-core would accept but that differs from
    what ships — and on 2026-09-17 that was the failure: the point fixtures omitted
    `evaluation_mode`, so the sniffer's `evaluation_mode='point'` check never fired, this
    test passed, and the real configs failed in production on exactly that check. A
    permissive validator plus an incomplete fixture is a green test that proves nothing.
    The remedy is upstream of this test — the fixtures must be *complete* copies, produced
    by the refresh command in the module docstring, not typed from a summary.

    Verified red-first on the fix: with the real `evaluation_mode` in and `aggregate_method`
    out, this fails for exactly the three point models with exactly the production message.

    Skipped where pipeline-core is absent; it is the only test here that needs it.
    """
    sniffer = pytest.importorskip(
        "views_pipeline_core.modules.validation.core_config_sniffer"
    )
    # The sniffer takes the OUTER partition dict, keyed by run_type, whose values are
    # {"train": (a, b), "test": (c, d)}. `_PARTITION` above is the inner one — the shape
    # views-baseline actually receives (`self._data_loader.partition_dict`), which is why
    # models read `partition_dict["test"]` directly. Test span must be
    # time_steps + MAX_SHIFT_COUNT = 36 + 12 = 48 months.
    sniffer.CoreConfigSniffer(
        MERGED_CONFIGS[algorithm],
        {"calibration": {"train": (121, 396), "test": (397, 444)}},
        target="model",
    ).sniff_all("calibration")
