"""Conformance guard (issue #85 / #84): every catalog builder must construct from a
**realistic merged views-models config**, not from a hand-rolled test fixture.

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
views-baseline, pushing the check to our side. `reproducibility_gate` already states the
intended direction — downstream repos import *us* to validate their configs — so the
matching obligation is that we validate against *their* config shape.

**Fixture provenance.** The configs below are copied from the shipped views-models
baselines, merged the way ``ConfigurationManager.get_combined_config`` merges them
(``config_hyperparameters`` then ``config_meta``, meta winning on collision), as of
**2026-09-09**:

* ``zero_pgmbaseline`` / ``locf_pgmbaseline`` / ``average_pgmbaseline`` — the point models
* ``light_strider`` (ConflictologyModel), ``black_ranger`` (MixtureBaseline)
* ``doctorish_dwarf`` (ParametricConflictology), ``bashful_dwarf`` (ParametricHurdle)

They are **copied, not imported**: nothing in this repo may read across the views-models
boundary. The cost of the copy is that it can drift from the live configs; refresh it when
views-models changes a baseline config's shape. It catches a *key rename* — the failure
that actually happened — not a *value* change.

**What this does not cover.** It constructs models; it does not fit or predict them, and
it does not verify that pipeline-core would accept these dicts (that would need the real
``ConfigurationManager``). Those belong to the runtime smoke test in views-models.
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

# Core keys every shipped baseline config carries, verified across all 29 on 2026-09-09.
#
# This block is sized to pipeline-core's `CoreConfigSniffer`, not to CORE_GENOME. An
# earlier version carried only the five genome keys, which made the module docstring's
# claim ("configs pipeline-core would accept") false: the sniffer runs as the first
# statement of `ModelManager.execute_single_run` and rejects a config missing `name`,
# `creator` or `rolling_origin_stride`, plus a regression metric key. A fixture trimmed
# to our own genome is a hand-roll, and the whole point of this file is not to build the
# guard on a hand-roll (#94 review, finding 7).
_CORE = {
    # CORE_GENOME
    "steps": [*range(1, 37)],
    "time_steps": 36,
    "prediction_format": "prediction_frame",
    "level": "pgm",
    # additionally required by CoreConfigSniffer.MANDATORY_KEYS_UNIVERSAL/_MODEL
    "name": "conformance_fixture",
    "creator": "views-baseline conformance suite",
    "rolling_origin_stride": 1,
    # from config_deployment.py. Kept on the LEGACY key rather than ADR-057's `maturity`
    # because all 29 shipped baseline configs are still on it — the fixture is supposed to
    # mirror what really ships, not what pipeline-core would prefer. It warns, and the
    # sniffer reads it as maturity='candidate'. Revisit when views-models migrates.
    "deployment_status": "shadow",
    # required by the sniffer whenever prediction_format == "prediction_frame"
    "skip_predictions_delivery": True,
}

MERGED_CONFIGS = {
    # zero_pgmbaseline / locf_pgmbaseline — no algorithm-specific keys
    "ZeroModel": {
        **_CORE, "algorithm": "ZeroModel",
        "regression_targets": ["lr_ged_sb"], "regression_point_metrics": ["MSE"],
    },
    "LocfModel": {
        **_CORE, "algorithm": "LocfModel",
        "regression_targets": ["lr_ged_sb"], "regression_point_metrics": ["MSE"],
    },
    # average_pgmbaseline
    "AverageModel": {
        **_CORE,
        "algorithm": "AverageModel",
        "regression_targets": ["lr_ged_sb"],
        "regression_point_metrics": ["MSE"],
        "window_months": 18,
    },
    # light_strider
    "ConflictologyModel": {
        **_CORE,
        "algorithm": "ConflictologyModel",
        "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        "regression_sample_metrics": ["twCRPS"],
        "window_months": 36,
        "n_samples": 64,
        "seed": 42,
    },
    # black_ranger
    "MixtureBaseline": {
        **_CORE,
        "algorithm": "MixtureBaseline",
        "regression_targets": ["lr_os_best"],
        "regression_sample_metrics": ["twCRPS"],
        "window_months": 18,
        "lambda_mix": 0.05,
        "n_samples": 256,
        "seed": 42,
    },
    # doctorish_dwarf — native-zero family, no transform
    "ParametricConflictology": {
        **_CORE,
        "algorithm": "ParametricConflictology",
        "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        "regression_sample_metrics": ["twCRPS"],
        "window_months": 36,
        "n_samples": 64,
        "seed": 42,
        "family": "nb",
        "transform": "none",
    },
    # bashful_dwarf — continuous positive-part family with log1p
    "ParametricHurdleConflictology": {
        **_CORE,
        "algorithm": "ParametricHurdleConflictology",
        "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        "regression_sample_metrics": ["twCRPS"],
        "window_months": 36,
        "n_samples": 64,
        "seed": 42,
        "family": "gamma",
        "transform": "log1p",
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

    `test_fixture_carries_no_retired_evaluation_key` compares one module literal against
    another and so cannot fail on any change to this package — it is fixture hygiene, not
    evidence. This test is the one that can fail: it hands each fixture to
    `CoreConfigSniffer`, which pipeline-core runs as the first statement of
    `ModelManager.execute_single_run`. If a fixture drifts into a shape no real run could
    produce, the conformance guard above is being built on a hand-roll — which is the
    precise mistake that let #84 survive.

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
