import re

import pytest

from views_baseline.infrastructure.exceptions import MissingHyperparameterError
from views_baseline.infrastructure.reproducibility_gate import ReproducibilityGate

# -----------------------------------------------------------------------
# Green Team — Structural correctness
# -----------------------------------------------------------------------


def test_core_genome_is_list_of_strings():
    genome = ReproducibilityGate.Config.CORE_GENOME
    assert isinstance(genome, list)
    assert all(isinstance(k, str) for k in genome)
    assert len(genome) > 0


# Every key here is dereferenced unconditionally on a real run, so its absence must be a
# contract violation and not a KeyError deep in a factory. Asserted by name, deliberately:
# a test parametrized over CORE_GENOME cannot catch a key being *removed* from it (the
# parametrize list simply shrinks and the case stops running). Found by the #89 guard
# audit — mutations M5/M6 deleted `regression_targets` and `level` from the genome and the
# entire 277-test suite stayed green.
_KEYS_DEREFERENCED_ON_EVERY_RUN = {
    "steps": "pipeline-core evaluation/forecast horizon",
    "time_steps": "manager._generate_predictions output_length",
    "prediction_format": "pipeline-core stage dispatch",
    "regression_targets": "BaselineModelCatalog.__init__ (all 7 builders)",
    "level": "manager._setup_model_and_data -> loa",
}


def test_core_genome_declares_every_unconditionally_read_key():
    """Removing a key from CORE_GENOME must break a test, not just production (#85)."""
    genome = set(ReproducibilityGate.Config.CORE_GENOME)
    missing = {k: why for k, why in _KEYS_DEREFERENCED_ON_EVERY_RUN.items() if k not in genome}
    assert not missing, (
        f"CORE_GENOME no longer declares {sorted(missing)}. Each is read unconditionally "
        f"on every run ({missing}), so dropping it from the genome restores the failure "
        f"mode of #84: the gate passes and the run dies on a bare KeyError."
    )


@pytest.mark.parametrize("missing_key", sorted(_KEYS_DEREFERENCED_ON_EVERY_RUN))
def test_each_core_key_is_enforced_not_merely_listed(missing_key):
    """A declared core key must actually be audited — listing is not enforcing.

    Asserts only that the absence is rejected *loudly*, not which of `audit_manifest`'s
    four checks fires. The #89 guard audit established that checks 1 and 4 overlap: a
    missing key reaches check 4 as `config.get(k) is None`, so it is rejected either way
    and only the message differs ("Missing core parameters" vs "set to None"). Mutation
    M10 — exempting a key from check 1 — survives for that reason and is an equivalent
    mutant, not a gap.
    """
    config = {
        "algorithm": "ZeroModel",
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
        "regression_targets": ["y1"],
        "level": "pgm",
    }
    del config[missing_key]
    with pytest.raises(MissingHyperparameterError, match=re.escape(missing_key)):
        ReproducibilityGate.Config.audit_manifest(config)


def test_algorithm_genomes_covers_all_catalog_models():
    expected = {
        "ZeroModel", "LocfModel", "AverageModel",
        "ConflictologyModel", "MixtureBaseline",
        "ParametricConflictology", "ParametricHurdleConflictology",
    }
    assert set(ReproducibilityGate.Config.ALGORITHM_GENOMES.keys()) == expected


def test_parametric_genomes_require_family_transform_seed():
    """ADR-022: family/transform/seed are required, audited keys for both parametric models."""
    genomes = ReproducibilityGate.Config.ALGORITHM_GENOMES
    for algo in ("ParametricConflictology", "ParametricHurdleConflictology"):
        assert {"family", "transform", "seed"}.issubset(genomes[algo])


def test_audit_manifest_accepts_valid_zero_model_config():
    config = {
        "algorithm": "ZeroModel",
        "regression_targets": ["y1"],
        "level": "pgm",
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
    }
    ReproducibilityGate.Config.audit_manifest(config)


def test_audit_manifest_accepts_valid_mixture_config():
    config = {
        "algorithm": "MixtureBaseline",
        "regression_targets": ["y1"],
        "level": "pgm",
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
        "window_months": 18,
        "lambda_mix": 0.05,
        "n_samples": 256,
        "seed": 42,
    }
    ReproducibilityGate.Config.audit_manifest(config)


def test_audit_manifest_rejects_missing_seed_for_distributional():
    """ADR-021: `seed` is a required, audited genome key for distributional models."""
    config = {
        "algorithm": "ConflictologyModel",
        "regression_targets": ["y1"],
        "level": "pgm",
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
        "window_months": 18,
        "n_samples": 256,
        # "seed" is missing
    }
    with pytest.raises(MissingHyperparameterError, match="seed"):
        ReproducibilityGate.Config.audit_manifest(config)


def test_audit_manifest_accepts_valid_parametric_config():
    config = {
        "algorithm": "ParametricConflictology",
        "regression_targets": ["y1"],
        "level": "pgm",
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
        "window_months": 18,
        "n_samples": 256,
        "seed": 42,
        "family": "nb",
        "transform": "none",
    }
    ReproducibilityGate.Config.audit_manifest(config)


@pytest.mark.parametrize("missing", ["family", "transform", "seed"])
def test_audit_manifest_rejects_missing_parametric_key(missing):
    """ADR-022: family/transform/seed are audited — omitting any one fails loud."""
    config = {
        "algorithm": "ParametricHurdleConflictology",
        "regression_targets": ["y1"],
        "level": "pgm",
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
        "window_months": 18,
        "n_samples": 256,
        "seed": 42,
        "family": "gumbel",
        "transform": "log1p",
    }
    del config[missing]
    with pytest.raises(MissingHyperparameterError, match=missing):
        ReproducibilityGate.Config.audit_manifest(config)


def test_audit_manifest_rejects_missing_core_key():
    config = {
        "algorithm": "ZeroModel",
        "regression_targets": ["y1"],
        "level": "pgm",
        "time_steps": 36,
        "prediction_format": "prediction_frame",
        # "steps" is missing — and only "steps", so the assertion below can name the
        # exact missing list. This config previously also omitted "prediction_format"
        # while its comment claimed otherwise; the loose match="steps" hid that.
    }
    # Asserted as the exact missing list, not `match="steps"`: that substring also
    # matches "time_steps", and after the #85 CORE_GENOME promotion it would match a
    # message naming keys this test says nothing about. A guard that can pass for a
    # reason other than the one it is named for is not a guard.
    with pytest.raises(MissingHyperparameterError, match=r"Missing core parameters: \['steps'\]"):
        ReproducibilityGate.Config.audit_manifest(config)


def test_audit_manifest_rejects_missing_algorithm_key():
    config = {
        "algorithm": "AverageModel",
        "regression_targets": ["y1"],
        "level": "pgm",
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
        # "window_months" is missing
    }
    with pytest.raises(MissingHyperparameterError, match="window_months"):
        ReproducibilityGate.Config.audit_manifest(config)


def test_audit_manifest_rejects_unknown_algorithm():
    config = {
        "algorithm": "NonExistentModel",
        "regression_targets": ["y1"],
        "level": "pgm",
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
    }
    with pytest.raises(MissingHyperparameterError, match="NonExistentModel"):
        ReproducibilityGate.Config.audit_manifest(config)


# -----------------------------------------------------------------------
# Beige Team — Cross-module integration
# -----------------------------------------------------------------------


def test_gate_genomes_match_catalog_genomes():
    """The catalog's MODEL_GENOMES must be the same object as the gate's ALGORITHM_GENOMES."""
    from views_baseline.model.catalog import BaselineModelCatalog

    assert BaselineModelCatalog.MODEL_GENOMES is ReproducibilityGate.Config.ALGORITHM_GENOMES


def test_manager_gate_rejects_incomplete_config(
    monkeypatch, manager_df, manager_partition_dict
):
    """End-to-end: the manager rejects a config missing core keys."""
    from conftest import make_manager

    import views_baseline.manager.baseline_manager as bm

    config = {
        "run_type": "eval",
        "level": "pg_id",
        "algorithm": "ZeroModel",
        "regression_targets": ["y1"],
        "prediction_format": "prediction_frame",
        # "steps" and "time_steps" are missing
    }

    mgr = make_manager(config, manager_partition_dict)
    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)

    # Exact missing list, for the same reason as above: "steps" alone would also be
    # satisfied by a message naming only "time_steps".
    with pytest.raises(
        MissingHyperparameterError,
        match=r"Missing core parameters: \['steps', 'time_steps'\]",
    ):
        mgr._setup_model_and_data()


def test_downstream_import_contract():
    """The gate is importable and exposes the expected interface."""
    # Deliberately re-import to verify the public import path works.
    from views_baseline.infrastructure.reproducibility_gate import ReproducibilityGate

    assert hasattr(ReproducibilityGate, "Config")
    assert hasattr(ReproducibilityGate.Config, "CORE_GENOME")
    assert hasattr(ReproducibilityGate.Config, "ALGORITHM_GENOMES")
    assert hasattr(ReproducibilityGate.Config, "audit_manifest")
    assert callable(ReproducibilityGate.Config.audit_manifest)


# -----------------------------------------------------------------------
# Red Team — Adversarial inputs
# -----------------------------------------------------------------------


def test_none_value_injection():
    """A required key present but set to None must be rejected."""
    config = {
        "algorithm": "AverageModel",
        "regression_targets": ["y1"],
        "level": "pgm",
        "steps": [*range(1, 37)],
        "time_steps": None,
        "prediction_format": "prediction_frame",
        "window_months": 6,
    }
    with pytest.raises(MissingHyperparameterError, match="None"):
        ReproducibilityGate.Config.audit_manifest(config)


def test_empty_string_algorithm():
    """An empty-string algorithm must be rejected as unknown."""
    config = {
        "algorithm": "",
        "regression_targets": ["y1"],
        "level": "pgm",
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
    }
    with pytest.raises(MissingHyperparameterError, match="Unknown algorithm"):
        ReproducibilityGate.Config.audit_manifest(config)


def test_missing_algorithm_key():
    """Config with no 'algorithm' key at all must be rejected explicitly."""
    config = {
        "regression_targets": ["y1"],
        "level": "pgm",
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
    }
    with pytest.raises(MissingHyperparameterError, match="algorithm"):
        ReproducibilityGate.Config.audit_manifest(config)


def test_extra_keys_ignored():
    """Surplus keys in the config must not cause errors."""
    config = {
        "algorithm": "ZeroModel",
        "regression_targets": ["y1"],
        "level": "pgm",
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
        "totally_unknown_key": "should be fine",
        "another_extra": 999,
    }
    ReproducibilityGate.Config.audit_manifest(config)
