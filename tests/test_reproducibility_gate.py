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
        "targets": ["y1"],
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
    }
    ReproducibilityGate.Config.audit_manifest(config)


def test_audit_manifest_accepts_valid_mixture_config():
    config = {
        "algorithm": "MixtureBaseline",
        "targets": ["y1"],
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
        "targets": ["y1"],
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
        "targets": ["y1"],
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
        "targets": ["y1"],
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
        "targets": ["y1"],
        "time_steps": 36,
        # "steps" is missing
    }
    with pytest.raises(MissingHyperparameterError, match="steps"):
        ReproducibilityGate.Config.audit_manifest(config)


def test_audit_manifest_rejects_missing_algorithm_key():
    config = {
        "algorithm": "AverageModel",
        "targets": ["y1"],
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
        "targets": ["y1"],
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
        "targets": ["y1"],
        "prediction_format": "prediction_frame",
        # "steps" and "time_steps" are missing
    }

    mgr = make_manager(config, manager_partition_dict)
    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)

    with pytest.raises(MissingHyperparameterError, match="steps"):
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
        "targets": ["y1"],
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
        "targets": ["y1"],
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
    }
    with pytest.raises(MissingHyperparameterError, match="Unknown algorithm"):
        ReproducibilityGate.Config.audit_manifest(config)


def test_missing_algorithm_key():
    """Config with no 'algorithm' key at all must be rejected explicitly."""
    config = {
        "targets": ["y1"],
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
        "targets": ["y1"],
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
        "totally_unknown_key": "should be fine",
        "another_extra": 999,
    }
    ReproducibilityGate.Config.audit_manifest(config)
