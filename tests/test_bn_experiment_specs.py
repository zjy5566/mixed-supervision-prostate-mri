import argparse
import csv
import os
import sys

import pytest

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
SCRIPTS_DIR = os.path.join(PROJECT_DIR, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from run_b_experiments import EXPERIMENTS as B_EXPERIMENTS
from run_n_experiments import EXPERIMENTS as N_EXPERIMENTS
from run_tbx_dice_ablation import EXPERIMENTS as T_EXPERIMENTS
from supervision_experiment_runner import (
    EXPERIMENT_EARLY_STOP_PATIENCE,
    EXPERIMENT_PROTOCOL_TAG,
    EXPERIMENT_TOP_K_CHECKPOINTS,
    EXPERIMENT_USE_EARLY_STOPPING,
    _dry_run_config,
    apply_experiment,
    validate_split_protocol,
)


def _matrix(spec):
    return (spec.use_dense, spec.use_tbx, spec.use_sbx, spec.use_patient)


def test_b_experiment_supervision_matrix_is_explicit():
    assert _matrix(B_EXPERIMENTS["b0"]) == (False, False, False, True)
    assert _matrix(B_EXPERIMENTS["b1"]) == (False, True, False, False)
    assert _matrix(B_EXPERIMENTS["b2"]) == (False, False, True, False)
    assert _matrix(B_EXPERIMENTS["b3"]) == (False, True, True, False)
    assert _matrix(B_EXPERIMENTS["b4"]) == (False, True, True, True)


def test_n_experiment_supervision_matrix_is_explicit():
    assert _matrix(N_EXPERIMENTS["n1"]) == (True, False, False, False)
    assert _matrix(N_EXPERIMENTS["n2"]) == (True, True, False, False)
    assert _matrix(N_EXPERIMENTS["n3"]) == (True, False, True, False)
    assert _matrix(N_EXPERIMENTS["n4"]) == (True, True, True, False)
    assert _matrix(N_EXPERIMENTS["n5"]) == (True, True, True, True)


def test_specs_use_strictly_aligned_parameters_and_checkpoint_selection():
    assert all(
        spec.design_tag == "StrictAbl"
        for spec in (*B_EXPERIMENTS.values(), *N_EXPERIMENTS.values())
    )
    assert all(
        (spec.lr, spec.pos_weight, spec.sys_pos_weight) == (1e-4, 2.0, 1.0)
        for spec in B_EXPERIMENTS.values()
    )
    for key in ("b1", "b2", "b3", "b4"):
        spec = B_EXPERIMENTS[key]
        assert not spec.use_curriculum
        if spec.use_tbx:
            assert (spec.tbx_weight, spec.tbx_start) == (1.0, 1)
        if spec.use_sbx:
            assert (spec.sbx_weight, spec.sbx_start) == (1.0, 1)

    assert all(
        (spec.lr, spec.pos_weight, spec.sys_pos_weight) == (1e-4, 2.0, 1.0)
        for spec in N_EXPERIMENTS.values()
    )
    n1 = N_EXPERIMENTS["n1"]
    assert (n1.dense_weight, n1.dense_start, n1.use_curriculum) == (1.0, 1, False)

    n2 = N_EXPERIMENTS["n2"]
    assert (n2.dense_weight, n2.tbx_weight) == (1.0, 0.05)
    assert n2.use_curriculum
    assert (n2.dense_start, n2.tbx_start) == (1, 15)

    n3 = N_EXPERIMENTS["n3"]
    assert (n3.dense_weight, n3.sbx_weight) == (1.0, 0.25)
    assert n3.use_curriculum
    assert (n3.dense_start, n3.sbx_start) == (1, 15)

    for key in ("n4", "n5"):
        spec = N_EXPERIMENTS[key]
        assert (spec.dense_weight, spec.tbx_weight, spec.sbx_weight) == (
            1.0,
            0.05,
            0.25,
        )
        assert spec.use_curriculum
        assert (spec.dense_start, spec.tbx_start, spec.sbx_start) == (1, 15, 15)

    expected_native = {
        "b0": "patient_auprc",
        "b1": "tbx_native",
        "b2": "region_auprc",
        "b3": "tbx_sbx_native",
        "b4": "tbx_sbx_patient_native",
        **{key: "lesion_dice" for key in N_EXPERIMENTS},
    }
    all_specs = {**B_EXPERIMENTS, **N_EXPERIMENTS}
    assert {
        key: spec.native_best_metric for key, spec in all_specs.items()
    } == expected_native
    assert all(
        spec.common_best_metric == "patient_auprc"
        for spec in all_specs.values()
    )
    assert all(
        spec.patient_pooling == "logit_lme"
        and spec.compare_patient_pooling is False
        for spec in B_EXPERIMENTS.values()
    )
    assert all(
        spec.patient_pooling == "logit_lme"
        and spec.compare_patient_pooling is False
        for spec in N_EXPERIMENTS.values()
    )
    assert B_EXPERIMENTS["b4"].patient_weight == 0.05
    assert N_EXPERIMENTS["n5"].patient_weight == 0.05


def test_tbx_dice_ablation_changes_only_dice_weight():
    assert {
        key: spec.tbx_dice_weight for key, spec in T_EXPERIMENTS.items()
    } == {"t0": 0.0, "t1": 0.25, "t2": 0.5, "t3": 1.0}
    for spec in T_EXPERIMENTS.values():
        assert _matrix(spec) == (False, True, False, False)
        assert spec.design_tag == "BestPrior"
        assert spec.tbx_weight == 1.0
        assert spec.native_best_metric == "tbx_native"
        assert spec.common_best_metric == "common_multilevel"


def test_all_redesigned_experiments_disable_early_stopping():
    assert EXPERIMENT_USE_EARLY_STOPPING is False
    # Retained only as a dormant compatibility value.
    assert EXPERIMENT_EARLY_STOP_PATIENCE == 30


def test_all_redesigned_experiments_retain_native_top_five_checkpoints():
    assert EXPERIMENT_TOP_K_CHECKPOINTS == 5


def _args():
    return argparse.Namespace(
        base_dir=None,
        dataset_root=None,
        exp_dir=None,
        epochs=None,
        seed=None,
        lr=None,
        pos_weight=None,
        sys_pos_weight=None,
        tbx_dice_weight=None,
        dropout_rate=None,
        dry_run=True,
        force=False,
    )


def test_redesigned_tags_separate_joint_plain_lme_protocol():
    for spec in (*B_EXPERIMENTS.values(), *N_EXPERIMENTS.values()):
        config = _dry_run_config(_args())
        apply_experiment(config, spec, _args())
        assert "StrictAbl" in config.EXPERIMENT_TAG
        assert EXPERIMENT_PROTOCOL_TAG in config.EXPERIMENT_TAG
        assert "StartsD" in config.EXPERIMENT_TAG
        if spec.experiment_family == "b":
            assert f"Native{spec.native_best_metric}" in config.EXPERIMENT_TAG


def test_strict_matrix_rejects_optimizer_or_tbx_dice_overrides():
    for field, value in (("lr", 5e-5), ("pos_weight", 1.0), ("tbx_dice_weight", 0.25)):
        args = _args()
        setattr(args, field, value)
        with pytest.raises(ValueError, match="StrictAbl requires"):
            apply_experiment(_dry_run_config(args), B_EXPERIMENTS["b1"], args)


def _write_protocol_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "patient_id",
        "source",
        "has_gland",
        "eligible_tcia_tbx_sbx",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_split_protocol_rejects_stale_or_gland_ineligible_cases(tmp_path):
    joint = {
        "patient_id": "TCIA_joint",
        "source": "TCIA",
        "has_gland": 1,
        "eligible_tcia_tbx_sbx": 1,
    }
    external = {
        "patient_id": "PROMIS_case",
        "source": "PROMIS",
        "has_gland": 1,
        "eligible_tcia_tbx_sbx": 0,
    }
    paths = {
        name: tmp_path / f"{name}.csv"
        for name in ("train", "val", "internal", "external")
    }
    for name in ("train", "val", "internal"):
        _write_protocol_csv(paths[name], [joint])
    _write_protocol_csv(paths["external"], [external])
    config = argparse.Namespace(
        TRAIN_CSV=str(paths["train"]),
        VAL_CSV=str(paths["val"]),
        INTERNAL_TEST_CSV=str(paths["internal"]),
        TEST_CSV=str(paths["external"]),
    )
    validate_split_protocol(config)

    stale = dict(joint, eligible_tcia_tbx_sbx=0)
    _write_protocol_csv(paths["train"], [stale])
    with pytest.raises(ValueError, match="non-joint TCIA"):
        validate_split_protocol(config)

    _write_protocol_csv(paths["train"], [joint])
    no_gland = dict(external, has_gland=0)
    _write_protocol_csv(paths["external"], [no_gland])
    with pytest.raises(ValueError, match="without a valid gland mask"):
        validate_split_protocol(config)
