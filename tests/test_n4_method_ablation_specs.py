import argparse
import math
import os
import sys

import torch


PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
SCRIPTS_DIR = os.path.join(PROJECT_DIR, "scripts")
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, SCRIPTS_DIR)

from config import EXPERIMENT_NAME_MAX_BYTES, limit_experiment_name_component  # noqa: E402
from model import ProstateSegMILNet  # noqa: E402
from run_n4_method_ablation import (  # noqa: E402
    ABLATION_COMPARISONS,
    EXPERIMENTS,
)
from supervision_experiment_runner import (  # noqa: E402
    _dry_run_config,
    apply_experiment,
)
from utils import LesionMILEvaluator  # noqa: E402


def _scientific_fields(spec):
    return {
        "use_dense": spec.use_dense,
        "use_tbx": spec.use_tbx,
        "use_sbx": spec.use_sbx,
        "use_patient": spec.use_patient,
        "dense_weight": spec.dense_weight,
        "tbx_weight": spec.tbx_weight,
        "sbx_weight": spec.sbx_weight,
        "patient_weight": spec.patient_weight,
        "use_curriculum": spec.use_curriculum,
        "dense_start": spec.dense_start,
        "tbx_start": spec.tbx_start,
        "sbx_start": spec.sbx_start,
        "patient_start": spec.patient_start,
        "mil_pooling": spec.mil_pooling,
        "region_pooling": spec.region_pooling,
    }


def _diff(left, right):
    left = _scientific_fields(left)
    right = _scientific_fields(right)
    return {key for key in left if left[key] != right[key]}


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


def test_ablation_manifest_reuses_one_reference_for_three_comparisons():
    assert set(EXPERIMENTS) == {
        "n4abl_ref",
        "n4abl_sbx_mean",
        "n4abl_sbx_max",
        "n4abl_patient",
        "n4abl_all_e1",
    }
    assert ABLATION_COMPARISONS == {
        "sbx_pooling": (
            "n4abl_sbx_mean",
            "n4abl_sbx_max",
            "n4abl_ref",
        ),
        "patient_supervision": ("n4abl_ref", "n4abl_patient"),
        "curriculum": ("n4abl_all_e1", "n4abl_ref"),
    }


def test_ablation_variants_change_only_the_intended_scientific_fields():
    ref = EXPERIMENTS["n4abl_ref"]
    assert _diff(ref, EXPERIMENTS["n4abl_sbx_mean"]) == {"mil_pooling"}
    assert _diff(ref, EXPERIMENTS["n4abl_sbx_max"]) == {"mil_pooling"}
    assert _diff(ref, EXPERIMENTS["n4abl_patient"]) == {
        "use_patient",
        "patient_weight",
        "patient_start",
    }
    assert _diff(ref, EXPERIMENTS["n4abl_all_e1"]) == {
        "use_curriculum",
        "tbx_start",
        "sbx_start",
    }


def test_all_ablation_specs_keep_n4_controls_and_lesion_dice_selection():
    for spec in EXPERIMENTS.values():
        assert spec.family == "n"
        assert spec.design_tag == "BestPrior"
        assert spec.train_csv == "N4_mixed_PUB_TCIA_train.csv"
        assert (spec.dense_weight, spec.tbx_weight, spec.sbx_weight) == (
            1.0,
            0.05,
            0.25,
        )
        assert spec.tbx_dice_weight == 0.0
        assert spec.lr == 1e-4
        assert spec.pos_weight == 2.0
        assert spec.sys_pos_weight == 1.0
        assert spec.native_best_metric == "lesion_dice"
        assert spec.common_best_metric == "patient_auprc"
        assert spec.patient_pooling == "logit_lme"
        assert spec.compare_patient_pooling is False
        assert spec.region_pooling == "top_percent"
        assert spec.lme_r == 8.0


def test_dry_run_resolves_unique_pooling_tags_and_formal_selector():
    tags = set()
    for spec in EXPERIMENTS.values():
        args = _args()
        config = _dry_run_config(args)
        apply_experiment(config, spec, args)
        assert config.MIL_POOLING == spec.mil_pooling
        assert config.SEG_REGION_POOLING == "top_percent"
        assert config.NATIVE_BEST_MODEL_METRIC == "lesion_dice"
        assert config.BEST_MODEL_METRIC == "lesion_dice"
        assert config.COMMON_BEST_MODEL_METRIC == "patient_auprc"
        tags.add(config.EXPERIMENT_TAG)
    assert len(tags) == len(EXPERIMENTS)


def test_experiment_directory_names_respect_linux_component_limit():
    short_name = "20260720_0923_N4ABL_REF"
    assert limit_experiment_name_component(short_name) == short_name

    overlong_name = "20260720_0923_" + ("N4_ablation_" * 30)
    shortened = limit_experiment_name_component(overlong_name)
    repeated = limit_experiment_name_component(overlong_name)
    other = limit_experiment_name_component(overlong_name + "other")

    assert len(shortened.encode("utf-8")) <= EXPERIMENT_NAME_MAX_BYTES
    assert shortened == repeated
    assert shortened != other


def test_patient_ablation_tag_leaves_room_for_standard_training_suffix():
    args = _args()
    config = _dry_run_config(args)
    apply_experiment(config, EXPERIMENTS["n4abl_patient"], args)
    representative_name = "_".join(
        [
            "20260720_0923",
            config.EXPERIMENT_TAG,
            "FixedW",
            "LesDense_LesTBxROI_LesSysSBx",
            "Clamp",
            "Curr",
            "EMlrX10",
            "LR0.0001",
            "lesion_dice",
        ]
    )

    assert "AblPatient" in config.EXPERIMENT_TAG
    assert "JTS-PLME" in config.EXPERIMENT_TAG
    assert len(representative_name.encode("utf-8")) <= EXPERIMENT_NAME_MAX_BYTES


def test_model_zone_pooling_variants_operate_in_logit_space():
    model = ProstateSegMILNet(
        in_channels=3,
        max_zones=1,
        base_channels=4,
        mil_pooling="lme",
        lme_r=8.0,
        return_dict=True,
    )
    logits = torch.tensor([[[[[0.0, 1.0, 2.0, 3.0]]]]])
    zones = torch.ones_like(logits)

    mean_value, valid = model.zone_mil_pooling(logits, zones, mode="mean")
    max_value, _ = model.zone_mil_pooling(logits, zones, mode="max")
    lme_value, _ = model.zone_mil_pooling(logits, zones, mode="lme")

    expected_lme = torch.logsumexp(logits.reshape(-1) * 8.0, dim=0) / 8.0
    expected_lme -= math.log(logits.numel()) / 8.0
    torch.testing.assert_close(mean_value[0, 0, 0], logits.mean())
    torch.testing.assert_close(max_value[0, 0, 0], logits.max())
    torch.testing.assert_close(lme_value[0, 0, 0], expected_lme)
    assert valid[0, 0]


def test_direct_sbx_mil_readout_uses_model_region_logits():
    evaluator = LesionMILEvaluator(
        prob_threshold=0.5,
        positive_threshold=3,
        invalid_sys_label=-1,
    )
    lesion_probs = torch.full((2, 1, 1, 1, 2), 0.5)
    region_logits = torch.tensor(
        [
            [[4.0], [-4.0]],
            [[-4.0], [4.0]],
        ]
    )
    batch = {
        "has_sys": torch.tensor([1.0, 1.0]),
        "has_target": torch.tensor([0.0, 0.0]),
        "sys_labels": torch.tensor([[3, 0], [0, 3]]),
    }
    evaluator.update_from_batch(
        lesion_probs,
        batch,
        region_logits=region_logits,
        region_valid_mask=torch.ones((2, 2), dtype=torch.bool),
    )
    metrics = evaluator.compute_metrics()

    assert metrics["region_n"] == 4
    assert metrics["region_auprc"] == 1.0
    assert metrics["region_auc"] == 1.0
