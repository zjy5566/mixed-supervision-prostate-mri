import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from Loss_function import MixedSupervisionLoss
from config import Config
from utils import (
    FROCEvaluator,
    LesionMILEvaluator,
    MetricTracker,
    SegRiskMapEvaluator,
    apply_patient_pooling_calibration,
    binary_entropy_bits_from_logits,
    bootstrap_auc_auprc_ci,
    compute_dice_per_case,
    compute_masked_tbx_dice_per_case,
    compute_topk_dice_per_case,
    flatten_frozen_thresholds,
    has_frozen_validation_thresholds,
    masked_logit_lme_features,
    operating_point_metrics,
    select_balanced_threshold,
)


def test_auc_auprc_bootstrap_ci_is_reproducible_and_cluster_aware():
    y_true = [0, 0, 1, 1, 0, 1, 0, 1]
    y_score = [0.05, 0.40, 0.35, 0.90, 0.20, 0.75, 0.60, 0.80]
    patient_groups = [0, 0, 1, 1, 2, 2, 3, 3]

    first = bootstrap_auc_auprc_ci(
        y_true,
        y_score,
        groups=patient_groups,
        confidence_level=0.95,
        n_resamples=200,
        seed=123,
    )
    second = bootstrap_auc_auprc_ci(
        y_true,
        y_score,
        groups=patient_groups,
        confidence_level=0.95,
        n_resamples=200,
        seed=123,
    )

    assert first == second
    assert first["ci_bootstrap_valid"] == 200
    assert 0.0 <= first["auc_ci_low"] <= first["auc_ci_high"] <= 1.0
    assert 0.0 <= first["auprc_ci_low"] <= first["auprc_ci_high"] <= 1.0


def test_auc_auprc_bootstrap_ci_is_unavailable_for_one_class():
    result = bootstrap_auc_auprc_ci(
        [0, 0, 0],
        [0.1, 0.2, 0.3],
        n_resamples=20,
        seed=1,
    )

    assert result["ci_bootstrap_valid"] == 0
    assert np.isnan(result["auc_ci_low"])
    assert np.isnan(result["auprc_ci_high"])


def test_tbx_pos_neg_bce_uses_sampled_roi_and_ignores_unsampled():
    criterion = MixedSupervisionLoss(
        positive_threshold=3,
        pos_weight_val=1.0,
        use_tbx_positive_only_loss=False,
        use_em_weighting=False,
        fixed_loss_weights={
            "lesion_dense": 0.0,
            "lesion_sparse": 1.0,
            "lesion_sys": 0.0,
        },
        task_switches={
            "lesion_dense": False,
            "lesion_sparse": True,
            "lesion_sys": False,
        },
        return_dict=True,
    )

    lesion_logits = torch.tensor([[[[[0.0, 1.0, -1.0, 2.0]]]]])
    target_mask = torch.tensor([[[[[0.0, 1.0, 3.0, 2.0]]]]])
    batch = {
        "target_mask": target_mask,
        "has_target": torch.tensor([1.0]),
        "has_lesion": torch.tensor([0.0]),
        "has_sys": torch.tensor([0.0]),
    }

    loss_dict = criterion({"lesion_logits": lesion_logits}, batch)

    valid_logits = torch.tensor([1.0, -1.0, 2.0])
    valid_targets = torch.tensor([0.0, 1.0, 0.0])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        valid_logits,
        valid_targets,
    )

    torch.testing.assert_close(loss_dict["loss_lesion_sparse"], expected)
    torch.testing.assert_close(loss_dict["loss_lesion_sparse_bce"], expected)
    torch.testing.assert_close(
        loss_dict["loss_lesion_sparse_dice"],
        torch.tensor(1.0)
        - (2.0 * torch.sigmoid(torch.tensor(-1.0)) + 1e-5)
        / (
            torch.sigmoid(torch.tensor([1.0, -1.0, 2.0])).sum()
            + 1.0
            + 1e-5
        ),
    )
    torch.testing.assert_close(loss_dict["total_loss"], expected)

    counts = loss_dict["loss_counts"]
    assert counts["lesion_sparse_voxels"] == 3
    assert counts["lesion_sparse_positive_voxels"] == 1
    assert counts["lesion_sparse_negative_voxels"] == 2
    torch.testing.assert_close(
        torch.tensor(counts["tbx_pos_prob_mean"]),
        torch.sigmoid(torch.tensor(-1.0)),
    )
    torch.testing.assert_close(
        torch.tensor(counts["tbx_neg_prob_mean"]),
        torch.sigmoid(torch.tensor([1.0, 2.0])).mean(),
    )
    torch.testing.assert_close(
        torch.tensor(counts["tbx_neg_1mp_mean"]),
        (1.0 - torch.sigmoid(torch.tensor([1.0, 2.0]))).mean(),
    )
    torch.testing.assert_close(
        torch.tensor(counts["tbx_pos_bce"]),
        torch.nn.functional.softplus(torch.tensor(1.0)),
    )
    torch.testing.assert_close(
        torch.tensor(counts["tbx_neg_bce"]),
        torch.nn.functional.softplus(torch.tensor([1.0, 2.0])).mean(),
    )

    tracker = MetricTracker()
    tracker.update_losses(loss_dict)
    assert tracker.tbx_pos_prob_mean.count == 1
    assert tracker.tbx_neg_prob_mean.count == 2
    torch.testing.assert_close(
        torch.tensor(tracker.tbx_pos_bce.avg),
        torch.nn.functional.softplus(torch.tensor(1.0)),
    )


def test_tbx_masked_dice_loss_is_weighted_and_ignores_unsampled_voxels():
    criterion = MixedSupervisionLoss(
        positive_threshold=3,
        pos_weight_val=1.0,
        use_tbx_positive_only_loss=False,
        tbx_dice_loss_weight=0.5,
        tbx_dice_smooth=1e-5,
        use_em_weighting=False,
        fixed_loss_weights={"lesion_sparse": 1.0},
        task_switches={
            "lesion_dense": False,
            "lesion_sparse": True,
            "lesion_sys": False,
        },
        return_dict=True,
    )
    logits = torch.tensor([[[[[20.0, -1.0, 0.0, 20.0]]]]])
    target_mask = torch.tensor([[[[[0.0, 1.0, 3.0, 0.0]]]]])
    batch = {
        "target_mask": target_mask,
        "has_target": torch.tensor([1.0]),
        "has_lesion": torch.tensor([0.0]),
        "has_sys": torch.tensor([0.0]),
    }

    loss_dict = criterion({"lesion_logits": logits}, batch)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([-1.0, 0.0]), torch.tensor([0.0, 1.0])
    )
    sampled_probs = torch.sigmoid(torch.tensor([-1.0, 0.0]))
    dice = 1.0 - (2.0 * sampled_probs[1] + 1e-5) / (
        sampled_probs.sum() + 1.0 + 1e-5
    )
    torch.testing.assert_close(loss_dict["loss_lesion_sparse_bce"], bce)
    torch.testing.assert_close(loss_dict["loss_lesion_sparse_dice"], dice)
    torch.testing.assert_close(loss_dict["loss_lesion_sparse"], bce + 0.5 * dice)
    assert loss_dict["loss_counts"]["lesion_sparse_dice_cases"] == 1

    masked_dice = compute_masked_tbx_dice_per_case(
        torch.sigmoid(logits),
        target_mask,
        positive_threshold=3,
        prob_threshold=0.5,
    )
    assert masked_dice.shape == (1,)
    assert abs(masked_dice[0] - 1.0) < 1e-6


def test_tbx_negative_only_case_keeps_bce_and_has_no_dice_term():
    criterion = MixedSupervisionLoss(
        positive_threshold=3,
        pos_weight_val=1.0,
        tbx_dice_loss_weight=1.0,
        use_em_weighting=False,
        fixed_loss_weights={"lesion_sparse": 1.0},
        task_switches={"lesion_sparse": True},
        return_dict=True,
    )
    logits = torch.tensor([[[[[0.0, 1.0]]]]])
    target_mask = torch.tensor([[[[[1.0, 2.0]]]]])
    loss_dict = criterion(
        {"lesion_logits": logits},
        {
            "target_mask": target_mask,
            "has_target": torch.tensor([1.0]),
            "has_lesion": torch.tensor([0.0]),
            "has_sys": torch.tensor([0.0]),
        },
    )
    torch.testing.assert_close(
        loss_dict["loss_lesion_sparse"], loss_dict["loss_lesion_sparse_bce"]
    )
    torch.testing.assert_close(loss_dict["loss_lesion_sparse_dice"], torch.tensor(0.0))
    assert loss_dict["loss_counts"]["lesion_sparse_dice_cases"] == 0


def test_prediction_entropy_tracks_available_supervision_subsets():
    tracker = MetricTracker()
    lesion_logits = torch.zeros((2, 1, 1, 1, 4), dtype=torch.float32)
    outputs = {
        "lesion_logits": lesion_logits,
        "region_logits": torch.zeros((2, 2), dtype=torch.float32),
        "region_valid_mask": torch.tensor(
            [[True, True], [True, False]], dtype=torch.bool
        ),
    }
    batch = {
        "gland_mask": torch.tensor(
            [
                [[[[1.0, 1.0, 0.0, 0.0]]]],
                [[[[1.0, 1.0, 0.0, 0.0]]]],
            ]
        ),
        "has_gland": torch.tensor([1.0, 1.0]),
        "lesion_mask": torch.tensor(
            [
                [[[[1.0, 0.0, 0.0, 0.0]]]],
                [[[[0.0, 0.0, 0.0, 0.0]]]],
            ]
        ),
        "has_lesion": torch.tensor([1.0, 0.0]),
        "target_mask": torch.tensor(
            [
                [[[[0.0, 0.0, 0.0, 0.0]]]],
                [[[[3.0, 1.0, 0.0, 0.0]]]],
            ]
        ),
        "has_target": torch.tensor([0.0, 1.0]),
        "sys_labels": torch.tensor([[3.0, 1.0], [1.0, -1.0]]),
        "has_sys": torch.tensor([1.0, 1.0]),
        "has_cls": torch.tensor([1.0, 1.0]),
        "cls_cspc_label": torch.tensor([1, 0]),
    }

    tracker.update_prediction_entropy(outputs, batch)

    expected_counts = {
        "lesion_all": 8,
        "lesion_gland": 4,
        "lesion_outside_gland": 4,
        "dense_positive": 1,
        "dense_negative_gland": 1,
        "tbx_positive": 1,
        "tbx_negative": 1,
        "region_all": 3,
        "region_positive": 1,
        "region_negative": 2,
        "patient_all": 2,
        "patient_positive": 1,
        "patient_negative": 1,
    }
    for key, count in expected_counts.items():
        meter = tracker.entropy_meters[key]
        assert meter.count == count
        assert abs(meter.avg - 1.0) < 1e-6

    entropy_dict = tracker.get_train_dict()
    assert entropy_dict["train_entropy_lesion_all_n"] == 8
    assert abs(entropy_dict["train_entropy_lesion_all_bits"] - 1.0) < 1e-6
    torch.testing.assert_close(
        binary_entropy_bits_from_logits(torch.tensor([0.0, 20.0, -20.0])),
        torch.tensor([1.0, 0.0, 0.0]),
        atol=1e-6,
        rtol=0.0,
    )


def test_pub_dense_cases_do_not_enter_patient_metrics():
    evaluator = LesionMILEvaluator(
        prob_threshold=0.5,
        positive_threshold=3,
        invalid_sys_label=-1,
    )

    lesion_probs = torch.tensor(
        [
            [[[[0.9]]]],
            [[[[0.8]]]],
        ],
        dtype=torch.float32,
    )
    batch = {
        "has_lesion": torch.tensor([1.0, 0.0]),
        "has_target": torch.tensor([0.0, 1.0]),
        "has_sys": torch.tensor([0.0, 0.0]),
        "lesion_mask": torch.tensor([[[[[1.0]]]], [[[[0.0]]]]]),
        "target_mask": torch.tensor([[[[[0.0]]]], [[[[3.0]]]]]),
    }

    evaluator.update_from_batch(lesion_probs, batch)
    metrics = evaluator.compute_metrics()

    assert metrics["patient_n"] == 1
    assert evaluator.patient_true == [1]
    assert abs(evaluator.patient_score[0] - 0.8) < 1e-6


def test_seg_risk_map_metrics_use_seg_patient_gt_and_sbx_region_gt():
    evaluator = SegRiskMapEvaluator(
        prob_threshold=0.5,
        positive_threshold=1,
        patient_pooling="top_percent",
        top_percent=50.0,
        max_zones=2,
        invalid_sys_label=-1,
    )

    lesion_probs = torch.tensor(
        [
            [[[[0.9, 0.8, 0.1, 0.1]]]],
            [[[[0.7, 0.1, 0.1, 0.1]]]],
        ],
        dtype=torch.float32,
    )
    batch = {
        "has_lesion": torch.tensor([1.0, 1.0]),
        "lesion_mask": torch.tensor(
            [
                [[[[1.0, 0.0, 0.0, 0.0]]]],
                [[[[0.0, 0.0, 0.0, 0.0]]]],
            ],
            dtype=torch.float32,
        ),
        "gland_mask": torch.ones_like(lesion_probs),
        "zones_mask": torch.tensor(
            [
                [[[[1.0, 1.0, 2.0, 2.0]]]],
                [[[[1.0, 1.0, 2.0, 2.0]]]],
            ],
            dtype=torch.float32,
        ),
        "sys_labels": torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    }

    evaluator.update_from_batch(lesion_probs, batch)
    metrics = evaluator.compute_metrics()

    assert evaluator.patient_true == [1, 0]
    assert abs(evaluator.patient_score[0] - 0.85) < 1e-6
    assert metrics["patient_n"] == 2
    assert metrics["patient_sens"] > 0.99
    assert metrics["patient_spec"] > 0.99
    assert metrics["region_n"] == 4
    assert evaluator.region_true == [1, 0, 0, 0]

    sbx_only_evaluator = SegRiskMapEvaluator(
        prob_threshold=0.5,
        positive_threshold=1,
        top_percent=50.0,
        max_zones=2,
        invalid_sys_label=-1,
    )
    sbx_only_batch = {
        "has_lesion": torch.tensor([0.0, 0.0]),
        "has_target": torch.tensor([0.0, 0.0]),
        "zones_mask": batch["zones_mask"],
        "gland_mask": batch["gland_mask"],
        "has_gland": torch.tensor([1.0, 1.0]),
        "sys_labels": batch["sys_labels"],
    }
    sbx_only_evaluator.update_from_batch(lesion_probs, sbx_only_batch)
    assert sbx_only_evaluator.patient_true == [1, 0]
    assert sbx_only_evaluator.region_true == [1, 0, 0, 0]


def test_seg_risk_map_patient_pooling_can_ignore_gland_mask_for_deployment():
    evaluator = SegRiskMapEvaluator(
        prob_threshold=0.5,
        positive_threshold=1,
        patient_pooling="top_percent",
        top_percent=25.0,
        max_zones=1,
        invalid_sys_label=-1,
        use_gland_mask_for_patient_pooling=False,
    )
    lesion_probs = torch.tensor([[[[[0.9, 0.2, 0.1, 0.1]]]]], dtype=torch.float32)
    batch = {
        "has_lesion": torch.tensor([1.0]),
        "has_cls": torch.tensor([1.0]),
        "cls_cspc_label": torch.tensor([1]),
        "lesion_mask": torch.tensor([[[[[1.0, 0.0, 0.0, 0.0]]]]], dtype=torch.float32),
        "gland_mask": torch.tensor([[[[[0.0, 1.0, 0.0, 0.0]]]]], dtype=torch.float32),
    }

    evaluator.update_from_batch(lesion_probs, batch)

    assert evaluator.patient_true == [1]
    assert abs(evaluator.patient_score[0] - 0.9) < 1e-6


def test_seg_risk_map_patient_pooling_uses_gland_mask_by_default():
    evaluator = SegRiskMapEvaluator(
        prob_threshold=0.5,
        positive_threshold=1,
        patient_pooling="max",
        invalid_sys_label=-1,
    )
    lesion_probs = torch.tensor([[[[[0.9, 0.2, 0.1, 0.1]]]]], dtype=torch.float32)
    batch = {
        "has_cls": torch.tensor([1.0]),
        "cls_cspc_label": torch.tensor([1]),
        "gland_mask": torch.tensor([[[[[0.0, 1.0, 0.0, 0.0]]]]], dtype=torch.float32),
        "has_gland": torch.tensor([1.0]),
    }

    evaluator.update_from_batch(lesion_probs, batch)

    assert evaluator.patient_true == [1]
    assert abs(evaluator.patient_score[0] - 0.2) < 1e-6


def test_contrast_logit_lme_is_invariant_to_additive_gland_logit_shift():
    base_logits = torch.tensor([[[-2.0, -1.0, 0.5, 2.0]]])
    shifted_logits = base_logits + 1.75
    gland = torch.ones_like(base_logits, dtype=torch.bool)

    base = masked_logit_lme_features(torch.sigmoid(base_logits), gland, lme_r=8.0)
    shifted = masked_logit_lme_features(
        torch.sigmoid(shifted_logits), gland, lme_r=8.0
    )

    assert base is not None and shifted is not None
    assert abs(shifted["absolute_logit"] - base["absolute_logit"] - 1.75) < 1e-5
    assert abs(shifted["gland_median_logit"] - base["gland_median_logit"] - 1.75) < 1e-5
    assert abs(shifted["contrast_logit"] - base["contrast_logit"]) < 1e-5


def test_validation_fits_contrast_pooling_and_test_reuses_frozen_coefficients():
    lesion_logits = torch.tensor(
        [
            [[[[ -2.0, -2.0, -2.0, -2.0]]]],
            [[[[  1.0,  1.0,  1.0,  1.0]]]],
            [[[[ -2.0, -2.0, -2.0,  2.5]]]],
            [[[[  1.0,  1.0,  1.0,  5.5]]]],
        ],
        dtype=torch.float32,
    )
    lesion_probs = torch.sigmoid(lesion_logits)
    batch = {
        "has_cls": torch.ones(4),
        "cls_cspc_label": torch.tensor([0, 0, 1, 1]),
        "has_gland": torch.ones(4),
        "gland_mask": torch.ones_like(lesion_probs),
    }

    validation = SegRiskMapEvaluator(
        patient_pooling="logit_lme_contrast",
        select_validation_thresholds=True,
    )
    validation.update_from_batch(lesion_probs, batch)
    validation_metrics = validation.compute_metrics()
    calibration = validation.patient_pooling_calibration

    assert calibration["fitted"] == 1
    assert calibration["n"] == 4
    assert calibration["positive_n"] == 2
    assert np.isfinite(calibration["alpha"])
    assert np.isfinite(calibration["beta"])
    assert np.isfinite(calibration["intercept"])
    assert validation_metrics["patient_auprc"] == validation_metrics[
        "patient_contrast_auprc"
    ]
    assert np.isfinite(validation_metrics["patient_logit_lme_auprc"])
    assert len(validation.patient_logit_lme_score) == 4

    frozen_test = SegRiskMapEvaluator(
        patient_pooling="logit_lme_contrast",
        patient_threshold=validation_metrics["patient_decision_threshold"],
        patient_logit_lme_threshold=validation_metrics[
            "patient_logit_lme_decision_threshold"
        ],
        patient_logit_lme_operating_thresholds={
            "decision": validation_metrics[
                "patient_logit_lme_decision_threshold"
            ],
            "at_fixed_specificity": validation_metrics[
                "patient_logit_lme_threshold_at_fixed_spec"
            ],
            "at_fixed_sensitivity": validation_metrics[
                "patient_logit_lme_threshold_at_fixed_sens"
            ],
        },
        patient_pooling_calibration=calibration,
        select_validation_thresholds=False,
    )
    frozen_test.update_from_batch(lesion_probs, batch)
    frozen_test_metrics = frozen_test.compute_metrics()

    np.testing.assert_allclose(
        frozen_test.patient_score,
        validation.patient_score,
        rtol=0.0,
        atol=1e-12,
    )
    first_features = {
        "absolute_logit": frozen_test.patient_absolute_logit[0],
        "contrast_logit": frozen_test.patient_contrast_logit[0],
    }
    expected = apply_patient_pooling_calibration(first_features, calibration)
    assert abs(expected["score"] - frozen_test.patient_score[0]) < 1e-12
    assert frozen_test_metrics["patient_logit_lme_decision_threshold"] == (
        validation_metrics["patient_logit_lme_decision_threshold"]
    )


def test_original_logit_lme_is_the_canonical_patient_auprc_without_contrast():
    lesion_probs = torch.tensor(
        [
            [[[[0.10, 0.15, 0.20, 0.25]]]],
            [[[[0.20, 0.25, 0.30, 0.35]]]],
            [[[[0.45, 0.50, 0.80, 0.85]]]],
            [[[[0.55, 0.60, 0.90, 0.95]]]],
        ],
        dtype=torch.float32,
    )
    batch = {
        "has_cls": torch.ones(4),
        "cls_cspc_label": torch.tensor([0, 0, 1, 1]),
        "has_gland": torch.ones(4),
        "gland_mask": torch.ones_like(lesion_probs),
    }
    evaluator = SegRiskMapEvaluator(
        patient_pooling="logit_lme",
        select_validation_thresholds=True,
    )
    evaluator.update_from_batch(lesion_probs, batch)
    metrics = evaluator.compute_metrics()

    assert metrics["patient_auprc"] == metrics["patient_logit_lme_auprc"]
    assert metrics["patient_auc"] == metrics["patient_logit_lme_auc"]
    assert metrics["patient_contrast_n"] == 0
    assert np.isnan(metrics["patient_contrast_auprc"])
    assert metrics["patient_pooling_calibration_fitted"] == 0


def test_seg_risk_map_patient_pooling_skips_missing_gland_mask():
    evaluator = SegRiskMapEvaluator(
        prob_threshold=0.5,
        positive_threshold=1,
        patient_pooling="max",
        invalid_sys_label=-1,
    )
    evaluator.update_from_batch(
        torch.tensor([[[[[0.9, 0.2]]]]], dtype=torch.float32),
        {
            "has_cls": torch.tensor([1.0]),
            "cls_cspc_label": torch.tensor([1]),
            "gland_mask": torch.zeros((1, 1, 1, 1, 2)),
            "has_gland": torch.tensor([0.0]),
        },
    )

    assert evaluator.patient_true == []
    assert evaluator.patient_score == []


def test_seg_risk_map_excludes_dense_ra_without_patient_pathology_label():
    evaluator = SegRiskMapEvaluator(
        prob_threshold=0.5,
        positive_threshold=3,
        patient_pooling="max",
        invalid_sys_label=-1,
    )
    lesion_probs = torch.tensor([[[[[0.95]]]]], dtype=torch.float32)
    batch = {
        "has_lesion": torch.tensor([1.0]),
        "has_target": torch.tensor([0.0]),
        "has_sys": torch.tensor([0.0]),
        "has_cls": torch.tensor([0.0]),
        "cls_cspc_label": torch.tensor([-1]),
        "lesion_mask": torch.tensor([[[[[1.0]]]]], dtype=torch.float32),
    }

    evaluator.update_from_batch(lesion_probs, batch)

    assert evaluator.patient_true == []
    assert evaluator.patient_score == []


def test_outside_gland_penalty_uses_only_outside_gland_voxels():
    criterion = MixedSupervisionLoss(
        use_em_weighting=False,
        fixed_loss_weights={
            "lesion_dense": 0.0,
            "lesion_sparse": 0.0,
            "lesion_sys": 0.0,
            "lesion_outside_gland": 0.5,
        },
        task_switches={
            "lesion_dense": False,
            "lesion_sparse": False,
            "lesion_sys": False,
            "lesion_outside_gland": True,
        },
        return_dict=True,
    )

    lesion_logits = torch.tensor([[[[[0.0, 1.0, -1.0, 2.0]]]]])
    gland_mask = torch.tensor([[[[[0.0, 1.0, 0.0, 1.0]]]]])
    batch = {
        "gland_mask": gland_mask,
        "has_gland": torch.tensor([1.0]),
        "has_target": torch.tensor([0.0]),
        "has_lesion": torch.tensor([0.0]),
        "has_sys": torch.tensor([0.0]),
    }

    loss_dict = criterion({"lesion_logits": lesion_logits}, batch)
    expected_raw = torch.nn.functional.softplus(torch.tensor([0.0, -1.0])).mean()

    torch.testing.assert_close(loss_dict["loss_lesion_outside_gland"], expected_raw)
    torch.testing.assert_close(loss_dict["total_loss"], expected_raw * 0.5)
    counts = loss_dict["loss_counts"]
    assert counts["lesion_outside_gland_cases"] == 1
    assert counts["lesion_outside_gland_voxels"] == 2
    torch.testing.assert_close(
        torch.tensor(counts["outside_gland_prob_mean"]),
        torch.sigmoid(torch.tensor([0.0, -1.0])).mean(),
    )


def test_patient_risk_loss_uses_gland_mask_by_default():
    old_use_gland = Config.PATIENT_RISK_USE_GLAND_MASK
    Config.PATIENT_RISK_USE_GLAND_MASK = True
    criterion = MixedSupervisionLoss(
        use_em_weighting=False,
        fixed_loss_weights={
            "lesion_dense": 0.0,
            "lesion_sparse": 0.0,
            "lesion_sys": 0.0,
            "lesion_outside_gland": 0.0,
            "lesion_patient": 0.5,
        },
        task_switches={
            "lesion_dense": False,
            "lesion_sparse": False,
            "lesion_sys": False,
            "lesion_outside_gland": False,
            "lesion_patient": True,
        },
        return_dict=True,
    )

    lesion_logits = torch.tensor([[[[[0.0, 2.0, -1.0, 1.0]]]]])
    gland_mask = torch.tensor([[[[[0.0, 1.0, 0.0, 1.0]]]]])
    batch = {
        "gland_mask": gland_mask,
        "has_gland": torch.tensor([1.0]),
        "has_cls": torch.tensor([1.0]),
        "cls_cspc_label": torch.tensor([1]),
        "has_target": torch.tensor([0.0]),
        "has_lesion": torch.tensor([0.0]),
        "has_sys": torch.tensor([0.0]),
    }

    loss_dict = criterion({"lesion_logits": lesion_logits}, batch)
    inside_logits = torch.tensor([2.0, 1.0])
    r = torch.tensor(8.0)
    pooled_logit = torch.logsumexp(inside_logits * r, dim=0) / r - torch.log(
        torch.tensor(float(inside_logits.numel()))
    ) / r
    expected_raw = torch.nn.functional.binary_cross_entropy_with_logits(
        pooled_logit.reshape(1),
        torch.tensor([1.0]),
    )

    torch.testing.assert_close(loss_dict["loss_lesion_patient"], expected_raw)
    torch.testing.assert_close(loss_dict["total_loss"], expected_raw * 0.5)
    counts = loss_dict["loss_counts"]
    assert counts["lesion_patient_cases"] == 1
    assert counts["lesion_patient_positive_cases"] == 1
    assert counts["lesion_patient_negative_cases"] == 0
    torch.testing.assert_close(
        torch.tensor(counts["patient_risk_prob_mean"]),
        torch.sigmoid(pooled_logit),
    )
    Config.PATIENT_RISK_USE_GLAND_MASK = old_use_gland


def test_patient_risk_loss_skips_case_without_valid_gland_mask():
    old_use_gland = Config.PATIENT_RISK_USE_GLAND_MASK
    Config.PATIENT_RISK_USE_GLAND_MASK = True
    criterion = MixedSupervisionLoss(
        use_em_weighting=False,
        fixed_loss_weights={"lesion_patient": 1.0},
        task_switches={
            "lesion_dense": False,
            "lesion_sparse": False,
            "lesion_sys": False,
            "lesion_outside_gland": False,
            "lesion_patient": True,
        },
        return_dict=True,
    )
    loss_dict = criterion(
        {"lesion_logits": torch.tensor([[[[[0.0, 2.0]]]]])},
        {
            "gland_mask": torch.zeros((1, 1, 1, 1, 2)),
            "has_gland": torch.tensor([0.0]),
            "has_cls": torch.tensor([1.0]),
            "cls_cspc_label": torch.tensor([1]),
            "has_target": torch.tensor([0.0]),
            "has_lesion": torch.tensor([0.0]),
            "has_sys": torch.tensor([0.0]),
        },
    )
    torch.testing.assert_close(loss_dict["loss_lesion_patient"], torch.tensor(0.0))
    torch.testing.assert_close(loss_dict["total_loss"], torch.tensor(0.0))
    assert loss_dict["active_tasks"]["lesion_patient"] == 0.0
    assert loss_dict["loss_counts"]["lesion_patient_cases"] == 0
    Config.PATIENT_RISK_USE_GLAND_MASK = old_use_gland


def test_target_cspca_dice_uses_only_positive_target_cases():
    tracker = MetricTracker()
    positive_threshold = 3
    prob_threshold = 0.5

    lesion_probs = torch.tensor(
        [
            [[[[0.9, 0.1, 0.8, 0.1]]]],
            [[[[0.9, 0.1, 0.1, 0.1]]]],
        ],
        dtype=torch.float32,
    )
    batch = {
        "has_target": torch.tensor([1.0, 1.0]),
        "target_mask": torch.tensor(
            [
                [[[[3.0, 0.0, 0.0, 0.0]]]],
                [[[[1.0, 0.0, 0.0, 0.0]]]],
            ],
            dtype=torch.float32,
        ),
    }

    target_cspca = (batch["target_mask"] >= positive_threshold).float()
    positive_target_cases = (batch["has_target"] > 0) & target_cspca.reshape(target_cspca.size(0), -1).any(dim=1)
    pred_bin = (lesion_probs[positive_target_cases] >= prob_threshold).float()
    tracker.update_target_cspca_dice_values(
        compute_dice_per_case(pred_bin, target_cspca[positive_target_cases])
    )

    assert tracker.target_cspca_dice_n == 1
    expected = (2.0 + 1e-5) / (3.0 + 1e-5)
    assert abs(tracker.target_cspca_dice.avg - expected) < 1e-6


def test_tbx_roi_metrics_report_sensitivity_at_fixed_roc_specificity():
    tracker = MetricTracker()

    tracker.update_tbx_roi_samples(
        y_true=[0, 0, 1, 1],
        y_score=[0.1, 0.2, 0.8, 0.9],
    )
    tracker.finalize_tbx_roi_metrics(threshold=0.5, compute_operating_metrics=True)

    assert tracker.tbx_roi_n == 4
    assert tracker.tbx_roi_fixed_spec_target == 0.95
    assert tracker.tbx_roi_actual_fpr_at_fixed_spec <= 0.05 + 1e-8
    assert tracker.tbx_roi_sens_at_fixed_spec == 1.0
    assert tracker.tbx_roi_spec_at_fixed_sens == 1.0
    assert tracker.tbx_roi_auc == 1.0


def test_voxel_operating_metrics_report_fixed_specificity_and_sensitivity():
    tracker = MetricTracker()

    tracker.update_voxel_operating_samples(
        "lesion",
        y_true=[0, 0, 1, 1],
        y_score=[0.1, 0.2, 0.8, 0.9],
    )
    tracker.update_voxel_operating_samples(
        "target_cspca",
        y_true=[0, 0, 1, 1],
        y_score=[0.1, 0.2, 0.8, 0.9],
    )
    tracker.finalize_voxel_operating_metrics("lesion")
    tracker.finalize_voxel_operating_metrics("target_cspca")

    assert tracker.lesion_voxel_n == 4
    assert tracker.lesion_voxel_sens_at_fixed_spec == 1.0
    assert tracker.lesion_voxel_actual_fpr_at_fixed_spec <= 0.05 + 1e-8
    assert tracker.lesion_voxel_spec_at_fixed_sens == 1.0
    assert tracker.target_cspca_voxel_n == 4
    assert tracker.target_cspca_voxel_sens_at_fixed_spec == 1.0
    assert tracker.target_cspca_voxel_spec_at_fixed_sens == 1.0


def test_dense_lesion_metrics_ignore_predictions_outside_gland():
    tracker = MetricTracker()
    probs = torch.tensor(
        [
            [[[[0.9, 0.9, 0.1, 0.9]]]],
            [[[[0.9, 0.9, 0.1, 0.9]]]],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [[[[1.0, 0.0, 0.0, 0.0]]]],
            [[[[1.0, 0.0, 0.0, 0.0]]]],
        ],
        dtype=torch.float32,
    )
    gland = torch.tensor(
        [
            [[[[1.0, 0.0, 1.0, 0.0]]]],
            [[[[0.0, 0.0, 0.0, 0.0]]]],
        ],
        dtype=torch.float32,
    )

    tracker.update_lesion_full_crop_dice_values(
        compute_dice_per_case((probs >= 0.5).float(), target)
    )
    tracker.update_lesion_gland_metrics(
        probs,
        target,
        gland,
        has_gland=torch.tensor([1.0, 0.0]),
        threshold=0.5,
        sweep_thresholds=True,
        compute_operating_metrics=True,
        compute_froc_metrics=True,
    )
    tracker.finalize_voxel_operating_metrics("lesion")
    tracker.finalize_froc_metrics()

    assert tracker.lesion_dice.avg == 1.0
    assert tracker.lesion_full_crop_dice.avg < 1.0
    assert tracker.lesion_gland_dice.avg == 1.0
    assert tracker.lesion_gland_cases == 1
    assert tracker.lesion_gland_missing_cases == 1
    assert tracker.lesion_gland_voxels == 2
    assert tracker.lesion_voxel_n == 2
    assert tracker.lesion_sens.avg == 1.0
    assert tracker.lesion_spec.avg == 1.0
    assert tracker.lesion_dice_sweep_values[0.5] == [1.0]
    metrics = tracker.get_val_dict()
    assert metrics["val_lesion_dice"] == 1.0
    assert metrics["val_lesion_full_crop_dice"] < 1.0
    assert metrics["val_lesion_gland_dice"] == 1.0
    assert metrics["val_lesion_gland_voxel_n"] == 2
    assert metrics["val_lesion_gland_froc_n"] == 1
    assert metrics["val_lesion_gland_froc_protocol"] == "picai_eval"


def test_froc_metrics_report_sensitivity_at_fixed_fp_per_patient():
    evaluator = FROCEvaluator(
        fp_per_patient_targets=(0.5,),
        candidate_threshold=0.5,
        candidate_min_voxels=0,
    )
    probs = torch.tensor(
        [
            [[[[0.90, 0.10, 0.80, 0.10]]]],
            [[[[0.10, 0.70, 0.10, 0.10]]]],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [[[[1.0, 0.0, 0.0, 0.0]]]],
            [[[[0.0, 0.0, 0.0, 0.0]]]],
        ],
        dtype=torch.float32,
    )

    evaluator.update_from_maps(probs, target)
    metrics = evaluator.compute_metrics(prefix="lesion_")

    assert metrics["lesion_froc_n"] == 2
    assert metrics["lesion_froc_num_gt"] == 1
    assert metrics["lesion_sens_at_fp_per_patient_0p5"] == 1.0
    assert metrics["lesion_actual_fp_per_patient_0p5"] == 0.5
    assert np.isclose(metrics["lesion_threshold_at_fp_per_patient_0p5"], 0.8)
    assert evaluator.official_metrics().lesion_TPR_at_FPR(0.5) == 1.0
    assert metrics["lesion_froc_protocol"] == "picai_eval"
    assert metrics["lesion_froc_connectivity"] == 26
    assert metrics["lesion_froc_min_overlap"] == 0.10


def test_picai_froc_uses_one_to_one_matching_for_multi_lesion_overlap():
    probability = np.ones((1, 1, 7), dtype=np.float32)
    target = np.zeros_like(probability)
    target[0, 0, 1] = 1.0
    target[0, 0, 5] = 1.0

    evaluator = FROCEvaluator(
        fp_per_patient_targets=(0.5,),
        candidate_threshold=0.5,
        candidate_min_voxels=0,
    )
    evaluator.update_from_maps(
        torch.from_numpy(probability)[None, ...],
        torch.from_numpy(target)[None, ...],
    )
    metrics = evaluator.compute_metrics(prefix="lesion_")

    assert metrics["lesion_froc_num_gt"] == 2
    assert metrics["lesion_sens_at_fp_per_patient_0p5"] == 0.5
    assert metrics["lesion_actual_fp_per_patient_0p5"] == 0.0


def test_picai_froc_rejects_diffuse_component_below_iou_threshold():
    probability = np.ones((1, 1, 30), dtype=np.float32)
    target = np.zeros_like(probability)
    target[0, 0, 10:12] = 1.0

    evaluator = FROCEvaluator(
        fp_per_patient_targets=(1.0,),
        candidate_threshold=0.5,
        candidate_min_voxels=0,
    )
    evaluator.update_from_maps(
        torch.from_numpy(probability)[None, ...],
        torch.from_numpy(target)[None, ...],
    )
    metrics = evaluator.compute_metrics(prefix="lesion_")

    assert metrics["lesion_froc_num_gt"] == 1
    assert metrics["lesion_sens_at_fp_per_patient_1p0"] == 0.0


def test_picai_froc_prediction_mask_does_not_crop_ground_truth():
    probability = torch.tensor([[[[[0.0, 1.0, 0.0, 0.0]]]]])
    target = torch.tensor([[[[[0.0, 1.0, 1.0, 0.0]]]]])
    gland = torch.tensor([[[[[0.0, 1.0, 0.0, 0.0]]]]])
    evaluator = FROCEvaluator(
        min_overlap=0.75,
        candidate_threshold=0.5,
        candidate_min_voxels=0,
    )

    evaluator.update_from_maps(probability, target, gland)

    case_results = evaluator.lesion_results["0"]
    assert (1, 0.0, 0.0) in case_results
    assert not any(is_lesion and confidence > 0 for is_lesion, confidence, _ in case_results)


def test_picai_froc_discards_duplicate_candidate_with_sufficient_overlap():
    probability = np.zeros((1, 1, 9), dtype=np.float32)
    probability[0, 0, 1] = 1.0
    probability[0, 0, 5] = 0.9
    probability[0, 0, 8] = 0.8
    target = np.zeros_like(probability)
    target[0, 0, 1:6] = 1.0

    evaluator = FROCEvaluator(
        fp_per_patient_targets=(0.0,),
        candidate_threshold=0.5,
        candidate_min_voxels=0,
    )
    evaluator.update_from_maps(
        torch.from_numpy(probability)[None, ...],
        torch.from_numpy(target)[None, ...],
    )
    metrics = evaluator.compute_metrics(prefix="lesion_")

    assert metrics["lesion_froc_num_gt"] == 1
    assert metrics["lesion_sens_at_fp_per_patient_0p0"] == 1.0
    assert metrics["lesion_actual_fp_per_patient_0p0"] == 0.0
    assert (
        metrics[
            "lesion_froc_allow_unmatched_candidates_with_minimal_overlap"
        ]
        == 1
    )


def test_target_cspca_aux_dice_reports_swept_threshold_and_topk_upper_bound():
    tracker = MetricTracker()
    lesion_probs = torch.tensor([[[[[0.90, 0.80, 0.70, 0.10]]]]], dtype=torch.float32)
    target = torch.tensor([[[[[1.0, 1.0, 0.0, 0.0]]]]], dtype=torch.float32)

    tracker.update_target_cspca_aux_dice(lesion_probs, target)
    tracker.finalize_target_cspca_aux_dice()

    assert tracker.target_cspca_best_threshold_dice == 1.0
    assert tracker.target_cspca_best_threshold > 0.5
    assert tracker.target_cspca_topk_dice.avg == 1.0
    assert compute_topk_dice_per_case(lesion_probs, target, mode="target_volume")[0] == 1.0


def test_frozen_operating_points_are_evaluated_without_test_refitting():
    val_true = [0, 0, 1, 1]
    val_score = [0.10, 0.20, 0.80, 0.90]
    selected = operating_point_metrics(
        val_true,
        val_score,
        fixed_specificity=0.95,
        fixed_sensitivity=0.90,
    )

    test_true = [0, 0, 1, 1]
    test_score = [0.70, 0.80, 0.60, 0.90]
    frozen = operating_point_metrics(
        test_true,
        test_score,
        fixed_specificity=0.95,
        fixed_sensitivity=0.90,
        threshold_at_fixed_spec=selected["threshold_at_fixed_spec"],
        threshold_at_fixed_sens=selected["threshold_at_fixed_sens"],
    )
    refit = operating_point_metrics(
        test_true,
        test_score,
        fixed_specificity=0.95,
        fixed_sensitivity=0.90,
    )

    assert frozen["threshold_at_fixed_spec"] == selected["threshold_at_fixed_spec"]
    assert frozen["threshold_at_fixed_sens"] == selected["threshold_at_fixed_sens"]
    assert frozen["threshold_at_fixed_spec"] != refit["threshold_at_fixed_spec"]
    assert frozen["actual_spec_at_fixed_spec"] == 0.5


def test_patient_threshold_selected_on_validation_stays_fixed_on_test():
    val_true = [0, 0, 1, 1]
    val_score = [0.10, 0.20, 0.80, 0.90]
    val_threshold = select_balanced_threshold(val_true, val_score)
    val_operating = operating_point_metrics(val_true, val_score)
    frozen_patient = {
        "decision": val_threshold,
        "at_fixed_specificity": val_operating["threshold_at_fixed_spec"],
        "at_fixed_sensitivity": val_operating["threshold_at_fixed_sens"],
    }

    evaluator = SegRiskMapEvaluator(
        prob_threshold=0.5,
        positive_threshold=1,
        patient_threshold=frozen_patient["decision"],
        patient_operating_thresholds=frozen_patient,
        select_validation_thresholds=False,
    )
    evaluator.patient_true = [0, 0, 1, 1]
    evaluator.patient_score = [0.70, 0.80, 0.60, 0.90]
    metrics = evaluator.compute_metrics()

    assert metrics["patient_decision_threshold"] == val_threshold
    assert metrics["patient_threshold_at_fixed_spec"] == frozen_patient[
        "at_fixed_specificity"
    ]
    assert metrics["patient_threshold_at_fixed_sens"] == frozen_patient[
        "at_fixed_sensitivity"
    ]


def test_metric_tracker_builds_versioned_validation_threshold_bundle(monkeypatch):
    monkeypatch.setattr(
        Config, "SEG_PATIENT_POOLING", "logit_lme_contrast", raising=False
    )
    monkeypatch.setattr(
        Config, "SEG_EVAL_COMPARE_PATIENT_POOLING", True, raising=False
    )
    tracker = MetricTracker()
    probs = torch.tensor([[[[[0.90, 0.80, 0.70, 0.10]]]]], dtype=torch.float32)
    target = torch.tensor([[[[[1.0, 1.0, 0.0, 0.0]]]]], dtype=torch.float32)
    tracker.update_lesion_dice_sweep(probs, target)
    tracker.update_target_cspca_aux_dice(probs, target)
    tracker.finalize_target_cspca_aux_dice()
    tracker.finalize_validation_dice_threshold()
    tracker.patient_decision_threshold = 0.75
    tracker.patient_balanced_accuracy_threshold = 0.75
    tracker.patient_threshold_at_fixed_spec = 0.85
    tracker.patient_threshold_at_fixed_sens = 0.65
    tracker.tbx_roi_decision_threshold = 0.70
    tracker.region_decision_threshold = 0.55
    tracker.region_balanced_accuracy_threshold = 0.55
    tracker.region_threshold_at_fixed_spec = 0.80
    tracker.region_threshold_at_fixed_sens = 0.40

    bundle = tracker.build_frozen_thresholds(validation_epoch=12)

    assert bundle["schema_version"] == 5
    assert bundle["source"] == "validation"
    assert bundle["validation_epoch"] == 12
    assert bundle["dice"]["segmentation"] > 0.5
    assert bundle["patient"]["decision"] == 0.65
    assert bundle["patient"]["balanced_accuracy"] == 0.75
    assert bundle["patient"]["decision_selection_rule"] == "fixed_sensitivity"
    assert bundle["patient"]["pooling_mode"] == "logit_lme_contrast"
    assert "patient_logit_lme" in bundle
    calibration = bundle["patient"]["pooling_calibration"]
    assert calibration["mode"] == "logit_lme_contrast"
    assert calibration["alpha"] == 1.0
    assert calibration["beta"] == 0.0
    assert has_frozen_validation_thresholds(bundle)
    assert bundle["tbx_roi"]["decision"] == 0.70
    assert bundle["tbx_roi"]["decision_selection_rule"] == "max_balanced_accuracy"
    assert bundle["region"]["decision"] == 0.80
    assert bundle["region"]["balanced_accuracy"] == 0.55
    assert bundle["region"]["decision_selection_rule"] == "fixed_specificity"
    flat = flatten_frozen_thresholds(bundle, prefix="validation_")
    assert flat["validation_patient_decision_threshold"] == 0.65
    assert flat["validation_patient_balanced_accuracy_threshold"] == 0.75
    assert flat["validation_patient_decision_selection_rule"] == "fixed_sensitivity"
    assert flat["validation_region_decision_threshold"] == 0.80
    assert flat["validation_region_balanced_accuracy_threshold"] == 0.55
    assert flat["validation_region_decision_selection_rule"] == "fixed_specificity"

    wrong_decision_bundle = {
        **bundle,
        "patient": {**bundle["patient"], "decision": 0.75},
    }
    assert not has_frozen_validation_thresholds(wrong_decision_bundle)
    invalid_dice_bundle = {
        **bundle,
        "dice": {**bundle["dice"], "segmentation": float("nan")},
    }
    assert not has_frozen_validation_thresholds(invalid_dice_bundle)

    legacy_bundle = {
        **bundle,
        "schema_version": 4,
        "patient": {
            key: value
            for key, value in bundle["patient"].items()
            if key != "pooling_calibration"
        },
    }
    assert not has_frozen_validation_thresholds(legacy_bundle)


def test_original_logit_lme_bundle_rejects_contrast_thresholds(monkeypatch):
    monkeypatch.setattr(Config, "SEG_PATIENT_POOLING", "logit_lme", raising=False)
    monkeypatch.setattr(
        Config, "SEG_EVAL_COMPARE_PATIENT_POOLING", False, raising=False
    )
    tracker = MetricTracker()
    tracker.patient_pooling_mode = "logit_lme"
    tracker.patient_decision_threshold = 0.61
    tracker.patient_balanced_accuracy_threshold = 0.61
    tracker.patient_threshold_at_fixed_spec = 0.72
    tracker.patient_threshold_at_fixed_sens = 0.43
    bundle = tracker.build_frozen_thresholds(validation_epoch=5)

    assert bundle["patient"]["pooling_mode"] == "logit_lme"
    assert bundle["patient"]["decision"] == 0.43
    assert bundle["patient"]["balanced_accuracy"] == 0.61
    assert bundle["patient"]["decision_selection_rule"] == "fixed_sensitivity"
    assert "pooling_calibration" not in bundle["patient"]
    assert has_frozen_validation_thresholds(bundle)

    contrast_bundle = {
        **bundle,
        "patient": {
            **bundle["patient"],
            "pooling_mode": "logit_lme_contrast",
            "pooling_calibration": {
                "mode": "logit_lme_contrast",
                "lme_r": 8.0,
                "alpha": 1.0,
                "beta": 1.0,
                "intercept": 0.0,
            },
        },
    }
    assert not has_frozen_validation_thresholds(contrast_bundle)


def test_validation_metric_output_is_backward_compatible_and_additive():
    columns = set(MetricTracker().get_val_dict())
    legacy_columns = {
        "val_loss_total",
        "val_loss_lesion_dense",
        "val_loss_lesion_sparse",
        "val_loss_lesion_sys",
        "val_lesion_dice",
        "val_lesion_full_crop_dice",
        "val_lesion_gland_dice",
        "val_segmentation_threshold",
        "val_target_cspca_dice",
        "val_tbx_masked_dice",
        "val_tbx_roi_auc",
        "val_tbx_roi_auprc",
        "val_patient_auc",
        "val_patient_auprc",
        "val_region_auc",
        "val_region_auprc",
        "val_lesion_froc_n",
        "val_target_cspca_froc_n",
        "val_entropy_lesion_all_bits",
    }
    dual_pooling_columns = {
        "val_patient_contrast_auc",
        "val_patient_contrast_auprc",
        "val_patient_contrast_decision_threshold",
        "val_patient_logit_lme_auc",
        "val_patient_logit_lme_auprc",
        "val_patient_logit_lme_decision_threshold",
    }

    assert legacy_columns <= columns
    assert dual_pooling_columns <= columns


if __name__ == "__main__":
    test_tbx_pos_neg_bce_uses_sampled_roi_and_ignores_unsampled()
    test_prediction_entropy_tracks_available_supervision_subsets()
    test_pub_dense_cases_do_not_enter_patient_metrics()
    test_seg_risk_map_metrics_use_seg_patient_gt_and_sbx_region_gt()
    test_seg_risk_map_patient_pooling_can_ignore_gland_mask_for_deployment()
    test_seg_risk_map_patient_pooling_uses_gland_mask_by_default()
    test_contrast_logit_lme_is_invariant_to_additive_gland_logit_shift()
    test_validation_fits_contrast_pooling_and_test_reuses_frozen_coefficients()
    test_seg_risk_map_patient_pooling_skips_missing_gland_mask()
    test_outside_gland_penalty_uses_only_outside_gland_voxels()
    test_patient_risk_loss_uses_gland_mask_by_default()
    test_patient_risk_loss_skips_case_without_valid_gland_mask()
    test_target_cspca_dice_uses_only_positive_target_cases()
    test_tbx_roi_metrics_report_sensitivity_at_fixed_roc_specificity()
    test_voxel_operating_metrics_report_fixed_specificity_and_sensitivity()
    test_dense_lesion_metrics_ignore_predictions_outside_gland()
    test_froc_metrics_report_sensitivity_at_fixed_fp_per_patient()
    test_picai_froc_uses_one_to_one_matching_for_multi_lesion_overlap()
    test_picai_froc_rejects_diffuse_component_below_iou_threshold()
    test_picai_froc_prediction_mask_does_not_crop_ground_truth()
    test_picai_froc_discards_duplicate_candidate_with_sufficient_overlap()
    test_target_cspca_aux_dice_reports_swept_threshold_and_topk_upper_bound()
