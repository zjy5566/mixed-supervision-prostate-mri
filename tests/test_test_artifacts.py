import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import test as test_module
from test import TestArtifactExporter
from utils import MetricTracker, SegRiskMapEvaluator


def test_checkpoint_pooling_metadata_restores_nonparameterised_model_settings(
    monkeypatch,
):
    model = SimpleNamespace(mil_pooling="lme", lme_r=8.0)
    monkeypatch.setattr(test_module.Config, "MIL_POOLING", "lme", raising=False)
    monkeypatch.setattr(test_module.Config, "LME_R", 8.0, raising=False)
    monkeypatch.setattr(
        test_module.Config, "SEG_REGION_POOLING", "top_percent", raising=False
    )

    test_module.restore_checkpoint_pooling_config(
        model,
        {
            "mil_pooling": "mean",
            "mil_lme_r": 8.0,
            "seg_region_pooling": "top_percent",
        },
    )

    assert model.mil_pooling == "mean"
    assert model.lme_r == 8.0
    assert test_module.Config.MIL_POOLING == "mean"


def test_checkpoint_pooling_metadata_rejects_invalid_region_mode(monkeypatch):
    model = SimpleNamespace(mil_pooling="lme", lme_r=8.0)
    monkeypatch.setattr(
        test_module.Config, "SEG_REGION_POOLING", "top_percent", raising=False
    )

    with pytest.raises(ValueError, match="canonical region pooling"):
        test_module.restore_checkpoint_pooling_config(
            model,
            {"seg_region_pooling": "not-a-pooling-mode"},
        )


def test_checkpoint_metadata_restores_patient_loss_semantics(monkeypatch):
    model = SimpleNamespace(mil_pooling="lme", lme_r=8.0)
    monkeypatch.setattr(
        test_module.Config, "USE_PATIENT_RISK_LOSS", False, raising=False
    )
    monkeypatch.setattr(
        test_module.Config, "PATIENT_RISK_LOSS_WEIGHT", 0.0, raising=False
    )

    test_module.restore_checkpoint_pooling_config(
        model,
        {
            "experiment_config": {
                "USE_PATIENT_RISK_LOSS": True,
                "PATIENT_RISK_LOSS_WEIGHT": 0.05,
                "PATIENT_RISK_START_EPOCH": 15,
                "SEG_PATIENT_POOLING": "logit_lme",
                "SEG_EVAL_USE_GLAND_MASK": True,
            }
        },
    )

    assert test_module.Config.USE_PATIENT_RISK_LOSS is True
    assert test_module.Config.PATIENT_RISK_LOSS_WEIGHT == 0.05
    assert test_module.Config.PATIENT_RISK_START_EPOCH == 15


def test_validation_summary_separates_primary_and_balanced_confusions(monkeypatch):
    monkeypatch.setattr(
        test_module.Config, "SEG_PATIENT_POOLING", "logit_lme", raising=False
    )
    tracker = MetricTracker()
    tracker.patient_pooling_mode = "logit_lme"
    tracker.patient_tp, tracker.patient_fn = 3, 1
    tracker.patient_tn, tracker.patient_fp = 5, 1
    tracker.patient_actual_sens_at_fixed_sens = 1.0
    tracker.patient_spec_at_fixed_sens = 0.5
    tracker.patient_sens_at_balanced_accuracy = 0.75
    tracker.patient_spec_at_balanced_accuracy = 5.0 / 6.0
    tracker.patient_bacc_at_balanced_accuracy = (
        tracker.patient_sens_at_balanced_accuracy
        + tracker.patient_spec_at_balanced_accuracy
    ) / 2.0
    tracker.patient_balanced_accuracy_threshold = 0.6
    tracker.patient_threshold_at_fixed_sens = 0.3

    tracker.region_tp, tracker.region_fn = 4, 1
    tracker.region_tn, tracker.region_fp = 4, 1
    tracker.region_sens_at_fixed_spec = 0.4
    tracker.region_actual_spec_at_fixed_spec = 1.0
    tracker.region_sens_at_balanced_accuracy = 0.8
    tracker.region_spec_at_balanced_accuracy = 0.8
    tracker.region_bacc_at_balanced_accuracy = 0.8
    tracker.region_balanced_accuracy_threshold = 0.55
    tracker.region_threshold_at_fixed_spec = 0.8

    bundle = tracker.build_frozen_thresholds(validation_epoch=7)
    row = test_module._validation_summary_row(
        tracker,
        validation_csv="validation.csv",
        checkpoint_label="best_native",
        checkpoint_epoch=7,
        checkpoint_path="checkpoint.pth",
        frozen_thresholds=bundle,
    )

    assert row["validation_patient_primary_tp"] == 4
    assert row["validation_patient_primary_fn"] == 0
    assert row["validation_patient_primary_tn"] == 3
    assert row["validation_patient_primary_fp"] == 3
    assert row["validation_patient_secondary_balanced_tp"] == 3
    assert row["validation_patient_secondary_balanced_fn"] == 1
    assert row["validation_region_primary_tp"] == 2
    assert row["validation_region_primary_fn"] == 3
    assert row["validation_region_primary_tn"] == 5
    assert row["validation_region_primary_fp"] == 0
    assert row["validation_region_secondary_balanced_tp"] == 4
    assert row["validation_region_secondary_balanced_fp"] == 1


def test_test_artifact_exporter_writes_every_sample_and_region(tmp_path):
    lesion_probs = torch.tensor(
        [
            [[[[0.90, 0.10, 0.70, 0.20]]]],
            [[[[0.80, 0.70, 0.10, 0.20]]]],
        ],
        dtype=torch.float32,
    )
    lesion_mask = torch.tensor(
        [
            [[[[1.0, 0.0, 0.0, 0.0]]]],
            [[[[0.0, 0.0, 0.0, 0.0]]]],
        ],
        dtype=torch.float32,
    )
    zones_mask = torch.tensor(
        [
            [[[[0.0, 0.0, 0.0, 0.0]]]],
            [[[[1.0, 1.0, 2.0, 2.0]]]],
        ],
        dtype=torch.float32,
    )
    sys_labels = torch.full((2, 20), -1, dtype=torch.long)
    sys_labels[1, 0] = 3
    sys_labels[1, 1] = 0
    gland_mask = torch.ones_like(lesion_mask)
    gland_mask[0, 0, 0, 0, 2] = 0.0
    batch = {
        "pid": ["PUB_case", "PROMIS_case"],
        "source": ["PUB", "PROMIS"],
        "input": torch.zeros((2, 3, 1, 1, 4), dtype=torch.float32),
        "lesion_mask": lesion_mask,
        "target_mask": torch.zeros_like(lesion_mask),
        "zones_mask": zones_mask,
        "gland_mask": gland_mask,
        "sys_labels": sys_labels,
        "has_lesion": torch.tensor([1.0, 0.0]),
        "has_target": torch.tensor([0.0, 0.0]),
        "has_sys": torch.tensor([0.0, 1.0]),
        "has_gland": torch.tensor([1.0, 1.0]),
    }
    evaluator = SegRiskMapEvaluator(
        prob_threshold=0.5,
        positive_threshold=3,
        patient_pooling="max",
        region_pooling="max",
        region_threshold=0.75,
        max_zones=2,
        invalid_sys_label=-1,
        use_gland_mask_for_patient_pooling=False,
    )
    evaluator.update_from_batch(lesion_probs, batch)
    region_info = evaluator.case_region_info(
        lesion_probs[1, 0], batch, 1, lesion_probs.device
    )
    assert region_info is not None
    assert region_info["zone_true"] == {1: 1, 2: 0}
    assert int(region_info["valid_map"].sum().item()) == 4
    assert int(region_info["label_map"].sum().item()) == 2

    exporter = TestArtifactExporter(
        str(tmp_path),
        dataset_label="external",
        dataset_csv="test.csv",
        checkpoint_label="best",
        checkpoint_path="best_checkpoint.pth",
        checkpoint_epoch=7,
        visualization_policy="none",
    )
    exporter.update(batch, lesion_probs, evaluator)
    sample_df = exporter.finalize()

    sample_csv = tmp_path / "per_sample_metrics.csv"
    region_csv = tmp_path / "per_region_metrics.csv"
    assert sample_csv.exists()
    assert region_csv.exists()
    assert len(sample_df) == 2
    assert list(sample_df["patient_id"]) == ["PUB_case", "PROMIS_case"]

    pub = sample_df.loc[sample_df["patient_id"] == "PUB_case"].iloc[0]
    assert pub["lesion_tp"] == 1
    assert pub["lesion_fp"] == 0
    assert pub["lesion_dice"] == 1.0
    assert abs(pub["lesion_full_crop_dice"] - 2.0 / 3.0) < 1e-8
    assert pub["patient_confusion"] == ""

    promis = sample_df.loc[sample_df["patient_id"] == "PROMIS_case"].iloc[0]
    assert np.isnan(promis["patient_score_contrast"])
    assert promis["patient_score_logit_lme"] == promis["patient_score"]
    assert "patient_logit_lme_probability_threshold" in sample_df.columns
    assert "patient_logit_lme_confusion" in sample_df.columns
    legacy_sample_columns = {
        "patient_id",
        "source",
        "probability_threshold",
        "lesion_dice",
        "lesion_full_crop_dice",
        "target_cspca_dice",
        "patient_label",
        "patient_score",
        "patient_pred",
        "patient_confusion",
        "region_n",
        "region_tp",
        "region_fp",
        "region_fn",
        "region_tn",
        "visualization_path",
        "visualization_reason",
    }
    assert legacy_sample_columns <= set(sample_df.columns)

    region_df = pd.read_csv(region_csv)
    assert len(region_df) == 2
    assert set(region_df["region_confusion"]) == {"TP", "TN"}
    assert region_df["region_correct"].sum() == 2
    assert set(region_df["region_probability_threshold"]) == {0.75}


def test_sbx_visual_slice_prioritizes_errors_then_positive_then_valid_risk():
    risk = np.zeros((3, 4, 4), dtype=np.float32)
    valid = np.zeros_like(risk)
    label = np.zeros_like(risk)
    pred = np.zeros_like(risk)

    valid[0, 0, 0] = 1.0
    label[0, 0, 0] = 1.0
    pred[0, 0, 0] = 1.0
    valid[1, :, :] = 1.0  # Large sampled-negative region must not win.
    assert test_module.choose_sbx_visual_slice(valid, label, pred, risk) == 0

    valid[2, 1:3, 1:3] = 1.0
    pred[2, 1:3, 1:3] = 1.0  # False-positive region has first priority.
    assert test_module.choose_sbx_visual_slice(valid, label, pred, risk) == 2

    label.fill(0.0)
    pred.fill(0.0)
    risk[1, 2, 2] = 0.8
    risk[0, 0, 0] = 0.2
    assert test_module.choose_sbx_visual_slice(valid, label, pred, risk) == 1


def test_mixed_tbx_sbx_visualization_uses_independent_sbx_slice(tmp_path, monkeypatch):
    lesion_probs = torch.full((1, 1, 3, 6, 6), 0.05, dtype=torch.float32)
    lesion_probs[0, 0, 0, 1:3, 1:3] = 0.90
    lesion_probs[0, 0, 2, 3, 3] = 0.90
    target_mask = torch.zeros_like(lesion_probs)
    target_mask[0, 0, 2, 3, 3] = 3.0
    zones_mask = torch.zeros_like(lesion_probs)
    zones_mask[0, 0, 0, 1:3, 1:3] = 1.0
    zones_mask[0, 0, 1, 1:5, 1:5] = 2.0
    sys_labels = torch.full((1, 20), -1, dtype=torch.long)
    sys_labels[0, 0] = 3
    sys_labels[0, 1] = 1
    batch = {
        "pid": ["TCIA_mixed_visual"],
        "source": ["TCIA"],
        "input": torch.zeros((1, 3, 3, 6, 6), dtype=torch.float32),
        "lesion_mask": torch.zeros_like(lesion_probs),
        "target_mask": target_mask,
        "zones_mask": zones_mask,
        "gland_mask": torch.ones_like(lesion_probs),
        "sys_labels": sys_labels,
        "has_lesion": torch.tensor([0.0]),
        "has_target": torch.tensor([1.0]),
        "has_sys": torch.tensor([1.0]),
        "has_gland": torch.tensor([1.0]),
    }
    captured = {}
    original_save = test_module.save_seg_mil_vis

    def save_with_slice_spy(**kwargs):
        sys_gt = test_module.make_sys_label_volume(
            kwargs["zones_mask"], kwargs["sys_labels"], -1
        )
        sys_pos = (sys_gt >= 3).astype(np.float32)
        captured["main_slice"] = test_module.choose_visual_slice(
            kwargs["lesion_prob"],
            kwargs["lesion_gt"],
            kwargs["target_gt"],
            sys_pos,
        )
        captured["sbx_slice"] = test_module.choose_sbx_visual_slice(
            kwargs["region_valid_map"],
            kwargs["region_label_map"],
            kwargs["region_pred_map"],
            kwargs["lesion_prob"],
        )
        return original_save(**kwargs)

    monkeypatch.setattr(test_module, "save_seg_mil_vis", save_with_slice_spy)
    evaluator = SegRiskMapEvaluator(
        prob_threshold=0.5,
        positive_threshold=3,
        patient_pooling="max",
        region_pooling="max",
        patient_threshold=0.5,
        region_threshold=0.5,
        tbx_roi_threshold=0.5,
        invalid_sys_label=-1,
        use_gland_mask_for_patient_pooling=False,
    )
    evaluator.update_from_batch(lesion_probs, batch)
    exporter = TestArtifactExporter(
        str(tmp_path),
        visualization_policy="all",
        max_visualizations=1,
    )
    exporter.update(batch, lesion_probs, evaluator)
    sample_df = exporter.finalize()

    assert captured["main_slice"] == 2
    assert captured["sbx_slice"] == 0
    relative_path = sample_df.iloc[0]["visualization_path"]
    assert relative_path
    assert (tmp_path / relative_path).exists()


def test_test_artifact_exporter_saves_selected_visualization(tmp_path, monkeypatch):
    lesion_probs = torch.zeros((1, 1, 4, 8, 8), dtype=torch.float32)
    lesion_probs[0, 0, 2, 4, 4] = 0.9
    lesion_probs[0, 0, 2, 0, 0] = 0.99
    lesion_mask = torch.zeros_like(lesion_probs)
    lesion_mask[0, 0, 2, 4, 4] = 1.0
    gland_mask = torch.zeros_like(lesion_mask)
    gland_mask[0, 0, 2, 3:6, 3:6] = 1.0
    batch = {
        "pid": ["PUB_visual"],
        "source": ["PUB"],
        "input": torch.zeros((1, 3, 4, 8, 8), dtype=torch.float32),
        "lesion_mask": lesion_mask,
        "target_mask": torch.zeros_like(lesion_mask),
        "zones_mask": torch.zeros_like(lesion_mask),
        "gland_mask": gland_mask,
        "sys_labels": torch.full((1, 20), -1, dtype=torch.long),
        "has_lesion": torch.tensor([1.0]),
        "has_target": torch.tensor([0.0]),
        "has_sys": torch.tensor([0.0]),
        "has_gland": torch.tensor([1.0]),
    }
    captured = {}
    original_save = test_module.save_seg_mil_vis

    def save_with_threshold_spy(**kwargs):
        captured["probability_threshold"] = kwargs["probability_threshold"]
        captured["masked_risk"] = test_module.mask_risk_map_to_gland(
            kwargs["lesion_prob"], kwargs["gland_mask"]
        )
        return original_save(**kwargs)

    monkeypatch.setattr(test_module, "save_seg_mil_vis", save_with_threshold_spy)

    evaluator = SegRiskMapEvaluator(
        prob_threshold=0.73,
        positive_threshold=3,
        patient_pooling="max",
        region_pooling="max",
        invalid_sys_label=-1,
        use_gland_mask_for_patient_pooling=False,
    )
    evaluator.update_from_batch(lesion_probs, batch)
    exporter = TestArtifactExporter(
        str(tmp_path),
        visualization_policy="representative",
        max_visualizations=1,
    )
    exporter.update(batch, lesion_probs, evaluator)
    sample_df = exporter.finalize()

    relative_path = sample_df.iloc[0]["visualization_path"]
    assert relative_path
    assert (tmp_path / relative_path).exists()
    assert sample_df.iloc[0]["visualization_reason"] == "good_lesion_dice"
    assert sample_df.iloc[0]["case_is_good"] == 1
    assert "visualizations/good" in relative_path.replace("\\", "/")
    assert captured["probability_threshold"] == 0.73
    assert abs(float(captured["masked_risk"][2, 4, 4]) - 0.9) < 1e-6
    assert captured["masked_risk"][2, 0, 0] == 0.0


def test_good_policy_rejects_empty_perfect_masks_and_accepts_positive_dice(tmp_path):
    exporter = TestArtifactExporter(
        str(tmp_path),
        visualization_policy="good",
        max_visualizations=4,
        max_good_visualizations=2,
        good_dice_threshold=0.8,
    )
    empty_case = {
        "source": "PUB",
        "lesion_dice": 1.0,
        "lesion_gt_positive_voxels": 0,
    }
    good_case = {
        "source": "PUB",
        "lesion_dice": 0.85,
        "lesion_gt_positive_voxels": 10,
    }

    assert exporter._good_visualization_reason(empty_case) == ""
    assert exporter._good_visualization_reason(good_case) == "good_lesion_dice"
    should_save, reason, _ = exporter._should_visualize(good_case)
    assert should_save
    assert reason == "good_lesion_dice"
