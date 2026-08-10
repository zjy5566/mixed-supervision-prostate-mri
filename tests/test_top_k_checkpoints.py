import os
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import train


def test_lesion_dice_checkpoint_selection_requires_eligible_ra_cases():
    with pytest.raises(RuntimeError, match="no eligible dense RA case"):
        train.validate_checkpoint_metric_support(
            SimpleNamespace(lesion_dice_n=0),
            "lesion_dice",
        )
    train.validate_checkpoint_metric_support(
        SimpleNamespace(lesion_dice_n=3),
        "lesion_dice",
    )


def test_patient_auprc_selection_requires_both_validation_classes():
    with pytest.raises(RuntimeError, match="both csPCa classes"):
        train.validate_checkpoint_metric_support(
            SimpleNamespace(
                patient_n=3,
                patient_tp=3,
                patient_fn=0,
                patient_tn=0,
                patient_fp=0,
            ),
            "patient_auprc",
        )
    train.validate_checkpoint_metric_support(
        SimpleNamespace(
            patient_n=4,
            patient_tp=2,
            patient_fn=0,
            patient_tn=2,
            patient_fp=0,
        ),
        "patient_auprc",
    )


def _valid_b_metric_tracker():
    return SimpleNamespace(
        lesion_dice=SimpleNamespace(avg=0.1),
        tbx_roi_auprc=0.8,
        tbx_masked_dice=SimpleNamespace(avg=0.4),
        tbx_masked_dice_n=3,
        tbx_roi_true=[0, 0, 1, 1],
        tbx_roi_n=4,
        region_auprc=0.6,
        region_n=4,
        region_tn=2,
        region_fp=0,
        region_fn=0,
        region_tp=2,
        patient_auprc=0.7,
        patient_n=4,
        patient_tn=2,
        patient_fp=0,
        patient_fn=0,
        patient_tp=2,
    )


def test_b_task_native_metric_formulas():
    tracker = _valid_b_metric_tracker()
    tbx_native = 0.70 * 0.8 + 0.30 * 0.4
    assert train.select_validation_metric(tracker, "tbx_native") == pytest.approx(
        tbx_native
    )
    assert train.select_validation_metric(
        tracker, "region_auprc"
    ) == pytest.approx(0.6)
    assert train.select_validation_metric(
        tracker, "tbx_sbx_native"
    ) == pytest.approx(0.50 * tbx_native + 0.50 * 0.6)
    assert train.select_validation_metric(
        tracker, "tbx_sbx_patient_native"
    ) == pytest.approx((tbx_native + 0.6 + 0.7) / 3.0)


def test_b_task_native_selection_requires_every_constituent():
    tracker = _valid_b_metric_tracker()
    for metric in (
        "tbx_native",
        "region_auprc",
        "tbx_sbx_native",
        "tbx_sbx_patient_native",
    ):
        train.validate_checkpoint_metric_support(tracker, metric)

    one_tbx_class = _valid_b_metric_tracker()
    one_tbx_class.tbx_roi_true = [1, 1]
    one_tbx_class.tbx_roi_n = 2
    with pytest.raises(RuntimeError, match="TBx ROI voxels from both"):
        train.validate_checkpoint_metric_support(one_tbx_class, "tbx_native")

    no_tbx_dice = _valid_b_metric_tracker()
    no_tbx_dice.tbx_masked_dice_n = 0
    with pytest.raises(RuntimeError, match="positive TBx case"):
        train.validate_checkpoint_metric_support(no_tbx_dice, "tbx_native")

    one_region_class = _valid_b_metric_tracker()
    one_region_class.region_tn = 0
    one_region_class.region_tp = 4
    with pytest.raises(RuntimeError, match="SBx regions from both"):
        train.validate_checkpoint_metric_support(
            one_region_class, "tbx_sbx_native"
        )

    one_patient_class = _valid_b_metric_tracker()
    one_patient_class.patient_tn = 0
    one_patient_class.patient_tp = 4
    with pytest.raises(RuntimeError, match="both csPCa classes"):
        train.validate_checkpoint_metric_support(
            one_patient_class, "tbx_sbx_patient_native"
        )


def test_native_top_k_retains_only_five_highest_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(train.Config, "TOP_K_CHECKPOINTS", 5, raising=False)

    def fake_save_checkpoint(path, *args, **kwargs):
        with open(path, "wb") as handle:
            handle.write(b"checkpoint")

    monkeypatch.setattr(train, "save_checkpoint", fake_save_checkpoint)
    records = []
    scores = [0.10, 0.40, 0.20, 0.50, 0.30, 0.60, 0.35]
    for epoch, score in enumerate(scores, start=1):
        records = train.update_top_k_native_checkpoints(
            str(tmp_path),
            records,
            model=object(),
            criterion=object(),
            optimizer=object(),
            scheduler=object(),
            epoch=epoch,
            native_metric=score,
            common_metric=score / 2,
            native_metric_name="patient_auprc",
            common_metric_name="patient_auprc",
            config_name="test",
        )

    assert [record["native_metric_value"] for record in records] == [
        0.60,
        0.50,
        0.40,
        0.35,
        0.30,
    ]
    retained_files = sorted(tmp_path.glob("top_native_epoch_*.pth"))
    assert len(retained_files) == 5
    assert {path.name for path in retained_files} == {
        "top_native_epoch_002.pth",
        "top_native_epoch_004.pth",
        "top_native_epoch_005.pth",
        "top_native_epoch_006.pth",
        "top_native_epoch_007.pth",
    }

    manifest = pd.read_csv(tmp_path / train.TOP_K_CHECKPOINT_MANIFEST)
    assert list(manifest["rank"]) == [1, 2, 3, 4, 5]
    assert list(manifest["native_metric_value"]) == [0.60, 0.50, 0.40, 0.35, 0.30]
    assert set(manifest["native_metric_name"]) == {"patient_auprc"}
    assert len(train.configured_top_k_checkpoint_files(str(tmp_path))) == 5


def test_checkpoint_contains_validation_threshold_bundle(tmp_path):
    model = torch.nn.Linear(2, 1)
    thresholds = {
        "schema_version": 1,
        "source": "validation",
        "validation_epoch": 4,
        "dice": {"segmentation": 0.65, "selection_metric": "lesion_dice"},
        "patient": {
            "decision": 0.75,
            "at_fixed_specificity": 0.85,
            "at_fixed_sensitivity": 0.65,
        },
    }
    path = tmp_path / "checkpoint.pth"

    train.save_checkpoint(
        str(path),
        model,
        criterion=None,
        optimizer=None,
        scheduler=None,
        epoch=4,
        best_metric=0.8,
        config_name="test",
        frozen_thresholds=thresholds,
    )
    checkpoint = torch.load(path, map_location="cpu")

    assert checkpoint["frozen_thresholds"] == thresholds
    assert checkpoint["mil_pooling"] == train.Config.MIL_POOLING
    assert checkpoint["mil_lme_r"] == train.Config.LME_R
    assert checkpoint["experiment_config"]["SEG_PATIENT_POOLING"] == "logit_lme"
    assert checkpoint["experiment_config"]["SEG_EVAL_USE_GLAND_MASK"] is True
