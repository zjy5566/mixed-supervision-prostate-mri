"""
Test/inference script for the revised lesion-segmentation + MIL setting.

This version reports:
  - gland-masked voxel-level lesion metrics when dense masks are available
  - patient-level metrics derived from segmentation risk maps
  - region-level metrics from risk maps and RA-lesion-to-zone IoU labels
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Mapping, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import ProstateUnifiedDataset
from Loss_function import MixedSupervisionLoss

try:
    from model import ProstateSegMILNet as ModelClass
except ImportError:  # pragma: no cover - transition compatibility
    from model import ProstateMixedSupervisionNet as ModelClass

import utils


VISUALIZATION_POLICIES = ("none", "errors", "good", "representative", "all")

# Only these checkpoint fields may alter runtime Config. Patient endpoint fields
# are included for provenance checks, but the formal endpoint must remain plain
# gland-restricted logit-LME.
CHECKPOINT_RUNTIME_CONFIG_FIELDS = {
    "EXPERIMENT_MODE",
    "EXPERIMENT_TAG",
    "TRAIN_CSV",
    "USE_LESION_DENSE_TASK",
    "USE_LESION_SPARSE_TASK",
    "USE_LESION_SYS_TASK",
    "USE_PATIENT_RISK_LOSS",
    "USE_OUTSIDE_GLAND_PENALTY",
    "LESION_DENSE_LOSS_WEIGHT",
    "LESION_SPARSE_LOSS_WEIGHT",
    "LESION_SYS_LOSS_WEIGHT",
    "PATIENT_RISK_LOSS_WEIGHT",
    "OUTSIDE_GLAND_LOSS_WEIGHT",
    "USE_EM_WEIGHTING",
    "USE_LOGVAR_CLAMP",
    "LOGVAR_MIN",
    "LOGVAR_MAX",
    "USE_CURRICULUM",
    "LESION_DENSE_START_EPOCH",
    "LESION_SPARSE_START_EPOCH",
    "LESION_SYS_START_EPOCH",
    "PATIENT_RISK_START_EPOCH",
    "OUTSIDE_GLAND_START_EPOCH",
    "FIXED_LOSS_WEIGHTS",
    "POS_WEIGHT_VAL",
    "SYS_POS_WEIGHT_VAL",
    "SYS_FOCAL_ALPHA",
    "SYS_FOCAL_GAMMA",
    "USE_SYS_CLASS_BALANCED_BCE",
    "USE_TBX_POSITIVE_ONLY_LOSS",
    "TBX_DICE_LOSS_WEIGHT",
    "TBX_DICE_SMOOTH",
    "TBX_POSITIVE_SOFT_LABEL",
    "TBX_NEGATIVE_SOFT_LABEL",
    "MASK_TARGET_IN_SYS",
    "PATIENT_RISK_POOLING",
    "PATIENT_RISK_LME_R",
    "PATIENT_RISK_USE_GLAND_MASK",
}


def _cfg(name: str, default: Any = None) -> Any:
    return getattr(Config, name, default)


def get_dataset_task(split: str = "test") -> str:
    if split == "val":
        return _cfg("VAL_DATASET_TASK", _cfg("TASK", _cfg("DATASET_TASK", "mixed")))
    return _cfg(
        "TEST_DATASET_TASK",
        _cfg("VAL_DATASET_TASK", _cfg("TASK", _cfg("DATASET_TASK", "mixed"))),
    )


def build_dataset(csv_path: str, split: str = "test"):
    task = get_dataset_task(split)
    try:
        return ProstateUnifiedDataset(
            csv_path=csv_path,
            data_root=Config.UNIFIED_DATA_DIR,
            is_train=False,
            task=task,
        )
    except TypeError:
        return ProstateUnifiedDataset(
            csv_path=csv_path,
            data_root=Config.UNIFIED_DATA_DIR,
            is_train=False,
        )


def build_model(device: torch.device):
    try:
        model = ModelClass(
            in_channels=_cfg("IN_CHANNELS", 3),
            max_zones=_cfg("MAX_ZONES", 20),
            base_channels=_cfg("BASE_CHANNELS", 32),
            dropout_rate=_cfg("DROPOUT_RATE", 0.0),
            mil_pooling=_cfg("MIL_POOLING", "lme"),
            lme_r=_cfg("LME_R", 8.0),
            return_dict=True,
        )
    except TypeError:
        model = ModelClass(
            in_channels=_cfg("IN_CHANNELS", 3),
            num_grade_classes=_cfg("NUM_CLASSES", 7),
            max_zones=_cfg("MAX_ZONES", 20),
        )
    return model.to(device)


def build_criterion(device: torch.device):
    """Build the same loss module used by training for test-loss reporting."""
    positive_threshold = _cfg("LESION_POSITIVE_THRESHOLD", _cfg("CSPC_THRESHOLD", 1))
    kwargs = {
        "positive_threshold": positive_threshold,
        "invalid_sys_label": _cfg("INVALID_SYS_LABEL", -1),
        "pos_weight_val": _cfg("POS_WEIGHT_VAL", 2.0),
        "sys_pos_weight_val": _cfg("SYS_POS_WEIGHT_VAL", _cfg("POS_WEIGHT_VAL", 2.0)),
        "sys_focal_alpha": _cfg("SYS_FOCAL_ALPHA", 0.75),
        "sys_focal_gamma": _cfg("SYS_FOCAL_GAMMA", 2.0),
        "use_sys_class_balanced_bce": _cfg("USE_SYS_CLASS_BALANCED_BCE", True),
        "use_tbx_positive_only_loss": _cfg("USE_TBX_POSITIVE_ONLY_LOSS", False),
        "return_dict": True,
    }
    try:
        criterion = MixedSupervisionLoss(**kwargs)
    except TypeError:  # pragma: no cover - compatibility with older loss class
        criterion = MixedSupervisionLoss(
            csPCa_threshold=positive_threshold,
            invalid_sys_label=_cfg("INVALID_SYS_LABEL", -1),
            pos_weight_val=_cfg("POS_WEIGHT_VAL", 2.0),
        )
    return criterion.to(device)


def load_model_weights(model, model_path: str, device: torch.device):
    """Load a checkpoint or a plain state_dict.

    The function first tries exact loading. If the architecture has been renamed
    during migration, it falls back to loading only matching keys.
    """
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    # Remove DataParallel prefix if present.
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value

    try:
        model.load_state_dict(cleaned, strict=True)
        print("Loaded model weights with strict=True")
    except RuntimeError as err:
        print(f"Strict loading failed: {err}")
        model_state = model.state_dict()
        matched = {
            k: v for k, v in cleaned.items()
            if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)
        }
        model_state.update(matched)
        model.load_state_dict(model_state, strict=False)
        print(f"Loaded {len(matched)}/{len(model_state)} matching tensors with strict=False")

    return checkpoint if isinstance(checkpoint, dict) else {"model_state_dict": cleaned}


def restore_checkpoint_pooling_config(model, checkpoint: Mapping[str, Any]) -> None:
    """Restore saved loss semantics and non-parameterised pooling choices."""
    experiment_config = checkpoint.get("experiment_config")
    if isinstance(experiment_config, Mapping):
        saved_patient_pooling = str(
            experiment_config.get("SEG_PATIENT_POOLING", "logit_lme")
        ).strip().lower()
        if saved_patient_pooling != "logit_lme":
            raise ValueError(
                "Checkpoint uses a legacy non-plain patient endpoint "
                f"({saved_patient_pooling!r}); do not reuse it as ordinary AUPRC."
            )
        if not bool(experiment_config.get("SEG_EVAL_USE_GLAND_MASK", True)):
            raise ValueError(
                "Checkpoint patient endpoint was not gland-restricted and is "
                "incompatible with the formal B/N protocol."
            )
        for name in CHECKPOINT_RUNTIME_CONFIG_FIELDS:
            if name in experiment_config:
                setattr(Config, name, experiment_config[name])
    if "mil_pooling" in checkpoint:
        mode = str(checkpoint["mil_pooling"]).strip().lower()
        if mode not in {"mean", "max", "lme"}:
            raise ValueError(f"Checkpoint has invalid MIL pooling mode: {mode!r}")
        Config.MIL_POOLING = mode
        if hasattr(model, "mil_pooling"):
            model.mil_pooling = mode
    if "mil_lme_r" in checkpoint:
        lme_r = float(checkpoint["mil_lme_r"])
        if not np.isfinite(lme_r) or lme_r <= 0.0:
            raise ValueError(f"Checkpoint has invalid MIL LME r: {lme_r!r}")
        Config.LME_R = lme_r
        if hasattr(model, "lme_r"):
            model.lme_r = lme_r
    if "seg_region_pooling" in checkpoint:
        region_mode = str(checkpoint["seg_region_pooling"]).strip().lower()
        if region_mode not in {
            "top_percent",
            "top-percent",
            "topk_mean",
            "max",
            "mean",
            "lme",
            "logit_lme",
            "logit-lme",
        }:
            raise ValueError(
                f"Checkpoint has invalid canonical region pooling mode: {region_mode!r}"
            )
        Config.SEG_REGION_POOLING = region_mode
    print(
        "[SBx MIL pooling] mode={} | r={} | canonical region pooling={}".format(
            _cfg("MIL_POOLING", "lme"),
            float(_cfg("LME_R", 8.0)),
            _cfg("SEG_REGION_POOLING", "top_percent"),
        )
    )


def unpack_model_output(raw_outputs):
    if hasattr(utils, "unpack_model_output"):
        return utils.unpack_model_output(raw_outputs)
    if isinstance(raw_outputs, dict):
        return raw_outputs
    if isinstance(raw_outputs, (tuple, list)) and len(raw_outputs) >= 5:
        return {
            "lesion_logits": raw_outputs[2],
            "region_logits": raw_outputs[3],
            "region_valid_mask": None,
        }
    if isinstance(raw_outputs, (tuple, list)) and len(raw_outputs) == 3:
        return {
            "lesion_logits": raw_outputs[0],
            "region_logits": raw_outputs[1],
            "region_valid_mask": raw_outputs[2],
        }
    raise TypeError("Unsupported model output format.")


def infer_dataset_type(batch: Dict, b: int) -> str:
    if hasattr(utils, "infer_dataset_type"):
        return utils.infer_dataset_type(batch, b)
    pid = str(batch.get("pid", [""])[b]) if "pid" in batch else ""
    if pid.startswith("PUB"):
        return "PUB"
    if pid.startswith("TCIA"):
        return "TCIA"
    if pid.startswith("PROMIS"):
        return "PROMIS"
    if batch.get("has_lesion", torch.zeros(1))[b].item() > 0:
        return "PUB"
    if batch.get("has_target", torch.zeros(1))[b].item() > 0:
        return "TCIA"
    if batch.get("has_sys", torch.zeros(1))[b].item() > 0:
        return "PROMIS"
    return "OTHER"


def compute_patient_label(batch: Dict, b: int, positive_threshold: int, invalid_sys_label: int) -> int:
    """Biopsy-based patient label; PUB dense masks are lesion-Dice labels only."""
    has_target = batch.get("has_target", torch.zeros(1))[b].item() > 0
    has_sys = batch.get("has_sys", torch.zeros(1))[b].item() > 0
    if not (has_target or has_sys):
        return invalid_sys_label

    label = 0

    if has_target:
        if batch["target_mask"][b].max().item() >= positive_threshold:
            label = 1

    if has_sys:
        labels = batch["sys_labels"][b]
        valid = labels != invalid_sys_label
        if valid.any() and labels[valid].max().item() >= positive_threshold:
            label = 1

    return int(label)


def compute_patient_score(lesion_prob: torch.Tensor, gland_mask: torch.Tensor | None = None) -> float:
    """Patient-level score as max lesion probability, restricted to gland when possible."""
    if gland_mask is not None and gland_mask.max().item() > 0:
        values = lesion_prob[gland_mask > 0]
        if values.numel() > 0:
            return float(values.max().detach().cpu().item())
    return float(lesion_prob.max().detach().cpu().item())


def make_sys_label_volume(zones_mask: np.ndarray, sys_labels: np.ndarray, invalid_sys_label: int) -> np.ndarray:
    out = np.zeros_like(zones_mask, dtype=np.float32)
    for z_idx in range(1, min(20, len(sys_labels)) + 1):
        if int(sys_labels[z_idx - 1]) != invalid_sys_label:
            out[zones_mask == z_idx] = float(sys_labels[z_idx - 1])
    return out


def make_sys_valid_volume(
    zones_mask: np.ndarray,
    sys_labels: np.ndarray,
    invalid_sys_label: int,
) -> np.ndarray:
    """Map valid sampled SBx labels to voxels, including negative regions."""
    out = np.zeros_like(zones_mask, dtype=np.float32)
    for z_idx in range(1, min(20, len(sys_labels)) + 1):
        if int(sys_labels[z_idx - 1]) != invalid_sys_label:
            out[zones_mask == z_idx] = 1.0
    return out


def choose_visual_slice(lesion_prob: np.ndarray, lesion_gt: np.ndarray, target_gt: np.ndarray, sys_gt: np.ndarray) -> int:
    if lesion_gt.sum() > 0:
        return int(np.argmax(lesion_gt.sum(axis=(1, 2))))
    if target_gt.sum() > 0:
        return int(np.argmax(target_gt.sum(axis=(1, 2))))
    if sys_gt.sum() > 0:
        return int(np.argmax(sys_gt.sum(axis=(1, 2))))
    if lesion_prob.sum() > 0:
        return int(np.argmax(lesion_prob.sum(axis=(1, 2))))
    return int(lesion_prob.shape[0] // 2)


def choose_sbx_visual_slice(
    region_valid_map: np.ndarray | None,
    region_label_map: np.ndarray | None,
    region_pred_map: np.ndarray | None,
    lesion_prob: np.ndarray,
) -> int | None:
    """Choose an SBx-specific slice without being overridden by TBx GT."""
    if region_valid_map is None:
        return None
    valid = np.asarray(region_valid_map) > 0
    lesion_prob = np.asarray(lesion_prob)
    if valid.shape != lesion_prob.shape:
        raise ValueError("SBx valid map and lesion risk map must have identical shapes.")
    if not valid.any():
        return None

    label = (
        np.zeros_like(valid)
        if region_label_map is None
        else np.asarray(region_label_map) > 0
    ) & valid
    pred = (
        np.zeros_like(valid)
        if region_pred_map is None
        else np.asarray(region_pred_map) > 0
    ) & valid
    if label.shape != valid.shape or pred.shape != valid.shape:
        raise ValueError("SBx label/prediction maps must match the valid-map shape.")

    # Prefer qualitative errors, then labelled positives, then predicted
    # positives. This makes both failure and successful-positive examples
    # display their clinically relevant region.
    for focus in (valid & (label != pred), label, pred):
        if focus.any():
            return int(np.argmax(focus.sum(axis=(1, 2))))

    # All sampled regions are correctly predicted negative. Show the slice
    # containing the highest residual risk within valid SBx supervision.
    finite_risk = np.where(np.isfinite(lesion_prob), lesion_prob, -np.inf)
    valid_risk = np.where(valid, finite_risk, -np.inf)
    per_slice_max = valid_risk.reshape(valid_risk.shape[0], -1).max(axis=1)
    if np.isfinite(per_slice_max).any():
        return int(np.argmax(per_slice_max))
    return int(np.argmax(valid.sum(axis=(1, 2))))


def mask_risk_map_to_gland(
    lesion_prob: np.ndarray,
    gland_mask: np.ndarray,
) -> np.ndarray:
    """Zero test-time risk outside the prostate before visualisation."""
    lesion_prob = np.asarray(lesion_prob)
    gland_mask = np.asarray(gland_mask)
    if lesion_prob.shape != gland_mask.shape:
        raise ValueError(
            "lesion_prob and gland_mask must have identical shapes for test "
            "visualisation."
        )
    return np.where(gland_mask > 0, lesion_prob, 0.0)


def save_seg_mil_vis(
    img_vol: np.ndarray,
    lesion_gt: np.ndarray,
    target_gt: np.ndarray,
    zones_mask: np.ndarray,
    sys_labels: np.ndarray,
    lesion_prob: np.ndarray,
    gland_mask: np.ndarray,
    region_valid_map: np.ndarray | None,
    region_label_map: np.ndarray | None,
    region_pred_map: np.ndarray | None,
    pid: str,
    save_path: str,
    annotation: str = "",
    probability_threshold: float = 0.5,
):
    invalid_sys_label = int(_cfg("INVALID_SYS_LABEL", -1))
    positive_threshold = int(_cfg("LESION_POSITIVE_THRESHOLD", _cfg("CSPC_THRESHOLD", 1)))
    sys_gt = make_sys_label_volume(zones_mask, sys_labels, invalid_sys_label)
    sys_pos = (sys_gt >= positive_threshold).astype(np.float32)
    sys_valid = make_sys_valid_volume(zones_mask, sys_labels, invalid_sys_label)
    lesion_prob = mask_risk_map_to_gland(lesion_prob, gland_mask)
    if region_valid_map is None:
        region_valid_map = sys_valid
    if region_label_map is None:
        region_label_map = sys_pos

    z = choose_visual_slice(lesion_prob, lesion_gt, target_gt, sys_pos)
    sbx_z = choose_sbx_visual_slice(
        region_valid_map,
        region_label_map,
        region_pred_map,
        lesion_prob,
    )
    s_img = img_vol[z]
    s_prob = lesion_prob[z]
    s_lesion = lesion_gt[z]
    s_target = target_gt[z]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    title = f"Patient: {pid} | Slice: {z}/{img_vol.shape[0]}"
    if annotation:
        title = f"{title}\n{annotation}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0, 0].imshow(s_img, cmap="gray")
    axes[0, 0].set_title("T2 MRI")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(s_img, cmap="gray")
    risk_masked = np.ma.masked_where(s_prob < 0.1, s_prob)
    im_risk = axes[0, 1].imshow(risk_masked, cmap="hot", alpha=0.55, vmin=0, vmax=1)
    axes[0, 1].set_title("Predicted lesion probability (gland-masked)")
    axes[0, 1].axis("off")
    fig.colorbar(im_risk, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[0, 2].imshow(s_img, cmap="gray")
    if s_lesion.sum() > 0:
        axes[0, 2].imshow(np.ma.masked_where(s_lesion == 0, s_lesion), cmap="autumn", alpha=0.55)
    axes[0, 2].set_title("GT dense lesion mask")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(s_img, cmap="gray")
    if s_target.sum() > 0:
        target_pos = (s_target >= positive_threshold).astype(np.float32)
        axes[1, 0].imshow(np.ma.masked_where(target_pos == 0, target_pos), cmap="autumn", alpha=0.55)
    axes[1, 0].set_title("TBx-positive target ROI voxels")
    axes[1, 0].axis("off")

    sbx_panel_z = z if sbx_z is None else sbx_z
    axes[1, 1].imshow(img_vol[sbx_panel_z], cmap="gray")
    if sbx_z is None:
        axes[1, 1].set_title("SBx: no valid sampled regions")
    else:
        sbx_valid = np.asarray(region_valid_map[sbx_z]) > 0
        sbx_positive = (np.asarray(region_label_map[sbx_z]) > 0) & sbx_valid
        sbx_negative = sbx_valid & (~sbx_positive)
        sbx_pred = (
            np.zeros_like(sbx_valid)
            if region_pred_map is None
            else (np.asarray(region_pred_map[sbx_z]) > 0) & sbx_valid
        )
        legend_handles = []
        if sbx_negative.any():
            axes[1, 1].imshow(
                np.ma.masked_where(~sbx_negative, sbx_negative.astype(np.float32)),
                cmap="Blues",
                alpha=0.42,
                vmin=0,
                vmax=1,
            )
            legend_handles.append(Patch(facecolor="tab:blue", alpha=0.42, label="GT csPCa−"))
        if sbx_positive.any():
            axes[1, 1].imshow(
                np.ma.masked_where(~sbx_positive, sbx_positive.astype(np.float32)),
                cmap="Greens",
                alpha=0.50,
                vmin=0,
                vmax=1,
            )
            legend_handles.append(Patch(facecolor="tab:green", alpha=0.50, label="GT csPCa+"))
        if sbx_pred.any():
            if (~sbx_pred).any():
                axes[1, 1].contour(
                    sbx_pred.astype(np.float32),
                    levels=[0.5],
                    colors="red",
                    linewidths=2,
                    linestyles="dashed",
                )
            else:
                axes[1, 1].imshow(
                    np.ma.masked_where(~sbx_pred, sbx_pred.astype(np.float32)),
                    cmap="Reds",
                    alpha=0.18,
                    vmin=0,
                    vmax=1,
                )
            legend_handles.append(
                Line2D([0], [0], color="red", linewidth=2, linestyle="--", label="Pred csPCa+")
            )
        if legend_handles:
            axes[1, 1].legend(handles=legend_handles, loc="lower right", fontsize=8)
        axes[1, 1].set_title(f"SBx GT / prediction (independent slice {sbx_z})")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(s_img, cmap="gray")
    if s_lesion.sum() > 0:
        axes[1, 2].contour(s_lesion, levels=[0.5], linewidths=2)
    pred_mask = s_prob >= float(probability_threshold)
    if pred_mask.any():
        axes[1, 2].contour(
            pred_mask.astype(np.float32),
            levels=[0.5],
            linewidths=2,
            linestyles="dashed",
        )
    axes[1, 2].set_title(
        "Contours: GT solid / "
        f"Pred dashed (threshold={float(probability_threshold):.3f})"
    )
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


class TestArtifactExporter:
    """Collect per-case/per-region metrics and selected QA visualisations.

    The exporter is deliberately independent from the aggregate metric tracker:
    aggregate AUC/FROC values remain dataset-level statistics, while this class
    writes only metrics that are well-defined for one test case or one region.
    """

    __test__ = False  # Prevent pytest from treating this helper as a test class.
    REGION_COLUMNS = (
        "sample_index",
        "patient_id",
        "source",
        "dataset_label",
        "dataset_csv",
        "checkpoint_label",
        "checkpoint_epoch",
        "checkpoint_path",
        "zone_id",
        "region_label",
        "region_score",
        "region_pred",
        "region_probability_threshold",
        "region_correct",
        "region_confusion",
    )

    def __init__(
        self,
        output_dir: str,
        *,
        dataset_label: str = "external",
        dataset_csv: str = "",
        checkpoint_label: str = "best",
        checkpoint_path: str = "",
        checkpoint_epoch: int = 0,
        visualization_policy: str = "representative",
        max_visualizations: int = 12,
        low_dice_threshold: float = 0.5,
        good_dice_threshold: float = 0.8,
        max_good_visualizations: int = 4,
    ):
        policy = str(visualization_policy).lower()
        if policy not in VISUALIZATION_POLICIES:
            raise ValueError(
                f"Unknown visualization policy {visualization_policy!r}; "
                f"choose from {VISUALIZATION_POLICIES}."
            )

        self.output_dir = os.path.abspath(output_dir)
        self.visualization_dir = os.path.join(self.output_dir, "visualizations")
        self.dataset_label = str(dataset_label)
        self.dataset_csv = str(dataset_csv)
        self.checkpoint_label = str(checkpoint_label)
        self.checkpoint_path = str(checkpoint_path)
        self.checkpoint_epoch = int(checkpoint_epoch)
        self.visualization_policy = policy
        self.max_visualizations = max(0, int(max_visualizations))
        self.low_dice_threshold = float(low_dice_threshold)
        self.good_dice_threshold = float(good_dice_threshold)
        self.max_good_visualizations = max(0, int(max_good_visualizations))
        if self.max_visualizations > 0:
            self.max_good_visualizations = min(
                self.max_good_visualizations, self.max_visualizations
            )
        self.sample_rows = []
        self.region_rows = []
        self.saved_visualizations = 0
        self.saved_good_visualizations = 0
        self.saved_representative_buckets = set()
        self.saved_good_bucket_counts: Dict[Tuple[str, str], int] = {}

    @staticmethod
    def _flag(batch: Dict, key: str, b: int) -> bool:
        if key not in batch:
            return False
        value = batch[key][b]
        if torch.is_tensor(value):
            return bool(value.item() > 0)
        return bool(value)

    @staticmethod
    def _binary_metrics(prefix: str, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        pred = pred.bool().reshape(-1)
        target = target.bool().reshape(-1)
        tp = int(torch.logical_and(pred, target).sum().item())
        fp = int(torch.logical_and(pred, ~target).sum().item())
        fn = int(torch.logical_and(~pred, target).sum().item())
        tn = int(torch.logical_and(~pred, ~target).sum().item())
        pred_pos = tp + fp
        gt_pos = tp + fn
        dice_denom = 2 * tp + fp + fn
        dice = 1.0 if dice_denom == 0 else (2.0 * tp) / dice_denom
        sensitivity = float("nan") if gt_pos == 0 else tp / gt_pos
        specificity = float("nan") if (tn + fp) == 0 else tn / (tn + fp)
        return {
            f"{prefix}_num_voxels": int(target.numel()),
            f"{prefix}_gt_positive_voxels": gt_pos,
            f"{prefix}_pred_positive_voxels": pred_pos,
            f"{prefix}_tp": tp,
            f"{prefix}_fp": fp,
            f"{prefix}_fn": fn,
            f"{prefix}_tn": tn,
            f"{prefix}_dice": float(dice),
            f"{prefix}_f1": float(dice),
            f"{prefix}_sensitivity": float(sensitivity),
            f"{prefix}_specificity": float(specificity),
        }

    @staticmethod
    def _safe_ratio(numerator: int, denominator: int) -> float:
        return float("nan") if denominator == 0 else float(numerator / denominator)

    def _visualization_reason(self, row: Dict[str, Any]) -> str:
        reasons = []
        if row.get("patient_correct") == 0:
            reasons.append("patient_error")
        if int(row.get("region_fp", 0)) + int(row.get("region_fn", 0)) > 0:
            reasons.append("region_error")

        lesion_dice = row.get("lesion_dice", np.nan)
        if np.isfinite(lesion_dice) and lesion_dice < self.low_dice_threshold:
            reasons.append("low_lesion_dice")
        target_dice = row.get("target_cspca_dice", np.nan)
        if np.isfinite(target_dice) and target_dice < self.low_dice_threshold:
            reasons.append("low_target_dice")
        return "+".join(reasons)

    def _good_visualization_reason(self, row: Dict[str, Any]) -> str:
        """Return the positive-task reasons that make a case qualitatively good.

        Empty/background-only cases are deliberately excluded: a Dice of one
        on an all-negative mask is not an informative qualitative result.
        """
        reasons = []
        lesion_dice = row.get("lesion_dice", np.nan)
        if (
            int(row.get("lesion_gt_positive_voxels", 0)) > 0
            and np.isfinite(lesion_dice)
            and lesion_dice >= self.good_dice_threshold
        ):
            reasons.append("good_lesion_dice")

        target_dice = row.get("target_cspca_dice", np.nan)
        if (
            int(row.get("tbx_roi_gt_positive_voxels", 0)) > 0
            and np.isfinite(target_dice)
            and target_dice >= self.good_dice_threshold
        ):
            reasons.append("good_tbx_dice")

        if (
            int(row.get("region_positive_gt", 0)) > 0
            and int(row.get("region_fp", 0)) == 0
            and int(row.get("region_fn", 0)) == 0
        ):
            reasons.append("good_region_prediction")

        if row.get("patient_confusion") == "TP":
            reasons.append("good_patient_tp")
        return "+".join(reasons)

    def _good_slot_available(self) -> bool:
        return (
            self.max_good_visualizations > 0
            and self.saved_good_visualizations < self.max_good_visualizations
        )

    def _good_bucket_available(self, bucket: Tuple[str, str]) -> bool:
        # Allow a small number of examples from the same supervision/source
        # combination without letting an early CSV block monopolise all slots.
        return self.saved_good_bucket_counts.get(bucket, 0) < 2

    def _non_good_slot_available(self) -> bool:
        if self.max_visualizations == 0:
            return True
        # Keep these slots unused until qualifying good cases arrive later in
        # the loader, rather than allowing early errors to consume the quota.
        reserved_good_slots = min(
            self.max_good_visualizations,
            max(1, self.max_visualizations // 3),
        )
        non_good_limit = self.max_visualizations - reserved_good_slots
        non_good_saved = self.saved_visualizations - self.saved_good_visualizations
        return non_good_saved < non_good_limit

    def _should_visualize(self, row: Dict[str, Any]) -> Tuple[bool, str, Optional[Tuple[str, str]]]:
        if self.visualization_policy == "none":
            return False, "", None
        if self.max_visualizations > 0 and self.saved_visualizations >= self.max_visualizations:
            return False, "", None

        error_reason = self._visualization_reason(row)
        if self.visualization_policy == "errors":
            return bool(error_reason), error_reason, None
        if self.visualization_policy == "all":
            return True, error_reason or "all", None

        good_reason = "" if error_reason else self._good_visualization_reason(row)
        good_bucket = (
            str(row.get("source", "OTHER")),
            good_reason.split("+", 1)[0],
        )
        if self.visualization_policy == "good":
            should_save = (
                bool(good_reason)
                and self._good_slot_available()
                and self._good_bucket_available(good_bucket)
            )
            return should_save, good_reason if should_save else "", good_bucket

        if (
            good_reason
            and self._good_slot_available()
            and self._good_bucket_available(good_bucket)
        ):
            return True, good_reason, good_bucket
        if not self._non_good_slot_available():
            return False, "", None

        label = row.get("patient_label", np.nan)
        label_bucket = "unlabelled" if not np.isfinite(label) else f"patient_{int(label)}"
        bucket = (str(row.get("source", "OTHER")), label_bucket)
        if error_reason:
            return True, error_reason, bucket
        if bucket not in self.saved_representative_buckets:
            return True, "representative", bucket
        return False, "", bucket

    def _save_visualization(
        self,
        *,
        batch: Dict,
        b: int,
        lesion_prob: torch.Tensor,
        region_valid_map: Optional[np.ndarray],
        region_label_map: Optional[np.ndarray],
        region_pred_map: Optional[np.ndarray],
        row: Dict[str, Any],
        reason: str,
        bucket: Optional[Tuple[str, str]],
    ) -> None:
        is_good = reason.startswith("good_")
        save_dir = (
            os.path.join(self.visualization_dir, "good")
            if is_good
            else self.visualization_dir
        )
        os.makedirs(save_dir, exist_ok=True)
        patient_id = str(row["patient_id"])
        filename = (
            f"{int(row['sample_index']):04d}_{utils.safe_vis_filename(row['source'])}_"
            f"{utils.safe_vis_filename(patient_id)}.png"
        )
        save_path = os.path.join(save_dir, filename)
        annotation_parts = [reason]
        for label, key in (
            ("Lesion Dice", "lesion_dice"),
            ("TBx Dice", "target_cspca_dice"),
            ("Region F1", "region_f1"),
            ("Patient score", "patient_score"),
        ):
            value = row.get(key, np.nan)
            if np.isfinite(value):
                annotation_parts.append(f"{label}={float(value):.3f}")
        empty_like = torch.zeros_like(lesion_prob)
        sys_labels = batch.get(
            "sys_labels",
            torch.full(
                (batch["input"].size(0), int(_cfg("MAX_ZONES", 20))),
                int(_cfg("INVALID_SYS_LABEL", -1)),
                device=lesion_prob.device,
            ),
        )
        save_seg_mil_vis(
            img_vol=batch["input"][b, 0].detach().cpu().numpy(),
            lesion_gt=utils.mask_for_visualisation(batch, "lesion_mask", b, empty_like),
            target_gt=utils.mask_for_visualisation(batch, "target_mask", b, empty_like),
            zones_mask=utils.mask_for_visualisation(batch, "zones_mask", b, empty_like),
            sys_labels=sys_labels[b].detach().cpu().numpy(),
            lesion_prob=lesion_prob.detach().cpu().numpy(),
            gland_mask=utils.mask_for_visualisation(batch, "gland_mask", b, empty_like),
            region_valid_map=region_valid_map,
            region_label_map=region_label_map,
            region_pred_map=region_pred_map,
            pid=patient_id,
            save_path=save_path,
            annotation=" | ".join(annotation_parts),
            probability_threshold=float(row["probability_threshold"]),
        )
        row["visualization_path"] = os.path.relpath(save_path, self.output_dir)
        row["visualization_reason"] = reason
        self.saved_visualizations += 1
        if is_good:
            self.saved_good_visualizations += 1
            if bucket is not None:
                self.saved_good_bucket_counts[bucket] = (
                    self.saved_good_bucket_counts.get(bucket, 0) + 1
                )
        elif bucket is not None:
            self.saved_representative_buckets.add(bucket)

    def update(self, batch: Dict, lesion_probs: torch.Tensor, seg_evaluator) -> None:
        """Add every case in one inference batch to the export tables."""
        prob_threshold = float(seg_evaluator.prob_threshold)
        patient_threshold = float(seg_evaluator.patient_threshold)
        patient_logit_lme_threshold = float(
            seg_evaluator.patient_logit_lme_threshold
        )
        contrast_enabled = utils.is_contrast_patient_pooling(
            seg_evaluator.patient_pooling
        )
        region_threshold = float(seg_evaluator.region_threshold)
        tbx_roi_threshold = float(seg_evaluator.tbx_roi_threshold)
        positive_threshold = int(seg_evaluator.positive_threshold)
        device = lesion_probs.device

        for b in range(lesion_probs.size(0)):
            patient_id = str(batch.get("pid", [f"case_{len(self.sample_rows) + 1}"])[b])
            source = infer_dataset_type(batch, b)
            lesion_prob = lesion_probs[b, 0]
            pred_binary = lesion_prob >= prob_threshold
            has_lesion = self._flag(batch, "has_lesion", b)
            has_target = self._flag(batch, "has_target", b)
            has_sys = self._flag(batch, "has_sys", b)
            has_gland = self._flag(batch, "has_gland", b)
            if "has_gland" not in batch and "gland_mask" in batch:
                has_gland = bool((batch["gland_mask"][b] > 0).any().item())
            gland = None
            if has_gland and "gland_mask" in batch:
                candidate_gland = batch["gland_mask"][b, 0] > 0
                if bool(candidate_gland.any().item()):
                    gland = candidate_gland

            row: Dict[str, Any] = {
                "sample_index": len(self.sample_rows) + 1,
                "patient_id": patient_id,
                "source": source,
                "dataset_label": self.dataset_label,
                "dataset_csv": self.dataset_csv,
                "checkpoint_label": self.checkpoint_label,
                "checkpoint_epoch": self.checkpoint_epoch,
                "checkpoint_path": self.checkpoint_path,
                "probability_threshold": prob_threshold,
                "patient_probability_threshold": patient_threshold,
                "patient_contrast_probability_threshold": (
                    patient_threshold if contrast_enabled else np.nan
                ),
                "patient_logit_lme_probability_threshold": patient_logit_lme_threshold,
                "region_probability_threshold": region_threshold,
                "tbx_roi_probability_threshold": tbx_roi_threshold,
                "positive_label_threshold": positive_threshold,
                "has_lesion": int(has_lesion),
                "has_target": int(has_target),
                "has_sys": int(has_sys),
                "has_gland": int(has_gland),
                "risk_min": float(lesion_prob.min().item()),
                "risk_mean": float(lesion_prob.mean().item()),
                "risk_max": float(lesion_prob.max().item()),
                "risk_std": float(lesion_prob.float().std(unbiased=False).item()),
                "pred_positive_voxels": int(pred_binary.sum().item()),
                "pred_positive_fraction": float(pred_binary.float().mean().item()),
                "lesion_dice": np.nan,
                "lesion_full_crop_dice": np.nan,
                "target_cspca_dice": np.nan,
                "patient_label": np.nan,
                "patient_score": np.nan,
                "patient_score_contrast": np.nan,
                "patient_score_logit_lme": np.nan,
                "patient_pooling_mode": str(seg_evaluator.patient_pooling),
                "patient_absolute_logit": np.nan,
                "patient_gland_median_logit": np.nan,
                "patient_contrast_logit": np.nan,
                "patient_calibrated_logit": np.nan,
                "patient_pooling_alpha": float(
                    seg_evaluator.patient_pooling_calibration.get("alpha", 1.0)
                ),
                "patient_pooling_beta": float(
                    seg_evaluator.patient_pooling_calibration.get("beta", 0.0)
                ),
                "patient_pooling_intercept": float(
                    seg_evaluator.patient_pooling_calibration.get("intercept", 0.0)
                ),
                "patient_pred": np.nan,
                "patient_correct": np.nan,
                "patient_confusion": "",
                "patient_logit_lme_pred": np.nan,
                "patient_logit_lme_correct": np.nan,
                "patient_logit_lme_confusion": "",
                "region_n": 0,
                "region_positive_gt": 0,
                "region_positive_pred": 0,
                "region_tp": 0,
                "region_fp": 0,
                "region_fn": 0,
                "region_tn": 0,
                "region_sensitivity": np.nan,
                "region_specificity": np.nan,
                "region_bacc": np.nan,
                "region_f1": np.nan,
                "case_has_error": 0,
                "case_is_good": 0,
                "good_visualization_reason": "",
                "visualization_path": "",
                "visualization_reason": "",
                "visualization_error": "",
            }

            if has_lesion and "lesion_mask" in batch:
                lesion_gt = batch["lesion_mask"][b, 0] > 0
                full_crop_metrics = self._binary_metrics(
                    "lesion_full_crop",
                    pred_binary,
                    lesion_gt,
                )
                row["lesion_full_crop_dice"] = full_crop_metrics[
                    "lesion_full_crop_dice"
                ]
                if gland is not None:
                    row.update(
                        self._binary_metrics(
                            "lesion",
                            pred_binary[gland],
                            lesion_gt[gland],
                        )
                    )

            if has_target and "target_mask" in batch:
                target_mask = batch["target_mask"][b, 0]
                sampled_roi = target_mask > 0
                target_gt = target_mask >= positive_threshold
                row["tbx_sampled_voxels"] = int(sampled_roi.sum().item())
                if sampled_roi.any():
                    tbx_roi_pred = lesion_prob >= tbx_roi_threshold
                    roi_metrics = self._binary_metrics(
                        "tbx_roi",
                        tbx_roi_pred[sampled_roi],
                        target_gt[sampled_roi],
                    )
                    row.update(roi_metrics)
                    if int(target_gt[sampled_roi].sum().item()) > 0:
                        target_metrics = self._binary_metrics(
                            "target_cspca",
                            pred_binary[sampled_roi],
                            target_gt[sampled_roi],
                        )
                        row["target_cspca_dice"] = target_metrics[
                            "target_cspca_dice"
                        ]
            else:
                row["tbx_sampled_voxels"] = 0

            patient_label = seg_evaluator._patient_label(batch, b, device)
            if patient_label is not None:
                patient_mask = seg_evaluator._patient_score_mask(batch, b, device, lesion_prob)
                details = (
                    seg_evaluator.patient_score_details(lesion_prob, patient_mask)
                    if patient_mask is not None
                    else None
                )
                if details is not None:
                    patient_score = float(details["score"])
                    patient_logit_lme_score = float(details["logit_lme_score"])
                    patient_pred = int(patient_score >= patient_threshold)
                    patient_logit_lme_pred = int(
                        patient_logit_lme_score >= patient_logit_lme_threshold
                    )
                    row["patient_label"] = int(patient_label)
                    row["patient_score"] = patient_score
                    row["patient_score_contrast"] = (
                        patient_score if contrast_enabled else np.nan
                    )
                    row["patient_score_logit_lme"] = patient_logit_lme_score
                    row["patient_absolute_logit"] = float(
                        details["absolute_logit"]
                    )
                    row["patient_gland_median_logit"] = float(
                        details["gland_median_logit"]
                    )
                    row["patient_contrast_logit"] = float(
                        details["contrast_logit"]
                    )
                    row["patient_calibrated_logit"] = float(
                        details["calibrated_logit"]
                    )
                    row["patient_pred"] = patient_pred
                    row["patient_correct"] = int(patient_pred == int(patient_label))
                    row["patient_confusion"] = (
                        "TP" if patient_label == 1 and patient_pred == 1
                        else "FN" if patient_label == 1
                        else "FP" if patient_pred == 1
                        else "TN"
                    )
                    row["patient_logit_lme_pred"] = patient_logit_lme_pred
                    row["patient_logit_lme_correct"] = int(
                        patient_logit_lme_pred == int(patient_label)
                    )
                    row["patient_logit_lme_confusion"] = (
                        "TP" if patient_label == 1 and patient_logit_lme_pred == 1
                        else "FN" if patient_label == 1
                        else "FP" if patient_logit_lme_pred == 1
                        else "TN"
                    )

            region_info = seg_evaluator.case_region_info(lesion_prob, batch, b, device)
            region_valid_map = None
            region_label_map = None
            region_pred_map = None
            if region_info is not None:
                region_valid_map = region_info["valid_map"].detach().cpu().numpy()
                region_label_map = region_info["label_map"].detach().cpu().numpy()
                region_pred_map = region_info["pred_map"].detach().cpu().numpy()
                zone_ids = sorted(region_info["zone_score"])
                for zone_id in zone_ids:
                    y_true = int(region_info["zone_true"][zone_id])
                    y_score = float(region_info["zone_score"][zone_id])
                    y_pred = int(region_info["zone_pred"][zone_id])
                    self.region_rows.append(
                        {
                            "sample_index": row["sample_index"],
                            "patient_id": patient_id,
                            "source": source,
                            "dataset_label": self.dataset_label,
                            "dataset_csv": self.dataset_csv,
                            "checkpoint_label": self.checkpoint_label,
                            "checkpoint_epoch": self.checkpoint_epoch,
                            "checkpoint_path": self.checkpoint_path,
                            "zone_id": int(zone_id),
                            "region_label": y_true,
                            "region_score": y_score,
                            "region_pred": y_pred,
                            "region_probability_threshold": region_threshold,
                            "region_correct": int(y_true == y_pred),
                            "region_confusion": (
                                "TP" if y_true == 1 and y_pred == 1
                                else "FN" if y_true == 1
                                else "FP" if y_pred == 1
                                else "TN"
                            ),
                        }
                    )

                y_true = np.asarray([region_info["zone_true"][z] for z in zone_ids], dtype=np.int64)
                y_pred = np.asarray([region_info["zone_pred"][z] for z in zone_ids], dtype=np.int64)
                tp = int(((y_true == 1) & (y_pred == 1)).sum())
                fp = int(((y_true == 0) & (y_pred == 1)).sum())
                fn = int(((y_true == 1) & (y_pred == 0)).sum())
                tn = int(((y_true == 0) & (y_pred == 0)).sum())
                sens = self._safe_ratio(tp, tp + fn)
                spec = self._safe_ratio(tn, tn + fp)
                finite = [value for value in (sens, spec) if np.isfinite(value)]
                row.update(
                    {
                        "region_n": len(zone_ids),
                        "region_positive_gt": int(y_true.sum()),
                        "region_positive_pred": int(y_pred.sum()),
                        "region_tp": tp,
                        "region_fp": fp,
                        "region_fn": fn,
                        "region_tn": tn,
                        "region_sensitivity": sens,
                        "region_specificity": spec,
                        "region_bacc": float(np.mean(finite)) if finite else np.nan,
                        "region_f1": self._safe_ratio(2 * tp, 2 * tp + fp + fn),
                    }
                )

            error_reason = self._visualization_reason(row)
            row["case_has_error"] = int(bool(error_reason))
            good_reason = "" if error_reason else self._good_visualization_reason(row)
            row["case_is_good"] = int(bool(good_reason))
            row["good_visualization_reason"] = good_reason
            should_save, reason, bucket = self._should_visualize(row)
            if should_save:
                try:
                    self._save_visualization(
                        batch=batch,
                        b=b,
                        lesion_prob=lesion_prob,
                        region_valid_map=region_valid_map,
                        region_label_map=region_label_map,
                        region_pred_map=region_pred_map,
                        row=row,
                        reason=reason,
                        bucket=bucket,
                    )
                except Exception as exc:  # Keep metrics even if rendering fails.
                    row["visualization_error"] = f"{type(exc).__name__}: {exc}"
                    print(f"Warning: failed to save test visualization for {patient_id}: {exc}")

            self.sample_rows.append(row)

    def finalize(self) -> pd.DataFrame:
        os.makedirs(self.output_dir, exist_ok=True)
        sample_path = os.path.join(self.output_dir, "per_sample_metrics.csv")
        region_path = os.path.join(self.output_dir, "per_region_metrics.csv")
        sample_df = pd.DataFrame(self.sample_rows)
        region_df = pd.DataFrame(self.region_rows, columns=self.REGION_COLUMNS)
        sample_df.to_csv(sample_path, index=False)
        region_df.to_csv(region_path, index=False)
        print(f"Per-sample test metrics saved to: {sample_path}")
        print(f"Per-region test metrics saved to: {region_path}")
        if self.saved_visualizations:
            print(
                f"Saved {self.saved_visualizations} selected test visualizations to: "
                f"{self.visualization_dir}"
            )
            if self.saved_good_visualizations:
                print(
                    f"  Good cases: {self.saved_good_visualizations} -> "
                    f"{os.path.join(self.visualization_dir, 'good')}"
                )
        return sample_df


def get_test_dir() -> str:
    return _cfg("TEST_DIR", os.path.join(_cfg("BASE_DIR", "."), "test"))


def get_test_csv() -> str:
    if _cfg("TEST_CSV", None) is not None:
        return _cfg("TEST_CSV")
    split_dir = _cfg("SPLIT_DIR", os.path.join(_cfg("UNIFIED_DATA_DIR", "."), "splits"))
    for name in [
        "N4_mixed_PROMIS_external_val.csv",
        "task_cls_external_val.csv",
        "external_val.csv",
        "test.csv",
    ]:
        candidate = os.path.join(split_dir, name)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(split_dir, "N4_mixed_PROMIS_external_val.csv")


def get_validation_csv() -> str:
    if _cfg("VAL_CSV", None) is not None:
        return _cfg("VAL_CSV")
    split_dir = _cfg("SPLIT_DIR", os.path.join(_cfg("UNIFIED_DATA_DIR", "."), "splits"))
    return os.path.join(split_dir, "N4_mixed_PUB_TCIA_internal_val.csv")


def get_model_path(test_dir: str) -> str:
    if _cfg("TEST_MODEL_PATH", None) is not None:
        return _cfg("TEST_MODEL_PATH")
    for name in ["best_checkpoint.pth", "best_model.pth"]:
        candidate = os.path.join(test_dir, name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Model file not found. Check TEST_MODEL_PATH or {test_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one checkpoint on one test CSV and export aggregate, per-case, "
            "per-region, and selected visual QA results."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        default=None,
        help="Directory containing best_checkpoint.pth/best_model.pth.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit checkpoint path; overrides checkpoint discovery in --experiment-dir.",
    )
    parser.add_argument(
        "--test-csv",
        default=None,
        help="Explicit test split CSV; defaults to Config.TEST_CSV.",
    )
    parser.add_argument(
        "--validation-csv",
        default=None,
        help=(
            "Validation split used only when a legacy checkpoint has no frozen "
            "threshold bundle."
        ),
    )
    parser.add_argument(
        "--frozen-thresholds",
        default=None,
        help=(
            "Shared validation artifact containing patient calibration and all "
            "decision thresholds. The same file can be reused for internal and "
            "external test."
        ),
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help=(
            "Run validation calibration/threshold selection only, write the "
            "shared frozen artifact, and exit without reading test data."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        default=os.environ.get("RP_DATASET_ROOT"),
        help="Dataset root containing Unified_Dataset (same meaning as RP_DATASET_ROOT).",
    )
    parser.add_argument(
        "--unified-data-dir",
        default=None,
        help="Explicit Unified_Dataset directory; overrides --dataset-root.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Artifact directory. Default: <experiment-dir>/test_artifacts/<dataset>/<checkpoint>.",
    )
    parser.add_argument("--dataset-label", default="external")
    parser.add_argument("--checkpoint-label", default="best")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default=None,
        help="Inference device. Defaults to Config.DEVICE.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--save-images",
        choices=VISUALIZATION_POLICIES,
        default=None,
        help=(
            "none, errors only, good positive cases, representative mixed cases, or all. "
            "Default: Config.TEST_VIS_POLICY (representative)."
        ),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum images for this run; 0 means unlimited.",
    )
    return parser.parse_args()


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    configured_test_name = os.path.basename(
        str(_cfg("TEST_CSV", "B1_PROMIS_external_val.csv"))
    )
    configured_validation_name = os.path.basename(
        str(_cfg("VAL_CSV", "N4_mixed_PUB_TCIA_internal_val.csv"))
    )
    if args.experiment_dir:
        Config.TEST_DIR = os.path.abspath(args.experiment_dir)
    if args.checkpoint:
        Config.TEST_MODEL_PATH = os.path.abspath(args.checkpoint)
        if not args.experiment_dir:
            Config.TEST_DIR = os.path.dirname(Config.TEST_MODEL_PATH)
    if args.dataset_root:
        dataset_root = os.path.abspath(args.dataset_root)
        os.environ["RP_DATASET_ROOT"] = dataset_root
        Config.DATASET_ROOT = dataset_root
        Config.UNIFIED_DATA_DIR = os.path.join(dataset_root, "Unified_Dataset")
        Config.SPLIT_DIR = os.path.join(Config.UNIFIED_DATA_DIR, "splits")
        if not args.test_csv:
            Config.TEST_CSV = os.path.join(Config.SPLIT_DIR, configured_test_name)
        if not args.validation_csv:
            Config.VAL_CSV = os.path.join(
                Config.SPLIT_DIR, configured_validation_name
            )
    if args.unified_data_dir:
        Config.UNIFIED_DATA_DIR = os.path.abspath(args.unified_data_dir)
        Config.SPLIT_DIR = os.path.join(Config.UNIFIED_DATA_DIR, "splits")
        if not args.test_csv:
            Config.TEST_CSV = os.path.join(Config.SPLIT_DIR, configured_test_name)
        if not args.validation_csv:
            Config.VAL_CSV = os.path.join(
                Config.SPLIT_DIR, configured_validation_name
            )
    if args.test_csv:
        Config.TEST_CSV = os.path.abspath(args.test_csv)
    if args.validation_csv:
        Config.VAL_CSV = os.path.abspath(args.validation_csv)
    if args.device:
        Config.DEVICE = args.device
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive.")
        Config.BATCH_SIZE = int(args.batch_size)
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("--num-workers cannot be negative.")
        Config.NUM_WORKERS = int(args.num_workers)


def _resolve_device() -> torch.device:
    requested = str(_cfg("DEVICE", "cpu"))
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested, but it is unavailable in this PyTorch build.")
    return torch.device(requested)


def _preflight_paths(
    model_path: str,
    test_csv: Optional[str] = None,
    validation_csv: Optional[str] = None,
) -> None:
    missing = []
    if not os.path.isfile(model_path):
        missing.append(f"checkpoint: {model_path}")
    if test_csv is not None and not os.path.isfile(test_csv):
        missing.append(f"test CSV: {test_csv}")
    if validation_csv is not None and not os.path.isfile(validation_csv):
        missing.append(f"validation CSV: {validation_csv}")
    unified_dir = str(_cfg("UNIFIED_DATA_DIR", ""))
    if not os.path.isdir(unified_dir):
        missing.append(f"Unified_Dataset: {unified_dir}")
    if missing:
        detail = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Test preflight failed; the following inputs are unavailable:\n"
            f"  - {detail}\n"
            "Pass --dataset-root/--unified-data-dir, --test-csv, and --checkpoint as needed."
        )


def derive_validation_thresholds(
    *,
    model,
    criterion,
    device: torch.device,
    checkpoint_epoch: int,
) -> Tuple[Dict[str, Any], Any, str]:
    """Select every decision/segmentation threshold on validation only."""
    validation_csv = os.path.abspath(get_validation_csv())
    if not os.path.isfile(validation_csv):
        raise FileNotFoundError(
            f"Validation CSV is unavailable: {validation_csv}. Pass --validation-csv."
        )
    validation_loader = DataLoader(
        build_dataset(validation_csv, split="val"),
        batch_size=int(_cfg("BATCH_SIZE", 1)),
        shuffle=False,
        num_workers=int(_cfg("NUM_WORKERS", 0)),
        pin_memory=device.type == "cuda",
    )
    print(f"[Threshold validation CSV] {validation_csv}")
    validation_tracker = utils.validate(
        model,
        validation_loader,
        criterion,
        device,
        checkpoint_epoch,
        save_dir="",
        compute_operating_metrics=True,
        compute_froc_metrics=False,
        compute_confidence_intervals=False,
    )
    thresholds = validation_tracker.build_frozen_thresholds(checkpoint_epoch)
    return thresholds, validation_tracker, validation_csv


def _print_frozen_threshold_audit(flat: Mapping[str, Any]) -> None:
    """Print the primary clinical decisions and retained BAcc thresholds."""
    for label, section in (
        ("Patient", "patient"),
        ("Region", "region"),
        ("TBx ROI", "tbx_roi"),
    ):
        print(
            "[{} threshold] primary={:.6f} ({}) | max-BAcc={:.6f}".format(
                label,
                float(flat[f"{section}_decision_threshold"]),
                flat[f"{section}_decision_selection_rule"],
                float(flat[f"{section}_balanced_accuracy_threshold"]),
            )
        )


def resolve_validation_thresholds(
    checkpoint: Dict[str, Any],
    *,
    model,
    criterion,
    device: torch.device,
    checkpoint_epoch: int,
    output_dir: str,
    frozen_thresholds_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Never select thresholds on test: load them or derive them on validation."""
    audit_path = os.path.join(output_dir, "frozen_validation_thresholds.json")
    shared_path = (
        os.path.abspath(frozen_thresholds_path)
        if frozen_thresholds_path
        else audit_path
    )
    thresholds = utils.load_frozen_thresholds_json(shared_path)
    if frozen_thresholds_path and os.path.isfile(shared_path) and thresholds is None:
        raise ValueError(
            f"Frozen threshold artifact is incompatible or invalid: {shared_path}"
        )
    if thresholds is not None:
        source = "shared validation artifact"
    else:
        thresholds = checkpoint.get("frozen_thresholds")
        if utils.has_frozen_validation_thresholds(thresholds):
            source = "checkpoint"
        else:
            thresholds, _, _ = derive_validation_thresholds(
                model=model,
                criterion=criterion,
                device=device,
                checkpoint_epoch=checkpoint_epoch,
            )
            source = "validation (legacy checkpoint fallback)"

    if checkpoint_epoch > 0 and int(thresholds.get("validation_epoch", 0)) != int(
        checkpoint_epoch
    ):
        raise ValueError(
            "Frozen validation artifact epoch does not match checkpoint: "
            f"artifact={thresholds.get('validation_epoch')} checkpoint={checkpoint_epoch}."
        )
    utils.save_frozen_thresholds_json(shared_path, thresholds)
    if os.path.abspath(audit_path) != os.path.abspath(shared_path):
        utils.save_frozen_thresholds_json(audit_path, thresholds)
    print(f"[Thresholds]  {source} -> {shared_path}")
    flat = utils.flatten_frozen_thresholds(thresholds)
    _print_frozen_threshold_audit(flat)
    if "patient_pooling_alpha" in flat:
        print(
            "[Patient pooling] {} | alpha={:.6f} | beta={:.6f} | "
            "intercept={:.6f} | n={}".format(
                flat.get("patient_pooling_mode", ""),
                float(flat["patient_pooling_alpha"]),
                float(flat["patient_pooling_beta"]),
                float(flat["patient_pooling_intercept"]),
                int(flat.get("patient_pooling_calibration_n", 0)),
            )
        )
    return thresholds


def _summary_row(
    tracker,
    *,
    dataset_label: str,
    test_csv: str,
    checkpoint_label: str,
    checkpoint_epoch: int,
    checkpoint_path: str,
    frozen_thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "test_dataset_label": dataset_label,
        "checkpoint_label": checkpoint_label,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_path": checkpoint_path,
        "test_csv": test_csv,
    }
    row.update(
        utils.flatten_frozen_thresholds(
            frozen_thresholds,
            prefix="test_",
        )
    )
    for key, value in tracker.get_val_dict().items():
        test_key = key.replace("val_", "test_", 1) if key.startswith("val_") else f"test_{key}"
        row[test_key] = value
    return row


def _validation_summary_row(
    tracker,
    *,
    validation_csv: str,
    checkpoint_label: str,
    checkpoint_epoch: int,
    checkpoint_path: str,
    frozen_thresholds: Dict[str, Any],
) -> Dict[str, Any]:
    def confusion_from_rates(
        *,
        positive_n: int,
        negative_n: int,
        sensitivity: float,
        specificity: float,
    ) -> Dict[str, int]:
        """Recover exact integer counts from validation rates and class totals."""
        positive_n = max(0, int(positive_n))
        negative_n = max(0, int(negative_n))
        tp = int(round(float(np.clip(sensitivity, 0.0, 1.0)) * positive_n))
        tn = int(round(float(np.clip(specificity, 0.0, 1.0)) * negative_n))
        tp = min(max(tp, 0), positive_n)
        tn = min(max(tn, 0), negative_n)
        return {
            "tn": tn,
            "fp": negative_n - tn,
            "fn": positive_n - tp,
            "tp": tp,
        }

    flattened = utils.flatten_frozen_thresholds(
        frozen_thresholds,
        prefix="validation_",
    )
    row: Dict[str, Any] = {
        "validation_checkpoint_label": checkpoint_label,
        "validation_checkpoint_epoch": checkpoint_epoch,
        "validation_checkpoint_path": checkpoint_path,
        "validation_csv": validation_csv,
    }
    row.update(flattened)
    row.update(tracker.get_val_dict())
    patient_sens = float(tracker.patient_actual_sens_at_fixed_sens)
    patient_spec = float(tracker.patient_spec_at_fixed_sens)
    region_sens = float(tracker.region_sens_at_fixed_spec)
    region_spec = float(tracker.region_actual_spec_at_fixed_spec)
    patient_primary_confusion = confusion_from_rates(
        positive_n=int(tracker.patient_tp) + int(tracker.patient_fn),
        negative_n=int(tracker.patient_tn) + int(tracker.patient_fp),
        sensitivity=patient_sens,
        specificity=patient_spec,
    )
    region_primary_confusion = confusion_from_rates(
        positive_n=int(tracker.region_tp) + int(tracker.region_fn),
        negative_n=int(tracker.region_tn) + int(tracker.region_fp),
        sensitivity=region_sens,
        specificity=region_spec,
    )
    row.update(
        {
            "validation_patient_primary_sens": patient_sens,
            "validation_patient_primary_spec": patient_spec,
            "validation_patient_primary_bacc": 0.5
            * (patient_sens + patient_spec),
            "validation_patient_primary_threshold": flattened[
                "validation_patient_decision_threshold"
            ],
            "validation_patient_primary_rule": flattened[
                "validation_patient_decision_selection_rule"
            ],
            **{
                f"validation_patient_primary_{key}": value
                for key, value in patient_primary_confusion.items()
            },
            "validation_patient_secondary_balanced_threshold": flattened[
                "validation_patient_balanced_accuracy_threshold"
            ],
            "validation_patient_secondary_balanced_sens": float(
                tracker.patient_sens_at_balanced_accuracy
            ),
            "validation_patient_secondary_balanced_spec": float(
                tracker.patient_spec_at_balanced_accuracy
            ),
            "validation_patient_secondary_balanced_bacc": float(
                tracker.patient_bacc_at_balanced_accuracy
            ),
            "validation_patient_secondary_balanced_tn": int(tracker.patient_tn),
            "validation_patient_secondary_balanced_fp": int(tracker.patient_fp),
            "validation_patient_secondary_balanced_fn": int(tracker.patient_fn),
            "validation_patient_secondary_balanced_tp": int(tracker.patient_tp),
            "validation_region_primary_sens": region_sens,
            "validation_region_primary_spec": region_spec,
            "validation_region_primary_bacc": 0.5
            * (region_sens + region_spec),
            "validation_region_primary_threshold": flattened[
                "validation_region_decision_threshold"
            ],
            "validation_region_primary_rule": flattened[
                "validation_region_decision_selection_rule"
            ],
            **{
                f"validation_region_primary_{key}": value
                for key, value in region_primary_confusion.items()
            },
            "validation_region_secondary_balanced_threshold": flattened[
                "validation_region_balanced_accuracy_threshold"
            ],
            "validation_region_secondary_balanced_sens": float(
                tracker.region_sens_at_balanced_accuracy
            ),
            "validation_region_secondary_balanced_spec": float(
                tracker.region_spec_at_balanced_accuracy
            ),
            "validation_region_secondary_balanced_bacc": float(
                tracker.region_bacc_at_balanced_accuracy
            ),
            "validation_region_secondary_balanced_tn": int(tracker.region_tn),
            "validation_region_secondary_balanced_fp": int(tracker.region_fp),
            "validation_region_secondary_balanced_fn": int(tracker.region_fn),
            "validation_region_secondary_balanced_tp": int(tracker.region_tp),
        }
    )
    return row


def main() -> None:
    args = _parse_args()
    _apply_cli_overrides(args)
    if hasattr(Config, "set_seed"):
        Config.set_seed()

    experiment_dir = get_test_dir()
    model_path = os.path.abspath(get_model_path(experiment_dir))
    validation_csv = os.path.abspath(get_validation_csv())
    test_csv = None if args.validation_only else os.path.abspath(get_test_csv())
    _preflight_paths(
        model_path,
        test_csv=test_csv,
        validation_csv=validation_csv if args.validation_only else None,
    )
    device = _resolve_device()

    checkpoint_label = str(args.checkpoint_label)
    dataset_label = "validation" if args.validation_only else str(args.dataset_label)
    output_dir = args.output_dir or os.path.join(
        experiment_dir,
        str(_cfg("TEST_ARTIFACT_SUBDIR", "test_artifacts")),
        utils.safe_vis_filename(dataset_label),
        utils.safe_vis_filename(checkpoint_label),
    )
    output_dir = os.path.abspath(output_dir)

    print(f"[Experiment]  {_cfg('EXPERIMENT_MODE', 'unknown')}")
    print(f"[Checkpoint]  {model_path}")
    if args.validation_only:
        print(f"[Validation]  {validation_csv}")
    else:
        print(f"[Test CSV]    {test_csv}")
    print(f"[Data root]   {_cfg('UNIFIED_DATA_DIR', '')}")
    print(f"[Device]      {device}")
    print(f"[Output]      {output_dir}")

    model = build_model(device)
    checkpoint = load_model_weights(model, model_path, device)
    restore_checkpoint_pooling_config(model, checkpoint)
    # Build after restoring checkpoint metadata so task switches, fixed weights,
    # curriculum starts, TBx Dice, and patient supervision match training.
    criterion = build_criterion(device)
    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    if checkpoint.get("criterion_state_dict") is not None:
        try:
            criterion.load_state_dict(checkpoint["criterion_state_dict"], strict=False)
        except TypeError:  # pragma: no cover - older torch compatibility
            criterion.load_state_dict(checkpoint["criterion_state_dict"])
    if hasattr(criterion, "set_epoch") and checkpoint_epoch > 0:
        criterion.set_epoch(checkpoint_epoch)

    shared_threshold_path = os.path.abspath(
        args.frozen_thresholds
        or os.path.join(output_dir, "frozen_validation_thresholds.json")
    )
    if args.validation_only:
        frozen_thresholds, validation_tracker, validation_csv = (
            derive_validation_thresholds(
                model=model,
                criterion=criterion,
                device=device,
                checkpoint_epoch=checkpoint_epoch,
            )
        )
        utils.save_frozen_thresholds_json(
            shared_threshold_path,
            frozen_thresholds,
        )
        audit_path = os.path.join(output_dir, "frozen_validation_thresholds.json")
        if os.path.abspath(audit_path) != shared_threshold_path:
            utils.save_frozen_thresholds_json(audit_path, frozen_thresholds)
        validation_summary = _validation_summary_row(
            validation_tracker,
            validation_csv=validation_csv,
            checkpoint_label=checkpoint_label,
            checkpoint_epoch=checkpoint_epoch,
            checkpoint_path=model_path,
            frozen_thresholds=frozen_thresholds,
        )
        summary_path = os.path.join(output_dir, "validation_summary_metrics.csv")
        pd.DataFrame([validation_summary]).to_csv(summary_path, index=False)
        flat = utils.flatten_frozen_thresholds(frozen_thresholds)
        _print_frozen_threshold_audit(flat)
        print("\n" + "=" * 60)
        if "patient_pooling_alpha" in flat:
            print(
                "Validation calibration | alpha={:.6f} beta={:.6f} "
                "intercept={:.6f} patient_threshold={:.6f}".format(
                    float(flat["patient_pooling_alpha"]),
                    float(flat["patient_pooling_beta"]),
                    float(flat["patient_pooling_intercept"]),
                    float(flat["patient_decision_threshold"]),
                )
            )
        else:
            print(
                "Validation patient pooling | mode={} | threshold={:.6f}".format(
                    flat.get("patient_pooling_mode", ""),
                    float(flat["patient_decision_threshold"]),
                )
            )
        print(f"Frozen validation artifact: {shared_threshold_path}")
        print(f"Validation metrics: {summary_path}")
        print("=" * 60)
        return

    test_loader = DataLoader(
        build_dataset(test_csv),
        batch_size=int(_cfg("BATCH_SIZE", 1)),
        shuffle=False,
        num_workers=int(_cfg("NUM_WORKERS", 0)),
        pin_memory=device.type == "cuda",
    )

    frozen_thresholds = resolve_validation_thresholds(
        checkpoint,
        model=model,
        criterion=criterion,
        device=device,
        checkpoint_epoch=checkpoint_epoch,
        output_dir=output_dir,
        frozen_thresholds_path=args.frozen_thresholds,
    )

    visualization_policy = args.save_images or str(
        _cfg("TEST_VIS_POLICY", "representative")
    )
    max_visualizations = (
        int(args.max_images)
        if args.max_images is not None
        else int(_cfg("TEST_VIS_MAX_PER_RUN", 12))
    )
    exporter = TestArtifactExporter(
        output_dir,
        dataset_label=dataset_label,
        dataset_csv=test_csv,
        checkpoint_label=checkpoint_label,
        checkpoint_path=model_path,
        checkpoint_epoch=checkpoint_epoch,
        visualization_policy=visualization_policy,
        max_visualizations=max_visualizations,
        low_dice_threshold=float(_cfg("TEST_VIS_LOW_DICE_THRESHOLD", 0.5)),
        good_dice_threshold=float(_cfg("TEST_VIS_GOOD_DICE_THRESHOLD", 0.8)),
        max_good_visualizations=int(_cfg("TEST_VIS_GOOD_MAX_PER_RUN", 4)),
    )

    tracker = utils.validate(
        model,
        test_loader,
        criterion,
        device,
        checkpoint_epoch,
        save_dir="",
        compute_operating_metrics=bool(_cfg("FINAL_TEST_COMPUTE_OPERATING_METRICS", True)),
        compute_froc_metrics=bool(_cfg("FINAL_TEST_COMPUTE_FROC_METRICS", True)),
        compute_confidence_intervals=bool(
            _cfg("FINAL_TEST_COMPUTE_CONFIDENCE_INTERVALS", True)
        ),
        sample_exporter=exporter,
        frozen_thresholds=frozen_thresholds,
    )
    summary = _summary_row(
        tracker,
        dataset_label=dataset_label,
        test_csv=test_csv,
        checkpoint_label=checkpoint_label,
        checkpoint_epoch=checkpoint_epoch,
        checkpoint_path=model_path,
        frozen_thresholds=frozen_thresholds,
    )
    summary_path = os.path.join(output_dir, "summary_metrics.csv")
    summary.update(
        {
            "test_artifact_dir": output_dir,
            "test_summary_metrics_csv": summary_path,
            "test_per_sample_metrics_csv": os.path.join(
                output_dir, "per_sample_metrics.csv"
            ),
            "test_per_region_metrics_csv": os.path.join(
                output_dir, "per_region_metrics.csv"
            ),
            "test_frozen_thresholds_json": os.path.join(
                output_dir, "frozen_validation_thresholds.json"
            ),
            "test_visualization_dir": exporter.visualization_dir,
            "test_saved_visualizations": int(exporter.saved_visualizations),
            "test_saved_good_visualizations": int(
                exporter.saved_good_visualizations
            ),
        }
    )
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    froc_artifacts = []
    for name, evaluator in (
        ("dense_lesion", tracker.lesion_froc),
        ("target_cspca", tracker.target_cspca_froc),
    ):
        froc_path = os.path.join(output_dir, f"picai_{name}_metrics.json")
        if evaluator.save_full(froc_path):
            froc_artifacts.append(froc_path)
            summary[f"test_{name}_froc_json"] = froc_path
    # Include the final FROC artifact paths in the same aggregate row.
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print(f"Test [{dataset_label}/{checkpoint_label}] | {tracker.print_val_summary()}")
    print(f"Aggregate test metrics saved to: {summary_path}")
    if froc_artifacts:
        print("Official PI-CAI FROC artifacts:")
        for froc_path in froc_artifacts:
            print(f"  - {froc_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
