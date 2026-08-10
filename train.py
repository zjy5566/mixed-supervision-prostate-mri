"""
Training script for the current lesion-risk-map + MIL setting.

Current training contract:
  - The model learns one voxel-level lesion risk map.
  - Only lesion-related losses are logged: PUB dense lesion, TCIA TBx ROI,
    and SBx MIL.
  - TCIA TBx ROI loss follows Config.USE_TBX_POSITIVE_ONLY_LOSS; the current
    B-series default is sampled positive + sampled negative ROI BCE.
  - Grade/gland outputs, losses, metrics, and best-model criteria are removed.
  - The script accepts the new dictionary model/loss outputs, but is tolerant of
    the old 5-output model during transition.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import ProstateUnifiedDataset
from Loss_function import MixedSupervisionLoss

# Prefer the new segmentation+MIL model class. Fall back to the old class name so
# that the script can still run while files are being migrated.
try:
    from model import ProstateSegMILNet as ModelClass
except ImportError:  # pragma: no cover - transition compatibility
    from model import ProstateMixedSupervisionNet as ModelClass

import utils


CHECKPOINT_EXPERIMENT_CONFIG_FIELDS = (
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
    "MIL_POOLING",
    "LME_R",
    "SEG_REGION_POOLING",
    "SEG_PATIENT_POOLING",
    "SEG_EVAL_USE_GLAND_MASK",
    "SEG_EVAL_COMPARE_PATIENT_POOLING",
    "NATIVE_BEST_MODEL_METRIC",
    "COMMON_BEST_MODEL_METRIC",
)


def checkpoint_experiment_config() -> Dict[str, Any]:
    """Serializable provenance needed to reconstruct loss/test semantics."""

    return {
        name: _cfg(name)
        for name in CHECKPOINT_EXPERIMENT_CONFIG_FIELDS
        if hasattr(Config, name)
    }


class Logger:
    """Write console output to both terminal and a log file."""

    def __init__(self, filename: str = "Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def _cfg(name: str, default: Any = None) -> Any:
    return getattr(Config, name, default)


def get_dataset_task(is_train: bool, split: str = "val") -> str:
    if is_train:
        return _cfg("TRAIN_DATASET_TASK", _cfg("TASK", _cfg("DATASET_TASK", "mixed")))
    if split == "test":
        return _cfg(
            "TEST_DATASET_TASK",
            _cfg("VAL_DATASET_TASK", _cfg("TASK", _cfg("DATASET_TASK", "mixed"))),
        )
    return _cfg("VAL_DATASET_TASK", _cfg("TASK", _cfg("DATASET_TASK", "mixed")))


def build_dataset(csv_path: str, is_train: bool, split: str = "val"):
    """Create dataset with split-specific task filtering when configured."""
    task = get_dataset_task(is_train, split=split)
    try:
        return ProstateUnifiedDataset(
            csv_path=csv_path,
            data_root=Config.UNIFIED_DATA_DIR,
            is_train=is_train,
            task=task,
        )
    except TypeError:
        return ProstateUnifiedDataset(
            csv_path=csv_path,
            data_root=Config.UNIFIED_DATA_DIR,
            is_train=is_train,
        )


def build_model(device: torch.device):
    """Instantiate either the new SegMIL model or the old transition model."""
    common_kwargs: Dict[str, Any] = {
        "in_channels": _cfg("IN_CHANNELS", 3),
        "max_zones": _cfg("MAX_ZONES", 20),
    }

    # New model signature.
    try:
        model = ModelClass(
            **common_kwargs,
            base_channels=_cfg("BASE_CHANNELS", 32),
            dropout_rate=_cfg("DROPOUT_RATE", 0.0),
            mil_pooling=_cfg("MIL_POOLING", "lme"),
            lme_r=_cfg("LME_R", 8.0),
            return_dict=True,
        )
    except TypeError:
        # Old model signature. num_grade_classes is ignored by new code paths.
        model = ModelClass(
            in_channels=_cfg("IN_CHANNELS", 3),
            num_grade_classes=_cfg("NUM_CLASSES", 7),
            max_zones=_cfg("MAX_ZONES", 20),
        )
    return model.to(device)


def build_criterion(device: torch.device):
    """Instantiate the lesion loss with Config-controlled TBx/SBx behaviour."""
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
        "tbx_dice_loss_weight": _cfg("TBX_DICE_LOSS_WEIGHT", 0.0),
        "tbx_dice_smooth": _cfg("TBX_DICE_SMOOTH", 1e-5),
        "return_dict": True,
    }
    try:
        criterion = MixedSupervisionLoss(**kwargs)
    except TypeError:
        # Compatibility with the old loss constructor.
        criterion = MixedSupervisionLoss(
            csPCa_threshold=positive_threshold,
            invalid_sys_label=_cfg("INVALID_SYS_LABEL", -1),
            pos_weight_val=_cfg("POS_WEIGHT_VAL", 2.0),
        )
    return criterion.to(device)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    if hasattr(utils, "move_batch_to_device"):
        return utils.move_batch_to_device(batch, device)
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


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


def call_loss(criterion, outputs, batch):
    if hasattr(utils, "call_criterion"):
        loss_output = utils.call_criterion(criterion, outputs, batch)
    else:
        loss_output = criterion(outputs, batch)

    if hasattr(utils, "normalise_loss_output"):
        return utils.normalise_loss_output(loss_output)

    if isinstance(loss_output, dict):
        return loss_output
    raise TypeError("The current utils.py cannot normalise this loss output.")


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))


def _mask_for_visualisation(batch: Dict[str, Any], key: str, b: int, fallback_like: torch.Tensor):
    if key not in batch:
        return fallback_like.detach().cpu().numpy()
    return batch[key][b, 0].detach().cpu().numpy()


def maybe_save_train_visualizations(
    batch: Dict[str, Any],
    outputs: Dict[str, torch.Tensor],
    save_dir: str,
    epoch: int,
    saved_patients: set,
    max_to_save: int,
) -> int:
    """Save a few non-repeated training predictions for visual QA."""
    lesion_logits = outputs.get("lesion_logits")
    if lesion_logits is None or max_to_save <= 0:
        return 0

    vis_dir = os.path.join(save_dir, _cfg("VIS_SUBDIR", "visualizations"), "train", f"epoch_{epoch:03d}")
    os.makedirs(vis_dir, exist_ok=True)

    lesion_probs = torch.sigmoid(lesion_logits.detach())
    saved_count = 0
    batch_size = lesion_probs.size(0)

    for b in range(batch_size):
        if saved_count >= max_to_save:
            break

        pid = batch["pid"][b] if "pid" in batch else f"case_{epoch}_{b}"
        pid = str(pid)
        if pid in saved_patients:
            continue

        empty_like = torch.zeros_like(lesion_probs[b, 0])
        gt_dict = {
            "type": utils.infer_dataset_type(batch, b),
            "lesion_mask": _mask_for_visualisation(batch, "lesion_mask", b, empty_like),
            "target_mask": _mask_for_visualisation(batch, "target_mask", b, empty_like),
            "zones_mask": _mask_for_visualisation(batch, "zones_mask", b, empty_like),
            "sys_labels": batch["sys_labels"][b].detach().cpu().numpy() if "sys_labels" in batch else None,
        }

        filename = f"{saved_count + 1:02d}_{gt_dict['type']}_{_safe_filename(pid)}.png"
        try:
            utils.visualize_predictions(
                input_tensor=batch["input"][b],
                risk_map=lesion_probs[b],
                gt_dict=gt_dict,
                save_path=os.path.join(vis_dir, filename),
                patient_id=pid,
            )
        except Exception as exc:
            print(f"Warning: failed to save training visualization for {pid}: {exc}")
            continue
        saved_patients.add(pid)
        saved_count += 1

    return saved_count


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    epoch: int,
    save_dir: str = "",
    saved_train_vis_patients=None,
):
    model.train()
    if hasattr(criterion, "set_epoch"):
        criterion.set_epoch(epoch)

    tracker = utils.MetricTracker()
    pbar = tqdm(loader, desc="Training")
    should_save_train_vis = (
        bool(_cfg("SAVE_TRAIN_VIS", False))
        and bool(save_dir)
        and int(_cfg("TRAIN_VIS_EVERY_N_EPOCHS", 0)) > 0
        and epoch % int(_cfg("TRAIN_VIS_EVERY_N_EPOCHS", 0)) == 0
    )
    max_train_vis = int(_cfg("TRAIN_VIS_MAX_PER_EPOCH", 0))
    saved_this_epoch = 0
    if saved_train_vis_patients is None:
        saved_train_vis_patients = set()

    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        imgs = batch["input"]
        zones_mask = batch.get("zones_mask", None)

        optimizer.zero_grad(set_to_none=True)

        raw_outputs = model(imgs, zones_mask)
        outputs = unpack_model_output(raw_outputs)
        loss_dict = call_loss(criterion, outputs, batch)
        total_loss = loss_dict["total_loss"]

        if bool(_cfg("MONITOR_PREDICTION_ENTROPY", True)):
            tracker.update_prediction_entropy(outputs, batch)

        if torch.is_tensor(total_loss) and total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=_cfg("GRAD_CLIP_NORM", 12.0))
            optimizer.step()

        tracker.update_losses(loss_dict)
        if should_save_train_vis and saved_this_epoch < max_train_vis:
            saved_this_epoch += maybe_save_train_visualizations(
                batch=batch,
                outputs=outputs,
                save_dir=save_dir,
                epoch=epoch,
                saved_patients=saved_train_vis_patients,
                max_to_save=max_train_vis - saved_this_epoch,
            )
        pbar.set_postfix({"Total Loss": f"{float(total_loss.detach().cpu()):.4f}"})

    return tracker


def select_validation_metric(v_track, metric_name: Optional[str] = None) -> float:
    """Metric for best-model saving. Higher is better."""
    if metric_name is None:
        metric_name = _cfg("BEST_MODEL_METRIC", "lesion_dice")
    metric_name = str(metric_name).lower()

    dense_dice = float(v_track.lesion_dice.avg)
    tbx_auprc = float(getattr(v_track, "tbx_roi_auprc", 0.0))
    tbx_masked_dice = float(
        getattr(getattr(v_track, "tbx_masked_dice", None), "avg", 0.0)
    )
    region_auprc = float(getattr(v_track, "region_auprc", 0.0))
    patient_auprc = float(getattr(v_track, "patient_auprc", 0.0))
    tbx_native = 0.70 * tbx_auprc + 0.30 * tbx_masked_dice

    if metric_name in {"loss", "val_loss", "val_loss_total"}:
        return -float(v_track.loss_total.avg)
    if metric_name in {"lesion_dice", "dice", "val_lesion_dice"}:
        return float(v_track.lesion_dice.avg)
    if metric_name in {
        "lesion_gland_dice",
        "within_prostate_dice",
        "val_lesion_gland_dice",
    }:
        return float(getattr(v_track, "lesion_gland_dice").avg)
    if metric_name in {"target_cspca_dice", "val_target_cspca_dice", "cspca_dice"}:
        return float(getattr(v_track, "target_cspca_dice").avg)
    if metric_name in {"tbx_roi_sens_at_fixed_spec", "val_tbx_roi_sens_at_fixed_spec", "tbx_roi_sens_at_fpr5"}:
        return float(getattr(v_track, "tbx_roi_sens_at_fixed_spec", 0.0))
    if metric_name in {"tbx_roi_auc", "val_tbx_roi_auc"}:
        return float(getattr(v_track, "tbx_roi_auc", 0.0))
    if metric_name in {"tbx_roi_auprc", "val_tbx_roi_auprc"}:
        return float(getattr(v_track, "tbx_roi_auprc", 0.0))
    if metric_name in {"tbx_roi_bacc", "val_tbx_roi_bacc"}:
        return float(getattr(v_track, "tbx_roi_bacc", 0.0))
    if metric_name in {"tbx_masked_dice", "val_tbx_masked_dice"}:
        return tbx_masked_dice
    if metric_name in {"tbx_native", "tbx_auprc_dice"}:
        return tbx_native
    for prefix in ("lesion", "target_cspca"):
        if metric_name.startswith(f"{prefix}_sens_at_fp_per_patient_"):
            return float(getattr(v_track, f"{prefix}_froc_metrics", {}).get(metric_name, 0.0))
        if metric_name.startswith(f"val_{prefix}_sens_at_fp_per_patient_"):
            key = metric_name[4:]
            return float(getattr(v_track, f"{prefix}_froc_metrics", {}).get(key, 0.0))
    if metric_name == "lesion_f1":
        return float(v_track.lesion_f1.avg)
    if metric_name == "patient_sens_at_fixed_spec":
        return float(getattr(v_track, "patient_sens_at_fixed_spec", 0.0))
    if metric_name == "patient_spec_at_fixed_sens":
        return float(getattr(v_track, "patient_spec_at_fixed_sens", 0.0))
    if metric_name == "region_sens_at_fixed_spec":
        return float(getattr(v_track, "region_sens_at_fixed_spec", 0.0))
    if metric_name == "region_spec_at_fixed_sens":
        return float(getattr(v_track, "region_spec_at_fixed_sens", 0.0))
    if metric_name == "region_bacc":
        return float(v_track.region_bacc)
    if metric_name == "region_auc":
        return float(getattr(v_track, "region_auc", 0.0))
    if metric_name in {"region_auprc", "val_region_auprc"}:
        return float(getattr(v_track, "region_auprc", 0.0))
    if metric_name == "patient_bacc":
        return float(getattr(v_track, "patient_bacc", 0.0))
    if metric_name == "patient_auc":
        return float(getattr(v_track, "patient_auc", 0.0))
    if metric_name in {"patient_auprc", "val_patient_auprc"}:
        return float(getattr(v_track, "patient_auprc", 0.0))
    if metric_name == "tbx_sbx_native":
        return 0.50 * tbx_native + 0.50 * region_auprc
    if metric_name == "tbx_sbx_patient_native":
        return (tbx_native + region_auprc + patient_auprc) / 3.0
    if metric_name == "dense_tbx_native":
        return 0.50 * dense_dice + 0.50 * tbx_native
    if metric_name == "dense_region_native":
        return 0.50 * dense_dice + 0.50 * region_auprc
    if metric_name == "dense_tbx_region_native":
        return (dense_dice + tbx_native + region_auprc) / 3.0
    if metric_name == "dense_tbx_region_patient_native":
        return (dense_dice + tbx_native + region_auprc + patient_auprc) / 4.0
    if metric_name == "common_multilevel":
        return (dense_dice + tbx_native + region_auprc + patient_auprc) / 4.0
    if metric_name == "ra_tbx_auprc_composite":
        return (
            0.50 * float(v_track.lesion_dice.avg)
            + 0.50 * float(getattr(v_track, "tbx_roi_auprc", 0.0))
        )
    if metric_name == "ra_sbx_auprc_composite":
        return (
            0.50 * float(v_track.lesion_dice.avg)
            + 0.50 * float(getattr(v_track, "region_auprc", 0.0))
        )
    if metric_name == "biopsy_auprc_composite":
        return (
            0.50 * float(getattr(v_track, "tbx_roi_auprc", 0.0))
            + 0.50 * float(getattr(v_track, "region_auprc", 0.0))
        )
    if metric_name == "mixed_auprc":
        return (
            0.40 * float(v_track.lesion_dice.avg)
            + 0.30 * float(getattr(v_track, "tbx_roi_auprc", 0.0))
            + 0.30 * float(getattr(v_track, "region_auprc", 0.0))
        )
    if metric_name == "clinical_bacc":
        return 0.5 * float(getattr(v_track, "patient_bacc", 0.0)) + 0.5 * float(v_track.region_bacc)
    if metric_name == "composite":
        return (
            0.50 * float(v_track.lesion_dice.avg)
            + 0.25 * float(getattr(v_track, "patient_bacc", 0.0))
            + 0.25 * float(v_track.region_bacc)
        )

    print(f"⚠️ Unknown BEST_MODEL_METRIC='{metric_name}', using lesion_dice.")
    return float(v_track.lesion_dice.avg)


def validate_checkpoint_metric_support(v_track, metric_name: str) -> None:
    """Fail rather than silently selecting epoch 1 from an unavailable metric."""
    metric_name = str(metric_name).lower()
    if metric_name in {"lesion_dice", "dice", "val_lesion_dice"}:
        if int(getattr(v_track, "lesion_dice_n", 0)) <= 0:
            raise RuntimeError(
                "N/native lesion-Dice checkpoint selection has no eligible "
                "dense RA case with a valid gland mask in validation."
            )
    requires_tbx = metric_name in {
        "tbx_native",
        "tbx_auprc_dice",
        "tbx_sbx_native",
        "tbx_sbx_patient_native",
    }
    if requires_tbx:
        tbx_true = np.asarray(
            getattr(v_track, "tbx_roi_true", []), dtype=np.int64
        ).reshape(-1)
        positive_n = int((tbx_true == 1).sum())
        negative_n = int((tbx_true == 0).sum())
        if (
            int(getattr(v_track, "tbx_roi_n", tbx_true.size)) <= 0
            or positive_n <= 0
            or negative_n <= 0
        ):
            raise RuntimeError(
                "TBx-native checkpoint selection requires sampled validation "
                "TBx ROI voxels from both csPCa classes."
            )
        if int(getattr(v_track, "tbx_masked_dice_n", 0)) <= 0:
            raise RuntimeError(
                "TBx-native checkpoint selection requires at least one positive "
                "TBx case for masked Dice."
            )

    requires_region = metric_name in {
        "region_auprc",
        "val_region_auprc",
        "tbx_sbx_native",
        "tbx_sbx_patient_native",
    }
    if requires_region:
        positive_n = int(getattr(v_track, "region_tp", 0)) + int(
            getattr(v_track, "region_fn", 0)
        )
        negative_n = int(getattr(v_track, "region_tn", 0)) + int(
            getattr(v_track, "region_fp", 0)
        )
        if (
            int(getattr(v_track, "region_n", 0)) <= 0
            or positive_n <= 0
            or negative_n <= 0
        ):
            raise RuntimeError(
                "Region-AUPRC checkpoint selection requires validation SBx "
                "regions from both csPCa classes."
            )

    requires_patient = metric_name in {
        "patient_auprc",
        "val_patient_auprc",
        "tbx_sbx_patient_native",
    }
    if requires_patient:
        positive_n = int(getattr(v_track, "patient_tp", 0)) + int(
            getattr(v_track, "patient_fn", 0)
        )
        negative_n = int(getattr(v_track, "patient_tn", 0)) + int(
            getattr(v_track, "patient_fp", 0)
        )
        if (
            int(getattr(v_track, "patient_n", 0)) <= 0
            or positive_n <= 0
            or negative_n <= 0
        ):
            raise RuntimeError(
                "Patient-AUPRC checkpoint selection requires gland-eligible "
                "validation patients from both csPCa classes."
            )


def save_checkpoint(
    path,
    model,
    criterion,
    optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    config_name: str,
    *,
    metric_name: Optional[str] = None,
    checkpoint_role: Optional[str] = None,
    metric_values: Optional[Dict[str, float]] = None,
    frozen_thresholds: Optional[Dict[str, Any]] = None,
):
    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "metric_name": metric_name,
            "checkpoint_role": checkpoint_role,
            "metric_values": metric_values or {},
            "frozen_thresholds": frozen_thresholds,
            "config_name": config_name,
            "mil_pooling": str(_cfg("MIL_POOLING", "lme")),
            "mil_lme_r": float(_cfg("LME_R", 8.0)),
            "seg_region_pooling": str(_cfg("SEG_REGION_POOLING", "top_percent")),
            "experiment_config": checkpoint_experiment_config(),
            "model_state_dict": model.state_dict(),
            "criterion_state_dict": criterion.state_dict() if criterion is not None else None,
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        },
        path,
    )


def get_best_model_path(save_path: str, role: str = "native") -> str:
    """Resolve a native/common best checkpoint with legacy native fallback."""
    role = str(role).lower()
    if role == "common":
        names = ["best_common_checkpoint.pth", "best_common_model.pth"]
    else:
        names = [
            "best_native_checkpoint.pth",
            "best_native_model.pth",
            "best_checkpoint.pth",
            "best_model.pth",
        ]
    for name in names:
        candidate = os.path.join(save_path, name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No {role} best model found in {save_path}")


TOP_K_CHECKPOINT_MANIFEST = "top_native_checkpoints.csv"


def configured_top_k_checkpoints() -> int:
    """Number of model-specific native-score checkpoints retained on disk."""
    try:
        return max(0, int(_cfg("TOP_K_CHECKPOINTS", 5)))
    except (TypeError, ValueError):
        return 5


def top_k_checkpoint_path(save_path: str, epoch: int) -> str:
    return os.path.join(save_path, f"top_native_epoch_{int(epoch):03d}.pth")


def _write_top_k_manifest(save_path: str, records: List[Dict[str, Any]]) -> None:
    rows = []
    for rank, record in enumerate(records, start=1):
        rows.append(
            {
                "rank": rank,
                "checkpoint_epoch": int(record["epoch"]),
                "native_metric_name": str(record["native_metric_name"]),
                "native_metric_value": float(record["native_metric_value"]),
                "common_metric_name": str(record["common_metric_name"]),
                "common_metric_value": float(record["common_metric_value"]),
                "checkpoint_path": os.path.basename(str(record["path"])),
            }
        )
    pd.DataFrame(rows).to_csv(
        os.path.join(save_path, TOP_K_CHECKPOINT_MANIFEST), index=False
    )


def update_top_k_native_checkpoints(
    save_path: str,
    records: List[Dict[str, Any]],
    *,
    model,
    criterion,
    optimizer,
    scheduler,
    epoch: int,
    native_metric: float,
    common_metric: float,
    native_metric_name: str,
    common_metric_name: str,
    config_name: str,
    frozen_thresholds: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Keep only the K highest native-validation-score epoch checkpoints."""
    top_k = configured_top_k_checkpoints()
    score = float(native_metric)
    if top_k <= 0 or not math.isfinite(score):
        return records

    if len(records) >= top_k:
        worst_score = min(float(item["native_metric_value"]) for item in records)
        if score <= worst_score:
            return records

    checkpoint_path = top_k_checkpoint_path(save_path, epoch)
    save_checkpoint(
        checkpoint_path,
        model,
        criterion,
        optimizer,
        scheduler,
        epoch,
        score,
        config_name,
        metric_name=native_metric_name,
        checkpoint_role="native_top_k",
        metric_values={"native": score, "common": float(common_metric)},
        frozen_thresholds=frozen_thresholds,
    )
    candidate = {
        "epoch": int(epoch),
        "native_metric_name": str(native_metric_name),
        "native_metric_value": score,
        "common_metric_name": str(common_metric_name),
        "common_metric_value": float(common_metric),
        "path": checkpoint_path,
    }
    ranked = sorted(
        [*records, candidate],
        key=lambda item: (-float(item["native_metric_value"]), int(item["epoch"])),
    )
    retained = ranked[:top_k]
    retained_paths = {os.path.realpath(str(item["path"])) for item in retained}
    for removed in ranked[top_k:]:
        removed_path = str(removed["path"])
        if os.path.realpath(removed_path) not in retained_paths and os.path.isfile(removed_path):
            os.remove(removed_path)
            print(f"--> Removed checkpoint outside native Top-{top_k}: {removed_path}")

    _write_top_k_manifest(save_path, retained)
    rank = next(
        idx
        for idx, item in enumerate(retained, start=1)
        if int(item["epoch"]) == int(epoch)
    )
    print(
        f"--> Native Top-{top_k} checkpoint saved: rank={rank}, epoch={epoch}, "
        f"{native_metric_name}={score:.4f}"
    )
    return retained


def configured_top_k_checkpoint_files(save_path: str) -> List[Tuple[str, str]]:
    """Read retained Top-K files in rank order, with a filename fallback."""
    manifest_path = os.path.join(save_path, TOP_K_CHECKPOINT_MANIFEST)
    candidates: List[Tuple[str, str]] = []
    if os.path.isfile(manifest_path):
        try:
            manifest = pd.read_csv(manifest_path).sort_values("rank")
            for _, row in manifest.iterrows():
                epoch = int(row["checkpoint_epoch"])
                rank = int(row["rank"])
                stored_path = str(row["checkpoint_path"])
                path = stored_path if os.path.isabs(stored_path) else os.path.join(save_path, stored_path)
                if os.path.isfile(path):
                    candidates.append((f"top_native_{rank:02d}_epoch_{epoch:03d}", path))
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            print(f"Warning: could not read Top-K checkpoint manifest: {exc}")
    if candidates:
        return candidates

    prefix = "top_native_epoch_"
    for filename in sorted(os.listdir(save_path)) if os.path.isdir(save_path) else []:
        if filename.startswith(prefix) and filename.endswith(".pth"):
            candidates.append((os.path.splitext(filename)[0], os.path.join(save_path, filename)))
    return candidates


def collect_final_test_checkpoints(save_path: str) -> List[Tuple[str, str]]:
    """Return labelled checkpoints to evaluate during final test."""
    candidates: List[Tuple[str, str]] = []

    dual_best = bool(_cfg("SAVE_DUAL_BEST_CHECKPOINTS", False))
    if dual_best and bool(_cfg("FINAL_TEST_INCLUDE_NATIVE_BEST", True)):
        try:
            candidates.append(("best_native", get_best_model_path(save_path, "native")))
        except FileNotFoundError:
            pass
    if dual_best and bool(_cfg("FINAL_TEST_INCLUDE_COMMON_BEST", True)):
        try:
            candidates.append(("best_common", get_best_model_path(save_path, "common")))
        except FileNotFoundError:
            pass
    if not dual_best and bool(_cfg("FINAL_TEST_INCLUDE_BEST", True)):
        try:
            candidates.append(("best", get_best_model_path(save_path, "native")))
        except FileNotFoundError:
            pass

    if bool(_cfg("FINAL_TEST_INCLUDE_TOP_K", True)):
        candidates.extend(configured_top_k_checkpoint_files(save_path))

    if bool(_cfg("FINAL_TEST_INCLUDE_LAST", True)):
        path = os.path.join(save_path, "last_checkpoint.pth")
        if os.path.exists(path):
            candidates.append(("last", path))

    if not candidates:
        candidates.append(("best_native", get_best_model_path(save_path, "native")))

    seen_paths = set()
    unique = []
    for label, path in candidates:
        real = os.path.realpath(path)
        if real in seen_paths:
            continue
        seen_paths.add(real)
        unique.append((label, path))
    return unique


def configured_final_test_datasets() -> List[Tuple[str, str]]:
    """Return labelled final-test datasets.

    Config.FINAL_TEST_DATASETS may be a sequence of (label, csv_path) pairs.
    When absent, use the shared internal and PROMIS external cohorts.
    """
    configured = _cfg("FINAL_TEST_DATASETS", None)
    if configured:
        datasets: List[Tuple[str, str]] = []
        for item in configured:
            if isinstance(item, dict):
                label = str(item.get("label", f"dataset_{len(datasets) + 1}"))
                csv_path = item.get("csv") or item.get("path")
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                label = str(item[0])
                csv_path = item[1]
            else:
                continue
            if csv_path:
                datasets.append((label, str(csv_path)))
        if datasets:
            return datasets

    return [
        (
            "internal",
            get_csv_path("COMMON_INTERNAL_TEST_CSV", "common_internal_test.csv"),
        ),
        (
            "external",
            get_csv_path(
                "COMMON_EXTERNAL_TEST_CSV",
                "N4_mixed_PROMIS_external_val.csv",
            ),
        ),
    ]


def load_best_weights(model, criterion, model_path: str, device: torch.device) -> Dict[str, Any]:
    """Load a best checkpoint or plain state_dict into the current model."""
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    cleaned = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value

    model.load_state_dict(cleaned, strict=True)

    if isinstance(checkpoint, dict) and checkpoint.get("criterion_state_dict") is not None:
        try:
            criterion.load_state_dict(checkpoint["criterion_state_dict"], strict=False)
        except TypeError:
            criterion.load_state_dict(checkpoint["criterion_state_dict"])

    return checkpoint if isinstance(checkpoint, dict) else {"model_state_dict": cleaned}


def resolve_validation_thresholds(
    checkpoint: Dict[str, Any],
    *,
    model,
    criterion,
    device: torch.device,
    save_path: str,
    checkpoint_label: str,
    checkpoint_epoch: int,
) -> Dict[str, Any]:
    """Load checkpoint thresholds or derive them once from validation."""
    threshold_dir = os.path.join(save_path, "frozen_thresholds")
    threshold_path = os.path.join(
        threshold_dir,
        f"{utils.safe_vis_filename(checkpoint_label)}.json",
    )
    thresholds = checkpoint.get("frozen_thresholds")
    if utils.has_frozen_validation_thresholds(thresholds):
        source_message = "loaded from checkpoint"
    else:
        thresholds = utils.load_frozen_thresholds_json(threshold_path)
        if thresholds is not None and checkpoint_epoch > 0 and int(
            thresholds.get("validation_epoch", 0)
        ) != int(checkpoint_epoch):
            thresholds = None
        if thresholds is not None:
            source_message = "loaded from legacy checkpoint sidecar"
        else:
            validation_csv = get_csv_path(
                "VAL_CSV",
                "N4_mixed_PUB_TCIA_internal_val.csv",
            )
            print(
                "Checkpoint has no frozen thresholds; deriving them from validation "
                f"only: {validation_csv}"
            )
            validation_loader = DataLoader(
                build_dataset(validation_csv, is_train=False, split="val"),
                batch_size=_cfg("BATCH_SIZE", 1),
                shuffle=False,
                num_workers=_cfg("NUM_WORKERS", 0),
                pin_memory=torch.cuda.is_available(),
            )
            validation_track = utils.validate(
                model,
                validation_loader,
                criterion,
                device,
                checkpoint_epoch,
                save_dir="",
                compute_operating_metrics=True,
                compute_froc_metrics=False,
            )
            thresholds = validation_track.build_frozen_thresholds(checkpoint_epoch)
            source_message = "derived from validation for legacy checkpoint"

    utils.save_frozen_thresholds_json(threshold_path, thresholds)
    print(f"Frozen thresholds ({source_message}): {threshold_path}")
    flat = utils.flatten_frozen_thresholds(thresholds)
    print(
        "  segmentation={:.4f} | patient={:.4f} | TBx ROI={:.4f} | "
        "region={:.4f}".format(
            float(flat["segmentation_threshold"]),
            float(flat["patient_decision_threshold"]),
            float(flat["tbx_roi_decision_threshold"]),
            float(flat["region_decision_threshold"]),
        )
    )
    if "patient_pooling_alpha" in flat:
        print(
            "  patient pooling={} | alpha={:.6f} | beta={:.6f} | "
            "intercept={:.6f} | n={}".format(
                flat.get("patient_pooling_mode", ""),
                float(flat["patient_pooling_alpha"]),
                float(flat["patient_pooling_beta"]),
                float(flat["patient_pooling_intercept"]),
                int(flat.get("patient_pooling_calibration_n", 0)),
            )
        )
    return thresholds


def run_final_test(
    model,
    criterion,
    device: torch.device,
    save_path: str,
    native_metric_name: str,
    common_metric_name: Optional[str] = None,
):
    """Evaluate saved checkpoints on configured final-test datasets."""
    from test import TestArtifactExporter

    test_datasets = configured_final_test_datasets()
    checkpoint_specs = collect_final_test_checkpoints(save_path)

    print("\n" + "=" * 60)
    print("Final test using selected checkpoints")
    print("=" * 60)
    print(f"Test dataset task: {get_dataset_task(is_train=False, split='test')}")
    print("Datasets:")
    for label, csv_path in test_datasets:
        print(f"  {label:<12} {csv_path}")
    print("Checkpoints:")
    for label, path in checkpoint_specs:
        print(f"  {label:<12} {path}")

    compute_operating_metrics = bool(_cfg("FINAL_TEST_COMPUTE_OPERATING_METRICS", True))
    compute_froc_metrics = bool(_cfg("FINAL_TEST_COMPUTE_FROC_METRICS", True))
    compute_confidence_intervals = bool(
        _cfg("FINAL_TEST_COMPUTE_CONFIDENCE_INTERVALS", True)
    )
    ci_checkpoint_labels = {
        str(item)
        for item in _cfg(
            "FINAL_TEST_CI_CHECKPOINT_LABELS",
            ("best", "best_native", "best_common"),
        )
    }
    rows = []
    last_track = None
    for label, checkpoint_path in checkpoint_specs:
        checkpoint_compute_confidence_intervals = (
            compute_confidence_intervals and label in ci_checkpoint_labels
        )
        checkpoint = load_best_weights(model, criterion, checkpoint_path, device)
        checkpoint_epoch = int(checkpoint.get("epoch", 0)) if isinstance(checkpoint, dict) else 0
        # Keep every requested checkpoint role even when two roles point to the
        # same epoch. B/N native/common selectors are both patient AUPRC, so
        # epoch-based de-duplication would otherwise remove named result tables,
        # FROC artifacts, and visualisations (including Top-5 rank 1).
        if hasattr(criterion, "set_epoch") and checkpoint_epoch > 0:
            criterion.set_epoch(checkpoint_epoch)

        frozen_thresholds = resolve_validation_thresholds(
            checkpoint,
            model=model,
            criterion=criterion,
            device=device,
            save_path=save_path,
            checkpoint_label=label,
            checkpoint_epoch=checkpoint_epoch,
        )

        for dataset_label, test_csv in test_datasets:
            artifact_dir = os.path.join(
                save_path,
                str(_cfg("TEST_ARTIFACT_SUBDIR", "test_artifacts")),
                utils.safe_vis_filename(dataset_label),
                utils.safe_vis_filename(label),
            )
            sample_exporter = TestArtifactExporter(
                artifact_dir,
                dataset_label=dataset_label,
                dataset_csv=test_csv,
                checkpoint_label=label,
                checkpoint_path=checkpoint_path,
                checkpoint_epoch=checkpoint_epoch,
                visualization_policy=str(_cfg("TEST_VIS_POLICY", "representative")),
                max_visualizations=int(_cfg("TEST_VIS_MAX_PER_RUN", 12)),
                low_dice_threshold=float(_cfg("TEST_VIS_LOW_DICE_THRESHOLD", 0.5)),
                good_dice_threshold=float(_cfg("TEST_VIS_GOOD_DICE_THRESHOLD", 0.8)),
                max_good_visualizations=int(_cfg("TEST_VIS_GOOD_MAX_PER_RUN", 4)),
            )
            test_loader = DataLoader(
                build_dataset(test_csv, is_train=False, split="test"),
                batch_size=_cfg("BATCH_SIZE", 1),
                shuffle=False,
                num_workers=_cfg("NUM_WORKERS", 0),
                pin_memory=torch.cuda.is_available(),
            )

            print("\n" + "-" * 60)
            print(
                f"Testing dataset={dataset_label} | checkpoint={label} | "
                f"epoch={checkpoint_epoch} | path={checkpoint_path}"
            )
            test_track = utils.validate(
                model,
                test_loader,
                criterion,
                device,
                checkpoint_epoch,
                save_dir="",
                compute_operating_metrics=compute_operating_metrics,
                compute_froc_metrics=compute_froc_metrics,
                compute_confidence_intervals=(
                    checkpoint_compute_confidence_intervals
                ),
                sample_exporter=sample_exporter,
                frozen_thresholds=frozen_thresholds,
            )
            print(f"Test [{dataset_label}/{label}] | {test_track.print_val_summary()}")
            last_track = test_track

            test_log = {
                "test_dataset_label": dataset_label,
                "checkpoint_label": label,
                "checkpoint_epoch": checkpoint_epoch,
                "checkpoint_path": checkpoint_path,
                "is_best_checkpoint": int(label.startswith("best")),
                "is_native_best_checkpoint": int(label in {"best", "best_native"}),
                "is_common_best_checkpoint": int(label == "best_common"),
                "is_top_k_checkpoint": int(label.startswith("top_native_")),
                "best_model_metric_name": checkpoint.get(
                    "metric_name",
                    common_metric_name if label == "best_common" else native_metric_name,
                )
                if isinstance(checkpoint, dict)
                else (common_metric_name if label == "best_common" else native_metric_name),
                "native_best_model_metric_name": native_metric_name,
                "common_best_model_metric_name": common_metric_name,
                "checkpoint_role": checkpoint.get("checkpoint_role", label)
                if isinstance(checkpoint, dict)
                else label,
                "checkpoint_best_metric_value": float(checkpoint.get("best_metric", math.nan))
                if isinstance(checkpoint, dict)
                else math.nan,
                "test_csv": test_csv,
                "test_compute_operating_metrics": int(compute_operating_metrics),
                "test_compute_froc_metrics": int(compute_froc_metrics),
                "test_compute_confidence_intervals": int(
                    checkpoint_compute_confidence_intervals
                ),
                "test_artifact_dir": artifact_dir,
                "test_per_sample_metrics_csv": os.path.join(artifact_dir, "per_sample_metrics.csv"),
                "test_per_region_metrics_csv": os.path.join(artifact_dir, "per_region_metrics.csv"),
            }
            test_log.update(
                utils.flatten_frozen_thresholds(
                    frozen_thresholds,
                    prefix="test_",
                )
            )
            for key, value in test_track.get_val_dict().items():
                test_key = key.replace("val_", "test_", 1) if key.startswith("val_") else f"test_{key}"
                test_log[test_key] = value
            os.makedirs(artifact_dir, exist_ok=True)
            summary_path = os.path.join(artifact_dir, "summary_metrics.csv")
            frozen_path = os.path.join(
                artifact_dir,
                "frozen_validation_thresholds.json",
            )
            utils.save_frozen_thresholds_json(frozen_path, frozen_thresholds)
            pd.DataFrame([test_log]).to_csv(summary_path, index=False)

            froc_paths = {}
            for froc_name, evaluator in (
                ("dense_lesion", test_track.lesion_froc),
                ("target_cspca", test_track.target_cspca_froc),
            ):
                froc_path = os.path.join(
                    artifact_dir,
                    f"picai_{froc_name}_metrics.json",
                )
                if evaluator.save_full(froc_path):
                    froc_paths[froc_name] = froc_path

            test_log.update(
                {
                    "test_summary_metrics_csv": summary_path,
                    "test_frozen_thresholds_json": frozen_path,
                    "test_visualization_dir": sample_exporter.visualization_dir,
                    "test_saved_visualizations": int(
                        sample_exporter.saved_visualizations
                    ),
                    "test_saved_good_visualizations": int(
                        sample_exporter.saved_good_visualizations
                    ),
                    "test_dense_lesion_froc_json": froc_paths.get(
                        "dense_lesion", ""
                    ),
                    "test_target_cspca_froc_json": froc_paths.get(
                        "target_cspca", ""
                    ),
                }
            )
            # Rewrite after adding artifact paths/counts so the per-run summary
            # and the experiment-level test_log expose the same complete row.
            pd.DataFrame([test_log]).to_csv(summary_path, index=False)
            rows.append(test_log)

    test_log_csv = os.path.join(save_path, "test_log.csv")
    pd.DataFrame(rows).to_csv(test_log_csv, index=False)
    print(f"Test log saved to: {test_log_csv}")
    return last_track


def get_experiment_name() -> str:
    if hasattr(Config, "get_experiment_name"):
        return Config.get_experiment_name()
    return str(_cfg("EXP_NAME", "SegMIL_experiment"))


def get_csv_path(name: str, fallback: str) -> str:
    value = _cfg(name, None)
    if value is not None:
        return value
    split_dir = _cfg("SPLIT_DIR", os.path.join(_cfg("UNIFIED_DATA_DIR", "."), "splits"))
    return os.path.join(split_dir, fallback)


def main():
    if hasattr(Config, "set_seed"):
        Config.set_seed()

    device = torch.device(_cfg("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    exp_name = get_experiment_name()
    save_path = os.path.join(_cfg("EXP_DIR", "experiments"), exp_name)
    os.makedirs(save_path, exist_ok=True)

    log_file_path = os.path.join(save_path, "console_output.log")
    sys.stdout = Logger(log_file_path)
    print(f"✅ Console outputs will be saved to: {log_file_path}")
    if hasattr(Config, "show"):
        Config.show()

    train_csv = get_csv_path("TRAIN_CSV", "N4_mixed_PUB_TCIA_train.csv")
    val_csv = get_csv_path("VAL_CSV", "N4_mixed_PUB_TCIA_internal_val.csv")
    test_csv = get_csv_path("TEST_CSV", "N4_mixed_PROMIS_external_val.csv")
    print(f"📄 Train CSV: {train_csv}")
    print(f"📄 Val CSV:   {val_csv}")
    print(f"📄 Test CSV:  {test_csv}")
    print(f"📌 Train dataset task: {get_dataset_task(is_train=True)}")
    print(f"📌 Val dataset task:   {get_dataset_task(is_train=False)}")
    print(f"📌 Test dataset task:  {get_dataset_task(is_train=False, split='test')}")
    print(
        f"📌 Checkpoint retention: native Top-{configured_top_k_checkpoints()} "
        "by validation score; fixed epoch snapshots disabled"
    )

    train_loader = DataLoader(
        build_dataset(train_csv, is_train=True),
        batch_size=_cfg("BATCH_SIZE", 1),
        shuffle=True,
        num_workers=_cfg("NUM_WORKERS", 0),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        build_dataset(val_csv, is_train=False),
        batch_size=_cfg("BATCH_SIZE", 1),
        shuffle=False,
        num_workers=_cfg("NUM_WORKERS", 0),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(device)
    criterion = build_criterion(device)

    em_lr_multiplier = float(_cfg("EM_LR_MULTIPLIER", 10.0))
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": _cfg("LR", 1e-4), "weight_decay": _cfg("WEIGHT_DECAY", 0.0)},
            {"params": criterion.parameters(), "lr": _cfg("LR", 1e-4) * em_lr_multiplier, "weight_decay": 0.0},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_cfg("NUM_EPOCHS", 100))

    best_native_metric = -float("inf")
    best_common_metric = -float("inf")
    use_early_stopping = bool(_cfg("USE_EARLY_STOPPING", False))
    early_stop_counter = 0
    history = []
    top_k_native_checkpoints: List[Dict[str, Any]] = []
    native_metric_name = str(
        _cfg("NATIVE_BEST_MODEL_METRIC", _cfg("BEST_MODEL_METRIC", "lesion_dice"))
    )
    common_metric_name = str(
        _cfg("COMMON_BEST_MODEL_METRIC", "common_multilevel")
    )
    save_dual_best = bool(_cfg("SAVE_DUAL_BEST_CHECKPOINTS", False))
    saved_train_vis_patients = set()
    last_epoch = 0
    last_validation_thresholds: Optional[Dict[str, Any]] = None
    if use_early_stopping:
        print(
            "Early stopping: enabled | "
            f"patience={int(_cfg('EARLY_STOP_PATIENCE', 30))} epochs"
        )
    else:
        print("Early stopping: disabled | training will run for all configured epochs")

    for epoch in range(1, int(_cfg("NUM_EPOCHS", 100)) + 1):
        last_epoch = epoch
        print(f"\nEpoch {epoch}/{_cfg('NUM_EPOCHS', 100)}")
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch)

        if hasattr(criterion, "is_enabled"):
            print(
                "Curriculum/task status | "
                f"Dense: {int(criterion.is_enabled('lesion_dense'))} | "
                f"TBx ROI BCE: {int(criterion.is_enabled('lesion_sparse'))} | "
                f"Sys MIL: {int(criterion.is_enabled('lesion_sys'))} | "
                f"Outside gland: {int(criterion.is_enabled('lesion_outside_gland'))} | "
                f"Patient risk: {int(criterion.is_enabled('lesion_patient'))}"
            )

        train_track = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            save_dir=save_path,
            saved_train_vis_patients=saved_train_vis_patients,
        )
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch)
        val_track = utils.validate(
            model,
            val_loader,
            criterion,
            device,
            epoch,
            save_path,
            compute_operating_metrics=bool(_cfg("VALIDATION_COMPUTE_OPERATING_METRICS", True)),
            compute_froc_metrics=bool(_cfg("VALIDATION_COMPUTE_FROC_METRICS", False)),
            compute_confidence_intervals=bool(
                _cfg("VALIDATION_COMPUTE_CONFIDENCE_INTERVALS", False)
            ),
        )

        print(f"Train | {train_track.print_train_summary()}")
        print(f"Val   | {val_track.print_val_summary()}")
        if bool(_cfg("MONITOR_PREDICTION_ENTROPY", True)):
            print(f"Train Entropy | {train_track.print_entropy_summary()}")
            print(f"Val Entropy   | {val_track.print_entropy_summary()}")

        current_weights = criterion.get_current_weights() if hasattr(criterion, "get_current_weights") else {}
        print("--- Lesion EM / Loss Multipliers ---")
        print(
            f"Dense: {current_weights.get('lesion_dense', 1.0):.3f} | "
            f"TBx ROI BCE: {current_weights.get('lesion_sparse', 1.0):.3f} | "
            f"Sys MIL: {current_weights.get('lesion_sys', 1.0):.3f} | "
            f"Outside gland: {current_weights.get('lesion_outside_gland', 0.0):.3f} | "
            f"Patient risk: {current_weights.get('lesion_patient', 0.0):.3f}"
        )

        validate_checkpoint_metric_support(val_track, native_metric_name)
        validate_checkpoint_metric_support(val_track, common_metric_name)
        native_metric = select_validation_metric(val_track, native_metric_name)
        common_metric = select_validation_metric(val_track, common_metric_name)
        validation_thresholds = val_track.build_frozen_thresholds(epoch)
        last_validation_thresholds = validation_thresholds
        print(
            f"Native selection ({native_metric_name}): {native_metric:.4f} | "
            f"Common selection ({common_metric_name}): {common_metric:.4f}"
        )

        epoch_log = {"epoch": epoch}
        epoch_log.update(train_track.get_train_dict())
        epoch_log.update(val_track.get_val_dict())
        epoch_log.update(
            utils.flatten_frozen_thresholds(
                validation_thresholds,
                prefix="val_frozen_",
            )
        )
        epoch_log.update(
            {
                "best_model_metric_name": native_metric_name,
                "native_best_model_metric_name": native_metric_name,
                "common_best_model_metric_name": common_metric_name,
                "native_selection_score": native_metric,
                "common_selection_score": common_metric,
                "mil_pooling": str(_cfg("MIL_POOLING", "lme")),
                "mil_lme_r": float(_cfg("LME_R", 8.0)),
                "seg_region_pooling": str(
                    _cfg("SEG_REGION_POOLING", "top_percent")
                ),
                "tbx_dice_loss_weight": float(_cfg("TBX_DICE_LOSS_WEIGHT", 0.0)),
                "use_em_weighting": int(_cfg("USE_EM_WEIGHTING", True)),
                "use_logvar_clamp": int(_cfg("USE_LOGVAR_CLAMP", False)),
                "use_curriculum": int(_cfg("USE_CURRICULUM", False)),
                "monitor_prediction_entropy": int(
                    _cfg("MONITOR_PREDICTION_ENTROPY", True)
                ),
                "use_early_stopping": int(use_early_stopping),
                "early_stop_patience": int(_cfg("EARLY_STOP_PATIENCE", 30)),
                "top_k_checkpoints": configured_top_k_checkpoints(),
                "em_lr_multiplier": em_lr_multiplier,
                "lesion_dense_enabled_this_epoch": int(criterion.is_enabled("lesion_dense")) if hasattr(criterion, "is_enabled") else 1,
                "lesion_sparse_enabled_this_epoch": int(criterion.is_enabled("lesion_sparse")) if hasattr(criterion, "is_enabled") else 1,
                "lesion_sys_enabled_this_epoch": int(criterion.is_enabled("lesion_sys")) if hasattr(criterion, "is_enabled") else 1,
                "lesion_outside_gland_enabled_this_epoch": int(criterion.is_enabled("lesion_outside_gland")) if hasattr(criterion, "is_enabled") else 0,
                "lesion_patient_enabled_this_epoch": int(criterion.is_enabled("lesion_patient")) if hasattr(criterion, "is_enabled") else 0,
            }
        )
        history.append(epoch_log)

        log_csv = os.path.join(save_path, "train_log.csv")
        pd.DataFrame(history).to_csv(log_csv, index=False)
        utils.plot_loss_curves(log_csv, os.path.join(save_path, "loss_curve.png"))
        if bool(_cfg("MONITOR_PREDICTION_ENTROPY", True)):
            utils.plot_entropy_curves(
                log_csv,
                os.path.join(save_path, "entropy_curve.png"),
            )

        stop_training = False
        metric_values = {
            "native": native_metric,
            "common": common_metric,
        }
        if native_metric > best_native_metric:
            best_native_metric = native_metric
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(save_path, "best_native_model.pth"))
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
            save_checkpoint(
                os.path.join(save_path, "best_native_checkpoint.pth"),
                model,
                criterion,
                optimizer,
                scheduler,
                epoch,
                best_native_metric,
                exp_name,
                metric_name=native_metric_name,
                checkpoint_role="native_best",
                metric_values=metric_values,
                frozen_thresholds=validation_thresholds,
            )
            # Legacy aliases always mirror the native selector.
            save_checkpoint(
                os.path.join(save_path, "best_checkpoint.pth"),
                model,
                criterion,
                optimizer,
                scheduler,
                epoch,
                best_native_metric,
                exp_name,
                metric_name=native_metric_name,
                checkpoint_role="native_best",
                metric_values=metric_values,
                frozen_thresholds=validation_thresholds,
            )
            print(
                f"--> Native best saved ({native_metric_name}: "
                f"{best_native_metric:.4f})"
            )
        elif use_early_stopping:
            early_stop_counter += 1
            if early_stop_counter >= int(_cfg("EARLY_STOP_PATIENCE", 30)):
                print(f"Early stop triggered at epoch {epoch}")
                stop_training = True

        if save_dual_best and common_metric > best_common_metric:
            best_common_metric = common_metric
            torch.save(model.state_dict(), os.path.join(save_path, "best_common_model.pth"))
            save_checkpoint(
                os.path.join(save_path, "best_common_checkpoint.pth"),
                model,
                criterion,
                optimizer,
                scheduler,
                epoch,
                best_common_metric,
                exp_name,
                metric_name=common_metric_name,
                checkpoint_role="common_best",
                metric_values=metric_values,
                frozen_thresholds=validation_thresholds,
            )
            print(
                f"--> Common best saved ({common_metric_name}: "
                f"{best_common_metric:.4f})"
            )

        top_k_native_checkpoints = update_top_k_native_checkpoints(
            save_path,
            top_k_native_checkpoints,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            native_metric=native_metric,
            common_metric=common_metric,
            native_metric_name=native_metric_name,
            common_metric_name=common_metric_name,
            config_name=exp_name,
            frozen_thresholds=validation_thresholds,
        )

        if stop_training:
            break

        scheduler.step()

    if last_epoch > 0:
        last_checkpoint_path = os.path.join(save_path, "last_checkpoint.pth")
        save_checkpoint(
            last_checkpoint_path,
            model,
            criterion,
            optimizer,
            scheduler,
            last_epoch,
            best_native_metric,
            exp_name,
            metric_name=native_metric_name,
            checkpoint_role="last",
            metric_values={
                "native_best": best_native_metric,
                "common_best": best_common_metric,
            },
            frozen_thresholds=last_validation_thresholds,
        )
        print(f"Last checkpoint saved: {last_checkpoint_path}")

    try:
        run_final_test(
            model,
            criterion,
            device,
            save_path,
            native_metric_name,
            common_metric_name,
        )
    except FileNotFoundError as exc:
        print(f"Warning: final test skipped because the best model was not found: {exc}")


if __name__ == "__main__":
    main()
