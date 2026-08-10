"""
Utility functions for the revised lesion-segmentation + MIL setting.

This version matches the new project setup after 2026-06-10:
  - main voxel-level task: lesion segmentation
  - weak supervision: TCIA TBx-confirmed target lesion ROIs and SBx/PROMIS region labels
  - no grade-prediction metrics, no gland-segmentation metrics

Expected model output from the revised model:
    outputs["lesion_logits"]      : (B, 1, D, H, W)
    outputs["region_logits"]      : (B, max_zones, 1) or (B, max_zones), optional
    outputs["region_valid_mask"]  : (B, max_zones), optional

Expected loss output from the revised loss:
    loss_dict["total_loss"]
    loss_dict["loss_lesion_total"]
    loss_dict["loss_lesion_dense"]
    loss_dict["loss_lesion_sparse"]
    loss_dict["loss_lesion_sys"]
    loss_dict["em_weights"]
    loss_dict["active_tasks"]
    loss_dict["curriculum_status"]
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
from typing import Dict, Mapping, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, average_precision_score, roc_curve
from tqdm import tqdm

from config import Config


# -----------------------------------------------------------------------------
# Basic metric helpers
# -----------------------------------------------------------------------------

def _cfg(name: str, default):
    return getattr(Config, name, default)


def tensor_to_float(value) -> float:
    """Convert tensor / numpy scalar / python number to a safe float."""
    if value is None:
        return 0.0
    if torch.is_tensor(value):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().reshape(-1)[0].cpu().item())
    try:
        return float(value)
    except Exception:
        return 0.0


def binary_entropy_bits_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return elementwise Bernoulli predictive entropy in bits.

    The stable logit-space expression avoids ``log(0)`` and has the intuitive
    range [0, 1]: zero is a confident binary prediction and one is p=0.5.
    The returned tensor is diagnostic only; callers should not add it to the
    optimisation objective.
    """
    logits = logits.float()
    probs = torch.sigmoid(logits)
    entropy_nats = F.softplus(logits) - logits * probs
    return entropy_nats / float(np.log(2.0))


def compute_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    """Batch-mean Dice for binary masks."""
    pred = pred.float().contiguous().view(pred.shape[0], -1)
    target = target.float().contiguous().view(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    denominator = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return float(dice.mean().detach().cpu().item())


def compute_dice_per_case(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> np.ndarray:
    """Per-case Dice values for reporting mean +/- SD."""
    pred = pred.float().contiguous().view(pred.shape[0], -1)
    target = target.float().contiguous().view(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    denominator = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return dice.detach().cpu().numpy().astype(np.float64)


def compute_masked_tbx_dice_per_case(
    probs: torch.Tensor,
    target_mask: torch.Tensor,
    positive_threshold: int,
    prob_threshold: float = 0.5,
    smooth: float = 1e-5,
) -> np.ndarray:
    """Per-case hard Dice inside sampled TBx ROIs for positive TBx cases.

    Voxels where target_mask == 0 are unsampled rather than known negatives and
    are therefore excluded from both prediction and target. Negative-only TBx
    cases are excluded because lesion Dice is undefined as a localisation
    quality measure for those cases; their information remains in ROI AUPRC.
    """
    if probs.shape != target_mask.shape:
        raise ValueError("probs and target_mask must have identical shapes.")
    sampled = target_mask > 0
    target = target_mask >= int(positive_threshold)
    positive_cases = target.reshape(target.size(0), -1).any(dim=1)
    if not bool(positive_cases.any().item()):
        return np.asarray([], dtype=np.float64)

    sampled = sampled[positive_cases]
    target = target[positive_cases]
    pred = (probs[positive_cases] >= float(prob_threshold)) & sampled
    target = target & sampled
    return compute_dice_per_case(pred.float(), target.float(), smooth=smooth)


def compute_topk_dice_per_case(
    prob: torch.Tensor,
    target: torch.Tensor,
    mode: str = "target_volume",
    top_percent: float = 1.0,
    smooth: float = 1e-5,
) -> np.ndarray:
    """Per-case Dice from top-scoring voxels.

    mode="target_volume" uses the ground-truth positive voxel count as k, so it
    is an optimistic localisation upper bound rather than a deployable metric.
    mode="percent" uses a fixed percentage of all voxels.
    """
    prob = prob.float().contiguous().view(prob.shape[0], -1)
    target = target.float().contiguous().view(target.shape[0], -1)
    values = []
    for case_idx in range(prob.shape[0]):
        target_flat = target[case_idx]
        num_voxels = int(target_flat.numel())
        if mode == "target_volume":
            k = int(target_flat.sum().detach().cpu().item())
        elif mode == "percent":
            k = int(np.ceil(num_voxels * max(float(top_percent), 0.0) / 100.0))
        else:
            raise ValueError(f"Unknown top-k Dice mode: {mode}")
        if k <= 0 or num_voxels <= 0:
            continue
        k = min(k, num_voxels)
        top_idx = torch.topk(prob[case_idx], k=k, largest=True, sorted=False).indices
        pred_flat = torch.zeros_like(target_flat)
        pred_flat[top_idx] = 1.0
        intersection = (pred_flat * target_flat).sum()
        denominator = pred_flat.sum() + target_flat.sum()
        dice = (2.0 * intersection + smooth) / (denominator + smooth)
        values.append(float(dice.detach().cpu().item()))
    return np.asarray(values, dtype=np.float64)


def summarise_values(values) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(arr.mean()), "std": std, "n": int(arr.size)}


def configured_target_dice_thresholds() -> Tuple[float, ...]:
    thresholds = _cfg("TARGET_DICE_SWEEP_THRESHOLDS", None)
    if thresholds is None:
        thresholds = np.arange(0.05, 1.00, 0.05)
    if isinstance(thresholds, str):
        thresholds = [float(item.strip()) for item in thresholds.split(",") if item.strip()]
    thresholds = tuple(
        sorted({round(float(th), 6) for th in thresholds if 0.0 <= float(th) <= 1.0})
    )
    return thresholds or (0.5,)


def compute_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds_np = preds.detach().cpu().numpy().astype(np.int64).flatten()
    targets_np = targets.detach().cpu().numpy().astype(np.int64).flatten()
    if targets_np.sum() == 0 and preds_np.sum() == 0:
        return 1.0
    return float(f1_score(targets_np, preds_np, zero_division=0))


def compute_sens(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds_np = preds.detach().cpu().numpy().astype(np.int64).flatten()
    targets_np = targets.detach().cpu().numpy().astype(np.int64).flatten()
    tn, fp, fn, tp = confusion_matrix(targets_np, preds_np, labels=[0, 1]).ravel()
    return float(tp / (tp + fn + 1e-7))


def compute_spec(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds_np = preds.detach().cpu().numpy().astype(np.int64).flatten()
    targets_np = targets.detach().cpu().numpy().astype(np.int64).flatten()
    tn, fp, fn, tp = confusion_matrix(targets_np, preds_np, labels=[0, 1]).ravel()
    return float(tn / (tn + fp + 1e-7))


def safe_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float32)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.0


def safe_auprc(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float32)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return 0.0


def bootstrap_auc_auprc_ci(
    y_true,
    y_score,
    *,
    groups=None,
    confidence_level: Optional[float] = None,
    n_resamples: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Percentile-bootstrap confidence intervals for AUROC and AUPRC.

    When ``groups`` is provided, whole groups are resampled so correlated
    observations from the same patient stay together. Without groups, positive
    and negative observations are resampled separately to keep both classes in
    every bootstrap replicate. Point estimates continue to use all observations;
    this helper only estimates uncertainty.
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true.size != y_score.size:
        raise ValueError("y_true and y_score must have the same number of values.")

    if confidence_level is None:
        confidence_level = float(_cfg("METRIC_CI_CONFIDENCE_LEVEL", 0.95))
    confidence_level = float(confidence_level)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1.")

    if n_resamples is None:
        n_resamples = int(_cfg("METRIC_CI_BOOTSTRAP_SAMPLES", 1000))
    n_resamples = int(n_resamples)
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")

    if seed is None:
        seed = int(_cfg("METRIC_CI_SEED", _cfg("SEED", 42)))

    valid = np.isfinite(y_score) & np.isin(y_true, (0, 1))
    y_true = y_true[valid]
    y_score = y_score[valid]
    group_values = None
    if groups is not None:
        group_values = np.asarray(groups).reshape(-1)
        if group_values.size != valid.size:
            raise ValueError("groups must match y_true and y_score in length.")
        group_values = group_values[valid]

    empty = {
        "auc_ci_low": float("nan"),
        "auc_ci_high": float("nan"),
        "auprc_ci_low": float("nan"),
        "auprc_ci_high": float("nan"),
        "ci_level": confidence_level,
        "ci_bootstrap_valid": 0,
    }
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return empty

    rng = np.random.default_rng(int(seed))
    auc_values = []
    auprc_values = []

    if group_values is None:
        negative_idx = np.flatnonzero(y_true == 0)
        positive_idx = np.flatnonzero(y_true == 1)
        for _ in range(n_resamples):
            sampled_idx = np.concatenate(
                [
                    rng.choice(negative_idx, size=negative_idx.size, replace=True),
                    rng.choice(positive_idx, size=positive_idx.size, replace=True),
                ]
            )
            sample_true = y_true[sampled_idx]
            sample_score = y_score[sampled_idx]
            auc_values.append(float(roc_auc_score(sample_true, sample_score)))
            auprc_values.append(
                float(average_precision_score(sample_true, sample_score))
            )
    else:
        unique_groups = np.unique(group_values)
        if unique_groups.size < 2:
            return empty
        indices_by_group = {
            group: np.flatnonzero(group_values == group) for group in unique_groups
        }
        max_attempts = max(n_resamples * 10, 100)
        attempts = 0
        while len(auc_values) < n_resamples and attempts < max_attempts:
            attempts += 1
            sampled_groups = rng.choice(
                unique_groups,
                size=unique_groups.size,
                replace=True,
            )
            sampled_idx = np.concatenate(
                [indices_by_group[group] for group in sampled_groups]
            )
            sample_true = y_true[sampled_idx]
            if np.unique(sample_true).size < 2:
                continue
            sample_score = y_score[sampled_idx]
            auc_values.append(float(roc_auc_score(sample_true, sample_score)))
            auprc_values.append(
                float(average_precision_score(sample_true, sample_score))
            )

    if not auc_values:
        return empty
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "auc_ci_low": float(np.quantile(auc_values, alpha)),
        "auc_ci_high": float(np.quantile(auc_values, 1.0 - alpha)),
        "auprc_ci_low": float(np.quantile(auprc_values, alpha)),
        "auprc_ci_high": float(np.quantile(auprc_values, 1.0 - alpha)),
        "ci_level": confidence_level,
        "ci_bootstrap_valid": int(len(auc_values)),
    }


FROZEN_THRESHOLD_SCHEMA_VERSION = 5
PATIENT_CONTRAST_POOLING_MODES = frozenset(
    {"logit_lme_contrast", "logit-lme-contrast"}
)
DECISION_THRESHOLD_RULES = frozenset(
    {"max_balanced_accuracy", "fixed_sensitivity", "fixed_specificity"}
)


def is_contrast_patient_pooling(mode: Optional[str] = None) -> bool:
    if mode is None:
        mode = _cfg("SEG_PATIENT_POOLING", "logit_lme")
    return str(mode).strip().lower() in PATIENT_CONTRAST_POOLING_MODES


def canonical_patient_pooling_mode(mode: Optional[str]) -> str:
    """Normalize aliases before matching a checkpoint to its score protocol."""
    value = str(mode or "").strip().lower()
    if value in PATIENT_CONTRAST_POOLING_MODES:
        return "logit_lme_contrast"
    if value in {"logit_lme", "logit-lme"}:
        return "logit_lme"
    return value


def canonical_decision_threshold_rule(rule: Optional[str]) -> str:
    """Normalize the validation rule that defines a primary hard decision."""
    value = str(rule or "").strip().lower().replace("-", "_")
    aliases = {
        "balanced_accuracy": "max_balanced_accuracy",
        "max_bacc": "max_balanced_accuracy",
        "bacc": "max_balanced_accuracy",
        "fixed_sens": "fixed_sensitivity",
        "fixed_spec": "fixed_specificity",
    }
    value = aliases.get(value, value)
    return value if value in DECISION_THRESHOLD_RULES else value


def _finite_probability_threshold(value, default: float = 0.5) -> float:
    """Return a finite probability threshold in [0, 1]."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = float(default)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        value = float(default)
    return value


def binary_metrics_at_threshold(y_true, y_score, threshold: float) -> Dict[str, float]:
    """Evaluate one already-selected binary decision threshold."""
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    y_score = np.asarray(y_score).astype(np.float32).reshape(-1)
    threshold = _finite_probability_threshold(threshold)
    if len(y_true) == 0:
        return {
            "threshold": threshold,
            "sens": 0.0,
            "spec": 0.0,
            "bacc": 0.0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }

    y_pred = (y_score >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if tp + fn > 0 else 0.0
    spec = float(tn / (tn + fp)) if tn + fp > 0 else 0.0
    return {
        "threshold": threshold,
        "sens": sens,
        "spec": spec,
        "bacc": float((sens + spec) / 2.0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_balanced_threshold(y_true, y_score, default: float = 0.5) -> float:
    """Select a validation threshold that maximises balanced accuracy.

    Ties are resolved by choosing the candidate closest to the configured
    default, then the higher threshold. This is deterministic and avoids using
    any test-set prevalence or labels.
    """
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    y_score = np.asarray(y_score).astype(np.float32).reshape(-1)
    default = _finite_probability_threshold(default)
    valid = np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return default

    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=False)
    finite = np.isfinite(thresholds) & (thresholds >= 0.0) & (thresholds <= 1.0)
    if not finite.any():
        return default
    thresholds = thresholds[finite]
    bacc = (tpr[finite] + (1.0 - fpr[finite])) / 2.0
    best = np.flatnonzero(np.isclose(bacc, np.max(bacc), rtol=0.0, atol=1e-12))
    distances = np.abs(thresholds[best] - default)
    closest = best[np.flatnonzero(np.isclose(distances, np.min(distances), atol=1e-12))]
    return _finite_probability_threshold(np.max(thresholds[closest]), default)


def operating_point_metrics(
    y_true,
    y_score,
    fixed_specificity: float = 0.95,
    fixed_sensitivity: float = 0.90,
    threshold_at_fixed_spec: Optional[float] = None,
    threshold_at_fixed_sens: Optional[float] = None,
) -> Dict[str, float]:
    """Return ROC operating-point metrics.

    Without explicit thresholds, select both operating points from validation.
    With explicit thresholds, only evaluate those frozen points on the supplied
    cohort; no threshold search is performed.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float32)
    use_frozen_spec = threshold_at_fixed_spec is not None and np.isfinite(
        float(threshold_at_fixed_spec)
    )
    use_frozen_sens = threshold_at_fixed_sens is not None and np.isfinite(
        float(threshold_at_fixed_sens)
    )

    if use_frozen_spec or use_frozen_sens:
        spec_threshold = _finite_probability_threshold(
            threshold_at_fixed_spec,
            threshold_at_fixed_sens if use_frozen_sens else 0.5,
        )
        sens_threshold = _finite_probability_threshold(
            threshold_at_fixed_sens,
            spec_threshold,
        )
        at_spec = binary_metrics_at_threshold(y_true, y_score, spec_threshold)
        at_sens = binary_metrics_at_threshold(y_true, y_score, sens_threshold)
        return {
            "fixed_spec_target": float(fixed_specificity),
            "sens_at_fixed_spec": at_spec["sens"],
            "actual_spec_at_fixed_spec": at_spec["spec"],
            "actual_fpr_at_fixed_spec": float(1.0 - at_spec["spec"]),
            "threshold_at_fixed_spec": spec_threshold,
            "fixed_sens_target": float(fixed_sensitivity),
            "spec_at_fixed_sens": at_sens["spec"],
            "actual_sens_at_fixed_sens": at_sens["sens"],
            "threshold_at_fixed_sens": sens_threshold,
        }

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {
            "fixed_spec_target": float(fixed_specificity),
            "sens_at_fixed_spec": 0.0,
            "actual_spec_at_fixed_spec": 0.0,
            "actual_fpr_at_fixed_spec": 0.0,
            "threshold_at_fixed_spec": float("nan"),
            "fixed_sens_target": float(fixed_sensitivity),
            "spec_at_fixed_sens": 0.0,
            "actual_sens_at_fixed_sens": 0.0,
            "threshold_at_fixed_sens": float("nan"),
        }

    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=False)
    specificity = 1.0 - fpr
    finite_threshold = (
        np.isfinite(thresholds) & (thresholds >= 0.0) & (thresholds <= 1.0)
    )

    spec_candidates = np.where(
        (specificity >= fixed_specificity) & finite_threshold
    )[0]
    if spec_candidates.size > 0:
        idx_spec = spec_candidates[np.argmax(tpr[spec_candidates])]
    else:
        valid = np.where(finite_threshold)[0]
        idx_spec = int(valid[np.argmax(specificity[valid])])

    sens_candidates = np.where((tpr >= fixed_sensitivity) & finite_threshold)[0]
    if sens_candidates.size > 0:
        idx_sens = sens_candidates[np.argmax(specificity[sens_candidates])]
    else:
        valid = np.where(finite_threshold)[0]
        idx_sens = int(valid[np.argmax(tpr[valid])])

    return {
        "fixed_spec_target": float(fixed_specificity),
        "sens_at_fixed_spec": float(tpr[idx_spec]),
        "actual_spec_at_fixed_spec": float(specificity[idx_spec]),
        "actual_fpr_at_fixed_spec": float(1.0 - specificity[idx_spec]),
        "threshold_at_fixed_spec": float(thresholds[idx_spec]),
        "fixed_sens_target": float(fixed_sensitivity),
        "spec_at_fixed_sens": float(specificity[idx_sens]),
        "actual_sens_at_fixed_sens": float(tpr[idx_sens]),
        "threshold_at_fixed_sens": float(thresholds[idx_sens]),
    }


def frozen_threshold_value(
    thresholds: Optional[Mapping],
    section: str,
    key: str,
    default: float = 0.5,
) -> float:
    """Read one threshold from a versioned validation threshold bundle."""
    if not isinstance(thresholds, Mapping):
        return _finite_probability_threshold(default)
    values = thresholds.get(section, {})
    if not isinstance(values, Mapping):
        return _finite_probability_threshold(default)
    return _finite_probability_threshold(values.get(key), default)


def has_frozen_validation_thresholds(thresholds: Optional[Mapping]) -> bool:
    """Whether a checkpoint contains a usable validation threshold bundle."""
    if not isinstance(thresholds, Mapping):
        return False
    if str(thresholds.get("source", "")).lower() != "validation":
        return False
    try:
        schema_version = int(thresholds.get("schema_version", 0))
    except (TypeError, ValueError):
        return False
    if schema_version < FROZEN_THRESHOLD_SCHEMA_VERSION:
        return False
    dice = thresholds.get("dice", {})
    if not isinstance(dice, Mapping) or "segmentation" not in dice:
        return False
    try:
        segmentation_threshold = float(dice.get("segmentation"))
    except (TypeError, ValueError):
        return False
    if not np.isfinite(segmentation_threshold) or not (
        0.0 <= segmentation_threshold <= 1.0
    ):
        return False
    if str(dice.get("selection_metric", "")) not in {
        "lesion_dice",
        "target_cspca_dice",
        "configured_default_no_dice_gt",
    }:
        return False

    # A patient threshold is reusable only with the score definition on which
    # it was fitted. This prevents contrast-calibrated B checkpoints from being
    # evaluated as if their threshold belonged to the original logit-LME score.
    patient = thresholds.get("patient", {})
    if not isinstance(patient, Mapping):
        return False
    calibration = patient.get("pooling_calibration", {})
    stored_pooling = patient.get("pooling_mode")
    if not stored_pooling and isinstance(calibration, Mapping):
        stored_pooling = calibration.get("mode")
    configured_pooling = canonical_patient_pooling_mode(
        _cfg("SEG_PATIENT_POOLING", "logit_lme")
    )
    if canonical_patient_pooling_mode(stored_pooling) != configured_pooling:
        return False
    expected_rules = {
        "patient": canonical_decision_threshold_rule(
            _cfg("PATIENT_DECISION_THRESHOLD_RULE", "fixed_sensitivity")
        ),
        "patient_logit_lme": canonical_decision_threshold_rule(
            _cfg("PATIENT_DECISION_THRESHOLD_RULE", "fixed_sensitivity")
        ),
        "tbx_roi": canonical_decision_threshold_rule(
            _cfg("TBX_ROI_DECISION_THRESHOLD_RULE", "max_balanced_accuracy")
        ),
        "region": canonical_decision_threshold_rule(
            _cfg("REGION_DECISION_THRESHOLD_RULE", "fixed_specificity")
        ),
    }
    for section_name, expected_rule in expected_rules.items():
        section = thresholds.get(section_name, {})
        if not isinstance(section, Mapping):
            return False
        section_thresholds = {}
        for key in (
            "decision",
            "balanced_accuracy",
            "at_fixed_specificity",
            "at_fixed_sensitivity",
        ):
            try:
                value = float(section.get(key))
            except (TypeError, ValueError):
                return False
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                return False
            section_thresholds[key] = value
        stored_rule = canonical_decision_threshold_rule(
            section.get("decision_selection_rule")
        )
        if stored_rule != expected_rule:
            return False
        selected_key = {
            "max_balanced_accuracy": "balanced_accuracy",
            "fixed_sensitivity": "at_fixed_sensitivity",
            "fixed_specificity": "at_fixed_specificity",
        }.get(stored_rule)
        if selected_key is None or not np.isclose(
            section_thresholds["decision"],
            section_thresholds[selected_key],
            rtol=0.0,
            atol=1e-12,
        ):
            return False

    if is_contrast_patient_pooling(configured_pooling):
        if not isinstance(calibration, Mapping):
            return False
        if not is_contrast_patient_pooling(calibration.get("mode")):
            return False
        for key in ("alpha", "beta", "intercept", "lme_r"):
            try:
                value = float(calibration.get(key))
            except (TypeError, ValueError):
                return False
            if not np.isfinite(value):
                return False
    model_pooling = thresholds.get("model_pooling")
    if model_pooling is not None:
        if not isinstance(model_pooling, Mapping):
            return False
        stored_mil_pooling = str(
            model_pooling.get("sbx_mil_pooling", "")
        ).strip().lower()
        if stored_mil_pooling not in {"mean", "max", "lme"}:
            return False
        if stored_mil_pooling != str(_cfg("MIL_POOLING", "lme")).lower():
            return False
        try:
            stored_lme_r = float(model_pooling.get("lme_r"))
        except (TypeError, ValueError):
            return False
        if not np.isfinite(stored_lme_r) or stored_lme_r <= 0.0:
            return False
        if not np.isclose(
            stored_lme_r,
            float(_cfg("LME_R", 8.0)),
            rtol=0.0,
            atol=1e-12,
        ):
            return False
        stored_region_pooling = str(
            model_pooling.get("canonical_region_pooling", "")
        ).strip().lower()
        if not stored_region_pooling:
            return False
        if stored_region_pooling != str(
            _cfg("SEG_REGION_POOLING", "top_percent")
        ).strip().lower():
            return False
    return True


def flatten_frozen_thresholds(
    thresholds: Optional[Mapping],
    prefix: str = "",
) -> Dict[str, Union[str, int, float]]:
    """Flatten the protocol fields for CSV audit logs."""
    if not has_frozen_validation_thresholds(thresholds):
        return {
            f"{prefix}threshold_source": "missing",
            f"{prefix}threshold_protocol_version": 0,
        }
    fields: Dict[str, Union[str, int, float]] = {
        f"{prefix}threshold_source": str(thresholds.get("source", "validation")),
        f"{prefix}threshold_protocol_version": int(
            thresholds.get("schema_version", FROZEN_THRESHOLD_SCHEMA_VERSION)
        ),
        f"{prefix}threshold_validation_epoch": int(thresholds.get("validation_epoch", 0)),
        f"{prefix}segmentation_threshold": frozen_threshold_value(
            thresholds, "dice", "segmentation"
        ),
    }
    for section in (
        "patient",
        "patient_logit_lme",
        "tbx_roi",
        "region",
        "lesion_voxel",
        "target_cspca_voxel",
    ):
        for key in ("decision", "at_fixed_specificity", "at_fixed_sensitivity"):
            fields[f"{prefix}{section}_{key}_threshold"] = frozen_threshold_value(
                thresholds,
                section,
                key,
                fields[f"{prefix}segmentation_threshold"],
            )
        section_values = thresholds.get(section, {})
        if isinstance(section_values, Mapping):
            if "balanced_accuracy" in section_values:
                fields[
                    f"{prefix}{section}_balanced_accuracy_threshold"
                ] = frozen_threshold_value(
                    thresholds,
                    section,
                    "balanced_accuracy",
                    fields[f"{prefix}{section}_decision_threshold"],
                )
            if "decision_selection_rule" in section_values:
                fields[
                    f"{prefix}{section}_decision_selection_rule"
                ] = canonical_decision_threshold_rule(
                    section_values.get("decision_selection_rule")
                )
    patient = thresholds.get("patient", {})
    calibration = (
        patient.get("pooling_calibration", {})
        if isinstance(patient, Mapping)
        else {}
    )
    stored_pooling = (
        patient.get("pooling_mode", "")
        if isinstance(patient, Mapping)
        else ""
    )
    if not stored_pooling and isinstance(calibration, Mapping):
        stored_pooling = calibration.get("mode", "")
    fields[f"{prefix}patient_pooling_mode"] = canonical_patient_pooling_mode(
        stored_pooling
    )
    model_pooling = thresholds.get("model_pooling", {})
    if isinstance(model_pooling, Mapping) and model_pooling:
        fields[f"{prefix}sbx_mil_pooling_mode"] = str(
            model_pooling.get("sbx_mil_pooling", "")
        )
        fields[f"{prefix}sbx_mil_pooling_lme_r"] = float(
            model_pooling.get("lme_r", 0.0)
        )
        fields[f"{prefix}canonical_region_pooling_mode"] = str(
            model_pooling.get("canonical_region_pooling", "")
        )
    if isinstance(calibration, Mapping) and calibration:
        fields.update(
            {
                f"{prefix}patient_pooling_lme_r": float(calibration.get("lme_r", 0.0)),
                f"{prefix}patient_pooling_alpha": float(calibration.get("alpha", 0.0)),
                f"{prefix}patient_pooling_beta": float(calibration.get("beta", 0.0)),
                f"{prefix}patient_pooling_intercept": float(calibration.get("intercept", 0.0)),
                f"{prefix}patient_pooling_calibration_n": int(calibration.get("n", 0)),
                f"{prefix}patient_pooling_calibration_positive_n": int(
                    calibration.get("positive_n", 0)
                ),
                f"{prefix}patient_pooling_calibration_fitted": int(
                    calibration.get("fitted", 0)
                ),
                f"{prefix}patient_pooling_calibration_status": str(
                    calibration.get("status", "")
                ),
            }
        )
    return fields


def save_frozen_thresholds_json(path: str, thresholds: Mapping) -> None:
    """Persist a human-readable copy beside a checkpoint/test log."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(thresholds, handle, indent=2, sort_keys=True, allow_nan=False)


def load_frozen_thresholds_json(path: str) -> Optional[Dict[str, object]]:
    """Load a previously frozen validation bundle, if it is valid."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            thresholds = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None
    return thresholds if has_frozen_validation_thresholds(thresholds) else None


def configured_fp_per_patient_targets() -> Tuple[float, ...]:
    targets = _cfg("FROC_FP_PER_PATIENT_TARGETS", (0.5, 1.0, 2.0))
    if isinstance(targets, str):
        targets = [float(item.strip()) for item in targets.split(",") if item.strip()]
    targets = tuple(sorted({float(target) for target in targets if float(target) >= 0.0}))
    return targets or (0.5,)


def metric_key_float(value: float) -> str:
    return str(float(value)).replace(".", "p").replace("-", "m")


_PICAI_EVAL_COMPONENTS = None


def _load_picai_eval_components():
    """Load the pinned official PI-CAI evaluator and candidate extractor."""
    global _PICAI_EVAL_COMPONENTS
    if _PICAI_EVAL_COMPONENTS is not None:
        return _PICAI_EVAL_COMPONENTS
    try:
        from picai_eval import Metrics
        from picai_eval.eval import evaluate_case
        from report_guided_annotation import extract_lesion_candidates
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise RuntimeError(
            "Official FROC requires picai_eval==1.4.13. Install "
            "requirements.txt in the inference environment."
        ) from exc
    _PICAI_EVAL_COMPONENTS = (Metrics, evaluate_case, extract_lesion_candidates)
    return _PICAI_EVAL_COMPONENTS


def _installed_package_version(name: str) -> str:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return "unavailable"


class FROCEvaluator:
    """Official PI-CAI lesion matching and FROC aggregation.

    Dense softmax maps are converted to detection maps with the candidate
    extractor recommended in the picai_eval documentation. Matching, IoU,
    connectivity, split/merge handling, and curve construction are delegated to
    the official package rather than reimplemented here.
    """

    def __init__(
        self,
        fp_per_patient_targets: Optional[Tuple[float, ...]] = None,
        min_overlap: Optional[float] = None,
        candidate_threshold: Optional[Union[str, float]] = None,
        candidate_min_voxels: Optional[int] = None,
        candidate_max_lesions: Optional[int] = None,
        candidate_dynamic_threshold_factor: Optional[float] = None,
        candidate_remove_adjacent: Optional[bool] = None,
    ):
        self.fp_per_patient_targets = fp_per_patient_targets or configured_fp_per_patient_targets()
        self.min_overlap = float(
            min_overlap
            if min_overlap is not None
            else _cfg("PICAI_FROC_MIN_OVERLAP", 0.10)
        )
        self.candidate_threshold = (
            candidate_threshold
            if candidate_threshold is not None
            else _cfg("PICAI_CANDIDATE_THRESHOLD", "dynamic-fast")
        )
        self.candidate_min_voxels = int(
            candidate_min_voxels
            if candidate_min_voxels is not None
            else _cfg("PICAI_CANDIDATE_MIN_VOXELS", 10)
        )
        self.candidate_max_lesions = int(
            candidate_max_lesions
            if candidate_max_lesions is not None
            else _cfg("PICAI_CANDIDATE_MAX_LESIONS", 5)
        )
        self.candidate_dynamic_threshold_factor = float(
            candidate_dynamic_threshold_factor
            if candidate_dynamic_threshold_factor is not None
            else _cfg("PICAI_CANDIDATE_DYNAMIC_THRESHOLD_FACTOR", 2.5)
        )
        self.candidate_remove_adjacent = bool(
            candidate_remove_adjacent
            if candidate_remove_adjacent is not None
            else _cfg("PICAI_CANDIDATE_REMOVE_ADJACENT", True)
        )
        if not 0.0 <= self.min_overlap <= 1.0:
            raise ValueError("PI-CAI min_overlap must be in [0, 1].")
        self.lesion_results: Dict[str, list] = {}
        self._next_case_id = 0
        self._metrics = None

    @property
    def num_cases(self) -> int:
        return len(self.lesion_results)

    def _candidate_detection_map(self, prob: np.ndarray) -> np.ndarray:
        _, _, extract_lesion_candidates = _load_picai_eval_components()
        detection_map, _, _ = extract_lesion_candidates(
            np.asarray(prob, dtype=np.float32),
            threshold=self.candidate_threshold,
            min_voxels_detection=self.candidate_min_voxels,
            num_lesions_to_extract=self.candidate_max_lesions,
            dynamic_threshold_factor=self.candidate_dynamic_threshold_factor,
            max_prob_round_decimals=None,
            remove_adjacent_lesion_candidates=self.candidate_remove_adjacent,
        )
        return np.asarray(detection_map, dtype=np.float32)

    def update_from_maps(self, probs: torch.Tensor, targets: torch.Tensor, scoring_masks: Optional[torch.Tensor] = None):
        _, evaluate_case, _ = _load_picai_eval_components()
        probs_np = probs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        masks_np = scoring_masks.detach().cpu().numpy() if scoring_masks is not None else None
        for case_idx in range(probs_np.shape[0]):
            prob = probs_np[case_idx, 0] if probs_np.ndim == 5 else probs_np[case_idx]
            target = targets_np[case_idx, 0] if targets_np.ndim == 5 else targets_np[case_idx]
            prob = np.clip(np.asarray(prob, dtype=np.float32), 0.0, 1.0)
            target = (np.asarray(target) > 0).astype(np.int32)
            if prob.ndim != 3 or target.ndim != 3 or prob.shape != target.shape:
                raise ValueError("PI-CAI FROC requires matching 3D probability and target maps.")
            if masks_np is not None:
                mask = masks_np[case_idx, 0] if masks_np.ndim == 5 else masks_np[case_idx]
                mask = np.asarray(mask).astype(bool)
                if mask.shape != target.shape:
                    raise ValueError("scoring_mask must match target shape for PI-CAI FROC.")
                if mask.any():
                    prob = np.where(mask, prob, 0.0)

            detection_map = self._candidate_detection_map(prob)
            lesion_results, *_ = evaluate_case(
                y_det=detection_map,
                y_true=target,
                min_overlap=self.min_overlap,
                overlap_func="IoU",
                case_confidence_func="max",
                allow_unmatched_candidates_with_minimal_overlap=True,
            )
            self.lesion_results[str(self._next_case_id)] = [
                (int(is_lesion), float(confidence), float(overlap))
                for is_lesion, confidence, overlap in lesion_results
            ]
            self._next_case_id += 1
        self._metrics = None

    def official_metrics(self):
        if self._metrics is None and self.lesion_results:
            Metrics, _, _ = _load_picai_eval_components()
            self._metrics = Metrics(lesion_results=self.lesion_results)
        return self._metrics

    def save_full(self, path: str) -> bool:
        metrics = self.official_metrics()
        if metrics is None:
            return False
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        metrics.save_full(path)
        return True

    def compute_metrics(self, prefix: str = "") -> Dict[str, object]:
        metrics_obj = self.official_metrics()
        if metrics_obj is None:
            sensitivity = np.asarray([], dtype=np.float64)
            fp_per_patient = np.asarray([], dtype=np.float64)
            thresholds = np.asarray([], dtype=np.float64)
            num_gt = 0
        else:
            sensitivity = np.asarray(metrics_obj.lesion_TPR, dtype=np.float64)
            fp_per_patient = np.asarray(metrics_obj.lesion_FPR, dtype=np.float64)
            thresholds = np.asarray(metrics_obj.thresholds, dtype=np.float64)
            num_gt = int(metrics_obj.num_lesions)

        metrics = {
            f"{prefix}froc_n": int(self.num_cases),
            f"{prefix}froc_num_gt": int(num_gt),
            f"{prefix}froc_threshold_count": int(len(thresholds)),
            f"{prefix}froc_protocol": "picai_eval",
            f"{prefix}froc_picai_eval_version": _installed_package_version("picai_eval"),
            f"{prefix}froc_candidate_extractor": "report_guided_annotation",
            f"{prefix}froc_candidate_extractor_version": _installed_package_version(
                "report_guided_annotation"
            ),
            f"{prefix}froc_candidate_threshold": str(self.candidate_threshold),
            f"{prefix}froc_candidate_min_voxels": int(self.candidate_min_voxels),
            f"{prefix}froc_candidate_max_lesions": int(self.candidate_max_lesions),
            f"{prefix}froc_candidate_dynamic_threshold_factor": float(
                self.candidate_dynamic_threshold_factor
            ),
            f"{prefix}froc_candidate_remove_adjacent": int(
                self.candidate_remove_adjacent
            ),
            f"{prefix}froc_overlap_func": "IoU",
            f"{prefix}froc_min_overlap": float(self.min_overlap),
            f"{prefix}froc_connectivity": 26,
            f"{prefix}froc_allow_unmatched_candidates_with_minimal_overlap": 1,
            f"{prefix}froc_one_to_one_matching": 1,
        }
        for target in self.fp_per_patient_targets:
            valid = np.flatnonzero(fp_per_patient <= float(target))
            if valid.size:
                point_idx = int(valid[-1])
                point_sensitivity = float(sensitivity[point_idx])
                point_fp = float(fp_per_patient[point_idx])
                point_threshold = (
                    float(thresholds[point_idx])
                    if point_idx < len(thresholds)
                    else float("nan")
                )
            else:
                point_sensitivity = 0.0
                point_fp = float("nan")
                point_threshold = float("nan")
            key = metric_key_float(target)
            metrics[f"{prefix}sens_at_fp_per_patient_{key}"] = point_sensitivity
            metrics[f"{prefix}actual_fp_per_patient_{key}"] = point_fp
            metrics[f"{prefix}threshold_at_fp_per_patient_{key}"] = point_threshold
        return metrics


def masked_logit_lme_features(
    prob_map: torch.Tensor,
    mask: torch.Tensor,
    lme_r: float = 8.0,
) -> Optional[Dict[str, float]]:
    """Return absolute and gland-relative logit-LME patient features."""
    if mask is None:
        return None
    values = prob_map[mask]
    if values.numel() == 0:
        return None

    values = values.float()
    eps = torch.finfo(values.dtype).eps
    logits = torch.logit(values.clamp(min=eps, max=1.0 - eps))
    r = float(lme_r)
    if not np.isfinite(r) or r <= 0.0:
        raise ValueError("lme_r must be a finite positive number.")
    n_values = logits.new_tensor(float(logits.numel()))
    absolute_logit = (
        torch.logsumexp(logits * r, dim=0) / r - torch.log(n_values) / r
    )
    gland_median_logit = torch.median(logits)
    contrast_logit = absolute_logit - gland_median_logit
    return {
        "absolute_logit": float(absolute_logit.detach().cpu().item()),
        "gland_median_logit": float(gland_median_logit.detach().cpu().item()),
        "contrast_logit": float(contrast_logit.detach().cpu().item()),
        "gland_voxels": int(logits.numel()),
    }


def default_patient_pooling_calibration(lme_r: Optional[float] = None) -> Dict[str, object]:
    """Absolute-LME fallback used until validation fits alpha/beta/intercept."""
    if lme_r is None:
        lme_r = float(_cfg("SEG_RISK_LME_R", _cfg("LME_R", 8.0)))
    return {
        "mode": "logit_lme_contrast",
        "lme_r": float(lme_r),
        "alpha": 1.0,
        "beta": 0.0,
        "intercept": 0.0,
        "regularization_c": float(_cfg("SEG_PATIENT_CALIBRATION_C", 1.0)),
        "n": 0,
        "positive_n": 0,
        "fitted": 0,
        "status": "absolute_lme_fallback",
    }


def apply_patient_pooling_calibration(
    features: Mapping[str, float],
    calibration: Optional[Mapping] = None,
) -> Dict[str, float]:
    """Apply frozen alpha*absolute + beta*contrast + intercept calibration."""
    resolved = default_patient_pooling_calibration(
        (calibration or {}).get("lme_r") if isinstance(calibration, Mapping) else None
    )
    if isinstance(calibration, Mapping):
        resolved.update(calibration)
    calibrated_logit = (
        float(resolved["alpha"]) * float(features["absolute_logit"])
        + float(resolved["beta"]) * float(features["contrast_logit"])
        + float(resolved["intercept"])
    )
    score = float(torch.sigmoid(torch.tensor(calibrated_logit, dtype=torch.float64)).item())
    return {"score": score, "calibrated_logit": float(calibrated_logit)}


def fit_patient_pooling_calibration(
    y_true,
    absolute_logits,
    contrast_logits,
    *,
    lme_r: Optional[float] = None,
    regularization_c: Optional[float] = None,
    max_iter: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """Fit validation-only logistic calibration for absolute/contrast LME."""
    calibration = default_patient_pooling_calibration(lme_r)
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    absolute_logits = np.asarray(absolute_logits, dtype=np.float64).reshape(-1)
    contrast_logits = np.asarray(contrast_logits, dtype=np.float64).reshape(-1)
    if not (y_true.size == absolute_logits.size == contrast_logits.size):
        raise ValueError("Patient labels and pooling features must have equal length.")

    valid = (
        np.isin(y_true, (0, 1))
        & np.isfinite(absolute_logits)
        & np.isfinite(contrast_logits)
    )
    y_true = y_true[valid]
    features = np.column_stack(
        [absolute_logits[valid], contrast_logits[valid]]
    )
    calibration["n"] = int(y_true.size)
    calibration["positive_n"] = int(y_true.sum()) if y_true.size else 0
    if y_true.size < 2 or np.unique(y_true).size < 2:
        calibration["status"] = "fallback_insufficient_validation_classes"
        return calibration

    if regularization_c is None:
        regularization_c = float(_cfg("SEG_PATIENT_CALIBRATION_C", 1.0))
    if max_iter is None:
        max_iter = int(_cfg("SEG_PATIENT_CALIBRATION_MAX_ITER", 1000))
    if seed is None:
        seed = int(_cfg("SEG_PATIENT_CALIBRATION_SEED", _cfg("SEED", 42)))
    regularization_c = float(regularization_c)
    if not np.isfinite(regularization_c) or regularization_c <= 0.0:
        raise ValueError("regularization_c must be a finite positive number.")

    estimator = LogisticRegression(
        C=regularization_c,
        solver="lbfgs",
        max_iter=int(max_iter),
        random_state=int(seed),
    )
    estimator.fit(features, y_true)
    calibration.update(
        {
            "alpha": float(estimator.coef_[0, 0]),
            "beta": float(estimator.coef_[0, 1]),
            "intercept": float(estimator.intercept_[0]),
            "regularization_c": regularization_c,
            "fitted": 1,
            "status": "validation_logistic_regression_l2",
        }
    )
    return calibration


def masked_probability_pool(
    prob_map: torch.Tensor,
    mask: torch.Tensor,
    mode: str = "top_percent",
    top_percent: float = 1.0,
    lme_r: float = 8.0,
) -> Optional[float]:
    """Pool a risk map inside a mask into one case/region probability score.

    ``logit_lme`` is the evaluation equivalent of the training patient loss:
    probabilities are mapped back to logits, normalised LME is applied in logit
    space, and the pooled logit is converted back to a probability.
    """
    values = prob_map[mask]
    if values.numel() == 0:
        return None

    mode = str(mode).lower()
    if mode in {"top_percent", "top-percent", "topk_mean"}:
        # Top-percent pooling is a smoother alternative to max pooling: it
        # rewards compact high-risk regions without letting one noisy voxel win.
        pct = max(float(top_percent), 0.0)
        k = int(np.ceil(values.numel() * pct / 100.0))
        k = min(max(k, 1), values.numel())
        score = torch.topk(values, k=k, largest=True, sorted=False).values.mean()
    elif mode == "max":
        score = values.max()
    elif mode == "mean":
        score = values.mean()
    elif mode == "lme":
        r = float(lme_r)
        score = torch.logsumexp(values * r, dim=0) / r - np.log(float(values.numel())) / r
    elif mode in {"logit_lme", "logit-lme"}:
        r = float(lme_r)
        eps = torch.finfo(values.dtype).eps
        logits = torch.logit(values.clamp(min=eps, max=1.0 - eps))
        pooled_logit = (
            torch.logsumexp(logits * r, dim=0) / r
            - np.log(float(logits.numel())) / r
        )
        score = torch.sigmoid(pooled_logit)
    else:
        raise ValueError(f"Unsupported segmentation risk-map pooling mode: {mode}")

    return float(score.detach().cpu().item())


def binary_confusion_counts(y_true, y_score, threshold: float) -> Dict[str, int]:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float32)
    if len(y_true) == 0:
        return {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    y_pred = (y_score >= float(threshold)).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def lesion_zone_iou_labels(
    lesion_label_map: torch.Tensor,
    zones: torch.Tensor,
    max_zones: int,
    iou_threshold: float,
) -> Dict[int, int]:
    """Map RA/MRI lesion instances to zones using PROMIS-style IoU threshold."""
    lesion_label_map = lesion_label_map.detach()
    zones = zones.detach().round().long()
    labels = torch.unique(lesion_label_map[lesion_label_map > 0])
    zone_labels = {zone_id: 0 for zone_id in range(1, int(max_zones) + 1)}
    if labels.numel() == 0:
        return zone_labels

    for lesion_id in labels.tolist():
        lesion_mask = lesion_label_map == lesion_id
        for zone_id in zone_labels:
            zone_mask = zones == zone_id
            if not bool(zone_mask.any().item()):
                continue
            intersection = torch.logical_and(lesion_mask, zone_mask).sum().item()
            if intersection <= 0:
                continue
            union = torch.logical_or(lesion_mask, zone_mask).sum().item()
            iou = float(intersection) / float(union) if union else 0.0
            if iou > float(iou_threshold):
                zone_labels[zone_id] = 1
    return zone_labels


# -----------------------------------------------------------------------------
# Visualisation helpers
# -----------------------------------------------------------------------------

def safe_vis_filename(value: str) -> str:
    """Sanitise patient identifiers before using them in visualisation names."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))


def mask_for_visualisation(batch: Mapping, key: str, b: int, fallback_like: torch.Tensor):
    """Return a CPU mask array, falling back to zeros when a dataset lacks it."""
    if key not in batch:
        return fallback_like.detach().cpu().numpy()
    value = batch[key][b]
    if value.dim() == fallback_like.dim() + 1:
        value = value[0]
    return value.detach().cpu().numpy()


# -----------------------------------------------------------------------------
# Model/loss compatibility helpers
# -----------------------------------------------------------------------------

def move_batch_to_device(batch: Mapping, device: torch.device) -> Dict:
    """Move tensor values in a batch dictionary to device; leave pid/list/string values unchanged."""
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def unpack_model_output(outputs) -> Dict[str, Optional[torch.Tensor]]:
    """Normalise model output to a dictionary.

    Preferred new output is already a dict. A legacy 5-tuple is also accepted:
        grade_pred, sys_grade_preds, lesion_pred, sys_lesion_preds, gland_pred
    In that case only lesion_pred and sys_lesion_preds are kept.
    """
    if isinstance(outputs, dict):
        return {
            "lesion_logits": outputs.get("lesion_logits"),
            "region_logits": outputs.get("region_logits"),
            "region_valid_mask": outputs.get("region_valid_mask"),
        }

    if isinstance(outputs, (tuple, list)):
        if len(outputs) == 3:
            lesion_logits, region_logits, region_valid_mask = outputs
            return {
                "lesion_logits": lesion_logits,
                "region_logits": region_logits,
                "region_valid_mask": region_valid_mask,
            }
        if len(outputs) >= 5:
            # Legacy model: (grade_pred, sys_grade_preds, lesion_pred, sys_lesion_preds, gland_pred)
            return {
                "lesion_logits": outputs[2],
                "region_logits": outputs[3],
                "region_valid_mask": None,
            }

    raise TypeError(
        "Unsupported model output. Expected dict, compact 3-tuple, or legacy 5-tuple."
    )


def normalise_loss_output(loss_output) -> Dict[str, object]:
    """Normalise loss output to a dictionary for clean logging.

    Preferred new output is a dict. Compact tuple from Loss_function_seg_mil.py is also accepted:
        total, lesion_total, dense, sparse, sys, em_weights, active_tasks, curriculum_status
    """
    if isinstance(loss_output, dict):
        return {
            "total_loss": loss_output.get("total_loss", 0.0),
            "loss_lesion_total": loss_output.get("loss_lesion_total", 0.0),
            "loss_lesion_dense": loss_output.get("loss_lesion_dense", 0.0),
            "loss_lesion_sparse": loss_output.get("loss_lesion_sparse", 0.0),
            "loss_lesion_sparse_bce": loss_output.get("loss_lesion_sparse_bce", 0.0),
            "loss_lesion_sparse_dice": loss_output.get("loss_lesion_sparse_dice", 0.0),
            "loss_lesion_sys": loss_output.get("loss_lesion_sys", 0.0),
            "loss_lesion_outside_gland": loss_output.get("loss_lesion_outside_gland", 0.0),
            "loss_lesion_patient": loss_output.get("loss_lesion_patient", 0.0),
            "em_weights": loss_output.get("em_weights", {}),
            "active_tasks": loss_output.get("active_tasks", {}),
            "curriculum_status": loss_output.get("curriculum_status", {}),
            "loss_counts": loss_output.get("loss_counts", {}),
        }

    if isinstance(loss_output, (tuple, list)):
        # New compact tuple.
        if len(loss_output) == 8:
            total, lesion_total, dense, sparse, sys, em_weights, active_tasks, curriculum_status = loss_output
            return {
                "total_loss": total,
                "loss_lesion_total": lesion_total,
                "loss_lesion_dense": dense,
                "loss_lesion_sparse": sparse,
                "loss_lesion_sparse_bce": sparse,
                "loss_lesion_sparse_dice": 0.0,
                "loss_lesion_sys": sys,
                "loss_lesion_outside_gland": 0.0,
                "loss_lesion_patient": 0.0,
                "em_weights": em_weights,
                "active_tasks": active_tasks,
                "curriculum_status": curriculum_status,
                "loss_counts": {},
            }

        # Legacy 12-tuple from the old multi-task loss. Grade/gland values are ignored.
        if len(loss_output) >= 12:
            return {
                "total_loss": loss_output[0],
                "loss_lesion_total": loss_output[4],
                "loss_lesion_dense": loss_output[5],
                "loss_lesion_sparse": loss_output[6],
                "loss_lesion_sparse_bce": loss_output[6],
                "loss_lesion_sparse_dice": 0.0,
                "loss_lesion_sys": loss_output[7],
                "loss_lesion_outside_gland": 0.0,
                "loss_lesion_patient": 0.0,
                "em_weights": loss_output[9],
                "active_tasks": loss_output[10],
                "curriculum_status": loss_output[11],
                "loss_counts": {},
            }

    raise ValueError(f"Unexpected loss output format: {type(loss_output)}")


def call_criterion(criterion, outputs: Dict[str, torch.Tensor], batch: Mapping):
    """Call either the new criterion(outputs, batch) or a legacy criterion signature."""
    try:
        return criterion(outputs, batch)
    except TypeError:
        # Compatibility with old MixedSupervisionLoss signature.
        lesion_logits = outputs["lesion_logits"]
        region_logits = outputs.get("region_logits")
        return criterion(
            None,
            None,
            lesion_logits,
            region_logits,
            None,
            batch.get("target_mask"),
            batch.get("sys_labels"),
            batch.get("lesion_mask"),
            batch.get("gland_mask"),
            batch.get("has_target"),
            batch.get("has_sys"),
            batch.get("has_lesion"),
            batch.get("has_gland", torch.zeros_like(batch.get("has_lesion"))),
        )


# -----------------------------------------------------------------------------
# Region / patient-level evaluator for MIL segmentation
# -----------------------------------------------------------------------------

class LesionMILEvaluator:
    """Patient-level and region-level cancer/csPCa evaluation from lesion probabilities.

    The evaluator can use either:
      - region_probs from model MIL pooling; or
      - voxel-level lesion probability map pooled manually inside zones.
    """

    def __init__(
        self,
        prob_threshold: float = 0.5,
        positive_threshold: int = 1,
        invalid_sys_label: int = -1,
        fixed_specificity: Optional[float] = None,
        fixed_sensitivity: Optional[float] = None,
    ):
        self.prob_threshold = float(prob_threshold)
        self.positive_threshold = int(positive_threshold)
        self.invalid_sys_label = int(invalid_sys_label)
        if fixed_specificity is None:
            fixed_specificity = _cfg("FIXED_SPECIFICITY_TARGET", 0.95)
        if fixed_sensitivity is None:
            fixed_sensitivity = _cfg("FIXED_SENSITIVITY_TARGET", 0.90)
        self.fixed_specificity = float(fixed_specificity)
        self.fixed_sensitivity = float(fixed_sensitivity)

        self.patient_true = []
        self.patient_score = []
        self.region_true = []
        self.region_score = []

    def update_from_batch(
        self,
        lesion_probs: torch.Tensor,
        batch: Mapping,
        region_logits: Optional[torch.Tensor] = None,
        region_valid_mask: Optional[torch.Tensor] = None,
    ):
        """Update evaluator from a batch.

        lesion_probs: (B, 1, D, H, W)
        region_logits: optional (B, Z, 1) or (B, Z)
        """
        B = lesion_probs.size(0)
        device = lesion_probs.device

        if region_logits is not None:
            region_probs = torch.sigmoid(region_logits)
            if region_probs.dim() == 3 and region_probs.size(-1) == 1:
                region_probs = region_probs.squeeze(-1)
        else:
            region_probs = None

        for b in range(B):
            has_sys = bool(batch.get("has_sys", torch.zeros(B))[b].item() > 0)
            has_target = bool(batch.get("has_target", torch.zeros(B))[b].item() > 0)
            # Patient-level GT is biopsy-based. PUB dense lesion masks are used
            # for lesion Dice only and must not enter patient BAcc/AUC.
            if not (has_sys or has_target):
                continue

            # Patient-level GT: positive if any available biopsy supervision is positive.
            patient_gt = 0
            if has_sys and "sys_labels" in batch:
                labels = batch["sys_labels"][b].to(device)
                valid = labels != self.invalid_sys_label
                if valid.any() and labels[valid].max().item() >= self.positive_threshold:
                    patient_gt = 1
            if has_target and "target_mask" in batch:
                target_mask = batch["target_mask"][b].to(device)
                if target_mask.max().item() >= self.positive_threshold:
                    patient_gt = 1

            patient_score = self._patient_score(lesion_probs[b, 0], batch, b, device)
            self.patient_true.append(patient_gt)
            self.patient_score.append(patient_score)

            # Region-level GT/pred only exists for systematic biopsy samples.
            if has_sys and "sys_labels" in batch:
                labels = batch["sys_labels"][b].to(device)
                for z_idx in range(labels.numel()):
                    z_label = int(labels[z_idx].item())
                    if z_label == self.invalid_sys_label:
                        continue

                    if region_valid_mask is not None:
                        if not bool(region_valid_mask[b, z_idx].item()):
                            continue

                    y_true = int(z_label >= self.positive_threshold)

                    if region_probs is not None and z_idx < region_probs.shape[1]:
                        y_score = float(region_probs[b, z_idx].detach().cpu().item())
                    else:
                        # Fallback: max pooling inside the zone mask.
                        zones_mask = batch["zones_mask"][b, 0].to(device)
                        voxels = zones_mask.round().long() == (z_idx + 1)
                        if not voxels.any():
                            continue
                        y_score = float(lesion_probs[b, 0][voxels].max().detach().cpu().item())

                    self.region_true.append(y_true)
                    self.region_score.append(y_score)

    def _patient_score(self, lesion_prob_3d: torch.Tensor, batch: Mapping, b: int, device: torch.device) -> float:
        """Patient-level score: max lesion probability inside gland if available, otherwise whole image."""
        if "gland_mask" in batch and batch["gland_mask"][b].numel() > 0:
            gland_mask = batch["gland_mask"][b, 0].to(device) > 0
            if gland_mask.any():
                return float(lesion_prob_3d[gland_mask].max().detach().cpu().item())
        return float(lesion_prob_3d.max().detach().cpu().item())

    def _binary_metrics(self, y_true, y_score, threshold: float) -> Dict[str, float]:
        y_true = np.asarray(y_true).astype(np.int64)
        y_score = np.asarray(y_score).astype(np.float32)
        if len(y_true) == 0:
            return {
                "sens": 0.0,
                "spec": 0.0,
                "bacc": 0.0,
                "auc": 0.0,
                "auprc": 0.0,
                "n": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "tp": 0,
                **operating_point_metrics(
                    y_true,
                    y_score,
                    fixed_specificity=self.fixed_specificity,
                    fixed_sensitivity=self.fixed_sensitivity,
                ),
            }
        y_pred = (y_score >= threshold).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        return {
            "sens": float(sens),
            "spec": float(spec),
            "bacc": float((sens + spec) / 2.0),
            "auc": safe_auc(y_true, y_score),
            "auprc": safe_auprc(y_true, y_score),
            "n": int(len(y_true)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            **operating_point_metrics(
                y_true,
                y_score,
                fixed_specificity=self.fixed_specificity,
                fixed_sensitivity=self.fixed_sensitivity,
            ),
        }

    def compute_metrics(self) -> Dict[str, float]:
        patient = self._binary_metrics(self.patient_true, self.patient_score, self.prob_threshold)
        region = self._binary_metrics(self.region_true, self.region_score, self.prob_threshold)
        return {
            "patient_sens": patient["sens"],
            "patient_spec": patient["spec"],
            "patient_bacc": patient["bacc"],
            "patient_auc": patient["auc"],
            "patient_auprc": patient["auprc"],
            "patient_n": patient["n"],
            "patient_tn": patient["tn"],
            "patient_fp": patient["fp"],
            "patient_fn": patient["fn"],
            "patient_tp": patient["tp"],
            "patient_fixed_spec_target": patient["fixed_spec_target"],
            "patient_sens_at_fixed_spec": patient["sens_at_fixed_spec"],
            "patient_actual_spec_at_fixed_spec": patient["actual_spec_at_fixed_spec"],
            "patient_actual_fpr_at_fixed_spec": patient["actual_fpr_at_fixed_spec"],
            "patient_threshold_at_fixed_spec": patient["threshold_at_fixed_spec"],
            "patient_fixed_sens_target": patient["fixed_sens_target"],
            "patient_spec_at_fixed_sens": patient["spec_at_fixed_sens"],
            "patient_actual_sens_at_fixed_sens": patient["actual_sens_at_fixed_sens"],
            "patient_threshold_at_fixed_sens": patient["threshold_at_fixed_sens"],
            "region_sens": region["sens"],
            "region_spec": region["spec"],
            "region_bacc": region["bacc"],
            "region_auc": region["auc"],
            "region_auprc": region["auprc"],
            "region_n": region["n"],
            "region_tn": region["tn"],
            "region_fp": region["fp"],
            "region_fn": region["fn"],
            "region_tp": region["tp"],
            "region_fixed_spec_target": region["fixed_spec_target"],
            "region_sens_at_fixed_spec": region["sens_at_fixed_spec"],
            "region_actual_spec_at_fixed_spec": region["actual_spec_at_fixed_spec"],
            "region_actual_fpr_at_fixed_spec": region["actual_fpr_at_fixed_spec"],
            "region_threshold_at_fixed_spec": region["threshold_at_fixed_spec"],
            "region_fixed_sens_target": region["fixed_sens_target"],
            "region_spec_at_fixed_sens": region["spec_at_fixed_sens"],
            "region_actual_sens_at_fixed_sens": region["actual_sens_at_fixed_sens"],
            "region_threshold_at_fixed_sens": region["threshold_at_fixed_sens"],
        }


# Backward-compatible alias for older imports.
BalancedAccuracyEvaluator = LesionMILEvaluator


class SegRiskMapEvaluator:
    """Patient/region metrics derived from segmentation risk maps.

    Patient labels use explicit or biopsy-derived csPCa targets. Dense RA masks
    without a matching patient pathology label are excluded from patient-level
    metrics. Region labels come from systematic biopsy labels.
    """

    def __init__(
        self,
        prob_threshold: float = 0.5,
        positive_threshold: int = 1,
        fixed_specificity: Optional[float] = None,
        fixed_sensitivity: Optional[float] = None,
        patient_pooling: Optional[str] = None,
        region_pooling: Optional[str] = None,
        top_percent: Optional[float] = None,
        lme_r: Optional[float] = None,
        max_zones: Optional[int] = None,
        invalid_sys_label: Optional[int] = None,
        use_gland_mask_for_patient_pooling: Optional[bool] = None,
        patient_threshold: Optional[float] = None,
        region_threshold: Optional[float] = None,
        tbx_roi_threshold: Optional[float] = None,
        patient_operating_thresholds: Optional[Mapping] = None,
        patient_logit_lme_threshold: Optional[float] = None,
        patient_logit_lme_operating_thresholds: Optional[Mapping] = None,
        patient_pooling_calibration: Optional[Mapping] = None,
        region_operating_thresholds: Optional[Mapping] = None,
        select_validation_thresholds: bool = False,
        compute_confidence_intervals: bool = False,
    ):
        self.prob_threshold = _finite_probability_threshold(prob_threshold)
        self.patient_threshold = _finite_probability_threshold(
            patient_threshold, self.prob_threshold
        )
        self.region_threshold = _finite_probability_threshold(
            region_threshold, self.prob_threshold
        )
        self.tbx_roi_threshold = _finite_probability_threshold(
            tbx_roi_threshold, self.prob_threshold
        )
        self.patient_operating_thresholds = dict(patient_operating_thresholds or {})
        self.patient_logit_lme_threshold = _finite_probability_threshold(
            patient_logit_lme_threshold, self.prob_threshold
        )
        self.patient_logit_lme_operating_thresholds = dict(
            patient_logit_lme_operating_thresholds or {}
        )
        self.region_operating_thresholds = dict(region_operating_thresholds or {})
        self.select_validation_thresholds = bool(select_validation_thresholds)
        self.compute_confidence_intervals = bool(compute_confidence_intervals)
        self.positive_threshold = int(positive_threshold)
        if fixed_specificity is None:
            fixed_specificity = _cfg("FIXED_SPECIFICITY_TARGET", 0.95)
        if fixed_sensitivity is None:
            fixed_sensitivity = _cfg("FIXED_SENSITIVITY_TARGET", 0.90)
        self.fixed_specificity = float(fixed_specificity)
        self.fixed_sensitivity = float(fixed_sensitivity)
        self.patient_pooling = str(
            patient_pooling or _cfg("SEG_PATIENT_POOLING", "top_percent")
        ).lower()
        self.region_pooling = str(region_pooling or _cfg("SEG_REGION_POOLING", "top_percent"))
        self.top_percent = float(top_percent if top_percent is not None else _cfg("SEG_RISK_TOP_PERCENT", 1.0))
        self.lme_r = float(lme_r if lme_r is not None else _cfg("SEG_RISK_LME_R", _cfg("LME_R", 8.0)))
        self.patient_pooling_calibration = default_patient_pooling_calibration(
            self.lme_r
        )
        if isinstance(patient_pooling_calibration, Mapping):
            self.patient_pooling_calibration.update(patient_pooling_calibration)
        if is_contrast_patient_pooling(self.patient_pooling):
            self.lme_r = float(
                self.patient_pooling_calibration.get("lme_r", self.lme_r)
            )
        else:
            self.patient_pooling_calibration.update(
                {
                    "mode": canonical_patient_pooling_mode(
                        self.patient_pooling
                    ),
                    "alpha": float("nan"),
                    "beta": float("nan"),
                    "intercept": float("nan"),
                    "n": 0,
                    "positive_n": 0,
                    "fitted": 0,
                    "status": "not_applicable_original_pooling",
                }
            )
        self.use_gland_mask_for_patient_pooling = bool(
            use_gland_mask_for_patient_pooling
            if use_gland_mask_for_patient_pooling is not None
            else _cfg("SEG_EVAL_USE_GLAND_MASK", True)
        )
        self.max_zones = int(max_zones if max_zones is not None else _cfg("MAX_ZONES", 20))
        self.invalid_sys_label = int(
            invalid_sys_label
            if invalid_sys_label is not None
            else _cfg("INVALID_SYS_LABEL", -1)
        )

        self.patient_true = []
        self.patient_score = []
        self.patient_logit_lme_score = []
        self.patient_group = []
        self.patient_absolute_logit = []
        self.patient_gland_median_logit = []
        self.patient_contrast_logit = []
        self.patient_calibrated_logit = []
        self.region_true = []
        self.region_score = []
        self.region_group = []
        self._next_case_group = 0

    def update_from_batch(self, lesion_probs: torch.Tensor, batch: Mapping):
        B = lesion_probs.size(0)
        device = lesion_probs.device

        for b in range(B):
            group_id = self._next_case_group
            self._next_case_group += 1
            prob_3d = lesion_probs[b, 0]

            patient_label = self._patient_label(batch, b, device)
            if patient_label is not None:
                patient_mask = self._patient_score_mask(batch, b, device, prob_3d)
                if patient_mask is not None:
                    details = self.patient_score_details(prob_3d, patient_mask)
                    if details is not None:
                        self.patient_true.append(int(patient_label))
                        self.patient_score.append(float(details["score"]))
                        self.patient_logit_lme_score.append(
                            float(details["logit_lme_score"])
                        )
                        self.patient_group.append(group_id)
                        self.patient_absolute_logit.append(
                            float(details["absolute_logit"])
                        )
                        self.patient_gland_median_logit.append(
                            float(details["gland_median_logit"])
                        )
                        self.patient_contrast_logit.append(
                            float(details["contrast_logit"])
                        )
                        self.patient_calibrated_logit.append(
                            float(details["calibrated_logit"])
                        )

            region_info = self.case_region_info(prob_3d, batch, b, device)
            if region_info is None:
                continue
            for zone_id, y_true in region_info["zone_true"].items():
                if zone_id not in region_info["zone_score"]:
                    continue
                self.region_true.append(int(y_true))
                self.region_score.append(float(region_info["zone_score"][zone_id]))
                self.region_group.append(group_id)

    def patient_score_details(
        self,
        prob_3d: torch.Tensor,
        patient_mask: torch.Tensor,
    ) -> Optional[Dict[str, float]]:
        """Compute auditable patient pooling features and the final score."""
        if is_contrast_patient_pooling(self.patient_pooling):
            features = masked_logit_lme_features(
                prob_3d,
                patient_mask,
                lme_r=self.lme_r,
            )
            if features is None:
                return None
            calibrated = apply_patient_pooling_calibration(
                features,
                self.patient_pooling_calibration,
            )
            original_score = float(
                torch.sigmoid(
                    torch.tensor(
                        features["absolute_logit"],
                        dtype=torch.float64,
                    )
                ).item()
            )
            return {
                **features,
                **calibrated,
                "logit_lme_score": original_score,
                "contrast_score": float(calibrated["score"]),
            }

        score = masked_probability_pool(
            prob_3d,
            patient_mask,
            mode=self.patient_pooling,
            top_percent=self.top_percent,
            lme_r=self.lme_r,
        )
        if score is None:
            return None
        eps = np.finfo(np.float64).eps
        clipped = float(np.clip(score, eps, 1.0 - eps))
        pooled_logit = float(np.log(clipped / (1.0 - clipped)))
        return {
            "score": float(score),
            "logit_lme_score": float(score),
            "contrast_score": float(score),
            "absolute_logit": pooled_logit,
            "gland_median_logit": float("nan"),
            "contrast_logit": float("nan"),
            "calibrated_logit": pooled_logit,
            "gland_voxels": int(patient_mask.sum().item()),
        }

    def case_region_info(
        self,
        prob_3d: torch.Tensor,
        batch: Mapping,
        b: int,
        device: torch.device,
    ) -> Optional[Dict[str, object]]:
        if "zones_mask" not in batch or "sys_labels" not in batch:
            return None
        zones = batch["zones_mask"][b, 0].to(device)
        if zones.shape != prob_3d.shape:
            zones = F.interpolate(
                zones[None, None].float(),
                size=prob_3d.shape,
                mode="nearest",
            )[0, 0]
        zones = zones.round().long()
        labels = batch["sys_labels"][b].to(device)
        zone_score = {}
        zone_true = {}
        zone_pred = {}
        valid_map = torch.zeros_like(prob_3d, dtype=torch.float32)
        label_map = torch.zeros_like(prob_3d, dtype=torch.float32)
        pred_map = torch.zeros_like(prob_3d, dtype=torch.float32)

        # SBx labels are zone-indexed vectors; rebuild voxel maps here so the
        # scalar region metrics and saved overlays use exactly the same labels.
        for zone_id in range(1, self.max_zones + 1):
            label_idx = zone_id - 1
            if label_idx >= labels.numel():
                continue
            z_label = int(labels[label_idx].item())
            if z_label == self.invalid_sys_label:
                continue
            zone_mask = zones == zone_id
            if not zone_mask.any():
                continue
            valid_map[zone_mask] = 1.0
            region_score = masked_probability_pool(
                prob_3d,
                zone_mask,
                mode=self.region_pooling,
                top_percent=self.top_percent,
                lme_r=self.lme_r,
            )
            if region_score is None:
                continue
            y_true = int(z_label >= self.positive_threshold)
            y_pred = int(region_score >= self.region_threshold)
            zone_true[zone_id] = y_true
            zone_score[zone_id] = float(region_score)
            zone_pred[zone_id] = y_pred
            if y_true:
                label_map[zone_mask] = 1.0
            if y_pred:
                pred_map[zone_mask] = 1.0

        return {
            "zones": zones,
            "zone_true": zone_true,
            "zone_score": zone_score,
            "zone_pred": zone_pred,
            "valid_map": valid_map,
            "label_map": label_map,
            "pred_map": pred_map,
        }

    def build_case_region_maps(
        self,
        lesion_probs: torch.Tensor,
        batch: Mapping,
        b: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        device = lesion_probs.device
        prob_3d = lesion_probs[b, 0]
        region_info = self.case_region_info(prob_3d, batch, b, device)
        if region_info is None:
            return None, None
        label_map = region_info["label_map"].detach().cpu().numpy()
        pred_map = region_info["pred_map"].detach().cpu().numpy()
        return label_map, pred_map

    def _segmentation_target(
        self,
        batch: Mapping,
        b: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if "lesion_mask" in batch and self._case_flag(batch, "has_lesion", b):
            target = batch["lesion_mask"][b, 0].to(device)
            return target
        if "target_mask" in batch and self._case_flag(batch, "has_target", b):
            target = batch["target_mask"][b, 0].to(device)
            return (target >= self.positive_threshold).to(target.dtype)
        return None

    def _patient_label(
        self,
        batch: Mapping,
        b: int,
        device: torch.device,
    ) -> Optional[int]:
        # Explicit/loader-derived case labels take precedence. The loader sets
        # has_cls only when biopsy or an explicit case label is available in the
        # redesigned protocol.
        if "has_cls" in batch and "cls_cspc_label" in batch:
            has_cls = self._case_flag(batch, "has_cls", b)
            label = int(batch["cls_cspc_label"][b].item())
            if has_cls and label != self.invalid_sys_label:
                return int(label > 0)

        # Backward-compatible derivation for evaluation batches that predate
        # cls_cspc_label. Crucially, lesion_mask is never used here.
        has_biopsy_label = False
        patient_label = 0

        if self._case_flag(batch, "has_target", b) and "target_mask" in batch:
            has_biopsy_label = True
            target = batch["target_mask"][b].to(device)
            if bool((target >= self.positive_threshold).any().item()):
                patient_label = 1

        if self._case_flag(batch, "has_sys", b) and "sys_labels" in batch:
            labels = batch["sys_labels"][b].to(device)
            valid = labels != self.invalid_sys_label
            if bool(valid.any().item()):
                has_biopsy_label = True
                if bool((labels[valid] >= self.positive_threshold).any().item()):
                    patient_label = 1

        return patient_label if has_biopsy_label else None

    def _patient_score_mask(
        self,
        batch: Mapping,
        b: int,
        device: torch.device,
        prob_3d: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.use_gland_mask_for_patient_pooling:
            return torch.ones_like(prob_3d, dtype=torch.bool)
        if not self._case_flag(batch, "has_gland", b):
            return None
        if "gland_mask" in batch and batch["gland_mask"][b].numel() > 0:
            gland = batch["gland_mask"][b, 0].to(device)
            if gland.shape != prob_3d.shape:
                gland = F.interpolate(
                    gland[None, None].float(),
                    size=prob_3d.shape,
                    mode="nearest",
                )[0, 0]
            gland = gland > 0
            if gland.any():
                return gland
        return None

    @staticmethod
    def _case_flag(batch: Mapping, key: str, b: int) -> bool:
        if key not in batch:
            return True
        value = batch[key][b]
        if torch.is_tensor(value):
            return bool(value.item() > 0)
        return bool(value)

    def _binary_metrics(
        self,
        y_true,
        y_score,
        threshold: float,
        operating_thresholds: Optional[Mapping] = None,
        groups=None,
    ) -> Dict[str, float]:
        y_true = np.asarray(y_true).astype(np.int64)
        y_score = np.asarray(y_score).astype(np.float32)
        operating_thresholds = operating_thresholds or {}
        if self.select_validation_thresholds:
            threshold = select_balanced_threshold(y_true, y_score, default=threshold)
        threshold = _finite_probability_threshold(threshold, self.prob_threshold)
        balanced_accuracy_threshold = _finite_probability_threshold(
            operating_thresholds.get("balanced_accuracy", threshold),
            threshold,
        )
        fixed_spec_threshold = operating_thresholds.get("at_fixed_specificity")
        fixed_sens_threshold = operating_thresholds.get("at_fixed_sensitivity")
        ci_metrics = {
            "auc_ci_low": float("nan"),
            "auc_ci_high": float("nan"),
            "auprc_ci_low": float("nan"),
            "auprc_ci_high": float("nan"),
            "ci_level": float(_cfg("METRIC_CI_CONFIDENCE_LEVEL", 0.95)),
            "ci_bootstrap_valid": 0,
        }
        if self.compute_confidence_intervals:
            ci_metrics = bootstrap_auc_auprc_ci(
                y_true,
                y_score,
                groups=groups,
            )
        if len(y_true) == 0:
            return {
                "sens": 0.0,
                "spec": 0.0,
                "bacc": 0.0,
                "auc": 0.0,
                "auprc": 0.0,
                "n": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "tp": 0,
                "decision_threshold": threshold,
                "balanced_accuracy_threshold": balanced_accuracy_threshold,
                "sens_at_balanced_accuracy": 0.0,
                "spec_at_balanced_accuracy": 0.0,
                "bacc_at_balanced_accuracy": 0.0,
                **ci_metrics,
                **operating_point_metrics(
                    y_true,
                    y_score,
                    fixed_specificity=self.fixed_specificity,
                    fixed_sensitivity=self.fixed_sensitivity,
                    threshold_at_fixed_spec=fixed_spec_threshold,
                    threshold_at_fixed_sens=fixed_sens_threshold,
                ),
            }
        decision = binary_metrics_at_threshold(y_true, y_score, threshold)
        balanced_decision = binary_metrics_at_threshold(
            y_true,
            y_score,
            balanced_accuracy_threshold,
        )
        return {
            "sens": decision["sens"],
            "spec": decision["spec"],
            "bacc": decision["bacc"],
            "auc": safe_auc(y_true, y_score),
            "auprc": safe_auprc(y_true, y_score),
            "n": int(len(y_true)),
            "tn": decision["tn"],
            "fp": decision["fp"],
            "fn": decision["fn"],
            "tp": decision["tp"],
            "decision_threshold": threshold,
            "balanced_accuracy_threshold": balanced_accuracy_threshold,
            "sens_at_balanced_accuracy": balanced_decision["sens"],
            "spec_at_balanced_accuracy": balanced_decision["spec"],
            "bacc_at_balanced_accuracy": balanced_decision["bacc"],
            **ci_metrics,
            **operating_point_metrics(
                y_true,
                y_score,
                fixed_specificity=self.fixed_specificity,
                fixed_sensitivity=self.fixed_sensitivity,
                threshold_at_fixed_spec=fixed_spec_threshold,
                threshold_at_fixed_sens=fixed_sens_threshold,
            ),
        }

    @staticmethod
    def _prefixed_binary_metrics(
        prefix: str,
        metrics: Mapping,
    ) -> Dict[str, float]:
        """Expose one binary evaluation under an explicit score-family name."""
        fields = (
            "sens",
            "spec",
            "bacc",
            "auc",
            "auprc",
            "auc_ci_low",
            "auc_ci_high",
            "auprc_ci_low",
            "auprc_ci_high",
            "ci_level",
            "ci_bootstrap_valid",
            "n",
            "tn",
            "fp",
            "fn",
            "tp",
            "decision_threshold",
            "balanced_accuracy_threshold",
            "sens_at_balanced_accuracy",
            "spec_at_balanced_accuracy",
            "bacc_at_balanced_accuracy",
            "fixed_spec_target",
            "sens_at_fixed_spec",
            "actual_spec_at_fixed_spec",
            "actual_fpr_at_fixed_spec",
            "threshold_at_fixed_spec",
            "fixed_sens_target",
            "spec_at_fixed_sens",
            "actual_sens_at_fixed_sens",
            "threshold_at_fixed_sens",
        )
        return {f"{prefix}_{field}": metrics[field] for field in fields}

    @staticmethod
    def _unavailable_binary_metrics(reference: Mapping) -> Dict[str, float]:
        """Shape-compatible empty result for a deliberately disabled endpoint."""
        unavailable = dict(reference)
        for field in (
            "sens",
            "spec",
            "bacc",
            "auc",
            "auprc",
            "auc_ci_low",
            "auc_ci_high",
            "auprc_ci_low",
            "auprc_ci_high",
            "decision_threshold",
            "balanced_accuracy_threshold",
            "sens_at_balanced_accuracy",
            "spec_at_balanced_accuracy",
            "bacc_at_balanced_accuracy",
            "sens_at_fixed_spec",
            "actual_spec_at_fixed_spec",
            "actual_fpr_at_fixed_spec",
            "threshold_at_fixed_spec",
            "spec_at_fixed_sens",
            "actual_sens_at_fixed_sens",
            "threshold_at_fixed_sens",
        ):
            unavailable[field] = float("nan")
        for field in ("ci_bootstrap_valid", "n", "tn", "fp", "fn", "tp"):
            unavailable[field] = 0
        return unavailable

    def compute_metrics(self) -> Dict[str, float]:
        has_complete_patient_features = (
            len(self.patient_absolute_logit)
            == len(self.patient_contrast_logit)
            == len(self.patient_true)
        )
        if (
            is_contrast_patient_pooling(self.patient_pooling)
            and has_complete_patient_features
        ):
            if self.select_validation_thresholds:
                self.patient_pooling_calibration = fit_patient_pooling_calibration(
                    self.patient_true,
                    self.patient_absolute_logit,
                    self.patient_contrast_logit,
                    lme_r=self.lme_r,
                )
            recalibrated_scores = []
            recalibrated_logits = []
            for absolute_logit, contrast_logit in zip(
                self.patient_absolute_logit,
                self.patient_contrast_logit,
            ):
                calibrated = apply_patient_pooling_calibration(
                    {
                        "absolute_logit": absolute_logit,
                        "contrast_logit": contrast_logit,
                    },
                    self.patient_pooling_calibration,
                )
                recalibrated_scores.append(calibrated["score"])
                recalibrated_logits.append(calibrated["calibrated_logit"])
            self.patient_score = recalibrated_scores
            self.patient_calibrated_logit = recalibrated_logits

        compare_patient_pooling = bool(
            _cfg("SEG_EVAL_COMPARE_PATIENT_POOLING", True)
        )
        if (
            compare_patient_pooling
            and is_contrast_patient_pooling(self.patient_pooling)
            and len(self.patient_logit_lme_score) == len(self.patient_true)
        ):
            patient_logit_lme = self._binary_metrics(
                self.patient_true,
                self.patient_logit_lme_score,
                self.patient_logit_lme_threshold,
                self.patient_logit_lme_operating_thresholds,
                self.patient_group,
            )
        else:
            # Non-contrast compatibility: the explicit aliases describe the
            # configured patient score rather than silently dropping columns.
            patient_logit_lme = None

        patient = self._binary_metrics(
            self.patient_true,
            self.patient_score,
            self.patient_threshold,
            self.patient_operating_thresholds,
            self.patient_group,
        )
        if patient_logit_lme is None:
            patient_logit_lme = patient
        patient_contrast = (
            patient
            if is_contrast_patient_pooling(self.patient_pooling)
            else self._unavailable_binary_metrics(patient)
        )
        region = self._binary_metrics(
            self.region_true,
            self.region_score,
            self.region_threshold,
            self.region_operating_thresholds,
            self.region_group,
        )
        self.patient_threshold = patient["decision_threshold"]
        self.patient_logit_lme_threshold = patient_logit_lme[
            "decision_threshold"
        ]
        self.region_threshold = region["decision_threshold"]
        metrics = {
            "patient_sens": patient["sens"],
            "patient_spec": patient["spec"],
            "patient_bacc": patient["bacc"],
            "patient_auc": patient["auc"],
            "patient_auprc": patient["auprc"],
            "patient_auc_ci_low": patient["auc_ci_low"],
            "patient_auc_ci_high": patient["auc_ci_high"],
            "patient_auprc_ci_low": patient["auprc_ci_low"],
            "patient_auprc_ci_high": patient["auprc_ci_high"],
            "patient_ci_level": patient["ci_level"],
            "patient_ci_bootstrap_valid": patient["ci_bootstrap_valid"],
            "patient_n": patient["n"],
            "patient_tn": patient["tn"],
            "patient_fp": patient["fp"],
            "patient_fn": patient["fn"],
            "patient_tp": patient["tp"],
            "patient_decision_threshold": patient["decision_threshold"],
            "patient_balanced_accuracy_threshold": patient[
                "balanced_accuracy_threshold"
            ],
            "patient_sens_at_balanced_accuracy": patient[
                "sens_at_balanced_accuracy"
            ],
            "patient_spec_at_balanced_accuracy": patient[
                "spec_at_balanced_accuracy"
            ],
            "patient_bacc_at_balanced_accuracy": patient[
                "bacc_at_balanced_accuracy"
            ],
            "patient_fixed_spec_target": patient["fixed_spec_target"],
            "patient_sens_at_fixed_spec": patient["sens_at_fixed_spec"],
            "patient_actual_spec_at_fixed_spec": patient["actual_spec_at_fixed_spec"],
            "patient_actual_fpr_at_fixed_spec": patient["actual_fpr_at_fixed_spec"],
            "patient_threshold_at_fixed_spec": patient["threshold_at_fixed_spec"],
            "patient_fixed_sens_target": patient["fixed_sens_target"],
            "patient_spec_at_fixed_sens": patient["spec_at_fixed_sens"],
            "patient_actual_sens_at_fixed_sens": patient["actual_sens_at_fixed_sens"],
            "patient_threshold_at_fixed_sens": patient["threshold_at_fixed_sens"],
            "patient_pooling_mode": self.patient_pooling,
            "patient_pooling_lme_r": float(self.lme_r),
            "patient_pooling_alpha": float(
                self.patient_pooling_calibration.get("alpha", 1.0)
            ),
            "patient_pooling_beta": float(
                self.patient_pooling_calibration.get("beta", 0.0)
            ),
            "patient_pooling_intercept": float(
                self.patient_pooling_calibration.get("intercept", 0.0)
            ),
            "patient_pooling_calibration_n": int(
                self.patient_pooling_calibration.get("n", 0)
            ),
            "patient_pooling_calibration_positive_n": int(
                self.patient_pooling_calibration.get("positive_n", 0)
            ),
            "patient_pooling_calibration_fitted": int(
                self.patient_pooling_calibration.get("fitted", 0)
            ),
            "patient_pooling_calibration_status": str(
                self.patient_pooling_calibration.get("status", "")
            ),
            "patient_pooling_regularization_c": float(
                self.patient_pooling_calibration.get("regularization_c", 1.0)
            ),
            "region_sens": region["sens"],
            "region_spec": region["spec"],
            "region_bacc": region["bacc"],
            "region_auc": region["auc"],
            "region_auprc": region["auprc"],
            "region_auc_ci_low": region["auc_ci_low"],
            "region_auc_ci_high": region["auc_ci_high"],
            "region_auprc_ci_low": region["auprc_ci_low"],
            "region_auprc_ci_high": region["auprc_ci_high"],
            "region_ci_level": region["ci_level"],
            "region_ci_bootstrap_valid": region["ci_bootstrap_valid"],
            "region_n": region["n"],
            "region_tn": region["tn"],
            "region_fp": region["fp"],
            "region_fn": region["fn"],
            "region_tp": region["tp"],
            "region_decision_threshold": region["decision_threshold"],
            "region_balanced_accuracy_threshold": region[
                "balanced_accuracy_threshold"
            ],
            "region_sens_at_balanced_accuracy": region[
                "sens_at_balanced_accuracy"
            ],
            "region_spec_at_balanced_accuracy": region[
                "spec_at_balanced_accuracy"
            ],
            "region_bacc_at_balanced_accuracy": region[
                "bacc_at_balanced_accuracy"
            ],
            "region_fixed_spec_target": region["fixed_spec_target"],
            "region_sens_at_fixed_spec": region["sens_at_fixed_spec"],
            "region_actual_spec_at_fixed_spec": region["actual_spec_at_fixed_spec"],
            "region_actual_fpr_at_fixed_spec": region["actual_fpr_at_fixed_spec"],
            "region_threshold_at_fixed_spec": region["threshold_at_fixed_spec"],
            "region_fixed_sens_target": region["fixed_sens_target"],
            "region_spec_at_fixed_sens": region["spec_at_fixed_sens"],
            "region_actual_sens_at_fixed_sens": region["actual_sens_at_fixed_sens"],
            "region_threshold_at_fixed_sens": region["threshold_at_fixed_sens"],
        }
        # ``patient_*`` is the configured canonical score (original logit-LME
        # for B, contrast for N). Explicit families keep output schemas stable.
        metrics.update(
            self._prefixed_binary_metrics("patient_contrast", patient_contrast)
        )
        metrics.update(
            self._prefixed_binary_metrics(
                "patient_logit_lme",
                patient_logit_lme,
            )
        )
        return metrics


# -----------------------------------------------------------------------------
# Metric tracker
# -----------------------------------------------------------------------------

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n: int = 1):
        if val is None:
            return
        val = tensor_to_float(val)
        if not np.isnan(val) and not np.isinf(val):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / max(self.count, 1)


class MetricTracker:
    """Tracks only segmentation/MIL losses and metrics."""

    ENTROPY_KEYS = (
        "lesion_all",
        "lesion_gland",
        "lesion_outside_gland",
        "dense_positive",
        "dense_negative_gland",
        "tbx_positive",
        "tbx_negative",
        "region_all",
        "region_positive",
        "region_negative",
        "patient_all",
        "patient_positive",
        "patient_negative",
    )
    PATIENT_SCORE_METRIC_FIELDS = (
        "sens",
        "spec",
        "bacc",
        "auc",
        "auprc",
        "auc_ci_low",
        "auc_ci_high",
        "auprc_ci_low",
        "auprc_ci_high",
        "ci_level",
        "ci_bootstrap_valid",
        "n",
        "tn",
        "fp",
        "fn",
        "tp",
        "decision_threshold",
        "balanced_accuracy_threshold",
        "sens_at_balanced_accuracy",
        "spec_at_balanced_accuracy",
        "bacc_at_balanced_accuracy",
        "fixed_spec_target",
        "sens_at_fixed_spec",
        "actual_spec_at_fixed_spec",
        "actual_fpr_at_fixed_spec",
        "threshold_at_fixed_spec",
        "fixed_sens_target",
        "spec_at_fixed_sens",
        "actual_sens_at_fixed_sens",
        "threshold_at_fixed_sens",
    )

    def __init__(self):
        self.loss_total = AverageMeter()
        self.loss_lesion = AverageMeter()
        self.loss_lesion_dense = AverageMeter()
        self.loss_lesion_sparse = AverageMeter()
        self.loss_lesion_sparse_bce = AverageMeter()
        self.loss_lesion_sparse_dice = AverageMeter()
        self.loss_lesion_sys = AverageMeter()
        self.loss_lesion_outside_gland = AverageMeter()
        self.loss_lesion_patient = AverageMeter()

        self.lesion_dice = AverageMeter()
        self.lesion_dice_values = []
        self.lesion_dice_std = 0.0
        self.lesion_dice_n = 0
        self.lesion_full_crop_dice = AverageMeter()
        self.lesion_full_crop_dice_values = []
        self.lesion_full_crop_dice_std = 0.0
        self.lesion_full_crop_dice_n = 0
        self.lesion_dice_sweep_thresholds = configured_target_dice_thresholds()
        self.lesion_dice_sweep_values = {
            threshold: [] for threshold in self.lesion_dice_sweep_thresholds
        }
        self.lesion_best_threshold_dice = 0.0
        self.lesion_best_threshold_dice_std = 0.0
        self.lesion_best_threshold_dice_n = 0
        self.lesion_best_threshold = float("nan")
        self.segmentation_threshold = float(_cfg("PRED_PROB_THRESHOLD", 0.5))
        self.metric_probability_threshold = self.segmentation_threshold
        self.segmentation_threshold_metric = "configured_default"
        self.threshold_source = "configured_default"
        self.threshold_validation_epoch = 0
        self.lesion_gland_dice = AverageMeter()
        self.lesion_gland_dice_values = []
        self.lesion_gland_dice_std = 0.0
        self.lesion_gland_dice_n = 0
        self.lesion_gland_cases = 0
        self.lesion_gland_missing_cases = 0
        self.lesion_gland_voxels = 0
        self.lesion_target_outside_gland_voxels = 0
        self.lesion_f1 = AverageMeter()
        self.lesion_sens = AverageMeter()
        self.lesion_spec = AverageMeter()
        self.lesion_voxel_true = []
        self.lesion_voxel_score = []
        self.lesion_voxel_n = 0
        self.lesion_voxel_fixed_spec_target = float(_cfg("FIXED_SPECIFICITY_TARGET", 0.95))
        self.lesion_voxel_sens_at_fixed_spec = 0.0
        self.lesion_voxel_actual_spec_at_fixed_spec = 0.0
        self.lesion_voxel_actual_fpr_at_fixed_spec = 0.0
        self.lesion_voxel_threshold_at_fixed_spec = float("nan")
        self.lesion_voxel_fixed_sens_target = float(_cfg("FIXED_SENSITIVITY_TARGET", 0.90))
        self.lesion_voxel_spec_at_fixed_sens = 0.0
        self.lesion_voxel_actual_sens_at_fixed_sens = 0.0
        self.lesion_voxel_threshold_at_fixed_sens = float("nan")
        self.target_cspca_dice = AverageMeter()
        self.target_cspca_dice_values = []
        self.target_cspca_dice_std = 0.0
        self.target_cspca_dice_n = 0
        self.tbx_masked_dice = AverageMeter()
        self.tbx_masked_dice_values = []
        self.tbx_masked_dice_std = 0.0
        self.tbx_masked_dice_n = 0
        self.target_cspca_dice_sweep_thresholds = configured_target_dice_thresholds()
        self.target_cspca_dice_sweep_values = {
            threshold: [] for threshold in self.target_cspca_dice_sweep_thresholds
        }
        self.target_cspca_best_threshold_dice = 0.0
        self.target_cspca_best_threshold_dice_std = 0.0
        self.target_cspca_best_threshold_dice_n = 0
        self.target_cspca_best_threshold = float("nan")
        self.target_cspca_topk_dice = AverageMeter()
        self.target_cspca_topk_dice_values = []
        self.target_cspca_topk_dice_std = 0.0
        self.target_cspca_topk_dice_n = 0
        self.target_cspca_top_percent = float(_cfg("TARGET_DICE_TOP_PERCENT", 1.0))
        self.target_cspca_top_percent_dice = AverageMeter()
        self.target_cspca_top_percent_dice_values = []
        self.target_cspca_top_percent_dice_std = 0.0
        self.target_cspca_top_percent_dice_n = 0
        self.lesion_froc = FROCEvaluator()
        self.target_cspca_froc = FROCEvaluator()
        self.lesion_froc_metrics = self.lesion_froc.compute_metrics(prefix="lesion_")
        self.target_cspca_froc_metrics = self.target_cspca_froc.compute_metrics(prefix="target_cspca_")
        self.target_cspca_voxel_true = []
        self.target_cspca_voxel_score = []
        self.target_cspca_voxel_n = 0
        self.target_cspca_voxel_fixed_spec_target = float(_cfg("FIXED_SPECIFICITY_TARGET", 0.95))
        self.target_cspca_voxel_sens_at_fixed_spec = 0.0
        self.target_cspca_voxel_actual_spec_at_fixed_spec = 0.0
        self.target_cspca_voxel_actual_fpr_at_fixed_spec = 0.0
        self.target_cspca_voxel_threshold_at_fixed_spec = float("nan")
        self.target_cspca_voxel_fixed_sens_target = float(_cfg("FIXED_SENSITIVITY_TARGET", 0.90))
        self.target_cspca_voxel_spec_at_fixed_sens = 0.0
        self.target_cspca_voxel_actual_sens_at_fixed_sens = 0.0
        self.target_cspca_voxel_threshold_at_fixed_sens = float("nan")
        self.tbx_roi_true = []
        self.tbx_roi_score = []
        self.tbx_roi_group = []
        self._next_tbx_roi_group = 0
        self.tbx_roi_bacc = 0.0
        self.tbx_roi_sens = 0.0
        self.tbx_roi_spec = 0.0
        self.tbx_roi_auc = 0.0
        self.tbx_roi_auprc = 0.0
        self.tbx_roi_auc_ci_low = float("nan")
        self.tbx_roi_auc_ci_high = float("nan")
        self.tbx_roi_auprc_ci_low = float("nan")
        self.tbx_roi_auprc_ci_high = float("nan")
        self.tbx_roi_ci_level = float(_cfg("METRIC_CI_CONFIDENCE_LEVEL", 0.95))
        self.tbx_roi_ci_bootstrap_valid = 0
        self.tbx_roi_n = 0
        self.tbx_roi_decision_threshold = float(_cfg("PRED_PROB_THRESHOLD", 0.5))
        self.tbx_roi_fixed_spec_target = float(_cfg("FIXED_SPECIFICITY_TARGET", 0.95))
        self.tbx_roi_sens_at_fixed_spec = 0.0
        self.tbx_roi_actual_spec_at_fixed_spec = 0.0
        self.tbx_roi_actual_fpr_at_fixed_spec = 0.0
        self.tbx_roi_threshold_at_fixed_spec = float("nan")
        self.tbx_roi_fixed_sens_target = float(_cfg("FIXED_SENSITIVITY_TARGET", 0.90))
        self.tbx_roi_spec_at_fixed_sens = 0.0
        self.tbx_roi_actual_sens_at_fixed_sens = 0.0
        self.tbx_roi_threshold_at_fixed_sens = float("nan")

        self.patient_bacc = 0.0
        self.patient_sens = 0.0
        self.patient_spec = 0.0
        self.patient_auc = 0.0
        self.patient_auprc = 0.0
        self.patient_auc_ci_low = float("nan")
        self.patient_auc_ci_high = float("nan")
        self.patient_auprc_ci_low = float("nan")
        self.patient_auprc_ci_high = float("nan")
        self.patient_ci_level = float(_cfg("METRIC_CI_CONFIDENCE_LEVEL", 0.95))
        self.patient_ci_bootstrap_valid = 0
        self.patient_n = 0
        self.patient_tn = 0
        self.patient_fp = 0
        self.patient_fn = 0
        self.patient_tp = 0
        self.patient_decision_threshold = float(_cfg("PRED_PROB_THRESHOLD", 0.5))
        self.patient_balanced_accuracy_threshold = float(
            _cfg("PRED_PROB_THRESHOLD", 0.5)
        )
        self.patient_sens_at_balanced_accuracy = 0.0
        self.patient_spec_at_balanced_accuracy = 0.0
        self.patient_bacc_at_balanced_accuracy = 0.0
        self.patient_fixed_spec_target = float(_cfg("FIXED_SPECIFICITY_TARGET", 0.95))
        self.patient_sens_at_fixed_spec = 0.0
        self.patient_actual_spec_at_fixed_spec = 0.0
        self.patient_actual_fpr_at_fixed_spec = 0.0
        self.patient_threshold_at_fixed_spec = float("nan")
        self.patient_fixed_sens_target = float(_cfg("FIXED_SENSITIVITY_TARGET", 0.90))
        self.patient_spec_at_fixed_sens = 0.0
        self.patient_actual_sens_at_fixed_sens = 0.0
        self.patient_threshold_at_fixed_sens = float("nan")
        self.patient_pooling_mode = str(
            _cfg("SEG_PATIENT_POOLING", "logit_lme")
        ).lower()
        self.patient_pooling_lme_r = float(
            _cfg("SEG_RISK_LME_R", _cfg("LME_R", 8.0))
        )
        self.patient_pooling_alpha = 1.0
        self.patient_pooling_beta = 0.0
        self.patient_pooling_intercept = 0.0
        self.patient_pooling_regularization_c = float(
            _cfg("SEG_PATIENT_CALIBRATION_C", 1.0)
        )
        self.patient_pooling_calibration_n = 0
        self.patient_pooling_calibration_positive_n = 0
        self.patient_pooling_calibration_fitted = 0
        self.patient_pooling_calibration_status = "not_evaluated"
        patient_metric_defaults = {
            "auc_ci_low": float("nan"),
            "auc_ci_high": float("nan"),
            "auprc_ci_low": float("nan"),
            "auprc_ci_high": float("nan"),
            "ci_level": float(_cfg("METRIC_CI_CONFIDENCE_LEVEL", 0.95)),
            "decision_threshold": float(_cfg("PRED_PROB_THRESHOLD", 0.5)),
            "balanced_accuracy_threshold": float(
                _cfg("PRED_PROB_THRESHOLD", 0.5)
            ),
            "fixed_spec_target": float(_cfg("FIXED_SPECIFICITY_TARGET", 0.95)),
            "threshold_at_fixed_spec": float("nan"),
            "fixed_sens_target": float(_cfg("FIXED_SENSITIVITY_TARGET", 0.90)),
            "threshold_at_fixed_sens": float("nan"),
        }
        for score_family in ("patient_contrast", "patient_logit_lme"):
            for field in self.PATIENT_SCORE_METRIC_FIELDS:
                setattr(
                    self,
                    f"{score_family}_{field}",
                    patient_metric_defaults.get(field, 0),
                )

        self.region_bacc = 0.0
        self.region_sens = 0.0
        self.region_spec = 0.0
        self.region_auc = 0.0
        self.region_auprc = 0.0
        self.region_auc_ci_low = float("nan")
        self.region_auc_ci_high = float("nan")
        self.region_auprc_ci_low = float("nan")
        self.region_auprc_ci_high = float("nan")
        self.region_ci_level = float(_cfg("METRIC_CI_CONFIDENCE_LEVEL", 0.95))
        self.region_ci_bootstrap_valid = 0
        self.region_n = 0
        self.region_tn = 0
        self.region_fp = 0
        self.region_fn = 0
        self.region_tp = 0
        self.region_decision_threshold = float(_cfg("PRED_PROB_THRESHOLD", 0.5))
        self.region_balanced_accuracy_threshold = float(
            _cfg("PRED_PROB_THRESHOLD", 0.5)
        )
        self.region_sens_at_balanced_accuracy = 0.0
        self.region_spec_at_balanced_accuracy = 0.0
        self.region_bacc_at_balanced_accuracy = 0.0
        self.region_fixed_spec_target = float(_cfg("FIXED_SPECIFICITY_TARGET", 0.95))
        self.region_sens_at_fixed_spec = 0.0
        self.region_actual_spec_at_fixed_spec = 0.0
        self.region_actual_fpr_at_fixed_spec = 0.0
        self.region_threshold_at_fixed_spec = float("nan")
        self.region_fixed_sens_target = float(_cfg("FIXED_SENSITIVITY_TARGET", 0.90))
        self.region_spec_at_fixed_sens = 0.0
        self.region_actual_sens_at_fixed_sens = 0.0
        self.region_threshold_at_fixed_sens = float("nan")

        # Direct SBx-MIL readout from the model's pooled region logits. This is
        # kept separate from canonical region_* metrics, which use the same
        # fixed risk-map pooling rule across every experiment.
        self.sbx_mil_pooling_mode = str(_cfg("MIL_POOLING", "lme")).lower()
        self.sbx_mil_lme_r = float(_cfg("LME_R", 8.0))
        self.sbx_mil_region_bacc = 0.0
        self.sbx_mil_region_sens = 0.0
        self.sbx_mil_region_spec = 0.0
        self.sbx_mil_region_auc = 0.0
        self.sbx_mil_region_auprc = 0.0
        self.sbx_mil_region_n = 0
        self.sbx_mil_region_tn = 0
        self.sbx_mil_region_fp = 0
        self.sbx_mil_region_fn = 0
        self.sbx_mil_region_tp = 0

        self.em_w_lesion_dense = AverageMeter()
        self.em_w_lesion_sparse = AverageMeter()
        self.em_w_lesion_sys = AverageMeter()
        self.em_w_lesion_outside_gland = AverageMeter()
        self.em_w_lesion_patient = AverageMeter()

        self.active_lesion_dense = AverageMeter()
        self.active_lesion_sparse = AverageMeter()
        self.active_lesion_sys = AverageMeter()
        self.active_lesion_outside_gland = AverageMeter()
        self.active_lesion_patient = AverageMeter()

        self.loss_num_batches = 0
        self.loss_num_cases = 0
        self.loss_dense_cases = 0
        self.loss_sparse_cases = 0
        self.loss_sparse_has_target_cases = 0
        self.loss_sparse_sampled_cases = 0
        self.loss_sparse_positive_cases = 0
        self.loss_sparse_negative_cases = 0
        self.loss_sparse_voxels = 0
        self.loss_sparse_positive_voxels = 0
        self.loss_sparse_negative_voxels = 0
        self.loss_sparse_dice_cases = 0
        self.loss_sys_cases = 0
        self.loss_sys_regions = 0
        self.loss_outside_gland_cases = 0
        self.loss_outside_gland_voxels = 0
        self.outside_gland_prob_mean = AverageMeter()
        self.loss_patient_cases = 0
        self.loss_patient_positive_cases = 0
        self.loss_patient_negative_cases = 0
        self.patient_risk_prob_mean = AverageMeter()
        self.patient_risk_positive_prob_mean = AverageMeter()
        self.patient_risk_negative_prob_mean = AverageMeter()
        self.tbx_pos_prob_mean = AverageMeter()
        self.tbx_neg_prob_mean = AverageMeter()
        self.tbx_neg_1mp_mean = AverageMeter()
        self.tbx_pos_bce = AverageMeter()
        self.tbx_neg_bce = AverageMeter()
        self.entropy_meters = {
            key: AverageMeter() for key in self.ENTROPY_KEYS
        }

    @staticmethod
    def _expand_case_mask(case_mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        case_mask = case_mask.to(device=reference.device).bool().reshape(-1)
        shape = (case_mask.shape[0],) + (1,) * (reference.ndim - 1)
        return case_mask.reshape(shape).expand_as(reference)

    def _update_entropy_meter(
        self,
        key: str,
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        if logits is None or logits.numel() == 0:
            return
        entropy = binary_entropy_bits_from_logits(logits.detach())
        if mask is not None:
            mask = mask.to(device=entropy.device).bool()
            if mask.shape != entropy.shape:
                try:
                    mask = mask.expand_as(entropy)
                except RuntimeError as exc:
                    raise ValueError(
                        f"Entropy mask shape {tuple(mask.shape)} does not match "
                        f"logit shape {tuple(entropy.shape)} for {key}."
                    ) from exc
            entropy = entropy[mask]
        else:
            entropy = entropy.reshape(-1)
        if entropy.numel() == 0:
            return
        self.entropy_meters[key].update(
            entropy.mean(),
            n=int(entropy.numel()),
        )

    def _update_patient_entropy(
        self,
        lesion_logits: torch.Tensor,
        batch: Mapping,
    ) -> None:
        has_cls = batch.get("has_cls")
        cls_label = batch.get("cls_cspc_label")
        if has_cls is None or cls_label is None:
            return

        invalid_label = int(_cfg("INVALID_SYS_LABEL", -1))
        labels = cls_label.to(device=lesion_logits.device).reshape(-1)
        valid = has_cls.to(device=lesion_logits.device).reshape(-1) > 0
        valid = valid & (labels != invalid_label)
        if not bool(valid.any().item()):
            return

        gland_mask = batch.get("gland_mask")
        has_gland = batch.get("has_gland")
        use_gland = bool(_cfg("PATIENT_RISK_USE_GLAND_MASK", True))
        pooling = str(_cfg("PATIENT_RISK_POOLING", "lme")).lower()
        lme_r = float(_cfg("PATIENT_RISK_LME_R", 8.0))
        pooled_logits = []
        pooled_labels = []

        for idx in torch.where(valid)[0]:
            logits_3d = lesion_logits[idx, 0]
            if use_gland:
                if gland_mask is None or has_gland is None or not bool(has_gland[idx].item()):
                    continue
                mask = gland_mask[idx, 0].to(device=lesion_logits.device) > 0
                if not bool(mask.any().item()):
                    continue
                values = logits_3d[mask]
            else:
                values = logits_3d.reshape(-1)
            if values.numel() == 0:
                continue
            if pooling == "max":
                pooled = values.max()
            elif pooling == "mean":
                pooled = values.mean()
            else:
                n_values = values.new_tensor(float(values.numel()))
                pooled = (
                    torch.logsumexp(values * lme_r, dim=0) / lme_r
                    - torch.log(n_values) / lme_r
                )
            pooled_logits.append(pooled)
            pooled_labels.append(labels[idx])

        if not pooled_logits:
            return
        logits = torch.stack(pooled_logits)
        labels = torch.stack(pooled_labels)
        self._update_entropy_meter("patient_all", logits)
        self._update_entropy_meter(
            "patient_positive", logits, labels > 0
        )
        self._update_entropy_meter(
            "patient_negative", logits, labels == 0
        )

    @torch.no_grad()
    def update_prediction_entropy(self, outputs: Mapping, batch: Mapping) -> None:
        """Accumulate prediction entropy without affecting gradients or loss."""
        lesion_logits = outputs.get("lesion_logits")
        if lesion_logits is None:
            return

        self._update_entropy_meter("lesion_all", lesion_logits)

        gland_mask = batch.get("gland_mask")
        has_gland = batch.get("has_gland")
        valid_gland_cases = None
        gland = None
        if gland_mask is not None:
            gland = gland_mask.to(device=lesion_logits.device) > 0
            if has_gland is None:
                valid_gland_cases = torch.ones(
                    lesion_logits.size(0),
                    dtype=torch.bool,
                    device=lesion_logits.device,
                )
            else:
                valid_gland_cases = has_gland.to(
                    device=lesion_logits.device
                ).reshape(-1) > 0
            valid_gland_voxels = self._expand_case_mask(
                valid_gland_cases, lesion_logits
            )
            self._update_entropy_meter(
                "lesion_gland", lesion_logits, valid_gland_voxels & gland
            )
            self._update_entropy_meter(
                "lesion_outside_gland",
                lesion_logits,
                valid_gland_voxels & (~gland),
            )

        has_lesion = batch.get("has_lesion")
        lesion_mask = batch.get("lesion_mask")
        if has_lesion is not None and lesion_mask is not None:
            dense_cases = self._expand_case_mask(
                has_lesion.to(device=lesion_logits.device).reshape(-1) > 0,
                lesion_logits,
            )
            dense_target = lesion_mask.to(device=lesion_logits.device) > 0
            self._update_entropy_meter(
                "dense_positive", lesion_logits, dense_cases & dense_target
            )
            if gland is not None and valid_gland_cases is not None:
                valid_gland_voxels = self._expand_case_mask(
                    valid_gland_cases, lesion_logits
                )
                self._update_entropy_meter(
                    "dense_negative_gland",
                    lesion_logits,
                    dense_cases & valid_gland_voxels & gland & (~dense_target),
                )

        target_mask = batch.get("target_mask")
        has_target = batch.get("has_target")
        if target_mask is not None and has_target is not None:
            positive_threshold = int(
                _cfg("LESION_POSITIVE_THRESHOLD", _cfg("CSPC_THRESHOLD", 1))
            )
            target_cases = self._expand_case_mask(
                has_target.to(device=lesion_logits.device).reshape(-1) > 0,
                lesion_logits,
            )
            target = target_mask.to(device=lesion_logits.device)
            self._update_entropy_meter(
                "tbx_positive",
                lesion_logits,
                target_cases & (target >= positive_threshold),
            )
            self._update_entropy_meter(
                "tbx_negative",
                lesion_logits,
                target_cases & (target > 0) & (target < positive_threshold),
            )

        region_logits = outputs.get("region_logits")
        if region_logits is not None:
            if region_logits.ndim == 3 and region_logits.size(-1) == 1:
                region_logits = region_logits.squeeze(-1)
            region_valid = outputs.get("region_valid_mask")
            sys_labels = batch.get("sys_labels")
            if region_valid is None and sys_labels is not None:
                region_valid = sys_labels != int(_cfg("INVALID_SYS_LABEL", -1))
            if region_valid is not None:
                if region_valid.ndim == 3 and region_valid.size(-1) == 1:
                    region_valid = region_valid.squeeze(-1)
                region_valid = region_valid.to(device=region_logits.device).bool()
                has_sys = batch.get("has_sys")
                if has_sys is not None:
                    region_valid = region_valid & self._expand_case_mask(
                        has_sys.to(device=region_logits.device).reshape(-1) > 0,
                        region_logits,
                    )
                self._update_entropy_meter(
                    "region_all", region_logits, region_valid
                )
                if sys_labels is not None:
                    labels = sys_labels.to(device=region_logits.device)
                    positive_threshold = int(
                        _cfg(
                            "LESION_POSITIVE_THRESHOLD",
                            _cfg("CSPC_THRESHOLD", 1),
                        )
                    )
                    self._update_entropy_meter(
                        "region_positive",
                        region_logits,
                        region_valid & (labels >= positive_threshold),
                    )
                    self._update_entropy_meter(
                        "region_negative",
                        region_logits,
                        region_valid & (labels < positive_threshold),
                    )

        self._update_patient_entropy(lesion_logits, batch)

    def get_entropy_dict(self, prefix: str) -> Dict[str, float]:
        out = {}
        for key, meter in self.entropy_meters.items():
            out[f"{prefix}_entropy_{key}_bits"] = meter.avg
            out[f"{prefix}_entropy_{key}_n"] = meter.count
        return out

    def print_entropy_summary(self) -> str:
        def value(key: str) -> str:
            meter = self.entropy_meters[key]
            return f"{meter.avg:.4f}" if meter.count else "n/a"

        return (
            "H(bits) | "
            f"Map {value('lesion_all')} | "
            f"Gland {value('lesion_gland')} | "
            f"Outside {value('lesion_outside_gland')} | "
            f"Dense +/- {value('dense_positive')}/{value('dense_negative_gland')} | "
            f"TBx +/- {value('tbx_positive')}/{value('tbx_negative')} | "
            f"Region +/- {value('region_positive')}/{value('region_negative')} | "
            f"Patient +/- {value('patient_positive')}/{value('patient_negative')}"
        )

    def update_losses(self, *args, em_weights=None, active_tasks=None, **kwargs):
        """Update loss meters from either a loss_dict or legacy positional args.

        Preferred:
            tracker.update_losses(loss_dict)

        Also accepts old call style:
            update_losses(total, g_tot, g_tbx, g_sbx, l_tot, l_dense, l_sparse, l_sys, gl, ...)
        Grade/gland values are ignored.
        """
        if len(args) == 1 and isinstance(args[0], dict):
            loss_dict = normalise_loss_output(args[0])
            total = loss_dict["total_loss"]
            l_tot = loss_dict["loss_lesion_total"]
            l_dense = loss_dict["loss_lesion_dense"]
            l_sparse = loss_dict["loss_lesion_sparse"]
            l_sparse_bce = loss_dict.get("loss_lesion_sparse_bce", l_sparse)
            l_sparse_dice = loss_dict.get("loss_lesion_sparse_dice", 0.0)
            l_sys = loss_dict["loss_lesion_sys"]
            l_outside_gland = loss_dict.get("loss_lesion_outside_gland", 0.0)
            l_patient = loss_dict.get("loss_lesion_patient", 0.0)
            em_weights = loss_dict.get("em_weights", em_weights)
            active_tasks = loss_dict.get("active_tasks", active_tasks)
            loss_counts = loss_dict.get("loss_counts", {})
        elif len(args) >= 9:
            # Legacy multi-task order. Ignore grade/gland.
            total = args[0]
            l_tot = args[4]
            l_dense = args[5]
            l_sparse = args[6]
            l_sparse_bce = l_sparse
            l_sparse_dice = 0.0
            l_sys = args[7]
            l_outside_gland = 0.0
            l_patient = 0.0
            loss_counts = {}
        elif len(args) >= 5:
            # Compact new order.
            total, l_tot, l_dense, l_sparse, l_sys = args[:5]
            l_sparse_bce = l_sparse
            l_sparse_dice = 0.0
            l_outside_gland = 0.0
            l_patient = 0.0
            loss_counts = {}
        else:
            total = kwargs.get("total", kwargs.get("total_loss", 0.0))
            l_tot = kwargs.get("loss_lesion_total", 0.0)
            l_dense = kwargs.get("loss_lesion_dense", 0.0)
            l_sparse = kwargs.get("loss_lesion_sparse", 0.0)
            l_sparse_bce = kwargs.get("loss_lesion_sparse_bce", l_sparse)
            l_sparse_dice = kwargs.get("loss_lesion_sparse_dice", 0.0)
            l_sys = kwargs.get("loss_lesion_sys", 0.0)
            l_outside_gland = kwargs.get("loss_lesion_outside_gland", 0.0)
            l_patient = kwargs.get("loss_lesion_patient", 0.0)
            loss_counts = kwargs.get("loss_counts", {})
        loss_counts = loss_counts or {}

        if active_tasks is not None:
            dense_active = float(active_tasks.get("lesion_dense", 0.0)) > 0
            sparse_active = float(active_tasks.get("lesion_sparse", 0.0)) > 0
            sys_active = float(active_tasks.get("lesion_sys", 0.0)) > 0
            outside_gland_active = float(active_tasks.get("lesion_outside_gland", 0.0)) > 0
            patient_active = float(active_tasks.get("lesion_patient", 0.0)) > 0
        else:
            dense_active = sparse_active = sys_active = outside_gland_active = patient_active = True

        batch_n = int(loss_counts.get("batch_size", 1) or 1)
        dense_n = int(loss_counts.get("lesion_dense_cases", 0) or 0)
        sparse_case_n = int(loss_counts.get("lesion_sparse_cases", 0) or 0)
        sparse_has_target_n = int(loss_counts.get("lesion_sparse_has_target_cases", sparse_case_n) or 0)
        sparse_sampled_n = int(loss_counts.get("lesion_sparse_sampled_cases", sparse_case_n) or 0)
        sparse_positive_n = int(loss_counts.get("lesion_sparse_positive_cases", 0) or 0)
        sparse_negative_n = int(loss_counts.get("lesion_sparse_negative_cases", 0) or 0)
        sparse_voxel_n = int(loss_counts.get("lesion_sparse_voxels", 0) or 0)
        sparse_positive_voxel_n = int(loss_counts.get("lesion_sparse_positive_voxels", 0) or 0)
        sparse_negative_voxel_n = int(loss_counts.get("lesion_sparse_negative_voxels", 0) or 0)
        sparse_dice_case_n = int(loss_counts.get("lesion_sparse_dice_cases", 0) or 0)
        sys_case_n = int(loss_counts.get("lesion_sys_cases", 0) or 0)
        sys_region_n = int(loss_counts.get("lesion_sys_regions", 0) or 0)
        outside_gland_case_n = int(loss_counts.get("lesion_outside_gland_cases", 0) or 0)
        outside_gland_voxel_n = int(loss_counts.get("lesion_outside_gland_voxels", 0) or 0)
        outside_gland_prob_mean = loss_counts.get("outside_gland_prob_mean", None)
        patient_case_n = int(loss_counts.get("lesion_patient_cases", 0) or 0)
        patient_positive_n = int(loss_counts.get("lesion_patient_positive_cases", 0) or 0)
        patient_negative_n = int(loss_counts.get("lesion_patient_negative_cases", 0) or 0)
        patient_risk_prob_mean = loss_counts.get("patient_risk_prob_mean", None)
        patient_risk_positive_prob_mean = loss_counts.get("patient_risk_positive_prob_mean", None)
        patient_risk_negative_prob_mean = loss_counts.get("patient_risk_negative_prob_mean", None)
        tbx_pos_prob_mean = loss_counts.get("tbx_pos_prob_mean", None)
        tbx_neg_prob_mean = loss_counts.get("tbx_neg_prob_mean", None)
        tbx_neg_1mp_mean = loss_counts.get("tbx_neg_1mp_mean", None)
        tbx_pos_bce = loss_counts.get("tbx_pos_bce", None)
        tbx_neg_bce = loss_counts.get("tbx_neg_bce", None)

        self.loss_num_batches += 1
        self.loss_num_cases += batch_n
        self.loss_dense_cases += dense_n
        self.loss_sparse_cases += sparse_case_n
        self.loss_sparse_has_target_cases += sparse_has_target_n
        self.loss_sparse_sampled_cases += sparse_sampled_n
        self.loss_sparse_positive_cases += sparse_positive_n
        self.loss_sparse_negative_cases += sparse_negative_n
        self.loss_sparse_voxels += sparse_voxel_n
        self.loss_sparse_positive_voxels += sparse_positive_voxel_n
        self.loss_sparse_negative_voxels += sparse_negative_voxel_n
        self.loss_sparse_dice_cases += sparse_dice_case_n
        self.loss_sys_cases += sys_case_n
        self.loss_sys_regions += sys_region_n
        # Track the auxiliary prior separately so it can be monitored without
        # conflating it with the biopsy-supervised TBx ROI BCE.
        self.loss_outside_gland_cases += outside_gland_case_n
        self.loss_outside_gland_voxels += outside_gland_voxel_n
        self.loss_patient_cases += patient_case_n
        self.loss_patient_positive_cases += patient_positive_n
        self.loss_patient_negative_cases += patient_negative_n

        self.loss_total.update(total, n=batch_n)
        self.loss_lesion.update(l_tot, n=batch_n)
        if dense_active:
            self.loss_lesion_dense.update(l_dense, n=max(dense_n, 1))
        if sparse_active:
            self.loss_lesion_sparse.update(l_sparse, n=max(sparse_voxel_n, sparse_case_n, 1))
            self.loss_lesion_sparse_bce.update(
                l_sparse_bce, n=max(sparse_voxel_n, sparse_case_n, 1)
            )
            if sparse_dice_case_n > 0:
                self.loss_lesion_sparse_dice.update(
                    l_sparse_dice, n=sparse_dice_case_n
                )
        if sys_active:
            self.loss_lesion_sys.update(l_sys, n=max(sys_region_n, sys_case_n, 1))
        if outside_gland_active:
            self.loss_lesion_outside_gland.update(
                l_outside_gland,
                n=max(outside_gland_voxel_n, outside_gland_case_n, 1),
            )
        if patient_active:
            self.loss_lesion_patient.update(l_patient, n=max(patient_case_n, 1))
        if outside_gland_voxel_n > 0:
            self.outside_gland_prob_mean.update(outside_gland_prob_mean, n=outside_gland_voxel_n)
        if patient_case_n > 0:
            self.patient_risk_prob_mean.update(patient_risk_prob_mean, n=patient_case_n)
        if patient_positive_n > 0:
            self.patient_risk_positive_prob_mean.update(patient_risk_positive_prob_mean, n=patient_positive_n)
        if patient_negative_n > 0:
            self.patient_risk_negative_prob_mean.update(patient_risk_negative_prob_mean, n=patient_negative_n)

        if sparse_positive_voxel_n > 0:
            self.tbx_pos_prob_mean.update(tbx_pos_prob_mean, n=sparse_positive_voxel_n)
            self.tbx_pos_bce.update(tbx_pos_bce, n=sparse_positive_voxel_n)
        if sparse_negative_voxel_n > 0:
            self.tbx_neg_prob_mean.update(tbx_neg_prob_mean, n=sparse_negative_voxel_n)
            self.tbx_neg_1mp_mean.update(tbx_neg_1mp_mean, n=sparse_negative_voxel_n)
            self.tbx_neg_bce.update(tbx_neg_bce, n=sparse_negative_voxel_n)

        if em_weights is not None:
            self.em_w_lesion_dense.update(em_weights.get("lesion_dense", 1.0))
            self.em_w_lesion_sparse.update(em_weights.get("lesion_sparse", 1.0))
            self.em_w_lesion_sys.update(em_weights.get("lesion_sys", 1.0))
            self.em_w_lesion_outside_gland.update(em_weights.get("lesion_outside_gland", 1.0))
            self.em_w_lesion_patient.update(em_weights.get("lesion_patient", 1.0))

        if active_tasks is not None:
            self.active_lesion_dense.update(active_tasks.get("lesion_dense", 0.0))
            self.active_lesion_sparse.update(active_tasks.get("lesion_sparse", 0.0))
            self.active_lesion_sys.update(active_tasks.get("lesion_sys", 0.0))
            self.active_lesion_outside_gland.update(active_tasks.get("lesion_outside_gland", 0.0))
            self.active_lesion_patient.update(active_tasks.get("lesion_patient", 0.0))

    def update_lesion_dice_values(self, values):
        """Update the primary dense Dice, defined inside valid gland masks."""
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        self.lesion_dice_values.extend(values.tolist())
        summary = summarise_values(self.lesion_dice_values)
        self.lesion_dice.avg = summary["mean"]
        self.lesion_dice.sum = summary["mean"] * summary["n"]
        self.lesion_dice.count = summary["n"]
        self.lesion_dice.val = float(values[-1])
        self.lesion_dice_std = summary["std"]
        self.lesion_dice_n = summary["n"]

    def update_lesion_full_crop_dice_values(self, values):
        """Retain whole-crop dense Dice as a diagnostic, not a selector."""
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        self.lesion_full_crop_dice_values.extend(values.tolist())
        summary = summarise_values(self.lesion_full_crop_dice_values)
        self.lesion_full_crop_dice.avg = summary["mean"]
        self.lesion_full_crop_dice.sum = summary["mean"] * summary["n"]
        self.lesion_full_crop_dice.count = summary["n"]
        self.lesion_full_crop_dice.val = float(values[-1])
        self.lesion_full_crop_dice_std = summary["std"]
        self.lesion_full_crop_dice_n = summary["n"]

    def update_lesion_dice_sweep(
        self,
        probs: torch.Tensor,
        target: torch.Tensor,
        scoring_mask: Optional[torch.Tensor] = None,
    ):
        """Collect validation Dice candidates inside an optional scoring mask."""
        if probs.numel() == 0 or target.numel() == 0:
            return
        if scoring_mask is not None:
            if scoring_mask.shape != probs.shape:
                raise ValueError(
                    "scoring_mask must have the same shape as dense probabilities."
                )
            scoring_mask = scoring_mask > 0
            target = target * scoring_mask.to(dtype=target.dtype)
        for threshold in self.lesion_dice_sweep_thresholds:
            pred = probs >= threshold
            if scoring_mask is not None:
                pred = pred & scoring_mask
            values = compute_dice_per_case(pred.float(), target)
            values = values[np.isfinite(values)]
            if values.size > 0:
                self.lesion_dice_sweep_values[threshold].extend(values.tolist())

    def update_lesion_gland_metrics(
        self,
        probs: torch.Tensor,
        target: torch.Tensor,
        gland_mask: torch.Tensor,
        has_gland: torch.Tensor,
        threshold: float,
        sweep_thresholds: bool = False,
        compute_operating_metrics: bool = False,
        compute_froc_metrics: bool = False,
    ) -> None:
        """Update dense-lesion metrics strictly inside valid prostate masks."""
        if probs.numel() == 0 or target.numel() == 0:
            return
        if probs.shape != target.shape or probs.shape != gland_mask.shape:
            raise ValueError(
                "probs, target, and gland_mask must have identical shapes for "
                "within-prostate evaluation."
            )

        gland = gland_mask > 0
        has_gland = has_gland.reshape(-1).to(device=probs.device).bool()
        nonempty_gland = gland.reshape(gland.size(0), -1).any(dim=1)
        valid_cases = has_gland & nonempty_gland
        self.lesion_gland_missing_cases += int((~valid_cases).sum().item())
        if not bool(valid_cases.any().item()):
            return

        probs = probs[valid_cases]
        target = (target[valid_cases] > 0).float()
        gland = gland[valid_cases]
        pred_bin = probs >= float(threshold)

        case_count = int(valid_cases.sum().item())
        gland_voxel_count = int(gland.sum().item())
        self.lesion_gland_cases += case_count
        self.lesion_gland_voxels += gland_voxel_count
        self.lesion_target_outside_gland_voxels += int(
            ((target > 0) & (~gland)).sum().item()
        )

        gland_float = gland.to(dtype=target.dtype)
        gland_dice_values = compute_dice_per_case(
            pred_bin.to(dtype=target.dtype) * gland_float,
            target * gland_float,
        )
        gland_dice_values = np.asarray(gland_dice_values, dtype=np.float64).reshape(-1)
        gland_dice_values = gland_dice_values[np.isfinite(gland_dice_values)]
        if gland_dice_values.size > 0:
            self.update_lesion_dice_values(gland_dice_values)
            self.lesion_gland_dice_values.extend(gland_dice_values.tolist())
            summary = summarise_values(self.lesion_gland_dice_values)
            self.lesion_gland_dice.avg = summary["mean"]
            self.lesion_gland_dice.sum = summary["mean"] * summary["n"]
            self.lesion_gland_dice.count = summary["n"]
            self.lesion_gland_dice.val = float(gland_dice_values[-1])
            self.lesion_gland_dice_std = summary["std"]
            self.lesion_gland_dice_n = summary["n"]

        if sweep_thresholds:
            self.update_lesion_dice_sweep(
                probs,
                target,
                scoring_mask=gland_float,
            )

        true_roi = (target[gland] > 0)
        pred_roi = pred_bin[gland]
        self.lesion_f1.update(
            compute_f1(pred_roi.to(dtype=torch.float32), true_roi.to(dtype=torch.float32))
        )
        tp = int((pred_roi & true_roi).sum().item())
        fn = int(((~pred_roi) & true_roi).sum().item())
        tn = int(((~pred_roi) & (~true_roi)).sum().item())
        fp = int((pred_roi & (~true_roi)).sum().item())
        positive_n = tp + fn
        negative_n = tn + fp
        if positive_n > 0:
            self.lesion_sens.update(tp / positive_n, n=positive_n)
        if negative_n > 0:
            self.lesion_spec.update(tn / negative_n, n=negative_n)

        if compute_froc_metrics:
            self.update_lesion_froc(
                probs,
                target,
                scoring_mask=gland_float,
            )
        if compute_operating_metrics:
            self.update_voxel_operating_samples(
                "lesion",
                true_roi.detach().cpu().numpy(),
                probs[gland].detach().cpu().numpy(),
            )

    def update_target_cspca_dice_values(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        self.target_cspca_dice_values.extend(values.tolist())
        summary = summarise_values(self.target_cspca_dice_values)
        self.target_cspca_dice.avg = summary["mean"]
        self.target_cspca_dice.sum = summary["mean"] * summary["n"]
        self.target_cspca_dice.count = summary["n"]
        self.target_cspca_dice.val = float(values[-1])
        self.target_cspca_dice_std = summary["std"]
        self.target_cspca_dice_n = summary["n"]

    def update_tbx_masked_dice_values(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        self.tbx_masked_dice_values.extend(values.tolist())
        summary = summarise_values(self.tbx_masked_dice_values)
        self.tbx_masked_dice.avg = summary["mean"]
        self.tbx_masked_dice.sum = summary["mean"] * summary["n"]
        self.tbx_masked_dice.count = summary["n"]
        self.tbx_masked_dice.val = float(values[-1])
        self.tbx_masked_dice_std = summary["std"]
        self.tbx_masked_dice_n = summary["n"]

    def update_target_cspca_aux_dice(
        self,
        probs: torch.Tensor,
        target: torch.Tensor,
        sweep_thresholds: bool = True,
    ):
        if probs.numel() == 0 or target.numel() == 0:
            return

        if sweep_thresholds:
            for threshold in self.target_cspca_dice_sweep_thresholds:
                values = compute_dice_per_case((probs >= threshold).float(), target)
                values = values[np.isfinite(values)]
                if values.size > 0:
                    self.target_cspca_dice_sweep_values[threshold].extend(values.tolist())

        topk_values = compute_topk_dice_per_case(probs, target, mode="target_volume")
        self._update_value_summary(
            topk_values,
            self.target_cspca_topk_dice_values,
            self.target_cspca_topk_dice,
            "target_cspca_topk_dice_std",
            "target_cspca_topk_dice_n",
        )

        top_percent_values = compute_topk_dice_per_case(
            probs,
            target,
            mode="percent",
            top_percent=self.target_cspca_top_percent,
        )
        self._update_value_summary(
            top_percent_values,
            self.target_cspca_top_percent_dice_values,
            self.target_cspca_top_percent_dice,
            "target_cspca_top_percent_dice_std",
            "target_cspca_top_percent_dice_n",
        )

    def finalize_target_cspca_aux_dice(
        self,
        frozen_threshold: Optional[float] = None,
    ):
        if frozen_threshold is not None:
            summary = summarise_values(self.target_cspca_dice_values)
            self.target_cspca_best_threshold = _finite_probability_threshold(
                frozen_threshold,
                self.segmentation_threshold,
            )
            self.target_cspca_best_threshold_dice = summary["mean"]
            self.target_cspca_best_threshold_dice_std = summary["std"]
            self.target_cspca_best_threshold_dice_n = summary["n"]
            return

        best_threshold = float("nan")
        best_summary = {"mean": 0.0, "std": 0.0, "n": 0}
        for threshold, values in self.target_cspca_dice_sweep_values.items():
            summary = summarise_values(values)
            if summary["n"] == 0:
                continue
            if summary["mean"] > best_summary["mean"]:
                best_threshold = float(threshold)
                best_summary = summary

        self.target_cspca_best_threshold = best_threshold
        self.target_cspca_best_threshold_dice = best_summary["mean"]
        self.target_cspca_best_threshold_dice_std = best_summary["std"]
        self.target_cspca_best_threshold_dice_n = best_summary["n"]

    @staticmethod
    def _best_dice_from_sweep(sweep_values) -> Tuple[float, Dict[str, float]]:
        best_threshold = float("nan")
        best_summary = {"mean": 0.0, "std": 0.0, "n": 0}
        for threshold, values in sweep_values.items():
            summary = summarise_values(values)
            if summary["n"] == 0:
                continue
            if summary["mean"] > best_summary["mean"]:
                best_threshold = float(threshold)
                best_summary = summary
        return best_threshold, best_summary

    def finalize_validation_dice_threshold(self) -> None:
        """Select one deployable segmentation threshold from validation Dice."""
        lesion_threshold, lesion_summary = self._best_dice_from_sweep(
            self.lesion_dice_sweep_values
        )
        self.lesion_best_threshold = lesion_threshold
        self.lesion_best_threshold_dice = lesion_summary["mean"]
        self.lesion_best_threshold_dice_std = lesion_summary["std"]
        self.lesion_best_threshold_dice_n = lesion_summary["n"]

        requested = str(
            _cfg("VALIDATION_DICE_THRESHOLD_METRIC", "lesion_dice")
        ).lower()
        candidates = {
            "lesion_dice": (
                self.lesion_best_threshold,
                self.lesion_best_threshold_dice,
                self.lesion_best_threshold_dice_n,
            ),
            "target_cspca_dice": (
                self.target_cspca_best_threshold,
                self.target_cspca_best_threshold_dice,
                self.target_cspca_best_threshold_dice_n,
            ),
        }
        fallback_order = (
            (requested,) if requested in candidates else tuple()
        ) + ("lesion_dice", "target_cspca_dice")
        for metric_name in dict.fromkeys(fallback_order):
            threshold, _, count = candidates[metric_name]
            if count > 0 and np.isfinite(threshold):
                self.segmentation_threshold = _finite_probability_threshold(
                    threshold,
                    self.segmentation_threshold,
                )
                self.segmentation_threshold_metric = metric_name
                self.threshold_source = "validation"
                return

        self.segmentation_threshold = _finite_probability_threshold(
            self.segmentation_threshold,
            _cfg("PRED_PROB_THRESHOLD", 0.5),
        )
        self.segmentation_threshold_metric = "configured_default_no_dice_gt"
        self.threshold_source = "validation"

    @staticmethod
    def _threshold_section(
        decision,
        at_fixed_specificity,
        at_fixed_sensitivity,
        default: float,
    ) -> Dict[str, float]:
        decision = _finite_probability_threshold(decision, default)
        return {
            "decision": decision,
            "at_fixed_specificity": _finite_probability_threshold(
                at_fixed_specificity, decision
            ),
            "at_fixed_sensitivity": _finite_probability_threshold(
                at_fixed_sensitivity, decision
            ),
        }

    @classmethod
    def _classification_threshold_section(
        cls,
        balanced_accuracy,
        at_fixed_specificity,
        at_fixed_sensitivity,
        default: float,
        decision_selection_rule: str,
    ) -> Dict[str, Union[str, float]]:
        """Freeze primary and secondary validation operating thresholds."""
        balanced_accuracy = _finite_probability_threshold(
            balanced_accuracy, default
        )
        section = cls._threshold_section(
            balanced_accuracy,
            at_fixed_specificity,
            at_fixed_sensitivity,
            balanced_accuracy,
        )
        rule = canonical_decision_threshold_rule(decision_selection_rule)
        if rule == "fixed_sensitivity":
            section["decision"] = section["at_fixed_sensitivity"]
        elif rule == "fixed_specificity":
            section["decision"] = section["at_fixed_specificity"]
        elif rule == "max_balanced_accuracy":
            section["decision"] = balanced_accuracy
        else:
            raise ValueError(
                f"Unsupported validation decision-threshold rule: {rule}"
            )
        section["balanced_accuracy"] = balanced_accuracy
        section["decision_selection_rule"] = rule
        return section

    def build_frozen_thresholds(self, validation_epoch: int) -> Dict[str, object]:
        """Build the versioned threshold bundle stored with a checkpoint."""
        segmentation = _finite_probability_threshold(
            self.segmentation_threshold,
            _cfg("PRED_PROB_THRESHOLD", 0.5),
        )
        segmentation_selection_metric = str(self.segmentation_threshold_metric)
        if segmentation_selection_metric not in {
            "lesion_dice",
            "target_cspca_dice",
            "configured_default_no_dice_gt",
        }:
            segmentation_selection_metric = "configured_default_no_dice_gt"
        patient_rule = canonical_decision_threshold_rule(
            _cfg("PATIENT_DECISION_THRESHOLD_RULE", "fixed_sensitivity")
        )
        region_rule = canonical_decision_threshold_rule(
            _cfg("REGION_DECISION_THRESHOLD_RULE", "fixed_specificity")
        )
        tbx_rule = canonical_decision_threshold_rule(
            _cfg("TBX_ROI_DECISION_THRESHOLD_RULE", "max_balanced_accuracy")
        )
        patient_section = self._classification_threshold_section(
            self.patient_balanced_accuracy_threshold,
            self.patient_threshold_at_fixed_spec,
            self.patient_threshold_at_fixed_sens,
            segmentation,
            patient_rule,
        )
        patient_section["pooling_mode"] = canonical_patient_pooling_mode(
            self.patient_pooling_mode
        )
        if is_contrast_patient_pooling(self.patient_pooling_mode):
            patient_section["pooling_calibration"] = {
                "mode": "logit_lme_contrast",
                "lme_r": float(self.patient_pooling_lme_r),
                "alpha": float(self.patient_pooling_alpha),
                "beta": float(self.patient_pooling_beta),
                "intercept": float(self.patient_pooling_intercept),
                "regularization_c": float(
                    self.patient_pooling_regularization_c
                ),
                "n": int(self.patient_pooling_calibration_n),
                "positive_n": int(
                    self.patient_pooling_calibration_positive_n
                ),
                "fitted": int(self.patient_pooling_calibration_fitted),
                "status": str(self.patient_pooling_calibration_status),
            }
        return {
            "schema_version": FROZEN_THRESHOLD_SCHEMA_VERSION,
            "source": "validation",
            "validation_epoch": int(validation_epoch),
            "dice": {
                "segmentation": segmentation,
                "selection_metric": segmentation_selection_metric,
                "lesion_best": _finite_probability_threshold(
                    self.lesion_best_threshold, segmentation
                ),
                "target_cspca_best": _finite_probability_threshold(
                    self.target_cspca_best_threshold, segmentation
                ),
            },
            "patient": patient_section,
            "patient_logit_lme": self._classification_threshold_section(
                self.patient_logit_lme_balanced_accuracy_threshold,
                self.patient_logit_lme_threshold_at_fixed_spec,
                self.patient_logit_lme_threshold_at_fixed_sens,
                segmentation,
                patient_rule,
            ),
            "tbx_roi": self._classification_threshold_section(
                self.tbx_roi_decision_threshold,
                self.tbx_roi_threshold_at_fixed_spec,
                self.tbx_roi_threshold_at_fixed_sens,
                segmentation,
                tbx_rule,
            ),
            "region": self._classification_threshold_section(
                self.region_balanced_accuracy_threshold,
                self.region_threshold_at_fixed_spec,
                self.region_threshold_at_fixed_sens,
                segmentation,
                region_rule,
            ),
            "lesion_voxel": self._threshold_section(
                segmentation,
                self.lesion_voxel_threshold_at_fixed_spec,
                self.lesion_voxel_threshold_at_fixed_sens,
                segmentation,
            ),
            "target_cspca_voxel": self._threshold_section(
                segmentation,
                self.target_cspca_voxel_threshold_at_fixed_spec,
                self.target_cspca_voxel_threshold_at_fixed_sens,
                segmentation,
            ),
            "targets": {
                "fixed_specificity": float(
                    _cfg("FIXED_SPECIFICITY_TARGET", 0.95)
                ),
                "fixed_sensitivity": float(
                    _cfg("FIXED_SENSITIVITY_TARGET", 0.90)
                ),
            },
            "model_pooling": {
                "sbx_mil_pooling": str(_cfg("MIL_POOLING", "lme")).lower(),
                "lme_r": float(_cfg("LME_R", 8.0)),
                "canonical_region_pooling": str(
                    _cfg("SEG_REGION_POOLING", "top_percent")
                ).lower(),
            },
        }

    def update_lesion_froc(self, probs: torch.Tensor, target: torch.Tensor, scoring_mask: Optional[torch.Tensor] = None):
        if probs.numel() == 0 or target.numel() == 0:
            return
        self.lesion_froc.update_from_maps(probs, target, scoring_mask)

    def update_target_cspca_froc(
        self,
        probs: torch.Tensor,
        target: torch.Tensor,
        scoring_mask: Optional[torch.Tensor] = None,
    ):
        if probs.numel() == 0 or target.numel() == 0:
            return
        self.target_cspca_froc.update_from_maps(probs, target, scoring_mask)

    def finalize_froc_metrics(self):
        self.lesion_froc_metrics = self.lesion_froc.compute_metrics(prefix="lesion_")
        self.target_cspca_froc_metrics = self.target_cspca_froc.compute_metrics(prefix="target_cspca_")

    def update_voxel_operating_samples(self, prefix: str, y_true, y_score):
        """Collect voxel-level labels/scores for ROC operating-point metrics."""
        y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
        y_score = np.asarray(y_score, dtype=np.float32).reshape(-1)
        valid = np.isfinite(y_score)
        if valid.size == 0 or not valid.any():
            return

        true_store = getattr(self, f"{prefix}_voxel_true")
        score_store = getattr(self, f"{prefix}_voxel_score")
        true_store.extend(y_true[valid].tolist())
        score_store.extend(y_score[valid].tolist())

    def finalize_voxel_operating_metrics(
        self,
        prefix: str,
        frozen_thresholds: Optional[Mapping] = None,
    ):
        y_true = np.asarray(getattr(self, f"{prefix}_voxel_true"), dtype=np.int64)
        y_score = np.asarray(getattr(self, f"{prefix}_voxel_score"), dtype=np.float32)
        setattr(self, f"{prefix}_voxel_n", int(len(y_true)))
        if len(y_true) == 0:
            return

        op = operating_point_metrics(
            y_true,
            y_score,
            fixed_specificity=getattr(self, f"{prefix}_voxel_fixed_spec_target"),
            fixed_sensitivity=getattr(self, f"{prefix}_voxel_fixed_sens_target"),
            threshold_at_fixed_spec=(frozen_thresholds or {}).get(
                "at_fixed_specificity"
            ),
            threshold_at_fixed_sens=(frozen_thresholds or {}).get(
                "at_fixed_sensitivity"
            ),
        )
        setattr(self, f"{prefix}_voxel_fixed_spec_target", op["fixed_spec_target"])
        setattr(self, f"{prefix}_voxel_sens_at_fixed_spec", op["sens_at_fixed_spec"])
        setattr(self, f"{prefix}_voxel_actual_spec_at_fixed_spec", op["actual_spec_at_fixed_spec"])
        setattr(self, f"{prefix}_voxel_actual_fpr_at_fixed_spec", op["actual_fpr_at_fixed_spec"])
        setattr(self, f"{prefix}_voxel_threshold_at_fixed_spec", op["threshold_at_fixed_spec"])
        setattr(self, f"{prefix}_voxel_fixed_sens_target", op["fixed_sens_target"])
        setattr(self, f"{prefix}_voxel_spec_at_fixed_sens", op["spec_at_fixed_sens"])
        setattr(self, f"{prefix}_voxel_actual_sens_at_fixed_sens", op["actual_sens_at_fixed_sens"])
        setattr(self, f"{prefix}_voxel_threshold_at_fixed_sens", op["threshold_at_fixed_sens"])

    def _update_value_summary(self, values, store, meter, std_attr: str, n_attr: str):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        store.extend(values.tolist())
        summary = summarise_values(store)
        meter.avg = summary["mean"]
        meter.sum = summary["mean"] * summary["n"]
        meter.count = summary["n"]
        meter.val = float(values[-1])
        setattr(self, std_attr, summary["std"])
        setattr(self, n_attr, summary["n"])

    def update_tbx_roi_samples(self, y_true, y_score):
        y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
        y_score = np.asarray(y_score, dtype=np.float32).reshape(-1)
        valid = np.isfinite(y_score)
        if valid.size == 0 or not valid.any():
            return
        group_id = self._next_tbx_roi_group
        self._next_tbx_roi_group += 1
        self.tbx_roi_true.extend(y_true[valid].tolist())
        self.tbx_roi_score.extend(y_score[valid].tolist())
        self.tbx_roi_group.extend([group_id] * int(valid.sum()))

    def finalize_tbx_roi_metrics(
        self,
        threshold: float,
        compute_operating_metrics: bool = False,
        compute_confidence_intervals: bool = False,
        frozen_thresholds: Optional[Mapping] = None,
        select_validation_threshold: bool = False,
    ):
        y_true = np.asarray(self.tbx_roi_true, dtype=np.int64)
        y_score = np.asarray(self.tbx_roi_score, dtype=np.float32)
        self.tbx_roi_n = int(len(y_true))
        if self.tbx_roi_n == 0:
            return

        frozen_thresholds = frozen_thresholds or {}
        if select_validation_threshold:
            threshold = select_balanced_threshold(y_true, y_score, default=threshold)
        threshold = _finite_probability_threshold(
            frozen_thresholds.get("decision", threshold), threshold
        )
        decision = binary_metrics_at_threshold(y_true, y_score, threshold)
        self.tbx_roi_decision_threshold = threshold
        self.tbx_roi_sens = decision["sens"]
        self.tbx_roi_spec = decision["spec"]
        self.tbx_roi_bacc = decision["bacc"]
        self.tbx_roi_auc = safe_auc(y_true, y_score)
        self.tbx_roi_auprc = safe_auprc(y_true, y_score)
        if compute_confidence_intervals:
            ci_metrics = bootstrap_auc_auprc_ci(
                y_true,
                y_score,
                groups=self.tbx_roi_group,
            )
            self.tbx_roi_auc_ci_low = ci_metrics["auc_ci_low"]
            self.tbx_roi_auc_ci_high = ci_metrics["auc_ci_high"]
            self.tbx_roi_auprc_ci_low = ci_metrics["auprc_ci_low"]
            self.tbx_roi_auprc_ci_high = ci_metrics["auprc_ci_high"]
            self.tbx_roi_ci_level = ci_metrics["ci_level"]
            self.tbx_roi_ci_bootstrap_valid = ci_metrics[
                "ci_bootstrap_valid"
            ]
        if not compute_operating_metrics:
            return

        op = operating_point_metrics(
            y_true,
            y_score,
            fixed_specificity=self.tbx_roi_fixed_spec_target,
            fixed_sensitivity=float(_cfg("FIXED_SENSITIVITY_TARGET", 0.90)),
            threshold_at_fixed_spec=frozen_thresholds.get(
                "at_fixed_specificity"
            ),
            threshold_at_fixed_sens=frozen_thresholds.get(
                "at_fixed_sensitivity"
            ),
        )
        self.tbx_roi_fixed_spec_target = op["fixed_spec_target"]
        self.tbx_roi_sens_at_fixed_spec = op["sens_at_fixed_spec"]
        self.tbx_roi_actual_spec_at_fixed_spec = op["actual_spec_at_fixed_spec"]
        self.tbx_roi_actual_fpr_at_fixed_spec = op["actual_fpr_at_fixed_spec"]
        self.tbx_roi_threshold_at_fixed_spec = op["threshold_at_fixed_spec"]
        self.tbx_roi_fixed_sens_target = op["fixed_sens_target"]
        self.tbx_roi_spec_at_fixed_sens = op["spec_at_fixed_sens"]
        self.tbx_roi_actual_sens_at_fixed_sens = op["actual_sens_at_fixed_sens"]
        self.tbx_roi_threshold_at_fixed_sens = op["threshold_at_fixed_sens"]

    @staticmethod
    def _ratio(num: int, den: int) -> float:
        return float(num) / float(den) if den else 0.0

    @staticmethod
    def _metric_ci_text(point: float, low: float, high: float) -> str:
        if np.isfinite(low) and np.isfinite(high):
            return f"{point:.4f} [{low:.4f}, {high:.4f}]"
        return f"{point:.4f} [CI unavailable]"

    def print_train_summary(self) -> str:
        return (
            f"Loss: {self.loss_total.avg:.4f} | "
            f"L_Les: {self.loss_lesion.avg:.4f} "
            f"(Dense {self.loss_lesion_dense.avg:.4f}, "
            f"Sparse {self.loss_lesion_sparse.avg:.4f}, "
            f"Sys {self.loss_lesion_sys.avg:.4f}, "
            f"OutGland {self.loss_lesion_outside_gland.avg:.4f}, "
            f"Patient {self.loss_lesion_patient.avg:.4f}) | "
            f"TBx p+: {self.tbx_pos_prob_mean.avg:.4f}, "
            f"p-: {self.tbx_neg_prob_mean.avg:.4f}, "
            f"1-p-: {self.tbx_neg_1mp_mean.avg:.4f}, "
            f"p(out): {self.outside_gland_prob_mean.avg:.4f}, "
            f"p(patient): {self.patient_risk_prob_mean.avg:.4f}"
        )

    def print_val_summary(self) -> str:
        primary_fp = configured_fp_per_patient_targets()[0]
        primary_fp_key = metric_key_float(primary_fp)
        lesion_froc_sens = self.lesion_froc_metrics.get(
            f"lesion_sens_at_fp_per_patient_{primary_fp_key}", 0.0
        )
        target_froc_sens = self.target_cspca_froc_metrics.get(
            f"target_cspca_sens_at_fp_per_patient_{primary_fp_key}", 0.0
        )
        if is_contrast_patient_pooling(self.patient_pooling_mode):
            patient_summary = (
                "Patient contrast AUROC/AUPRC (95% CI): "
                f"{self._metric_ci_text(self.patient_contrast_auc, self.patient_contrast_auc_ci_low, self.patient_contrast_auc_ci_high)}/"
                f"{self._metric_ci_text(self.patient_contrast_auprc, self.patient_contrast_auprc_ci_low, self.patient_contrast_auprc_ci_high)} | "
                "Patient original logit-LME AUROC/AUPRC (95% CI): "
                f"{self._metric_ci_text(self.patient_logit_lme_auc, self.patient_logit_lme_auc_ci_low, self.patient_logit_lme_auc_ci_high)}/"
                f"{self._metric_ci_text(self.patient_logit_lme_auprc, self.patient_logit_lme_auprc_ci_low, self.patient_logit_lme_auprc_ci_high)} | "
            )
        else:
            patient_summary = (
                "Patient original logit-LME AUROC/AUPRC (95% CI): "
                f"{self._metric_ci_text(self.patient_auc, self.patient_auc_ci_low, self.patient_auc_ci_high)}/"
                f"{self._metric_ci_text(self.patient_auprc, self.patient_auprc_ci_low, self.patient_auprc_ci_high)} | "
            )
        return (
            f"Loss: {self.loss_total.avg:.4f} | "
            f"Metric threshold: {self.metric_probability_threshold:.2f} | "
            f"Selected threshold: {self.segmentation_threshold:.2f} ({self.threshold_source}) | "
            f"Les-Dice gland: {self.lesion_dice.avg:.4f}+/-{self.lesion_dice_std:.4f} "
            f"(n={self.lesion_dice_n}) | "
            f"Les-Dice full (diagnostic): {self.lesion_full_crop_dice.avg:.4f}"
            f"+/-{self.lesion_full_crop_dice_std:.4f} "
            f"(n={self.lesion_full_crop_dice_n}) | "
            f"Les voxel Sens/Spec@{self.metric_probability_threshold:.2f} gland: "
            f"{self.lesion_sens.avg:.4f}/{self.lesion_spec.avg:.4f} | "
            f"Les-Sens@{primary_fp:g}FP/pat gland: {lesion_froc_sens:.4f} | "
            f"Target-csPCa Dice@{self.metric_probability_threshold:.2f}: "
            f"{self.target_cspca_dice.avg:.4f}+/-{self.target_cspca_dice_std:.4f} "
            f"(n={self.target_cspca_dice_n}) | "
            f"TBx masked Dice@{self.metric_probability_threshold:.2f}: "
            f"{self.tbx_masked_dice.avg:.4f}+/-{self.tbx_masked_dice_std:.4f} "
            f"(n={self.tbx_masked_dice_n}) | "
            f"DenseBestThrDice(gland): {self.lesion_best_threshold_dice:.4f}"
            f"@{self.lesion_best_threshold:.2f} | "
            f"TargetBestThrDice: {self.target_cspca_best_threshold_dice:.4f}"
            f"@{self.target_cspca_best_threshold:.2f} | "
            f"TopKDice: {self.target_cspca_topk_dice.avg:.4f} | "
            f"Target-csPCa Sens@{primary_fp:g}FP/case: {target_froc_sens:.4f} | "
            f"TBx p+: {self.tbx_pos_prob_mean.avg:.4f}, "
            f"p-: {self.tbx_neg_prob_mean.avg:.4f}, "
            f"p(out): {self.outside_gland_prob_mean.avg:.4f}, "
            f"p(patient): {self.patient_risk_prob_mean.avg:.4f}, "
            "TBx ROI AUROC/AUPRC (95% CI): "
            f"{self._metric_ci_text(self.tbx_roi_auc, self.tbx_roi_auc_ci_low, self.tbx_roi_auc_ci_high)}/"
            f"{self._metric_ci_text(self.tbx_roi_auprc, self.tbx_roi_auprc_ci_low, self.tbx_roi_auprc_ci_high)} | "
            f"{patient_summary}"
            "Region AUROC/AUPRC (95% CI): "
            f"{self._metric_ci_text(self.region_auc, self.region_auc_ci_low, self.region_auc_ci_high)}/"
            f"{self._metric_ci_text(self.region_auprc, self.region_auprc_ci_low, self.region_auprc_ci_high)} | "
            f"SBx-MIL[{self.sbx_mil_pooling_mode}] AUPRC: "
            f"{self.sbx_mil_region_auprc:.4f} (n={self.sbx_mil_region_n}) | "
            f"Pat Sens@Spec{self.patient_fixed_spec_target:.2f}: {self.patient_sens_at_fixed_spec:.4f} | "
            f"Pat Spec@Sens{self.patient_fixed_sens_target:.2f}: {self.patient_spec_at_fixed_sens:.4f} | "
            f"Pat TP/FP/FN/TN: {self.patient_tp}/{self.patient_fp}/{self.patient_fn}/{self.patient_tn} | "
            f"Region Sens@Spec{self.region_fixed_spec_target:.2f}: {self.region_sens_at_fixed_spec:.4f} | "
            f"Region TP/FP/FN/TN: {self.region_tp}/{self.region_fp}/{self.region_fn}/{self.region_tn}"
        )

    def get_train_dict(self) -> Dict[str, float]:
        train_dict = {
            "train_loss_total": self.loss_total.avg,
            "train_loss_lesion": self.loss_lesion.avg,
            "train_loss_lesion_dense": self.loss_lesion_dense.avg,
            "train_loss_lesion_sparse": self.loss_lesion_sparse.avg,
            "train_loss_lesion_sparse_bce": self.loss_lesion_sparse_bce.avg,
            "train_loss_lesion_sparse_dice": self.loss_lesion_sparse_dice.avg,
            "train_loss_lesion_sys": self.loss_lesion_sys.avg,
            "train_loss_lesion_outside_gland": self.loss_lesion_outside_gland.avg,
            "train_loss_lesion_patient": self.loss_lesion_patient.avg,
            "em_w_lesion_dense": self.em_w_lesion_dense.avg,
            "em_w_lesion_sparse": self.em_w_lesion_sparse.avg,
            "em_w_lesion_sys": self.em_w_lesion_sys.avg,
            "em_w_lesion_outside_gland": self.em_w_lesion_outside_gland.avg,
            "em_w_lesion_patient": self.em_w_lesion_patient.avg,
            "active_lesion_dense": self.active_lesion_dense.avg,
            "active_lesion_sparse": self.active_lesion_sparse.avg,
            "active_lesion_sys": self.active_lesion_sys.avg,
            "active_lesion_outside_gland": self.active_lesion_outside_gland.avg,
            "active_lesion_patient": self.active_lesion_patient.avg,
            "active_lesion_dense_batch_rate": self.active_lesion_dense.avg,
            "active_lesion_sparse_batch_rate": self.active_lesion_sparse.avg,
            "active_lesion_sys_batch_rate": self.active_lesion_sys.avg,
            "active_lesion_outside_gland_batch_rate": self.active_lesion_outside_gland.avg,
            "active_lesion_patient_batch_rate": self.active_lesion_patient.avg,
            "train_loss_num_batches": self.loss_num_batches,
            "train_loss_num_cases": self.loss_num_cases,
            "train_loss_dense_cases": self.loss_dense_cases,
            "train_loss_sparse_cases": self.loss_sparse_cases,
            "train_loss_sparse_has_target_cases": self.loss_sparse_has_target_cases,
            "train_loss_sparse_sampled_cases": self.loss_sparse_sampled_cases,
            "train_loss_sparse_positive_cases": self.loss_sparse_positive_cases,
            "train_loss_sparse_negative_cases": self.loss_sparse_negative_cases,
            "train_loss_sparse_positive_case_rate": self._ratio(
                self.loss_sparse_positive_cases, self.loss_sparse_has_target_cases
            ),
            "train_loss_sparse_negative_case_rate": self._ratio(
                self.loss_sparse_negative_cases, self.loss_sparse_has_target_cases
            ),
            "train_loss_sparse_voxels": self.loss_sparse_voxels,
            "train_loss_sparse_positive_voxels": self.loss_sparse_positive_voxels,
            "train_loss_sparse_negative_voxels": self.loss_sparse_negative_voxels,
            "train_loss_sparse_dice_cases": self.loss_sparse_dice_cases,
            "train_tbx_pos_prob_mean": self.tbx_pos_prob_mean.avg,
            "train_tbx_neg_prob_mean": self.tbx_neg_prob_mean.avg,
            "train_tbx_neg_1mp_mean": self.tbx_neg_1mp_mean.avg,
            "train_tbx_pos_bce": self.tbx_pos_bce.avg,
            "train_tbx_neg_bce": self.tbx_neg_bce.avg,
            "train_loss_outside_gland_cases": self.loss_outside_gland_cases,
            "train_loss_outside_gland_voxels": self.loss_outside_gland_voxels,
            "train_outside_gland_prob_mean": self.outside_gland_prob_mean.avg,
            "train_loss_patient_cases": self.loss_patient_cases,
            "train_loss_patient_positive_cases": self.loss_patient_positive_cases,
            "train_loss_patient_negative_cases": self.loss_patient_negative_cases,
            "train_patient_risk_prob_mean": self.patient_risk_prob_mean.avg,
            "train_patient_risk_positive_prob_mean": self.patient_risk_positive_prob_mean.avg,
            "train_patient_risk_negative_prob_mean": self.patient_risk_negative_prob_mean.avg,
            "train_loss_sys_cases": self.loss_sys_cases,
            "train_loss_sys_regions": self.loss_sys_regions,
        }
        train_dict.update(self.get_entropy_dict("train"))
        return train_dict

    def get_val_dict(self) -> Dict[str, float]:
        val_dict = {
            "val_loss_total": self.loss_total.avg,
            "val_loss_lesion": self.loss_lesion.avg,
            "val_loss_lesion_dense": self.loss_lesion_dense.avg,
            "val_loss_lesion_sparse": self.loss_lesion_sparse.avg,
            "val_loss_lesion_sparse_bce": self.loss_lesion_sparse_bce.avg,
            "val_loss_lesion_sparse_dice": self.loss_lesion_sparse_dice.avg,
            "val_loss_lesion_sys": self.loss_lesion_sys.avg,
            "val_loss_lesion_outside_gland": self.loss_lesion_outside_gland.avg,
            "val_loss_lesion_patient": self.loss_lesion_patient.avg,
            "val_lesion_dice": self.lesion_dice.avg,
            "val_lesion_dice_mean": self.lesion_dice.avg,
            "val_lesion_dice_std": self.lesion_dice_std,
            "val_lesion_dice_n": self.lesion_dice_n,
            "val_lesion_best_threshold_dice": self.lesion_best_threshold_dice,
            "val_lesion_best_threshold_dice_mean": self.lesion_best_threshold_dice,
            "val_lesion_best_threshold_dice_std": self.lesion_best_threshold_dice_std,
            "val_lesion_best_threshold_dice_n": self.lesion_best_threshold_dice_n,
            "val_lesion_best_threshold": self.lesion_best_threshold,
            "val_segmentation_threshold": self.segmentation_threshold,
            "val_metric_probability_threshold": self.metric_probability_threshold,
            "val_segmentation_threshold_metric": self.segmentation_threshold_metric,
            "val_threshold_source": self.threshold_source,
            "val_threshold_validation_epoch": self.threshold_validation_epoch,
            "val_lesion_full_crop_dice": self.lesion_full_crop_dice.avg,
            "val_lesion_full_crop_dice_mean": self.lesion_full_crop_dice.avg,
            "val_lesion_full_crop_dice_std": self.lesion_full_crop_dice_std,
            "val_lesion_full_crop_dice_n": self.lesion_full_crop_dice_n,
            "val_lesion_gland_dice": self.lesion_gland_dice.avg,
            "val_lesion_gland_dice_mean": self.lesion_gland_dice.avg,
            "val_lesion_gland_dice_std": self.lesion_gland_dice_std,
            "val_lesion_gland_dice_n": self.lesion_gland_dice_n,
            "val_lesion_gland_cases": self.lesion_gland_cases,
            "val_lesion_gland_missing_cases": self.lesion_gland_missing_cases,
            "val_lesion_gland_voxels": self.lesion_gland_voxels,
            "val_lesion_target_outside_gland_voxels": self.lesion_target_outside_gland_voxels,
            "val_lesion_f1": self.lesion_f1.avg,
            "val_lesion_sens": self.lesion_sens.avg,
            "val_lesion_spec": self.lesion_spec.avg,
            "val_lesion_gland_sens_at_prob_threshold": self.lesion_sens.avg,
            "val_lesion_gland_spec_at_prob_threshold": self.lesion_spec.avg,
            "val_lesion_gland_sens_at_0p5": (
                self.lesion_sens.avg
                if np.isclose(self.metric_probability_threshold, 0.5)
                else float("nan")
            ),
            "val_lesion_gland_spec_at_0p5": (
                self.lesion_spec.avg
                if np.isclose(self.metric_probability_threshold, 0.5)
                else float("nan")
            ),
            "val_lesion_voxel_n": self.lesion_voxel_n,
            "val_lesion_voxel_fixed_spec_target": self.lesion_voxel_fixed_spec_target,
            "val_lesion_voxel_sens_at_fixed_spec": self.lesion_voxel_sens_at_fixed_spec,
            "val_lesion_voxel_actual_spec_at_fixed_spec": self.lesion_voxel_actual_spec_at_fixed_spec,
            "val_lesion_voxel_actual_fpr_at_fixed_spec": self.lesion_voxel_actual_fpr_at_fixed_spec,
            "val_lesion_voxel_threshold_at_fixed_spec": self.lesion_voxel_threshold_at_fixed_spec,
            "val_lesion_voxel_fixed_sens_target": self.lesion_voxel_fixed_sens_target,
            "val_lesion_voxel_spec_at_fixed_sens": self.lesion_voxel_spec_at_fixed_sens,
            "val_lesion_voxel_actual_sens_at_fixed_sens": self.lesion_voxel_actual_sens_at_fixed_sens,
            "val_lesion_voxel_threshold_at_fixed_sens": self.lesion_voxel_threshold_at_fixed_sens,
            "val_lesion_gland_voxel_n": self.lesion_voxel_n,
            "val_lesion_gland_fixed_spec_target": self.lesion_voxel_fixed_spec_target,
            "val_lesion_gland_sens_at_fixed_spec": self.lesion_voxel_sens_at_fixed_spec,
            "val_lesion_gland_actual_spec_at_fixed_spec": self.lesion_voxel_actual_spec_at_fixed_spec,
            "val_lesion_gland_threshold_at_fixed_spec": self.lesion_voxel_threshold_at_fixed_spec,
            "val_lesion_gland_fixed_sens_target": self.lesion_voxel_fixed_sens_target,
            "val_lesion_gland_spec_at_fixed_sens": self.lesion_voxel_spec_at_fixed_sens,
            "val_lesion_gland_actual_sens_at_fixed_sens": self.lesion_voxel_actual_sens_at_fixed_sens,
            "val_lesion_gland_threshold_at_fixed_sens": self.lesion_voxel_threshold_at_fixed_sens,
            "val_target_cspca_dice": self.target_cspca_dice.avg,
            "val_target_cspca_dice_at_prob_threshold": self.target_cspca_dice.avg,
            "val_target_cspca_dice_mean": self.target_cspca_dice.avg,
            "val_target_cspca_dice_std": self.target_cspca_dice_std,
            "val_target_cspca_dice_n": self.target_cspca_dice_n,
            "val_tbx_masked_dice": self.tbx_masked_dice.avg,
            "val_tbx_masked_dice_mean": self.tbx_masked_dice.avg,
            "val_tbx_masked_dice_std": self.tbx_masked_dice_std,
            "val_tbx_masked_dice_n": self.tbx_masked_dice_n,
            "val_target_cspca_best_threshold_dice": self.target_cspca_best_threshold_dice,
            "val_target_cspca_best_threshold_dice_mean": self.target_cspca_best_threshold_dice,
            "val_target_cspca_best_threshold_dice_std": self.target_cspca_best_threshold_dice_std,
            "val_target_cspca_best_threshold_dice_n": self.target_cspca_best_threshold_dice_n,
            "val_target_cspca_best_threshold": self.target_cspca_best_threshold,
            # Preferred protocol-aware aliases. The legacy best-threshold fields
            # above remain for existing report scripts.
            "val_target_cspca_selected_threshold_dice": self.target_cspca_best_threshold_dice,
            "val_target_cspca_selected_threshold_dice_mean": self.target_cspca_best_threshold_dice,
            "val_target_cspca_selected_threshold_dice_std": self.target_cspca_best_threshold_dice_std,
            "val_target_cspca_selected_threshold_dice_n": self.target_cspca_best_threshold_dice_n,
            "val_target_cspca_selected_threshold": self.target_cspca_best_threshold,
            "val_target_cspca_threshold_source": self.threshold_source,
            "val_target_cspca_topk_dice": self.target_cspca_topk_dice.avg,
            "val_target_cspca_topk_dice_mean": self.target_cspca_topk_dice.avg,
            "val_target_cspca_topk_dice_std": self.target_cspca_topk_dice_std,
            "val_target_cspca_topk_dice_n": self.target_cspca_topk_dice_n,
            "val_target_cspca_top_percent": self.target_cspca_top_percent,
            "val_target_cspca_top_percent_dice": self.target_cspca_top_percent_dice.avg,
            "val_target_cspca_top_percent_dice_mean": self.target_cspca_top_percent_dice.avg,
            "val_target_cspca_top_percent_dice_std": self.target_cspca_top_percent_dice_std,
            "val_target_cspca_top_percent_dice_n": self.target_cspca_top_percent_dice_n,
            "val_target_cspca_voxel_n": self.target_cspca_voxel_n,
            "val_target_cspca_voxel_fixed_spec_target": self.target_cspca_voxel_fixed_spec_target,
            "val_target_cspca_voxel_sens_at_fixed_spec": self.target_cspca_voxel_sens_at_fixed_spec,
            "val_target_cspca_voxel_actual_spec_at_fixed_spec": self.target_cspca_voxel_actual_spec_at_fixed_spec,
            "val_target_cspca_voxel_actual_fpr_at_fixed_spec": self.target_cspca_voxel_actual_fpr_at_fixed_spec,
            "val_target_cspca_voxel_threshold_at_fixed_spec": self.target_cspca_voxel_threshold_at_fixed_spec,
            "val_target_cspca_voxel_fixed_sens_target": self.target_cspca_voxel_fixed_sens_target,
            "val_target_cspca_voxel_spec_at_fixed_sens": self.target_cspca_voxel_spec_at_fixed_sens,
            "val_target_cspca_voxel_actual_sens_at_fixed_sens": self.target_cspca_voxel_actual_sens_at_fixed_sens,
            "val_target_cspca_voxel_threshold_at_fixed_sens": self.target_cspca_voxel_threshold_at_fixed_sens,
            "val_tbx_roi_bacc": self.tbx_roi_bacc,
            "val_tbx_roi_sens": self.tbx_roi_sens,
            "val_tbx_roi_spec": self.tbx_roi_spec,
            "val_tbx_roi_auc": self.tbx_roi_auc,
            "val_tbx_roi_auprc": self.tbx_roi_auprc,
            "val_tbx_roi_auc_ci_low": self.tbx_roi_auc_ci_low,
            "val_tbx_roi_auc_ci_high": self.tbx_roi_auc_ci_high,
            "val_tbx_roi_auprc_ci_low": self.tbx_roi_auprc_ci_low,
            "val_tbx_roi_auprc_ci_high": self.tbx_roi_auprc_ci_high,
            "val_tbx_roi_ci_level": self.tbx_roi_ci_level,
            "val_tbx_roi_ci_bootstrap_valid": self.tbx_roi_ci_bootstrap_valid,
            "val_tbx_roi_n": self.tbx_roi_n,
            "val_tbx_roi_decision_threshold": self.tbx_roi_decision_threshold,
            "val_tbx_roi_fixed_spec_target": self.tbx_roi_fixed_spec_target,
            "val_tbx_roi_sens_at_fixed_spec": self.tbx_roi_sens_at_fixed_spec,
            "val_tbx_roi_actual_spec_at_fixed_spec": self.tbx_roi_actual_spec_at_fixed_spec,
            "val_tbx_roi_actual_fpr_at_fixed_spec": self.tbx_roi_actual_fpr_at_fixed_spec,
            "val_tbx_roi_threshold_at_fixed_spec": self.tbx_roi_threshold_at_fixed_spec,
            "val_tbx_roi_fixed_sens_target": self.tbx_roi_fixed_sens_target,
            "val_tbx_roi_spec_at_fixed_sens": self.tbx_roi_spec_at_fixed_sens,
            "val_tbx_roi_actual_sens_at_fixed_sens": self.tbx_roi_actual_sens_at_fixed_sens,
            "val_tbx_roi_threshold_at_fixed_sens": self.tbx_roi_threshold_at_fixed_sens,
            "val_patient_bacc": self.patient_bacc,
            "val_patient_sens": self.patient_sens,
            "val_patient_spec": self.patient_spec,
            "val_patient_auc": self.patient_auc,
            "val_patient_auprc": self.patient_auprc,
            "val_patient_auc_ci_low": self.patient_auc_ci_low,
            "val_patient_auc_ci_high": self.patient_auc_ci_high,
            "val_patient_auprc_ci_low": self.patient_auprc_ci_low,
            "val_patient_auprc_ci_high": self.patient_auprc_ci_high,
            "val_patient_ci_level": self.patient_ci_level,
            "val_patient_ci_bootstrap_valid": self.patient_ci_bootstrap_valid,
            "val_patient_n": self.patient_n,
            "val_patient_tn": self.patient_tn,
            "val_patient_fp": self.patient_fp,
            "val_patient_fn": self.patient_fn,
            "val_patient_tp": self.patient_tp,
            "val_patient_decision_threshold": self.patient_decision_threshold,
            "val_patient_balanced_accuracy_threshold": self.patient_balanced_accuracy_threshold,
            "val_patient_sens_at_balanced_accuracy": self.patient_sens_at_balanced_accuracy,
            "val_patient_spec_at_balanced_accuracy": self.patient_spec_at_balanced_accuracy,
            "val_patient_bacc_at_balanced_accuracy": self.patient_bacc_at_balanced_accuracy,
            "val_patient_fixed_spec_target": self.patient_fixed_spec_target,
            "val_patient_sens_at_fixed_spec": self.patient_sens_at_fixed_spec,
            "val_patient_actual_spec_at_fixed_spec": self.patient_actual_spec_at_fixed_spec,
            "val_patient_actual_fpr_at_fixed_spec": self.patient_actual_fpr_at_fixed_spec,
            "val_patient_threshold_at_fixed_spec": self.patient_threshold_at_fixed_spec,
            "val_patient_fixed_sens_target": self.patient_fixed_sens_target,
            "val_patient_spec_at_fixed_sens": self.patient_spec_at_fixed_sens,
            "val_patient_actual_sens_at_fixed_sens": self.patient_actual_sens_at_fixed_sens,
            "val_patient_threshold_at_fixed_sens": self.patient_threshold_at_fixed_sens,
            "val_patient_pooling_mode": self.patient_pooling_mode,
            "val_patient_pooling_lme_r": self.patient_pooling_lme_r,
            "val_patient_pooling_alpha": self.patient_pooling_alpha,
            "val_patient_pooling_beta": self.patient_pooling_beta,
            "val_patient_pooling_intercept": self.patient_pooling_intercept,
            "val_patient_pooling_regularization_c": self.patient_pooling_regularization_c,
            "val_patient_pooling_calibration_n": self.patient_pooling_calibration_n,
            "val_patient_pooling_calibration_positive_n": self.patient_pooling_calibration_positive_n,
            "val_patient_pooling_calibration_fitted": self.patient_pooling_calibration_fitted,
            "val_patient_pooling_calibration_status": self.patient_pooling_calibration_status,
            "val_region_bacc": self.region_bacc,
            "val_region_sens": self.region_sens,
            "val_region_spec": self.region_spec,
            "val_region_auc": self.region_auc,
            "val_region_auprc": self.region_auprc,
            "val_region_auc_ci_low": self.region_auc_ci_low,
            "val_region_auc_ci_high": self.region_auc_ci_high,
            "val_region_auprc_ci_low": self.region_auprc_ci_low,
            "val_region_auprc_ci_high": self.region_auprc_ci_high,
            "val_region_ci_level": self.region_ci_level,
            "val_region_ci_bootstrap_valid": self.region_ci_bootstrap_valid,
            "val_region_n": self.region_n,
            "val_region_tn": self.region_tn,
            "val_region_fp": self.region_fp,
            "val_region_fn": self.region_fn,
            "val_region_tp": self.region_tp,
            "val_region_decision_threshold": self.region_decision_threshold,
            "val_region_balanced_accuracy_threshold": self.region_balanced_accuracy_threshold,
            "val_region_sens_at_balanced_accuracy": self.region_sens_at_balanced_accuracy,
            "val_region_spec_at_balanced_accuracy": self.region_spec_at_balanced_accuracy,
            "val_region_bacc_at_balanced_accuracy": self.region_bacc_at_balanced_accuracy,
            "val_region_fixed_spec_target": self.region_fixed_spec_target,
            "val_region_sens_at_fixed_spec": self.region_sens_at_fixed_spec,
            "val_region_actual_spec_at_fixed_spec": self.region_actual_spec_at_fixed_spec,
            "val_region_actual_fpr_at_fixed_spec": self.region_actual_fpr_at_fixed_spec,
            "val_region_threshold_at_fixed_spec": self.region_threshold_at_fixed_spec,
            "val_region_fixed_sens_target": self.region_fixed_sens_target,
            "val_region_spec_at_fixed_sens": self.region_spec_at_fixed_sens,
            "val_region_actual_sens_at_fixed_sens": self.region_actual_sens_at_fixed_sens,
            "val_region_threshold_at_fixed_sens": self.region_threshold_at_fixed_sens,
            "val_sbx_mil_pooling_mode": self.sbx_mil_pooling_mode,
            "val_sbx_mil_lme_r": self.sbx_mil_lme_r,
            "val_sbx_mil_region_bacc": self.sbx_mil_region_bacc,
            "val_sbx_mil_region_sens": self.sbx_mil_region_sens,
            "val_sbx_mil_region_spec": self.sbx_mil_region_spec,
            "val_sbx_mil_region_auc": self.sbx_mil_region_auc,
            "val_sbx_mil_region_auprc": self.sbx_mil_region_auprc,
            "val_sbx_mil_region_n": self.sbx_mil_region_n,
            "val_sbx_mil_region_tn": self.sbx_mil_region_tn,
            "val_sbx_mil_region_fp": self.sbx_mil_region_fp,
            "val_sbx_mil_region_fn": self.sbx_mil_region_fn,
            "val_sbx_mil_region_tp": self.sbx_mil_region_tp,
            "val_loss_num_batches": self.loss_num_batches,
            "val_loss_num_cases": self.loss_num_cases,
            "val_loss_dense_cases": self.loss_dense_cases,
            "val_loss_sparse_cases": self.loss_sparse_cases,
            "val_loss_sparse_has_target_cases": self.loss_sparse_has_target_cases,
            "val_loss_sparse_sampled_cases": self.loss_sparse_sampled_cases,
            "val_loss_sparse_positive_cases": self.loss_sparse_positive_cases,
            "val_loss_sparse_negative_cases": self.loss_sparse_negative_cases,
            "val_loss_sparse_positive_case_rate": self._ratio(
                self.loss_sparse_positive_cases, self.loss_sparse_has_target_cases
            ),
            "val_loss_sparse_negative_case_rate": self._ratio(
                self.loss_sparse_negative_cases, self.loss_sparse_has_target_cases
            ),
            "val_loss_sparse_voxels": self.loss_sparse_voxels,
            "val_loss_sparse_positive_voxels": self.loss_sparse_positive_voxels,
            "val_loss_sparse_negative_voxels": self.loss_sparse_negative_voxels,
            "val_loss_sparse_dice_cases": self.loss_sparse_dice_cases,
            "val_tbx_pos_prob_mean": self.tbx_pos_prob_mean.avg,
            "val_tbx_neg_prob_mean": self.tbx_neg_prob_mean.avg,
            "val_tbx_neg_1mp_mean": self.tbx_neg_1mp_mean.avg,
            "val_tbx_pos_bce": self.tbx_pos_bce.avg,
            "val_tbx_neg_bce": self.tbx_neg_bce.avg,
            "val_loss_outside_gland_cases": self.loss_outside_gland_cases,
            "val_loss_outside_gland_voxels": self.loss_outside_gland_voxels,
            "val_outside_gland_prob_mean": self.outside_gland_prob_mean.avg,
            "val_loss_patient_cases": self.loss_patient_cases,
            "val_loss_patient_positive_cases": self.loss_patient_positive_cases,
            "val_loss_patient_negative_cases": self.loss_patient_negative_cases,
            "val_patient_risk_prob_mean": self.patient_risk_prob_mean.avg,
            "val_patient_risk_positive_prob_mean": self.patient_risk_positive_prob_mean.avg,
            "val_patient_risk_negative_prob_mean": self.patient_risk_negative_prob_mean.avg,
            "val_loss_sys_cases": self.loss_sys_cases,
            "val_loss_sys_regions": self.loss_sys_regions,
        }
        val_dict.update({f"val_{key}": value for key, value in self.lesion_froc_metrics.items()})
        val_dict.update(
            {
                f"val_lesion_gland_{key[len('lesion_'):]}": value
                for key, value in self.lesion_froc_metrics.items()
                if key.startswith("lesion_")
            }
        )
        val_dict.update({f"val_{key}": value for key, value in self.target_cspca_froc_metrics.items()})
        for score_family in ("patient_contrast", "patient_logit_lme"):
            for field in self.PATIENT_SCORE_METRIC_FIELDS:
                val_dict[f"val_{score_family}_{field}"] = getattr(
                    self,
                    f"{score_family}_{field}",
                )
        val_dict.update(self.get_entropy_dict("val"))
        return val_dict


# -----------------------------------------------------------------------------
# Validation loop
# -----------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    epoch,
    save_dir,
    compute_operating_metrics: bool = False,
    compute_froc_metrics: bool = False,
    compute_confidence_intervals: bool = False,
    sample_exporter=None,
    frozen_thresholds: Optional[Mapping] = None,
):
    """Validate the segmentation + MIL model.

    This function accepts the new dict-output model and loss. It also tolerates
    the old 5-tuple model output for easier transition, but only lesion-related
    outputs are used.
    """
    model.eval()
    tracker = MetricTracker()

    positive_threshold = int(_cfg("LESION_POSITIVE_THRESHOLD", _cfg("CSPC_THRESHOLD", 1)))
    if frozen_thresholds is not None and not has_frozen_validation_thresholds(
        frozen_thresholds
    ):
        raise ValueError(
            "frozen_thresholds must be a validation-derived threshold bundle."
        )
    using_frozen_thresholds = has_frozen_validation_thresholds(frozen_thresholds)
    configured_threshold = float(_cfg("PRED_PROB_THRESHOLD", 0.5))
    prob_threshold = frozen_threshold_value(
        frozen_thresholds,
        "dice",
        "segmentation",
        configured_threshold,
    )
    invalid_sys_label = int(_cfg("INVALID_SYS_LABEL", -1))

    patient_thresholds = (
        frozen_thresholds.get("patient", {}) if using_frozen_thresholds else {}
    )
    patient_logit_lme_thresholds = (
        frozen_thresholds.get("patient_logit_lme", {})
        if using_frozen_thresholds
        else {}
    )
    region_thresholds = (
        frozen_thresholds.get("region", {}) if using_frozen_thresholds else {}
    )
    tbx_roi_thresholds = (
        frozen_thresholds.get("tbx_roi", {}) if using_frozen_thresholds else {}
    )
    tracker.metric_probability_threshold = prob_threshold
    tracker.segmentation_threshold = prob_threshold
    if using_frozen_thresholds:
        tracker.threshold_source = "validation_frozen"
        tracker.threshold_validation_epoch = int(
            frozen_thresholds.get("validation_epoch", 0)
        )
        tracker.segmentation_threshold_metric = str(
            frozen_thresholds.get("dice", {}).get(
                "selection_metric", "validation"
            )
        )

    seg_evaluator = SegRiskMapEvaluator(
        prob_threshold=prob_threshold,
        positive_threshold=positive_threshold,
        invalid_sys_label=invalid_sys_label,
        patient_threshold=(patient_thresholds or {}).get("decision"),
        region_threshold=(region_thresholds or {}).get("decision"),
        tbx_roi_threshold=(tbx_roi_thresholds or {}).get("decision"),
        patient_operating_thresholds=patient_thresholds,
        patient_logit_lme_threshold=(patient_logit_lme_thresholds or {}).get(
            "decision"
        ),
        patient_logit_lme_operating_thresholds=patient_logit_lme_thresholds,
        patient_pooling_calibration=(patient_thresholds or {}).get(
            "pooling_calibration"
        ),
        region_operating_thresholds=region_thresholds,
        select_validation_thresholds=not using_frozen_thresholds,
        compute_confidence_intervals=compute_confidence_intervals,
    )
    sbx_mil_evaluator = LesionMILEvaluator(
        prob_threshold=prob_threshold,
        positive_threshold=positive_threshold,
        invalid_sys_label=invalid_sys_label,
    )

    saved_counts = {"PUB": 0, "TCIA": 0, "PROMIS": 0, "OTHER": 0}
    max_saves_per_type = int(_cfg("VAL_VIS_MAX_PER_TYPE", 2))
    # Validation visualisations are deterministic per dataset type and epoch:
    # save only the first few cases per source instead of random sampling.
    save_val_vis = (
        bool(_cfg("SAVE_VAL_VIS", False))
        and bool(save_dir)
        and int(_cfg("VAL_VIS_EVERY_N_EPOCHS", 0)) > 0
        and epoch % int(_cfg("VAL_VIS_EVERY_N_EPOCHS", 0)) == 0
        and max_saves_per_type > 0
    )
    vis_dir = os.path.join(save_dir, _cfg("VIS_SUBDIR", "visualizations"), "val", f"epoch_{epoch:03d}")
    vis_dir_created = False

    for batch in tqdm(loader, desc="Validation"):
        batch = move_batch_to_device(batch, device)
        imgs = batch["input"]
        zones_mask = batch.get("zones_mask", None)

        raw_outputs = model(imgs, zones_mask)
        outputs = unpack_model_output(raw_outputs)
        lesion_logits = outputs["lesion_logits"]
        if lesion_logits is None:
            raise ValueError("Model output does not contain lesion logits.")

        loss_output = call_criterion(criterion, outputs, batch)
        loss_dict = normalise_loss_output(loss_output)
        tracker.update_losses(loss_dict)
        if bool(_cfg("MONITOR_PREDICTION_ENTROPY", True)):
            tracker.update_prediction_entropy(outputs, batch)

        lesion_probs = torch.sigmoid(lesion_logits)

        # Dense segmentation metrics only for PUB/radiologist-annotated cases.
        if "has_lesion" in batch and batch["has_lesion"].sum() > 0:
            idx = batch["has_lesion"] > 0
            pred_bin = (lesion_probs[idx] >= prob_threshold).float()
            target = batch["lesion_mask"][idx].float()
            tracker.update_lesion_full_crop_dice_values(
                compute_dice_per_case(pred_bin, target)
            )
            gland_mask = batch.get("gland_mask", torch.zeros_like(batch["lesion_mask"]))[idx]
            if "has_gland" in batch:
                has_gland = batch["has_gland"][idx]
            else:
                has_gland = gland_mask.reshape(gland_mask.size(0), -1).any(dim=1)
            tracker.update_lesion_gland_metrics(
                lesion_probs[idx],
                target,
                gland_mask.float(),
                has_gland,
                threshold=prob_threshold,
                sweep_thresholds=not using_frozen_thresholds,
                compute_operating_metrics=compute_operating_metrics,
                compute_froc_metrics=compute_froc_metrics,
            )

        # B-series csPCa localisation metric on biopsy-confirmed target ROIs.
        if "has_target" in batch and "target_mask" in batch and batch["has_target"].sum() > 0:
            target_cspca = (batch["target_mask"] >= positive_threshold).float()
            target_cases = batch["has_target"] > 0
            if compute_froc_metrics:
                target_froc_mask = None
                if "gland_mask" in batch:
                    target_froc_mask = batch["gland_mask"][target_cases].float()
                tracker.update_target_cspca_froc(
                    lesion_probs[target_cases],
                    target_cspca[target_cases],
                    scoring_mask=target_froc_mask,
                )
            positive_target_cases = (batch["has_target"] > 0) & target_cspca.reshape(target_cspca.size(0), -1).any(dim=1)
            if positive_target_cases.any():
                positive_probs = lesion_probs[positive_target_cases]
                positive_target = target_cspca[positive_target_cases]
                pred_bin = (positive_probs >= prob_threshold).float()
                tracker.update_target_cspca_dice_values(
                    compute_dice_per_case(pred_bin, positive_target)
                )
                tracker.update_target_cspca_aux_dice(
                    positive_probs,
                    positive_target,
                    sweep_thresholds=not using_frozen_thresholds,
                )

            tracker.update_tbx_masked_dice_values(
                compute_masked_tbx_dice_per_case(
                    lesion_probs[target_cases],
                    batch["target_mask"][target_cases],
                    positive_threshold=positive_threshold,
                    prob_threshold=prob_threshold,
                )
            )

            sampled_tbx_roi = (batch["has_target"] > 0).view(-1, 1, 1, 1, 1) & (batch["target_mask"] > 0)
            if sampled_tbx_roi.any():
                for case_idx in torch.nonzero(
                    target_cases,
                    as_tuple=False,
                ).reshape(-1):
                    case_roi = batch["target_mask"][case_idx] > 0
                    if not case_roi.any():
                        continue
                    tracker.update_tbx_roi_samples(
                        target_cspca[case_idx][case_roi].detach().cpu().numpy(),
                        lesion_probs[case_idx][case_roi].detach().cpu().numpy(),
                    )
                if compute_operating_metrics:
                    sampled_true = target_cspca[sampled_tbx_roi].detach().cpu().numpy()
                    sampled_score = lesion_probs[sampled_tbx_roi].detach().cpu().numpy()
                    tracker.update_voxel_operating_samples(
                        "target_cspca",
                        sampled_true,
                        sampled_score,
                    )

        # Patient/region metrics derived from segmentation risk maps and mask GT.
        seg_evaluator.update_from_batch(lesion_probs=lesion_probs, batch=batch)
        sbx_mil_evaluator.update_from_batch(
            lesion_probs=lesion_probs,
            batch=batch,
            region_logits=outputs.get("region_logits"),
            region_valid_mask=outputs.get("region_valid_mask"),
        )
        if sample_exporter is not None:
            sample_exporter.update(batch, lesion_probs, seg_evaluator)

        if save_val_vis:
            for b in range(imgs.size(0)):
                d_type = infer_dataset_type(batch, b)
                if saved_counts.get(d_type, 0) >= max_saves_per_type:
                    continue

                empty_like = torch.zeros_like(lesion_probs[b, 0])
                # Reuse evaluator logic for visual overlays so QA images mirror
                # the reported region-level confusion counts.
                region_label_map, region_pred_map = seg_evaluator.build_case_region_maps(
                    lesion_probs=lesion_probs,
                    batch=batch,
                    b=b,
                )
                gt_dict = {
                    "type": d_type,
                    "lesion_mask": mask_for_visualisation(batch, "lesion_mask", b, empty_like),
                    "target_mask": mask_for_visualisation(batch, "target_mask", b, empty_like),
                    "zones_mask": mask_for_visualisation(batch, "zones_mask", b, empty_like),
                    "sys_labels": batch["sys_labels"][b].detach().cpu().numpy() if "sys_labels" in batch else np.asarray([]),
                    "region_label_map": region_label_map,
                    "region_pred_map": region_pred_map,
                }
                pid = batch["pid"][b] if "pid" in batch else f"case_{epoch}_{b}"
                filename = f"{d_type}_{saved_counts.get(d_type, 0) + 1:02d}_{safe_vis_filename(pid)}.png"
                try:
                    if not vis_dir_created:
                        os.makedirs(vis_dir, exist_ok=True)
                        vis_dir_created = True
                    visualize_predictions(
                        input_tensor=imgs[b],
                        risk_map=lesion_probs[b],
                        gt_dict=gt_dict,
                        save_path=os.path.join(vis_dir, filename),
                        patient_id=str(pid),
                    )
                except Exception as exc:
                    print(f"Warning: failed to save validation visualization for {pid}: {exc}")
                    continue
                saved_counts[d_type] = saved_counts.get(d_type, 0) + 1

    seg_metrics = seg_evaluator.compute_metrics()
    sbx_mil_metrics = sbx_mil_evaluator.compute_metrics()
    tracker.finalize_target_cspca_aux_dice(
        frozen_threshold=prob_threshold if using_frozen_thresholds else None
    )
    if using_frozen_thresholds:
        frozen_lesion_summary = summarise_values(tracker.lesion_dice_values)
        tracker.lesion_best_threshold = prob_threshold
        tracker.lesion_best_threshold_dice = frozen_lesion_summary["mean"]
        tracker.lesion_best_threshold_dice_std = frozen_lesion_summary["std"]
        tracker.lesion_best_threshold_dice_n = frozen_lesion_summary["n"]
    else:
        tracker.finalize_validation_dice_threshold()
        tracker.threshold_validation_epoch = int(epoch)
    if compute_froc_metrics:
        tracker.finalize_froc_metrics()
    if compute_operating_metrics:
        tracker.finalize_voxel_operating_metrics(
            "lesion",
            frozen_thresholds=(frozen_thresholds or {}).get("lesion_voxel"),
        )
        tracker.finalize_voxel_operating_metrics(
            "target_cspca",
            frozen_thresholds=(frozen_thresholds or {}).get(
                "target_cspca_voxel"
            ),
        )
    tracker.finalize_tbx_roi_metrics(
        frozen_threshold_value(
            frozen_thresholds,
            "tbx_roi",
            "decision",
            prob_threshold,
        ),
        compute_operating_metrics=compute_operating_metrics,
        compute_confidence_intervals=compute_confidence_intervals,
        frozen_thresholds=tbx_roi_thresholds,
        select_validation_threshold=not using_frozen_thresholds,
    )

    tracker.patient_sens = seg_metrics["patient_sens"]
    tracker.patient_spec = seg_metrics["patient_spec"]
    tracker.patient_bacc = seg_metrics["patient_bacc"]
    tracker.patient_auc = seg_metrics["patient_auc"]
    tracker.patient_auprc = seg_metrics["patient_auprc"]
    tracker.patient_auc_ci_low = seg_metrics["patient_auc_ci_low"]
    tracker.patient_auc_ci_high = seg_metrics["patient_auc_ci_high"]
    tracker.patient_auprc_ci_low = seg_metrics["patient_auprc_ci_low"]
    tracker.patient_auprc_ci_high = seg_metrics["patient_auprc_ci_high"]
    tracker.patient_ci_level = seg_metrics["patient_ci_level"]
    tracker.patient_ci_bootstrap_valid = seg_metrics[
        "patient_ci_bootstrap_valid"
    ]
    tracker.patient_n = seg_metrics["patient_n"]
    tracker.patient_tn = seg_metrics["patient_tn"]
    tracker.patient_fp = seg_metrics["patient_fp"]
    tracker.patient_fn = seg_metrics["patient_fn"]
    tracker.patient_tp = seg_metrics["patient_tp"]
    tracker.patient_decision_threshold = seg_metrics[
        "patient_decision_threshold"
    ]
    tracker.patient_balanced_accuracy_threshold = seg_metrics[
        "patient_balanced_accuracy_threshold"
    ]
    tracker.patient_sens_at_balanced_accuracy = seg_metrics[
        "patient_sens_at_balanced_accuracy"
    ]
    tracker.patient_spec_at_balanced_accuracy = seg_metrics[
        "patient_spec_at_balanced_accuracy"
    ]
    tracker.patient_bacc_at_balanced_accuracy = seg_metrics[
        "patient_bacc_at_balanced_accuracy"
    ]
    tracker.patient_fixed_spec_target = seg_metrics["patient_fixed_spec_target"]
    tracker.patient_sens_at_fixed_spec = seg_metrics["patient_sens_at_fixed_spec"]
    tracker.patient_actual_spec_at_fixed_spec = seg_metrics["patient_actual_spec_at_fixed_spec"]
    tracker.patient_actual_fpr_at_fixed_spec = seg_metrics["patient_actual_fpr_at_fixed_spec"]
    tracker.patient_threshold_at_fixed_spec = seg_metrics["patient_threshold_at_fixed_spec"]
    tracker.patient_fixed_sens_target = seg_metrics["patient_fixed_sens_target"]
    tracker.patient_spec_at_fixed_sens = seg_metrics["patient_spec_at_fixed_sens"]
    tracker.patient_actual_sens_at_fixed_sens = seg_metrics["patient_actual_sens_at_fixed_sens"]
    tracker.patient_threshold_at_fixed_sens = seg_metrics["patient_threshold_at_fixed_sens"]
    tracker.patient_pooling_mode = seg_metrics["patient_pooling_mode"]
    tracker.patient_pooling_lme_r = seg_metrics["patient_pooling_lme_r"]
    tracker.patient_pooling_alpha = seg_metrics["patient_pooling_alpha"]
    tracker.patient_pooling_beta = seg_metrics["patient_pooling_beta"]
    tracker.patient_pooling_intercept = seg_metrics["patient_pooling_intercept"]
    tracker.patient_pooling_regularization_c = seg_metrics[
        "patient_pooling_regularization_c"
    ]
    tracker.patient_pooling_calibration_n = seg_metrics[
        "patient_pooling_calibration_n"
    ]
    tracker.patient_pooling_calibration_positive_n = seg_metrics[
        "patient_pooling_calibration_positive_n"
    ]
    tracker.patient_pooling_calibration_fitted = seg_metrics[
        "patient_pooling_calibration_fitted"
    ]
    tracker.patient_pooling_calibration_status = seg_metrics[
        "patient_pooling_calibration_status"
    ]
    for score_family in ("patient_contrast", "patient_logit_lme"):
        for field in tracker.PATIENT_SCORE_METRIC_FIELDS:
            setattr(
                tracker,
                f"{score_family}_{field}",
                seg_metrics[f"{score_family}_{field}"],
            )

    tracker.region_sens = seg_metrics["region_sens"]
    tracker.region_spec = seg_metrics["region_spec"]
    tracker.region_bacc = seg_metrics["region_bacc"]
    tracker.region_auc = seg_metrics["region_auc"]
    tracker.region_auprc = seg_metrics["region_auprc"]
    tracker.region_auc_ci_low = seg_metrics["region_auc_ci_low"]
    tracker.region_auc_ci_high = seg_metrics["region_auc_ci_high"]
    tracker.region_auprc_ci_low = seg_metrics["region_auprc_ci_low"]
    tracker.region_auprc_ci_high = seg_metrics["region_auprc_ci_high"]
    tracker.region_ci_level = seg_metrics["region_ci_level"]
    tracker.region_ci_bootstrap_valid = seg_metrics[
        "region_ci_bootstrap_valid"
    ]
    tracker.region_n = seg_metrics["region_n"]
    tracker.region_tn = seg_metrics["region_tn"]
    tracker.region_fp = seg_metrics["region_fp"]
    tracker.region_fn = seg_metrics["region_fn"]
    tracker.region_tp = seg_metrics["region_tp"]
    tracker.region_decision_threshold = seg_metrics[
        "region_decision_threshold"
    ]
    tracker.region_balanced_accuracy_threshold = seg_metrics[
        "region_balanced_accuracy_threshold"
    ]
    tracker.region_sens_at_balanced_accuracy = seg_metrics[
        "region_sens_at_balanced_accuracy"
    ]
    tracker.region_spec_at_balanced_accuracy = seg_metrics[
        "region_spec_at_balanced_accuracy"
    ]
    tracker.region_bacc_at_balanced_accuracy = seg_metrics[
        "region_bacc_at_balanced_accuracy"
    ]
    tracker.region_fixed_spec_target = seg_metrics["region_fixed_spec_target"]
    tracker.region_sens_at_fixed_spec = seg_metrics["region_sens_at_fixed_spec"]
    tracker.region_actual_spec_at_fixed_spec = seg_metrics["region_actual_spec_at_fixed_spec"]
    tracker.region_actual_fpr_at_fixed_spec = seg_metrics["region_actual_fpr_at_fixed_spec"]
    tracker.region_threshold_at_fixed_spec = seg_metrics["region_threshold_at_fixed_spec"]
    tracker.region_fixed_sens_target = seg_metrics["region_fixed_sens_target"]
    tracker.region_spec_at_fixed_sens = seg_metrics["region_spec_at_fixed_sens"]
    tracker.region_actual_sens_at_fixed_sens = seg_metrics["region_actual_sens_at_fixed_sens"]
    tracker.region_threshold_at_fixed_sens = seg_metrics["region_threshold_at_fixed_sens"]
    tracker.sbx_mil_pooling_mode = str(_cfg("MIL_POOLING", "lme")).lower()
    tracker.sbx_mil_lme_r = float(_cfg("LME_R", 8.0))
    for field in (
        "bacc",
        "sens",
        "spec",
        "auc",
        "auprc",
        "n",
        "tn",
        "fp",
        "fn",
        "tp",
    ):
        setattr(
            tracker,
            f"sbx_mil_region_{field}",
            sbx_mil_metrics[f"region_{field}"],
        )

    if sample_exporter is not None:
        sample_exporter.finalize()

    return tracker


def infer_dataset_type(batch: Mapping, b: int) -> str:
    """Infer dataset type for logging/visualisation."""
    if "source" in batch:
        source = batch["source"][b]
        if isinstance(source, str):
            return source

    pid = str(batch.get("pid", [""])[b]) if "pid" in batch else ""
    if pid.startswith("PUB"):
        return "PUB"
    if pid.startswith("TCIA"):
        return "TCIA"
    if pid.startswith("PROMIS"):
        return "PROMIS"

    if "has_lesion" in batch and batch["has_lesion"][b].item() > 0:
        return "PUB"
    if "has_target" in batch and batch["has_target"][b].item() > 0:
        return "TCIA"
    if "has_sys" in batch and batch["has_sys"][b].item() > 0:
        return "PROMIS"
    return "OTHER"


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_loss_curves(log_path: str, save_path: str):
    """Plot lesion-related training/validation losses and EM weights."""
    try:
        df = pd.read_csv(log_path)
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        ax1 = axes[0]
        for col, label in [
            ("train_loss_total", "Train Total"),
            ("val_loss_total", "Val Total"),
            ("train_loss_lesion", "Train Lesion Total"),
            ("train_loss_lesion_dense", "Train Dense"),
            ("train_loss_lesion_sparse", "Train TBx ROI"),
            ("train_loss_lesion_sparse_bce", "Train TBx BCE"),
            ("train_loss_lesion_sparse_dice", "Train TBx Dice"),
            ("train_loss_lesion_sys", "Train Sys MIL"),
            ("val_loss_lesion", "Val Lesion Total"),
            ("val_loss_lesion_dense", "Val Dense"),
            ("val_loss_lesion_sparse", "Val TBx ROI"),
            ("val_loss_lesion_sparse_bce", "Val TBx BCE"),
            ("val_loss_lesion_sparse_dice", "Val TBx Dice"),
            ("val_loss_lesion_sys", "Val Sys MIL"),
        ]:
            if col in df.columns:
                ax1.plot(df["epoch"], df[col], label=label, linewidth=2 if "total" in col else 1.2)

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Segmentation + MIL Loss Curves")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, linestyle="--", alpha=0.4)

        ax2 = axes[1]
        for col, label in [
            ("em_w_lesion_dense", "Dense Weight"),
            ("em_w_lesion_sparse", "TBx ROI Weight"),
            ("em_w_lesion_sys", "Sys MIL Weight"),
        ]:
            if col in df.columns:
                ax2.plot(df["epoch"], df[col], label=label)

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learned multiplier exp(-log_var)")
        ax2.set_title("EM / Uncertainty Weights")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    except Exception as e:
        print(f"Plot failed: {e}")


def plot_entropy_curves(log_path: str, save_path: str):
    """Plot train/validation predictive entropy separately from task metrics."""
    try:
        df = pd.read_csv(log_path)
        panels = [
            (
                "Voxel-map entropy",
                [
                    ("lesion_all", "All voxels"),
                    ("lesion_gland", "Inside gland"),
                    ("lesion_outside_gland", "Outside gland"),
                ],
            ),
            (
                "Supervised voxel entropy",
                [
                    ("dense_positive", "Dense positive"),
                    ("dense_negative_gland", "Dense negative gland"),
                    ("tbx_positive", "TBx positive"),
                    ("tbx_negative", "TBx negative"),
                ],
            ),
            (
                "Pooled-output entropy",
                [
                    ("region_positive", "Region positive"),
                    ("region_negative", "Region negative"),
                    ("patient_positive", "Patient positive"),
                    ("patient_negative", "Patient negative"),
                ],
            ),
        ]
        available = any(
            f"{split}_entropy_{key}_bits" in df.columns
            for _, series in panels
            for key, _ in series
            for split in ("train", "val")
        )
        if not available:
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
        for ax, (title, series) in zip(axes, panels):
            for key, label in series:
                for split, style in (("train", "-"), ("val", "--")):
                    col = f"{split}_entropy_{key}_bits"
                    count_col = f"{split}_entropy_{key}_n"
                    if col not in df.columns:
                        continue
                    values = df[col].copy()
                    if count_col in df.columns:
                        values = values.where(df[count_col] > 0)
                    ax.plot(
                        df["epoch"],
                        values,
                        linestyle=style,
                        label=f"{split.title()} {label}",
                    )
            ax.set_ylim(-0.02, 1.02)
            ax.set_ylabel("Entropy (bits)")
            ax.set_title(title)
            ax.grid(True, linestyle="--", alpha=0.4)
            if ax.lines:
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

        axes[-1].set_xlabel("Epoch")
        fig.suptitle("Predictive Entropy During Training (0=certain, 1=max uncertain)")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    except Exception as e:
        print(f"Entropy plot failed: {e}")


def visualize_predictions(input_tensor, risk_map, gt_dict, save_path: str, patient_id: str):
    """Visualise lesion risk map and available ground truth.

    Removed the old grade-map row. The figure now shows:
      row 1: T2 + predicted lesion risk
      row 2: available ground truth / biopsy supervision
      row 3: systematic zones if available
    """
    t2 = input_tensor[0].detach().cpu().numpy()
    risk = risk_map[0].detach().cpu().numpy()

    mid = t2.shape[0] // 2
    slices = [max(0, mid - 5), mid, min(t2.shape[0] - 1, mid + 5)]

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.suptitle(f"Patient: {patient_id} | Dataset Type: {gt_dict['type']}", fontsize=16, y=0.98)

    for i, s_idx in enumerate(slices):
        # Row 1: predicted lesion risk map.
        axes[0, i].imshow(t2[s_idx], cmap="gray")
        risk_overlay = np.ma.masked_where(risk[s_idx] < 0.2, risk[s_idx])
        im1 = axes[0, i].imshow(risk_overlay, cmap="hot", alpha=0.5, vmin=0, vmax=1)
        axes[0, i].set_title(f"Prediction: Lesion Risk (Slice {s_idx})")
        axes[0, i].axis("off")
        if i == 2:
            fig.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)

        # Row 2: available supervision.
        axes[1, i].imshow(t2[s_idx], cmap="gray")
        gt_slice, title, cmap, vmin, vmax = _build_gt_slice(gt_dict, s_idx)
        if gt_slice is not None:
            gt_overlay = np.ma.masked_where(gt_slice == 0, gt_slice)
            im2 = axes[1, i].imshow(gt_overlay, cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax)
            if i == 2:
                fig.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
        axes[1, i].set_title(title)
        axes[1, i].axis("off")

        # Row 3: region-level label/prediction, useful for localisation QA.
        axes[2, i].imshow(t2[s_idx], cmap="gray")
        z_slice = gt_dict.get("zones_mask", None)
        if z_slice is not None and np.max(z_slice) > 0:
            zone_overlay = np.ma.masked_where(z_slice[s_idx] == 0, z_slice[s_idx])
            axes[2, i].imshow(zone_overlay, cmap="tab20", alpha=0.18)
        region_label = gt_dict.get("region_label_map", None)
        region_pred = gt_dict.get("region_pred_map", None)
        has_region_overlay = False
        if region_label is not None and np.max(region_label) > 0:
            label_overlay = np.ma.masked_where(region_label[s_idx] == 0, region_label[s_idx])
            axes[2, i].imshow(label_overlay, cmap="Greens", alpha=0.42, vmin=0, vmax=1)
            has_region_overlay = True
        if region_pred is not None and np.max(region_pred) > 0:
            pred_overlay = np.ma.masked_where(region_pred[s_idx] == 0, region_pred[s_idx])
            axes[2, i].imshow(pred_overlay, cmap="Reds", alpha=0.42, vmin=0, vmax=1)
            has_region_overlay = True
        if has_region_overlay:
            axes[2, i].set_title(f"Regions: label green / pred red (Slice {s_idx})")
        elif z_slice is not None and np.max(z_slice) > 0:
            axes[2, i].set_title(f"Systematic Zones (Slice {s_idx})")
        else:
            axes[2, i].set_title(f"No systematic zones (Slice {s_idx})")
        axes[2, i].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def _build_gt_slice(gt_dict: Mapping, s_idx: int):
    d_type = gt_dict.get("type", "OTHER")
    invalid = int(_cfg("INVALID_SYS_LABEL", -1))
    positive_threshold = int(_cfg("LESION_POSITIVE_THRESHOLD", _cfg("CSPC_THRESHOLD", 1)))

    if d_type == "PUB":
        lesion = gt_dict.get("lesion_mask")
        if lesion is not None:
            return lesion[s_idx], f"GT: Radiologist Lesion Mask (Slice {s_idx})", "autumn", 0, 1

    if d_type == "TCIA":
        target = gt_dict.get("target_mask")
        if target is not None and np.max(target) > 0:
            binary_target = (target[s_idx] >= positive_threshold).astype(np.float32)
            if binary_target.max() == 0:
                # Show TBx-confirmed target ROI even when benign-labelled.
                binary_target = (target[s_idx] > 0).astype(np.float32)
            return binary_target, f"GT: TBx-confirmed Target ROI (Slice {s_idx})", "autumn", 0, 1

    if d_type in {"TCIA", "PROMIS"}:
        zones = gt_dict.get("zones_mask")
        sys_labels = gt_dict.get("sys_labels", np.asarray([]))
        if zones is not None and len(sys_labels) > 0:
            z_slice = zones[s_idx]
            gt_slice = np.zeros_like(z_slice, dtype=np.float32)
            for z_idx in range(1, min(len(sys_labels), 20) + 1):
                label = sys_labels[z_idx - 1]
                if label != invalid:
                    gt_slice[z_slice == z_idx] = float(label >= positive_threshold)
            return gt_slice, f"GT: SBx Positive Regions (Slice {s_idx})", "autumn", 0, 1

    return None, f"GT: No dense supervision (Slice {s_idx})", "autumn", 0, 1
