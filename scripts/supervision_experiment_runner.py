#!/usr/bin/env python3
"""Shared runner utilities for the redesigned B/N supervision experiments.

The B/N scripts deliberately separate optimisation hyperparameters from
supervision membership:

  D = dense radiologist annotation (RA)
  T = targeted-biopsy ROI supervision (TBx)
  S = systematic-biopsy region supervision (SBx)
  P = biopsy-confirmed patient-level supervision

Outside-gland suppression is disabled in the main B/N matrix. It is an anatomy
ablation, not one of the four cancer-supervision levels above.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional, Tuple


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


EXPERIMENT_USE_EARLY_STOPPING = False
EXPERIMENT_EARLY_STOP_PATIENCE = 30
EXPERIMENT_TOP_K_CHECKPOINTS = 5
# Compact protocol marker used in every redesigned run directory.  It prevents
# the new joint-TBx/SBx cohort and plain-logit-LME endpoint from being mistaken
# for an older run that used the same split filename or contrast scoring.
EXPERIMENT_PROTOCOL_TAG = "JTS-PLME"
B_NATIVE_BEST_METRICS = {
    "b0": "patient_auprc",
    "b1": "tbx_native",
    "b2": "region_auprc",
    "b3": "tbx_sbx_native",
    "b4": "tbx_sbx_patient_native",
}


@dataclass(frozen=True)
class ExperimentSpec:
    """One fully specified B/N experiment and its comparison protocol."""

    key: str
    mode: str
    description: str
    parameter_source: str
    train_csv: str
    family: str = ""
    train_dataset_task: str = "mixed"
    use_dense: bool = False
    use_tbx: bool = False
    use_sbx: bool = False
    use_patient: bool = False
    dense_weight: float = 0.0
    tbx_weight: float = 0.0
    sbx_weight: float = 0.0
    patient_weight: float = 0.0
    use_curriculum: bool = False
    dense_start: int = 1
    tbx_start: int = 1
    sbx_start: int = 1
    patient_start: int = 1
    lr: float = 1e-4
    pos_weight: float = 2.0
    sys_pos_weight: float = 1.0
    tbx_dice_weight: float = 0.0
    native_best_metric: str = "patient_auprc"
    common_best_metric: str = "patient_auprc"
    patient_pooling: str = "logit_lme"
    compare_patient_pooling: bool = False
    mil_pooling: str = "lme"
    region_pooling: str = "top_percent"
    lme_r: float = 8.0
    design_tag: str = "BestPrior"
    tag_tokens: Tuple[str, ...] = ()

    @property
    def best_metric(self) -> str:
        """Backward-compatible alias for the task-native selector."""
        return self.native_best_metric

    @property
    def experiment_family(self) -> str:
        family = str(self.family).strip().lower()
        if family:
            return family
        key = str(self.key).strip().lower()
        return key[:1]

    @property
    def supervision_code(self) -> str:
        active = []
        if self.use_dense:
            active.append("D")
        if self.use_tbx:
            active.append("T")
        if self.use_sbx:
            active.append("S")
        if self.use_patient:
            active.append("P")
        return "".join(active) or "None"


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add safe runtime overrides shared by both family runners."""

    parser.add_argument("--base-dir", default=os.environ.get("RP_BASE_DIR"))
    parser.add_argument("--dataset-root", default=os.environ.get("RP_DATASET_ROOT"))
    parser.add_argument("--exp-dir", default=os.environ.get("RP_EXP_DIR"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--sys-pos-weight", type=float, default=None)
    parser.add_argument(
        "--tbx-dice-weight",
        type=float,
        default=None,
        help="Override the masked TBx Dice weight while keeping TBx BCE enabled.",
    )
    parser.add_argument("--dropout-rate", type=float, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configuration without starting training.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if an identical tag already completed internal/external tests.",
    )


def _number(value: float) -> str:
    return f"{float(value):g}"


def _resolved_value(cli_value: Optional[float], default: float) -> float:
    return float(default if cli_value is None else cli_value)


def _resolved_exp_dir(args: argparse.Namespace) -> str:
    if args.exp_dir:
        return os.path.abspath(args.exp_dir)
    if args.base_dir:
        return os.path.join(os.path.abspath(args.base_dir), "Experiments")
    return str(PROJECT_DIR / "Experiments")


def _dry_run_config(args: argparse.Namespace) -> SimpleNamespace:
    """Build the small Config surface needed for dependency-free dry runs."""

    base_dir = (
        os.path.abspath(args.base_dir) if args.base_dir else str(PROJECT_DIR)
    )
    dataset_root = (
        os.path.abspath(args.dataset_root)
        if args.dataset_root
        else os.path.join(base_dir, "data")
    )
    unified = os.path.join(dataset_root, "Unified_Dataset")
    split_dir = os.path.join(unified, "splits")
    return SimpleNamespace(
        BASE_DIR=base_dir,
        DATASET_ROOT=dataset_root,
        UNIFIED_DATA_DIR=unified,
        SPLIT_DIR=split_dir,
        COMMON_INTERNAL_VAL_CSV=os.path.join(split_dir, "common_internal_evaluation.csv"),
        COMMON_INTERNAL_TEST_CSV=os.path.join(split_dir, "common_internal_test.csv"),
        COMMON_EXTERNAL_TEST_CSV=os.path.join(split_dir, "N4_mixed_PROMIS_external_val.csv"),
        EXP_DIR=(os.path.abspath(args.exp_dir) if args.exp_dir else os.path.join(base_dir, "Experiments")),
        NUM_EPOCHS=150,
        SEED=42,
        DROPOUT_RATE=0.2,
        USE_EARLY_STOPPING=EXPERIMENT_USE_EARLY_STOPPING,
        EARLY_STOP_PATIENCE=EXPERIMENT_EARLY_STOP_PATIENCE,
        TOP_K_CHECKPOINTS=EXPERIMENT_TOP_K_CHECKPOINTS,
        FINAL_TEST_INCLUDE_TOP_K=True,
    )


def _completed_run(exp_dir: str, experiment_tag: str) -> Optional[str]:
    root = Path(exp_dir)
    if not root.is_dir():
        return None

    for run_dir in sorted(root.glob(f"*{experiment_tag}*"), reverse=True):
        test_log = run_dir / "test_log.csv"
        last_checkpoint = run_dir / "last_checkpoint.pth"
        if not test_log.is_file() or not last_checkpoint.is_file():
            continue
        try:
            with test_log.open(newline="") as handle:
                labels = {
                    str(row.get("test_dataset_label", "")).strip().lower()
                    for row in csv.DictReader(handle)
                }
        except (OSError, csv.Error):
            continue
        if {"internal", "external"}.issubset(labels):
            return str(run_dir)
    return None


def _csv_flag(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _validate_protocol_csv(
    path: str,
    *,
    label: str,
    require_joint_tcia_marker: bool,
) -> None:
    """Fail fast on stale pre-joint splits or invalid gland support."""

    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {label} split CSV: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        required = {"patient_id", "source", "has_gland"}
        if require_joint_tcia_marker:
            required.add("eligible_tcia_tbx_sbx")
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(
                f"{label} split is incompatible with the JTS-PLME protocol; "
                f"missing columns {missing}: {csv_path}. Copy/regenerate the "
                "current local split files before training."
            )

        seen: set[tuple[str, str]] = set()
        row_count = 0
        for row_count, row in enumerate(reader, start=1):
            patient_id = str(row.get("patient_id", "")).strip()
            source = str(row.get("source", "")).strip().upper()
            identity = (source, patient_id)
            if not patient_id or not source:
                raise ValueError(f"{label} split contains an empty case identity: {csv_path}")
            if identity in seen:
                raise ValueError(
                    f"{label} split contains duplicate case {identity}: {csv_path}"
                )
            seen.add(identity)

            if not _csv_flag(row.get("has_gland")):
                raise ValueError(
                    f"{label} split contains a case without a valid gland mask: "
                    f"{identity}. Gland-restricted patient scoring would silently "
                    "change the effective cohort."
                )
            if (
                require_joint_tcia_marker
                and source == "TCIA"
                and not _csv_flag(row.get("eligible_tcia_tbx_sbx"))
            ):
                raise ValueError(
                    f"{label} split contains non-joint TCIA case {identity}: {csv_path}. "
                    "All TCIA cases must have both TBx and SBx data."
                )

        if row_count == 0:
            raise ValueError(f"{label} split is empty: {csv_path}")


def validate_split_protocol(Config: Any) -> None:
    """Validate the exact cohort contract used by formal B/N experiments."""

    _validate_protocol_csv(
        Config.TRAIN_CSV,
        label="training",
        require_joint_tcia_marker=True,
    )
    _validate_protocol_csv(
        Config.VAL_CSV,
        label="validation",
        require_joint_tcia_marker=True,
    )
    _validate_protocol_csv(
        Config.INTERNAL_TEST_CSV,
        label="internal test",
        require_joint_tcia_marker=True,
    )
    _validate_protocol_csv(
        Config.TEST_CSV,
        label="external test",
        require_joint_tcia_marker=False,
    )


def _refresh_paths(
    Config: Any,
    spec: ExperimentSpec,
    *,
    base_dir: Optional[str],
    dataset_root: Optional[str],
    exp_dir: Optional[str],
) -> None:
    if base_dir:
        Config.BASE_DIR = os.path.abspath(base_dir)
    if dataset_root:
        Config.DATASET_ROOT = os.path.abspath(dataset_root)
    elif base_dir:
        Config.DATASET_ROOT = os.path.join(Config.BASE_DIR, "data")

    if base_dir or dataset_root:
        Config.UNIFIED_DATA_DIR = os.path.join(Config.DATASET_ROOT, "Unified_Dataset")
        Config.SPLIT_DIR = os.path.join(Config.UNIFIED_DATA_DIR, "splits")
        Config.COMMON_INTERNAL_VAL_CSV = os.path.join(
            Config.SPLIT_DIR, "common_internal_evaluation.csv"
        )
        Config.COMMON_INTERNAL_TEST_CSV = os.path.join(
            Config.SPLIT_DIR, "common_internal_test.csv"
        )
        Config.COMMON_EXTERNAL_TEST_CSV = os.path.join(
            Config.SPLIT_DIR, "N4_mixed_PROMIS_external_val.csv"
        )

    Config.TRAIN_CSV = os.path.join(Config.SPLIT_DIR, spec.train_csv)
    Config.VAL_CSV = Config.COMMON_INTERNAL_VAL_CSV
    Config.INTERNAL_TEST_CSV = Config.COMMON_INTERNAL_TEST_CSV
    Config.TEST_CSV = Config.COMMON_EXTERNAL_TEST_CSV
    Config.COMMON_FINAL_TEST_DATASETS = (
        ("internal", Config.INTERNAL_TEST_CSV),
        ("external", Config.TEST_CSV),
    )
    Config.FINAL_TEST_DATASETS = Config.COMMON_FINAL_TEST_DATASETS

    if exp_dir:
        Config.EXP_DIR = os.path.abspath(exp_dir)
    elif base_dir:
        Config.EXP_DIR = os.path.join(Config.BASE_DIR, "Experiments")


def _experiment_tag(
    spec: ExperimentSpec,
    *,
    epochs: int,
    seed: int,
    lr: float,
    pos_weight: float,
    sys_pos_weight: float,
    tbx_dice_weight: float,
    use_early_stopping: bool,
    early_stop_patience: int,
) -> str:
    curriculum = "Curr" if spec.use_curriculum else "NoCurr"
    protocol_detail_tokens = []
    if str(spec.design_tag) == "StrictAbl":
        protocol_detail_tokens.append(
            "Starts"
            f"D{int(spec.dense_start)}"
            f"T{int(spec.tbx_start)}"
            f"S{int(spec.sbx_start)}"
            f"P{int(spec.patient_start)}"
        )
    if spec.experiment_family == "b":
        protocol_detail_tokens.append(f"Native{spec.native_best_metric}")
    return "_".join(
        [
            spec.key.upper(),
            f"Sup{spec.supervision_code}",
            str(spec.design_tag),
            EXPERIMENT_PROTOCOL_TAG,
            *protocol_detail_tokens,
            *spec.tag_tokens,
            f"D{_number(spec.dense_weight)}",
            f"T{_number(spec.tbx_weight)}",
            f"TDice{_number(tbx_dice_weight)}",
            f"S{_number(spec.sbx_weight)}",
            f"P{_number(spec.patient_weight)}",
            "A0",
            curriculum,
            f"LR{_number(lr)}",
            f"PosW{_number(pos_weight)}",
            f"SysPosW{_number(sys_pos_weight)}",
            (
                f"ES{int(early_stop_patience)}"
                if use_early_stopping
                else "ESOff"
            ),
            f"TopK{EXPERIMENT_TOP_K_CHECKPOINTS}",
            f"Seed{seed}",
            f"E{epochs}",
        ]
    )


def _refresh_fixed_loss_weights(Config: Any) -> None:
    Config.FIXED_LOSS_WEIGHTS = {
        "grade_tbx": 0.0,
        "grade_sbx": 0.0,
        "lesion_dense": float(Config.LESION_DENSE_LOSS_WEIGHT),
        "lesion_sparse": float(Config.LESION_SPARSE_LOSS_WEIGHT),
        "lesion_sys": float(Config.LESION_SYS_LOSS_WEIGHT),
        "lesion_outside_gland": float(Config.OUTSIDE_GLAND_LOSS_WEIGHT),
        "lesion_patient": float(Config.PATIENT_RISK_LOSS_WEIGHT),
        "gland": 0.0,
    }


def apply_experiment(Config: Any, spec: ExperimentSpec, args: argparse.Namespace) -> None:
    """Resolve one spec into Config without inheriting mode-dependent auxiliaries."""

    epochs = int(Config.NUM_EPOCHS if args.epochs is None else args.epochs)
    seed = int(Config.SEED if args.seed is None else args.seed)
    lr = _resolved_value(args.lr, spec.lr)
    pos_weight = _resolved_value(args.pos_weight, spec.pos_weight)
    sys_pos_weight = _resolved_value(args.sys_pos_weight, spec.sys_pos_weight)
    tbx_dice_weight = _resolved_value(
        getattr(args, "tbx_dice_weight", None), spec.tbx_dice_weight
    )
    early_stop_patience = EXPERIMENT_EARLY_STOP_PATIENCE

    Config.EXPERIMENT_MODE = spec.mode
    Config.EXPERIMENT_TAG = _experiment_tag(
        spec,
        epochs=epochs,
        seed=seed,
        lr=lr,
        pos_weight=pos_weight,
        sys_pos_weight=sys_pos_weight,
        tbx_dice_weight=tbx_dice_weight,
        use_early_stopping=EXPERIMENT_USE_EARLY_STOPPING,
        early_stop_patience=early_stop_patience,
    )

    Config.TASK = "mixed"
    Config.DATASET_TASK = "mixed"
    Config.TRAIN_DATASET_TASK = spec.train_dataset_task
    Config.VAL_DATASET_TASK = "mixed"
    Config.TEST_DATASET_TASK = "mixed"

    Config.USE_GRADE_TBX_TASK = False
    Config.USE_GRADE_SBX_TASK = False
    Config.USE_GLAND_TASK = False
    Config.USE_LESION_DENSE_TASK = bool(spec.use_dense)
    Config.USE_LESION_SPARSE_TASK = bool(spec.use_tbx)
    Config.USE_LESION_SYS_TASK = bool(spec.use_sbx)

    Config.LESION_DENSE_LOSS_WEIGHT = float(spec.dense_weight)
    Config.LESION_SPARSE_LOSS_WEIGHT = float(spec.tbx_weight)
    Config.LESION_SYS_LOSS_WEIGHT = float(spec.sbx_weight)
    Config.USE_EM_WEIGHTING = False
    Config.USE_CURRICULUM = bool(spec.use_curriculum)
    Config.LESION_DENSE_START_EPOCH = int(spec.dense_start)
    Config.LESION_SPARSE_START_EPOCH = int(spec.tbx_start)
    Config.LESION_SYS_START_EPOCH = int(spec.sbx_start)

    # A is excluded from the main B/N matrix. P is activated only by the spec.
    Config.USE_OUTSIDE_GLAND_PENALTY = False
    Config.OUTSIDE_GLAND_LOSS_WEIGHT = 0.0
    Config.OUTSIDE_GLAND_START_EPOCH = 1
    Config.USE_PATIENT_RISK_LOSS = bool(spec.use_patient)
    Config.PATIENT_RISK_LOSS_WEIGHT = float(spec.patient_weight)
    Config.PATIENT_RISK_START_EPOCH = int(spec.patient_start)

    # Training and evaluation use the same strict prostate-mask support. The
    # experiment spec chooses the pre-specified patient endpoint without
    # changing model training.
    Config.PATIENT_RISK_POOLING = "lme"
    Config.PATIENT_RISK_LME_R = float(spec.lme_r)
    Config.PATIENT_RISK_USE_GLAND_MASK = True
    Config.SEG_PATIENT_POOLING = str(spec.patient_pooling)
    Config.SEG_RISK_LME_R = float(spec.lme_r)
    Config.SEG_PATIENT_CALIBRATION_C = 1.0
    Config.SEG_PATIENT_CALIBRATION_MAX_ITER = 1000
    Config.SEG_PATIENT_CALIBRATION_SEED = int(seed)
    Config.SEG_EVAL_USE_GLAND_MASK = True
    Config.SEG_EVAL_COMPARE_PATIENT_POOLING = bool(
        spec.compare_patient_pooling
    )
    Config.MIL_POOLING = str(spec.mil_pooling).lower()
    Config.SEG_REGION_POOLING = str(spec.region_pooling).lower()
    Config.LME_R = float(spec.lme_r)
    Config.PATIENT_DECISION_THRESHOLD_RULE = "fixed_sensitivity"
    Config.REGION_DECISION_THRESHOLD_RULE = "fixed_specificity"
    Config.TBX_ROI_DECISION_THRESHOLD_RULE = "max_balanced_accuracy"
    Config.USE_RA_LESION_PRESENCE_AS_PATIENT_LABEL = False

    Config.LR = lr
    Config.POS_WEIGHT_VAL = pos_weight
    Config.SYS_POS_WEIGHT_VAL = sys_pos_weight
    Config.NUM_EPOCHS = epochs
    Config.SEED = seed
    Config.USE_EARLY_STOPPING = EXPERIMENT_USE_EARLY_STOPPING
    Config.EARLY_STOP_PATIENCE = early_stop_patience
    Config.NATIVE_BEST_MODEL_METRIC = str(spec.native_best_metric)
    Config.COMMON_BEST_MODEL_METRIC = str(spec.common_best_metric)
    Config.BEST_MODEL_METRIC = Config.NATIVE_BEST_MODEL_METRIC
    Config.SAVE_DUAL_BEST_CHECKPOINTS = True
    Config.FINAL_TEST_INCLUDE_NATIVE_BEST = True
    Config.FINAL_TEST_INCLUDE_COMMON_BEST = True
    Config.TOP_K_CHECKPOINTS = EXPERIMENT_TOP_K_CHECKPOINTS
    Config.FINAL_TEST_INCLUDE_TOP_K = True
    Config.WEIGHT_DECAY = 1e-4
    Config.BATCH_SIZE = 4
    Config.DROPOUT_RATE = (
        float(Config.DROPOUT_RATE)
        if args.dropout_rate is None
        else float(args.dropout_rate)
    )
    Config.USE_AUGMENTATION = True
    Config.USE_TBX_POSITIVE_ONLY_LOSS = False
    Config.TBX_DICE_LOSS_WEIGHT = tbx_dice_weight
    Config.TBX_DICE_SMOOTH = float(getattr(Config, "TBX_DICE_SMOOTH", 1e-5))
    Config.USE_SYS_CLASS_BALANCED_BCE = True
    Config.SYS_FOCAL_ALPHA = 0.75
    Config.SYS_FOCAL_GAMMA = 2.0
    Config.MASK_TARGET_IN_SYS = False

    _refresh_fixed_loss_weights(Config)
    validate_resolved_config(Config, spec)


def validate_resolved_config(Config: Any, spec: ExperimentSpec) -> None:
    active = (
        bool(Config.USE_LESION_DENSE_TASK),
        bool(Config.USE_LESION_SPARSE_TASK),
        bool(Config.USE_LESION_SYS_TASK),
        bool(Config.USE_PATIENT_RISK_LOSS),
    )
    expected = (spec.use_dense, spec.use_tbx, spec.use_sbx, spec.use_patient)
    if active != expected:
        raise ValueError(f"Resolved supervision {active} does not match {spec.key} {expected}.")
    if not any(active):
        raise ValueError(f"{spec.key} has no active supervision.")
    if bool(Config.USE_OUTSIDE_GLAND_PENALTY) or float(Config.OUTSIDE_GLAND_LOSS_WEIGHT) != 0.0:
        raise ValueError("Outside-gland supervision must be disabled in the main B/N matrix.")
    if spec.use_patient and float(Config.PATIENT_RISK_LOSS_WEIGHT) <= 0.0:
        raise ValueError(f"{spec.key} enables P but has a non-positive patient weight.")
    if not spec.use_patient and float(Config.PATIENT_RISK_LOSS_WEIGHT) != 0.0:
        raise ValueError(f"{spec.key} disables P but retained a patient weight.")
    if float(Config.TBX_DICE_LOSS_WEIGHT) < 0.0:
        raise ValueError("TBX_DICE_LOSS_WEIGHT must be non-negative.")
    if not spec.use_tbx and float(Config.TBX_DICE_LOSS_WEIGHT) != 0.0:
        raise ValueError(f"{spec.key} disables TBx but retained a TBx Dice weight.")
    family = spec.experiment_family
    if str(spec.design_tag) == "StrictAbl":
        shared_values = (
            float(Config.LR),
            float(Config.POS_WEIGHT_VAL),
            float(Config.SYS_POS_WEIGHT_VAL),
            float(Config.TBX_DICE_LOSS_WEIGHT),
        )
        if shared_values != (1e-4, 2.0, 1.0, 0.0):
            raise ValueError(
                "StrictAbl requires LR=1e-4, PosW=2, SysPosW=1, and TBx Dice=0."
            )
        if family == "b":
            if bool(Config.USE_CURRICULUM):
                raise ValueError("Strict B experiments must start all active losses at epoch 1.")
            if spec.use_tbx and (
                float(Config.LESION_SPARSE_LOSS_WEIGHT),
                int(Config.LESION_SPARSE_START_EPOCH),
            ) != (1.0, 1):
                raise ValueError("Strict B TBx supervision requires weight 1 and start epoch 1.")
            if spec.use_sbx and (
                float(Config.LESION_SYS_LOSS_WEIGHT),
                int(Config.LESION_SYS_START_EPOCH),
            ) != (1.0, 1):
                raise ValueError("Strict B SBx supervision requires weight 1 and start epoch 1.")
        elif family == "n":
            if (
                float(Config.LESION_DENSE_LOSS_WEIGHT),
                int(Config.LESION_DENSE_START_EPOCH),
            ) != (1.0, 1):
                raise ValueError("Strict N experiments require dense weight 1 from epoch 1.")
            if spec.use_tbx and (
                float(Config.LESION_SPARSE_LOSS_WEIGHT),
                int(Config.LESION_SPARSE_START_EPOCH),
            ) != (0.05, 15):
                raise ValueError("Strict N TBx supervision requires weight 0.05 and start epoch 15.")
            if spec.use_sbx and (
                float(Config.LESION_SYS_LOSS_WEIGHT),
                int(Config.LESION_SYS_START_EPOCH),
            ) != (0.25, 15):
                raise ValueError("Strict N SBx supervision requires weight 0.25 and start epoch 15.")
            if bool(Config.USE_CURRICULUM) != bool(spec.use_tbx or spec.use_sbx):
                raise ValueError("Strict N curriculum must be enabled exactly when a weak branch is active.")
        if spec.use_patient and (
            float(Config.PATIENT_RISK_LOSS_WEIGHT),
            int(Config.PATIENT_RISK_START_EPOCH),
        ) != (0.05, 1):
            raise ValueError("Strict patient supervision requires weight 0.05 and start epoch 1.")
    if bool(Config.MASK_TARGET_IN_SYS):
        raise ValueError("MASK_TARGET_IN_SYS must remain disabled.")
    if not bool(Config.PATIENT_RISK_USE_GLAND_MASK):
        raise ValueError("Patient-risk loss must pool inside gland_mask.")
    if not bool(Config.SEG_EVAL_USE_GLAND_MASK):
        raise ValueError("Patient-risk evaluation must pool inside gland_mask.")
    if str(Config.SEG_PATIENT_POOLING).lower() != str(spec.patient_pooling).lower():
        raise ValueError("Resolved patient pooling does not match the experiment spec.")
    if bool(Config.SEG_EVAL_COMPARE_PATIENT_POOLING) != bool(
        spec.compare_patient_pooling
    ):
        raise ValueError(
            "Resolved patient-pooling comparison flag does not match the spec."
        )
    if str(Config.PATIENT_DECISION_THRESHOLD_RULE) != "fixed_sensitivity":
        raise ValueError("Patient decision threshold must target fixed sensitivity.")
    if str(Config.REGION_DECISION_THRESHOLD_RULE) != "fixed_specificity":
        raise ValueError("Region decision threshold must target fixed specificity.")
    if str(Config.TBX_ROI_DECISION_THRESHOLD_RULE) != "max_balanced_accuracy":
        raise ValueError("TBx ROI decision threshold must retain max balanced accuracy.")
    if str(Config.MIL_POOLING).lower() not in {"mean", "max", "lme"}:
        raise ValueError("MIL_POOLING must be one of: mean, max, lme.")
    if str(Config.MIL_POOLING).lower() != str(spec.mil_pooling).lower():
        raise ValueError("Resolved SBx MIL pooling does not match the experiment spec.")
    if str(Config.SEG_REGION_POOLING).lower() != str(spec.region_pooling).lower():
        raise ValueError("Resolved canonical region pooling does not match the spec.")
    if float(Config.LME_R) <= 0.0 or float(Config.LME_R) != float(spec.lme_r):
        raise ValueError("Resolved LME_R must match the positive experiment value.")
    if family == "b":
        expected_b_native = B_NATIVE_BEST_METRICS.get(str(spec.key).lower())
        if expected_b_native is None:
            raise ValueError(f"No task-native checkpoint metric is registered for {spec.key}.")
        if str(Config.NATIVE_BEST_MODEL_METRIC) != expected_b_native:
            raise ValueError(
                f"{spec.key.upper()} native/Top-5 checkpoint selection must use "
                f"{expected_b_native}."
            )
        if str(Config.COMMON_BEST_MODEL_METRIC) != "patient_auprc":
            raise ValueError("B common checkpoint selection must use patient AUPRC.")
        if str(Config.SEG_PATIENT_POOLING).lower() != "logit_lme":
            raise ValueError("B-family patient AUPRC must use original logit-LME.")
        if bool(Config.SEG_EVAL_COMPARE_PATIENT_POOLING):
            raise ValueError("B-family experiments must not use contrast pooling.")
    if family == "n":
        if str(Config.NATIVE_BEST_MODEL_METRIC) != "lesion_dice":
            raise ValueError(
                "N native/best/Top-5 checkpoint selection must use lesion Dice."
            )
        if str(Config.COMMON_BEST_MODEL_METRIC) != "patient_auprc":
            raise ValueError("N common checkpoint selection must use patient AUPRC.")
        if not bool(Config.USE_LESION_DENSE_TASK):
            raise ValueError("N lesion-Dice selection requires dense RA supervision.")
        if str(Config.SEG_PATIENT_POOLING).lower() != "logit_lme":
            raise ValueError("N-family patient AUPRC must use original logit-LME.")
        if bool(Config.SEG_EVAL_COMPARE_PATIENT_POOLING):
            raise ValueError("N-family experiments must not use contrast pooling.")
    if str(Config.NATIVE_BEST_MODEL_METRIC) != str(spec.native_best_metric):
        raise ValueError("Resolved native checkpoint metric does not match the spec.")
    if str(Config.COMMON_BEST_MODEL_METRIC) != str(spec.common_best_metric):
        raise ValueError("Resolved common checkpoint metric does not match the spec.")
    if bool(Config.USE_EARLY_STOPPING):
        raise ValueError("Early stopping must remain disabled in redesigned experiments.")
    if int(Config.TOP_K_CHECKPOINTS) != EXPERIMENT_TOP_K_CHECKPOINTS:
        raise ValueError("All redesigned experiments must retain native Top-5 checkpoints.")
    if not bool(Config.FINAL_TEST_INCLUDE_TOP_K):
        raise ValueError("Native Top-5 checkpoints must be included in final test.")


def print_resolved_config(Config: Any, spec: ExperimentSpec) -> None:
    keys: Iterable[str] = [
        "EXPERIMENT_MODE",
        "EXPERIMENT_TAG",
        "TRAIN_CSV",
        "VAL_CSV",
        "INTERNAL_TEST_CSV",
        "TEST_CSV",
        "FINAL_TEST_DATASETS",
        "EXP_DIR",
        "NUM_EPOCHS",
        "USE_EARLY_STOPPING",
        "EARLY_STOP_PATIENCE",
        "SEED",
        "LR",
        "WEIGHT_DECAY",
        "BATCH_SIZE",
        "DROPOUT_RATE",
        "MIL_POOLING",
        "LME_R",
        "SEG_REGION_POOLING",
        "POS_WEIGHT_VAL",
        "SYS_POS_WEIGHT_VAL",
        "BEST_MODEL_METRIC",
        "NATIVE_BEST_MODEL_METRIC",
        "COMMON_BEST_MODEL_METRIC",
        "SAVE_DUAL_BEST_CHECKPOINTS",
        "TOP_K_CHECKPOINTS",
        "FINAL_TEST_INCLUDE_TOP_K",
        "USE_LESION_DENSE_TASK",
        "USE_LESION_SPARSE_TASK",
        "USE_LESION_SYS_TASK",
        "USE_PATIENT_RISK_LOSS",
        "USE_OUTSIDE_GLAND_PENALTY",
        "LESION_DENSE_LOSS_WEIGHT",
        "LESION_SPARSE_LOSS_WEIGHT",
        "LESION_SYS_LOSS_WEIGHT",
        "TBX_DICE_LOSS_WEIGHT",
        "PATIENT_RISK_LOSS_WEIGHT",
        "USE_CURRICULUM",
        "LESION_DENSE_START_EPOCH",
        "LESION_SPARSE_START_EPOCH",
        "LESION_SYS_START_EPOCH",
        "PATIENT_RISK_START_EPOCH",
        "PATIENT_RISK_POOLING",
        "PATIENT_RISK_USE_GLAND_MASK",
        "SEG_PATIENT_POOLING",
        "SEG_PATIENT_CALIBRATION_C",
        "SEG_PATIENT_CALIBRATION_MAX_ITER",
        "SEG_PATIENT_CALIBRATION_SEED",
        "SEG_EVAL_USE_GLAND_MASK",
        "SEG_EVAL_COMPARE_PATIENT_POOLING",
        "PATIENT_DECISION_THRESHOLD_RULE",
        "REGION_DECISION_THRESHOLD_RULE",
        "TBX_ROI_DECISION_THRESHOLD_RULE",
        "PICAI_FROC_MIN_OVERLAP",
        "PICAI_CANDIDATE_THRESHOLD",
        "PICAI_CANDIDATE_MIN_VOXELS",
        "PICAI_CANDIDATE_MAX_LESIONS",
        "PICAI_CANDIDATE_DYNAMIC_THRESHOLD_FACTOR",
        "PICAI_CANDIDATE_REMOVE_ADJACENT",
        "MASK_TARGET_IN_SYS",
        "FIXED_LOSS_WEIGHTS",
    ]
    print(f"Selected {spec.key.upper()}: {spec.description}")
    print(f"Supervision: {spec.supervision_code}; anatomy A: disabled")
    print(f"Parameter protocol: {spec.parameter_source}")
    print(
        "Checkpoint selection: "
        f"native={spec.native_best_metric}; common={spec.common_best_metric}"
    )
    print("Patient label: biopsy/explicit csPCa only; RA lesion-presence fallback disabled")
    for key in keys:
        print(f"{key:<36}: {getattr(Config, key, None)}")


def execute_experiment(spec: ExperimentSpec, args: argparse.Namespace) -> None:
    if args.dataset_root:
        os.environ["RP_DATASET_ROOT"] = os.path.abspath(args.dataset_root)

    if args.dry_run:
        # Configuration inspection should work on login/local machines that do
        # not have the CUDA/PyTorch training environment installed.
        Config = _dry_run_config(args)
    else:
        # Import after the dataset-root environment override so config.py derives
        # its initial paths from the requested training environment.
        from config import Config

    _refresh_paths(
        Config,
        spec,
        base_dir=args.base_dir,
        dataset_root=args.dataset_root,
        exp_dir=args.exp_dir,
    )
    apply_experiment(Config, spec, args)
    print_resolved_config(Config, spec)

    if args.dry_run:
        return

    validate_split_protocol(Config)

    if not args.force:
        completed = _completed_run(_resolved_exp_dir(args), Config.EXPERIMENT_TAG)
        if completed:
            print(f"Skipping completed {spec.key.upper()} experiment: {completed}")
            return

    import train

    train.main()
