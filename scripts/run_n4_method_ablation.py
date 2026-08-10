#!/usr/bin/env python3
"""Run the shared-reference N4 method ablations.

Five unique trainings cover three comparisons without retraining the identical
LME / patient-off / staged reference three times.
"""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

from supervision_experiment_runner import (
    ExperimentSpec,
    add_common_arguments,
    execute_experiment,
)


def _spec(
    *,
    key: str,
    description: str,
    tag_tokens: Tuple[str, ...],
    mil_pooling: str = "lme",
    use_patient: bool = False,
    patient_weight: float = 0.0,
    patient_start: int = 1,
    use_curriculum: bool = True,
    tbx_start: int = 15,
    sbx_start: int = 15,
) -> ExperimentSpec:
    return ExperimentSpec(
        key=key,
        family="n",
        mode="N4_MIXED",
        description=description,
        parameter_source=(
            "N4 shared ablation protocol: D/T/S=1/0.05/0.25, "
            "LR1e-4, PosW2, SysPosW1"
        ),
        train_csv="N4_mixed_PUB_TCIA_train.csv",
        use_dense=True,
        use_tbx=True,
        use_sbx=True,
        use_patient=use_patient,
        dense_weight=1.0,
        tbx_weight=0.05,
        sbx_weight=0.25,
        patient_weight=patient_weight,
        use_curriculum=use_curriculum,
        dense_start=1,
        tbx_start=tbx_start,
        sbx_start=sbx_start,
        patient_start=patient_start,
        lr=1e-4,
        pos_weight=2.0,
        sys_pos_weight=1.0,
        tbx_dice_weight=0.0,
        native_best_metric="lesion_dice",
        common_best_metric="patient_auprc",
        patient_pooling="logit_lme",
        compare_patient_pooling=False,
        mil_pooling=mil_pooling,
        lme_r=8.0,
        tag_tokens=tag_tokens,
    )


EXPERIMENTS: Dict[str, ExperimentSpec] = {
    "n4abl_ref": _spec(
        key="n4abl_ref",
        description=(
            "Shared N4 reference: SBx LME, patient loss off, staged TBx/SBx"
        ),
        tag_tokens=("AblSharedRef", "SBxPoolLME", "StartsD1T15S15"),
    ),
    "n4abl_sbx_mean": _spec(
        key="n4abl_sbx_mean",
        description="SBx pooling ablation: logit-space mean",
        tag_tokens=("AblSBxPooling", "SBxPoolMean", "StartsD1T15S15"),
        mil_pooling="mean",
    ),
    "n4abl_sbx_max": _spec(
        key="n4abl_sbx_max",
        description="SBx pooling ablation: max",
        tag_tokens=("AblSBxPooling", "SBxPoolMax", "StartsD1T15S15"),
        mil_pooling="max",
    ),
    "n4abl_patient": _spec(
        key="n4abl_patient",
        description=(
            "Patient-supervision ablation: add gland-restricted patient loss "
            "from epoch 15"
        ),
        tag_tokens=(
            "AblPatient",
            "SBxPoolLME",
            "StartsD1T15S15P15",
        ),
        use_patient=True,
        patient_weight=0.05,
        patient_start=15,
    ),
    "n4abl_all_e1": _spec(
        key="n4abl_all_e1",
        description="Curriculum ablation: dense, TBx and SBx all active at epoch 1",
        tag_tokens=("AblCurriculum", "SBxPoolLME", "StartsD1T1S1"),
        use_curriculum=False,
        tbx_start=1,
        sbx_start=1,
    ),
}


ABLATION_COMPARISONS = {
    "sbx_pooling": (
        "n4abl_sbx_mean",
        "n4abl_sbx_max",
        "n4abl_ref",
    ),
    "patient_supervision": ("n4abl_ref", "n4abl_patient"),
    "curriculum": ("n4abl_all_e1", "n4abl_ref"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one N4 method ablation.")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    add_common_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    execute_experiment(EXPERIMENTS[args.experiment], args)


if __name__ == "__main__":
    main()
