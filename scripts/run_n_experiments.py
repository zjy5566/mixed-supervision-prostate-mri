#!/usr/bin/env python3
"""Run one strictly aligned N-family experiment (N1--N5) from scripts/.

N experiments are anchored by dense RA supervision. N5 is the only N-family
main experiment that adds the explicit patient-level supervision pathway.
Every active TBx/SBx branch uses the N4 weak-loss weight and starts at epoch 15.
Outside-gland suppression is reserved for a separate anatomy ablation.
"""

from __future__ import annotations

import argparse
from typing import Dict

from supervision_experiment_runner import (
    ExperimentSpec,
    add_common_arguments,
    execute_experiment,
)


EXPERIMENTS: Dict[str, ExperimentSpec] = {
    "n1": ExperimentSpec(
        key="n1",
        mode="N1_RADIOLOGIST_ONLY",
        description="Dense RA voxel-level supervision only",
        parameter_source="strict N ablation dense reference",
        design_tag="StrictAbl",
        train_csv="N1_radiologist_only_train.csv",
        train_dataset_task="radiologist_only",
        use_dense=True,
        dense_weight=1.0,
        lr=1e-4,
        pos_weight=2.0,
        sys_pos_weight=1.0,
        native_best_metric="lesion_dice",
        common_best_metric="patient_auprc",
        patient_pooling="logit_lme",
        compare_patient_pooling=False,
    ),
    "n2": ExperimentSpec(
        key="n2",
        mode="N2_PUB_TCIA_TBX_ONLY",
        description="Dense RA + TBx target-ROI supervision",
        parameter_source="strict N ablation: D/T=1/0.05, TBx from epoch 15",
        design_tag="StrictAbl",
        train_csv="N2_PUB_TCIA_TBx_only_train.csv",
        use_dense=True,
        use_tbx=True,
        dense_weight=1.0,
        tbx_weight=0.05,
        use_curriculum=True,
        dense_start=1,
        tbx_start=15,
        lr=1e-4,
        pos_weight=2.0,
        sys_pos_weight=1.0,
        native_best_metric="lesion_dice",
        common_best_metric="patient_auprc",
        patient_pooling="logit_lme",
        compare_patient_pooling=False,
    ),
    "n3": ExperimentSpec(
        key="n3",
        mode="N3_PUB_TCIA_SBX_ONLY",
        description="Dense RA + SBx region supervision",
        parameter_source="strict N ablation: D/S=1/0.25, SBx from epoch 15",
        design_tag="StrictAbl",
        train_csv="N3_PUB_TCIA_SBx_only_train.csv",
        use_dense=True,
        use_sbx=True,
        dense_weight=1.0,
        sbx_weight=0.25,
        use_curriculum=True,
        dense_start=1,
        sbx_start=15,
        lr=1e-4,
        pos_weight=2.0,
        sys_pos_weight=1.0,
        native_best_metric="lesion_dice",
        common_best_metric="patient_auprc",
        patient_pooling="logit_lme",
        compare_patient_pooling=False,
    ),
    "n4": ExperimentSpec(
        key="n4",
        mode="N4_MIXED_CLEAN",
        description="Dense RA + TBx + SBx supervision",
        parameter_source="strict N ablation: D/T/S=1/0.05/0.25, weak losses from epoch 15",
        design_tag="StrictAbl",
        train_csv="N4_mixed_PUB_TCIA_train.csv",
        use_dense=True,
        use_tbx=True,
        use_sbx=True,
        dense_weight=1.0,
        tbx_weight=0.05,
        sbx_weight=0.25,
        use_curriculum=True,
        dense_start=1,
        tbx_start=15,
        sbx_start=15,
        lr=1e-4,
        pos_weight=2.0,
        sys_pos_weight=1.0,
        native_best_metric="lesion_dice",
        common_best_metric="patient_auprc",
        patient_pooling="logit_lme",
        compare_patient_pooling=False,
    ),
    "n5": ExperimentSpec(
        key="n5",
        mode="N5_MIXED_PATIENT",
        description="Dense RA + TBx + SBx + biopsy-confirmed patient supervision",
        parameter_source="strict N ablation weights/schedule + PatientRiskW0.05",
        design_tag="StrictAbl",
        train_csv="N4_mixed_PUB_TCIA_train.csv",
        use_dense=True,
        use_tbx=True,
        use_sbx=True,
        use_patient=True,
        dense_weight=1.0,
        tbx_weight=0.05,
        sbx_weight=0.25,
        patient_weight=0.05,
        use_curriculum=True,
        dense_start=1,
        tbx_start=15,
        sbx_start=15,
        patient_start=1,
        lr=1e-4,
        pos_weight=2.0,
        sys_pos_weight=1.0,
        native_best_metric="lesion_dice",
        common_best_metric="patient_auprc",
        patient_pooling="logit_lme",
        compare_patient_pooling=False,
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one N1--N5 experiment.")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    add_common_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    execute_experiment(EXPERIMENTS[args.experiment], args)


if __name__ == "__main__":
    main()
