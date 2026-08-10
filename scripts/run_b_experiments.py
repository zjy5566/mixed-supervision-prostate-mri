#!/usr/bin/env python3
"""Run one strictly aligned B-family experiment (B0--B4) from scripts/.

B experiments do not use dense RA supervision. B0--B4 share one optimiser
configuration, and every active biopsy branch starts at epoch 1. P remains an
explicit supervision level and A remains disabled.
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
    "b0": ExperimentSpec(
        key="b0",
        mode="B0_PATIENT_ONLY",
        description="Biopsy-confirmed patient-level supervision only",
        parameter_source="strict B ablation shared optimiser; PatientRiskW0.05",
        design_tag="StrictAbl",
        train_csv="B3_TCIA_TBx_SBx_train.csv",
        use_patient=True,
        patient_weight=0.05,
        lr=1e-4,
        pos_weight=2.0,
        sys_pos_weight=1.0,
        native_best_metric="patient_auprc",
        common_best_metric="patient_auprc",
        patient_pooling="logit_lme",
        compare_patient_pooling=False,
    ),
    "b1": ExperimentSpec(
        key="b1",
        mode="B1_TCIA_TBX_BASELINE",
        description="TBx target-ROI supervision only",
        parameter_source="strict B ablation shared optimiser and epoch-1 schedule",
        design_tag="StrictAbl",
        train_csv="B1_TCIA_TBx_baseline_train.csv",
        use_tbx=True,
        tbx_weight=1.0,
        lr=1e-4,
        pos_weight=2.0,
        sys_pos_weight=1.0,
        native_best_metric="tbx_native",
        common_best_metric="patient_auprc",
        patient_pooling="logit_lme",
        compare_patient_pooling=False,
    ),
    "b2": ExperimentSpec(
        key="b2",
        mode="B2_TCIA_SBX_ONLY",
        description="SBx region-level supervision only",
        parameter_source="strict B ablation shared optimiser and epoch-1 schedule",
        design_tag="StrictAbl",
        train_csv="B2_TCIA_SBx_only_train.csv",
        use_sbx=True,
        sbx_weight=1.0,
        lr=1e-4,
        pos_weight=2.0,
        sys_pos_weight=1.0,
        native_best_metric="region_auprc",
        common_best_metric="patient_auprc",
        patient_pooling="logit_lme",
        compare_patient_pooling=False,
    ),
    "b3": ExperimentSpec(
        key="b3",
        mode="B3_TCIA_TBX_SBX",
        description="TBx target-ROI + SBx region supervision",
        parameter_source="strict B ablation shared optimiser and epoch-1 schedule",
        design_tag="StrictAbl",
        train_csv="B3_TCIA_TBx_SBx_train.csv",
        use_tbx=True,
        use_sbx=True,
        tbx_weight=1.0,
        sbx_weight=1.0,
        use_curriculum=False,
        tbx_start=1,
        sbx_start=1,
        lr=1e-4,
        pos_weight=2.0,
        sys_pos_weight=1.0,
        native_best_metric="tbx_sbx_native",
        common_best_metric="patient_auprc",
        patient_pooling="logit_lme",
        compare_patient_pooling=False,
    ),
    "b4": ExperimentSpec(
        key="b4",
        mode="B4_TCIA_TBX_SBX_PATIENT",
        description="TBx + SBx + biopsy-confirmed patient-level supervision",
        parameter_source="strict B ablation shared optimiser/schedule; PatientRiskW0.05",
        design_tag="StrictAbl",
        train_csv="B3_TCIA_TBx_SBx_train.csv",
        use_tbx=True,
        use_sbx=True,
        use_patient=True,
        tbx_weight=1.0,
        sbx_weight=1.0,
        patient_weight=0.05,
        use_curriculum=False,
        tbx_start=1,
        sbx_start=1,
        patient_start=1,
        lr=1e-4,
        pos_weight=2.0,
        sys_pos_weight=1.0,
        native_best_metric="tbx_sbx_patient_native",
        common_best_metric="patient_auprc",
        patient_pooling="logit_lme",
        compare_patient_pooling=False,
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one B0--B4 experiment.")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    add_common_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    execute_experiment(EXPERIMENTS[args.experiment], args)


if __name__ == "__main__":
    main()
