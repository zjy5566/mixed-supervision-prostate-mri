#!/usr/bin/env python3
"""Run the T0--T3 masked-TBx-Dice ablation on the B1 TBx-only setting.

All runs retain the sampled positive/negative TBx BCE baseline. The sole
experimental variable is the coefficient of positive-case Dice calculated
inside sampled target ROIs.
"""

from __future__ import annotations

import argparse
from typing import Dict

from supervision_experiment_runner import (
    ExperimentSpec,
    add_common_arguments,
    execute_experiment,
)


def _spec(key: str, dice_weight: float) -> ExperimentSpec:
    return ExperimentSpec(
        key=key,
        mode=f"{key.upper()}_B1_TBX_DICE_ABLATION",
        description=(
            "B1 TBx sampled pos/neg BCE + masked positive-case TBx Dice "
            f"(weight={dice_weight:g})"
        ),
        parameter_source="B1 LR5e-5 + PosW1 selected run",
        train_csv="B1_TCIA_TBx_baseline_train.csv",
        use_tbx=True,
        tbx_weight=1.0,
        tbx_dice_weight=dice_weight,
        lr=5e-5,
        pos_weight=1.0,
        sys_pos_weight=1.0,
        native_best_metric="tbx_native",
        common_best_metric="common_multilevel",
    )


EXPERIMENTS: Dict[str, ExperimentSpec] = {
    "t0": _spec("t0", 0.0),
    "t1": _spec("t1", 0.25),
    "t2": _spec("t2", 0.50),
    "t3": _spec("t3", 1.00),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one T0--T3 TBx Dice ablation.")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    add_common_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    execute_experiment(EXPERIMENTS[args.experiment], args)


if __name__ == "__main__":
    main()
