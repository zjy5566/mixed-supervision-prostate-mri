#!/usr/bin/env python3
"""Freeze validation-only score settings, then evaluate internal/external data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Optional


PROJECT_DIR = Path(__file__).resolve().parents[1]
TEST_SCRIPT = PROJECT_DIR / "test.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select thresholds (and fit patient pooling only when configured) "
            "on validation once, then reuse the frozen artifact for internal "
            "and external test."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--internal-csv", required=True)
    parser.add_argument("--external-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--checkpoint-label", default="best_common")
    parser.add_argument(
        "--experiment-mode",
        default=None,
        help=(
            "Config.EXPERIMENT_MODE used by test.py. The value is injected "
            "before Config is imported (for example N4_MIXED)."
        ),
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--unified-data-dir", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--save-images",
        choices=("none", "errors", "good", "representative", "all"),
        default="representative",
    )
    parser.add_argument("--max-images", type=int, default=12)
    return parser.parse_args()


def absolute_existing(path: str, label: str) -> str:
    resolved = os.path.abspath(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def run(command: list[str], experiment_mode: Optional[str] = None) -> None:
    print("\n[Run] " + " ".join(command), flush=True)
    environment = os.environ.copy()
    if experiment_mode:
        environment["RP_EXPERIMENT_MODE"] = experiment_mode
    subprocess.run(command, cwd=PROJECT_DIR, env=environment, check=True)


def main() -> None:
    args = parse_args()
    checkpoint = absolute_existing(args.checkpoint, "checkpoint")
    validation_csv = absolute_existing(args.validation_csv, "validation CSV")
    internal_csv = absolute_existing(args.internal_csv, "internal CSV")
    external_csv = absolute_existing(args.external_csv, "external CSV")
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    frozen_path = output_root / "frozen_validation_thresholds.json"

    if args.experiment_mode:
        print(f"[Experiment mode] {args.experiment_mode}", flush=True)

    common = [
        sys.executable,
        str(TEST_SCRIPT),
        "--checkpoint",
        checkpoint,
        "--experiment-dir",
        os.path.abspath(args.experiment_dir or os.path.dirname(checkpoint)),
        "--validation-csv",
        validation_csv,
        "--checkpoint-label",
        args.checkpoint_label,
        "--frozen-thresholds",
        str(frozen_path),
    ]
    if args.dataset_root:
        common.extend(["--dataset-root", os.path.abspath(args.dataset_root)])
    if args.unified_data_dir:
        common.extend(
            ["--unified-data-dir", os.path.abspath(args.unified_data_dir)]
        )
    if args.device:
        common.extend(["--device", args.device])
    if args.batch_size is not None:
        common.extend(["--batch-size", str(args.batch_size)])
    if args.num_workers is not None:
        common.extend(["--num-workers", str(args.num_workers)])

    run(
        common
        + [
            "--validation-only",
            "--output-dir",
            str(output_root / "validation"),
        ],
        args.experiment_mode,
    )

    for label, csv_path in (("internal", internal_csv), ("external", external_csv)):
        run(
            common
            + [
                "--test-csv",
                csv_path,
                "--dataset-label",
                label,
                "--output-dir",
                str(output_root / label),
                "--save-images",
                args.save_images,
                "--max-images",
                str(args.max_images),
            ],
            args.experiment_mode,
        )

    print("\nFrozen evaluation complete")
    print(f"  thresholds: {frozen_path}")
    print(f"  validation: {output_root / 'validation'}")
    print(f"  internal:   {output_root / 'internal'}")
    print(f"  external:   {output_root / 'external'}")


if __name__ == "__main__":
    main()
