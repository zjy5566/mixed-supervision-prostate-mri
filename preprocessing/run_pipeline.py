#!/usr/bin/env python3
"""Run the supported dataset preprocessing workflows from explicit paths.

The pipeline never writes into the downloaded source directories. All
intermediate, processed, unified, split, and QA artifacts are written below a
separate workspace selected by the user.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Callable, Iterable, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATASET_ORDER = ("pub", "promis", "tcia")
STAGE_ORDER = ("preprocess", "unify", "qa")
SPLIT_MODES = ("auto", "experiment", "none")


class PipelineError(RuntimeError):
    """Raised for a user-correctable pipeline configuration error."""


@dataclass(frozen=True)
class PipelineConfig:
    workspace: Path
    datasets: tuple[str, ...]
    stages: tuple[str, ...]
    pub_root: Optional[Path]
    promis_mri_root: Optional[Path]
    promis_biopsy_root: Optional[Path]
    tcia_dicom_root: Optional[Path]
    tcia_stl_root: Optional[Path]
    tcia_biopsy_table: Optional[Path]
    split_mode: str
    allow_existing_output: bool
    dry_run: bool
    qa_workers: int
    qa_visuals: str

    @property
    def processed_root(self) -> Path:
        return self.workspace / "processed"

    @property
    def pub_processed(self) -> Path:
        return self.processed_root / "pub"

    @property
    def promis_processed(self) -> Path:
        return self.processed_root / "promis"

    @property
    def tcia_extracted(self) -> Path:
        return self.workspace / "intermediate" / "tcia_extracted"

    @property
    def tcia_processed(self) -> Path:
        return self.processed_root / "tcia"

    @property
    def unified_root(self) -> Path:
        return self.workspace / "Unified_Dataset"

    @property
    def qa_root(self) -> Path:
        return self.workspace / "qa" / "input_alignment"


def _path(value: Optional[str]) -> Optional[Path]:
    return Path(value).expanduser().resolve() if value else None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess supported prostate MRI datasets into the unified "
            "training layout without relying on machine-specific paths."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace", required=True, help="Separate output workspace.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        choices=DATASET_ORDER,
        help="Datasets to preprocess and include.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_ORDER,
        default=("preprocess", "unify"),
        help="Pipeline stages. They always execute in canonical order.",
    )

    parser.add_argument(
        "--pub-root",
        help="PUB/RA root containing imagesTr, labelsTr, and zonesTr.",
    )
    parser.add_argument(
        "--promis-mri-root",
        help="PROMIS MRI root containing P-* case directories.",
    )
    parser.add_argument(
        "--promis-biopsy-root",
        help="PROMIS template-biopsy directory containing P-*.csv.",
    )
    parser.add_argument(
        "--tcia-dicom-root",
        help="TCIA root containing Prostate-MRI-US-Biopsy-* DICOM directories.",
    )
    parser.add_argument(
        "--tcia-stl-root",
        help="TCIA directory containing the downloaded .STL files.",
    )
    parser.add_argument(
        "--tcia-biopsy-table",
        help="TCIA biopsy XLSX/CSV with MRI coordinates and Gleason labels.",
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help=(
            "Allow non-empty derived directories. Existing files may be "
            "overwritten and stale files are not removed"
        ),
    )
    parser.add_argument(
        "--split-mode",
        choices=SPLIT_MODES,
        default="auto",
        help=(
            "auto creates experiment splits only for a compatible joint-TCIA "
            "cohort; experiment requires them; none writes only the registry"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the plan without writing files.",
    )
    parser.add_argument(
        "--qa-workers",
        type=int,
        default=1,
        help="Workers for the optional alignment QA stage.",
    )
    parser.add_argument(
        "--qa-visuals",
        choices=("none", "suspect", "top"),
        default="none",
        help="Optional case-level QA figures; none minimizes derived artifacts.",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    datasets = tuple(name for name in DATASET_ORDER if name in set(args.datasets))
    stages = tuple(name for name in STAGE_ORDER if name in set(args.stages))
    if args.qa_workers < 1:
        raise PipelineError("--qa-workers must be at least 1.")
    if "preprocess" in stages and "qa" in stages and "unify" not in stages:
        raise PipelineError(
            "--stages preprocess qa is ambiguous: add unify to audit the new "
            "dataset, or use qa alone to audit an existing Unified_Dataset."
        )
    promis_mri_root = _discover_case_parent(
        _path(args.promis_mri_root),
        "P-*",
    )
    promis_biopsy_root = _discover_file_parent(
        _path(args.promis_biopsy_root),
        "P-*.csv",
    )
    tcia_dicom_root = _discover_case_parent(
        _path(args.tcia_dicom_root),
        "Prostate-MRI-US-Biopsy-*",
    )
    tcia_stl_root = _discover_file_parent(
        _path(args.tcia_stl_root),
        "*.STL",
        case_insensitive_suffix=".stl",
    )
    config = PipelineConfig(
        workspace=Path(args.workspace).expanduser().resolve(),
        datasets=datasets,
        stages=stages,
        pub_root=_path(args.pub_root),
        promis_mri_root=promis_mri_root,
        promis_biopsy_root=promis_biopsy_root,
        tcia_dicom_root=tcia_dicom_root,
        tcia_stl_root=tcia_stl_root,
        tcia_biopsy_table=_path(args.tcia_biopsy_table),
        split_mode=str(args.split_mode),
        allow_existing_output=bool(args.allow_existing_output),
        dry_run=bool(args.dry_run),
        qa_workers=int(args.qa_workers),
        qa_visuals=str(args.qa_visuals),
    )
    validate_workspace(config)
    return config


def _discover_case_parent(root: Optional[Path], pattern: str) -> Optional[Path]:
    """Accept either a canonical case root or one unambiguous wrapper folder."""
    if root is None or not root.is_dir():
        return root
    if any(path.is_dir() for path in root.glob(pattern)):
        return root
    parents = {
        path.parent
        for path in root.rglob(pattern)
        if path.is_dir()
    }
    return next(iter(parents)) if len(parents) == 1 else root


def _discover_file_parent(
    root: Optional[Path],
    pattern: str,
    *,
    case_insensitive_suffix: Optional[str] = None,
) -> Optional[Path]:
    """Accept a file directory or a uniquely nested archive directory."""
    if root is None or not root.is_dir():
        return root

    def matches(path: Path) -> bool:
        if not path.is_file():
            return False
        if case_insensitive_suffix is not None:
            return path.suffix.lower() == case_insensitive_suffix
        return path.match(pattern)

    if any(matches(path) for path in root.iterdir()):
        return root
    parents = {path.parent for path in root.rglob("*") if matches(path)}
    return next(iter(parents)) if len(parents) == 1 else root


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def validate_workspace(config: PipelineConfig) -> None:
    """Keep generated clinical artifacts outside source data and the Git tree."""
    workspace = config.workspace
    if workspace == Path(workspace.anchor):
        raise PipelineError("--workspace cannot be a filesystem root.")
    if workspace == PROJECT_ROOT or PROJECT_ROOT in workspace.parents:
        raise PipelineError(
            "--workspace must be outside the Git repository so generated "
            "clinical artifacts cannot be committed accidentally."
        )
    source_dirs = (
        config.pub_root,
        config.promis_mri_root,
        config.promis_biopsy_root,
        config.tcia_dicom_root,
        config.tcia_stl_root,
        config.tcia_biopsy_table,
    )
    for source in source_dirs:
        if source is not None and _paths_overlap(workspace, source):
            raise PipelineError(
                "--workspace must not contain, equal, or sit inside a raw "
                f"input directory: {source}"
            )


def _require_dir(path: Optional[Path], label: str) -> Path:
    if path is None:
        raise PipelineError(f"{label} is required for the selected stages.")
    if not path.is_dir():
        raise PipelineError(f"{label} is not a directory: {path}")
    return path


def _require_file(path: Optional[Path], label: str) -> Path:
    if path is None:
        raise PipelineError(f"{label} is required for the selected stages.")
    if not path.is_file():
        raise PipelineError(f"{label} is not a file: {path}")
    return path


def _count_complete_pub_cases(root: Path) -> tuple[int, int]:
    images = root / "imagesTr"
    labels = root / "labelsTr"
    zones = root / "zonesTr"
    for directory, label in (
        (images, "PUB imagesTr"),
        (labels, "PUB labelsTr"),
        (zones, "PUB zonesTr"),
    ):
        _require_dir(directory, label)

    t2_files = sorted(images.glob("*_0000.nii.gz"))
    complete = 0
    for t2_path in t2_files:
        patient_id = t2_path.name[: -len("_0000.nii.gz")]
        companions = (
            images / f"{patient_id}_0001.nii.gz",
            images / f"{patient_id}_0002.nii.gz",
            labels / f"{patient_id}.nii.gz",
            zones / f"{patient_id}.nii.gz",
        )
        if all(path.is_file() for path in companions):
            complete += 1
    return len(t2_files), complete


def validate_pub_inputs(root: Path) -> str:
    discovered, complete = _count_complete_pub_cases(root)
    if complete == 0:
        raise PipelineError(
            "PUB input has no complete case with T2/ADC/DWI, lesion label, "
            f"and zone mask under {root}."
        )
    return f"PUB: {complete}/{discovered} discovered cases are complete"


def validate_promis_inputs(mri_root: Path, biopsy_root: Path) -> str:
    case_dirs = sorted(path for path in mri_root.glob("P-*") if path.is_dir())
    if not case_dirs:
        raise PipelineError(f"PROMIS MRI root has no P-* case directories: {mri_root}")
    required = (
        "t2.nii.gz",
        "adc.nii.gz",
        "dwi.nii.gz",
        "gland.nii.gz",
        "gland_zone_20level_set1.nii.gz",
    )
    complete_dirs = [
        case_dir
        for case_dir in case_dirs
        if all((case_dir / filename).is_file() for filename in required)
    ]
    if not complete_dirs:
        zone_masks = sum(
            (case_dir / "gland_zone_20level_set1.nii.gz").is_file()
            for case_dir in case_dirs
        )
        if zone_masks == 0:
            raise PipelineError(
                "PROMIS cases are missing gland_zone_20level_set1.nii.gz. "
                "This repository does not redistribute the third-party "
                "20-zone generator, so these masks must be generated with "
                "separately licensed tooling before this pipeline can run."
            )
        raise PipelineError(
            "PROMIS MRI input has no complete P-* case with the five required "
            f"NIfTI files under {mri_root}."
        )

    csv_files = sorted(biopsy_root.glob("P-*.csv"))
    if not csv_files:
        raise PipelineError(
            f"PROMIS biopsy input has no P-*.csv files under {biopsy_root}."
        )
    required_columns = {
        "zone_id",
        "samtaken",
        "zprescancer",
        "zprimgleason",
        "zsecondgleason",
        "maxccisup",
    }
    invalid_schema: list[tuple[Path, list[str]]] = []
    for csv_path in csv_files:
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            header = set(next(csv.reader(handle), []))
        missing = sorted(required_columns - header)
        if missing:
            invalid_schema.append((csv_path, missing))
    if invalid_schema:
        bad_path, missing = invalid_schema[0]
        raise PipelineError(
            "PROMIS biopsy CSVs must use the schema consumed by this "
            f"implementation. Missing {missing} in {bad_path}"
        )

    complete_ids = {case_dir.name for case_dir in complete_dirs}
    csv_ids = {path.stem for path in csv_files}
    matched = complete_ids & csv_ids
    if not matched:
        raise PipelineError(
            "PROMIS MRI case IDs and biopsy CSV filenames do not overlap. "
            "Expected matching names such as P-0001/ and P-0001.csv."
        )
    return (
        f"PROMIS: {len(complete_dirs)}/{len(case_dirs)} MRI cases are complete; "
        f"{len(matched)} have matching biopsy CSVs"
    )


TCIA_BIOPSY_COLUMNS = {
    "Patient Number",
    "Series Instance UID (MRI)",
    "Core Label",
    "Primary Gleason",
    "Secondary Gleason",
    "Bx Tip X (MRI Coord)",
    "Bx Tip Y (MRI Coord)",
    "Bx Tip Z (MRI Coord)",
    "Bx Base X (MRI Coord)",
    "Bx Base Y (MRI Coord)",
    "Bx Base Z (MRI Coord)",
}


def _table_columns(path: Path) -> set[str]:
    import pandas as pd

    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, nrows=0)
    else:
        frame = pd.read_excel(path, nrows=0)
    # Downstream collection-specific readers use exact column names, so the
    # preflight must not accept whitespace they would later reject.
    return {str(column) for column in frame.columns}


def validate_tcia_inputs(
    dicom_root: Path,
    stl_root: Path,
    biopsy_table: Path,
) -> str:
    patients = sorted(
        path
        for path in dicom_root.iterdir()
        if path.is_dir() and path.name.startswith("Prostate-MRI-US-Biopsy-")
    )
    if not patients:
        raise PipelineError(
            "TCIA DICOM root has no Prostate-MRI-US-Biopsy-* directories: "
            f"{dicom_root}"
        )
    if not any(
        path.is_file() and path.suffix.lower() == ".dcm"
        for patient in patients
        for path in patient.rglob("*")
    ):
        raise PipelineError(
            "TCIA extraction currently requires DICOM files ending in .dcm."
        )

    stls = sorted(
        path
        for path in stl_root.iterdir()
        if path.is_file() and path.suffix.lower() == ".stl"
    )
    if not stls:
        raise PipelineError(
            "TCIA STL root has no .STL files at its top level: "
            f"{stl_root}"
        )

    columns = _table_columns(biopsy_table)
    missing = sorted(TCIA_BIOPSY_COLUMNS - columns)
    if missing:
        raise PipelineError(
            "TCIA biopsy table is missing columns required by the current "
            f"implementation: {missing}"
        )
    return (
        f"TCIA: {len(patients)} patient directories, {len(stls)} STL files, "
        "and a compatible biopsy table found"
    )


def validate_raw_inputs(config: PipelineConfig) -> list[str]:
    summaries = []
    if "pub" in config.datasets:
        summaries.append(validate_pub_inputs(_require_dir(config.pub_root, "--pub-root")))
    if "promis" in config.datasets:
        summaries.append(
            validate_promis_inputs(
                _require_dir(config.promis_mri_root, "--promis-mri-root"),
                _require_dir(config.promis_biopsy_root, "--promis-biopsy-root"),
            )
        )
    if "tcia" in config.datasets:
        summaries.append(
            validate_tcia_inputs(
                _require_dir(config.tcia_dicom_root, "--tcia-dicom-root"),
                _require_dir(config.tcia_stl_root, "--tcia-stl-root"),
                _require_file(config.tcia_biopsy_table, "--tcia-biopsy-table"),
            )
        )
    return summaries


def _nonempty(path: Path) -> bool:
    return path.is_dir() and next(path.iterdir(), None) is not None


def _prepare_output(path: Path, allow_existing: bool) -> None:
    if path.exists() and not path.is_dir():
        raise PipelineError(f"Derived output path is not a directory: {path}")
    if _nonempty(path) and not allow_existing:
        raise PipelineError(
            f"Derived output is not empty: {path}. Use a new --workspace or "
            "pass --allow-existing-output after reviewing overwrite and stale-file risks."
        )
    path.mkdir(parents=True, exist_ok=True)


def validate_output_state(config: PipelineConfig) -> None:
    """Check every selected output before the first stage writes anything."""
    outputs: list[Path] = []
    if "preprocess" in config.stages:
        if "pub" in config.datasets:
            outputs.append(config.pub_processed)
        if "promis" in config.datasets:
            outputs.append(config.promis_processed)
        if "tcia" in config.datasets:
            outputs.extend((config.tcia_extracted, config.tcia_processed))
    if "unify" in config.stages:
        outputs.append(config.unified_root)
    if "qa" in config.stages:
        outputs.append(config.qa_root)

    always_fresh = {config.unified_root, config.qa_root}
    for output in dict.fromkeys(outputs):
        if output.exists() and not output.is_dir():
            raise PipelineError(f"Derived output path is not a directory: {output}")
        if _nonempty(output) and output in always_fresh:
            raise PipelineError(
                f"Derived output must be fresh for this stage: {output}. "
                "Use a new --workspace so stale registry, split, or QA files "
                "cannot be mistaken for current outputs."
            )
        if _nonempty(output) and not config.allow_existing_output:
            raise PipelineError(
                f"Derived output is not empty: {output}. Use a new --workspace "
                "or pass --allow-existing-output after reviewing overwrite "
                "and stale-file risks."
            )


def _count_dirs_with(root: Path, required: Iterable[str]) -> int:
    if not root.is_dir():
        return 0
    names = tuple(required)
    return sum(
        all((case_dir / name).is_file() for name in names)
        for case_dir in root.iterdir()
        if case_dir.is_dir()
    )


def _tcia_supervision_counts(root: Path) -> tuple[int, int]:
    """Return counts of eligible and joint-TBx/SBx processed TCIA cases."""
    eligible = 0
    joint = 0
    case_dirs = root.iterdir() if root.is_dir() else ()
    for case_dir in case_dirs:
        if not case_dir.is_dir() or not (case_dir / "input_tensor.npy").is_file():
            continue
        has_target = (case_dir / "target_mask.nii.gz").is_file()
        has_sbx = (
            (case_dir / "zones_mask.nii.gz").is_file()
            and (case_dir / "systematic_labels.npy").is_file()
        )
        if has_target or has_sbx:
            eligible += 1
        if has_target and has_sbx:
            joint += 1
    return eligible, joint


def _joint_tcia_duplicate_patients(root: Path) -> list[str]:
    """Find original TCIA patients represented by multiple joint study cases."""
    counts: dict[str, int] = {}
    case_dirs = root.iterdir() if root.is_dir() else ()
    for case_dir in case_dirs:
        if not case_dir.is_dir() or not (case_dir / "input_tensor.npy").is_file():
            continue
        has_target = (case_dir / "target_mask.nii.gz").is_file()
        has_sbx = (
            (case_dir / "zones_mask.nii.gz").is_file()
            and (case_dir / "systematic_labels.npy").is_file()
        )
        if has_target and has_sbx:
            base_patient = case_dir.name.split("_", 1)[0]
            counts[base_patient] = counts.get(base_patient, 0) + 1
    return sorted(patient for patient, count in counts.items() if count > 1)


def validate_processed_inputs(config: PipelineConfig) -> list[str]:
    summaries = []
    if "pub" in config.datasets:
        image_suffix = "_img.npy"
        count = sum(
            (config.pub_processed / f"{path.name[:-len(image_suffix)]}_lab.npy").is_file()
            and (config.pub_processed / f"{path.name[:-len(image_suffix)]}_zone.npy").is_file()
            for path in config.pub_processed.glob(f"*{image_suffix}")
        )
        if count == 0:
            raise PipelineError(
                "No complete processed PUB image/label/zone triplets found in "
                f"{config.pub_processed}."
            )
        summaries.append(f"processed PUB cases: {count}")
    if "promis" in config.datasets:
        count = _count_dirs_with(
            config.promis_processed,
            ("input_tensor.npy", "zones_mask.nii.gz", "systematic_labels.npy"),
        )
        if count == 0:
            raise PipelineError(
                f"No complete processed PROMIS cases found in {config.promis_processed}."
            )
        summaries.append(f"processed PROMIS cases: {count}")
    if "tcia" in config.datasets:
        eligible, joint = _tcia_supervision_counts(config.tcia_processed)
        if eligible == 0:
            raise PipelineError(
                "No processed TCIA case has an image tensor plus TBx or SBx "
                f"supervision in {config.tcia_processed}."
            )
        summaries.append(
            f"eligible processed TCIA cases: {eligible} ({joint} joint TBx+SBx)"
        )
    if config.split_mode == "experiment":
        if "tcia" not in config.datasets:
            raise PipelineError(
                "--split-mode experiment requires selecting the TCIA dataset."
            )
        _, joint = _tcia_supervision_counts(config.tcia_processed)
        if joint == 0:
            raise PipelineError(
                "--split-mode experiment requires at least one processed TCIA "
                "case with both TBx and SBx supervision."
            )
        duplicate_patients = _joint_tcia_duplicate_patients(config.tcia_processed)
        if duplicate_patients:
            raise PipelineError(
                "Experiment splitting currently requires one derived MRI study "
                "per original TCIA patient. Resolve duplicate study cases before "
                f"splitting: {duplicate_patients[:5]}"
            )
    return summaries


def print_plan(config: PipelineConfig, summaries: Sequence[str]) -> None:
    print("Preprocessing plan")
    print(f"  workspace: {config.workspace}")
    print(f"  datasets:  {', '.join(config.datasets)}")
    print(f"  stages:    {', '.join(config.stages)}")
    print(f"  split mode: {config.split_mode}")
    if "qa" in config.stages:
        print(f"  QA visuals: {config.qa_visuals}")
    for summary in summaries:
        print(f"  input:     {summary}")

    step = 1
    if "preprocess" in config.stages:
        if "pub" in config.datasets:
            print(f"  {step}. PUB MRI/label preprocessing -> {config.pub_processed}")
            step += 1
        if "promis" in config.datasets:
            print(f"  {step}. PROMIS MRI preprocessing -> {config.promis_processed}")
            step += 1
            print(f"  {step}. PROMIS biopsy labels -> {config.promis_processed}")
            step += 1
        if "tcia" in config.datasets:
            print(f"  {step}. TCIA DICOM/STL extraction -> {config.tcia_extracted}")
            step += 1
            print(f"  {step}. TCIA gland/target masks -> {config.tcia_extracted}")
            step += 1
            print(f"  {step}. TCIA systematic-biopsy labels -> {config.tcia_extracted}")
            step += 1
            print(f"  {step}. TCIA resample/crop/tensor export -> {config.tcia_processed}")
            step += 1
    if "unify" in config.stages:
        print(
            f"  {step}. Unified dataset, registry, and eligible splits "
            f"-> {config.unified_root}"
        )
        step += 1
    if "qa" in config.stages:
        print(f"  {step}. Read-only multimodal alignment QA -> {config.qa_root}")


def run_pub(config: PipelineConfig) -> None:
    from preprocessing.radiologist_preprocessor import ProstateDataPreprocessor

    _prepare_output(config.pub_processed, config.allow_existing_output)
    processor = ProstateDataPreprocessor(
        str(config.pub_root),
        str(config.pub_processed),
    )
    processor.run_all()
    image_suffix = "_img.npy"
    count = sum(
        (config.pub_processed / f"{path.name[:-len(image_suffix)]}_lab.npy").is_file()
        and (config.pub_processed / f"{path.name[:-len(image_suffix)]}_zone.npy").is_file()
        for path in config.pub_processed.glob(f"*{image_suffix}")
    )
    if count == 0:
        raise PipelineError(
            "PUB preprocessing produced no complete image/label/zone triplet."
        )


def run_promis(config: PipelineConfig) -> None:
    from preprocessing.PROMIS_preprocessor_MRI import batch_preprocess
    from preprocessing.PROMIS_preprocessor_cvs import batch_convert_csv_to_npy

    _prepare_output(config.promis_processed, config.allow_existing_output)
    batch_preprocess(
        str(config.promis_mri_root),
        str(config.promis_processed),
        lesion_only=False,
        overwrite_lesion=False,
    )
    if _count_dirs_with(config.promis_processed, ("input_tensor.npy",)) == 0:
        raise PipelineError("PROMIS MRI preprocessing produced no image tensors.")
    batch_convert_csv_to_npy(
        str(config.promis_biopsy_root),
        str(config.promis_processed),
    )
    complete = _count_dirs_with(
        config.promis_processed,
        ("input_tensor.npy", "zones_mask.nii.gz", "systematic_labels.npy"),
    )
    if complete == 0:
        raise PipelineError(
            "PROMIS preprocessing produced no case with an image tensor, "
            "zone mask, and biopsy label vector."
        )


def run_tcia(config: PipelineConfig) -> None:
    from tqdm import tqdm

    from preprocessing.TCIA_preprocessing_MRI_extract import batch_extract_mri
    from preprocessing.TCIA_stl2mask import batch_convert_stls, read_table
    from preprocessing.TCIA_preprocessing_biopsydata import process_patient_folder
    from preprocessing.TCIA_preprocessing_MRI import process_single_patient

    _prepare_output(config.tcia_extracted, config.allow_existing_output)
    _prepare_output(config.tcia_processed, config.allow_existing_output)

    batch_extract_mri(
        str(config.tcia_dicom_root),
        str(config.tcia_extracted),
        str(config.tcia_stl_root),
    )
    extracted_folders = sorted(
        path
        for path in config.tcia_extracted.iterdir()
        if path.is_dir() and path.name.startswith("Prostate-MRI-US-Biopsy-")
    )
    if not extracted_folders:
        raise PipelineError("TCIA extraction produced no case directories.")

    batch_convert_stls(
        str(config.tcia_extracted),
        None,
        str(config.tcia_biopsy_table),
    )
    if _count_dirs_with(config.tcia_extracted, ("gland_mask.nii.gz",)) == 0:
        raise PipelineError("TCIA STL conversion produced no gland masks.")

    biopsy_frame = read_table(str(config.tcia_biopsy_table))
    if biopsy_frame is None:
        raise PipelineError("TCIA biopsy table could not be read.")
    for folder in tqdm(extracted_folders, desc="TCIA biopsy labels"):
        process_patient_folder(folder.name, biopsy_frame, str(config.tcia_extracted))

    results: dict[str, int] = {}
    for folder in tqdm(extracted_folders, desc="TCIA tensor preprocessing"):
        result = process_single_patient(
            folder.name,
            str(folder),
            str(config.tcia_processed),
        )
        results[result] = results.get(result, 0) + 1
    if results.get("SUCCESS", 0) == 0:
        raise PipelineError(
            "TCIA tensor preprocessing produced no complete T2/ADC/DWI/gland case."
        )
    print(f"TCIA tensor summary: {results}")
    eligible, joint = _tcia_supervision_counts(config.tcia_processed)
    if eligible == 0:
        raise PipelineError(
            "TCIA preprocessing produced image tensors but no case with usable "
            "TBx or SBx supervision. Review STL overlap and biopsy matching."
        )
    print(f"TCIA supervision summary: eligible={eligible}, joint_tbx_sbx={joint}")


def run_unify(config: PipelineConfig) -> None:
    from preprocessing.Dataset_settingup import create_unified_dataset

    validate_processed_inputs(config)
    _prepare_output(config.unified_root, allow_existing=False)
    disabled_root = config.workspace / ".not_selected"
    effective_split_mode = config.split_mode
    duplicate_patients = _joint_tcia_duplicate_patients(config.tcia_processed)
    if effective_split_mode == "auto" and duplicate_patients:
        print(
            "Experiment splits skipped: multiple joint-TCIA study cases map to "
            "the same original patient, and row-level splitting could leak a "
            "patient across partitions. Use a patient-level curation policy "
            "before requesting experiment splits."
        )
        effective_split_mode = "none"

    result = create_unified_dataset(
        str(config.workspace),
        pub_dir=(
            str(config.pub_processed)
            if "pub" in config.datasets
            else str(disabled_root / "pub")
        ),
        promis_dir=(
            str(config.promis_processed)
            if "promis" in config.datasets
            else str(disabled_root / "promis")
        ),
        promis_labels_dir=(
            str(config.promis_processed)
            if "promis" in config.datasets
            else str(disabled_root / "promis_labels")
        ),
        tcia_dir=(
            str(config.tcia_processed)
            if "tcia" in config.datasets
            else str(disabled_root / "tcia")
        ),
        output_dir=str(config.unified_root),
        split_mode=effective_split_mode,
    )
    if result is None:
        raise PipelineError("Unified dataset creation found no eligible cases.")


def run_qa(config: PipelineConfig) -> None:
    if not config.unified_root.is_dir():
        raise PipelineError(
            f"Unified dataset does not exist for QA: {config.unified_root}"
        )
    _prepare_output(config.qa_root, allow_existing=False)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "preprocessing" / "check_input_alignment.py"),
        "--dataset-root",
        str(config.unified_root),
        "--output-dir",
        str(config.qa_root),
        "--sources",
        *(name.upper() for name in config.datasets),
        "--workers",
        str(config.qa_workers),
        "--visuals",
        config.qa_visuals,
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _execute_stage(name: str, operation: Callable[[], None]) -> None:
    """Add stage context and a partial-output warning to runtime failures."""
    try:
        operation()
    except PipelineError as exc:
        raise PipelineError(
            f"{name} failed: {exc} The workspace may contain partial derived "
            "outputs; inspect it or use a new workspace before retrying."
        ) from exc
    except Exception as exc:
        raise PipelineError(
            f"{name} failed ({type(exc).__name__}: {exc}). The workspace may "
            "contain partial derived outputs; inspect it or use a new workspace "
            "before retrying."
        ) from exc


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        config = build_config(parse_args(argv))
        summaries: list[str] = []
        if "preprocess" in config.stages:
            summaries.extend(validate_raw_inputs(config))
        elif "unify" in config.stages:
            summaries.extend(validate_processed_inputs(config))
        if "qa" in config.stages and "unify" not in config.stages:
            if not config.unified_root.is_dir():
                raise PipelineError(
                    f"QA requires an existing unified dataset: {config.unified_root}"
                )

        validate_output_state(config)

        print_plan(config, summaries)
        if config.dry_run:
            print("Dry run complete. No files were written.")
            return 0

        config.workspace.mkdir(parents=True, exist_ok=True)
        if "preprocess" in config.stages:
            if "pub" in config.datasets:
                _execute_stage("PUB preprocessing", lambda: run_pub(config))
            if "promis" in config.datasets:
                _execute_stage("PROMIS preprocessing", lambda: run_promis(config))
            if "tcia" in config.datasets:
                _execute_stage("TCIA preprocessing", lambda: run_tcia(config))
        if "unify" in config.stages:
            _execute_stage("dataset unification", lambda: run_unify(config))
        if "qa" in config.stages:
            _execute_stage("alignment QA", lambda: run_qa(config))

        print("Pipeline complete.")
        if config.unified_root.is_dir():
            print(f"Unified dataset: {config.unified_root}")
        else:
            print(f"Processed datasets: {config.processed_root}")
        return 0
    except PipelineError as exc:
        print(f"Pipeline configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Pipeline interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(
            f"Pipeline failed ({type(exc).__name__}: {exc}). "
            "No further stages were run.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
