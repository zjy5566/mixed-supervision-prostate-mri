from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from preprocessing.Dataset_settingup import create_unified_dataset
from preprocessing.run_pipeline import (
    PipelineError,
    _execute_stage,
    build_config,
    parse_args,
    validate_output_state,
    validate_promis_inputs,
    validate_pub_inputs,
)


def _touch_files(root: Path, relative_paths: tuple[str, ...]) -> None:
    for relative_path in relative_paths:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def _make_complete_pub_raw_layout(root: Path, patient_id: str = "case001") -> None:
    _touch_files(
        root,
        (
            f"imagesTr/{patient_id}_0000.nii.gz",
            f"imagesTr/{patient_id}_0001.nii.gz",
            f"imagesTr/{patient_id}_0002.nii.gz",
            f"labelsTr/{patient_id}.nii.gz",
            f"zonesTr/{patient_id}.nii.gz",
        ),
    )


def _make_promis_mri_case(
    root: Path,
    patient_id: str = "P-0001",
    *,
    include_zone_mask: bool = True,
) -> None:
    required = [
        "t2.nii.gz",
        "adc.nii.gz",
        "dwi.nii.gz",
        "gland.nii.gz",
    ]
    if include_zone_mask:
        required.append("gland_zone_20level_set1.nii.gz")
    _touch_files(root / patient_id, tuple(required))


def _make_promis_biopsy_layout(root: Path, patient_id: str = "P-0001") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{patient_id}.csv").write_text(
        (
            "zone_id,samtaken,zprescancer,zprimgleason,"
            "zsecondgleason,maxccisup\n1,1,1,3,4,2\n"
        ),
        encoding="utf-8",
    )


def _make_pub_processed_case(root: Path, patient_id: str = "case_001") -> None:
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / f"{patient_id}_img.npy", np.zeros((3, 2, 2, 2), dtype=np.float32))
    np.save(root / f"{patient_id}_lab.npy", np.zeros((2, 2, 2), dtype=np.uint8))
    np.save(root / f"{patient_id}_zone.npy", np.ones((2, 2, 2), dtype=np.uint8))


def test_pub_preflight_accepts_a_complete_layout_without_reading_images(tmp_path):
    pub_root = tmp_path / "pub_raw"
    _make_complete_pub_raw_layout(pub_root)

    summary = validate_pub_inputs(pub_root)

    assert summary == "PUB: 1/1 discovered cases are complete"


def test_promis_preflight_accepts_complete_mri_and_biopsy_layout(tmp_path):
    mri_root = tmp_path / "promis_mri"
    biopsy_root = tmp_path / "promis_biopsy"
    _make_promis_mri_case(mri_root)
    _make_promis_biopsy_layout(biopsy_root)

    summary = validate_promis_inputs(mri_root, biopsy_root)

    assert summary == "PROMIS: 1/1 MRI cases are complete; 1 have matching biopsy CSVs"


def test_promis_preflight_fails_clearly_when_zone_mask_is_missing(tmp_path):
    mri_root = tmp_path / "promis_mri"
    biopsy_root = tmp_path / "promis_biopsy"
    _make_promis_mri_case(mri_root, include_zone_mask=False)
    _make_promis_biopsy_layout(biopsy_root)

    with pytest.raises(
        PipelineError,
        match=r"PROMIS cases are missing gland_zone_20level_set1\.nii\.gz",
    ):
        validate_promis_inputs(mri_root, biopsy_root)


def test_promis_preflight_rejects_whitespace_mangled_column_names(tmp_path):
    mri_root = tmp_path / "promis_mri"
    biopsy_root = tmp_path / "promis_biopsy"
    _make_promis_mri_case(mri_root)
    _make_promis_biopsy_layout(biopsy_root)
    csv_path = biopsy_root / "P-0001.csv"
    csv_path.write_text(
        csv_path.read_text(encoding="utf-8").replace("zone_id", " zone_id"),
        encoding="utf-8",
    )

    with pytest.raises(PipelineError, match="Missing .*zone_id"):
        validate_promis_inputs(mri_root, biopsy_root)


def test_nested_download_wrappers_are_resolved_when_unambiguous(tmp_path):
    mri_root = tmp_path / "promis_archive"
    biopsy_root = tmp_path / "biopsy_archive"
    nested_mri = mri_root / "release" / "MRI"
    nested_biopsy = biopsy_root / "Template_biopsy" / "Template_biopsy"
    _make_promis_mri_case(nested_mri)
    _make_promis_biopsy_layout(nested_biopsy)

    args = parse_args(
        [
            "--workspace",
            str(tmp_path / "workspace"),
            "--datasets",
            "promis",
            "--promis-mri-root",
            str(mri_root),
            "--promis-biopsy-root",
            str(biopsy_root),
        ]
    )
    config = build_config(args)

    assert config.promis_mri_root == nested_mri
    assert config.promis_biopsy_root == nested_biopsy


def test_preprocess_and_qa_require_unify_between_them(tmp_path):
    args = parse_args(
        [
            "--workspace",
            str(tmp_path / "workspace"),
            "--datasets",
            "pub",
            "--stages",
            "preprocess",
            "qa",
        ]
    )

    with pytest.raises(PipelineError, match="add unify"):
        build_config(args)


def test_workspace_must_not_overlap_raw_inputs(tmp_path):
    pub_root = tmp_path / "pub_raw"
    _make_complete_pub_raw_layout(pub_root)
    args = parse_args(
        [
            "--workspace",
            str(pub_root / "derived"),
            "--datasets",
            "pub",
            "--pub-root",
            str(pub_root),
        ]
    )

    with pytest.raises(PipelineError, match="must not contain, equal, or sit inside"):
        build_config(args)


def test_existing_output_must_be_a_directory_even_when_reuse_is_allowed(tmp_path):
    args = parse_args(
        [
            "--workspace",
            str(tmp_path / "workspace"),
            "--datasets",
            "pub",
            "--stages",
            "preprocess",
            "--allow-existing-output",
        ]
    )
    config = build_config(args)
    config.pub_processed.parent.mkdir(parents=True)
    config.pub_processed.write_text("not a directory", encoding="utf-8")

    with pytest.raises(PipelineError, match="is not a directory"):
        validate_output_state(config)


def test_unified_output_must_be_fresh_even_when_existing_output_is_allowed(tmp_path):
    args = parse_args(
        [
            "--workspace",
            str(tmp_path / "workspace"),
            "--datasets",
            "pub",
            "--stages",
            "unify",
            "--allow-existing-output",
        ]
    )
    config = build_config(args)
    stale_split = config.unified_root / "splits" / "B1_TCIA_TBx_baseline_train.csv"
    stale_split.parent.mkdir(parents=True)
    stale_split.write_text("patient_id\nold_case\n", encoding="utf-8")

    with pytest.raises(PipelineError, match="must be fresh"):
        validate_output_state(config)


def test_stage_errors_include_partial_output_context():
    def fail_after_write():
        raise PipelineError("no complete processed cases")

    with pytest.raises(
        PipelineError,
        match="PROMIS preprocessing failed.*partial derived outputs",
    ):
        _execute_stage("PROMIS preprocessing", fail_after_write)


@pytest.mark.parametrize("split_mode", ["none", "auto"])
def test_pub_only_unification_writes_registry_without_experiment_splits(
    tmp_path,
    split_mode,
):
    pub_processed = tmp_path / "processed" / "pub"
    unified_root = tmp_path / f"unified_{split_mode}"
    _make_pub_processed_case(pub_processed)

    result = create_unified_dataset(
        tmp_path,
        pub_dir=pub_processed,
        promis_dir=tmp_path / "missing_promis",
        tcia_dir=tmp_path / "missing_tcia",
        output_dir=unified_root,
        split_mode=split_mode,
    )

    registry_path = unified_root / "splits" / "dataset_registry.csv"
    assert Path(result) == unified_root
    assert registry_path.is_file()

    registry = pd.read_csv(registry_path)
    assert registry.to_dict("records") == [
        {
            "patient_id": "PUB_case_001",
            "source": "PUB",
            "has_target": 0,
            "has_sys_12": 0,
            "has_sys_20": 0,
            "has_lesion": 1,
            "has_gland": 1,
        }
    ]
    assert (unified_root / "PUB_case_001" / "input_tensor.npy").is_file()
    assert (unified_root / "PUB_case_001" / "lesion_mask.npy").is_file()
    assert (unified_root / "PUB_case_001" / "gland_mask.npy").is_file()
    assert not (unified_root / "splits" / "split_summary.csv").exists()


def test_auto_split_skips_duplicate_joint_tcia_studies(tmp_path):
    tcia_processed = tmp_path / "processed" / "tcia"
    for uid in ("1.2.3.4", "1.2.3.5"):
        case_dir = tcia_processed / f"Prostate-MRI-US-Biopsy-0001_{uid}"
        case_dir.mkdir(parents=True)
        np.save(case_dir / "input_tensor.npy", np.zeros((3, 2, 2, 2)))
        np.save(case_dir / "systematic_labels.npy", np.ones(12, dtype=np.int64))
        (case_dir / "target_mask.nii.gz").touch()
        (case_dir / "zones_mask.nii.gz").touch()

    unified_root = tmp_path / "unified_duplicate_tcia"
    result = create_unified_dataset(
        tmp_path,
        pub_dir=tmp_path / "missing_pub",
        promis_dir=tmp_path / "missing_promis",
        tcia_dir=tcia_processed,
        output_dir=unified_root,
        split_mode="auto",
    )

    assert Path(result) == unified_root
    registry = pd.read_csv(unified_root / "splits" / "dataset_registry.csv")
    assert len(registry) == 2
    assert not (unified_root / "splits" / "B1_TCIA_TBx_baseline_train.csv").exists()
