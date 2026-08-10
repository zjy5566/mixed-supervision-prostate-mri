"""Convert PROMIS template-biopsy CSVs to SBx labels and zone label masks.

Input conditions:
  - `input_folder` contains `P-*.csv` template-biopsy files with `zone_id`,
    `samtaken`, `zprescancer`, Gleason, and/or `maxccisup` columns.
  - `output_folder/<patient_id>/zones_mask.nii.gz` already exists from
    PROMIS MRI preprocessing. This anatomical zone-id mask is preserved.

Outputs per patient:
  - `systematic_labels.npy`: 20-zone vector using the project convention:
    -1 = invalid/unsampled/no supervision,
    0 = background/old negative placeholder,
    1 = sampled benign, 2 = ISUP1, ..., 6 = ISUP5.
  - `zone_label_mask.nii.gz`: voxel mask filled with the project label of
    each anatomical zone, generated from `zones_mask.nii.gz`.
"""

import os

import numpy as np
import pandas as pd
import SimpleITK as sitk

NUM_PROMIS_ZONES = 20
LABEL_INVALID = -1
LABEL_BACKGROUND = 0
LABEL_BENIGN = 1
LABEL_ISUP_OFFSET = 1


def as_int_or_none(value):
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def project_label_from_isup(isup_grade):
    """Map raw ISUP grade 1-5 to the project labels 2-6."""
    isup = as_int_or_none(isup_grade)
    if isup is None or isup <= 0:
        return LABEL_BENIGN
    return int(np.clip(isup, 1, 5)) + LABEL_ISUP_OFFSET


def get_isup_label(primary, secondary):
    """Map Gleason primary/secondary scores to the project label convention."""
    p = as_int_or_none(primary)
    s = as_int_or_none(secondary)
    if p is None or s is None or p <= 0 or s <= 0:
        return LABEL_BENIGN

    score = p + s
    if score <= 6:
        return 2
    if score == 7:
        return 3 if p == 3 else 4
    if score == 8:
        return 5
    if score >= 9:
        return 6
    return LABEL_BENIGN


def process_promis_sbx_csv(file_path):
    """Convert one PROMIS biopsy CSV into a 20-zone project-label vector."""
    try:
        df = pd.read_csv(file_path)
        if "zone_id" not in df.columns:
            raise ValueError("CSV is missing required column: zone_id")

        df = df[df["zone_id"].between(1, NUM_PROMIS_ZONES)].copy()
        labels_vector = np.full(NUM_PROMIS_ZONES, LABEL_INVALID, dtype=np.int64)

        for _, row in df.iterrows():
            zone_idx = int(row["zone_id"]) - 1
            sampled = as_int_or_none(row.get("samtaken", 1))
            if sampled == 0:
                labels_vector[zone_idx] = LABEL_INVALID
                continue

            has_cancer = as_int_or_none(row.get("zprescancer", 0))
            if has_cancer != 1:
                labels_vector[zone_idx] = LABEL_BENIGN
                continue

            label = get_isup_label(
                row.get("zprimgleason", np.nan),
                row.get("zsecondgleason", np.nan),
            )
            if label == LABEL_BENIGN:
                label = project_label_from_isup(row.get("maxccisup", np.nan))
            if label == LABEL_BENIGN:
                label = 2

            labels_vector[zone_idx] = label

        return labels_vector
    except Exception as exc:
        print(f"Failed to process {file_path}: {exc}")
        return None


def find_zone_mask(patient_folder):
    for filename in ("zones_mask.nii.gz", "gland_zone_20level_set1.nii.gz"):
        path = os.path.join(patient_folder, filename)
        if os.path.exists(path):
            return path
    return None


def save_zone_label_mask(patient_folder, labels):
    zone_mask_path = find_zone_mask(patient_folder)
    if zone_mask_path is None:
        print(
            "Warning: zone mask not found for "
            f"{patient_folder}; skipped zone_label_mask.nii.gz"
        )
        return False

    zone_img = sitk.ReadImage(zone_mask_path)
    zone_arr = sitk.GetArrayFromImage(zone_img).round().astype(np.int64)
    label_arr = np.full_like(zone_arr, LABEL_BACKGROUND, dtype=np.int16)

    for zone_id in range(1, NUM_PROMIS_ZONES + 1):
        label_arr[zone_arr == zone_id] = int(labels[zone_id - 1])

    label_img = sitk.GetImageFromArray(label_arr)
    label_img.CopyInformation(zone_img)
    sitk.WriteImage(label_img, os.path.join(patient_folder, "zone_label_mask.nii.gz"))
    return True


def batch_convert_csv_to_npy(input_dir, output_root):
    """Convert all PROMIS CSVs and create per-patient label masks."""
    if not os.path.exists(output_root):
        print(f"Warning: output root does not exist: {output_root}")
        return

    count = 0
    mask_count = 0
    for filename in os.listdir(input_dir):
        if not (filename.endswith(".csv") and filename.startswith("P-")):
            continue

        file_path = os.path.join(input_dir, filename)
        labels = process_promis_sbx_csv(file_path)
        if labels is None:
            continue

        patient_id = os.path.splitext(filename)[0]
        patient_folder = os.path.join(output_root, patient_id)
        os.makedirs(patient_folder, exist_ok=True)

        np.save(os.path.join(patient_folder, "systematic_labels.npy"), labels)
        if save_zone_label_mask(patient_folder, labels):
            mask_count += 1

        count += 1
        if count % 50 == 0:
            print(f"Processed {count} PROMIS CSV files...")

    print(f"\nDone. Converted {count} PROMIS label CSV files.")
    print(f"Saved zone_label_mask.nii.gz for {mask_count} patients.")
    print(f"Outputs are under: {output_root}")


if __name__ == "__main__":
    DATASET_ROOT = os.environ.get("RP_DATASET_ROOT")
    if not DATASET_ROOT:
        raise SystemExit("Set RP_DATASET_ROOT before preprocessing data.")
    input_folder = os.path.join(
        DATASET_ROOT,
        "derived PROMIS data set",
        "Template_biopsy",
        "Template_biopsy",
    )
    output_folder = os.path.join(
        DATASET_ROOT,
        "derived PROMIS data set",
        "Processed_PROMIS_dwi",
    )

    # Run batch conversion.
    batch_convert_csv_to_npy(input_folder, output_folder)
