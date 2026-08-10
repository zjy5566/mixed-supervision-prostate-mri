import os
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    from Dataset_split_generator import create_split_csvs, RANDOM_STATE
except ImportError:
    from preprocessing.Dataset_split_generator import create_split_csvs, RANDOM_STATE


def first_existing_dir(*paths):
    for path in paths:
        if path and os.path.isdir(path):
            return path
    return None


def first_existing_file(*paths):
    """Return the first existing file among current and legacy mask names."""
    for path in paths:
        if path and os.path.isfile(path):
            return path
    return None


def copy_sbx_labels_preserve_project_convention(src_path, dst_path):
    """Copy SBx labels without changing the current project label convention.

    Current convention:
      -1 = invalid / unsampled / no supervision
       0 = background / old negative placeholder
       1 = benign / negative biopsy
       2-6 = shifted ISUP labels
    """
    labels = np.load(src_path).astype(np.int64)
    unexpected = sorted(set(labels.tolist()) - set(range(-1, 7)))
    if unexpected:
        print(f"[Warning] Unexpected SBx labels in {src_path}: {unexpected}")
    np.save(dst_path, labels)

def create_unified_dataset(
    base_dir,
    *,
    pub_dir=None,
    promis_dir=None,
    promis_labels_dir=None,
    tcia_dir=None,
    output_dir=None,
    split_mode="auto",
):
    """Create the unified dataset from explicit or legacy processed paths.

    The explicit path arguments are used by the public preprocessing pipeline
    and make the integration step independent of the original workstation
    layout. Omitting them preserves the legacy ``RP_DATASET_ROOT`` layout.

    ``split_mode`` controls experiment CSV generation:
      - ``auto`` writes experiment splits only when at least one TCIA case has
        both TBx and SBx supervision;
      - ``experiment`` requires that cohort and propagates validation errors;
      - ``none`` writes only ``dataset_registry.csv``.

    A registry can therefore be built for a single downloaded dataset without
    pretending that it is sufficient to reproduce the mixed-supervision
    experiments.
    """
    if split_mode not in {"auto", "experiment", "none"}:
        raise ValueError(
            "split_mode must be one of 'auto', 'experiment', or 'none'; "
            f"got {split_mode!r}"
        )
    base_dir = os.path.abspath(base_dir)
    src_pub = (
        os.path.abspath(pub_dir)
        if pub_dir
        else first_existing_dir(
            os.path.join(base_dir, 'Dataset_prostate_MRI', 'Dataset_prostate_MRI_dwi'),
            os.path.join(base_dir, 'Dataset_prostate_MRI', 'Dataset_prostate_MRI'),
        )
    )
    src_promis = (
        os.path.abspath(promis_dir)
        if promis_dir
        else first_existing_dir(
            os.path.join(base_dir, 'derived PROMIS data set', 'Processed_PROMIS_dwi'),
        )
    )
    if promis_labels_dir:
        src_promis_labels = os.path.abspath(promis_labels_dir)
    elif promis_dir:
        src_promis_labels = src_promis
    else:
        src_promis_labels = first_existing_dir(
            os.path.join(base_dir, 'derived PROMIS data set', 'Processed_PROMIS'),
        )
    src_tcia = (
        os.path.abspath(tcia_dir)
        if tcia_dir
        else os.path.join(base_dir, 'Target biosy', 'Processed_TCIA')
    )

    dst_root = os.path.abspath(output_dir or os.path.join(base_dir, 'Unified_Dataset'))
    os.makedirs(dst_root, exist_ok=True)

    registry_data = []
    joint_tcia_groups = []
    print("\nDetected dataset layout:")
    print(f"  PUB processed        : {src_pub or 'MISSING'}")
    print(f"  TCIA processed       : {src_tcia if os.path.isdir(src_tcia) else 'MISSING'}")
    print(f"  PROMIS MRI+labels    : {src_promis or 'MISSING'}")
    print(f"  PROMIS labels-only   : {src_promis_labels or 'MISSING'}")
    print(f"  Unified destination  : {dst_root}")

    # --- 2. 整合 TCIA 靶向活检集 ---
    print("Processing TCIA Target Biopsy Dataset...")
    if os.path.exists(src_tcia):
        tcia_search_dir = src_tcia
        # 兼容你的嵌套结构：Target biosy\Processed_TCIA\Processed_PROMIS
        if os.path.exists(os.path.join(src_tcia, 'Processed_PROMIS')):
            tcia_search_dir = os.path.join(src_tcia, 'Processed_PROMIS')

        tcia_patients = sorted(
            d for d in os.listdir(tcia_search_dir)
            if d.startswith('Prostate-MRI-US-Biopsy-')
        )

        for pid in tqdm(tcia_patients):
            src_p_dir = os.path.join(tcia_search_dir, pid)

            # 必须包含 input_tensor
            if not os.path.exists(os.path.join(src_p_dir, 'input_tensor.npy')):
                continue

            # Prefer canonical names, but accept the older *_crop exports so
            # regenerated datasets and historical preprocessing runs both work.
            target_path = first_existing_file(
                os.path.join(src_p_dir, 'target_mask.nii.gz'),
                os.path.join(src_p_dir, 'target_mask_crop.nii.gz'),
            )
            sys_mask_path = first_existing_file(
                os.path.join(src_p_dir, 'zones_mask.nii.gz'),
                os.path.join(src_p_dir, 'zones_mask_crop.nii.gz'),
            )
            sys_label_path = os.path.join(src_p_dir, 'systematic_labels.npy')
            gland_path = first_existing_file(
                os.path.join(src_p_dir, 'gland_mask.nii.gz'),
                os.path.join(src_p_dir, 'gland_mask_crop.nii.gz'),
            )

            has_target = 1 if target_path else 0
            has_sys_12 = 1 if (sys_mask_path and os.path.exists(sys_label_path)) else 0
            has_gland = 1 if gland_path else 0

            # 如果既没有靶向也没有系统活检，这个病人就是无用数据，跳过
            if has_target == 0 and has_sys_12 == 0:
                continue

            new_pid = f"TCIA_{pid.split('-')[-1]}"
            dst_p_dir = os.path.join(dst_root, new_pid)
            os.makedirs(dst_p_dir, exist_ok=True)

            # 拷贝并统一命名
            shutil.copy2(os.path.join(src_p_dir, 'input_tensor.npy'), os.path.join(dst_p_dir, 'input_tensor.npy'))

            if has_sys_12:
                shutil.copy2(sys_mask_path, os.path.join(dst_p_dir, 'zones_mask.nii.gz'))
                copy_sbx_labels_preserve_project_convention(sys_label_path, os.path.join(dst_p_dir, 'systematic_labels_12.npy'))

            if has_target:
                shutil.copy2(target_path, os.path.join(dst_p_dir, 'target_mask.nii.gz'))

            if has_gland:
                shutil.copy2(gland_path, os.path.join(dst_p_dir, 'gland_mask.nii.gz'))

            registry_data.append({
                'patient_id': new_pid, 'source': 'TCIA',
                'has_target': has_target, 'has_sys_12': has_sys_12, 'has_sys_20': 0, 'has_lesion': 0, 'has_gland': has_gland
            })
            if has_target and has_sys_12:
                joint_tcia_groups.append(pid.split('_', 1)[0])

    # --- 3. 整合 PROMIS 数据集 ---
    print("\nProcessing PROMIS Dataset...")
    if src_promis and os.path.exists(src_promis):
        promis_patients = sorted(
            d for d in os.listdir(src_promis) if d.startswith('P-')
        )

        for pid in tqdm(promis_patients):
            src_p_dir = os.path.join(src_promis, pid)

            sys_label_path = first_existing_file(
                os.path.join(src_p_dir, 'systematic_labels.npy'),
                os.path.join(src_promis_labels, pid, 'systematic_labels.npy') if src_promis_labels else None,
            )
            req_files = ['input_tensor.npy', 'zones_mask.nii.gz']
            if not all(os.path.exists(os.path.join(src_p_dir, f)) for f in req_files) or not sys_label_path:
                continue

            has_gland = 1 if os.path.exists(os.path.join(src_p_dir, 'gland_mask.nii.gz')) else 0

            new_pid = f"PROMIS_{pid}"
            dst_p_dir = os.path.join(dst_root, new_pid)
            os.makedirs(dst_p_dir, exist_ok=True)

            shutil.copy2(os.path.join(src_p_dir, 'input_tensor.npy'), os.path.join(dst_p_dir, 'input_tensor.npy'))
            shutil.copy2(os.path.join(src_p_dir, 'zones_mask.nii.gz'), os.path.join(dst_p_dir, 'zones_mask.nii.gz'))
            copy_sbx_labels_preserve_project_convention(
                sys_label_path,
                os.path.join(dst_p_dir, 'systematic_labels_20.npy')
            )

            if has_gland:
                shutil.copy2(os.path.join(src_p_dir, 'gland_mask.nii.gz'), os.path.join(dst_p_dir, 'gland_mask.nii.gz'))

            registry_data.append({
                'patient_id': new_pid, 'source': 'PROMIS',
                'has_target': 0, 'has_sys_12': 0, 'has_sys_20': 1, 'has_lesion': 0, 'has_gland': has_gland
            })

    # --- 4. 整合 公开 MRI 数据集 (PUB) ---
    print("\nProcessing PUB Radiologist Annotation Dataset...")
    if src_pub and os.path.exists(src_pub):
        pub_files = os.listdir(src_pub)
        image_suffix = '_img.npy'
        pub_ids = sorted({
            filename[:-len(image_suffix)]
            for filename in pub_files
            if filename.endswith(image_suffix)
        })

        for pid in tqdm(pub_ids):
            src_img = os.path.join(src_pub, f"{pid}_img.npy")
            src_lab = os.path.join(src_pub, f"{pid}_lab.npy") # Lesion 病灶
            src_zone = os.path.join(src_pub, f"{pid}_zone.npy") # Gland 腺体区域

            if not all(os.path.exists(f) for f in [src_img, src_lab, src_zone]):
                continue

            new_pid = f"PUB_{pid}"
            dst_p_dir = os.path.join(dst_root, new_pid)
            os.makedirs(dst_p_dir, exist_ok=True)

            # 拷贝并规范化命名，抹平模态差异
            shutil.copy2(src_img, os.path.join(dst_p_dir, 'input_tensor.npy'))
            shutil.copy2(src_lab, os.path.join(dst_p_dir, 'lesion_mask.npy'))
            shutil.copy2(src_zone, os.path.join(dst_p_dir, 'gland_mask.npy'))

            registry_data.append({
                'patient_id': new_pid, 'source': 'PUB',
                'has_target': 0, 'has_sys_12': 0, 'has_sys_20': 0, 'has_lesion': 1, 'has_gland': 1
            })

    # --- 5. Generate the registry and, when requested, experiment split CSVs ---
    print("\nGenerating the dataset registry...")
    if len(registry_data) == 0:
        print("Error: No valid data found! Please check your source directories.")
        return None

    df = pd.DataFrame(registry_data)

    splits_dir = os.path.join(dst_root, 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    registry_csv = os.path.join(splits_dir, 'dataset_registry.csv')
    df.to_csv(registry_csv, index=False)
    tcia_joint = df[
        (df['source'] == 'TCIA')
        & (df['has_target'] == 1)
        & ((df['has_sys_12'] == 1) | (df['has_sys_20'] == 1))
    ]
    duplicate_joint_groups = sorted({
        group_id
        for group_id in joint_tcia_groups
        if joint_tcia_groups.count(group_id) > 1
    })
    if duplicate_joint_groups and split_mode == "experiment":
        raise ValueError(
            "Experiment splitting requires one derived MRI study per original "
            "TCIA patient. Resolve duplicate study cases before splitting: "
            f"{duplicate_joint_groups[:5]}"
        )
    should_generate_splits = split_mode == "experiment" or (
        split_mode == "auto"
        and len(tcia_joint) > 0
        and not duplicate_joint_groups
    )

    if should_generate_splits:
        print(
            "Generating internal validation/test and PROMIS "
            "external-validation experiment splits..."
        )
        create_split_csvs(
            registry_csv=registry_csv,
            splits_dir=splits_dir,
            external_source="PROMIS",
            val_size=0.2,
            internal_test_size=0.1,
            random_state=RANDOM_STATE,
        )
    elif split_mode == "auto":
        if duplicate_joint_groups:
            print(
                "Experiment splits skipped: multiple derived joint-TCIA "
                "studies map to the same original patient, so row-level "
                "splitting would risk patient leakage."
            )
        else:
            print(
                "Experiment splits skipped: the registry has no TCIA case with "
                "both TBx and SBx supervision. The registry is still usable for "
                "inspection and custom downstream splitting."
            )
    else:
        print("Experiment splits skipped because split_mode='none'.")

    print(f"\nDone! Total Patients: {len(df)}")
    print(df['source'].value_counts().to_string())
    print(f"Unified dataset successfully created at: {dst_root}")
    return dst_root

if __name__ == "__main__":
    BASE_DIR = os.environ.get("RP_DATASET_ROOT")
    if not BASE_DIR:
        raise SystemExit("Set RP_DATASET_ROOT before preparing the dataset.")
    create_unified_dataset(BASE_DIR)
