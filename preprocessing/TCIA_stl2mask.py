import os
import vtk
from vtk.util import numpy_support
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd
import glob
import re

# ==========================================
# 核心转换引擎 (已修复 VTK 内存泄漏与挖空逻辑)
# ==========================================
def stl_to_numpy_mask(stl_path, template_img):
    try:
        reader = vtk.vtkSTLReader()
        reader.SetFileName(stl_path)
        reader.Update()
        polydata = reader.GetOutput()

        dims = template_img.GetSize()
        spacing = template_img.GetSpacing()

        # 1. 创建底图
        white_image = vtk.vtkImageData()
        white_image.SetDimensions(dims)
        white_image.SetSpacing(spacing)
        white_image.SetOrigin(0, 0, 0)
        white_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        # 强制将底图的所有像素初始化为 1 (前景)
        scalars = white_image.GetPointData().GetScalars()
        scalars.Fill(1)

        # 2. 将 STL 表面转换为 3D 模板 (Stencil)
        pol2stenc = vtk.vtkPolyDataToImageStencil()
        pol2stenc.SetInputData(polydata)
        pol2stenc.SetOutputOrigin(template_img.GetOrigin())
        pol2stenc.SetOutputSpacing(template_img.GetSpacing())
        pol2stenc.SetOutputWholeExtent(0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1)
        pol2stenc.Update()

        # 3. 执行模板切割
        imgstenc = vtk.vtkImageStencil()
        imgstenc.SetInputData(white_image)
        imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())

        # 关闭反转，外部强制设为 BackgroundValue(0)
        imgstenc.ReverseStencilOff()
        imgstenc.SetBackgroundValue(0)
        imgstenc.Update()

        # 4. 提取为 Numpy 数组并重塑维度 (Z, Y, X)
        vtk_data = imgstenc.GetOutput().GetPointData().GetScalars()
        numpy_mask = numpy_support.vtk_to_numpy(vtk_data).reshape(dims[::-1])

        return numpy_mask
    except Exception as e:
        print(f"Error converting {os.path.basename(stl_path)}: {e}")
        return None

def save_numpy_to_nifti(numpy_array, template_img, output_path):
    final_mask = sitk.GetImageFromArray(numpy_array.astype(np.uint8))
    final_mask.CopyInformation(template_img)
    sitk.WriteImage(final_mask, output_path)

# ==========================================
# 标签映射字典
# ==========================================
def get_isup_label(primary, secondary):
    """
    Map biopsy Gleason pattern to the project label convention:
      1 = benign / no cancer, 2 = ISUP1, 3 = ISUP2, ..., 6 = ISUP5.
    """
    if pd.isna(primary) or pd.isna(secondary):
        return 1

    try:
        p, s = int(primary), int(secondary)
    except (TypeError, ValueError):
        return 1

    if p + s <= 6:
        return 2
    if p + s == 7:
        return 3 if p == 3 else 4
    if p + s == 8:
        return 5
    if p + s >= 9:
        return 6
    return 1

def map_ucla_to_isup(ucla_score):
    """
    UCLA Score 映射规则：0/1/2=1, 3=2, 4=3, 5=4, 6=5, 7=6
    """
    try:
        score = int(ucla_score)
        if score in [0, 1, 2]: return 1
        elif score == 3: return 2
        elif score == 4: return 3
        elif score == 5: return 4
        elif score == 6: return 5
        elif score == 7: return 6
        else: return 1
    except:
        return 1

# ==========================================
# 批量处理主函数
# ==========================================
def read_table(path):
    if path is None or not os.path.exists(path):
        return None
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)

def first_existing_column(df, candidates):
    if df is None:
        return None
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    return None

def normalize_target_no(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\.0$", "", text)
    text = re.sub(r"(?i)^target\s*", "", text)
    match = re.search(r"\d+", text)
    return match.group(0) if match else text

def create_needle_mask(template_img, tip_phys, base_phys, radius=2, num_points=200):
    """Create a temporary biopsy-core track mask for matching cores to target ROIs."""
    img_shape = sitk.GetArrayViewFromImage(template_img).shape
    depth, height, width = img_shape
    mask_arr = np.zeros(img_shape, dtype=np.uint8)

    try:
        tip_idx = template_img.TransformPhysicalPointToIndex([float(v) for v in tip_phys])
        base_idx = template_img.TransformPhysicalPointToIndex([float(v) for v in base_phys])
    except Exception:
        return None

    xs = np.linspace(tip_idx[0], base_idx[0], num_points)
    ys = np.linspace(tip_idx[1], base_idx[1], num_points)
    zs = np.linspace(tip_idx[2], base_idx[2], num_points)

    for x, y, z in zip(xs, ys, zs):
        ix, iy, iz = int(round(x)), int(round(y)), int(round(z))
        if 0 <= iz < depth and 0 <= iy < height and 0 <= ix < width:
            mask_arr[iz, iy, ix] = 1

    mask_sitk = sitk.GetImageFromArray(mask_arr)
    mask_sitk.CopyInformation(template_img)
    dilated = sitk.BinaryDilate(mask_sitk > 0, [radius, radius, radius], sitk.sitkBall)
    return sitk.GetArrayFromImage(dilated).astype(bool)

def ucla_fallback_labels(target_df):
    """Read target metadata labels for fallback only."""
    labels = {}
    if target_df is None:
        return labels

    patient_col = first_existing_column(target_df, ["Patient ID", "Patient Number", "PatientID"])
    target_col = first_existing_column(target_df, ["Target No.", "Target No", "Target Number", "Target"])
    ucla_col = first_existing_column(
        target_df,
        ["UCLA Score (Similar to PIRADS v2)", "UCLA Score", "PIRADS", "PI-RADS"],
    )
    if patient_col is None or target_col is None or ucla_col is None:
        return labels

    for _, row in target_df.iterrows():
        pid = str(row[patient_col]).strip()
        target_no = normalize_target_no(row[target_col])
        if not pid or not target_no:
            continue
        labels.setdefault(pid, {})[target_no] = map_ucla_to_isup(row[ucla_col])
    return labels

def biopsy_verified_labels_by_spatial_overlap(
    biopsy_df,
    target_masks,
    template_img,
    base_pid,
    uid_suffix,
    min_overlap_voxels=1,
    needle_radius=2,
):
    """
    Assign target-biopsy cores to target lesions by needle-track/mask overlap.

    For each target biopsy core from the same patient and MRI, a needle mask is
    generated from Bx Tip/Base MRI coordinates. The core is assigned to the
    target mask with the largest non-zero overlap. Multiple cores assigned to
    one target are reduced by max ISUP.
    """
    stats = {
        "matched_cores": 0,
        "unmatched_cores": 0,
        "ambiguous_cores": 0,
        "missing_coord_cores": 0,
    }
    if biopsy_df is None or not target_masks:
        return {}, stats

    patient_col = first_existing_column(biopsy_df, ["Patient Number", "Patient ID", "PatientID"])
    mr_col = first_existing_column(biopsy_df, ["Series Instance UID (MRI)", "seriesInstanceUID_MR"])
    core_col = first_existing_column(biopsy_df, ["Core Label", "Core", "Label"])
    primary_col = first_existing_column(biopsy_df, ["Primary Gleason", "Primary"])
    secondary_col = first_existing_column(biopsy_df, ["Secondary Gleason", "Secondary"])
    coord_cols = [
        "Bx Tip X (MRI Coord)",
        "Bx Tip Y (MRI Coord)",
        "Bx Tip Z (MRI Coord)",
        "Bx Base X (MRI Coord)",
        "Bx Base Y (MRI Coord)",
        "Bx Base Z (MRI Coord)",
    ]
    coord_col_map = {c: first_existing_column(biopsy_df, [c]) for c in coord_cols}
    required = [patient_col, mr_col, core_col, primary_col, secondary_col] + list(coord_col_map.values())
    if any(col is None for col in required):
        return {}, stats

    rows = biopsy_df[
        biopsy_df[patient_col].astype(str).str.strip() == str(base_pid).strip()
    ].copy()
    if uid_suffix:
        rows = rows[
            rows[mr_col].astype(str).str.strip().str.endswith(str(uid_suffix))
        ].copy()
    rows = rows[
        rows[core_col].astype(str).str.upper().str.contains("TARGET", na=False)
    ].copy()

    labels = {}
    for _, row in rows.iterrows():
        if any(pd.isna(row[coord_col_map[col]]) for col in coord_cols):
            stats["missing_coord_cores"] += 1
            continue

        tip_phys = [
            row[coord_col_map["Bx Tip X (MRI Coord)"]],
            row[coord_col_map["Bx Tip Y (MRI Coord)"]],
            row[coord_col_map["Bx Tip Z (MRI Coord)"]],
        ]
        base_phys = [
            row[coord_col_map["Bx Base X (MRI Coord)"]],
            row[coord_col_map["Bx Base Y (MRI Coord)"]],
            row[coord_col_map["Bx Base Z (MRI Coord)"]],
        ]
        needle_mask = create_needle_mask(
            template_img,
            tip_phys,
            base_phys,
            radius=needle_radius,
        )
        if needle_mask is None or not needle_mask.any():
            stats["missing_coord_cores"] += 1
            continue

        overlaps = []
        for target_no, target_mask in target_masks.items():
            overlap = int(np.logical_and(needle_mask, target_mask > 0).sum())
            overlaps.append((target_no, overlap))

        overlaps.sort(key=lambda item: item[1], reverse=True)
        best_target, best_overlap = overlaps[0]
        second_overlap = overlaps[1][1] if len(overlaps) > 1 else -1

        if best_overlap < min_overlap_voxels:
            stats["unmatched_cores"] += 1
            continue
        if best_overlap == second_overlap:
            stats["ambiguous_cores"] += 1
            continue

        isup_label = get_isup_label(row[primary_col], row[secondary_col])
        labels[best_target] = max(labels.get(best_target, 1), isup_label)
        stats["matched_cores"] += 1

    return labels, stats

def batch_convert_stls(processed_dir_root, target_excel_path, biopsy_excel_path=None, use_ucla_fallback=False):
    print("Loading target metadata...")
    try:
        target_df = read_table(target_excel_path)
        biopsy_df = read_table(biopsy_excel_path)
    except Exception as e:
        print(f"Failed to read clinical files: {e}")
        return

    fallback_labels = ucla_fallback_labels(target_df)
    if biopsy_df is None and use_ucla_fallback:
        print("Warning: biopsy Excel was not provided. Falling back to UCLA/PIRADS-like labels.")
    elif biopsy_df is None:
        print("Warning: biopsy Excel was not provided. TBx-labelled target masks will be skipped.")
    else:
        print("Target labels will use biopsy-verified Gleason/ISUP when available.")

    folders = [d for d in os.listdir(processed_dir_root) if os.path.isdir(os.path.join(processed_dir_root, d))]

    gland_success = 0
    target_binary_success = 0
    target_success = 0
    target_biopsy_level = 0
    target_ucla_fallback = 0
    target_unmatched = 0
    target_matched_cores = 0
    target_unmatched_cores = 0
    target_ambiguous_cores = 0
    target_missing_coord_cores = 0

    for folder in tqdm(folders, desc="Converting STLs to Masks"):
        folder_path = os.path.join(processed_dir_root, folder)
        mri_template_path = os.path.join(folder_path, 't2.nii.gz')

        if not os.path.exists(mri_template_path):
            continue

        template_img = sitk.ReadImage(mri_template_path)
        if "_" in folder:
            base_pid, uid_suffix = folder.split("_", 1)
        else:
            base_pid, uid_suffix = folder, ""

        # ---------------------------------------------------------
        # 任务 A: 转换 Gland Mask (移除覆盖保护)
        # ---------------------------------------------------------
        stl_path_gland = os.path.join(folder_path, 'prostate_surface.stl')
        output_gland_path = os.path.join(folder_path, 'gland_mask.nii.gz')

        if os.path.exists(stl_path_gland):
            gland_np = stl_to_numpy_mask(stl_path_gland, template_img)
            if gland_np is not None:
                save_numpy_to_nifti(gland_np, template_img, output_gland_path)
                gland_success += 1

        # ---------------------------------------------------------
        # 任务 B: 转换 Target Masks 并合并 (移除覆盖保护)
        # ---------------------------------------------------------
        output_target_path = os.path.join(folder_path, 'target_mask.nii.gz')
        output_target_binary_path = os.path.join(folder_path, 'target_mask_binary.nii.gz')

        target_stl_files = glob.glob(os.path.join(folder_path, 'target_*.stl'))

        if len(target_stl_files) > 0:
            dims = template_img.GetSize()
            combined_target_mask = np.zeros(dims[::-1], dtype=np.uint8)
            combined_target_binary = np.zeros(dims[::-1], dtype=np.uint8)
            target_masks = {}
            found_any = False

            for t_file in target_stl_files:
                t_num = os.path.basename(t_file).replace('target_', '').replace('.stl', '')
                t_num = normalize_target_no(t_num)

                target_np = stl_to_numpy_mask(t_file, template_img)
                if target_np is not None:
                    target_masks[t_num] = target_np
                    combined_target_binary = np.maximum(
                        combined_target_binary,
                        (target_np > 0).astype(np.uint8),
                    )

            if len(target_masks) > 0:
                save_numpy_to_nifti(combined_target_binary, template_img, output_target_binary_path)
                target_binary_success += 1

            verified_labels, match_stats = biopsy_verified_labels_by_spatial_overlap(
                biopsy_df,
                target_masks,
                template_img,
                base_pid,
                uid_suffix,
            )
            target_matched_cores += int(match_stats.get("matched_cores", 0))
            target_unmatched_cores += int(match_stats.get("unmatched_cores", 0))
            target_ambiguous_cores += int(match_stats.get("ambiguous_cores", 0))
            target_missing_coord_cores += int(match_stats.get("missing_coord_cores", 0))
            fallback_patient_labels = fallback_labels.get(base_pid, {})

            for t_num, target_np in target_masks.items():
                if t_num in verified_labels:
                    matched_label = verified_labels[t_num]
                    target_biopsy_level += 1
                elif use_ucla_fallback and t_num in fallback_patient_labels:
                    matched_label = fallback_patient_labels[t_num]
                    target_ucla_fallback += 1
                else:
                    target_unmatched += 1
                    continue

                labeled_target = target_np * matched_label
                combined_target_mask = np.maximum(combined_target_mask, labeled_target)
                found_any = True

            if found_any:
                save_numpy_to_nifti(combined_target_mask, template_img, output_target_path)
                target_success += 1

    print(f"\n=============================================")
    print(f"Summary:")
    print(f" - Gland Masks Generated:  {gland_success}")
    print(f" - Binary Target Masks Generated: {target_binary_success}")
    print(f" - Biopsy-Labeled Target Masks Generated: {target_success}")
    print(f" - Target labels from biopsy spatial overlap: {target_biopsy_level}")
    print(f" - Target biopsy cores matched by overlap: {target_matched_cores}")
    print(f" - Target biopsy cores with no overlap: {target_unmatched_cores}")
    print(f" - Target biopsy cores with tied overlap: {target_ambiguous_cores}")
    print(f" - Target biopsy cores missing MRI coords: {target_missing_coord_cores}")
    print(f" - Target labels from UCLA fallback: {target_ucla_fallback}")
    print(f" - Target labels unmatched/skipped: {target_unmatched}")
    print(f"=============================================")


if __name__ == "__main__":
    DATASET_ROOT = os.environ.get("RP_DATASET_ROOT")
    if not DATASET_ROOT:
        raise SystemExit("Set RP_DATASET_ROOT before converting STL masks.")
    EXTRACTED_ROOT = os.path.join(DATASET_ROOT, 'Target biosy', 'Extracted_Target_Biopsy')
    TARGET_EXCEL_PATH = os.path.join(
        DATASET_ROOT,
        'Target biosy',
        'unprocessed_data',
        'Target-Data_2019-12-05-2.xlsx',
    )
    BIOPSY_EXCEL_PATH = os.path.join(
        DATASET_ROOT,
        'Target biosy',
        'unprocessed_data',
        'TCIA-Biopsy-Data_2020-07-14.xlsx',
    )

    batch_convert_stls(EXTRACTED_ROOT, TARGET_EXCEL_PATH, BIOPSY_EXCEL_PATH)
