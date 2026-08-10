"""Preprocess PROMIS MRI volumes into model-ready tensors and anatomical masks.

Input conditions:
  - Each case folder is named `P-*` and contains `t2.nii.gz`, `adc.nii.gz`,
    `dwi.nii.gz`, `gland.nii.gz`, and `gland_zone_20level_set1.nii.gz`.
    `lesion_a1.nii.gz` is optional and is processed when available.
  - The zone mask stores anatomical systematic-biopsy zone ids: 0 background,
    1-20 PROMIS zones. It is not a label mask and should stay independent from
    `systematic_labels.npy`.

Outputs per patient:
  - `input_tensor.npy`: stacked T2/DWI/ADC tensor in (C, D, H, W).
  - `input_tensor.nii.gz`: vector-image copy for visual QA.
  - `gland_mask.nii.gz`: cropped binary gland mask.
  - `zones_mask.nii.gz`: cropped anatomical zone-id mask for region pooling.
  - `lesion_a1_mask.nii.gz`: cropped radiologist lesion label map, preserving
    lesion instance ids 0, 1, 2, ...
"""

import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

NUM_PROMIS_ZONES = 20
DEFAULT_TARGET_SPACING = (1.0, 1.0, 2.24)
DEFAULT_CROP_SIZE = (64, 64, 32)

# ==========================================
# 1. 图像配准函数 (保持不变)
# ==========================================
def register_images(fixed_image, moving_image, is_label=False):
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.15, 42)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(fixed_image, moving_image)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(final_transform)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(moving_image)


def save_input_tensor_nifti(input_tensor, reference_image, output_path):
    vector_arr = np.moveaxis(input_tensor.astype(np.float32), 0, -1)
    vector_img = sitk.GetImageFromArray(vector_arr, isVector=True)
    vector_img.CopyInformation(reference_image)
    sitk.WriteImage(vector_img, output_path)


def warn_unexpected_zone_ids(zones_img, case_id):
    zone_arr = sitk.GetArrayFromImage(zones_img).round().astype(np.int64)
    zone_ids = set(np.unique(zone_arr).tolist())
    unexpected = sorted(zone_ids - set(range(NUM_PROMIS_ZONES + 1)))
    if unexpected:
        print(f"\n[Warning] Unexpected zone ids in {case_id}: {unexpected}")


def resample_to_spacing(image, target_spacing=DEFAULT_TARGET_SPACING, is_label=False):
    new_size = [
        int(round(image.GetSize()[i] * image.GetSpacing()[i] / target_spacing[i]))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)


def resample_label_to_reference(image, reference):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)


def pad_if_needed(img, target_size=DEFAULT_CROP_SIZE):
    curr_size = img.GetSize()
    pad_lower = [0, 0, 0]
    pad_upper = [0, 0, 0]
    need_pad = False
    for i in range(3):
        if curr_size[i] < target_size[i]:
            diff = target_size[i] - curr_size[i]
            pad_lower[i] = diff // 2
            pad_upper[i] = diff - pad_lower[i]
            need_pad = True
    return sitk.ConstantPad(img, pad_lower, pad_upper, 0.0) if need_pad else img


def crop_lesion_like_promis_case(
    t2,
    gland,
    lesion,
    target_spacing=DEFAULT_TARGET_SPACING,
    crop_size=DEFAULT_CROP_SIZE,
):
    """Fast lesion-only crop using the same gland-centred ROI as full preprocessing."""
    t2_res = resample_to_spacing(t2, target_spacing=target_spacing)
    gland_res = resample_to_spacing(gland, target_spacing=target_spacing, is_label=True)
    lesion_res = resample_label_to_reference(lesion, t2_res)

    gland_res = sitk.BinaryThreshold(
        gland_res,
        lowerThreshold=0.5,
        upperThreshold=1e9,
        insideValue=1,
        outsideValue=0,
    )

    t2_res = pad_if_needed(t2_res, crop_size)
    gland_res = pad_if_needed(gland_res, crop_size)
    lesion_res = pad_if_needed(lesion_res, crop_size)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(gland_res)
    if not stats.HasLabel(1):
        centroid_index = [s // 2 for s in t2_res.GetSize()]
    else:
        centroid_world = stats.GetCentroid(1)
        centroid_index = t2_res.TransformPhysicalPointToIndex(centroid_world)

    img_size = t2_res.GetSize()
    roi_start = [
        max(0, min(int(centroid_index[i] - crop_size[i] // 2), img_size[i] - crop_size[i]))
        for i in range(3)
    ]
    return sitk.RegionOfInterest(lesion_res, crop_size, roi_start)


def save_lesion_label_map(lesion_img, output_path, max_foreground_fraction=0.5):
    lesion_img = sitk.Cast(lesion_img, sitk.sitkUInt16)
    lesion_arr = np.rint(sitk.GetArrayFromImage(lesion_img)).astype(np.uint16)
    foreground_fraction = float(np.mean(lesion_arr > 0))
    if foreground_fraction > max_foreground_fraction:
        return False, foreground_fraction

    out_img = sitk.GetImageFromArray(lesion_arr)
    out_img.CopyInformation(lesion_img)
    sitk.WriteImage(out_img, output_path)
    return True, foreground_fraction


def backfill_lesion_a1_case(case_dir, output_dir, overwrite=False):
    """Add lesion_a1_mask.nii.gz for an already preprocessed PROMIS case."""
    case_id = os.path.basename(case_dir)
    save_path = os.path.join(output_dir, case_id)
    lesion_path = os.path.join(case_dir, 'lesion_a1.nii.gz')
    output_path = os.path.join(save_path, 'lesion_a1_mask.nii.gz')

    if not os.path.exists(lesion_path):
        return "missing_lesion"
    if os.path.exists(output_path) and not overwrite:
        return "exists"

    t2_path = os.path.join(case_dir, 't2.nii.gz')
    gland_path = os.path.join(case_dir, 'gland.nii.gz')
    if not os.path.exists(t2_path) or not os.path.exists(gland_path):
        return "missing_reference"

    t2 = sitk.ReadImage(t2_path)
    gland = sitk.ReadImage(gland_path)
    lesion = sitk.ReadImage(lesion_path)
    lesion_crop = crop_lesion_like_promis_case(t2, gland, lesion)
    saved, foreground_fraction = save_lesion_label_map(lesion_crop, output_path)
    if not saved:
        print(
            f"\n[Warning] Skipped suspicious lesion mask for {case_id}: "
            f"foreground_fraction={foreground_fraction:.4f}"
        )
        return "suspicious_full_mask"
    return "saved"


def first_existing_file(*paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


# ==========================================
# 2. 单病例处理流程 (引入前景归一化)
# ==========================================
def preprocess_promis_case(
    case_dir,
    output_dir,
    target_spacing=DEFAULT_TARGET_SPACING,
    crop_size=DEFAULT_CROP_SIZE,
):
    case_id = os.path.basename(case_dir)
    save_path = os.path.join(output_dir, case_id)

    os.makedirs(save_path, exist_ok=True)

    try:
        t2 = sitk.ReadImage(os.path.join(case_dir, 't2.nii.gz'))
        adc = sitk.ReadImage(os.path.join(case_dir, 'adc.nii.gz'))
        dwi = sitk.ReadImage(os.path.join(case_dir, 'dwi.nii.gz'))
        gland = sitk.ReadImage(os.path.join(case_dir, 'gland.nii.gz'))
        zones = sitk.ReadImage(os.path.join(case_dir, 'gland_zone_20level_set1.nii.gz'))
    except Exception as e:
        print(f"\n[Error] Missing files in {case_id}: {e}")
        return

    lesion_path = os.path.join(case_dir, 'lesion_a1.nii.gz')
    lesion = sitk.ReadImage(lesion_path) if os.path.exists(lesion_path) else None

    def resample(image, is_label=False):
        new_size = [int(round(image.GetSize()[i] * image.GetSpacing()[i] / target_spacing[i])) for i in range(3)]
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
        return resampler.Execute(image)

    # --- 对所有模态和Mask进行重采样 ---
    t2_res = resample(t2)
    gland_res = resample(gland, is_label=True)
    zones_res = resample(zones, is_label=True)
    lesion_res = resample(lesion, is_label=True) if lesion is not None else None
    adc_res_init = resample(adc)
    dwi_res_init = resample(dwi)
    gland_res = sitk.BinaryThreshold(
        gland_res,
        lowerThreshold=0.5,
        upperThreshold=1e9,
        insideValue=1,
        outsideValue=0,
    )

    try:
        adc_reg = register_images(t2_res, adc_res_init)
        dwi_reg = register_images(t2_res, dwi_res_init)
    except Exception as e:
        print(f"\n[Error] Registration failed for {case_id}: {e}")
        return

    def pad_if_needed(img, target_size):
        curr_size = img.GetSize()
        pad_lower = [0, 0, 0]; pad_upper = [0, 0, 0]; need_pad = False
        for i in range(3):
            if curr_size[i] < target_size[i]:
                diff = target_size[i] - curr_size[i]
                pad_lower[i] = diff // 2; pad_upper[i] = diff - pad_lower[i]
                need_pad = True
        return sitk.ConstantPad(img, pad_lower, pad_upper, 0.0) if need_pad else img

    # --- 对所有模态和Mask进行越界补齐 ---
    t2_res = pad_if_needed(t2_res, crop_size)
    adc_reg = pad_if_needed(adc_reg, crop_size)
    dwi_reg = pad_if_needed(dwi_reg, crop_size)
    gland_res = pad_if_needed(gland_res, crop_size)
    zones_res = pad_if_needed(zones_res, crop_size)
    lesion_res = pad_if_needed(lesion_res, crop_size) if lesion_res is not None else None

    # --- 以 Gland 的质心计算裁剪坐标 ---
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(gland_res)
    if not stats.HasLabel(1):
        centroid_index = [s // 2 for s in t2_res.GetSize()]
    else:
        centroid_world = stats.GetCentroid(1)
        centroid_index = t2_res.TransformPhysicalPointToIndex(centroid_world)

    img_size = t2_res.GetSize()
    roi_start = [max(0, min(int(centroid_index[i] - crop_size[i] // 2), img_size[i] - crop_size[i])) for i in range(3)]

    # --- 执行同步裁剪 ---
    t2_crop = sitk.RegionOfInterest(t2_res, crop_size, roi_start)
    adc_crop = sitk.RegionOfInterest(adc_reg, crop_size, roi_start)
    dwi_crop = sitk.RegionOfInterest(dwi_reg, crop_size, roi_start)
    gland_crop = sitk.RegionOfInterest(gland_res, crop_size, roi_start)
    zones_crop = sitk.RegionOfInterest(zones_res, crop_size, roi_start)
    lesion_crop = (
        sitk.RegionOfInterest(lesion_res, crop_size, roi_start)
        if lesion_res is not None
        else None
    )
    gland_crop = sitk.Cast(gland_crop, sitk.sitkUInt8)
    zones_crop = sitk.Cast(zones_crop, sitk.sitkUInt8)
    if lesion_crop is not None:
        lesion_crop = sitk.Cast(lesion_crop, sitk.sitkUInt16)
    warn_unexpected_zone_ids(zones_crop, case_id)

    # ========================================================
    # 【核心修改点】：前景 Z-score 局部归一化逻辑
    # ========================================================
    # 先提取出 Gland 的 Numpy 数组，作为计算统计量的掩膜标尺
    gland_arr = sitk.GetArrayFromImage(gland_crop).astype(np.uint8)

    def finalize_foreground(img, mask_arr):
        """仅利用前列腺腺体内部的像素计算均值和方差"""
        arr = sitk.GetArrayFromImage(img).astype(np.float32)

        # 提取前列腺内部有效像素
        valid_pixels = arr[mask_arr > 0]

        # 防御性回退：如果掩膜为空，则全局归一化
        if len(valid_pixels) == 0:
            mean_val = np.mean(arr)
            std_val = np.std(arr)
        else:
            mean_val = np.mean(valid_pixels)
            std_val = np.std(valid_pixels)

        return (arr - mean_val) / (std_val + 1e-8)

    # 保存 1: 多模态输入张量 (传入 gland_arr 作为前景计算标尺)
    input_tensor = np.stack([
        finalize_foreground(t2_crop, gland_arr),
        finalize_foreground(dwi_crop, gland_arr),
        finalize_foreground(adc_crop, gland_arr)
    ], axis=0)

    np.save(os.path.join(save_path, 'input_tensor.npy'), input_tensor)
    save_input_tensor_nifti(
        input_tensor,
        t2_crop,
        os.path.join(save_path, 'input_tensor.nii.gz'),
    )

    # 保存 2: Gland 腺体掩膜
    sitk.WriteImage(gland_crop, os.path.join(save_path, 'gland_mask.nii.gz'))

    # 保存 3: 裁剪并对齐后的 20 Zones 掩膜
    sitk.WriteImage(zones_crop, os.path.join(save_path, 'zones_mask.nii.gz'))

    if lesion_crop is not None:
        saved, foreground_fraction = save_lesion_label_map(
            lesion_crop,
            os.path.join(save_path, 'lesion_a1_mask.nii.gz'),
        )
        if not saved:
            print(
                f"\n[Warning] Skipped suspicious lesion mask for {case_id}: "
                f"foreground_fraction={foreground_fraction:.4f}"
            )


# ==========================================
# 3. 批量处理逻辑
# ==========================================
def batch_preprocess(src_root, dst_root, lesion_only=False, overwrite_lesion=False):
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    all_cases = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d)) and d.startswith('P-')]

    print(f"Total cases found: {len(all_cases)}")

    if lesion_only:
        counts = {
            "saved": 0,
            "exists": 0,
            "missing_lesion": 0,
            "missing_reference": 0,
            "suspicious_full_mask": 0,
        }
        for case_id in tqdm(all_cases, desc="Backfilling PROMIS lesion_a1"):
            case_dir = os.path.join(src_root, case_id)
            status = backfill_lesion_a1_case(
                case_dir,
                dst_root,
                overwrite=overwrite_lesion,
            )
            counts[status] = counts.get(status, 0) + 1

        print("\nLesion-only backfill complete.")
        for key, value in counts.items():
            print(f"  {key}: {value}")
        return

    for case_id in tqdm(all_cases, desc="Batch Processing PROMIS"):
        case_dir = os.path.join(src_root, case_id)
        preprocess_promis_case(case_dir, dst_root)

if __name__ == "__main__":
    DATASET_ROOT = os.environ.get("RP_DATASET_ROOT")
    if not DATASET_ROOT:
        raise SystemExit("Set RP_DATASET_ROOT before preprocessing data.")
    MRI_DATA_ROOT = os.path.join(DATASET_ROOT, 'derived PROMIS data set', 'MRI')
    PROCESSED_ROOT = os.path.join(
        DATASET_ROOT,
        'derived PROMIS data set',
        'Processed_PROMIS_dwi',
    )
    PROCESS_LESION_ONLY = True
    OVERWRITE_EXISTING_LESION = True

    batch_preprocess(
        MRI_DATA_ROOT,
        PROCESSED_ROOT,
        lesion_only=PROCESS_LESION_ONLY,
        overwrite_lesion=OVERWRITE_EXISTING_LESION,
    )
    print("\nAll done!")
