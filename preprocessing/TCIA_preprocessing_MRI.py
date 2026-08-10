import os
import shutil
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

def find_existing_file(src_path, candidates):
    for name in candidates:
        path = os.path.join(src_path, name)
        if os.path.exists(path):
            return path
    return None

# --- 1. 核心配准函数 (保持不变) ---
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
    return resampler.Execute(moving_image)

# --- 2. 重采样函数 (保持不变) ---
def resample_to_spacing(image, target_spacing=[1.0, 1.0, 2.24], is_label=False):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(round(original_size[i] * original_spacing[i] / target_spacing[i])) for i in range(3)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resampler.Execute(image)

# --- 3. 以前列腺掩膜为中心的裁剪函数 ---
def mask_centered_crop(img, mask, target_size=[64, 64, 32]):
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(mask)

    if label_shape_filter.GetNumberOfLabels() == 0:
        center_index = [s // 2 for s in img.GetSize()]
    else:
        centroid_world = label_shape_filter.GetCentroid(1)
        center_index = img.TransformPhysicalPointToIndex(centroid_world)

    roi_start = [center_index[i] - target_size[i] // 2 for i in range(3)]
    img_size = img.GetSize()
    pad_lower = [max(0, -roi_start[i]) for i in range(3)]
    pad_upper = [max(0, roi_start[i] + target_size[i] - img_size[i]) for i in range(3)]

    if sum(pad_lower) + sum(pad_upper) > 0:
        img = sitk.ConstantPad(img, pad_lower, pad_upper, 0)
        roi_start = [roi_start[i] + pad_lower[i] for i in range(3)]

    return sitk.RegionOfInterest(img, target_size, roi_start)

# ========================================================
# --- 4. 归一化函数 (前景 Z-score 局部归一化) ---
# ========================================================
def normalize_array(img, mask_arr):
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    valid_pixels = arr[mask_arr > 0]

    if len(valid_pixels) == 0:
        mean_val = np.mean(arr)
        std_val = np.std(arr)
    else:
        mean_val = np.mean(valid_pixels)
        std_val = np.std(valid_pixels)

    return (arr - mean_val) / (std_val + 1e-8)


def crop_and_save_label(label_path, mask_res, reference_crop, output_path):
    label = sitk.ReadImage(label_path)
    label_res = resample_to_spacing(label, is_label=True)
    label_crop = mask_centered_crop(label_res, mask_res)
    label_crop.CopyInformation(reference_crop)
    sitk.WriteImage(label_crop, output_path)


# --- 5. 单个病例处理流程 ---
def process_single_patient(folder_name, src_path, dst_root):
    t2_file = find_existing_file(src_path, ['t2.nii.gz', 'T2.nii.gz', 't2w.nii.gz'])
    adc_file = find_existing_file(src_path, ['adc.nii.gz', 'ADC.nii.gz'])
    dwi_file = find_existing_file(src_path, ['dwi.nii.gz', 'DWI.nii.gz', 'hbv.nii.gz'])
    mask_file = find_existing_file(
        src_path,
        ['gland_mask.nii.gz', 'prostate_mask.nii.gz', 'prostate_surface_mask.nii.gz'],
    )
    target_mask_file = find_existing_file(
        src_path,
        ['target_mask.nii.gz', 'target_lesion_mask.nii.gz'],
    )
    zones_mask_file = find_existing_file(
        src_path,
        ['zones_mask.nii.gz', 'systematic_zones_mask.nii.gz'],
    )
    sys_labels_file = find_existing_file(
        src_path,
        ['systematic_labels.npy', 'systematic_labels_12.npy'],
    )

    save_dir = os.path.join(dst_root, folder_name)

    # Required inputs for building the concatenated MRI tensor.
    if not all([t2_file, adc_file, dwi_file, mask_file]):
        return "MISSING_DATA"

    try:
        os.makedirs(save_dir, exist_ok=True)

        t2 = sitk.ReadImage(t2_file)
        adc = sitk.ReadImage(adc_file)
        dwi = sitk.ReadImage(dwi_file)
        mask = sitk.ReadImage(mask_file)

        # 重采样
        t2_res = resample_to_spacing(t2)
        adc_res = resample_to_spacing(adc)
        dwi_res = resample_to_spacing(dwi)
        mask_res = resample_to_spacing(mask, is_label=True)

        # 配准
        adc_reg = register_images(t2_res, adc_res)
        dwi_reg = register_images(t2_res, dwi_res)

        # 中心裁剪 (核心金标准位置)
        t2_crop = mask_centered_crop(t2_res, mask_res)
        adc_crop = mask_centered_crop(adc_reg, mask_res)
        dwi_crop = mask_centered_crop(dwi_reg, mask_res)
        mask_crop = mask_centered_crop(mask_res, mask_res)

        # ========================================================
        # 提取 mask 数组并传入归一化函数构建多通道张量
        # ========================================================
        mask_arr = sitk.GetArrayFromImage(mask_crop).astype(np.uint8)

        input_tensor = np.stack([
            normalize_array(t2_crop, mask_arr),
            normalize_array(dwi_crop, mask_arr),
            normalize_array(adc_crop, mask_arr)
        ], axis=0)

        np.save(os.path.join(save_dir, 'input_tensor.npy'), input_tensor)

        input_tensor_vector = np.moveaxis(input_tensor.astype(np.float32), 0, -1)
        input_tensor_img = sitk.GetImageFromArray(input_tensor_vector, isVector=True)
        input_tensor_img.CopyInformation(t2_crop)
        sitk.WriteImage(input_tensor_img, os.path.join(save_dir, 'input_tensor.nii.gz'))

        # Persist the prostate gland crop alongside the image tensor so later
        # training can apply the optional outside-gland lesion-risk penalty.
        mask_crop.CopyInformation(t2_crop)
        sitk.WriteImage(mask_crop, os.path.join(save_dir, 'gland_mask.nii.gz'))

        if target_mask_file:
            crop_and_save_label(
                target_mask_file,
                mask_res,
                t2_crop,
                os.path.join(save_dir, 'target_mask.nii.gz'),
            )

        if zones_mask_file:
            crop_and_save_label(
                zones_mask_file,
                mask_res,
                t2_crop,
                os.path.join(save_dir, 'zones_mask.nii.gz'),
            )

        if sys_labels_file:
            shutil.copy2(sys_labels_file, os.path.join(save_dir, 'systematic_labels.npy'))

        return "SUCCESS"
    except Exception as e:
        print(f"\n[Error] {folder_name}: {e}")
        return "FAILED"

# --- 6. 主程序入口 ---
if __name__ == "__main__":
    DATASET_ROOT = os.environ.get("RP_DATASET_ROOT")
    if not DATASET_ROOT:
        raise SystemExit("Set RP_DATASET_ROOT before preprocessing data.")
    SRC_ROOT = os.path.join(DATASET_ROOT, 'Target biosy', 'Extracted_Target_Biopsy')
    DST_ROOT = os.path.join(DATASET_ROOT, 'Target biosy', 'Processed_TCIA')

    # 这里读取真实的文件夹名称 (无论是 Prostate-MRI-US-Biopsy-0159 还是 Prostate-MRI-US-Biopsy-0159_12345 都会被抓取)
    folders = [d for d in os.listdir(SRC_ROOT) if d.startswith('Prostate-MRI-US-Biopsy-') and os.path.isdir(os.path.join(SRC_ROOT, d))]
    print(f"Total potential cases found: {len(folders)}")

    stats = {
        "complete_modalities": 0,
        "newly_processed": 0,
        "already_exists": 0,
        "failed": 0,
        "missing": 0
    }

    pbar = tqdm(folders, desc="Batch Processing")
    for folder_name in pbar:
        src_path = os.path.join(SRC_ROOT, folder_name)
        result = process_single_patient(folder_name, src_path, DST_ROOT)

        if result == "SUCCESS":
            stats["complete_modalities"] += 1
            stats["newly_processed"] += 1
        elif result == "ALREADY_PROCESSED":
            stats["complete_modalities"] += 1
            stats["already_exists"] += 1
        elif result == "MISSING_DATA":
            stats["missing"] += 1
        elif result == "FAILED":
            stats["failed"] += 1

        pbar.set_postfix({
            "Complete": stats["complete_modalities"],
            "New": stats["newly_processed"],
            "Exist": stats["already_exists"]
        })

    print(f"\n" + "="*40)
    print(f"Preprocessing Summary:")
    print(f"  - Total Complete Cases (T2+ADC+DWI+Mask): {stats['complete_modalities']}")
    print(f"    └─ Newly Processed: {stats['newly_processed']}")
    print(f"    └─ Already Exists:  {stats['already_exists']}")
    print(f"  - Incomplete / Missing Data: {stats['missing']}")
    print(f"  - Processing Failed:         {stats['failed']}")
    print(f"  - Output Directory: {DST_ROOT}")
    print("="*40)
