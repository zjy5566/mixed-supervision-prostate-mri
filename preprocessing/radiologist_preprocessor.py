import os
import glob
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

class ProstateDataPreprocessor:
    def __init__(self, root_dir, output_dir, target_spacing=[1.0, 1.0, 2.24], target_size=[64, 64, 32]):
        """
        :param root_dir: 包含 imagesTr, labelsTr, zonesTr 的根目录
        :param output_dir: 处理后数据的保存目录
        """
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.target_spacing = target_spacing
        self.target_size = target_size
        os.makedirs(output_dir, exist_ok=True)

    # --- 1. 重采样函数 ---
    def resample_to_spacing(self, image, is_label=False):
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        new_size = [int(round(original_size[i] * original_spacing[i] / self.target_spacing[i])) for i in range(3)]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(self.target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
        return resampler.Execute(image)

    # --- 2. 以前列腺掩膜为中心的裁剪函数 ---
    def mask_centered_crop(self, img, mask):
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        binary_mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
        label_shape_filter.Execute(binary_mask)

        if label_shape_filter.GetNumberOfLabels() == 0:
            center_index = [s // 2 for s in img.GetSize()]
        else:
            centroid_world = label_shape_filter.GetCentroid(1)
            center_index = img.TransformPhysicalPointToIndex(centroid_world)

        roi_start = [center_index[i] - self.target_size[i] // 2 for i in range(3)]

        img_size = img.GetSize()
        pad_lower = [max(0, -roi_start[i]) for i in range(3)]
        pad_upper = [max(0, roi_start[i] + self.target_size[i] - img_size[i]) for i in range(3)]

        if sum(pad_lower) + sum(pad_upper) > 0:
            img = sitk.ConstantPad(img, pad_lower, pad_upper, 0)
            roi_start = [roi_start[i] + pad_lower[i] for i in range(3)]

        return sitk.RegionOfInterest(img, self.target_size, roi_start)

    # --- 3. 归一化函数 (已修改为局部/前景归一化) ---
    def normalize_array(self, img_arr, mask_arr):
        """
        局部 Z-score 归一化 (Foreground Normalization)
        仅利用前列腺区域 (mask > 0) 的像素计算均值和方差，并应用到整张图。
        """
        img_arr = img_arr.astype(np.float32)

        # 提取前列腺内部的有效像素
        valid_pixels = img_arr[mask_arr > 0]

        # 防御性编程：万一遇到了极其罕见的空 mask，回退到全局归一化
        if len(valid_pixels) == 0:
            mean_val = np.mean(img_arr)
            std_val = np.std(img_arr)
        else:
            mean_val = np.mean(valid_pixels)
            std_val = np.std(valid_pixels)

        # 将局部算出的均值和方差，应用到整张图像上
        norm_arr = (img_arr - mean_val) / (std_val if std_val > 1e-7 else 1.0)

        # 【进阶可选操作】：如果你希望网络完全不受周围脂肪和直肠的干扰，
        # 可以取消下面这行的注释，直接把前列腺外面的背景全部设为纯黑(0)。
        # norm_arr[mask_arr == 0] = 0.0

        return norm_arr

    def process_patient(self, patient_id):
        # 1. 构建路径
        t2_path = os.path.join(self.root_dir, 'imagesTr', f"{patient_id}_0000.nii.gz")
        adc_path = os.path.join(self.root_dir, 'imagesTr', f"{patient_id}_0001.nii.gz")
        dwi_path = os.path.join(self.root_dir, 'imagesTr', f"{patient_id}_0002.nii.gz")
        label_path = os.path.join(self.root_dir, 'labelsTr', f"{patient_id}.nii.gz")
        zone_path = os.path.join(self.root_dir, 'zonesTr', f"{patient_id}.nii.gz")

        if not all([os.path.exists(p) for p in [t2_path, adc_path, dwi_path, zone_path, label_path]]):
            return

        # 2. 读取数据
        t2 = sitk.ReadImage(t2_path)
        adc = sitk.ReadImage(adc_path)
        dwi = sitk.ReadImage(dwi_path)
        zone = sitk.ReadImage(zone_path)
        label = sitk.ReadImage(label_path)

        # 3. 统一重采样
        t2_res = self.resample_to_spacing(t2)
        adc_res = self.resample_to_spacing(adc)
        dwi_res = self.resample_to_spacing(dwi)
        zone_res = self.resample_to_spacing(zone, is_label=True)
        label_res = self.resample_to_spacing(label, is_label=True)

        # 4. 以前列腺质心进行中心裁剪
        t2_crop = self.mask_centered_crop(t2_res, zone_res)
        adc_crop = self.mask_centered_crop(adc_res, zone_res)
        dwi_crop = self.mask_centered_crop(dwi_res, zone_res)
        label_crop = self.mask_centered_crop(label_res, zone_res)
        zone_crop = self.mask_centered_crop(zone_res, zone_res)

        # 5. 转为 Numpy 数组
        t2_arr = sitk.GetArrayFromImage(t2_crop)
        adc_arr = sitk.GetArrayFromImage(adc_crop)
        dwi_arr = sitk.GetArrayFromImage(dwi_crop)

        # 【修改逻辑】：提前提取 zone_mask，因为归一化函数现在需要用到它
        zone_arr = sitk.GetArrayFromImage(zone_crop)
        final_zone = zone_arr.astype(np.uint8)

        # 6. 多模态堆叠并执行前景局部归一化 (传入 final_zone)
        stacked_img = np.stack([
            self.normalize_array(t2_arr, final_zone),
            self.normalize_array(dwi_arr, final_zone),
            self.normalize_array(adc_arr, final_zone)
        ], axis=0).astype(np.float32)

        # 7. 标签与区域掩膜处理
        label_arr = sitk.GetArrayFromImage(label_crop)
        final_label = (label_arr > 0).astype(np.uint8)

        # 8. 保存处理后的结果
        np.save(os.path.join(self.output_dir, f"{patient_id}_img.npy"), stacked_img)
        # stacked_img_nii = sitk.GetImageFromArray(stacked_img, isVector=False)
        # stacked_img_nii.CopyInformation(t2_crop)
        # sitk.WriteImage(stacked_img_nii, os.path.join(self.output_dir, f"{patient_id}_img.nii.gz"))
        np.save(os.path.join(self.output_dir, f"{patient_id}_lab.npy"), final_label)
        np.save(os.path.join(self.output_dir, f"{patient_id}_zone.npy"), final_zone)

    def run_all(self):
        t2_files = glob.glob(os.path.join(self.root_dir, 'imagesTr', '*_0000.nii.gz'))
        patient_ids = sorted([os.path.basename(f).replace('_0000.nii.gz', '') for f in t2_files])

        valid_count = 0
        for pid in tqdm(patient_ids, desc="Processing MRI Dataset"):
            try:
                label_path = os.path.join(self.root_dir, 'labelsTr', f"{pid}.nii.gz")
                if not os.path.exists(label_path):
                    continue

                self.process_patient(pid)
                valid_count += 1
            except Exception as e:
                print(f"Error processing {pid}: {e}")

        print(f"Processing complete. Successfully processed {valid_count} patients with valid lesion masks.")

# --- 使用示例 ---
if __name__ == "__main__":
    DATASET_ROOT = os.environ.get("RP_DATASET_ROOT")
    if not DATASET_ROOT:
        raise SystemExit("Set RP_DATASET_ROOT before preprocessing data.")
    raw_data_path = os.path.join(DATASET_ROOT, "Dataset_prostate_MRI")
    processed_path = os.path.join(
        DATASET_ROOT,
        "Dataset_prostate_MRI",
        "Dataset_prostate_MRI",
    )

    preprocessor = ProstateDataPreprocessor(raw_data_path, processed_path)
    print("Script started...")
    preprocessor.run_all()
    print(f"All done! Output saved to: {processed_path}")
