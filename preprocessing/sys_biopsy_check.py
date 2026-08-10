import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm  # 引入 tqdm 以显示进度条

def map_sys_labels_to_3d_mask(patient_dir):
    """
    将 1D 的 systematic_labels 映射回 3D 的 zones_mask 中，
    生成体素级别带有 ISUP 分级的 NIfTI 图像，用于可视化检查。
    """
    # 1. 定义文件路径
    zones_mask_path = os.path.join(patient_dir, 'zones_mask.nii.gz')
    labels_12_path = os.path.join(patient_dir, 'systematic_labels.npy')
    labels_20_path = os.path.join(patient_dir, 'systematic_labels.npy')

    # 确定使用的 label 文件 (优先 12 区，兼容 20 区)
    labels_path = labels_12_path if os.path.exists(labels_12_path) else labels_20_path

    if not os.path.exists(zones_mask_path) or not os.path.exists(labels_path):
        # 调试用：如果缺失文件可以不打印，避免打乱进度条
        # print(f"[{os.path.basename(patient_dir)}] 缺少 zones_mask 或 labels 文件，跳过。")
        return False

    try:
        # 2. 读取 3D 分区掩膜和 1D 标签数组
        zones_img = sitk.ReadImage(zones_mask_path)
        zones_arr = sitk.GetArrayFromImage(zones_img)

        sys_labels = np.load(labels_path)

        # 3. 创建一个与 zones_mask 形状相同的空白数组
        mapped_arr = np.zeros_like(zones_arr, dtype=np.uint8)

        # 4. 执行映射逻辑
        # zone_arr 中的值是 1~12 (或 1~20)，背景是 0
        # sys_labels 数组中的索引是 0~11 (对应 zone 1~12)
        max_zone_id = int(np.max(zones_arr))

        for zone_id in range(1, max_zone_id + 1):
            # 获取该分区在 1D 数组中的病理结果 (注意索引减 1)
            # 数组长度是 20，足够包容 12 和 20 的情况
            isup_grade = sys_labels[zone_id - 1]

            # 如果该区域有活检结果 (ISUP > 0)，则赋值给 3D 数组对应的体素
            if isup_grade > 0:
                mapped_arr[zones_arr == zone_id] = isup_grade

        # 5. 转回 SimpleITK Image 并复制空间元数据
        mapped_img = sitk.GetImageFromArray(mapped_arr)
        mapped_img.CopyInformation(zones_img)

        # 6. 保存为新的 NIfTI 文件
        output_path = os.path.join(patient_dir, 'systematic_bx_mapped.nii.gz')
        sitk.WriteImage(mapped_img, output_path)
        return True

    except Exception as e:
        print(f"\n❌ 处理 {os.path.basename(patient_dir)} 时发生错误: {e}")
        return False

if __name__ == "__main__":
    DATASET_ROOT = os.environ.get("RP_DATASET_ROOT")
    if not DATASET_ROOT:
        raise SystemExit("Set RP_DATASET_ROOT before checking biopsy labels.")
    dataset_root = os.path.join(DATASET_ROOT, "Target biosy", "Processed_TCIA")
    # dataset_root = os.path.join(DATASET_ROOT, "derived PROMIS data set", "Processed_PROMIS")

    # 修改了此处的 startswith 判断逻辑，匹配实际的文件夹命名
    patients = [d for d in os.listdir(dataset_root)
                if os.path.isdir(os.path.join(dataset_root, d)) and d.startswith('Prostate-MRI-US-Biopsy-')]
    # patients = [d for d in os.listdir(dataset_root)
    #              if os.path.isdir(os.path.join(dataset_root, d)) and d.startswith('P-')]
    print(f"\n--- 正在批量处理 {len(patients)} 个病人 ---")

    # 使用 tqdm 包装循环，动态显示进度条
    success_count = 0
    for pid in tqdm(patients, desc="Mapping SYS Labels"):
        is_success = map_sys_labels_to_3d_mask(os.path.join(dataset_root, pid))
        if is_success:
            success_count += 1

    print(f"\n✅ 批量处理完成！成功生成 {success_count} / {len(patients)} 个 3D 系统活检掩膜文件。")
