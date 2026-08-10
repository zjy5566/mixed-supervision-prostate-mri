import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

# --- 1. 12分区文献定义字典 ---
ZONE_DICT = {
    'LEFT LATERAL BASE': 1,
    'LEFT LATERAL MID': 2,
    'LEFT LATERAL APEX': 3,
    'LEFT BASE': 4,
    'LEFT MID': 5,
    'LEFT APEX': 6,
    'RIGHT BASE': 7,
    'RIGHT MID': 8,
    'RIGHT APEX': 9,
    'RIGHT LATERAL BASE': 10,
    'RIGHT LATERAL MID': 11,
    'RIGHT LATERAL APEX': 12
}

def get_isup_label(primary, secondary):
    """
    计算 Gleason 分数并映射为训练用的 ISUP Label
    """
    if pd.isna(primary) or pd.isna(secondary):
        return 1

    p, s = int(primary), int(secondary)
    if p + s <= 6: return 2
    if p + s == 7: return 3 if p == 3 else 4
    if p + s == 8: return 5
    if p + s >= 9: return 6
    return 1

def create_needle_mask(original_img, tip_idx, base_idx, radius=2):
    """
    在原始 3D 空间中生成一根柱状针道
    注意：这里现在接收的是原始尺寸的 T2，而不是 64x64x32
    """
    img_shape = sitk.GetArrayViewFromImage(original_img).shape
    D, H, W = img_shape
    mask_arr = np.zeros(img_shape, dtype=np.uint8)

    num_points = 200 # 由于原始图像较大，增加采样点让针道更连贯
    points_z = np.linspace(tip_idx[2], base_idx[2], num_points)
    points_y = np.linspace(tip_idx[1], base_idx[1], num_points)
    points_x = np.linspace(tip_idx[0], base_idx[0], num_points)

    for z, y, x in zip(points_z, points_y, points_x):
        iz, iy, ix = int(round(z)), int(round(y)), int(round(x))
        if 0 <= iz < D and 0 <= iy < H and 0 <= ix < W:
            mask_arr[iz, iy, ix] = 1

    mask_sitk = sitk.GetImageFromArray(mask_arr)
    mask_sitk.CopyInformation(original_img)

    # 因为原始图像的分辨率较高（比如 0.5x0.5mm），使用球形膨胀加粗针道
    dilated_mask = sitk.BinaryDilate(mask_sitk > 0, [radius, radius, radius], sitk.sitkBall)
    return sitk.GetArrayFromImage(dilated_mask)

def generate_12_zone_mask(original_img, gland_binary):
    """基于物理坐标空间，自动将前列腺分割为 12 个解剖区域"""
    zones_mask = np.zeros_like(gland_binary, dtype=np.uint8)
    if not np.any(gland_binary):
        return sitk.GetImageFromArray(zones_mask)

    z_idx, y_idx, x_idx = np.where(gland_binary > 0)
    phys_coords = [original_img.TransformIndexToPhysicalPoint((int(x), int(y), int(z))) for z, y, x in zip(z_idx, y_idx, x_idx)]
    phys_coords = np.array(phys_coords)

    X, Z = phys_coords[:, 0], phys_coords[:, 2]
    x_min, x_max = X.min(), X.max()
    z_min, z_max = Z.min(), Z.max()

    z_thresh1 = z_min + (z_max - z_min) / 3
    z_thresh2 = z_min + 2 * (z_max - z_min) / 3

    x_mid = (x_min + x_max) / 2
    x_left_mid = x_mid + (x_max - x_mid) / 2
    x_right_mid = x_min + (x_mid - x_min) / 2

    for i in range(len(z_idx)):
        x_val, z_val = X[i], Z[i]

        if z_val > z_thresh2: z_str = "BASE"
        elif z_val > z_thresh1: z_str = "MID"
        else: z_str = "APEX"

        if x_val > x_left_mid: x_str = "LEFT LATERAL"
        elif x_val > x_mid: x_str = "LEFT"
        elif x_val > x_right_mid: x_str = "RIGHT"
        else: x_str = "RIGHT LATERAL"

        zone_name = f"{x_str} {z_str}"
        zones_mask[z_idx[i], y_idx[i], x_idx[i]] = ZONE_DICT.get(zone_name, 0)

    zones_sitk = sitk.GetImageFromArray(zones_mask)
    zones_sitk.CopyInformation(original_img)
    return zones_sitk

def process_patient_folder(folder_name, biopsy_df, extracted_root):
    """
    处理 Extracted_Target_Biopsy 中的单个文件夹，例如 Prostate-MRI-US-Biopsy-0001_90221
    """
    folder_path = os.path.join(extracted_root, folder_name)

    # 提取病人的基准 ID 和 UID 后缀
    if "_" in folder_name:
        base_pid = folder_name.split('_')[0]
        uid_suffix = folder_name.split('_')[1]
    else:
        base_pid = folder_name
        uid_suffix = ""

    t2_path = os.path.join(folder_path, 't2.nii.gz')
    gland_mask_path = os.path.join(folder_path, 'gland_mask.nii.gz') # 这是上一道工序 stl2mask 生成的

    if not os.path.exists(t2_path) or not os.path.exists(gland_mask_path):
        return

    try:
        t2_img = sitk.ReadImage(t2_path)
        gland_mask = sitk.ReadImage(gland_mask_path)
        gland_binary = (sitk.GetArrayFromImage(gland_mask) > 0).astype(np.uint8)
    except Exception as e:
        print(f"Error reading images for {folder_name}: {e}")
        return

    # 从 Excel 中找到这个病人所有的穿刺记录
    p_data = biopsy_df[biopsy_df['Patient Number'] == base_pid]

    # 【核心匹配逻辑】：只保留那些 "Series Instance UID (MRI)" 尾数能够与当前文件夹后缀对得上的穿刺记录
    if uid_suffix != "":
        # 由于可能存在 NaN 的情况，先将该列转为字符串并剔除首尾空格
        valid_uids_mask = p_data['Series Instance UID (MRI)'].astype(str).str.strip().str.endswith(uid_suffix)
        p_data = p_data[valid_uids_mask]

    if p_data.empty:
        return # 这个文件夹（特定的一组MRI）没有对应的活检记录

    # 数据初始化 (基于原始 t2 尺寸)
    t2_shape = sitk.GetArrayViewFromImage(t2_img).shape
    target_mask_arr = np.zeros(t2_shape, dtype=np.uint8)
    sys_labels = np.zeros(12, dtype=np.uint8)

    found_any_target = False
    found_any_sys = False

    # 遍历筛选出的属于这个病人的、且属于这次 MRI 扫描的穿刺点
    for _, row in p_data.iterrows():
        try:
            core_label = str(row['Core Label']).strip().upper()
            isup_label = get_isup_label(row['Primary Gleason'], row['Secondary Gleason'])

            # --- 分支 A: 靶向穿刺生成 Mask ---
            if core_label == 'TARGET OR PRIOR POSITIVE':
                if pd.isna(row['Bx Tip X (MRI Coord)']) or pd.isna(row['Bx Base X (MRI Coord)']):
                    continue

                tip_phys = [row['Bx Tip X (MRI Coord)'], row['Bx Tip Y (MRI Coord)'], row['Bx Tip Z (MRI Coord)']]
                base_phys = [row['Bx Base X (MRI Coord)'], row['Bx Base Y (MRI Coord)'], row['Bx Base Z (MRI Coord)']]

                tip_idx = t2_img.TransformPhysicalPointToIndex(tip_phys)
                base_idx = t2_img.TransformPhysicalPointToIndex(base_phys)

                needle_arr = create_needle_mask(t2_img, tip_idx, base_idx)

                target_mask_arr = np.maximum(target_mask_arr, needle_arr * isup_label)
                found_any_target = True

            # --- 分支 B: 系统穿刺提取 Array ---
            else:
                if core_label in ZONE_DICT:
                    idx = ZONE_DICT[core_label] - 1
                    sys_labels[idx] = max(sys_labels[idx], isup_label)
                    found_any_sys = True

        except Exception as e:
            continue

    # 保存靶向活检 Mask（如果存在）
    if found_any_target:
        # 使用 gland_binary 进行约束，保证针道只存在于前列腺腺体内部
        target_mask_arr = target_mask_arr * gland_binary
        t_mask = sitk.GetImageFromArray(target_mask_arr)
        t_mask.CopyInformation(t2_img)
        sitk.WriteImage(t_mask, os.path.join(folder_path, 'target_bx_needle.nii.gz')) # 命名为 needle 区分刚才 stl 提取出的靶点

    # 保存系统活检分区及标签（如果存在）
    if found_any_sys:
        zones_sitk = generate_12_zone_mask(t2_img, gland_binary)
        sitk.WriteImage(zones_sitk, os.path.join(folder_path, 'zones_mask.nii.gz'))
        np.save(os.path.join(folder_path, 'systematic_labels.npy'), sys_labels)

# --- 执行主流程 ---
if __name__ == "__main__":
    DATASET_ROOT = os.environ.get("RP_DATASET_ROOT")
    if not DATASET_ROOT:
        raise SystemExit("Set RP_DATASET_ROOT before preprocessing data.")
    BIOPSY_EXCEL = os.path.join(
        DATASET_ROOT,
        'Target biosy',
        'unprocessed_data',
        'TCIA-Biopsy-Data_2020-07-14.xlsx',
    )
    EXTRACTED_ROOT = os.path.join(DATASET_ROOT, 'Target biosy', 'Extracted_Target_Biopsy')

    if not os.path.exists(BIOPSY_EXCEL):
        print(f"Error: Cannot find Excel file at {BIOPSY_EXCEL}")
    else:
        print("Loading TCIA Biopsy Excel...")
        df = pd.read_excel(BIOPSY_EXCEL)

        # 遍历 Extracted_Target_Biopsy 里的所有文件夹
        folders = [d for d in os.listdir(EXTRACTED_ROOT) if d.startswith('Prostate-MRI-US-Biopsy-') and os.path.isdir(os.path.join(EXTRACTED_ROOT, d))]
        print(f"Total potential extracted folders found: {len(folders)}")

        for folder_name in tqdm(folders, desc="Processing Biopsy Data"):
            process_patient_folder(folder_name, df, EXTRACTED_ROOT)

        print("\nSuccess: Check your folders in Extracted_Target_Biopsy")
        print("Now you can run the Process script to crop and resample everything!")
