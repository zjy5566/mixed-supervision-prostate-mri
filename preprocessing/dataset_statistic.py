import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from collections import defaultdict

def calculate_dataset_statistics(base_dir):
    unified_dir = os.path.join(base_dir, 'Unified_Dataset')

    # 初始化统计字典
    stats = {
        'PUB': {
            'lesion_pixels': 0,
            'background_pixels': 0
        },
        'TCIA_Target': defaultdict(int), # 统计各 ISUP 级别的像素数 (0为背景)
        'TCIA_Sys': defaultdict(int),    # 统计各 ISUP 级别的分区个数
        'PROMIS_Sys': defaultdict(int)   # 统计各 ISUP 级别的分区个数
    }

    patients = [d for d in os.listdir(unified_dir) if os.path.isdir(os.path.join(unified_dir, d))]

    for pid in tqdm(patients, desc="Scanning Dataset"):
        p_dir = os.path.join(unified_dir, pid)

        # ==========================================
        # 1. 统计 PUB 数据集 (Lesion vs Background 像素)
        # ==========================================
        if pid.startswith('PUB_'):
            lesion_path = os.path.join(p_dir, 'lesion_mask.npy')
            if os.path.exists(lesion_path):
                mask = np.load(lesion_path)
                # 假设 mask 中大于 0 的都是 lesion
                stats['PUB']['lesion_pixels'] += np.sum(mask > 0)
                stats['PUB']['background_pixels'] += np.sum(mask == 0)

        # ==========================================
        # 2. 统计 TCIA 数据集 (Target 像素 + Sys 分区)
        # ==========================================
        elif pid.startswith('TCIA_'):
            # 2.1 统计 Target Biopsy 像素
            target_path = os.path.join(p_dir, 'target_mask.nii.gz')
            if os.path.exists(target_path):
                img = sitk.ReadImage(target_path)
                mask = sitk.GetArrayFromImage(img)
                unique_vals, counts = np.unique(mask, return_counts=True)
                for val, count in zip(unique_vals, counts):
                    stats['TCIA_Target'][int(val)] += count

            # 2.2 统计 Systematic Biopsy 12 分区
            sys_path = os.path.join(p_dir, 'systematic_labels_12.npy')
            if os.path.exists(sys_path):
                labels = np.load(sys_path)
                unique_vals, counts = np.unique(labels, return_counts=True)
                for val, count in zip(unique_vals, counts):
                    stats['TCIA_Sys'][int(val)] += count

        # ==========================================
        # 3. 统计 PROMIS 数据集 (Sys 20 分区)
        # ==========================================
        elif pid.startswith('PROMIS_'):
            sys_path = os.path.join(p_dir, 'systematic_labels_20.npy')
            if os.path.exists(sys_path):
                labels = np.load(sys_path)
                unique_vals, counts = np.unique(labels, return_counts=True)
                for val, count in zip(unique_vals, counts):
                    stats['PROMIS_Sys'][int(val)] += count

    # ==========================================
    # 打印统计结果
    # ==========================================
    print("\n" + "="*50)
    print(" DATASET STATISTICS SUMMARY ".center(50, "="))
    print("="*50)

    # 1. PUB 打印
    pub_lesion = stats['PUB']['lesion_pixels']
    pub_bg = stats['PUB']['background_pixels']
    if pub_bg > 0:
        pub_ratio = pub_lesion / pub_bg
        print(f"\n[PUB Dataset] (Pixel-level)")
        print(f" - Lesion Pixels:     {pub_lesion:,}")
        print(f" - Background Pixels: {pub_bg:,}")
        print(f" - Ratio (Lesion:Bg): 1 : {pub_bg/pub_lesion:.2f}  (约 {pub_ratio:.4%})")

    # 2. TCIA Target 打印
    print(f"\n[TCIA Dataset - TBx-confirmed Target Lesion ROI] (Pixel-level)")
    target_bg = stats['TCIA_Target'].get(0, 0)
    target_lesion_total = sum(v for k, v in stats['TCIA_Target'].items() if k > 0)
    print(f" - Background Pixels: {target_bg:,}")
    print(f" - Target ROI Pixels (All ISUP): {target_lesion_total:,}")
    for k in sorted(stats['TCIA_Target'].keys()):
        if k > 0:
            print(f"    * ISUP {k}: {stats['TCIA_Target'][k]:,}")
    if target_lesion_total > 0:
        print(f" - Ratio (Target:Bg): 1 : {target_bg/target_lesion_total:.2f} (约 {target_lesion_total/target_bg:.6%})")

    # 3. TCIA Sys 打印
    print(f"\n[TCIA Dataset - Systematic Biopsy] (Zone-level)")
    tcia_sys_neg = stats['TCIA_Sys'].get(0, 0)
    tcia_sys_pos = sum(v for k, v in stats['TCIA_Sys'].items() if k > 0)
    print(f" - Benign/Bg Zones (ISUP 0): {tcia_sys_neg:,}")
    print(f" - Positive Zones (All ISUP): {tcia_sys_pos:,}")
    for k in sorted(stats['TCIA_Sys'].keys()):
        if k > 0:
            print(f"    * ISUP {k}: {stats['TCIA_Sys'][k]:,}")
    if tcia_sys_pos > 0:
        print(f" - Ratio (Pos Zone:Neg Zone): 1 : {tcia_sys_neg/tcia_sys_pos:.2f}")

    # 4. PROMIS Sys 打印
    print(f"\n[PROMIS Dataset - Systematic Biopsy] (Zone-level)")
    promis_sys_neg = stats['PROMIS_Sys'].get(0, 0)
    promis_sys_pos = sum(v for k, v in stats['PROMIS_Sys'].items() if k > 0)
    print(f" - Benign/Bg Zones (ISUP 0): {promis_sys_neg:,}")
    print(f" - Positive Zones (All ISUP): {promis_sys_pos:,}")
    for k in sorted(stats['PROMIS_Sys'].keys()):
        if k > 0:
            print(f"    * ISUP {k}: {stats['PROMIS_Sys'][k]:,}")
    if promis_sys_pos > 0:
        print(f" - Ratio (Pos Zone:Neg Zone): 1 : {promis_sys_neg/promis_sys_pos:.2f}")

    print("\n" + "="*50)

if __name__ == "__main__":
    BASE_DIR = os.environ.get("RP_DATASET_ROOT")
    if not BASE_DIR:
        raise SystemExit("Set RP_DATASET_ROOT before computing dataset statistics.")
    calculate_dataset_statistics(BASE_DIR)
