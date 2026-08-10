import os
import shutil
import SimpleITK as sitk
from tqdm import tqdm

def identify_folder_type(root):
    folder_name = os.path.basename(root).lower()
    parent_name = os.path.basename(os.path.dirname(root)).lower()
    combined_name = folder_name + " " + parent_name

    if 'adc' in combined_name or 'apparent diffusion' in combined_name:
        if 'report' not in combined_name: return 'adc'

    dwi_indicators = ['dwi', 'diff', 'b1600', 'b2000', 'b1000', 'b800', 'calcbval']
    if any(x in combined_name for x in dwi_indicators):
        if 'adc' not in folder_name: return 'dwi'

    if 't2' in combined_name:
        if not any(x in combined_name for x in ['sag', 'cor', 'haste', 'loc']): return 't2'

    return None

def read_dicom_series(directory):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    if not dicom_names: return None
    reader.SetFileNames(dicom_names)
    try: return reader.Execute()
    except Exception: return None

def extract_and_save_case(patient_id, patient_path, dst_root, stl_dir):
    # 1. 查找该病人名下的前列腺表面 STL (用于锚定 T2 UID)
    patient_surface_stls = {}
    if os.path.exists(stl_dir):
        for f in sorted(os.listdir(stl_dir)):
            if (
                f.startswith(patient_id)
                and 'prostatesurface' in f.lower()
                and os.path.splitext(f)[1].lower() == '.stl'
            ):
                parts = f.split('-seriesUID-')
                if len(parts) == 2:
                    uid = os.path.splitext(parts[1])[0].strip()
                    patient_surface_stls[uid] = os.path.join(stl_dir, f)

    # 2. 遍历病人的所有 DICOM 文件夹，读取元数据并按 Study UID 归组
    studies = {}
    for root, dirs, files in os.walk(patient_path):
        dirs.sort()
        files.sort()
        dicoms = [f for f in files if f.lower().endswith('.dcm')]
        if dicoms:
            mri_type = identify_folder_type(root)
            if mri_type:
                dcm_path = os.path.join(root, dicoms[0])
                reader = sitk.ImageFileReader()
                reader.SetFileName(dcm_path)
                reader.LoadPrivateTagsOn()
                try:
                    reader.ReadImageInformation()
                    series_uid = reader.GetMetaData('0020|000e').strip()
                    study_uid = reader.GetMetaData('0020|000d').strip()
                except Exception:
                    continue

                if study_uid not in studies:
                    studies[study_uid] = {'t2': [], 'adc': [], 'dwi': [], 't2_uids': []}

                studies[study_uid][mri_type].append((series_uid, root))
                if mri_type == 't2':
                    studies[study_uid]['t2_uids'].append(series_uid)

    # 3. 匹配并提取
    extracted_count = 0
    already_exists_count = 0

    for study_uid, modalities in studies.items():
        matched_uid = None
        for t2_uid in modalities['t2_uids']:
            if t2_uid in patient_surface_stls:
                matched_uid = t2_uid
                break

        if matched_uid:
            # Preserve the complete Series Instance UID. Truncating it to five
            # characters can merge distinct studies or associate the wrong
            # biopsy rows.
            save_folder = os.path.join(dst_root, f"{patient_id}_{matched_uid}")
            surface_stl_to_copy = patient_surface_stls[matched_uid]

            # [新增] 寻找属于这个病人且对应这个 UID 的所有 Target STL 文件
            # 例如: Prostate-MRI-US-Biopsy-0159-Target1-seriesUID-xxxxxx.STL
            target_stls_to_copy = []
            for target_name in sorted(os.listdir(stl_dir)):
                if os.path.splitext(target_name)[1].lower() != '.stl':
                    continue
                target_parts = target_name.split('-seriesUID-')
                if len(target_parts) != 2:
                    continue
                target_uid = os.path.splitext(target_parts[1])[0].strip()
                if (
                    not target_name.startswith(f"{patient_id}-Target")
                    or target_uid != matched_uid
                ):
                    continue
                t_file = os.path.join(stl_dir, target_name)
                # 提取 Target 编号 (例如 Target1 -> 1)
                t_name = os.path.basename(t_file)
                try:
                    t_num = t_name.split('-Target')[1].split('-seriesUID')[0]
                    target_stls_to_copy.append((t_num, t_file))
                except:
                    pass

        elif len(patient_surface_stls) == 0 and len(modalities['t2']) > 0:
            save_folder = os.path.join(dst_root, patient_id)
            surface_stl_to_copy = None
            target_stls_to_copy = []
        else:
            continue

        # 检查是否已经存在
        req_files = ['t2.nii.gz']
        if surface_stl_to_copy: req_files.append('prostate_surface.stl')
        if target_stls_to_copy: req_files.extend([f"target_{num}.stl" for num, _ in target_stls_to_copy])

        # if os.path.exists(save_folder):
        #     if all(f in os.listdir(save_folder) for f in req_files):
        #         already_exists_count += 1
        #         continue

        os.makedirs(save_folder, exist_ok=True)

        success = False
        for m_type in ['t2', 'adc', 'dwi']:
            candidates = sorted(modalities[m_type])
            if candidates:
                if m_type == 't2' and matched_uid:
                    candidates = [
                        candidate
                        for candidate in candidates
                        if candidate[0] == matched_uid
                    ]
                    if not candidates:
                        continue
                _, mri_folder = candidates[0]
                out_path = os.path.join(save_folder, f"{m_type}.nii.gz")
                if not os.path.exists(out_path):
                    img = read_dicom_series(mri_folder)
                    if img:
                        sitk.WriteImage(img, out_path)
                        success = True
                else:
                    success = True

        # 拷贝 Gland STL
        if success and surface_stl_to_copy:
            dest_stl = os.path.join(save_folder, "prostate_surface.stl")
            if not os.path.exists(dest_stl):
                shutil.copy2(surface_stl_to_copy, dest_stl)

        # [新增] 拷贝所有 Target STL
        if success and len(target_stls_to_copy) > 0:
            for t_num, t_file in target_stls_to_copy:
                dest_t_stl = os.path.join(save_folder, f"target_{t_num}.stl")
                if not os.path.exists(dest_t_stl):
                    shutil.copy2(t_file, dest_t_stl)

        if success: extracted_count += 1

    if extracted_count > 0: return "SUCCESS"
    elif already_exists_count > 0: return "ALREADY_EXISTS"
    else: return "FAILED"

def batch_extract_mri(src_root, dst_root, stl_dir):
    os.makedirs(dst_root, exist_ok=True)

    patients = sorted(
        d for d in os.listdir(src_root)
        if d.startswith('Prostate-MRI-US-Biopsy-')
    )
    print(f"Found {len(patients)} potential patients.")

    new_count = skip_count = fail_count = 0
    pbar = tqdm(patients, desc="Extracting MRI")
    for p_id in pbar:
        p_path = os.path.join(src_root, p_id)
        result = extract_and_save_case(p_id, p_path, dst_root, stl_dir)

        if result == "SUCCESS": new_count += 1
        elif result == "ALREADY_EXISTS": skip_count += 1
        else: fail_count += 1

        pbar.set_postfix({"New": new_count, "Skip": skip_count, "Fail": fail_count})

    print("\n" + "="*30)
    print(f"Extraction Summary:")
    print(f"  - Newly Extracted: {new_count}")
    print(f"  - Already Exists:  {skip_count}")
    print(f"  - Failed/Missing:  {fail_count}")
    print("="*30)

if __name__ == "__main__":
    DATASET_ROOT = os.environ.get("RP_DATASET_ROOT")
    if not DATASET_ROOT:
        raise SystemExit("Set RP_DATASET_ROOT before extracting data.")
    SRC_DIR = os.path.join(
        DATASET_ROOT,
        'Target biosy',
        'unprocessed_data',
        'manifest-1694710246744',
        'Prostate-MRI-US-Biopsy',
    )
    DST_DIR = os.path.join(DATASET_ROOT, 'Target biosy', 'Extracted_Target_Biopsy')
    STL_DIR = os.path.join(
        DATASET_ROOT,
        'Target biosy',
        'unprocessed_data',
        'Prostate-MRI-US-Biopsy',
        'STLs',
        'STLs',
    )

    batch_extract_mri(SRC_DIR, DST_DIR, STL_DIR)
