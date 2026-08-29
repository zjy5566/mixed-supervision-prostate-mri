"""
Generate experiment split CSV files from an existing dataset registry.

This script does not copy, rename, or preprocess image files. It only reads an
existing registry CSV and creates reproducible train/internal-validation/internal-
test/external-validation CSV files.

Current experiment design:
    Every TCIA cohort is drawn from the same patients with both TBx and SBx.
    B1: expose TCIA TBx-confirmed radiologist target lesion ROIs only.
    B2: expose TCIA SBx region supervision only.
    B3: expose both TCIA TBx-confirmed target ROIs and TCIA SBx supervision.

Legacy/reference splits are also written:
    N1: dense radiologist annotations only.
    N2: dense radiologist annotations + TCIA TBx supervision only.
    N3: dense radiologist annotations + TCIA SBx supervision only.
    N4: dense radiologist annotations + TCIA TBx and SBx supervision.

Training supervision differs across experiments. The B-series trains on TCIA,
but uses the shared dense-annotation + TCIA internal evaluation cohort so the
baseline can report lesion Dice alongside biopsy-based TCIA/PROMIS metrics.

PROMIS is held out as the external validation source for every experiment.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
DEFAULT_VAL_SIZE = 0.2
DEFAULT_INTERNAL_TEST_SIZE = 0.1


def _safe_train_val_split(
    df: pd.DataFrame,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split one source into train and internal validation sets safely."""
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if len(df) == 0:
        return df.copy(), df.copy()

    if len(df) < 5:
        # Avoid sklearn errors for very small datasets while keeping at least
        # one validation case whenever possible.
        n_val = max(1, int(round(len(df) * val_size)))
        n_val = min(n_val, len(df) - 1) if len(df) > 1 else 1
        val_df = df.iloc[:n_val].copy()
        train_df = df.iloc[n_val:].copy()
        return train_df, val_df

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _safe_train_val_test_split(
    df: pd.DataFrame,
    val_size: float = DEFAULT_VAL_SIZE,
    internal_test_size: float = DEFAULT_INTERNAL_TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split one source into train, internal validation, and internal test sets.

    val_size and internal_test_size are fractions of the original source-level
    dataframe. The validation fraction is therefore preserved after carving out
    the internal test set.
    """
    if val_size < 0.0 or internal_test_size < 0.0 or val_size + internal_test_size >= 1.0:
        raise ValueError(
            "val_size and internal_test_size must be non-negative and sum to < 1; "
            f"got val_size={val_size}, internal_test_size={internal_test_size}"
        )

    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if len(df) == 0:
        return df.copy(), df.copy(), df.copy()

    if len(df) < 5:
        n_test = 1 if len(df) >= 3 and internal_test_size > 0 else 0
        n_val = 1 if len(df) - n_test >= 2 and val_size > 0 else 0
        test_df = df.iloc[:n_test].copy()
        val_df = df.iloc[n_test:n_test + n_val].copy()
        train_df = df.iloc[n_test + n_val:].copy()
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    if internal_test_size == 0.0:
        train_df, val_df = _safe_train_val_split(
            df,
            val_size=val_size,
            random_state=random_state,
        )
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            df.iloc[0:0].copy(),
        )

    train_val_df, test_df = train_test_split(
        df,
        test_size=internal_test_size,
        random_state=random_state,
        shuffle=True,
    )

    if val_size == 0.0:
        return (
            train_val_df.reset_index(drop=True),
            df.iloc[0:0].copy(),
            test_df.reset_index(drop=True),
        )

    adjusted_val_size = val_size / (1.0 - internal_test_size)
    train_df, val_df = _safe_train_val_split(
        train_val_df,
        val_size=adjusted_val_size,
        random_state=random_state,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _write_split(df: pd.DataFrame, path: str) -> None:
    """Write one split CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.reset_index(drop=True).to_csv(path, index=False)


def _prepare_registry(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the registry and add supervision-availability columns."""
    df = df.copy()

    required_columns = {
        "patient_id",
        "source",
        "has_target",
        "has_sys_12",
        "has_sys_20",
        "has_lesion",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "Registry CSV is missing required columns: "
            f"{sorted(missing)}"
        )

    if "has_gland" not in df.columns:
        df["has_gland"] = 0

    if df["patient_id"].isna().any():
        raise ValueError("Registry CSV contains a missing patient_id.")
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["source"] = df["source"].astype(str).str.upper().str.strip()
    if (df["patient_id"] == "").any():
        raise ValueError("Registry CSV contains an empty patient_id.")
    duplicate_mask = df.duplicated(["source", "patient_id"], keep=False)
    if duplicate_mask.any():
        duplicate_keys = (
            df.loc[duplicate_mask, ["source", "patient_id"]]
            .drop_duplicates()
            .sort_values(["source", "patient_id"])
        )
        preview = duplicate_keys.head(10).to_dict("records")
        raise ValueError(
            "Registry must contain one row per source/patient_id before "
            f"splitting; duplicate keys include: {preview}"
        )

    binary_columns = [
        "has_target",
        "has_sys_12",
        "has_sys_20",
        "has_lesion",
        "has_gland",
    ]
    for column in binary_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)

    df["can_seg"] = df["has_lesion"].astype(int)
    df["can_tbx"] = df["has_target"].astype(int)
    df["can_sbx"] = (
        (df["has_sys_12"] == 1) | (df["has_sys_20"] == 1)
    ).astype(int)
    df["can_cls"] = (
        (df["can_tbx"] == 1) | (df["can_sbx"] == 1)
    ).astype(int)
    # Cohort eligibility is computed before task-specific views hide labels.
    # It therefore remains an auditable provenance flag in B1/B2 and N2/N3.
    df["eligible_tcia_tbx_sbx"] = (
        (df["source"] == "TCIA")
        & (df["can_tbx"] == 1)
        & (df["can_sbx"] == 1)
    ).astype(int)

    def supervision_type(row: pd.Series) -> str:
        if row["source"] == "PUB":
            return "dense_radiologist_annotation"
        if row["source"] == "TCIA":
            if row["can_tbx"] and row["can_sbx"]:
                return "tbx_confirmed_roi_and_sbx"
            if row["can_tbx"]:
                return "tbx_confirmed_roi_only"
            if row["can_sbx"]:
                return "sbx_only"
        if row["source"] == "PROMIS" and row["can_sbx"]:
            return "sbx_only"
        return "unknown"

    df["supervision_type"] = df.apply(supervision_type, axis=1)
    return df


def _select_joint_tcia_biopsy_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Select the common TCIA cohort with both TBx and SBx available."""
    return df[df["eligible_tcia_tbx_sbx"] == 1].copy()


def _make_tbx_only_view(df: pd.DataFrame) -> pd.DataFrame:
    """Keep TBx-labelled TCIA cases and hide SBx supervision in the CSV.

    Cases that possess both TBx and SBx labels are retained, but their SBx
    availability flags are set to zero. Therefore this view exposes only
    TBx-confirmed target ROI supervision to the dataset/loss pipeline.
    """
    out = df[df["can_tbx"] == 1].copy()
    out["has_sys_12"] = 0
    out["has_sys_20"] = 0
    out["can_sbx"] = 0
    out["can_cls"] = out["can_tbx"]
    out["supervision_type"] = "tbx_confirmed_roi_only_for_experiment"
    return out


def _make_sbx_only_view(df: pd.DataFrame) -> pd.DataFrame:
    """Keep SBx-labelled TCIA cases and hide TBx supervision in the CSV.

    Cases that possess both TBx and SBx labels are retained, but their TBx
    availability flag is set to zero. Therefore this view exposes only SBx
    supervision to the dataset/loss pipeline.
    """
    out = df[df["can_sbx"] == 1].copy()
    out["has_target"] = 0
    out["can_tbx"] = 0
    out["can_cls"] = out["can_sbx"]
    out["supervision_type"] = "sbx_only_for_experiment"
    return out


def _concat_pub_tcia(
    pub_df: pd.DataFrame,
    tcia_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine dense-annotation cases and selected TCIA supervision cases."""
    return pd.concat(
        [
            pub_df[pub_df["can_seg"] == 1].copy(),
            tcia_df.copy(),
        ],
        ignore_index=True,
    )


def create_split_csvs(
    registry_csv: str,
    splits_dir: str,
    external_source: str = "PROMIS",
    val_size: float = DEFAULT_VAL_SIZE,
    internal_test_size: float = DEFAULT_INTERNAL_TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Dict[str, pd.DataFrame]:
    """Create experiment CSV files from an existing registry CSV.

    The dense-annotation and full TCIA registries are each split once. Joint
    TBx/SBx eligibility is then applied inside every TCIA partition so
    established patient assignments remain stable. B-series training CSVs are
    TCIA-only; legacy N-series CSVs keep the dense-annotation + TCIA setup for
    comparison. PROMIS is never included in training, internal validation, or
    internal test.
    """
    if not os.path.exists(registry_csv):
        raise FileNotFoundError(f"Registry CSV not found: {registry_csv}")

    if not 0.0 <= val_size < 1.0:
        raise ValueError(f"val_size must be between 0 and 1, got {val_size}")
    if not 0.0 <= internal_test_size < 1.0:
        raise ValueError(
            "internal_test_size must be between 0 and 1, "
            f"got {internal_test_size}"
        )
    if val_size + internal_test_size >= 1.0:
        raise ValueError(
            "val_size + internal_test_size must be < 1, "
            f"got {val_size + internal_test_size}"
        )

    os.makedirs(splits_dir, exist_ok=True)

    registry = _prepare_registry(pd.read_csv(registry_csv))
    _write_split(registry, os.path.join(splits_dir, "dataset_registry.csv"))

    pub_df = registry[registry["source"] == "PUB"].copy()
    tcia_all_df = registry[registry["source"] == "TCIA"].copy()
    tcia_df = _select_joint_tcia_biopsy_pool(tcia_all_df)
    external_df = registry[
        registry["source"] == external_source.upper()
    ].copy()

    if len(tcia_all_df) == 0:
        raise ValueError("Registry contains no TCIA cases.")
    if len(tcia_df) == 0:
        raise ValueError(
            "No TCIA cases have both TBx and SBx supervision. "
            "The experiment cohort requires has_target=1 and either "
            "has_sys_12=1 or has_sys_20=1."
        )

    print(
        "TCIA joint TBx+SBx cohort: "
        f"included={len(tcia_df)}, excluded={len(tcia_all_df) - len(tcia_df)}, "
        f"total={len(tcia_all_df)}"
    )

    pub_train, pub_internal_val, pub_internal_test = _safe_train_val_test_split(
        pub_df,
        val_size=val_size,
        internal_test_size=internal_test_size,
        random_state=random_state,
    )
    # Preserve the established seed-42 source partitions, then apply the joint
    # eligibility rule inside each partition. This keeps every already-joint
    # patient's train/validation/test assignment stable while removing TCIA
    # cases that have only one biopsy type.
    (
        tcia_train_all,
        tcia_internal_val_all,
        tcia_internal_test_all,
    ) = _safe_train_val_test_split(
        tcia_all_df,
        val_size=val_size,
        internal_test_size=internal_test_size,
        random_state=random_state,
    )
    tcia_train = _select_joint_tcia_biopsy_pool(tcia_train_all)
    tcia_internal_val = _select_joint_tcia_biopsy_pool(tcia_internal_val_all)
    tcia_internal_test = _select_joint_tcia_biopsy_pool(tcia_internal_test_all)

    print(
        "TCIA joint cohort partitions: "
        f"train={len(tcia_train)}, validation={len(tcia_internal_val)}, "
        f"test={len(tcia_internal_test)}"
    )

    # External validation uses PROMIS systematic-biopsy supervision.
    promis_external = external_df[external_df["can_sbx"] == 1].copy()

    # ------------------------------------------------------------------
    # Common internal validation/test cohorts for every experiment
    # ------------------------------------------------------------------
    # Keep both original TCIA supervision flags in validation/test. Training
    # views may hide TBx or SBx labels, but evaluation retains both labels so
    # that every experiment can calculate:
    #   1) patient-level BACC on TCIA cases with can_cls == 1;
    #   2) region-level BACC on the subset with can_sbx == 1.
    # Dense radiologist-annotation validation cases are also included so Dice can be
    # measured on the same run. Metric code must use has_target/has_sys or
    # can_cls/can_sbx masks and must not treat dense-annotation cases as negative biopsy
    # labels.
    tcia_common_internal_eval = tcia_internal_val.copy()
    tcia_common_internal_test = tcia_internal_test.copy()
    common_internal_val = _concat_pub_tcia(
        pub_internal_val,
        tcia_common_internal_eval,
    )
    common_internal_test = _concat_pub_tcia(
        pub_internal_test,
        tcia_common_internal_test,
    )

    # ------------------------------------------------------------------
    # B-series: new TCIA-centred experiment story
    # ------------------------------------------------------------------
    # B1/B2/B3 use identical TCIA patients. Task-specific views only hide the
    # supervision that a given experiment is not allowed to optimise.
    b1_train = _make_tbx_only_view(tcia_train)
    b1_internal_val = common_internal_val.copy()
    b1_internal_test = common_internal_test.copy()
    b1_external_val = promis_external.copy()

    b2_train = _make_sbx_only_view(tcia_train)
    b2_internal_val = common_internal_val.copy()
    b2_internal_test = common_internal_test.copy()
    b2_external_val = promis_external.copy()

    b3_train = tcia_train.copy()
    b3_internal_val = common_internal_val.copy()
    b3_internal_test = common_internal_test.copy()
    b3_external_val = promis_external.copy()

    # ------------------------------------------------------------------
    # Legacy/reference N-series: dense radiologist-annotation setup
    # ------------------------------------------------------------------
    # N1: train with dense radiologist annotations only.
    n1_train = pub_train[pub_train["can_seg"] == 1].copy()
    n1_internal_val = common_internal_val.copy()
    n1_internal_test = common_internal_test.copy()
    # Supports patient/region evaluation, not external lesion Dice.
    n1_external_val = promis_external.copy()

    # N2/N3/N4 use identical dense-annotation and TCIA patients. Labels are hidden
    # only in the corresponding training view.
    tcia_tbx_train = _make_tbx_only_view(tcia_train)
    n2_train = _concat_pub_tcia(pub_train, tcia_tbx_train)
    n2_internal_val = common_internal_val.copy()
    n2_internal_test = common_internal_test.copy()
    n2_external_val = promis_external.copy()

    # N3: train with dense radiologist annotations + TCIA SBx only.
    tcia_sbx_train = _make_sbx_only_view(tcia_train)
    n3_train = _concat_pub_tcia(pub_train, tcia_sbx_train)
    n3_internal_val = common_internal_val.copy()
    n3_internal_test = common_internal_test.copy()
    n3_external_val = promis_external.copy()

    # N4: train with dense radiologist annotations + joint TCIA supervision.
    tcia_mixed_train = tcia_train.copy()
    n4_train = _concat_pub_tcia(pub_train, tcia_mixed_train)
    n4_internal_val = common_internal_val.copy()
    n4_internal_test = common_internal_test.copy()
    n4_external_val = promis_external.copy()

    # Task-specific validation views remain available for diagnostic use, but
    # they are not the recommended model-selection CSVs for N1-N4.
    tcia_tbx_internal_val = _make_tbx_only_view(tcia_internal_val)
    tcia_tbx_internal_test = _make_tbx_only_view(tcia_internal_test)
    tcia_sbx_internal_val = _make_sbx_only_view(tcia_internal_val)
    tcia_sbx_internal_test = _make_sbx_only_view(tcia_internal_test)

    splits: Dict[str, pd.DataFrame] = {
        # Source-level reference files.
        "internal_pool.csv": registry[
            (registry["source"] != external_source.upper())
            & (
                (registry["source"] != "TCIA")
                | (registry["eligible_tcia_tbx_sbx"] == 1)
            )
        ].copy(),
        "external_val.csv": external_df.copy(),
        "TCIA_joint_TBx_SBx_pool.csv": tcia_df.copy(),
        "common_internal_evaluation.csv": common_internal_val.copy(),
        "common_internal_test.csv": common_internal_test.copy(),
        "TCIA_common_internal_evaluation.csv": (
            tcia_common_internal_eval.copy()
        ),
        "TCIA_common_internal_test.csv": (
            tcia_common_internal_test.copy()
        ),

        # B-series: TCIA-centred baseline and ablations.
        "B1_TCIA_TBx_baseline_train.csv": b1_train,
        "B1_TCIA_TBx_baseline_internal_val.csv": b1_internal_val,
        "B1_TCIA_TBx_baseline_internal_test.csv": b1_internal_test,
        "B1_PROMIS_external_val.csv": b1_external_val,

        "B2_TCIA_SBx_only_train.csv": b2_train,
        "B2_TCIA_SBx_only_internal_val.csv": b2_internal_val,
        "B2_TCIA_SBx_only_internal_test.csv": b2_internal_test,
        "B2_PROMIS_external_val.csv": b2_external_val,

        "B3_TCIA_TBx_SBx_train.csv": b3_train,
        "B3_TCIA_TBx_SBx_internal_val.csv": b3_internal_val,
        "B3_TCIA_TBx_SBx_internal_test.csv": b3_internal_test,
        "B3_PROMIS_external_val.csv": b3_external_val,

        # N1.
        "N1_radiologist_only_train.csv": n1_train,
        "N1_radiologist_only_internal_val.csv": n1_internal_val,
        "N1_radiologist_only_internal_test.csv": n1_internal_test,
        "N1_PROMIS_external_val.csv": n1_external_val,

        # N2: dense radiologist annotations + TCIA TBx only.
        "N2_PUB_TCIA_TBx_only_train.csv": n2_train,
        "N2_PUB_TCIA_TBx_only_internal_val.csv": n2_internal_val,
        "N2_PUB_TCIA_TBx_only_internal_test.csv": n2_internal_test,
        "N2_PROMIS_external_val.csv": n2_external_val,

        # N3: dense radiologist annotations + TCIA SBx only.
        "N3_PUB_TCIA_SBx_only_train.csv": n3_train,
        "N3_PUB_TCIA_SBx_only_internal_val.csv": n3_internal_val,
        "N3_PUB_TCIA_SBx_only_internal_test.csv": n3_internal_test,
        "N3_PROMIS_external_val.csv": n3_external_val,

        # N4: dense radiologist annotations + joint TCIA biopsy supervision.
        "N4_mixed_PUB_TCIA_train.csv": n4_train,
        "N4_mixed_PUB_TCIA_internal_val.csv": n4_internal_val,
        "N4_mixed_PUB_TCIA_internal_test.csv": n4_internal_test,
        "N4_mixed_PROMIS_external_val.csv": n4_external_val,

        # Optional TCIA-only task files.
        "task_tbx_train.csv": tcia_tbx_train,
        "task_tbx_internal_val.csv": tcia_tbx_internal_val,
        "task_tbx_internal_test.csv": tcia_tbx_internal_test,
        "task_sbx_train.csv": tcia_sbx_train,
        "task_sbx_internal_val.csv": tcia_sbx_internal_val,
        "task_sbx_internal_test.csv": tcia_sbx_internal_test,
        "task_sbx_external_val.csv": promis_external.copy(),
    }

    for filename, split_df in splits.items():
        _write_split(split_df, os.path.join(splits_dir, filename))

    summary_rows = []
    for filename, split_df in splits.items():
        source_counts = (
            split_df["source"].value_counts().to_dict()
            if len(split_df) > 0 and "source" in split_df.columns
            else {}
        )
        summary_rows.append(
            {
                "file": filename,
                "n": len(split_df),
                "source_counts": source_counts,
                "n_seg": int(split_df["can_seg"].sum()) if len(split_df) else 0,
                "n_tbx": int(split_df["can_tbx"].sum()) if len(split_df) else 0,
                "n_sbx": int(split_df["can_sbx"].sum()) if len(split_df) else 0,
                "n_tcia_joint_tbx_sbx": (
                    int(split_df["eligible_tcia_tbx_sbx"].sum())
                    if len(split_df)
                    and "eligible_tcia_tbx_sbx" in split_df.columns
                    else 0
                ),
                # Patient metrics use all cases with at least one biopsy label.
                "n_patient_eval": (
                    int(split_df["can_cls"].sum()) if len(split_df) else 0
                ),
                # Region metrics require systematic-biopsy region labels.
                "n_region_eval": (
                    int(split_df["can_sbx"].sum()) if len(split_df) else 0
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    _write_split(summary_df, os.path.join(splits_dir, "split_summary.csv"))

    print("\nSplit CSV files created successfully.")
    print(summary_df.to_string(index=False))
    print(f"\nSaved to: {splits_dir}")

    return splits


if __name__ == "__main__":
    DATASET_ROOT = os.environ.get("RP_DATASET_ROOT")
    if not DATASET_ROOT:
        raise SystemExit("Set RP_DATASET_ROOT before generating dataset splits.")
    UNIFIED_DATA_DIR = os.path.join(DATASET_ROOT, "Unified_Dataset")
    REGISTRY_CSV = os.path.join(
        UNIFIED_DATA_DIR,
        "splits",
        "dataset_registry.csv",
    )
    SPLITS_DIR = os.path.join(UNIFIED_DATA_DIR, "splits")

    create_split_csvs(
        registry_csv=REGISTRY_CSV,
        splits_dir=SPLITS_DIR,
        external_source="PROMIS",
        val_size=0.2,
        internal_test_size=DEFAULT_INTERNAL_TEST_SIZE,
        random_state=RANDOM_STATE,
    )
