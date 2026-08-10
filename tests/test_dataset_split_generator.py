import tempfile
from pathlib import Path

import pandas as pd

from preprocessing.Dataset_split_generator import create_split_csvs


def _row(
    patient_id: str,
    source: str,
    *,
    target: int = 0,
    sbx12: int = 0,
    sbx20: int = 0,
    lesion: int = 0,
) -> dict:
    return {
        "patient_id": patient_id,
        "source": source,
        "has_target": target,
        "has_sys_12": sbx12,
        "has_sys_20": sbx20,
        "has_lesion": lesion,
        "has_gland": 1,
    }


def _synthetic_registry() -> pd.DataFrame:
    rows = []
    rows.extend(_row(f"PUB_{i:02d}", "PUB", lesion=1) for i in range(12))
    rows.extend(
        _row(f"TCIA_joint_{i:02d}", "TCIA", target=1, sbx12=1)
        for i in range(20)
    )
    rows.extend(
        _row(f"TCIA_tbx_only_{i:02d}", "TCIA", target=1)
        for i in range(5)
    )
    rows.extend(
        _row(f"TCIA_sbx_only_{i:02d}", "TCIA", sbx12=1)
        for i in range(7)
    )
    rows.extend(
        _row(f"TCIA_neither_{i:02d}", "TCIA")
        for i in range(2)
    )
    rows.extend(
        _row(f"PROMIS_{i:02d}", "PROMIS", sbx20=1)
        for i in range(8)
    )
    return pd.DataFrame(rows)


def _patient_ids(df: pd.DataFrame, source: str | None = None) -> set[str]:
    if source is not None:
        df = df[df["source"] == source]
    return set(df["patient_id"].astype(str))


def test_all_tcia_experiments_use_the_same_joint_tbx_sbx_patients():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        registry_csv = root / "registry.csv"
        splits_dir = root / "splits"
        _synthetic_registry().to_csv(registry_csv, index=False)

        splits = create_split_csvs(
            str(registry_csv),
            str(splits_dir),
            val_size=0.2,
            internal_test_size=0.1,
            random_state=42,
        )

        b1_ids = _patient_ids(splits["B1_TCIA_TBx_baseline_train.csv"])
        b2_ids = _patient_ids(splits["B2_TCIA_SBx_only_train.csv"])
        b3_ids = _patient_ids(splits["B3_TCIA_TBx_SBx_train.csv"])
        assert b1_ids == b2_ids == b3_ids

        n2_tcia_ids = _patient_ids(
            splits["N2_PUB_TCIA_TBx_only_train.csv"], "TCIA"
        )
        n3_tcia_ids = _patient_ids(
            splits["N3_PUB_TCIA_SBx_only_train.csv"], "TCIA"
        )
        n4_tcia_ids = _patient_ids(
            splits["N4_mixed_PUB_TCIA_train.csv"], "TCIA"
        )
        assert n2_tcia_ids == n3_tcia_ids == n4_tcia_ids == b1_ids
        assert (
            _patient_ids(splits["N2_PUB_TCIA_TBx_only_train.csv"])
            == _patient_ids(splits["N3_PUB_TCIA_SBx_only_train.csv"])
            == _patient_ids(splits["N4_mixed_PUB_TCIA_train.csv"])
        )

        expected_joint_ids = {f"TCIA_joint_{i:02d}" for i in range(20)}
        val_ids = _patient_ids(splits["TCIA_common_internal_evaluation.csv"])
        test_ids = _patient_ids(splits["TCIA_common_internal_test.csv"])
        assert b1_ids.isdisjoint(val_ids)
        assert b1_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)
        assert b1_ids | val_ids | test_ids == expected_joint_ids

        for split_df in splits.values():
            if "source" not in split_df.columns:
                continue
            tcia_rows = split_df[split_df["source"] == "TCIA"]
            assert all(
                patient_id.startswith("TCIA_joint_")
                for patient_id in tcia_rows["patient_id"].astype(str)
            )


def test_task_views_hide_labels_without_changing_joint_cohort_membership():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        registry_csv = root / "registry.csv"
        _synthetic_registry().to_csv(registry_csv, index=False)
        splits = create_split_csvs(str(registry_csv), str(root / "splits"))

        b1 = splits["B1_TCIA_TBx_baseline_train.csv"]
        b2 = splits["B2_TCIA_SBx_only_train.csv"]
        b3 = splits["B3_TCIA_TBx_SBx_train.csv"]

        assert (b1["has_target"] == 1).all()
        assert (b1[["has_sys_12", "has_sys_20", "can_sbx"]] == 0).all().all()
        assert (b2["has_target"] == 0).all()
        assert (b2["can_tbx"] == 0).all()
        assert (b2["can_sbx"] == 1).all()
        assert (b3[["can_tbx", "can_sbx"]] == 1).all().all()
        assert (b1["eligible_tcia_tbx_sbx"] == 1).all()
        assert (b2["eligible_tcia_tbx_sbx"] == 1).all()
        assert (b3["eligible_tcia_tbx_sbx"] == 1).all()

        prepared_registry = pd.read_csv(root / "splits" / "dataset_registry.csv")
        excluded = prepared_registry[
            prepared_registry["patient_id"].str.contains("only|neither")
        ]
        assert len(excluded) == 14
        assert (excluded["eligible_tcia_tbx_sbx"] == 0).all()


def test_duplicate_source_patient_ids_are_rejected_before_splitting():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        registry = _synthetic_registry()
        registry = pd.concat([registry, registry.iloc[[0]]], ignore_index=True)
        registry_csv = root / "registry.csv"
        registry.to_csv(registry_csv, index=False)

        try:
            create_split_csvs(str(registry_csv), str(root / "splits"))
        except ValueError as exc:
            assert "duplicate keys" in str(exc)
            assert "PUB_00" in str(exc)
        else:
            raise AssertionError("Duplicate source/patient_id rows were not rejected.")
