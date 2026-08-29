#!/usr/bin/env python3
"""Audit spatial alignment of T2, DWI, and ADC in ``input_tensor.npy``.

The unified tensor has channel order ``T2, DWI, ADC`` and array order
``(C, D, H, W)``. Because NPY files do not retain image geometry metadata,
this script checks *content alignment*, not origin/spacing/direction. It uses:

1. phase correlation of multi-modal gradient-magnitude volumes to propose
   small residual integer translations; and
2. normalised mutual information (NMI) inside a dilated gland region to test
   whether a proposed translation improves alignment over the stored tensor.

A case is marked SUSPECT only when a non-trivial shift also produces a
meaningful NMI gain. This avoids treating a noisy translation estimate as
proof of misregistration. Optional figures show T2 together with DWI/ADC
before and after the estimated residual shift.

This is a screening/QC tool. Multi-modal similarity cannot prove perfect
registration, so flagged cases must be confirmed visually. The dataset is
never modified.

Example
-------
conda run -n medical_ai python preprocessing/check_input_alignment.py \
  --dataset-root /path/to/Unified_Dataset \
  --output-dir output/input_alignment_audit \
  --workers 6
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage


CHANNEL_NAMES = ("T2", "DWI", "ADC")
PAIR_SPECS = ((0, 1, "T2-DWI"), (0, 2, "T2-ADC"))
ORIENTATION_HYPOTHESES = {
    "identity": (),
    "flip_z": (0,),
    "flip_y": (1,),
    "flip_x": (2,),
    "flip_zy": (0, 1),
    "flip_zx": (0, 2),
    "flip_yx": (1, 2),
    "flip_zyx": (0, 1, 2),
}
SOURCE_ALIASES = {
    "radiologist": "PUB",
    "pub": "PUB",
    "promis": "PROMIS",
    "tcia": "TCIA",
}


def _source_name(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in SOURCE_ALIASES:
        return SOURCE_ALIASES[normalized]
    raise argparse.ArgumentTypeError(
        "expected one of: radiologist, promis, tcia"
    )


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Check T2/DWI/ADC content alignment in Unified_Dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the prepared Unified_Dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "output" / "input_alignment_audit",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        type=_source_name,
        metavar="{radiologist,promis,tcia}",
        default=("radiologist", "promis", "tcia"),
        help="Input cohorts to audit.",
    )
    parser.add_argument(
        "--max-shift",
        type=int,
        default=5,
        help="Maximum residual translation searched along each axis, in voxels.",
    )
    parser.add_argument(
        "--phase-candidates",
        type=int,
        default=20,
        help="Number of phase-correlation translation candidates validated by NMI.",
    )
    parser.add_argument("--nmi-bins", type=int, default=32)
    parser.add_argument(
        "--gland-dilation",
        type=int,
        default=3,
        help="Binary dilation iterations for the NMI evaluation region.",
    )
    parser.add_argument(
        "--flag-shift",
        type=int,
        default=2,
        help="Minimum max-axis residual shift required for a shift flag.",
    )
    parser.add_argument(
        "--min-nmi-gain",
        type=float,
        default=0.02,
        help="Minimum NMI improvement required for a shift flag.",
    )
    parser.add_argument(
        "--min-orientation-gain",
        type=float,
        default=0.05,
        help="Minimum NMI improvement required to flag a possible axis flip.",
    )
    parser.add_argument(
        "--min-roi-voxels",
        type=int,
        default=256,
        help="Minimum valid voxels required for an NMI calculation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of cases for a quick test run.",
    )
    parser.add_argument(
        "--visuals",
        choices=("suspect", "top", "none"),
        default="suspect",
        help="Which cases receive PNG QC figures.",
    )
    parser.add_argument(
        "--visual-count",
        type=int,
        default=50,
        help="Maximum number of QC figures.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a synthetic shift-direction test and exit.",
    )
    args = parser.parse_args()
    args.sources = tuple(_source_name(source) for source in args.sources)
    return args


def discover_cases(dataset_root: Path, sources: Sequence[str]) -> List[Path]:
    allowed = set(sources)
    cases = []
    for input_path in dataset_root.glob("*/input_tensor.npy"):
        case_dir = input_path.parent
        source = case_dir.name.split("_", 1)[0]
        if source in allowed:
            cases.append(case_dir)
    return sorted(cases, key=lambda p: p.name)


def load_input_tensor(case_dir: Path) -> np.ndarray:
    tensor = np.load(case_dir / "input_tensor.npy", allow_pickle=False)
    tensor = np.asarray(tensor)
    if tensor.ndim != 4:
        raise ValueError(f"input_tensor must be 4D, got {tensor.shape}")
    if tensor.shape[0] == 3:
        pass
    elif tensor.shape[-1] == 3:
        tensor = np.moveaxis(tensor, -1, 0)
    else:
        raise ValueError(
            f"cannot locate three modality channels in input_tensor shape {tensor.shape}"
        )
    return tensor.astype(np.float32, copy=False)


def load_gland_mask(case_dir: Path, expected_shape: Tuple[int, int, int]) -> np.ndarray:
    npy_path = case_dir / "gland_mask.npy"
    nii_path = case_dir / "gland_mask.nii.gz"
    if npy_path.exists():
        gland = np.load(npy_path, allow_pickle=False)
    elif nii_path.exists():
        gland = sitk.GetArrayFromImage(sitk.ReadImage(str(nii_path)))
    else:
        raise FileNotFoundError("neither gland_mask.npy nor gland_mask.nii.gz exists")

    gland = np.asarray(gland).squeeze()
    if gland.shape != expected_shape:
        raise ValueError(
            f"gland mask shape {gland.shape} != image shape {expected_shape}"
        )
    gland = np.isfinite(gland) & (gland > 0)
    if not gland.any():
        raise ValueError("gland mask is empty")
    return gland


def dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask.copy()
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    return ndimage.binary_dilation(mask, structure=structure, iterations=iterations)


def robust_quantize(
    image: np.ndarray,
    roi: np.ndarray,
    bins: int,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    valid = roi & np.isfinite(image)
    values = image[valid]
    if values.size == 0:
        raise ValueError("no finite image voxels in ROI")
    lo, hi = np.percentile(values, [lower_percentile, upper_percentile])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-8:
        raise ValueError("image is constant or invalid inside ROI")
    scaled = (np.clip(image, lo, hi) - lo) / (hi - lo)
    quantized = np.floor(scaled * bins).astype(np.int16)
    return np.clip(quantized, 0, bins - 1)


def slices_for_shift(
    shape: Sequence[int], shift_zyx: Sequence[int]
) -> Tuple[Tuple[slice, ...], Tuple[slice, ...]]:
    """Return fixed/moving overlap slices for shifting moving by ``shift_zyx``."""
    fixed_slices: List[slice] = []
    moving_slices: List[slice] = []
    for size, shift in zip(shape, shift_zyx):
        shift = int(shift)
        if abs(shift) >= size:
            raise ValueError(f"shift {shift} is too large for axis size {size}")
        if shift >= 0:
            fixed_slices.append(slice(shift, size))
            moving_slices.append(slice(0, size - shift))
        else:
            fixed_slices.append(slice(0, size + shift))
            moving_slices.append(slice(-shift, size))
    return tuple(fixed_slices), tuple(moving_slices)


def normalised_mutual_information(
    fixed_q: np.ndarray,
    moving_q: np.ndarray,
    fixed_roi: np.ndarray,
    shift_zyx: Sequence[int],
    bins: int,
    min_voxels: int,
) -> float:
    fixed_slices, moving_slices = slices_for_shift(fixed_q.shape, shift_zyx)
    roi = fixed_roi[fixed_slices]
    if int(roi.sum()) < min_voxels:
        return float("nan")

    fixed_values = fixed_q[fixed_slices][roi].astype(np.int64, copy=False)
    moving_values = moving_q[moving_slices][roi].astype(np.int64, copy=False)
    joint = np.bincount(
        fixed_values * bins + moving_values,
        minlength=bins * bins,
    ).reshape(bins, bins)
    total = float(joint.sum())
    if total <= 0:
        return float("nan")

    probability = joint / total
    px = probability.sum(axis=1)
    py = probability.sum(axis=0)
    nz_joint = probability > 0
    nz_x = px > 0
    nz_y = py > 0
    h_joint = -float(np.sum(probability[nz_joint] * np.log(probability[nz_joint])))
    h_x = -float(np.sum(px[nz_x] * np.log(px[nz_x])))
    h_y = -float(np.sum(py[nz_y] * np.log(py[nz_y])))
    mutual_information = h_x + h_y - h_joint
    denominator = h_x + h_y
    return 2.0 * mutual_information / denominator if denominator > 0 else float("nan")


def bounding_box(mask: np.ndarray, margin: int) -> Tuple[slice, slice, slice]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return tuple(slice(0, n) for n in mask.shape)  # type: ignore[return-value]
    lower = np.maximum(coords.min(axis=0) - margin, 0)
    upper = np.minimum(coords.max(axis=0) + margin + 1, mask.shape)
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))  # type: ignore[return-value]


def gradient_feature(image: np.ndarray) -> np.ndarray:
    finite = np.isfinite(image)
    if not finite.any():
        raise ValueError("image contains no finite values")
    values = image[finite]
    lo, hi = np.percentile(values, [1.0, 99.0])
    if hi <= lo + 1e-8:
        raise ValueError("image is constant")
    normalised = (np.clip(image, lo, hi) - lo) / (hi - lo)
    normalised = ndimage.gaussian_filter(normalised, sigma=0.8)
    gradients = np.gradient(normalised)
    magnitude = np.sqrt(sum(component * component for component in gradients))
    magnitude -= float(magnitude.mean())
    std = float(magnitude.std())
    if std > 1e-8:
        magnitude /= std

    # A smooth window suppresses circular FFT wraparound without introducing a
    # sharp shared mask boundary that would artificially favour zero shift.
    windows = [np.hanning(max(2, size)) for size in magnitude.shape]
    window_3d = (
        windows[0][:, None, None]
        * windows[1][None, :, None]
        * windows[2][None, None, :]
    )
    return magnitude * window_3d


def phase_shift_candidates(
    fixed: np.ndarray,
    moving: np.ndarray,
    max_shift: int,
    top_k: int,
) -> Tuple[List[Tuple[int, int, int]], Dict[Tuple[int, int, int], float]]:
    fixed_feature = gradient_feature(fixed)
    moving_feature = gradient_feature(moving)
    fixed_fft = np.fft.fftn(fixed_feature)
    moving_fft = np.fft.fftn(moving_feature)
    cross_power = fixed_fft * np.conj(moving_fft)
    cross_power /= np.maximum(np.abs(cross_power), 1e-12)
    correlation = np.real(np.fft.ifftn(cross_power))

    scores: Dict[Tuple[int, int, int], float] = {}
    limits = [min(max_shift, size - 1) for size in fixed.shape]
    for shift in itertools.product(
        range(-limits[0], limits[0] + 1),
        range(-limits[1], limits[1] + 1),
        range(-limits[2], limits[2] + 1),
    ):
        index = tuple(int(value % size) for value, size in zip(shift, fixed.shape))
        scores[tuple(int(v) for v in shift)] = float(correlation[index])

    ranked = sorted(scores, key=scores.get, reverse=True)
    candidates = ranked[: max(1, top_k)]
    zero = (0, 0, 0)
    if zero not in candidates:
        candidates.append(zero)
    return candidates, scores


def orientation_diagnostic(
    fixed_q: np.ndarray,
    moving_q: np.ndarray,
    roi: np.ndarray,
    bins: int,
    min_voxels: int,
) -> Tuple[str, float, float, Dict[str, float]]:
    scores: Dict[str, float] = {}
    for name, axes in ORIENTATION_HYPOTHESES.items():
        candidate = np.flip(moving_q, axis=axes) if axes else moving_q
        scores[name] = normalised_mutual_information(
            fixed_q,
            candidate,
            roi,
            (0, 0, 0),
            bins,
            min_voxels,
        )
    finite_scores = {k: v for k, v in scores.items() if np.isfinite(v)}
    if not finite_scores:
        return "invalid", float("nan"), float("nan"), scores
    best_name = max(finite_scores, key=finite_scores.get)
    identity = scores.get("identity", float("nan"))
    gain = finite_scores[best_name] - identity
    return best_name, finite_scores[best_name], gain, scores


def analyse_pair(
    fixed: np.ndarray,
    moving: np.ndarray,
    roi: np.ndarray,
    crop: Tuple[slice, slice, slice],
    config: Mapping[str, float | int],
) -> Dict[str, object]:
    bins = int(config["nmi_bins"])
    min_voxels = int(config["min_roi_voxels"])
    fixed_q = robust_quantize(fixed, roi, bins)
    moving_q = robust_quantize(moving, roi, bins)

    phase_candidates, phase_scores = phase_shift_candidates(
        fixed[crop],
        moving[crop],
        max_shift=int(config["max_shift"]),
        top_k=int(config["phase_candidates"]),
    )
    shift_scores: Dict[Tuple[int, int, int], float] = {}
    for shift in phase_candidates:
        shift_scores[shift] = normalised_mutual_information(
            fixed_q,
            moving_q,
            roi,
            shift,
            bins,
            min_voxels,
        )

    finite_shift_scores = {
        shift: score for shift, score in shift_scores.items() if np.isfinite(score)
    }
    if not finite_shift_scores:
        raise ValueError("all NMI translation evaluations were invalid")
    best_shift = max(finite_shift_scores, key=finite_shift_scores.get)
    nmi_identity = shift_scores.get((0, 0, 0), float("nan"))
    nmi_best = finite_shift_scores[best_shift]
    nmi_gain = nmi_best - nmi_identity

    best_orientation, orientation_nmi, orientation_gain, orientation_scores = (
        orientation_diagnostic(fixed_q, moving_q, roi, bins, min_voxels)
    )

    max_abs_shift = max(abs(int(v)) for v in best_shift)
    shift_suspect = (
        max_abs_shift >= int(config["flag_shift"])
        and nmi_gain >= float(config["min_nmi_gain"])
    )
    orientation_suspect = (
        best_orientation not in ("identity", "invalid")
        and orientation_gain >= float(config["min_orientation_gain"])
    )
    if shift_suspect and orientation_suspect:
        status = "SUSPECT_SHIFT_AND_ORIENTATION"
    elif shift_suspect:
        status = "SUSPECT_SHIFT"
    elif orientation_suspect:
        status = "SUSPECT_ORIENTATION"
    else:
        status = "PASS"

    return {
        "status": status,
        "best_shift_z": int(best_shift[0]),
        "best_shift_y": int(best_shift[1]),
        "best_shift_x": int(best_shift[2]),
        "max_abs_shift": int(max_abs_shift),
        "shift_l2_voxels": float(math.sqrt(sum(int(v) ** 2 for v in best_shift))),
        "nmi_identity": float(nmi_identity),
        "nmi_best_shift": float(nmi_best),
        "nmi_gain": float(nmi_gain),
        "phase_score_identity": float(phase_scores.get((0, 0, 0), float("nan"))),
        "phase_score_best_shift": float(phase_scores.get(best_shift, float("nan"))),
        "best_orientation": best_orientation,
        "orientation_nmi": float(orientation_nmi),
        "orientation_gain": float(orientation_gain),
        "orientation_scores": ";".join(
            f"{name}:{score:.6f}" for name, score in orientation_scores.items()
        ),
    }


def analyse_case(task: Tuple[str, Dict[str, float | int]]) -> List[Dict[str, object]]:
    case_dir_string, config = task
    case_dir = Path(case_dir_string)
    patient_id = case_dir.name
    source = patient_id.split("_", 1)[0]
    base = {
        "patient_id": patient_id,
        "source": source,
        "case_dir": str(case_dir),
    }
    try:
        tensor = load_input_tensor(case_dir)
        if not np.isfinite(tensor).all():
            nonfinite = [
                int((~np.isfinite(tensor[channel])).sum()) for channel in range(3)
            ]
            raise ValueError(f"non-finite voxels per channel: {nonfinite}")
        gland = load_gland_mask(case_dir, tuple(tensor.shape[1:]))
        roi = dilate_mask(gland, int(config["gland_dilation"]))
        crop = bounding_box(roi, margin=int(config["max_shift"]) + 2)

        rows = []
        for fixed_channel, moving_channel, pair_name in PAIR_SPECS:
            diagnostics = analyse_pair(
                tensor[fixed_channel],
                tensor[moving_channel],
                roi,
                crop,
                config,
            )
            rows.append(
                {
                    **base,
                    "pair": pair_name,
                    "input_shape": "x".join(str(v) for v in tensor.shape),
                    "gland_voxels": int(gland.sum()),
                    "roi_voxels": int(roi.sum()),
                    **diagnostics,
                    "error": "",
                }
            )
        return rows
    except Exception as exc:
        return [
            {
                **base,
                "pair": "LOAD_OR_SHAPE_CHECK",
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=3),
            }
        ]


def integer_shift_for_display(volume: np.ndarray, shift_zyx: Sequence[int]) -> np.ndarray:
    return ndimage.shift(
        volume,
        shift=tuple(int(v) for v in shift_zyx),
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )


def display_limits(volume: np.ndarray, roi: np.ndarray) -> Tuple[float, float]:
    values = volume[roi & np.isfinite(volume)]
    if values.size == 0:
        values = volume[np.isfinite(volume)]
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def choose_slices(gland: np.ndarray) -> List[int]:
    areas = gland.sum(axis=(1, 2))
    center = int(np.argmax(areas))
    nonempty = np.flatnonzero(areas > 0)
    if nonempty.size >= 3:
        quantile_indices = np.rint(
            np.quantile(nonempty, [0.25, 0.50, 0.75])
        ).astype(int)
        return [int(v) for v in quantile_indices]
    return [max(0, center - 2), center, min(gland.shape[0] - 1, center + 2)]


def render_case_figure(
    case_dir: Path,
    pair_rows: pd.DataFrame,
    output_path: Path,
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent.parent / ".mplconfig"))
    import matplotlib.pyplot as plt

    tensor = load_input_tensor(case_dir)
    gland = load_gland_mask(case_dir, tuple(tensor.shape[1:]))
    roi = dilate_mask(gland, 3)
    slices = choose_slices(gland)

    row_by_pair = {row["pair"]: row for _, row in pair_rows.iterrows()}
    dwi_row = row_by_pair.get("T2-DWI")
    adc_row = row_by_pair.get("T2-ADC")
    dwi_shift = (
        int(dwi_row["best_shift_z"]),
        int(dwi_row["best_shift_y"]),
        int(dwi_row["best_shift_x"]),
    )
    adc_shift = (
        int(adc_row["best_shift_z"]),
        int(adc_row["best_shift_y"]),
        int(adc_row["best_shift_x"]),
    )
    volumes = [
        ("T2 reference", tensor[0]),
        ("DWI stored", tensor[1]),
        (f"DWI shifted {dwi_shift}", integer_shift_for_display(tensor[1], dwi_shift)),
        ("ADC stored", tensor[2]),
        (f"ADC shifted {adc_shift}", integer_shift_for_display(tensor[2], adc_shift)),
    ]

    fig, axes = plt.subplots(len(volumes), len(slices), figsize=(11, 14), squeeze=False)
    for row_index, (row_name, volume) in enumerate(volumes):
        vmin, vmax = display_limits(volume, roi)
        for col_index, z_index in enumerate(slices):
            axis = axes[row_index, col_index]
            axis.imshow(volume[z_index], cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
            if gland[z_index].any():
                axis.contour(
                    gland[z_index].astype(np.uint8),
                    levels=[0.5],
                    colors=["#FFD400"],
                    linewidths=0.8,
                    origin="lower",
                )
            if row_index == 0:
                axis.set_title(f"axial z={z_index}")
            if col_index == 0:
                axis.set_ylabel(row_name)
            axis.set_xticks([])
            axis.set_yticks([])

    dwi_text = (
        f"T2-DWI: shift={dwi_shift}, NMI gain={float(dwi_row['nmi_gain']):.4f}, "
        f"status={dwi_row['status']}"
    )
    adc_text = (
        f"T2-ADC: shift={adc_shift}, NMI gain={float(adc_row['nmi_gain']):.4f}, "
        f"status={adc_row['status']}"
    )
    fig.suptitle(f"{case_dir.name}\n{dwi_text}\n{adc_text}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_case_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for patient_id, group in pair_df.groupby("patient_id", sort=True):
        statuses = group["status"].astype(str).tolist()
        if any(status == "ERROR" for status in statuses):
            case_status = "ERROR"
        elif any(status.startswith("SUSPECT") for status in statuses):
            case_status = "SUSPECT"
        else:
            case_status = "PASS"
        numeric_gain = pd.to_numeric(group.get("nmi_gain"), errors="coerce")
        numeric_shift = pd.to_numeric(group.get("max_abs_shift"), errors="coerce")
        records.append(
            {
                "patient_id": patient_id,
                "source": group["source"].iloc[0],
                "status": case_status,
                "max_nmi_gain": float(numeric_gain.max()) if numeric_gain.notna().any() else np.nan,
                "max_abs_shift": float(numeric_shift.max()) if numeric_shift.notna().any() else np.nan,
                "pair_statuses": ";".join(
                    f"{row['pair']}={row['status']}" for _, row in group.iterrows()
                ),
                "error": ";".join(
                    str(value) for value in group.get("error", []) if str(value)
                ),
                "case_dir": group["case_dir"].iloc[0],
            }
        )
    return pd.DataFrame.from_records(records)


def make_summary_text(pair_df: pd.DataFrame, case_df: pd.DataFrame, args: argparse.Namespace) -> str:
    lines = [
        "Input tensor alignment audit",
        "============================",
        f"Dataset: {args.dataset_root}",
        f"Cases: {len(case_df)}",
        f"Threshold: max-axis shift >= {args.flag_shift} voxels AND NMI gain >= {args.min_nmi_gain}",
        f"Orientation threshold: NMI gain >= {args.min_orientation_gain}",
        "",
        "Case status counts:",
        case_df.groupby(["source", "status"]).size().to_string(),
        "",
        "Pair diagnostic distributions (valid rows):",
    ]
    valid = pair_df[pair_df["pair"].isin([spec[2] for spec in PAIR_SPECS])].copy()
    for column in ("max_abs_shift", "nmi_identity", "nmi_gain", "orientation_gain"):
        valid[column] = pd.to_numeric(valid[column], errors="coerce")
    if valid.empty:
        lines.append("No valid modality-pair results.")
    else:
        stats = valid.groupby(["source", "pair"])[
            ["max_abs_shift", "nmi_identity", "nmi_gain", "orientation_gain"]
        ].agg(["median", "mean", "max"])
        lines.append(stats.to_string())
    lines.extend(
        [
            "",
            "Interpretation:",
            "- PASS means this screen found no better small translation/orientation hypothesis.",
            "- SUSPECT means a shifted/flipped hypothesis improved NMI enough to require visual review.",
            "- This content-based test does not certify deformable or sub-voxel alignment.",
            "- Positive shifts are the integer (z,y,x) shift applied to the moving DWI/ADC volume.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_self_test() -> None:
    rng = np.random.default_rng(42)
    fixed = ndimage.gaussian_filter(rng.normal(size=(24, 32, 40)), sigma=2.0)
    injected_shift = (2, -3, 4)
    moving = ndimage.shift(
        fixed,
        shift=injected_shift,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    expected_correction = tuple(-v for v in injected_shift)
    candidates, _ = phase_shift_candidates(fixed, moving, max_shift=5, top_k=5)
    if expected_correction not in candidates:
        raise AssertionError(
            f"phase test failed: expected {expected_correction}, got {candidates}"
        )
    print(
        "Self-test passed: injected moving shift",
        injected_shift,
        "requires correction",
        expected_correction,
    )


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if args.max_shift < 0:
        raise ValueError("--max-shift must be non-negative")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {args.dataset_root}")

    case_dirs = discover_cases(args.dataset_root, args.sources)
    if args.limit is not None:
        case_dirs = case_dirs[: args.limit]
    if not case_dirs:
        raise RuntimeError("no input_tensor.npy cases matched the requested sources")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".mplconfig"))
    config: Dict[str, float | int] = {
        "max_shift": args.max_shift,
        "phase_candidates": args.phase_candidates,
        "nmi_bins": args.nmi_bins,
        "gland_dilation": args.gland_dilation,
        "flag_shift": args.flag_shift,
        "min_nmi_gain": args.min_nmi_gain,
        "min_orientation_gain": args.min_orientation_gain,
        "min_roi_voxels": args.min_roi_voxels,
    }
    tasks = [(str(case_dir), config) for case_dir in case_dirs]
    print(f"Auditing {len(tasks)} cases with {args.workers} worker(s)...")

    all_rows: List[Dict[str, object]] = []
    if args.workers == 1:
        iterator: Iterable[List[Dict[str, object]]] = map(analyse_case, tasks)
        for index, rows in enumerate(iterator, start=1):
            all_rows.extend(rows)
            if index % 50 == 0 or index == len(tasks):
                print(f"  processed {index}/{len(tasks)}")
    else:
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                for index, rows in enumerate(
                    executor.map(analyse_case, tasks, chunksize=4), start=1
                ):
                    all_rows.extend(rows)
                    if index % 50 == 0 or index == len(tasks):
                        print(f"  processed {index}/{len(tasks)}")
        except (OSError, PermissionError) as exc:
            # Some managed/sandboxed systems disable POSIX semaphores required
            # by ProcessPoolExecutor. Keep the script usable by falling back to
            # deterministic single-process execution.
            print(f"  multiprocessing unavailable ({exc}); falling back to 1 worker")
            all_rows.clear()
            for index, rows in enumerate(map(analyse_case, tasks), start=1):
                all_rows.extend(rows)
                if index % 50 == 0 or index == len(tasks):
                    print(f"  processed {index}/{len(tasks)}")

    pair_df = pd.DataFrame.from_records(all_rows)
    pair_df = pair_df.sort_values(["source", "patient_id", "pair"]).reset_index(drop=True)
    case_df = build_case_summary(pair_df)
    pair_csv = args.output_dir / "alignment_pairs.csv"
    case_csv = args.output_dir / "alignment_cases.csv"
    summary_path = args.output_dir / "summary.txt"
    # Keep absolute paths in memory for optional rendering, but do not persist
    # workstation or storage paths in the exported audit tables.
    pair_df.drop(columns=["case_dir"], errors="ignore").to_csv(pair_csv, index=False)
    case_df.drop(columns=["case_dir"], errors="ignore").to_csv(case_csv, index=False)
    summary_text = make_summary_text(pair_df, case_df, args)
    summary_path.write_text(summary_text, encoding="utf-8")

    if args.visuals != "none" and args.visual_count > 0:
        if args.visuals == "suspect":
            selected = case_df[case_df["status"] == "SUSPECT"]
        else:
            selected = case_df[case_df["status"] != "ERROR"].sort_values(
                "max_nmi_gain", ascending=False
            )
        selected = selected.head(args.visual_count)
        visual_dir = args.output_dir / "visuals"
        for _, case_row in selected.iterrows():
            patient_id = str(case_row["patient_id"])
            rows = pair_df[pair_df["patient_id"] == patient_id]
            if set(rows["pair"]) >= {"T2-DWI", "T2-ADC"}:
                try:
                    render_case_figure(
                        Path(str(case_row["case_dir"])),
                        rows,
                        visual_dir / f"{patient_id}.png",
                    )
                except Exception as exc:
                    print(f"  warning: could not render {patient_id}: {exc}")

    print(summary_text)
    print(f"Pair results : {pair_csv}")
    print(f"Case results : {case_csv}")
    print(f"Summary      : {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
