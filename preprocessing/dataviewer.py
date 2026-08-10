import argparse
import gzip
import os
import struct
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    import SimpleITK as sitk
except ImportError:  # Keep npy viewing usable on environments without SimpleITK.
    sitk = None


class VolumeViewer:
    def __init__(self, npy_path):
        self.data = np.load(npy_path)
        print(f"Raw loaded data shape: {self.data.shape}")

        if self.data.ndim == 3:
            self.data = self.data[np.newaxis, ...]
        elif self.data.ndim != 4:
            raise ValueError(
                f"Expected 3D (D,H,W) or 4D (C,D,H,W) data, got {self.data.shape}"
            )

        self.num_channels = self.data.shape[0]
        self.max_slices = self.data.shape[1]
        self.slice_idx = self.max_slices // 2

        if self.num_channels == 3:
            self.channels = ["T2", "DWI", "ADC"]
        elif self.num_channels == 1:
            self.channels = ["Label / Mask"]
        else:
            self.channels = [f"Channel {i}" for i in range(self.num_channels)]

        self.fig, axes = plt.subplots(
            1, self.num_channels, figsize=(5 * self.num_channels, 5)
        )
        self.fig.canvas.manager.set_window_title(
            f"Deep Viewer - Slice {self.slice_idx}"
        )

        self.axes = [axes] if self.num_channels == 1 else axes
        self.images = []
        for i in range(self.num_channels):
            img_slice = self.data[i, self.slice_idx, :, :]
            img = self.axes[i].imshow(img_slice, cmap="gray")
            self.axes[i].set_title(self.channels[i])
            self.axes[i].axis("off")
            self.images.append(img)

        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        plt.tight_layout()
        print(f"Viewer data shape: {self.data.shape}. Scroll mouse wheel for slices.")
        plt.show()

    def update_display(self):
        for i in range(self.num_channels):
            self.images[i].set_data(self.data[i, self.slice_idx, :, :])

        self.fig.canvas.manager.set_window_title(
            f"Slice {self.slice_idx} / {self.max_slices - 1}"
        )
        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        if event.button == "up":
            self.slice_idx = min(self.slice_idx + 1, self.max_slices - 1)
        elif event.button == "down":
            self.slice_idx = max(self.slice_idx - 1, 0)

        self.update_display()


def npy_viewer(npy_path):
    data = np.load(npy_path)
    print("\n--- NPY overview ---")
    print("Shape:", data.shape)
    print("Dtype:", data.dtype)

    if data.ndim == 1:
        print("\n--- Label details ---")
        for i, isup in enumerate(data):
            print(f"Region {i + 1}: {isup}")
    else:
        print("This is image-like array data. Use VolumeViewer for visual inspection.")


def _print_value_counts(name: str, values: np.ndarray, max_rows: int = 30):
    flat = values.reshape(-1)
    unique_vals, counts = np.unique(flat, return_counts=True)
    print(f"\n--- {name} value counts ---")
    print(f"Voxel count: {flat.size}")
    print(f"Unique value count: {len(unique_vals)}")

    for value, count in zip(unique_vals[:max_rows], counts[:max_rows]):
        pct = 100.0 * float(count) / float(flat.size) if flat.size else 0.0
        print(f"  {value!r}: {int(count)} ({pct:.4f}%)")

    if len(unique_vals) > max_rows:
        print(f"  ... truncated {len(unique_vals) - max_rows} more unique values")


def _load_nii_array(path: str) -> np.ndarray:
    if sitk is None:
        print("\n--- NIfTI metadata ---")
        print("Path:", path)
        print("Reader: raw NIfTI fallback (SimpleITK not installed)")
        return _load_nii_array_raw(path)
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    print("\n--- NIfTI metadata ---")
    print("Path:", path)
    print("SimpleITK pixel type:", img.GetPixelIDTypeAsString())
    print("Array shape (z,y,x):", arr.shape)
    print("Array dtype:", arr.dtype)
    print("Image size (x,y,z):", img.GetSize())
    print("Spacing:", img.GetSpacing())
    print("Origin:", img.GetOrigin())
    return arr

def _load_nii_array_raw(path: str) -> np.ndarray:
    """Small NIfTI-1 reader for exact value QA when SimpleITK is unavailable."""
    with (gzip.open(path, "rb") if path.lower().endswith(".gz") else open(path, "rb")) as f:
        data = f.read()

    if len(data) < 352:
        raise ValueError(f"File is too small to be a NIfTI-1 image: {path}")

    sizeof_hdr_le = struct.unpack("<i", data[:4])[0]
    if sizeof_hdr_le == 348:
        endian = "<"
    elif struct.unpack(">i", data[:4])[0] == 348:
        endian = ">"
    else:
        raise ValueError("Unsupported NIfTI header: sizeof_hdr is not 348")

    dims = struct.unpack(endian + "8h", data[40:56])
    ndim = int(dims[0])
    shape_xyz = tuple(int(v) for v in dims[1 : 1 + ndim] if int(v) > 0)
    datatype = struct.unpack(endian + "h", data[70:72])[0]
    bitpix = struct.unpack(endian + "h", data[72:74])[0]
    pixdim = struct.unpack(endian + "8f", data[76:108])
    vox_offset = int(round(struct.unpack(endian + "f", data[108:112])[0]))
    scl_slope = struct.unpack(endian + "f", data[112:116])[0]
    scl_inter = struct.unpack(endian + "f", data[116:120])[0]
    qoffset = struct.unpack(endian + "3f", data[268:280])

    dtype_map = {
        2: "u1",
        4: "i2",
        8: "i4",
        16: "f4",
        64: "f8",
        256: "i1",
        512: "u2",
        768: "u4",
        1024: "i8",
        1280: "u8",
    }
    if datatype not in dtype_map:
        raise ValueError(f"Unsupported NIfTI datatype code: {datatype}")

    dtype = np.dtype(endian + dtype_map[datatype])
    n_voxels = int(np.prod(shape_xyz))
    arr = np.frombuffer(data, dtype=dtype, count=n_voxels, offset=vox_offset)
    arr = arr.reshape(shape_xyz[::-1])

    if scl_slope not in (0.0, 1.0):
        arr = arr.astype(np.float32) * scl_slope + scl_inter
    elif scl_inter != 0.0:
        arr = arr.astype(np.float32) + scl_inter

    print("Raw NIfTI datatype code:", datatype)
    print("Raw bitpix:", bitpix)
    print("Array shape (z,y,x):", arr.shape)
    print("Array dtype:", arr.dtype)
    print("Image size (x,y,z...):", shape_xyz)
    print("Spacing:", tuple(float(v) for v in pixdim[1 : 1 + ndim]))
    print("qoffset approx:", qoffset)
    return arr

def inspect_target_mask(
    target_mask_path: str,
    gland_mask_path: Optional[str] = None,
    zones_mask_path: Optional[str] = None,
):
    """Inspect target_mask values and optional gland/zones split checks.

    Project convention reminder:
      - target_mask == 0 means background / unlabeled voxels.
      - target_mask >= 2 means cancer label inside a TBx-confirmed target ROI.
      - -1 is not currently used by the target_mask loss path; invalid/unsampled
        regions are represented by sys_labels == -1 for SBx supervision.
    """
    if not os.path.exists(target_mask_path):
        raise FileNotFoundError(target_mask_path)

    arr = _load_nii_array(target_mask_path)
    _print_value_counts("whole target_mask", arr)

    print("\n--- Direct checks ---")
    print("Contains 0:", bool(np.any(arr == 0)))
    print("Contains -1:", bool(np.any(arr == -1)))
    print("Contains positive labels:", bool(np.any(arr > 0)))
    print("Minimum value:", arr.min() if arr.size else "empty")
    print("Maximum value:", arr.max() if arr.size else "empty")

    if gland_mask_path is None:
        candidate = os.path.join(os.path.dirname(target_mask_path), "gland_mask.nii.gz")
        gland_mask_path = candidate if os.path.exists(candidate) else None

    if zones_mask_path is None:
        candidate = os.path.join(os.path.dirname(target_mask_path), "zones_mask.nii.gz")
        zones_mask_path = candidate if os.path.exists(candidate) else None

    if gland_mask_path:
        gland = _load_nii_array(gland_mask_path)
        if gland.shape != arr.shape:
            print("\n[Warning] gland_mask shape does not match target_mask.")
        else:
            gland_fg = gland > 0
            _print_value_counts("target_mask outside gland (gland==0)", arr[~gland_fg])
            _print_value_counts("target_mask inside gland (gland>0)", arr[gland_fg])

    if zones_mask_path:
        zones = _load_nii_array(zones_mask_path)
        if zones.shape != arr.shape:
            print("\n[Warning] zones_mask shape does not match target_mask.")
        else:
            sampled = zones > 0
            _print_value_counts("target_mask outside sampled zones (zones==0)", arr[~sampled])
            _print_value_counts("target_mask inside sampled zones (zones>0)", arr[sampled])

    print("\n--- Interpretation note ---")
    print(
        "If ITK-SNAP shows everything as 0, this file may genuinely contain no -1. "
        "For this project, unsampled SBx regions are stored in systematic_labels_12/20.npy "
        "as -1, not necessarily in target_mask.nii.gz."
    )


def main():
    parser = argparse.ArgumentParser(description="View NPY files or inspect NIfTI masks.")
    parser.add_argument(
        "path",
        help="Path to .npy, .nii, or .nii.gz file.",
    )
    parser.add_argument(
        "--inspect-mask",
        action="store_true",
        help="Print exact NIfTI mask values instead of opening the image viewer.",
    )
    parser.add_argument("--gland-mask", default=None, help="Optional gland_mask.nii.gz.")
    parser.add_argument("--zones-mask", default=None, help="Optional zones_mask.nii.gz.")
    args = parser.parse_args()

    lower_path = args.path.lower()
    if args.inspect_mask or lower_path.endswith(".nii") or lower_path.endswith(".nii.gz"):
        inspect_target_mask(args.path, args.gland_mask, args.zones_mask)
    elif lower_path.endswith(".npy"):
        data = np.load(args.path)
        if data.ndim in (3, 4):
            VolumeViewer(args.path)
        else:
            npy_viewer(args.path)
    else:
        raise ValueError(f"Unsupported file type: {args.path}")


if __name__ == "__main__":
    main()
