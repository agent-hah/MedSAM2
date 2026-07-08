"""
Preprocess RatData DICOM files into MedSAM2-compatible NPZ format.

Handles three DICOM storage patterns:
  1. Multi-frame DICOMs (Cios/Pheno/Wistar) — single large .dcm with NumberOfFrames
  2. Single-frame series (DCE) — many individual .dcm files in one directory
  3. Zipped DICOMs — .zip files containing DICOMs inside

Usage:
    python preprocess_ratdata.py \
        --input_dir data/medsam_preprocessed/RatData/RatData \
        --output_dir data/medsam_preprocessed/RatData_NPZ \
        --image_size 512
"""

import os
import re
import glob
import argparse
import zipfile
import tempfile
from collections import defaultdict

import numpy as np
import cv2
import pydicom
from tqdm import tqdm


# ===========================================================================
# Utility helpers
# ===========================================================================

def sanitize_name(name: str) -> str:
    """Replace spaces, dashes, and other non-alphanumeric chars with underscores,
    then collapse multiple underscores and strip leading/trailing ones."""
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def derive_npz_name_from_path(rel_path: str, stem: str = "") -> str:
    """Build a descriptive NPZ filename from a path relative to the RatData root.

    Examples:
        ('SRG - Angio/SRG 1 - DCE/FILES/S1 DCE 5Jun22', '')
            -> 'SRG_Angio__SRG_1_DCE__S1_DCE_5Jun22.npz'
        ('Wistar_Angio/rat id 636 - selected/tumor A', 'IM-0001-0012')
            -> 'Wistar_Angio__rat_id_636_selected__tumor_A__IM_0001_0012.npz'
    """
    parts = rel_path.split(os.sep)
    sanitized_parts = [sanitize_name(p) for p in parts if p]
    # Drop generic intermediate directory names that add no information
    skip_names = {"FILES", "ACQUISITIONS", "DSA_angio"}
    sanitized_parts = [p for p in sanitized_parts if p not in skip_names]
    if stem:
        sanitized_parts.append(sanitize_name(stem))
    return "__".join(sanitized_parts) + ".npz"


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Min-max normalize an arbitrary-dtype array to uint8 [0, 255]."""
    arr = array.astype(np.float64)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max == arr_min:
        return np.zeros(array.shape, dtype=np.uint8)
    normalized = ((arr - arr_min) / (arr_max - arr_min) * 255.0)
    return normalized.astype(np.uint8)


def resize_volume(volume_3d: np.ndarray, image_size: int) -> np.ndarray:
    """Resize each frame in a (D, H, W) volume to (image_size, image_size)."""
    D = volume_3d.shape[0]
    resized = np.zeros((D, image_size, image_size), dtype=np.uint8)
    for i in range(D):
        resized[i] = cv2.resize(
            volume_3d[i], (image_size, image_size),
            interpolation=cv2.INTER_LINEAR,
        )
    return resized


# ===========================================================================
# DICOM reading
# ===========================================================================

def read_dicom_pixel_array(dcm_path: str) -> np.ndarray:
    """Read a DICOM file and return its pixel array.

    Returns
    -------
    np.ndarray
        Shape (D, H, W) for multi-frame or (H, W) for single-frame.
    """
    ds = pydicom.dcmread(dcm_path)
    pixel_array = ds.pixel_array  # pydicom handles decompression
    return pixel_array


def is_multiframe(dcm_path: str) -> bool:
    """Check whether a DICOM file is multi-frame without reading pixel data."""
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    n_frames = getattr(ds, "NumberOfFrames", 1)
    return int(n_frames) > 1


# ===========================================================================
# Processing functions
# ===========================================================================

def process_multiframe_dicom(dcm_path: str, image_size: int) -> np.ndarray:
    """Process a single multi-frame DICOM into a (D, image_size, image_size) uint8 volume."""
    pixel_array = read_dicom_pixel_array(dcm_path)

    # Handle potential (D, H, W, C) colour DICOMs by taking luminance
    if pixel_array.ndim == 4:
        # Convert RGB to grayscale
        pixel_array = np.mean(pixel_array, axis=-1)

    # Ensure 3D: (D, H, W)
    if pixel_array.ndim == 2:
        pixel_array = pixel_array[np.newaxis, :, :]

    volume = normalize_to_uint8(pixel_array)
    volume = resize_volume(volume, image_size)
    return volume


def process_singleframe_directory(dcm_paths: list, image_size: int) -> np.ndarray:
    """Process a directory of single-frame DICOMs into a (D, image_size, image_size) uint8 volume.

    Files are sorted by filename so frames are in the correct temporal order.
    """
    dcm_paths = sorted(dcm_paths)
    frames = []
    for path in dcm_paths:
        try:
            pixel_array = read_dicom_pixel_array(path)
            if pixel_array.ndim == 3:
                # Unexpected multi-frame in a "single-frame" directory — take first frame
                pixel_array = pixel_array[0]
            if pixel_array.ndim == 3 and pixel_array.shape[-1] in (3, 4):
                # Colour image → grayscale
                pixel_array = np.mean(pixel_array, axis=-1)
            frames.append(pixel_array)
        except Exception as e:
            print(f"  ⚠ Skipping {os.path.basename(path)}: {e}")

    if not frames:
        return None

    volume = np.stack(frames, axis=0)
    volume = normalize_to_uint8(volume)
    volume = resize_volume(volume, image_size)
    return volume


def extract_and_find_dcms(zip_path: str, tmp_dir: str) -> list:
    """Extract a zip file and return a list of .dcm file paths found inside."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)
    dcm_files = glob.glob(os.path.join(tmp_dir, "**", "*.dcm"), recursive=True)
    return sorted(dcm_files)


# ===========================================================================
# Main pipeline
# ===========================================================================

def preprocess_ratdata(input_dir: str, output_dir: str, image_size: int):
    """Walk the RatData directory tree, convert all DICOMs to NPZ files."""
    os.makedirs(output_dir, exist_ok=True)

    # ── Phase 1: Discover all DICOM files and zip files ──────────────────
    print(f"Scanning {input_dir} for DICOM and ZIP files...")
    dcm_by_dir = defaultdict(list)       # dir_path → list of .dcm paths
    zip_files = []                        # list of .zip paths

    for root, dirs, files in os.walk(input_dir):
        for f in files:
            full_path = os.path.join(root, f)
            if f.lower().endswith(".dcm"):
                dcm_by_dir[root].append(full_path)
            elif f.lower().endswith(".zip"):
                zip_files.append(full_path)

    total_dcm_dirs = len(dcm_by_dir)
    total_zips = len(zip_files)
    print(f"  Found {total_dcm_dirs} directories containing .dcm files")
    print(f"  Found {total_zips} .zip files")

    npz_created = 0
    errors = []

    # ── Phase 2: Process each DICOM directory ────────────────────────────
    print(f"\nProcessing DICOM directories...")
    
    # Pre-filter directories that have 'dsa' in path or filenames
    dsa_dirs = []
    for dir_path, files in dcm_by_dir.items():
        rel_dir = os.path.relpath(dir_path, input_dir)
        has_dsa = "dsa" in rel_dir.lower() or any("dsa" in os.path.basename(f).lower() for f in files)
        if has_dsa:
            dsa_dirs.append(dir_path)
            
    for dir_path in tqdm(sorted(dsa_dirs), desc="DICOM dirs (DSA only)"):
        dcm_files = dcm_by_dir[dir_path]
        rel_dir = os.path.relpath(dir_path, input_dir)

        try:
            # Decide: multi-frame or single-frame series?
            # Check the first DICOM to determine type
            first_dcm = sorted(dcm_files)[0]
            multiframe = is_multiframe(first_dcm)

            if multiframe:
                # Each multi-frame DICOM → one NPZ
                for dcm_path in sorted(dcm_files):
                    if not dcm_path.lower().endswith(".dcm"):
                        continue
                    
                    # If the directory doesn't have 'dsa', ensure this specific file does
                    if "dsa" not in rel_dir.lower() and "dsa" not in os.path.basename(dcm_path).lower():
                        continue

                    stem = os.path.splitext(os.path.basename(dcm_path))[0]
                    npz_name = derive_npz_name_from_path(rel_dir, stem)
                    out_path = os.path.join(output_dir, npz_name)

                    if os.path.exists(out_path):
                        print(f"  ⏭ Already exists: {npz_name}")
                        continue

                    volume = process_multiframe_dicom(dcm_path, image_size)
                    np.savez_compressed(out_path, imgs=volume)
                    npz_created += 1
            else:
                # All single-frame DICOMs in directory → one NPZ
                npz_name = derive_npz_name_from_path(rel_dir)
                out_path = os.path.join(output_dir, npz_name)

                if os.path.exists(out_path):
                    print(f"  ⏭ Already exists: {npz_name}")
                    continue

                volume = process_singleframe_directory(dcm_files, image_size)
                if volume is not None:
                    np.savez_compressed(out_path, imgs=volume)
                    npz_created += 1
                else:
                    errors.append(f"No valid frames in {rel_dir}")

        except Exception as e:
            err_msg = f"Error processing {rel_dir}: {e}"
            print(f"  ✗ {err_msg}")
            errors.append(err_msg)

    # ── Phase 3: Process zipped DICOMs ───────────────────────────────────
    if zip_files:
        # Pre-filter zip files
        dsa_zips = [z for z in zip_files if "dsa" in os.path.relpath(z, input_dir).lower()]
        
        if dsa_zips:
            print(f"\nProcessing {len(dsa_zips)} ZIP files...")
            for zip_path in tqdm(dsa_zips, desc="ZIP files (DSA only)"):
                rel_zip = os.path.relpath(zip_path, input_dir)

                zip_stem = os.path.splitext(os.path.basename(zip_path))[0]
                rel_dir = os.path.relpath(os.path.dirname(zip_path), input_dir)
                npz_name = derive_npz_name_from_path(rel_dir, zip_stem)
                out_path = os.path.join(output_dir, npz_name)

                if os.path.exists(out_path):
                    print(f"  ⏭ Already exists: {npz_name}")
                    continue

                try:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        dcm_files = extract_and_find_dcms(zip_path, tmp_dir)
                        if not dcm_files:
                            errors.append(f"No .dcm files in {rel_zip}")
                            continue

                        # Check if multi-frame or single-frame
                        if is_multiframe(dcm_files[0]):
                            # Process each extracted multi-frame DICOM
                            for dcm_path in dcm_files:
                                stem = os.path.splitext(os.path.basename(dcm_path))[0]
                                sub_npz_name = derive_npz_name_from_path(rel_dir, f"{zip_stem}__{stem}")
                                sub_out_path = os.path.join(output_dir, sub_npz_name)
                                volume = process_multiframe_dicom(dcm_path, image_size)
                                np.savez_compressed(sub_out_path, imgs=volume)
                                npz_created += 1
                        else:
                            # All extracted DICOMs → one NPZ
                            volume = process_singleframe_directory(dcm_files, image_size)
                            if volume is not None:
                                np.savez_compressed(out_path, imgs=volume)
                                npz_created += 1
                            else:
                                errors.append(f"No valid frames in {rel_zip}")

                except Exception as e:
                    err_msg = f"Error processing ZIP {rel_zip}: {e}"
                    print(f"  ✗ {err_msg}")
                    errors.append(err_msg)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("       RATDATA PREPROCESSING SUMMARY       ")
    print("=" * 55)
    print(f"  Input directory  : {input_dir}")
    print(f"  Output directory : {output_dir}")
    print(f"  Image size       : {image_size}×{image_size}")
    print(f"  DICOM directories: {total_dcm_dirs}")
    print(f"  ZIP files        : {total_zips}")
    print(f"  NPZ files created: {npz_created}")
    if errors:
        print(f"  Errors / skips   : {len(errors)}")
        for err in errors:
            print(f"    • {err}")
    else:
        print(f"  Errors / skips   : 0")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert RatData DICOMs to MedSAM2-compatible NPZ files"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.path.join("data", "medsam_preprocessed", "RatData", "RatData"),
        help="Path to the root RatData directory containing SRG and Wistar subdirectories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("data", "medsam_preprocessed", "RatData_NPZ"),
        help="Path to the output directory for NPZ files",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Target spatial resolution (default: 512)",
    )

    args = parser.parse_args()
    preprocess_ratdata(args.input_dir, args.output_dir, args.image_size)
