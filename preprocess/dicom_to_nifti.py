"""
Preprocess DICOM files into raw NIfTI (.nii.gz) format.
Handles HumanData and RatData.

Usage:
    python run_preprocessing.py dicom2nifti --dataset rat|human \
        --input_dir <path> --output_dir <path>
"""

import os
import re
import glob
import argparse
import zipfile
import tempfile
from collections import defaultdict

import numpy as np
import pydicom
import nibabel as nib
from tqdm import tqdm

# ===========================================================================
# Utility helpers
# ===========================================================================

def is_dicom_file(filepath: str, check_magic: bool = True) -> bool:
    """Check if a file is a DICOM file by checking extension or magic bytes."""
    if filepath.lower().endswith(".dcm"):
        return True
    if not check_magic:
        return False
    if filepath.endswith(".Identifier") or filepath.endswith(".JPG") or filepath.endswith(".zip") or filepath.endswith(".npz") or filepath.endswith(".nii.gz"):
        return False
    try:
        with open(filepath, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False

def sanitize_name(name: str) -> str:
    """Replace spaces, dashes, and other non-alphanumeric chars with underscores."""
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")

def derive_nifti_name_from_path(rel_path: str, stem: str = "") -> str:
    """Build a descriptive NIfTI filename from a path."""
    parts = rel_path.split(os.sep)
    sanitized_parts = [sanitize_name(p) for p in parts if p]
    skip_names = {"FILES", "ACQUISITIONS", "DSA_angio"}
    sanitized_parts = [p for p in sanitized_parts if p not in skip_names]
    if stem:
        sanitized_parts.append(sanitize_name(stem))
    base = "__".join(sanitized_parts)
    return base + "_image.nii.gz"

# ===========================================================================
# DICOM reading
# ===========================================================================

def read_dicom_pixel_array(dcm_path: str):
    """Read a DICOM file and return its pixel array and dataset."""
    ds = pydicom.dcmread(dcm_path)
    pixel_array = ds.pixel_array
    return pixel_array, ds

def is_multiframe(dcm_path: str) -> bool:
    """Check whether a DICOM file is multi-frame without reading pixel data."""
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    n_frames = getattr(ds, "NumberOfFrames", 1)
    return int(n_frames) > 1

def get_affine(ds):
    """
    Get affine matrix from DICOM dataset.
    Uses PixelSpacing or ImagerPixelSpacing if available, otherwise defaults to 1.0mm.
    """
    dy, dx = 1.0, 1.0
    if hasattr(ds, "PixelSpacing"):
        dy, dx = map(float, ds.PixelSpacing)
    elif hasattr(ds, "ImagerPixelSpacing"):
        dy, dx = map(float, ds.ImagerPixelSpacing)
    
    # We create an affine where Z spacing is 1.0 (temporal axis)
    # Nibabel uses (X, Y, Z) ordering. We'll map (X=col spacing, Y=row spacing)
    affine = np.diag([dx, dy, 1.0, 1.0])
    return affine

def save_nifti(volume, affine, out_path):
    """
    Save volume as NIfTI.
    volume is expected to be in (D, H, W) order.
    nibabel expects (X, Y, Z) which corresponds to (W, H, D).
    """
    volume_t = np.transpose(volume, (2, 1, 0))
    nifti_img = nib.Nifti1Image(volume_t, affine)
    nib.save(nifti_img, out_path)

# ===========================================================================
# Processing functions
# ===========================================================================

def process_multiframe_dicom(dcm_path: str, out_path: str, args):
    """Process a single multi-frame DICOM into a raw NIfTI."""
    pixel_array, ds = read_dicom_pixel_array(dcm_path)
    
    if pixel_array.ndim == 4:
        pixel_array = np.mean(pixel_array, axis=-1)
    
    if pixel_array.ndim == 2:
        pixel_array = pixel_array[np.newaxis, :, :]
        
    affine = get_affine(ds)
    save_nifti(pixel_array, affine, out_path)

def process_singleframe_directory(dcm_paths: list, out_path: str, args) -> bool:
    """Process a directory of single-frame DICOMs into a raw NIfTI."""
    def get_sort_key(path):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            inst = getattr(ds, "InstanceNumber", None)
            if inst is not None:
                return (int(inst), path)
        except Exception:
            pass
        match = re.search(r'(\d+)', os.path.basename(path))
        num = int(match.group(1)) if match else 0
        return (num, path)

    dcm_paths = sorted(dcm_paths, key=get_sort_key)
    frames = []
    ds_first = None
    
    for i, path in enumerate(dcm_paths):
        try:
            pixel_array, ds = read_dicom_pixel_array(path)
            if i == 0:
                ds_first = ds
            if pixel_array.ndim == 3:
                pixel_array = pixel_array[0]
            if pixel_array.ndim == 3 and pixel_array.shape[-1] in (3, 4):
                pixel_array = np.mean(pixel_array, axis=-1)
            frames.append(pixel_array)
        except Exception as e:
            print(f"  ⚠ Skipping {os.path.basename(path)}: {e}")

    if not frames:
        return False
        
    volume = np.stack(frames, axis=0)
    affine = get_affine(ds_first) if ds_first else np.eye(4)
    save_nifti(volume, affine, out_path)
    return True

def extract_and_find_dcms(zip_path: str, tmp_dir: str, check_magic: bool) -> list:
    """Extract a zip file and return a list of DICOM file paths found inside."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)
    dcm_files = []
    for root, _, files in os.walk(tmp_dir):
        for f in files:
            full_path = os.path.join(root, f)
            if is_dicom_file(full_path, check_magic):
                dcm_files.append(full_path)
    return sorted(dcm_files)

# ===========================================================================
# Main pipeline
# ===========================================================================

def preprocess_dicom_to_nifti(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    dataset = args.dataset
    dsa_only = not getattr(args, 'process_all', False)
    
    os.makedirs(output_dir, exist_ok=True)
    check_magic = (dataset == 'human')

    print(f"Scanning {input_dir} for DICOM and ZIP files...")
    dcm_by_dir = defaultdict(list)
    zip_files = []

    for root, dirs, files in os.walk(input_dir):
        for f in files:
            full_path = os.path.join(root, f)
            if f.lower().endswith(".zip"):
                zip_files.append(full_path)
            elif is_dicom_file(full_path, check_magic):
                dcm_by_dir[root].append(full_path)

    total_dcm_dirs = len(dcm_by_dir)
    total_zips = len(zip_files)
    print(f"  Found {total_dcm_dirs} directories containing DICOMs")
    print(f"  Found {total_zips} ZIP files")

    nii_created = 0
    errors = []

    print("\nProcessing DICOM directories...")
    target_dirs = []
    for dir_path, files in dcm_by_dir.items():
        rel_dir = os.path.relpath(dir_path, input_dir)
        has_dsa = "dsa" in rel_dir.lower() or any("dsa" in os.path.basename(f).lower() for f in files)
        
        if dataset == 'human' and dsa_only and not has_dsa and files:
            try:
                ds = pydicom.dcmread(files[0], stop_before_pixels=True)
                desc = getattr(ds, "SeriesDescription", "").lower()
                prot = getattr(ds, "ProtocolName", "").lower()
                if "dsa" in desc or "dsa" in prot:
                    has_dsa = True
            except Exception:
                pass
                
        if not dsa_only or has_dsa:
            target_dirs.append(dir_path)

    desc = "DICOM dirs (DSA only)" if dsa_only else "DICOM dirs"
    for dir_path in tqdm(sorted(target_dirs), desc=desc):
        dcm_files = dcm_by_dir[dir_path]
        rel_dir = os.path.relpath(dir_path, input_dir)

        try:
            first_dcm = sorted(dcm_files)[0]
            multiframe = is_multiframe(first_dcm)

            if multiframe:
                for dcm_path in sorted(dcm_files):
                    if dataset == 'rat' and not dcm_path.lower().endswith(".dcm"):
                        continue
                    
                    if dsa_only:
                        if dataset == 'human' and "dsa" not in rel_dir.lower() and "dsa" not in os.path.basename(dcm_path).lower():
                            try:
                                ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                                desc = getattr(ds, "SeriesDescription", "").lower()
                                prot = getattr(ds, "ProtocolName", "").lower()
                                if "dsa" not in desc and "dsa" not in prot:
                                    continue
                            except Exception:
                                continue
                        elif dataset == 'rat' and "dsa" not in rel_dir.lower() and "dsa" not in os.path.basename(dcm_path).lower():
                            continue

                    stem = os.path.splitext(os.path.basename(dcm_path))[0]
                    img_name = derive_nifti_name_from_path(rel_dir, stem)
                    out_path = os.path.join(output_dir, img_name)

                    if not getattr(args, 'overwrite', False) and os.path.exists(out_path):
                        print(f"  ⏭ Already exists: {img_name}")
                        continue

                    process_multiframe_dicom(dcm_path, out_path, args)
                    nii_created += 1
            else:
                img_name = derive_nifti_name_from_path(rel_dir)
                out_path = os.path.join(output_dir, img_name)

                if not getattr(args, 'overwrite', False) and os.path.exists(out_path):
                    print(f"  ⏭ Already exists: {img_name}")
                    continue

                if process_singleframe_directory(dcm_files, out_path, args):
                    nii_created += 1
                else:
                    errors.append(f"No valid frames in {rel_dir}")

        except Exception as e:
            err_msg = f"Error processing {rel_dir}: {e}"
            print(f"  ✗ {err_msg}")
            errors.append(err_msg)

    if zip_files:
        target_zips = zip_files
        if dsa_only:
            target_zips = [z for z in zip_files if "dsa" in os.path.relpath(z, input_dir).lower()]
        
        if target_zips:
            print(f"\nProcessing {len(target_zips)} ZIP files...")
            desc_zip = "ZIP files (DSA only)" if dsa_only else "ZIP files"
            for zip_path in tqdm(target_zips, desc=desc_zip):
                rel_zip = os.path.relpath(zip_path, input_dir)
                zip_stem = os.path.splitext(os.path.basename(zip_path))[0]
                rel_dir = os.path.relpath(os.path.dirname(zip_path), input_dir)
                img_name = derive_nifti_name_from_path(rel_dir, zip_stem)
                out_path = os.path.join(output_dir, img_name)

                if not getattr(args, 'overwrite', False) and os.path.exists(out_path):
                    print(f"  ⏭ Already exists: {img_name}")
                    continue

                try:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        dcm_files = extract_and_find_dcms(zip_path, tmp_dir, check_magic)
                        if not dcm_files:
                            errors.append(f"No DICOM files in {rel_zip}")
                            continue

                        if is_multiframe(dcm_files[0]):
                            for dcm_path in dcm_files:
                                stem = os.path.splitext(os.path.basename(dcm_path))[0]
                                sub_img_name = derive_nifti_name_from_path(rel_dir, f"{zip_stem}____{stem}")
                                sub_out_path = os.path.join(output_dir, sub_img_name)
                                process_multiframe_dicom(dcm_path, sub_out_path, args)
                                nii_created += 1
                        else:
                            if process_singleframe_directory(dcm_files, out_path, args):
                                nii_created += 1
                            else:
                                errors.append(f"No valid frames in {rel_zip}")
                except Exception as e:
                    err_msg = f"Error processing ZIP {rel_zip}: {e}"
                    print(f"  ✗ {err_msg}")
                    errors.append(err_msg)

    print("\n" + "=" * 55)
    print("      DICOM TO NIFTI CONVERSION SUMMARY      ")
    print("=" * 55)
    print(f"  Dataset          : {dataset.capitalize()}Data")
    print(f"  Input directory  : {input_dir}")
    print(f"  Output directory : {output_dir}")
    print(f"  DSA only         : {dsa_only}")
    print(f"  DICOM directories: {total_dcm_dirs}")
    print(f"  ZIP files        : {total_zips}")
    print(f"  NIfTI files      : {nii_created}")
    if errors:
        print(f"  Errors / skips   : {len(errors)}")
        for err in errors:
            print(f"    • {err}")
    else:
        print(f"  Errors / skips   : 0")
    print("=" * 55)

parser = argparse.ArgumentParser(description="Convert DICOMs to raw NIfTI files")
parser.add_argument("--dataset", type=str, choices=["rat", "human"], required=True, help="Dataset type (rat or human)")
parser.add_argument("--input_dir", type=str, required=True, help="Path to input dataset directory")
parser.add_argument("--output_dir", type=str, required=True, help="Path to output NIfTI directory")
parser.add_argument("--process_all", action="store_true", help="Process all data instead of filtering for 'dsa' files")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing NIfTI files")

if __name__ == "__main__":
    args = parser.parse_args()
    preprocess_dicom_to_nifti(args)
