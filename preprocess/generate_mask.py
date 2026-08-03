"""
Generate hessian-based preliminary masks from raw NIfTI images.

Usage:
    python run_preprocessing.py generate-mask --input_dir <path> [--output_dir <path>]
"""

import os
import argparse
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm

try:
    from sporco.admm import tvl1
    SPORCO_AVAILABLE = True
except ImportError:
    SPORCO_AVAILABLE = False
    print("WARNING: sporco package not found. TVL1 denoising will be skipped.")

from gt_generation import hessian

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.preprocess import get_crop_roi

def tvl1_denoise_volume(volume3D, lmbda=0.6, eps=1e-8):
    """
    Denoise volume frame-by-frame using TV-L1.
    volume3D: (T, H, W)
    """
    if not SPORCO_AVAILABLE:
        return volume3D
        
    vol = np.asarray(volume3D, dtype=np.float32)
    T, H, W = vol.shape
    out = np.empty_like(vol, dtype=np.float32)

    for t in range(T):
        img = vol[t, :, :]
        vmin = float(img.min())
        vmax = float(img.max())
        if vmax - vmin < eps:
            out[t, :, :] = 0.0
            continue
            
        img01 = (img - vmin) / (vmax - vmin)
        solver = tvl1.TVL1Denoise(img01, lmbda=lmbda)
        out[t, :, :] = solver.solve().astype(np.float32) * (vmax - vmin) + vmin

    return out

def generate_masks(args):
    input_dir = args.input_dir
    output_dir = getattr(args, 'output_dir', input_dir)
    if output_dir is None:
        output_dir = input_dir
        
    os.makedirs(output_dir, exist_ok=True)
    
    nifti_files = glob.glob(os.path.join(input_dir, "**", "*_image.nii.gz"), recursive=True)
    print(f"Found {len(nifti_files)} raw NIfTI images for mask generation.")
    
    errors = []
    masks_created = 0
    
    for img_path in tqdm(nifti_files, desc="Generating masks"):
        rel_path = os.path.relpath(img_path, input_dir)
        mask_rel_path = rel_path.replace("_image.nii.gz", "_mask.nii.gz")
        out_path = os.path.join(output_dir, mask_rel_path)
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        if not getattr(args, 'overwrite', False) and os.path.exists(out_path):
            print(f"  ⏭ Already exists: {mask_rel_path}")
            continue
            
        try:
            nii = nib.load(img_path)
            # nibabel loads data as (X, Y, Z), which for our saving logic is (W, H, D).
            img_data = nii.get_fdata()
            # Transpose to (D, H, W) for processing
            volume = np.transpose(img_data, (2, 1, 0))
            
            # ROI Cropping
            seq_name = os.path.basename(img_path).replace("_image.nii.gz", "")
            roi = get_crop_roi(volume, seq_name, cache_file="crop_cache.json", skip_cropping=args.skip_cropping)
            x, y, w, h = roi
            
            cropped_volume = volume[:, y:y+h, x:x+w]
            
            # TVL1 Denoising
            if args.lambda_tvl1 > 0:
                cropped_volume = tvl1_denoise_volume(cropped_volume, lmbda=args.lambda_tvl1)
                
            # Hessian segmentation
            # Get spacing from affine. Assuming diagonal affine [dx, dy, dz, 1]
            affine = nii.affine
            dx = abs(affine[0, 0])
            dy = abs(affine[1, 1])
            spacing = (dy, dx)
            
            sigmas = np.arange(0.5, args.sigma_max, 0.3)
            _, _, V_seg3D_cropped, _ = hessian.segment_vessel_3d_sequence(
                cropped_volume, 
                sigmas=sigmas, 
                spacing=spacing, 
                tau=1.0, 
                bright_vessels=False,
                thresh_mode=args.thresh_mode, 
                rel_thresh=args.rel_thresh,
                selem_radius=1, 
                hole_area=4,
                keep_largest=not args.keep_all, 
                max_iterations=2
            )
            
            # Reconstruct full-size mask
            V_seg3D_full = np.zeros_like(volume, dtype=bool)
            V_seg3D_full[:, y:y+h, x:x+w] = V_seg3D_cropped
            
            # Transpose mask back to (W, H, D)
            mask_data = np.transpose(V_seg3D_full.astype(np.uint8), (2, 1, 0))
            
            # Save NIfTI mask with the EXACT same affine and header
            mask_nii = nib.Nifti1Image(mask_data, nii.affine, nii.header)
            nib.save(mask_nii, out_path)
            masks_created += 1
            
        except Exception as e:
            err_msg = f"Error processing {rel_path}: {e}"
            print(f"  ✗ {err_msg}")
            errors.append(err_msg)
            
    print("\n" + "=" * 55)
    print("        HESSIAN MASK GENERATION SUMMARY      ")
    print("=" * 55)
    print(f"  Input directory  : {input_dir}")
    print(f"  Output directory : {output_dir}")
    print(f"  Masks generated  : {masks_created}")
    if errors:
        print(f"  Errors / skips   : {len(errors)}")
        for err in errors:
            print(f"    • {err}")
    else:
        print(f"  Errors / skips   : 0")
    print("=" * 55)


parser = argparse.ArgumentParser(description="Generate hessian-based masks from raw NIfTI images")
parser.add_argument("--input_dir", type=str, required=True, help="Path to input raw NIfTI directory")
parser.add_argument("--output_dir", type=str, required=False, help="Path to output masks directory (defaults to input_dir)")
parser.add_argument("--lambda_tvl1", type=float, default=0.6, help="TVL1 denoising lambda parameter (0 to disable)")
parser.add_argument("--sigma_max", type=float, default=3.5, help="Maximum sigma for Hessian filtering")
parser.add_argument("--thresh_mode", type=str, choices=["otsu", "relative"], default="relative", help="Thresholding mode")
parser.add_argument("--rel_thresh", type=float, default=0.35, help="Relative threshold value")
parser.add_argument("--skip_cropping", action="store_true", help="Skip interactive ROI cropping")
parser.add_argument("--keep_all", action="store_true", help="Keep all vessel components instead of only the largest one")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing NIfTI masks")

if __name__ == "__main__":
    args, unknown = parser.parse_known_args()
    generate_masks(args)
