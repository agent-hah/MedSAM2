"""
Preprocess raw NIfTI images with destructive transforms (window/level, CLAHE, min-max).

Usage:
    python run_preprocessing.py preprocess-nifti --input_dir <path> --output_dir <path>
"""

import os
import argparse
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm
import sys
import shutil
import cv2

# Ensure utils is importable if run directly
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.preprocess import apply_full_preprocessing, get_crop_roi

def preprocess_nifti(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    nifti_files = glob.glob(os.path.join(input_dir, "**", "*_image.nii.gz"), recursive=True)
    print(f"Found {len(nifti_files)} raw NIfTI images for preprocessing.")
    
    errors = []
    processed = 0
    
    for img_path in tqdm(nifti_files, desc="Preprocessing NIfTI"):
        rel_path = os.path.relpath(img_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        if not getattr(args, 'overwrite', False) and os.path.exists(out_path):
            print(f"  ⏭ Already exists: {rel_path}")
            continue
            
        try:
            nii = nib.load(img_path)
            # nibabel loads data as (X, Y, Z), which corresponds to (W, H, D) for us.
            img_data = nii.get_fdata()
            
            # Transpose to (D, H, W) for get_crop_roi compatibility
            volume = np.transpose(img_data, (2, 1, 0))
            
            # Retrieve ROI from cache (avoid interactive cropping window)
            seq_name = os.path.basename(img_path).replace("_image.nii.gz", "")
            
            x, y, w, h = 0, 0, img_data.shape[0], img_data.shape[1]
            if os.path.exists("crop_cache.json"):
                import json
                with open("crop_cache.json", "r") as f:
                    cache = json.load(f)
                if seq_name in cache:
                    x, y, w, h = cache[seq_name]
                    
            # Crop the raw data (W, H, D)
            img_data_cropped = img_data[x:x+w, y:y+h, :]
            W_new, H_new, D_new = img_data_cropped.shape
            
            # To get dtype, we process one slice first
            sample_hw = img_data_cropped[:, :, 0].T
            sample_out = apply_full_preprocessing(
                sample_hw, 
                window_center=args.window_center, 
                window_width=args.window_width, 
                use_clahe=not args.no_clahe, 
                use_minmax=not args.no_minmax
            )
            sample_out_resized = cv2.resize(sample_out, (512, 512), interpolation=cv2.INTER_LINEAR)
            
            out_data = np.zeros((512, 512, D_new), dtype=sample_out_resized.dtype)
            out_data[:, :, 0] = sample_out_resized.T
            
            for d in range(1, D_new):
                slice_wh = img_data_cropped[:, :, d]
                slice_hw = slice_wh.T
                
                processed_hw = apply_full_preprocessing(
                    slice_hw, 
                    window_center=args.window_center, 
                    window_width=args.window_width, 
                    use_clahe=not args.no_clahe, 
                    use_minmax=not args.no_minmax
                )
                
                processed_resized = cv2.resize(processed_hw, (512, 512), interpolation=cv2.INTER_LINEAR)
                out_data[:, :, d] = processed_resized.T
                
            out_nii = nib.Nifti1Image(out_data, nii.affine, nii.header)
            nib.save(out_nii, out_path)
            processed += 1
            
            # Also crop and save the mask over if it exists, so the preprocessed folder has the complete cropped pair
            mask_in_path = img_path.replace("_image.nii.gz", "_mask.nii.gz")
            mask_out_path = out_path.replace("_image.nii.gz", "_mask.nii.gz")
            if os.path.exists(mask_in_path):
                if getattr(args, 'overwrite', False) or not os.path.exists(mask_out_path):
                    mask_nii = nib.load(mask_in_path)
                    mask_data = mask_nii.get_fdata()
                    
                    # Crop mask with the exact same ROI
                    mask_data_cropped = mask_data[x:x+w, y:y+h, :]
                    
                    # Resize mask to 512x512
                    mask_resized = np.zeros((512, 512, D_new), dtype=np.uint8)
                    for d in range(D_new):
                        mask_slice_hw = mask_data_cropped[:, :, d].T
                        # using INTER_NEAREST to keep binary labels
                        mask_slice_resized = cv2.resize(mask_slice_hw, (512, 512), interpolation=cv2.INTER_NEAREST)
                        mask_resized[:, :, d] = mask_slice_resized.T
                    
                    mask_out_nii = nib.Nifti1Image(mask_resized, mask_nii.affine, mask_nii.header)
                    nib.save(mask_out_nii, mask_out_path)
            
        except Exception as e:
            err_msg = f"Error processing {rel_path}: {e}"
            print(f"  ✗ {err_msg}")
            errors.append(err_msg)
            
    print("\n" + "=" * 55)
    print("           NIFTI PREPROCESSING SUMMARY       ")
    print("=" * 55)
    print(f"  Input directory  : {input_dir}")
    print(f"  Output directory : {output_dir}")
    print(f"  Images processed : {processed}")
    if errors:
        print(f"  Errors / skips   : {len(errors)}")
        for err in errors:
            print(f"    • {err}")
    else:
        print(f"  Errors / skips   : 0")
    print("=" * 55)


parser = argparse.ArgumentParser(description="Preprocess raw NIfTI images")
parser.add_argument("--input_dir", type=str, required=True, help="Path to input raw NIfTI directory")
parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
parser.add_argument("--window_center", type=float, default=None, help="Window center for CT/MRI")
parser.add_argument("--window_width", type=float, default=None, help="Window width for CT/MRI")
parser.add_argument("--no_clahe", action="store_true", help="Disable CLAHE")
parser.add_argument("--no_minmax", action="store_true", help="Disable min-max normalization")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing NIfTI images")

if __name__ == "__main__":
    args, unknown = parser.parse_known_args()
    preprocess_nifti(args)
