import os
import glob
import re
import argparse
import numpy as np
import cv2
from tqdm import tqdm

import utils.preprocess as preprocess_utils

def preprocess_dias_for_medsam2(base_data_dir, output_dir, label_type="standard", args=None):
    """
    Preprocesses the DIAS dataset into MedSAM2-compliant 3D .npz files.
    Includes fallback logic to duplicate the previous frame if a slice is missing.
    """
    img_dir = os.path.join(base_data_dir, "images")

    # Configure output directories based on the label type
    if label_type == "standard":
        label_dir = os.path.join(base_data_dir, "labels")
        save_dir = os.path.join(output_dir, "DIAS_Standard_NPZ")
    else:
        label_dir = os.path.join(base_data_dir, "scribble_labels", label_type)
        save_dir = os.path.join(output_dir, f"DIAS_Scribble_{label_type}_NPZ")

    os.makedirs(save_dir, exist_ok=True)

    # Identify unique sequence IDs by finding the first slice (i0)
    img_paths = glob.glob(os.path.join(img_dir, "image_s*_i0.png"))
    seq_ids = [re.search(r"image_s(\d+)_i0\.png", os.path.basename(p)).group(1) for p in img_paths]

    print(f"Processing {label_type}... Found {len(seq_ids)} sequences.")

    for seq_id in tqdm(seq_ids):
        label_path = os.path.join(label_dir, f"label_s{seq_id}.png")
        if not os.path.exists(label_path):
            continue

        sequence_imgs = []
        is_valid_sequence = True

        # Load and stack all 8 slices into a temporal volume
        for i in range(8):
            img_path = os.path.join(img_dir, f"image_s{seq_id}_i{i}.png")

            # --- Fallback Logic (Option A) ---
            if not os.path.exists(img_path):
                print(f"\nWarning: Missing slice {i} for sequence {seq_id}. Duplicating previous frame.")
                if len(sequence_imgs) > 0:
                    # Append the previous frame again to pad the sequence
                    sequence_imgs.append(sequence_imgs[-1])
                    continue
                else:
                    # If the very first frame (i0) is somehow missing, we cannot duplicate.
                    is_valid_sequence = False
                    break
            # ---------------------------------
            
            # Process normal frames
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            sequence_imgs.append(img)

        # --- Optional Preprocessing with utils ---
        if len(sequence_imgs) > 0 and args is not None:
            roi = preprocess_utils.get_crop_roi(
                sequence_imgs, f"DIAS_s{seq_id}", clear_cache=args.clear_cache
            )
            
            processed_imgs = []
            for img in sequence_imgs:
                cropped = preprocess_utils.crop_image(img, roi)
                processed = preprocess_utils.apply_full_preprocessing(
                    cropped,
                    window_center=args.window_center,
                    window_width=args.window_width,
                    tv_weight=args.tv_weight,
                    use_frangi=not args.no_frangi
                )
                processed_imgs.append(cv2.resize(processed, (512, 512), interpolation=cv2.INTER_LINEAR))
            sequence_imgs = processed_imgs
        else:
            # Fallback if args is not provided
            processed_imgs = []
            for img in sequence_imgs:
                processed_imgs.append(cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR))
            sequence_imgs = processed_imgs
            
        # Failsafe: only save if we successfully constructed an 8-frame sequence
        if not is_valid_sequence or len(sequence_imgs) != 8:
            print(f"Skipping sequence s{seq_id} due to unrecoverable missing data.")
            continue

        # Shape: (8, 512, 512) - leave as uint8 [0, 255], MedSAM2 normalizes internally
        imgs_np = np.stack(sequence_imgs, axis=0)

        # Load label and apply same ROI and replicate it across the time dimension
        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        if args is not None and len(sequence_imgs) > 0:
            mask = preprocess_utils.crop_image(mask, roi)
            
        mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask_resized > 0).astype(np.uint8)

        # Shape: (8, 512, 512)
        gts_np = np.repeat(mask_binary[np.newaxis, :, :], 8, axis=0)

        # Save compressed NPZ file for MedSAM2
        np.savez_compressed(os.path.join(save_dir, f"s{seq_id}.npz"), imgs=imgs_np, gts=gts_np)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DIAS dataset to NPZ with preprocessing")
    parser.add_argument("--base_data_dir", type=str, default=os.path.expanduser("~/projects/lab/DIAS/d_data/DIAS/training"))
    parser.add_argument("--output_dir", type=str, default=os.path.expanduser("~/projects/git/MedSAM2/data/medsam_preprocessed"))
    parser.add_argument("--clear_cache", action="store_true", help="Clear the ROI cache")
    parser.add_argument("--window_center", type=float, default=None, help="Window center")
    parser.add_argument("--window_width", type=float, default=None, help="Window width")
    parser.add_argument("--tv_weight", type=float, default=0.1, help="Total variation denoising weight")
    parser.add_argument("--no_frangi", action="store_true", help="Disable Frangi/Hessian filtering")
    
    args = parser.parse_args()

    # Process all three label types
    preprocess_dias_for_medsam2(args.base_data_dir, args.output_dir, label_type="standard", args=args)
    preprocess_dias_for_medsam2(args.base_data_dir, args.output_dir, label_type="SALE", args=args)
    preprocess_dias_for_medsam2(args.base_data_dir, args.output_dir, label_type="RDFA", args=args)

    print("\nPreprocessing complete!")