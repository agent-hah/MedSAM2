import os
import glob
import re
import numpy as np
import cv2
from tqdm import tqdm

def preprocess_dias_test_for_medsam2(base_data_dir, output_dir):
    """
    Preprocesses the DIAS Test dataset into MedSAM2-compliant 3D .npz files.
    """
    img_dir = os.path.join(base_data_dir, "images")
    label_dir = os.path.join(base_data_dir, "labels")
    save_dir = os.path.join(output_dir, "DIAS_Test_NPZ")

    os.makedirs(save_dir, exist_ok=True)

    # Identify unique sequence IDs by finding the first slice (i0)
    img_paths = glob.glob(os.path.join(img_dir, "image_s*_i0.png"))
    seq_ids = [re.search(r"image_s(\d+)_i0\.png", os.path.basename(p)).group(1) for p in img_paths]

    print(f"Processing Test Data... Found {len(seq_ids)} sequences.")

    for seq_id in tqdm(seq_ids):
        sequence_imgs = []
        is_valid_sequence = True

        # Load and stack all 8 slices into a temporal volume
        for i in range(8):
            img_path = os.path.join(img_dir, f"image_s{seq_id}_i{i}.png")

            # --- Fallback Logic ---
            if not os.path.exists(img_path):
                print(f"\nWarning: Missing slice {i} for sequence {seq_id}. Duplicating previous frame.")
                if len(sequence_imgs) > 0:
                    sequence_imgs.append(sequence_imgs[-1])
                    continue
                else:
                    is_valid_sequence = False
                    break

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            sequence_imgs.append(img_resized)

        if not is_valid_sequence or len(sequence_imgs) != 8:
            print(f"Skipping sequence s{seq_id} due to unrecoverable missing data.")
            continue

        imgs_np = np.stack(sequence_imgs, axis=0)

        # Include ground truth in the NPZ to match training format exactly
        label_path = os.path.join(label_dir, f"label_s{seq_id}.png")
        if os.path.exists(label_path):
            mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            mask_binary = (mask_resized > 0).astype(np.uint8)
            gts_np = np.repeat(mask_binary[np.newaxis, :, :], 8, axis=0)
        else:
            gts_np = np.zeros_like(imgs_np)

        np.savez_compressed(os.path.join(save_dir, f"s{seq_id}.npz"), imgs=imgs_np, gts=gts_np)

if __name__ == "__main__":
    HOME_DIR = os.path.expanduser("~")
    # Point this to the test directory instead of the training directory
    DIAS_TEST_DIR = os.path.join(HOME_DIR, "projects/lab/DIAS/d_data/DIAS/test")
    OUTPUT_DIR = os.path.join(HOME_DIR, "projects/git/MedSAM2/data/medsam_preprocessed")

    preprocess_dias_test_for_medsam2(DIAS_TEST_DIR, OUTPUT_DIR)
    print("\nTest data preprocessing complete!")