"""
Inference script for RatData (no ground truth).

Mirrors the original RatData directory structure in the output, producing
for each NPZ volume:
  - A side-by-side comparison PNG of the input MIP and the prediction overlay
  - Per-slice comparison PNGs (input slice | prediction overlay) in a subfolder
  - The raw 3D segmentation mask as a .npy file

Uses the same MIP-guided automatic prompting as evaluate_medsam2.py, but with a
sliding window approach for prompting: the entire sequence is loaded as context, but
prompts are placed at the center of each sliding window and propagation is confined
to that window, keeping error propagation localized.

Usage:
    python infer_ratdata.py \
        -c sam2.1_hiera_t512.yaml \
        -ckpt work_dir/medsam2_checkpoint.pt \
        -d data/medsam_preprocessed/RatData_NPZ \
        -o results/RatData_Inference \
        --window_size 8
"""

import os
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

# Import MedSAM2 native builder
from sam2.build_sam import build_sam2_video_predictor_npz


# ============================================================================
# Shared helpers (same as evaluate_medsam2.py)
# ============================================================================

def resize_grayscale_to_rgb_and_resize(array, image_size):
    """Utility to format the 3D volume for MedSAM2 Video Predictor."""
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))
    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)
        resized_array[i] = img_array
    return resized_array


def get_mip_guided_prompt(volume_3d, frame_idx):
    """
    Uses a 2.5D Maximum Intensity Projection to find the global center of the
    arterial tree, then maps that coordinate to the specified slice to extract
    safe, verified 5-point prompts.
    
    Args:
        volume_3d: 3D volume array
        frame_idx: The frame index to use for contour detection
    """
    # 1. Generate the 2.5D MIP across the entire Z-axis (Depth)
    # Vessels appear dark, so minimum intensity projection captures them better
    mip_2d = np.min(volume_3d, axis=0)

    # Helper function to extract contours
    def get_contours(img_slice):
        if img_slice.max() <= 1.0:
            img_uint8 = (img_slice * 255).astype(np.uint8)
        else:
            img_uint8 = img_slice.astype(np.uint8)
        blurred = cv2.GaussianBlur(img_uint8, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    mip_contours = get_contours(mip_2d)
    frame_contours = get_contours(volume_3d[frame_idx])

    h, w = mip_2d.shape
    mip_cx, mip_cy = w / 2.0, h / 2.0

    # 2. Find the global Center of Mass from the largest object in the MIP
    max_area = 0
    for cnt in mip_contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                mip_cx = int(M["m10"] / M["m00"])
                mip_cy = int(M["m01"] / M["m00"])
                max_area = area

    # 3. Find the contour on the specified frame closest to the MIP's global center
    best_frame_contour = None
    min_dist = float('inf')
    best_cx, best_cy = mip_cx, mip_cy

    for cnt in frame_contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Distance from this frame contour to the global MIP center
            dist = (cx - mip_cx)**2 + (cy - mip_cy)**2
            if dist < min_dist:
                min_dist = dist
                best_frame_contour = cnt
                best_cx, best_cy = cx, cy

    # 4. Extract the 5 points from the verified frame contour
    if best_frame_contour is not None:
        extLeft = tuple(best_frame_contour[best_frame_contour[:, :, 0].argmin()][0])
        extRight = tuple(best_frame_contour[best_frame_contour[:, :, 0].argmax()][0])
        extTop = tuple(best_frame_contour[best_frame_contour[:, :, 1].argmin()][0])
        extBottom = tuple(best_frame_contour[best_frame_contour[:, :, 1].argmax()][0])

        pts = [[best_cx, best_cy], extLeft, extRight, extTop, extBottom]
        return np.array(pts, dtype=np.float32), np.array([1, 1, 1, 1, 1], dtype=np.int32)
    else:
        # No vessels detected on this slice
        return None, None


# ============================================================================
# Visualization helpers
# ============================================================================

def make_side_by_side(img_gray, mask, label_left="Input", label_right="Prediction"):
    """Create a side-by-side comparison: input with colorbar (left) | B&W mask with colorbar (right)."""
    # Compute real min/max from the original image data before normalization
    v_min = float(img_gray.min())
    v_max = float(img_gray.max())
    
    ptp = v_max - v_min
    if ptp > 0:
        img_uint8 = np.clip((img_gray - v_min) / ptp * 255, 0, 255).astype(np.uint8)
    else:
        img_uint8 = np.clip(img_gray, 0, 255).astype(np.uint8)
        
    if img_uint8.ndim > 2:
        img_uint8 = img_uint8[:, :, 0]
        
    # Apply CLAHE to improve contrast of the base image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_uint8)

    left = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2BGR)
    right = left.copy()
    
    # Red overlay for prediction
    overlay = right.copy()
    overlay[mask > 0] = (0, 0, 255)
    cv2.addWeighted(overlay, 0.5, right, 0.5, 0, right)
    
    # Add a thick red outline
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(right, contours, -1, (0, 0, 255), 2)
    
    h, w = left.shape[:2]

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, h / 800)
    thickness = max(1, int(h / 400))
    
    # Add labels with black outline for readability
    cv2.putText(left, label_left, (10, 25), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(left, label_left, (10, 25), font, font_scale, (0, 255, 255), thickness)
    cv2.putText(right, label_right, (10, 25), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(right, label_right, (10, 25), font, font_scale, (0, 255, 255), thickness)

    # --- Vertical Color Bar ---
    bar_w = max(20, int(w * 0.04))
    pad_y = int(h * 0.1)
    bar_h = h - 2 * pad_y
    
    def make_vertical_colorbar(panel_h, panel_w, val_min, val_max):
        """Create a vertical colorbar image to append to the right of a panel."""
        cbar_img = np.zeros((panel_h, bar_w + 40, 3), dtype=np.uint8)
        # Draw vertical gradient (top = max, bottom = min)
        for y in range(bar_h):
            val = int(255 * (1.0 - y / max(1, bar_h - 1)))
            cv2.line(cbar_img, (5, pad_y + y), (5 + bar_w, pad_y + y), (val, val, val), 1)
        # Draw border around the gradient
        cv2.rectangle(cbar_img, (5, pad_y), (5 + bar_w, pad_y + bar_h), (200, 200, 200), 1)
        # Max label (top)
        text_max = f"{val_max:.1f}"
        cv2.putText(cbar_img, text_max, (5, pad_y - 5), font, font_scale * 0.7, (255, 255, 255), max(1, thickness - 1))
        # Min label (bottom)
        text_min = f"{val_min:.1f}"
        cv2.putText(cbar_img, text_min, (5, pad_y + bar_h + int(font_scale * 15) + 5), font, font_scale * 0.7, (255, 255, 255), max(1, thickness - 1))
        return cbar_img
    
    # Both colorbars show the actual image intensity range
    left_cbar = make_vertical_colorbar(h, w, v_min, v_max)
    right_cbar = make_vertical_colorbar(h, w, v_min, v_max)

    # Assemble: [left | left_cbar | separator | right | right_cbar]
    separator = np.ones((h, 2, 3), dtype=np.uint8) * 255
    combined = np.hstack([left, left_cbar, separator, right, right_cbar])
    return combined


def npz_name_to_subdir(npz_filename):
    """
    Convert flat NPZ filename back into a mirror directory path.
    NPZ names use double-underscore (__) as the path separator.
    e.g. 'SRG_Angio__SRG_69_Pheno__Aorta_1_angio.npz'
         -> 'SRG_Angio/SRG_69_Pheno/Aorta_1_angio'
    """
    stem = npz_filename.replace(".npz", "")
    parts = stem.split("__")
    return os.path.join(*parts)


# ============================================================================
# Main inference loop
# ============================================================================

def run_inference(config_name, checkpoint_path, data_dir, output_dir, window_size=8):
    """
    Run MedSAM2 inference on all RatData NPZ volumes and save visual
    comparison outputs mirroring the original directory hierarchy.
    """
    print(f"Building model using config: {config_name}")
    print(f"Loading checkpoint from {checkpoint_path}...")
    predictor = build_sam2_video_predictor_npz(config_name, checkpoint_path)

    npz_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])
    print(f"Found {len(npz_files)} NPZ volumes in {data_dir}")

    summary_rows = []

    with torch.no_grad():
        for filename in tqdm(npz_files, desc="Inference"):
            seq_id = filename.replace(".npz", "")
            file_path = os.path.join(data_dir, filename)

            # Reconstruct mirror directory
            mirror_subdir = npz_name_to_subdir(filename)
            seq_output_dir = os.path.join(output_dir, mirror_subdir)
            os.makedirs(seq_output_dir, exist_ok=True)

            # Load images (no ground truth expected)
            data = np.load(file_path, allow_pickle=True)
            img_3D_ori = data["imgs"]

            D = img_3D_ori.shape[0]
            video_height, video_width = img_3D_ori.shape[1:3]

            # Prepare for model
            img_resized = resize_grayscale_to_rgb_and_resize(img_3D_ori, 512)
            img_resized = img_resized / 255.0
            img_resized = torch.from_numpy(img_resized).cuda()

            # Z-score normalization (computed per-sequence at inference time)
            seq_mean = img_resized.mean()
            seq_std = img_resized.std() + 1e-8
            img_resized = (img_resized - seq_mean) / seq_std

            with torch.autocast("cuda", dtype=torch.bfloat16):
                inference_state = predictor.init_state(img_resized, video_height, video_width)

                segs_3D = np.zeros(img_3D_ori.shape, dtype=np.uint8)
                total_prompt_points = 0
                
                # ==========================================
                # MULTI-FRAME AUTOMATIC PROMPTING
                # ==========================================
                prompted_frames = []
                for i in range(D):
                    point_prompts, labels = get_mip_guided_prompt(img_3D_ori, i)
                    
                    if point_prompts is not None:
                        total_prompt_points += len(point_prompts)
                        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=i,
                            obj_id=1,
                            points=point_prompts,
                            labels=labels
                        )
                        segs_3D[i] = (out_mask_logits[0] > 0.0).cpu().numpy()[0].astype(np.uint8)
                        prompted_frames.append(i)
                
                # ==========================================
                # VIDEO PROPAGATION PHASE (Global)
                # ==========================================
                if len(prompted_frames) > 0:
                    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                        segs_3D[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()[0].astype(np.uint8)
                else:
                    print(f"Warning: No vessels detected in any slice for {seq_id}")

            # ==========================================
            # SAVE OUTPUTS
            # ==========================================

            # 1. Summary comparison (input vs. merged prediction mask)
            input_minip = np.min(img_3D_ori, axis=0)
            merged_mask = np.max(segs_3D, axis=0)

            summary_img = make_side_by_side(
                input_minip, merged_mask,
                label_left="Input (middle frame)", label_right="Prediction"
            )
            cv2.imwrite(os.path.join(seq_output_dir, "summary_comparison.png"), summary_img)

            # 3. Per-slice comparisons
            slices_dir = os.path.join(seq_output_dir, "slices")
            os.makedirs(slices_dir, exist_ok=True)
            for i in range(D):
                slice_img = make_side_by_side(
                    img_3D_ori[i], segs_3D[i],
                    label_left=f"Slice {i}/{D-1}",
                    label_right="Prediction"
                )
                cv2.imwrite(os.path.join(slices_dir, f"slice_{i:04d}.png"), slice_img)

            # Track summary statistics (no metrics, just counts)
            n_positive_slices = int(np.any(segs_3D, axis=(1, 2)).sum())
            total_vessel_pixels = int(segs_3D.sum())
            summary_rows.append({
                "sequence": seq_id,
                "num_slices": D,
                "resolution": f"{video_height}x{video_width}",
                "positive_slices": n_positive_slices,
                "total_vessel_pixels": total_vessel_pixels,
                "num_prompt_points": total_prompt_points,
            })

    # ==========================================
    # WRITE SUMMARY REPORT
    # ==========================================
    import csv
    csv_path = os.path.join(output_dir, "inference_summary.csv")
    if summary_rows:
        fieldnames = summary_rows[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print("\n" + "=" * 60)
    print("       RATDATA INFERENCE COMPLETE")
    print("=" * 60)
    print(f"  Volumes processed : {len(npz_files)}")
    print(f"  Output directory  : {output_dir}")
    print(f"  Summary CSV       : {csv_path}")
    print()
    print("  For each volume, the output directory contains:")
    print("    comparison.png      — Side-by-side Middle Frame vs prediction overlay")
    print("    segmentation_3d.npy — Raw 3D binary mask")
    print("    slices/             — Per-slice input vs prediction PNGs")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run MedSAM2 inference on RatData and produce visual comparisons (no ground truth needed)"
    )
    parser.add_argument("-c", "--config", required=True,
                        help="Base inference config (e.g., sam2.1_hiera_t512.yaml)")
    parser.add_argument("-ckpt", "--checkpoint", required=True,
                        help="Path to fine-tuned checkpoint.pt")
    parser.add_argument("-d", "--data_dir", required=True,
                        help="Path to preprocessed RatData NPZ folder")
    parser.add_argument("-o", "--output_dir", required=True,
                        help="Path to save mirrored output directory")

    args = parser.parse_args()

    run_inference(
        config_name=args.config,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

if __name__ == "__main__":
    main()
