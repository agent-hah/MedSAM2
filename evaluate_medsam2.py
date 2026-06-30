import os
import argparse
import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

# Import MedSAM2 native builder
from sam2.build_sam import build_sam2_video_predictor_npz

# Import DIAS evaluation utilities
from utils.metrics import get_metrics, get_color, AverageMeter

def resize_grayscale_to_rgb_and_resize(array, image_size):
    """Utility to format the 3D volume for MedSAM2 Video Predictor"""
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))
    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)
        resized_array[i] = img_array
    return resized_array

def get_mip_guided_prompt(volume_3d, mid_idx):
    """
    Uses a 2.5D Maximum Intensity Projection to find the global center of the
    arterial tree, then maps that coordinate to the middle slice to extract
    safe, verified 5-point prompts.
    """
    # 1. Generate the 2.5D MIP across the entire Z-axis (Depth)
    mip_2d = np.max(volume_3d, axis=0)

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
    mid_contours = get_contours(volume_3d[mid_idx])

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

    # 3. Find the contour on the MIDDLE slice closest to the MIP's global center
    best_mid_contour = None
    min_dist = float('inf')
    best_cx, best_cy = mip_cx, mip_cy

    for cnt in mid_contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Distance from this middle-slice contour to the global MIP center
            dist = (cx - mip_cx)**2 + (cy - mip_cy)**2
            if dist < min_dist:
                min_dist = dist
                best_mid_contour = cnt
                best_cx, best_cy = cx, cy

    # 4. Extract the 5 points from the verified middle-slice contour
    if best_mid_contour is not None:
        extLeft = tuple(best_mid_contour[best_mid_contour[:, :, 0].argmin()][0])
        extRight = tuple(best_mid_contour[best_mid_contour[:, :, 0].argmax()][0])
        extTop = tuple(best_mid_contour[best_mid_contour[:, :, 1].argmin()][0])
        extBottom = tuple(best_mid_contour[best_mid_contour[:, :, 1].argmax()][0])

        pts = [[best_cx, best_cy], extLeft, extRight, extTop, extBottom]
        return np.array(pts, dtype=np.float32), np.array([1, 1, 1, 1, 1], dtype=np.int32)
    else:
        # Absolute fallback if the middle slice is mysteriously empty
        return np.array([[mip_cx, mip_cy]], dtype=np.float32), np.array([1], dtype=np.int32)

def run_inference_and_evaluate(config_name, checkpoint_path, test_data_dir, output_dir):
    # 1. Replicate DIAS Tester directory structure
    pred_folder = os.path.join(output_dir, 'pred')
    gt_folder = os.path.join(output_dir, 'gt')
    color_folder = os.path.join(output_dir, 'color_imgs')

    os.makedirs(pred_folder, exist_ok=True)
    os.makedirs(gt_folder, exist_ok=True)
    os.makedirs(color_folder, exist_ok=True)

    # Initialize DIAS metric trackers
    results = {
        "DSC": AverageMeter(),
        "Acc": AverageMeter(),
        "Sen": AverageMeter(),
        "Spe": AverageMeter(),
        "Pre": AverageMeter(),
        "IOU": AverageMeter(),
        "AUC": AverageMeter(),
        "cldice": AverageMeter()
    }

    print(f"Building model using config: {config_name}")
    print(f"Loading checkpoint from {checkpoint_path}...")
    predictor = build_sam2_video_predictor_npz(config_name, checkpoint_path)

    print(f"Running Fully Automatic Inference & Evaluation on: {test_data_dir}")
    test_files = [f for f in os.listdir(test_data_dir) if f.endswith(".npz")]

    with torch.no_grad():
        for filename in tqdm(test_files, desc="Testing"):
            seq_id = filename.split(".")[0]
            file_path = os.path.join(test_data_dir, filename)

            # Load images and ground truth
            data = np.load(file_path, allow_pickle=True)
            img_3D_ori = data["imgs"]
            gts_3D = data["gts"]

            video_height, video_width = img_3D_ori.shape[1:3]
            img_resized = resize_grayscale_to_rgb_and_resize(img_3D_ori, 512)
            img_resized = img_resized / 255.0
            img_resized = torch.from_numpy(img_resized).cuda()

            dias_mean = 0.5371
            dias_std = 0.2546

            img_mean = torch.tensor((dias_mean, dias_mean, dias_mean), dtype=torch.float32)[:, None, None].cuda()
            img_std = torch.tensor((dias_std, dias_std, dias_std), dtype=torch.float32)[:, None, None].cuda()

            img_resized -= img_mean
            img_resized /= img_std

            with torch.autocast("cuda", dtype=torch.bfloat16):
                inference_state = predictor.init_state(img_resized, video_height, video_width)

                # ==========================================
                # FULLY AUTOMATIC PROMPTING (Bidirectional + 2.5D MIP)
                # ==========================================
                # Calculate the middle slice
                D = img_3D_ori.shape[0]
                mid_frame_idx = D // 2

                # Feed the entire 3D volume to the MIP generator
                point_prompts, labels = get_mip_guided_prompt(img_3D_ori, mid_frame_idx)

                # Pass the MIP-verified points to the middle frame
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=mid_frame_idx,
                    obj_id=1,
                    points=point_prompts,
                    labels=labels
                )

                segs_3D = np.zeros(img_3D_ori.shape, dtype=np.uint8)
                segs_3D[mid_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()[0].astype(np.uint8)

                # ==========================================
                # VIDEO PROPAGATION PHASE (Bidirectional)
                # ==========================================
                # Track Forward
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                        inference_state, start_frame_idx=mid_frame_idx, reverse=False
                ):
                    segs_3D[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()[0].astype(np.uint8)

                # Track Backward
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                        inference_state, start_frame_idx=mid_frame_idx, reverse=True
                ):
                    segs_3D[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()[0].astype(np.uint8)

                # ==========================================
                # DIAS 2.5D EVALUATION & EXPORT LOGIC
                # ==========================================
                # 1. Squash the 3D tracked prediction into a single 2D MIP mask
                # If the vessel was tracked in ANY slice, it becomes 1 in the final 2D mask.
                predict_2d_mip = np.max(segs_3D, axis=0)

                # 2. Extract the single ground truth mask
                # (Handling cases where gts_3D might be shape (1, H, W) or just (H, W))
                target_2d = gts_3D[0] if gts_3D.ndim == 3 else gts_3D

                # 3. Save raw numpy arrays (Now saving the 2D versions to match DIAS)
                np.save(os.path.join(pred_folder, f'{seq_id}.npy'), predict_2d_mip)
                np.save(os.path.join(gt_folder, f'{seq_id}.npy'), target_2d)

                # 4. Generate and save a single Color PNG per sequence
                img_color = get_color(predict_2d_mip, target_2d)
                cv2.imwrite(os.path.join(color_folder, f'{seq_id}.png'), img_color)

                # 5. Calculate metrics ONCE per sequence and update trackers
                metric = get_metrics(predict_2d_mip, target_2d, run_clDice=True)
                for k in results.keys():
                    results[k].update(metric[k])

    # ==========================================
    # FINAL METRICS EXPORT
    # ==========================================
    # Save the tracked metrics to results.csv
    df = pd.DataFrame()
    for k in results.keys():
        df[k] = [results[k].mean, results[k].std]

    # Label the rows for clarity (row 0 = mean, row 1 = std)
    df.index = ['mean', 'std']

    csv_path = os.path.join(output_dir, 'results.csv')
    df.to_csv(csv_path, index=True)

    print("\n" + "="*45)
    print("      DIAS BENCHMARK RESULTS      ")
    print("="*45)
    for k, v in results.items():
        print(f"{k:6}: {v.mean:.4f} ± {v.std:.4f}")
    print("="*45)
    print(f"Metrics saved to {csv_path}")
    print(f"Color overlays saved to {color_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MedSAM2 identically to DIAS 3D full-supervised benchmark")
    parser.add_argument("-c", "--config", required=True, help="Base inference config (e.g., sam2.1_hiera_t512.yaml)")
    parser.add_argument("-ckpt", "--checkpoint", required=True, help="Path to fine-tuned checkpoint.pt")
    parser.add_argument("-d", "--data_dir", required=True, help="Path to preprocessed DIAS Test NPZ folder")
    parser.add_argument("-o", "--output_dir", required=True, help="Path to save pred/, gt/, color_imgs/, and results.csv")

    args = parser.parse_args()

    run_inference_and_evaluate(
        config_name=args.config,
        checkpoint_path=args.checkpoint,
        test_data_dir=args.data_dir,
        output_dir=args.output_dir,
    )