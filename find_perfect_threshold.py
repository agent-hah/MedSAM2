import os
import argparse
import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

# Import MedSAM2 native builder and DIAS utilities
from sam2.build_sam import build_sam2_video_predictor_npz
from utils.metrics import get_metrics, AverageMeter


# ==========================================
# UTILITIES & PROMPTING
# ==========================================
def resize_grayscale_to_rgb_and_resize(array, image_size):
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
    mip_2d = np.max(volume_3d, axis=0)

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

    max_area = 0
    for cnt in mip_contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                mip_cx = int(M["m10"] / M["m00"])
                mip_cy = int(M["m01"] / M["m00"])
                max_area = area

    best_mid_contour = None
    min_dist = float('inf')
    best_cx, best_cy = mip_cx, mip_cy

    for cnt in mid_contours:
        area = cv2.contourArea(cnt)
        if area < 20: continue
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            dist = (cx - mip_cx) ** 2 + (cy - mip_cy) ** 2
            if dist < min_dist:
                min_dist = dist
                best_mid_contour = cnt
                best_cx, best_cy = cx, cy

    if best_mid_contour is not None:
        extLeft = tuple(best_mid_contour[best_mid_contour[:, :, 0].argmin()][0])
        extRight = tuple(best_mid_contour[best_mid_contour[:, :, 0].argmax()][0])
        extTop = tuple(best_mid_contour[best_mid_contour[:, :, 1].argmin()][0])
        extBottom = tuple(best_mid_contour[best_mid_contour[:, :, 1].argmax()][0])
        pts = [[best_cx, best_cy], extLeft, extRight, extTop, extBottom]
        return np.array(pts, dtype=np.float32), np.array([1, 1, 1, 1, 1], dtype=np.int32)
    return np.array([[mip_cx, mip_cy]], dtype=np.float32), np.array([1], dtype=np.int32)


# ==========================================
# MAIN GRID SEARCH LOGIC
# ==========================================
def run_threshold_search(config_name, checkpoint_path, test_data_dir):
    print(f"Loading MedSAM2 for Threshold Optimization...")
    predictor = build_sam2_video_predictor_npz(config_name, checkpoint_path)

    test_files = [f for f in os.listdir(test_data_dir) if f.endswith(".npz")]

    # Define the grid of thresholds to test
    thresholds = np.round(np.arange(-2.0, 0.2, 0.2), 1)

    # Initialize trackers for every threshold in the grid
    trackers = {
        t: {"DSC": AverageMeter(), "clDice": AverageMeter(), "Sen": AverageMeter(), "Pre": AverageMeter()}
        for t in thresholds
    }

    print(f"Running Inference & Evaluating {len(thresholds)} Thresholds: {thresholds.tolist()}")

    with torch.no_grad():
        for filename in tqdm(test_files, desc="Processing Volumes"):
            file_path = os.path.join(test_data_dir, filename)
            data = np.load(file_path, allow_pickle=True)
            img_3D_ori = data["imgs"]
            gts_3D = data["gts"]

            D, video_height, video_width = img_3D_ori.shape
            mid_frame_idx = D // 2

            # Format and apply CUSTOM DIAS STATS
            img_resized = resize_grayscale_to_rgb_and_resize(img_3D_ori, 512)
            img_resized = img_resized / 255.0
            img_resized = torch.from_numpy(img_resized).cuda()

            img_mean = torch.tensor((0.5371, 0.5371, 0.5371), dtype=torch.float32)[:, None, None].cuda()
            img_std = torch.tensor((0.2546, 0.2546, 0.2546), dtype=torch.float32)[:, None, None].cuda()
            img_resized -= img_mean
            img_resized /= img_std

            with torch.autocast("cuda", dtype=torch.bfloat16):
                inference_state = predictor.init_state(img_resized, video_height, video_width)

                # Retrieve 2.5D MIP Prompts
                point_prompts, labels = get_mip_guided_prompt(img_3D_ori, mid_frame_idx)

                _, _, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=mid_frame_idx,
                    obj_id=1,
                    points=point_prompts,
                    labels=labels
                )

                # Pre-allocate an array to hold the RAW LOGITS, not binary masks
                logits_3D = np.zeros(img_3D_ori.shape, dtype=np.float32)
                logits_3D[mid_frame_idx] = out_mask_logits[0].cpu().numpy()[0]

                # Track Forward
                for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state,
                                                                                      start_frame_idx=mid_frame_idx,
                                                                                      reverse=False):
                    logits_3D[out_frame_idx] = out_mask_logits[0].cpu().numpy()[0]

                # Track Backward
                for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state,
                                                                                      start_frame_idx=mid_frame_idx,
                                                                                      reverse=True):
                    logits_3D[out_frame_idx] = out_mask_logits[0].cpu().numpy()[0]

            # ==========================================
            # APPLY THRESHOLDS AND CALCULATE METRICS
            # ==========================================
            for t in thresholds:
                # Apply the current threshold to generate the binary mask
                segs_3D_t = (logits_3D > t).astype(np.uint8)

                for j in range(D):
                    predict_slice = segs_3D_t[j]
                    target_slice = gts_3D[j]

                    metric = get_metrics(predict_slice, target_slice, run_clDice=True)
                    trackers[t]["DSC"].update(metric["DSC"])
                    trackers[t]["clDice"].update(metric["cldice"])
                    trackers[t]["Sen"].update(metric["Sen"])
                    trackers[t]["Pre"].update(metric["Pre"])

    # ==========================================
    # DISPLAY RESULTS
    # ==========================================
    print("\n" + "=" * 70)
    print(f"{'Threshold':<12} | {'DSC':<10} | {'clDice':<10} | {'Sensitivity':<15} | {'Precision':<10}")
    print("-" * 70)

    best_thresh = 0.0
    best_dsc = 0.0

    for t in thresholds:
        dsc = trackers[t]['DSC'].mean
        cldice = trackers[t]['clDice'].mean
        sen = trackers[t]['Sen'].mean
        pre = trackers[t]['Pre'].mean

        if dsc > best_dsc:
            best_dsc = dsc
            best_thresh = t

        print(f"{t:<12.1f} | {dsc:<10.4f} | {cldice:<10.4f} | {sen:<15.4f} | {pre:<10.4f}")

    print("=" * 70)
    print(f" OPTIMAL THRESHOLD: {best_thresh} (Yields max DSC of {best_dsc:.4f})")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Base inference config")
    parser.add_argument("-ckpt", "--checkpoint", required=True, help="Path to fine-tuned checkpoint.pt")
    # You can pass your validation set here to optimize, before locking it in for the test set
    parser.add_argument("-d", "--data_dir", required=True, help="Path to DIAS Test/Val NPZ folder")

    args = parser.parse_args()

    run_threshold_search(args.config, args.checkpoint, args.data_dir)