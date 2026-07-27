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
from utils.metrics import get_metrics, get_color, AverageMeter, count_connect_component

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
        # No vessels detected on this slice
        return None, None

def make_side_by_side_dias(img_gray, mask, label_left="Input", label_right="Prediction"):
    """Create a side-by-side comparison: input with colorbar (left) | red overlay with colorbar (right)."""
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
        
    left = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
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

def save_side_by_side_video(img_3D_ori, segs_3D, video_path):
    """Saves a side-by-side MP4 video comparing the target slice and the prediction overlay."""
    D, H, W = img_3D_ori.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    first_combined = make_side_by_side_dias(
        img_3D_ori[0], segs_3D[0],
        label_left=f"Target Slice 0", label_right="Prediction"
    )
    video_h, video_w = first_combined.shape[:2]
    
    video = cv2.VideoWriter(video_path, fourcc, 10, (video_w, video_h))
    
    for i in range(D):
        combined = make_side_by_side_dias(
            img_3D_ori[i], segs_3D[i],
            label_left=f"Target Slice {i}", label_right="Prediction"
        )
        video.write(combined)
    video.release()

def run_inference_and_evaluate(config_name, checkpoint_path, test_data_dir, output_dir):
    # 1. Replicate DIAS Tester directory structure exactly
    pred_folder = os.path.join(output_dir, 'pred')
    pred_b_folder = os.path.join(output_dir, 'pred_b')
    gt_folder = os.path.join(output_dir, 'gt')
    color_folder = os.path.join(output_dir, 'color')
    videos_folder = os.path.join(output_dir, 'videos')

    os.makedirs(pred_folder, exist_ok=True)
    os.makedirs(pred_b_folder, exist_ok=True)
    os.makedirs(gt_folder, exist_ok=True)
    os.makedirs(color_folder, exist_ok=True)
    os.makedirs(videos_folder, exist_ok=True)

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
    vc_meter = AverageMeter()
    individual_results = []

    print(f"Building model using config: {config_name}")
    print(f"Loading checkpoint from {checkpoint_path}...")
    predictor = build_sam2_video_predictor_npz(config_name, checkpoint_path)

    print(f"Running Fully Automatic Inference & Evaluation on: {test_data_dir}")
    test_files = [f for f in os.listdir(test_data_dir) if f.endswith(".npz")]

    with torch.no_grad():
        for j, filename in enumerate(tqdm(test_files, desc="Testing", ncols=150)):
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

            # Z-score normalization (computed per-sequence at inference time)
            seq_mean = img_resized.mean()
            seq_std = img_resized.std() + 1e-8
            img_resized = (img_resized - seq_mean) / seq_std

            with torch.autocast("cuda", dtype=torch.bfloat16):
                inference_state = predictor.init_state(img_resized, video_height, video_width)

                # ==========================================
                # MULTI-FRAME PROMPTING (Bidirectional + 2.5D MIP)
                # ==========================================
                D = img_3D_ori.shape[0]
                
                prompted_frames = []
                for i in range(D):
                    point_prompts, labels = get_mip_guided_prompt(img_3D_ori, i)
                    
                    if point_prompts is not None:
                        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=i,
                            obj_id=1,
                            points=point_prompts,
                            labels=labels
                        )
                        prompted_frames.append(i)
                
                segs_3D = np.zeros(img_3D_ori.shape, dtype=np.uint8)
                
                # ==========================================
                # VIDEO PROPAGATION PHASE (Global)
                # ==========================================
                if len(prompted_frames) > 0:
                    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                        segs_3D[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()[0].astype(np.uint8)
                else:
                    print(f"Warning: No vessels detected in any slice for {seq_id}")

                # ==========================================
                # DIAS 2.5D EVALUATION & EXPORT LOGIC
                # ==========================================
                # 1. Squash the 3D tracked prediction into a single 2D MIP mask
                predict_2d_mip = np.max(segs_3D, axis=0)
                predict_b = np.where(predict_2d_mip >= 0.5, 1, 0)

                # 2. Extract the single ground truth mask
                target_2d = gts_3D[0] if gts_3D.ndim == 3 else gts_3D

                # 3. Format images as uint8 contiguous arrays
                gt_img = np.ascontiguousarray((target_2d * 255).astype(np.uint8))
                pre_img = np.ascontiguousarray((predict_2d_mip * 255).astype(np.uint8))
                pre_b_img = np.ascontiguousarray((predict_b * 255).astype(np.uint8))

                # 4. Save PNG images using index j
                cv2.imwrite(os.path.join(gt_folder, f"gt{j}.png"), gt_img)
                cv2.imwrite(os.path.join(pred_folder, f"pre{j}.png"), pre_img)
                cv2.imwrite(os.path.join(pred_b_folder, f"pre_b{j}.png"), pre_b_img)

                img_color = get_color(predict_b, target_2d)
                cv2.imwrite(os.path.join(color_folder, f"color_b{j}.png"), img_color.astype(np.uint8))
                
                # 5. Save the side-by-side MP4 video for this sequence
                video_path = os.path.join(videos_folder, f"video_{j}.mp4")
                save_side_by_side_video(img_3D_ori, segs_3D, video_path)

                # 6. Calculate metrics per sequence and update trackers
                metric = get_metrics(predict_2d_mip, target_2d, run_clDice=True)
                for k in results.keys():
                    results[k].update(metric[k])
                
                vc_val = count_connect_component(predict_b, target_2d)
                vc_meter.update(vc_val)
                
                ind_metric = {'image_id': f"image_{j}"}
                ind_metric.update(metric)
                ind_metric['VC'] = vc_val
                individual_results.append(ind_metric)

    # ==========================================
    # FINAL METRICS EXPORT
    # ==========================================
    model_name = "medsam2"
    mean_data = [results[k].mean for k in results.keys()] + [vc_meter.mean]
    std_data = [results[k].std for k in results.keys()] + [vc_meter.std]
    columns = list(results.keys()) + ["VC"]
    
    formatted_data = [rf"{mean:.4f}$\pm${std:.4f}" for mean, std in zip(mean_data, std_data)]
    data_dict = {col: [val] for col, val in zip(columns, formatted_data)}
    
    df = pd.DataFrame(data_dict)
    df.to_csv(os.path.join(output_dir, f"{model_name}_result.csv"), index=False)
    
    df_ind = pd.DataFrame(individual_results)
    df_ind.to_csv(os.path.join(output_dir, f"{model_name}_individual_results.csv"), index=False)

    print("\n" + "="*45)
    print("      DIAS BENCHMARK RESULTS      ")
    print("="*45)
    for k in results.keys():
        print(f"{k:6} (Mean) : {results[k].mean:.4f}")
    for k in results.keys():
        print(f"{k:6} (Std)  : {results[k].std:.4f}")
    print(f"VC     (Mean) : {vc_meter.mean:.4f}")
    print(f"VC     (Std)  : {vc_meter.std:.4f}")
    print("="*45)
    print(f"Metrics saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MedSAM2 identically to DIAS 3D full-supervised benchmark")
    parser.add_argument("-c", "--config", required=True, help="Base inference config (e.g., sam2.1_hiera_t512.yaml)")
    parser.add_argument("-ckpt", "--checkpoint", required=True, help="Path to fine-tuned checkpoint.pt")
    parser.add_argument("-d", "--data_dir", required=True, help="Path to preprocessed DIAS Test NPZ folder")
    parser.add_argument("-o", "--output_dir", required=True, help="Path to save pred/, gt/, color/, videos/, and results")

    args = parser.parse_args()

    run_inference_and_evaluate(
        config_name=args.config,
        checkpoint_path=args.checkpoint,
        test_data_dir=args.data_dir,
        output_dir=args.output_dir,
    )