"""
Inference Script for MedSAM2 on Human Data for DIAS Analysis
"""

import numpy as np
import torch
import os
import argparse
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from skimage import measure
from scipy import ndimage

import models


def extract_patches(image: np.ndarray, patch_size: int = 256, overlap: float = 0.15) -> dict:
    """
    Extract overlapping patches from a binary mask image.

    Args:
        image: Input binary mask (H, W)
        patch_size: Size of each patch (patch_size x patch_size)
        overlap: Overlap ratio between adjacent patches

    Returns:
        Dictionary with 'patches' (list), 'coords' (list of (x1, y1, x2, y2))
    """
    H, W = image.shape
    step = int(patch_size * (1 - overlap))
    coords = []
    patches = []

    for y in range(0, H - patch_size + 1, step):
        for x in range(0, W - patch_size + 1, step):
            coords.append((x, y, x + patch_size, y + patch_size))
            patches.append(image[y:y + patch_size, x:x + patch_size])

    # Handle right boundary
    x = W - patch_size
    if x > 0 and x != coords[-1][0] if coords else True:
        coords.append((x, 0, W, patch_size))
        patches.append(image[x:x + patch_size, 0:patch_size])

    # Handle bottom boundary
    y = H - patch_size
    if y > 0:
        x_end = W - patch_size
        start_x = coords[-1][0] if coords else 0
        for x in range(start_x, x_end + 1, step):
            coords.append((x, y, x + patch_size, H))
            patches.append(image[x:x + patch_size, y:y + patch_size])

    # Handle corner
    if coords[-1][0] != W - patch_size and coords[-1][1] != H - patch_size:
        coords.append((W - patch_size, H - patch_size, W, H))
        patches.append(image[W - patch_size:, H - patch_size:])

    return {"patches": np.array(patches), "coords": coords}


def recompone_overlap(preds: np.ndarray, coords: list, img_size: tuple, img_size_original: tuple, alpha: float = 1.5) -> np.ndarray:
    """
    Recompose image from overlapping patches by averaging overlap regions.

    Args:
        preds: Predicted patches (N, H_p, W_p) or (N, 1, H_p, W_p)
        coords: Coordinates of each patch
        img_size: Size of each patch (H, W)
        img_size_original: Size of original image (H, W)
        alpha: Power factor for overlap weighting

    Returns:
        Recomposed image (H_original, W_original)
    """
    H, W = img_size_original
    output = np.zeros(img_size_original, dtype=preds.dtype)
    count = np.zeros(img_size_original, dtype=preds.dtype)

    for pred, (x1, y1, x2, y2) in zip(preds, coords):
        pred = np.squeeze(pred)
        output[y1:y2, x1:x2] += pred ** alpha
        count[y1:y2, x1:x2] += 1

    # Avoid division by zero
    count[count == 0] = 1
    output = output / count
    return output


def build_window_tensor(img_np: np.ndarray, window_start: int, window_size: int) -> torch.Tensor:
    """
    Build a tensor from a sliding window of frames.

    Extracts `window_size` frames starting at window_start, clamping to
    the volume boundaries and repeating the edge frame to pad if needed.

    Args:
        img_np: Preprocessed frames array (D, H, W)
        window_start: Starting index for this window
        window_size: Total number of frames in the window

    Returns:
        Tensor of shape (1, window_size, H, W), values [0, 255]
    """
    D = len(img_np)
    w_end = min(window_start + window_size, D)
    frames = img_np[window_start:w_end].astype(np.float32)
    # Pad by repeating the last frame if we hit the volume boundary
    if len(frames) < window_size:
        pad = np.stack([frames[-1]] * (window_size - len(frames)))
        frames = np.concatenate([frames, pad], axis=0)
        
    ptp = frames.max() - frames.min()
    if ptp > 0:
        frames = np.clip((frames - frames.min()) / ptp * 255, 0, 255).astype(np.uint8)
    else:
        frames = frames.astype(np.uint8)
        
    # Create (1, window_size, H, W) tensor
    tensor_list = [frame2tensor(f) for f in frames]
    return torch.stack(tensor_list, dim=0).unsqueeze(0)


def frame2tensor(frame: np.ndarray) -> torch.Tensor:
    """Convert a single frame to a tensor."""
    if torch.cuda.is_available():
        return torch.from_numpy(frame).requires_grad_(False).cuda().to(torch.float16)
    else:
        return torch.from_numpy(frame).requires_grad_(False).to(torch.float16)


def build_model(model_path: str, model_type: str = "UNet", window_size: int = 8, device: str = "cuda"):
    """Build the DIAS model (e.g., UNet) that takes `window_size` frames as input."""
    model_class = getattr(models, model_type)
    # List of models from d3tod2 that expect 1 channel (3D inputs)
    is_3d_model = "3D" in model_type or model_type in ["IPN", "PSC", "SVS_Net", "QS_UNet", "VSS_Net", "ST_UNet"]

    print("Loading model ...", end="\t")
    
    # Load state dict, handling potential nested structures
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "state_dict1" in checkpoint:
        state_dict = checkpoint["state_dict1"]
    else:
        state_dict = checkpoint

    # Try to dynamically guess num_classes from the state_dict
    num_classes = 2
    for k in state_dict.keys():
        if "outc.conv.weight" in k or k == "out_conv.weight" or k.endswith(".outc.conv.weight"):
            num_classes = state_dict[k].shape[0]
            break
        elif k.endswith("final_conv.weight") or k.endswith("classifier.weight"):
            num_classes = state_dict[k].shape[0]
            break

    if is_3d_model:
        model = model_class(num_classes=num_classes, num_channels=1)
    else:
        model = model_class(num_classes=num_classes, num_channels=window_size)
        
    # Handle possible "module." prefix from DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
        
    model.to(device)
    model.eval()
    print("Done")
    return model


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Dice coefficient between two binary masks."""
    intersection = np.sum(mask1 * mask2)
    total = np.sum(mask1) + np.sum(mask2)
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / total


def make_side_by_side_dias(img_gray, mask, label_left="Input", label_right="Prediction"):
    """Create a side-by-side comparison with enhanced visibility."""
    ptp = img_gray.max() - img_gray.min()
    if ptp > 0:
        img_uint8 = np.clip((img_gray - img_gray.min()) / ptp * 255, 0, 255).astype(np.uint8)
    else:
        img_uint8 = np.clip(img_gray, 0, 255).astype(np.uint8)

    if img_uint8.ndim > 2:
        img_uint8 = img_uint8[:, :, 0]
        
    # Apply CLAHE to improve contrast of the base image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_uint8)

    left = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2BGR)
    right = left.copy()
    
    # Fill with semi-transparent red
    overlay = right.copy()
    overlay[mask > 0] = (0, 0, 255)
    cv2.addWeighted(overlay, 0.5, right, 0.5, 0, right)
    
    # Add a thick red outline
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(right, contours, -1, (0, 0, 255), 2)
    
    h, w = left.shape[:2]

    # Add labels with black outline for readability
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, h / 800)
    thickness = max(1, int(h / 400))
    
    # Left text (Yellow with black outline)
    cv2.putText(left, label_left, (10, 25), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(left, label_left, (10, 25), font, font_scale, (0, 255, 255), thickness)
    
    # Right text (Yellow with black outline)
    cv2.putText(right, label_right, (10, 25), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(right, label_right, (10, 25), font, font_scale, (0, 255, 255), thickness)

    # Add a thin white separator
    separator = np.ones((h, 2, 3), dtype=np.uint8) * 255
    combined = np.hstack([left, separator, right])
    return combined


def extract_dias_metrics(segmentation: np.ndarray) -> dict:
    """
    Extract DIAS (Distal Inferior Artery Segment) metrics from a 2D segmentation.

    Args:
        segmentation: Binary mask where vessel = 1

    Returns:
        Dictionary with DIAS metrics
    """
    metrics = {}

    # Find contours
    contours = measure.find_contours(segmentation, 0.5)

    if len(contours) == 0:
        metrics["max_dias"] = 0.0
        metrics["num_vessels"] = 0
        return metrics

    # Find the most distal (lowest Y) and rightmost contour
    max_dias_y = -1
    max_dias_contour = None
    for contour in contours:
        max_y = np.max(contour[:, 1])
        if max_y > max_dias_y:
            max_dias_y = max_y
            max_dias_contour = contour

    if max_dias_contour is not None:
        metrics["max_dias_y"] = float(max_dias_y)
        metrics["max_dias_x"] = float(np.mean(max_dias_contour[max_dias_contour[:, 1] == max_dias_y][:, 0]))
        metrics["max_dias"] = float(max_dias_y)
        metrics["num_vessels"] = len(contours)
    else:
        metrics["max_dias"] = 0.0
        metrics["num_vessels"] = 0

    return metrics


def npz_name_to_subdir(npz_filename):
    """
    Convert flat NPZ filename back into a mirror directory path.
    NPZ names use double-underscore (__) as the path separator.
    """
    stem = npz_filename.replace(".npz", "")
    parts = stem.split("__")
    return os.path.join(*parts)


def process_sequence(seq_name: str, npz_path: str, model, output_base: str, window_size: int = 8, verbose: bool = False) -> dict:
    """
    Process a single NPZ file with sliding window inference across ALL slices.

    For each target slice i, builds a window of `window_size` frames centered on i,
    runs inference, and saves the prediction for slice i individually (no merging).

    Slices that share the same window are grouped to avoid redundant inference.

    Args:
        seq_name: Sequence identifier
        npz_path: Path to the NPZ file containing frames
        model: Loaded MedSAM2 model
        output_base: Base output directory
        window_size: Width of the sliding window (number of frames)
        verbose: Whether to print detailed progress

    Returns:
        Dictionary with inference results and statistics
    """
    # Load data
    data = np.load(npz_path)
    frames = data["imgs"]  # (D, H, W) or (D, H, W, C)
    if len(frames.shape) == 4:
        frames = frames[:, :, :, 0]  # Take first channel if C channel exists

    # Load pre-existing mask for Dice computation
    mirror_subdir = npz_name_to_subdir(seq_name)
    seq_output_dir = os.path.join(output_base, mirror_subdir)
    os.makedirs(seq_output_dir, exist_ok=True)

    original_mask = data["gts"] if "gts" in data else None
    if original_mask is not None:
        if len(original_mask.shape) == 4:
            original_mask = original_mask[:, :, :, 0]
        print(f"  Loaded ground truth mask from NPZ")

    H, W = frames.shape[1], frames.shape[2]
    D = len(frames)

    # Create windows subdirectory
    windows_base_dir = os.path.join(seq_output_dir, "windows")
    os.makedirs(windows_base_dir, exist_ok=True)

    # We slide over by window_size
    if D < window_size:
        pad_size = window_size - D
        print(f"\n  Sequence {seq_name} is shorter than window_size {window_size} (has {D} slices). Padding with {pad_size} duplicated first frames.")
        pad_frames = np.repeat(frames[0:1], pad_size, axis=0)
        frames = np.concatenate([pad_frames, frames], axis=0)
        if original_mask is not None:
            pad_masks = np.repeat(original_mask[0:1], pad_size, axis=0)
            original_mask = np.concatenate([pad_masks, original_mask], axis=0)
        D = len(frames)

    all_window_starts = list(range(0, D - window_size + 1, window_size))
    num_windows = len(all_window_starts)
    center = D / 2
    # Sort by distance to center
    all_window_starts.sort(key=lambda x: abs((x + window_size/2) - center))
    
    target_count = min(8, len(all_window_starts))
    print(f"\n  {D} slices → searching for {target_count} {window_size}-frame windows with predictions (stride {window_size})")

    valid_windows = []
    empty_windows = []
    total_voxels = 0
    segmented_voxels = 0

    for w_start in tqdm(all_window_starts, desc="Predicting windows", leave=False):
        w_end = w_start + window_size
        window_frames = frames[w_start:w_end].astype(np.float32)
        ptp = window_frames.max() - window_frames.min()
        if ptp > 0:
            window_frames = np.clip((window_frames - window_frames.min()) / ptp * 255, 0, 255)
            
        # The DIAS repository applies Z-score standardization per sequence
        window_frames = (window_frames - window_frames.mean()) / (window_frames.std() + 1e-8)
            
        window_tensor = torch.from_numpy(window_frames).unsqueeze(0).cuda().to(torch.float32)
        model_name = model.class_name if hasattr(model, 'class_name') else model.__class__.__name__
        
        is_3d_model = "3D" in model_name or model_name in ["IPN", "PSC", "SVS_Net", "QS_UNet", "VSS_Net", "ST_UNet"]
        if is_3d_model:
            window_tensor = window_tensor.unsqueeze(1)
            
        from contextlib import nullcontext
        use_autocast = model_name in ["MAA_Net", "IPN"]
        amp_context = torch.autocast("cuda", dtype=torch.float16) if use_autocast else nullcontext()
        
        force_patch = getattr(model, 'force_patch_inference', False)
        
        if not force_patch:
            try:
                with torch.no_grad(), amp_context:
                    output = model(window_tensor)
            except torch.cuda.OutOfMemoryError:
                print(f"\n  OOM on full resolution, switching to patch-based inference for the remainder of the run...")
                torch.cuda.empty_cache()
                model.force_patch_inference = True
                model.working_patch_size = 256
                force_patch = True
                
        if force_patch:
            patch_size = getattr(model, 'working_patch_size', 256)
            success = False
            while patch_size >= 64 and not success:
                try:
                    step = int(patch_size * 0.75)
                    H_in, W_in = window_tensor.shape[-2:]
                    
                    output_full = None
                    count_full = None
                    
                    for y in range(0, H_in, step):
                        for x in range(0, W_in, step):
                            y1 = y
                            y2 = min(y + patch_size, H_in)
                            x1 = x
                            x2 = min(x + patch_size, W_in)
                            
                            if y2 - y1 < patch_size:
                                y1 = max(0, H_in - patch_size)
                                y2 = H_in
                            if x2 - x1 < patch_size:
                                x1 = max(0, W_in - patch_size)
                                x2 = W_in
                                
                            patch = window_tensor[..., y1:y2, x1:x2]
                                
                            with torch.no_grad(), amp_context:
                                patch_out = model(patch)
                                if isinstance(patch_out, tuple):
                                    patch_out = patch_out[0]
                                
                            if output_full is None:
                                out_shape = list(patch_out.shape)
                                out_shape[-2:] = [H_in, W_in]
                                output_full = torch.zeros(out_shape, device=window_tensor.device)
                                count_full = torch.zeros(out_shape, device=window_tensor.device)
                            
                            output_full[..., y1:y2, x1:x2] += patch_out
                            count_full[..., y1:y2, x1:x2] += 1
                            
                    output = output_full / count_full
                    success = True
                    model.working_patch_size = patch_size
                except torch.cuda.OutOfMemoryError:
                    print(f"  OOM on patch_size {patch_size}, trying {patch_size//2}...")
                    torch.cuda.empty_cache()
                    patch_size //= 2
                    
            if not success:
                raise RuntimeError("OOM even with patch size 64!")
            
        if isinstance(output, tuple):
            output = output[0]

        if len(output.shape) == 5:
            if output.shape[-2:] != (H, W):
                output = torch.nn.functional.interpolate(output, size=(output.shape[2], H, W), mode='trilinear', align_corners=False)
            prob = torch.softmax(output, dim=1)[0, 1, window_size // 2].cpu().numpy()
        else:
            if output.shape[-2:] != (H, W):
                output = torch.nn.functional.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
            prob = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
            
        binary_mask = (prob > 0.5).astype(np.uint8)

        # Remove border artifacts (zero-padding artifact from CNNs)
        border_thickness = 2
        binary_mask[:border_thickness, :] = 0
        binary_mask[-border_thickness:, :] = 0
        binary_mask[:, :border_thickness] = 0
        binary_mask[:, -border_thickness:] = 0

        slice_voxels = int(np.sum(binary_mask))
        target_idx = w_start + window_size // 2

        slice_metrics = extract_dias_metrics(binary_mask)
        slice_metrics["slice_idx"] = target_idx
        slice_metrics["voxel_count"] = slice_voxels
        slice_metrics["window_start"] = w_start
        slice_metrics["w_end"] = w_end
        slice_metrics["prob"] = prob
        slice_metrics["mask"] = binary_mask

        if slice_voxels > 0:
            valid_windows.append(slice_metrics)
        else:
            empty_windows.append(slice_metrics)

        if len(valid_windows) == target_count:
            break

    selected_windows = valid_windows + empty_windows
    selected_windows = selected_windows[:target_count]
    
    # Sort selected windows chronologically for consistent saving/output
    selected_windows.sort(key=lambda x: x["window_start"])

    all_results = []
    prob_3d = np.zeros((D, H, W), dtype=np.float32)

    for res in selected_windows:
        w_start = res["window_start"]
        w_end = res["w_end"]
        target_idx = res["slice_idx"]
        binary_mask = res["mask"]
        prob = res["prob"]
        slice_voxels = res["voxel_count"]

        total_voxels += H * W
        segmented_voxels += slice_voxels

        window_dir = os.path.join(windows_base_dir, f"window_{w_start:04d}")
        os.makedirs(window_dir, exist_ok=True)
        
        vis_frame = frames[target_idx]
        slice_img = make_side_by_side_dias(
            vis_frame, binary_mask,
            label_left=f"Target Slice {target_idx}",
            label_right="Prediction"
        )
        cv2.imwrite(os.path.join(window_dir, "comparison.png"), slice_img)

        if verbose:
            print(f"\n  Window [{w_start}, {w_end}) -> Target {target_idx}: {slice_voxels} voxels")
            if original_mask is not None:
                dice_val = compute_dice(binary_mask, original_mask[target_idx] if len(original_mask.shape) == 3 else original_mask)
                print(f"    Dice: {dice_val:.4f}")

        prob_3d[target_idx] = prob
        
        # Free memory of arrays
        del res["prob"]
        del res["mask"]
        del res["w_end"]
        all_results.append(res)

    dice = None
    if original_mask is not None:
        dice = compute_dice((prob_3d > 0.5).astype(np.uint8), original_mask)

    binary_3d = (prob_3d > 0.5).astype(np.uint8)

    # Summary comparison (Middle frame of input vs. prediction overlay)
    input_frame = frames[D // 2]
    pred_frame = binary_3d[D // 2]
    
    summary_img = make_side_by_side_dias(
        input_frame, pred_frame,
        label_left=f"Input (Middle Frame {D // 2})", label_right="Prediction"
    )
    cv2.imwrite(os.path.join(seq_output_dir, "comparison.png"), summary_img)

    stats = {
        "seq_name": seq_name,
        "num_slices": D,
        "resolution": f"{H}x{W}",
        "num_windows": num_windows,
        "window_size": window_size,
        "total_voxels": total_voxels,
        "segmented_voxels": segmented_voxels,
        "segmented_fraction": segmented_voxels / total_voxels if total_voxels > 0 else 0,
        "dice": dice,
        "per_slice": all_results,
    }

    return stats


def run_inference(input_dir: str, model_path: str, output_base: str, window_size: int = 8, verbose: bool = False, model_type: str = "UNet") -> list:
    """
    Run sliding window inference on all NPZ files in input directory.

    Args:
        input_dir: Directory containing preprocessed NPZ files
        model_path: Path to the MedSAM2 model weights
        output_base: Base directory for outputs
        window_size: Width of the sliding window (number of frames)
        verbose: Whether to print detailed per-slice information

    Returns:
        List of result dictionaries for each sequence
    """
    # Setup model
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        print("WARNING: CUDA not available, using CPU")

    model = build_model(model_path, model_type=model_type, device=device)

    # Find NPZ files
    npz_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]
    if not npz_files:
        raise ValueError(f"No NPZ files found in {input_dir}")

    # Process each sequence
    all_stats = []
    summary_rows = []
    for npz_file in tqdm(npz_files, desc="Sequences"):
        seq_name = npz_file[:-4]
        npz_path = os.path.join(input_dir, npz_file)
        stats = process_sequence(seq_name, npz_path, model, output_base, window_size=window_size, verbose=verbose)
        all_stats.append(stats)
        
        summary_rows.append({
            "sequence": stats["seq_name"],
            "num_slices": stats["num_slices"],
            "resolution": stats["resolution"],
            "segmented_voxels": stats["segmented_voxels"],
            "segmented_fraction": stats["segmented_fraction"],
            "dice": stats["dice"] if stats["dice"] is not None else ""
        })

    import csv
    csv_path = os.path.join(output_base, "inference_summary.csv")
    if summary_rows:
        fieldnames = summary_rows[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    return all_stats


def print_summary(all_stats: list):
    """Print inference summary."""
    print("\n" + "=" * 80)
    print("INFERENCE SUMMARY")
    print("=" * 80)

    valid_stats = [s for s in all_stats if s is not None]
    dice_values = [s["dice"] for s in valid_stats if s["dice"] is not None]

    for stats in valid_stats:
        print(f"\nSequence: {stats['seq_name']}")
        print(f"  Slices: {stats['num_slices']}")
        print(f"  Windows: {stats['num_windows']} (window_size={stats['window_size']})")
        print(f"  Segmented fraction: {stats['segmented_fraction']:.4f} "
              f"({stats['segmented_voxels']:,}/{stats['total_voxels']:,} voxels)")
        if stats['dice'] is not None:
            print(f"  Dice coefficient: {stats['dice']:.4f}")

    if dice_values:
        print(f"\nAverage Dice: {np.mean(dice_values):.4f} ± {np.std(dice_values):.4f}")

    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedSAM2 Inference for DIAS Analysis")
    # Original args
    parser.add_argument("--input_dir", type=str,
                        default="/home/ashmithandoo/projects/git/MedSAM2/data/humandata_preprocessed/predictions/MoCo-300ep",
                        help="Directory containing preprocessed NPZ files")
    parser.add_argument("--model_path", type=str,
                        default="/home/ashmithandoo/projects/git/MedSAM2/models/medsam2_vit_t_b16.pt",
                        help="Path to MedSAM2 model weights")
    parser.add_argument("--output_base", type=str,
                        default="/home/ashmithandoo/projects/git/MedSAM2/data/humandata_preprocessed/predictions/MoCo-300ep_results",
                        help="Base directory for output")
    parser.add_argument("--model_type", type=str,
                        default=None,
                        help="Model type from models package (e.g. UNet, VSS_Net, etc.)")
    parser.add_argument("--window_size", type=int, default=8,
                        help="Width of the sliding window in frames (default: 8).")
    parser.add_argument("--verbose", action="store_true", help="Print detailed per-slice information")
    
    # New args passed by the bash script
    parser.add_argument("--data_dir", type=str, default=None, help="Same as input_dir")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory containing best_model.pth")
    parser.add_argument("--output_dir", type=str, default=None, help="Same as output_base")
    parser.add_argument("--dias_repo", type=str, default="", help="Ignored")
    parser.add_argument("--supervision", type=str, default="", help="Ignored")
    parser.add_argument("--wsl_variant", type=str, default="", help="Ignored")
    
    # Parse known args just in case there are others
    args, _ = parser.parse_known_args()

    # Map new args to old args
    input_dir = args.data_dir if args.data_dir else args.input_dir
    output_base = args.output_dir if args.output_dir else args.output_base
    
    # Handle checkpoint_dir -> model_path
    if args.checkpoint_dir:
        if os.path.isdir(args.checkpoint_dir):
            model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
        else:
            model_path = args.checkpoint_dir
    else:
        model_path = args.model_path
        
    # Auto-infer model_type if not provided or if we're running from DIAS script
    model_type = args.model_type
    if not model_type and args.checkpoint_dir:
        known_models = [
            "UNet_Nested_3D", "Att_UNet_3D", "Res_UNet_3D", "CSNet_3D", "FR_UNet_3D", "UNet_3D",
            "UNet_Nested", "Att_UNet", "Res_UNet", "CSNet", "FR_UNet", "MAA_Net", "SVS_Net", 
            "QS_UNet", "VSS_Net", "ST_UNet", "UNet_CCT", "UNet_DP", "UNet", "IPN", "PSC"
        ]
        # Check longest first
        for km in known_models:
            if km in args.checkpoint_dir:
                model_type = km
                break
        if not model_type:
            model_type = "UNet" # fallback
    elif not model_type:
        model_type = "UNet"

    print(f"Running inference with model_type={model_type} from {model_path}")

    # Run inference
    all_stats = run_inference(
        input_dir=input_dir,
        model_path=model_path,
        output_base=output_base,
        window_size=args.window_size,
        verbose=args.verbose,
        model_type=model_type
    )

    # Print summary
    print_summary(all_stats)
