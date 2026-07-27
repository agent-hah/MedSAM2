import os
import json
import cv2
import numpy as np
from skimage.filters import frangi, threshold_otsu
from skimage.restoration import denoise_tv_bregman
from scipy.ndimage import distance_transform_edt

def apply_window_level(image, center, width):
    """
    Applies Window Level to a raw image (e.g. DICOM).
    Values below (center - width/2) are clamped to 0.
    Values above (center + width/2) are clamped to 255.
    Output is uint8.
    """
    img = image.astype(np.float32)
    min_val = center - (width / 2.0)
    max_val = center + (width / 2.0)
    
    # Clip and scale to 0-255
    img = np.clip(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val) * 255.0
    return img.astype(np.uint8)

def apply_tv_denoising(image, weight=0.35):
    """
    Applies Total Variation (TV) denoising to the image.
    Image is expected to be in range [0, 255] or [0, 1].
    Returns float32 image in range [0, 1].
    """
    # denoise_tv_bregman returns float64 in range [0, 1] if input is float or uint8.
    # It handles scaling automatically if the image is float.
    img_float = image.astype(np.float32) / 255.0 if image.max() > 1.0 else image.astype(np.float32)
    denoised = denoise_tv_bregman(img_float, weight=weight)
    return denoised.astype(np.float32)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Expects a uint8 image [0, 255]. Returns a uint8 image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def apply_frangi_filter(image, black_ridges=False, dynamic_sigmas=False):
    """
    Applies Frangi (Hessian) filter to enhance vessel-like structures.
    Expects input to be a 2D array. Returns output normalized to [0, 255] as uint8.
    By default detects bright ridges (black_ridges=False) as typically seen in MRA/CTA.
    """
    # Use a reasonable default range for vessel sizes in pixels
    sigmas = range(1, 10, 2)

    # frangi handles intensity scaling automatically, usually returns values in a very small range
    filtered = frangi(image, sigmas=sigmas, black_ridges=black_ridges)
    
    f_min, f_max = filtered.min(), filtered.max()
    if f_max > f_min:
        filtered = (filtered - f_min) / (f_max - f_min) * 255.0
    else:
        filtered = np.zeros_like(filtered)
        
    return filtered.astype(np.uint8)

def get_crop_roi(sequence_frames, seq_name, cache_file="crop_cache.json", clear_cache=False, skip_cropping=False):
    """
    Gets cropping coordinates (x, y, w, h) for a sequence.
    If cached, returns the cached coordinates.
    Otherwise, pops up an interactive cv2 window to let the user scrub through frames and select the ROI.
    """
    if skip_cropping:
        h, w = sequence_frames[0].shape[:2]
        return [0, 0, w, h]

    cache = {}
    if not clear_cache and os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            try:
                cache = json.load(f)
            except json.JSONDecodeError:
                cache = {}
                
    if seq_name in cache:
        return cache[seq_name]
        
    # Process all frames for display (to uint8)
    display_frames = []
    
    # sequence_frames could be a list of arrays or a 3D numpy array
    for frame in sequence_frames:
        display_img = frame.copy()
        if display_img.dtype != np.uint8:
            if display_img.max() > 255:
                # Basic windowing for display
                mean_v = np.mean(display_img)
                std_v = np.std(display_img)
                display_img = apply_window_level(display_img, center=mean_v, width=std_v*4)
            else:
                display_img = (display_img * 255).astype(np.uint8)
        display_frames.append(display_img)
            
    print(f"\nSelect ROI for sequence: {seq_name}")
    print("Use 'a' / 'd' to scrub through frames.")
    print("Press SPACE or ENTER on the best frame to start cropping.")
    print("Press 'c' to cancel/skip crop entirely.")
    
    h, w = display_frames[0].shape[:2]
    
    if not os.environ.get("DISPLAY"):
        print(f"No display found. Skipping interactive ROI selection for {seq_name}.")
        final_roi = [0, 0, w, h]
        cache[seq_name] = final_roi
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=4)
        return final_roi
        
    max_dim = 800
    scale = 1.0
    if h > max_dim or w > max_dim:
        scale = max_dim / max(h, w)
        display_frames = [cv2.resize(f, (int(w * scale), int(h * scale))) for f in display_frames]
        
    window_name = f"Preview - {seq_name} (Scrub before crop)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    idx = 0
    num_frames = len(display_frames)
    
    while True:
        img_show = display_frames[idx].copy()
        cv2.putText(img_show, f"Frame {idx+1}/{num_frames} - Press ENTER to crop here", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, 1.0 * scale), (255, 255, 255), 2)
        cv2.imshow(window_name, img_show)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('a') or key == 81:  # 81 is left arrow on some systems
            idx = max(0, idx - 1)
        elif key == ord('d') or key == 83:  # 83 is right arrow
            idx = min(num_frames - 1, idx + 1)
        elif key == 13 or key == 32:  # Enter or Space
            break
        elif key == ord('c') or key == 27:  # 'c' or ESC
            cv2.destroyWindow(window_name)
            final_roi = [0, 0, w, h]
            cache[seq_name] = final_roi
            with open(cache_file, "w") as f:
                json.dump(cache, f, indent=4)
            return final_roi

    cv2.destroyWindow(window_name)
    
    roi = cv2.selectROI(f"Select ROI - {seq_name}", display_frames[idx], showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    
    # cv2.selectROI returns (x, y, w, h). If (0, 0, 0, 0), user cancelled.
    rx, ry, rw, rh = roi
    
    if rw > 0 and rh > 0:
        # Scale back coordinates to original resolution
        rx = int(rx / scale)
        ry = int(ry / scale)
        rw = int(rw / scale)
        rh = int(rh / scale)
        final_roi = [rx, ry, rw, rh]
    else:
        # User cancelled, default to full image
        final_roi = [0, 0, w, h]
        
    cache[seq_name] = final_roi
    
    # Save cache
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=4)
        
    return final_roi

def crop_image(image, roi):
    """
    Crops an image given an ROI [x, y, w, h].
    """
    x, y, w, h = roi
    return image[y:y+h, x:x+w]

def apply_full_preprocessing(image, window_center=None, window_width=None, tv_weight=0.0, use_frangi=False, dynamic_sigmas=False, black_ridges=False, use_clahe=True, clahe_clip=2.0, clahe_tile=(8,8), use_minmax=True):
    """
    Convenience function that applies the full preprocessing pipeline to a single 2D slice.
    """
    img = image.copy()
    
    if window_center is not None and window_width is not None:
        img = apply_window_level(img, window_center, window_width)
    elif use_minmax:
        # If no window given but it's raw dicom, just min-max normalize
        if img.max() > 255:
            if img.max() == img.min():
                img = np.zeros_like(img, dtype=np.uint8)
            else:
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    else:
        img = img.astype(np.float32)
            
    if tv_weight > 0:
        img = apply_tv_denoising(img, weight=tv_weight)
        
    if use_frangi:
        img = apply_frangi_filter(img, black_ridges=black_ridges, dynamic_sigmas=dynamic_sigmas)
    else:
        # If no frangi, ensure it's uint8
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
            
    if use_clahe:
        if img.dtype != np.uint8:
            raise ValueError("CLAHE requires a uint8 image. Ensure use_minmax is True when using CLAHE.")
        img = apply_clahe(img, clip_limit=clahe_clip, tile_grid_size=clahe_tile)
        
    return img
