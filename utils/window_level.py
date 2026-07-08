import os
import argparse
import glob
import numpy as np
import pydicom
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import zipfile
import tempfile

def read_image(path):
    if path.lower().endswith(".dcm"):
        ds = pydicom.dcmread(path, stop_before_pixels=False)
        return ds.pixel_array
    elif path.lower().endswith(".png"):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return None

def analyze_dataset(data_dir, num_samples=100):
    """
    Analyzes a dataset to recommend Window Center (C) and Window Width (W).
    """
    # Find all dicom and png files
    files = glob.glob(os.path.join(data_dir, "**", "*.dcm"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True)
        
    if not files:
        print(f"No .dcm or .png files found in {data_dir}")
        return

    # Randomly sample to speed up discovery
    np.random.seed(42)
    sample_files = np.random.choice(files, min(num_samples, len(files)), replace=False)
    
    all_pixels = []
    
    for f in tqdm(sample_files, desc=f"Analyzing {os.path.basename(data_dir)}"):
        try:
            arr = read_image(f)
            if arr is not None:
                # Subsample pixels to save memory
                arr = arr.astype(np.float32)
                all_pixels.append(np.random.choice(arr.flatten(), min(10000, arr.size), replace=False))
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_pixels:
        print("No valid images could be read.")
        return

    all_pixels = np.concatenate(all_pixels)
    
    # Calculate percentiles to ignore outliers
    p1 = np.percentile(all_pixels, 1)
    p5 = np.percentile(all_pixels, 5)
    p95 = np.percentile(all_pixels, 95)
    p99 = np.percentile(all_pixels, 99)
    
    mean_val = np.mean(all_pixels)
    std_val = np.std(all_pixels)
    
    print("\n" + "="*50)
    print(f"Results for dataset: {data_dir}")
    print(f"  Min/Max: {np.min(all_pixels):.2f} / {np.max(all_pixels):.2f}")
    print(f"  Mean (Std): {mean_val:.2f} ({std_val:.2f})")
    print(f"  1st - 99th Percentile: {p1:.2f} to {p99:.2f}")
    print(f"  5th - 95th Percentile: {p5:.2f} to {p95:.2f}")
    
    # Recommend Window Level based on 1st-99th percentile
    rec_center = (p1 + p99) / 2
    rec_width = p99 - p1
    print("\nRecommended Settings (1st to 99th percentile):")
    print(f"  Window Center (C) = {rec_center:.2f}")
    print(f"  Window Width (W)  = {rec_width:.2f}")
    
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover optimal Window Level settings.")
    parser.add_argument("--human_dir", type=str, default="data/medsam_preprocessed/HumanData/HumanData", help="Path to HumanData")
    parser.add_argument("--rat_dir", type=str, default="data/medsam_preprocessed/RatData/RatData", help="Path to RatData")
    parser.add_argument("--dias_dir", type=str, default="~/projects/lab/DIAS/d_data/DIAS/training", help="Path to DIAS data")
    
    args = parser.parse_args()
    
    if os.path.exists(os.path.expanduser(args.human_dir)):
        analyze_dataset(os.path.expanduser(args.human_dir))
    
    if os.path.exists(os.path.expanduser(args.rat_dir)):
        analyze_dataset(os.path.expanduser(args.rat_dir))
        
    if os.path.exists(os.path.expanduser(args.dias_dir)):
        analyze_dataset(os.path.expanduser(args.dias_dir))
