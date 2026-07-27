import os
import cv2
from pathlib import Path

RESULTS_DIR = Path("results")

# The comparison models from generate_comparison_pptx.py
COMPARISON_PAIRS = [
    ("fsl_ST_UNet", "fsl_VSS_Net"),
    ("ssl_10_60_ite_3_student", "ssl_10_60_SDA_ite_3_student"),
    ("wsl_RDFA_wsl_train_sscr_ablation", "wsl_SALE_wsl_train_sscr_ablation"),
]

# The selected cases from generate_comparison_pptx.py
SELECTED_CASES = [
    # Rat cases
    ("RatData_DIAS_Inference", "SRG_Angio/SRG_24_Cios/Aorta_1_dsa"),
    ("RatData_DIAS_Inference", "SRG_Angio/SRG_29_Cios/Aorta_dsa"),
    # Human cases
    ("HumanData_DIAS_Inference", "82A1/DSA/Segment_8_day_of_Tx_1"),
    ("HumanData_DIAS_Inference", "82A1/DSA/Left_Hepatic_Artery_Segment_4_Treatment_2/LHA_Treatment_2_Slightly_More_Proximal"),
]

def generate_mp4_for_series(image_paths, output_mp4_path, fps=5):
    """Combines a list of image paths into an MP4 video."""
    if not image_paths:
        return False
        
    try:
        # Read first image to get dimensions
        first_img = cv2.imread(str(image_paths[0]))
        if first_img is None:
            return False
            
        height, width, layers = first_img.shape
        
        # mp4v codec is widely supported for mp4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video = cv2.VideoWriter(str(output_mp4_path), fourcc, fps, (width, height))
        
        for p in image_paths:
            img = cv2.imread(str(p))
            if img is not None:
                # Resize if necessary to match the first frame exactly
                if img.shape[:2] != (height, width):
                    img = cv2.resize(img, (width, height))
                video.write(img)
                
        video.release()
        return True
    except Exception as e:
        print(f"Error creating {output_mp4_path}: {e}")
        return False

def main():
    # 1. Process DIAS models
    dias_models = set()
    for m1, m2 in COMPARISON_PAIRS:
        dias_models.add(m1)
        dias_models.add(m2)
        
    for category, case_path in SELECTED_CASES:
        for model in dias_models:
            case_dir = RESULTS_DIR / category / model / case_path
            windows_dir = case_dir / "windows"
            
            if not windows_dir.exists():
                print(f"Skipping DIAS (not found): {windows_dir}")
                continue
                
            # Find all comparison.png in window_* dirs
            image_paths = sorted(list(windows_dir.glob("window_*/comparison.png")))
            if not image_paths:
                print(f"No comparison images found in {windows_dir}")
                continue
                
            out_mp4 = case_dir / "slice_movie.mp4"
            print(f"Generating MP4 ({len(image_paths):02d} frames) -> {out_mp4}")
            generate_mp4_for_series(image_paths, out_mp4, fps=5)

    # 2. Process MedSAM2 models
    for category, case_path in SELECTED_CASES:
        # Map DIAS category to standard category (e.g. RatData_DIAS_Inference -> RatData_Inference)
        medsam2_category = category.replace("_DIAS", "")
        
        case_dir = RESULTS_DIR / medsam2_category / "medsam2" / case_path
        slices_dir = case_dir / "slices"
        
        if not slices_dir.exists():
            print(f"Skipping MedSAM2 (not found): {slices_dir}")
            continue
            
        # Find all slice_*.png
        image_paths = sorted(list(slices_dir.glob("slice_*.png")))
        if not image_paths:
            print(f"No slice images found in {slices_dir}")
            continue
            
        out_mp4 = case_dir / "slice_movie.mp4"
        print(f"Generating MP4 ({len(image_paths):02d} frames) -> {out_mp4}")
        generate_mp4_for_series(image_paths, out_mp4, fps=10) # Faster fps for MedSAM2 since it has more frames

if __name__ == "__main__":
    print("Starting MP4 generation...")
    main()
    print("Done!")
