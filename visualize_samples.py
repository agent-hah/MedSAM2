import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def create_graphic(base_dir, samples, output_path="samples_graphic.png"):
    color_imgs_dir = os.path.join(base_dir, "color_imgs")
    gt_dir = os.path.join(base_dir, "gt")
    
    num_samples = len(samples)
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8), squeeze=False)
    
    for i, sample in enumerate(samples):
        # Paths
        gt_path = os.path.join(gt_dir, f"{sample}.npy")
        color_path = os.path.join(color_imgs_dir, f"{sample}.png")
        
        # Load data
        try:
            gt_data = np.load(gt_path)
            if gt_data.ndim == 3 and gt_data.shape[0] in [1, 3]:
                gt_data = gt_data.transpose(1, 2, 0)
            if gt_data.ndim == 3 and gt_data.shape[-1] == 1:
                gt_data = gt_data[..., 0]
        except Exception as e:
            print(f"Could not load GT for {sample}: {e}")
            gt_data = np.zeros((256, 256)) # Placeholder

        try:
            color_img = Image.open(color_path)
        except Exception as e:
            print(f"Could not load color image for {sample}: {e}")
            color_img = Image.new('RGB', (256, 256), color='black') # Placeholder

        # Plot Ground Truth
        ax = axes[0, i]
        if gt_data.ndim == 2:
            ax.imshow(gt_data, cmap='gray')
        else:
            ax.imshow(gt_data)
        ax.set_title(sample)
        if i == 0:
            ax.set_ylabel("Ground Truth", fontsize=14, rotation=90, labelpad=10, va='center')
        ax.set_xticks([])
        ax.set_yticks([])

        # Plot Color Image
        ax = axes[1, i]
        ax.imshow(color_img)
        if i == 0:
            ax.set_ylabel("Color Image", fontsize=14, rotation=90, labelpad=10, va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved graphic to {output_path}")

if __name__ == "__main__":
    base_dir = "exp_log/MedSAM2_DIAS_Standard/take3/eval7"
    # Select some sample names based on the directory contents
    samples = ["s40", "s41", "s42", "s43", "s44", "s45", "s46"] 
    create_graphic(base_dir, samples)
