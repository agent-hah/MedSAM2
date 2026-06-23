import os
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from training.utils.checkpoint_utils import load_state_dict_into_model


def run_inference(config_name, checkpoint_path, test_data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialize Hydra and Model
    initialize_config_module("sam2", version_base="1.2")
    cfg = compose(config_name=config_name)

    print(f"Building model using config: {config_name}")
    model = instantiate(cfg.trainer.model, _convert_="all")
    model.cuda()
    model.eval()

    # 2. Load Checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    with open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    load_state_dict_into_model(model, checkpoint["model"])

    # 3. Run Inference Loop
    print(f"Running inference on dataset: {test_data_dir}")
    test_files = [f for f in os.listdir(test_data_dir) if f.endswith(".npz")]

    with torch.no_grad():
        for filename in tqdm(test_files):
            seq_id = filename.split(".")[0]  # e.g., 's42'
            file_path = os.path.join(test_data_dir, filename)

            # Load your preprocessed test data (matching the 'imgs' key from preprocessing)
            data = np.load(file_path)
            img_tensor = torch.from_numpy(data["imgs"]).float().cuda()

            # Ensure proper shape: (Batch, Channels, H, W)
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.unsqueeze(0)

            # Forward pass
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                outputs = model(img_tensor)

                # Extract the probability mask for the foreground class
                if isinstance(outputs, dict):
                    logits = outputs["pred_masks"]
                else:
                    logits = outputs

                prob_mask = torch.softmax(logits, dim=1)[:, 1, :, :]

            # Convert to binary mask (0 and 255) for OpenCV
            prob_mask_np = prob_mask.cpu().numpy().squeeze()
            binary_mask = np.where(prob_mask_np >= 0.5, 255, 0).astype(np.uint8)

            # Save the prediction
            save_path = os.path.join(output_dir, f"pred_{seq_id}.png")
            cv2.imwrite(save_path, binary_mask)

    print(f"Inference complete! Predictions saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MedSAM2 Inference on DIAS Test Data"
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Hydra config name (e.g., sam2.1_hiera_tiny512_DIAS_Standard)",
    )
    parser.add_argument(
        "-ckpt", "--checkpoint", required=True, help="Path to checkpoint.pt"
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        required=True,
        help="Path to preprocessed DIAS_Test_NPZ folder",
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Path to save prediction PNGs"
    )

    args = parser.parse_args()

    run_inference(
        config_name=args.config,
        checkpoint_path=args.checkpoint,
        test_data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

