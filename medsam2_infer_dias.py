import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor_npz


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


def run_inference(config_name, checkpoint_path, test_data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Building model using config: {config_name}")
    print(f"Loading checkpoint from {checkpoint_path}...")
    predictor = build_sam2_video_predictor_npz(config_name, checkpoint_path)

    print(f"Running fully automatic inference on dataset: {test_data_dir}")
    test_files = [f for f in os.listdir(test_data_dir) if f.endswith(".npz")]

    with torch.no_grad():
        for filename in tqdm(test_files):
            seq_id = filename.split(".")[0]
            file_path = os.path.join(test_data_dir, filename)

            data = np.load(file_path, allow_pickle=True)
            img_3D_ori = data["imgs"]

            video_height, video_width = img_3D_ori.shape[1:3]
            img_resized = resize_grayscale_to_rgb_and_resize(img_3D_ori, 512)
            img_resized = img_resized / 255.0
            img_resized = torch.from_numpy(img_resized).cuda()

            img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[
                :, None, None
            ].cuda()
            img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[
                :, None, None
            ].cuda()
            img_resized -= img_mean
            img_resized /= img_std

            with torch.autocast("cuda", dtype=torch.bfloat16):
                inference_state = predictor.init_state(
                    img_resized, video_height, video_width
                )

                # ==========================================
                # FULLY AUTOMATIC PROMPTING PHASE (Center-Point Prior)
                # ==========================================
                # Calculate the exact center of the original video dimensions
                center_x = video_width / 2.0
                center_y = video_height / 2.0

                # Create a single positive point prompt
                points = np.array([[center_x, center_y]], dtype=np.float32)
                labels = np.array([1], dtype=np.int32)

                # Feed the automatic center coordinate into the predictor
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=1,
                    points=points,
                    labels=labels,
                )

                # Initialize output volume and populate frame 0 with the predicted mask
                binary_mask = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                segs_3D = np.zeros(img_3D_ori.shape, dtype=np.uint8)
                segs_3D[0] = binary_mask * 255

                # ==========================================
                # VIDEO PROPAGATION PHASE
                # ==========================================
                # Track forward through the remaining slices (frames 1-7)
                for (
                    out_frame_idx,
                    out_obj_ids,
                    out_mask_logits,
                ) in predictor.propagate_in_video(
                    inference_state, start_frame_idx=0, reverse=False
                ):
                    binary_mask = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                    segs_3D[out_frame_idx] = binary_mask * 255

            save_path = os.path.join(output_dir, f"pred_{seq_id}.npz")
            np.savez_compressed(save_path, segs=segs_3D)

    print(f"Inference complete! Predictions saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Fully Automatic MedSAM2 Inference on DIAS Test Data"
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Hydra config name (e.g., sam2.1_hiera_t512.yaml)",
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
        "-o", "--output_dir", required=True, help="Path to save prediction NPZs"
    )

    args = parser.parse_args()

    run_inference(
        config_name=args.config,
        checkpoint_path=args.checkpoint,
        test_data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
