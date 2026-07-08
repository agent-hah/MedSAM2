python run_inference.py rat \
    -c configs/sam2.1_hiera_t512.yaml \
    -ckpt exp_log/MedSAM2_DIAS_Standard/take3/checkpoints/checkpoint.pt \
    -d data/medsam_preprocessed/RatData_NPZ \
    -o results/RatData_Inference/medsam2
