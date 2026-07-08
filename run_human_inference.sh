python infer_humandata.py \
    -c configs/sam2.1_hiera_t512.yaml \
    -ckpt exp_log/MedSAM2_DIAS_Standard/take3/checkpoints/checkpoint.pt \
    -d data/medsam_preprocessed/HumanData_NPZ \
    -o results/HumanData_Inference/medsam2