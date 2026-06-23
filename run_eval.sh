python medsam2_infer_dias.py \
  -c configs/sam2.1_hiera_tiny512_DIAS_Standard \
  -ckpt exp_log/MedSAM2_DIAS_Standard/checkpoints/checkpoint.pt \
  -d data/medsam_preprocessed/DIAS_Test_NPZ \
  -o exp_log/MedSAM2_DIAS_Standard/predictions

python medsam2_infer_dias.py \
  -c configs/sam2.1_hiera_tiny512_DIAS_SALE \
  -ckpt exp_log/MedSAM2_DIAS_SALE/checkpoints/checkpoint.pt \
  -d data/medsam_preprocessed/DIAS_Test_NPZ \
  -o exp_log/MedSAM2_DIAS_SALE/predictions

python medsam2_infer_dias.py \
  -c configs/sam2.1_hiera_tiny512_DIAS_RDFA \
  -ckpt exp_log/MedSAM2_DIAS_RDFA/checkpoints/checkpoint.pt \
  -d data/medsam_preprocessed/DIAS_Test_NPZ \
  -o exp_log/MedSAM2_DIAS_RDFA/predictions

