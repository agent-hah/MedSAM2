#python medsam2_infer_dias.py \
#  -c configs/sam2.1_hiera_t512.yaml \
#  -ckpt exp_log/MedSAM2_DIAS_Standard/checkpoints/checkpoint.pt \
#  -d data/medsam_preprocessed/DIAS_Test_NPZ \
#  -o exp_log/MedSAM2_DIAS_Standard/predictions
#
#python medsam2_infer_dias.py \
#  -c configs/sam2.1_hiera_t512.yaml \
#  -ckpt exp_log/MedSAM2_DIAS_SALE/checkpoints/checkpoint.pt \
#  -d data/medsam_preprocessed/DIAS_Test_NPZ \
#  -o exp_log/MedSAM2_DIAS_SALE/predictions
#
#python medsam2_infer_dias.py \
#  -c configs/sam2.1_hiera_t512.yaml \
#  -ckpt exp_log/MedSAM2_DIAS_RDFA/checkpoints/checkpoint.pt \
#  -d data/medsam_preprocessed/DIAS_Test_NPZ \
#  -o exp_log/MedSAM2_DIAS_RDFA/predictions

python evaluate_medsam2.py \
  -c configs/sam2.1_hiera_t512.yaml \
  -ckpt exp_log/MedSAM2_DIAS_Standard/take3/checkpoints/checkpoint.pt \
  -d data/medsam_preprocessed/DIAS_Test_NPZ \
  -o exp_log/MedSAM2_DIAS_Standard/take3/eval7

#python find_perfect_threshold.py \
#  -c configs/sam2.1_hiera_t512.yaml \
#  -ckpt exp_log/MedSAM2_DIAS_Standard/take3/checkpoints/checkpoint.pt \
#  -d data/medsam_preprocessed/DIAS_Standard_NPZ \

#python evaluate_medsam2.py \
#  -c configs/sam2.1_hiera_t512.yaml \
#  -ckpt exp_log/MedSAM2_DIAS_SALE/take2/checkpoints/checkpoint.pt \
#  -d data/medsam_preprocessed/DIAS_Test_NPZ \
#  -o exp_log/MedSAM2_DIAS_SALE/take2/eval2
#
#python evaluate_medsam2.py \
#  -c configs/sam2.1_hiera_t512.yaml \
#  -ckpt exp_log/MedSAM2_DIAS_RDFA/take2/checkpoints/checkpoint.pt \
#  -d data/medsam_preprocessed/DIAS_Test_NPZ \
#  -o exp_log/MedSAM2_DIAS_RDFA/take2/eval2

