 CUDA_VISIBLE_DEVICES=0 python training/train.py \
     -c configs/sam2.1_hiera_tiny512_DIAS_Standard \
     --output-path ./exp_log/MedSAM2_DIAS_Standard/take3 \
     --use-cluster 0 \
     --num-gpus 1 \
     --num-nodes 1

#CUDA_VISIBLE_DEVICES=0 python training/train.py \
#  -c configs/sam2.1_hiera_tiny512_DIAS_SALE.yaml \
#  --output-path ./exp_log/MedSAM2_DIAS_SALE/take2 \
#  --use-cluster 0 \
#  --num-gpus 1 \
#  --num-nodes 1
#
#CUDA_VISIBLE_DEVICES=0 python training/train.py \
#  -c configs/sam2.1_hiera_tiny512_DIAS_RDFA.yaml \
#  --output-path ./exp_log/MedSAM2_DIAS_RDFA/take2 \
#  --use-cluster 0 \
#  --num-gpus 1 \
#  --num-nodes 1
