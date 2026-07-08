# PowerShell script to run DIAS models inference on RatData
# Ensure you have activated the environment: conda activate medsam2

# Run FSL model: Att_UNet
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/fully_supervised/Att_UNet/Att_UNet_NN_260527_222210 --supervision fsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/fsl_Att_UNet

# Run FSL model: CSNet
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/fully_supervised/CSNet/CSNet_NN_260527_231159 --supervision fsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/fsl_CSNet

# Run FSL model: CSNet_3D
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/fully_supervised/CSNet_3D/CSNet_3D_NN_260528_122908 --supervision fsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/fsl_CSNet_3D

# Run FSL model: FR_UNet
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/fully_supervised/FR_UNet/FR_UNet_NN_260529_184205 --supervision fsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/fsl_FR_UNet

# Run FSL model: FR_UNet_3D
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/fully_supervised/FR_UNet_3D/FR_UNet_3D_NN_260528_094634 --supervision fsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/fsl_FR_UNet_3D

# Run FSL model: IPN
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/fully_supervised/IPN/IPN_NN_260606_041238 --supervision fsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/fsl_IPN

# Run FSL model: MAA_Net
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/fully_supervised/MAA_Net/MAA_Net_NN_260606_084342 --supervision fsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/fsl_MAA_Net

# Run FSL model: PSC
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/fully_supervised/PSC/PSC_NN_260528_220236 --supervision fsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/fsl_PSC

# Run FSL model: Res_UNet
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/fully_supervised/Res_UNet/Res_UNet_NN_260528_011013 --supervision fsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/fsl_Res_UNet

# Run FSL model: Res_UNet_3D
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/fully_supervised/Res_UNet_3D/Res_UNet_3D_NN_260528_182747 --supervision fsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/fsl_Res_UNet_3D

# Run FSL model: ST_UNet
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/fully_supervised/ST_UNet/ST_UNet_NN_260529_194151 --supervision fsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/fsl_ST_UNet

# Run FSL model: SVS_Net
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/fully_supervised/SVS_Net/SVS_Net_NN_260528_230109 --supervision fsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/fsl_SVS_Net

# Run FSL model: UNet
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/fully_supervised/UNet/UNet_NN_260529_175103 --supervision fsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/fsl_UNet

# Run FSL model: UNet_3D
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/fully_supervised/UNet_3D/UNet_3D_NN_260528_021342 --supervision fsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/fsl_UNet_3D

# Run FSL model: UNet_Nested
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/fully_supervised/UNet_Nested/UNet_Nested_NN_260528_000323 --supervision fsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/fsl_UNet_Nested

# Run FSL model: UNet_Nested_3D
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/fully_supervised/UNet_Nested_3D/UNet_Nested_3D_NN_260528_195427 --supervision fsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/fsl_UNet_Nested_3D

# Run FSL model: VSS_Net
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/fully_supervised/VSS_Net/VSS_Net_NN_260527_210416 --supervision fsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/fsl_VSS_Net

# Run FSL model: VSS_Net
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/fully_supervised/VSS_Net/VSS_Net_NN_260615_092204 --supervision fsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/fsl_VSS_Net

# Run FSL model: VSS_Net
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/fully_supervised/VSS_Net/VSS_Net_NN_260618_125334 --supervision fsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/fsl_VSS_Net

# Run SSL model: 10_30_SDA_ite_1_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/10_30_SDA/FR_UNet_260531_074329/ite_1_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_10_30_SDA_ite_1_student

# Run SSL model: 10_30_SDA_ite_1_teacher
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/10_30_SDA/FR_UNet_260531_074329/ite_1_teacher --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_10_30_SDA_ite_1_teacher

# Run SSL model: 10_30_SDA_ite_2_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/10_30_SDA/FR_UNet_260531_074329/ite_2_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_10_30_SDA_ite_2_student

# Run SSL model: 10_30_SDA_ite_3_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/10_30_SDA/FR_UNet_260531_074329/ite_3_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_10_30_SDA_ite_3_student

# Run SSL model: 10_60_ite_1_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/10_60/FR_UNet_260602_094212/ite_1_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_10_60_ite_1_student

# Run SSL model: 10_60_ite_1_teacher
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/10_60/FR_UNet_260602_094212/ite_1_teacher --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_10_60_ite_1_teacher

# Run SSL model: 10_60_ite_2_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/10_60/FR_UNet_260602_094212/ite_2_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_10_60_ite_2_student

# Run SSL model: 10_60_ite_3_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/10_60/FR_UNet_260602_094212/ite_3_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_10_60_ite_3_student

# Run SSL model: 10_60_SDA_ite_1_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/10_60_SDA/FR_UNet_260602_190000/ite_1_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_10_60_SDA_ite_1_student

# Run SSL model: 10_60_SDA_ite_1_teacher
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/10_60_SDA/FR_UNet_260602_190000/ite_1_teacher --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_10_60_SDA_ite_1_teacher

# Run SSL model: 10_60_SDA_ite_2_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/10_60_SDA/FR_UNet_260602_190000/ite_2_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_10_60_SDA_ite_2_student

# Run SSL model: 10_60_SDA_ite_3_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/10_60_SDA/FR_UNet_260602_190000/ite_3_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_10_60_SDA_ite_3_student

# Run SSL model: 1_30_SDA_ite_1_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/1_30_SDA/FR_UNet_260530_043833/ite_1_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_1_30_SDA_ite_1_student

# Run SSL model: 1_30_SDA_ite_1_teacher
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/1_30_SDA/FR_UNet_260530_043833/ite_1_teacher --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_1_30_SDA_ite_1_teacher

# Run SSL model: 1_30_SDA_ite_2_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/1_30_SDA/FR_UNet_260530_043833/ite_2_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_1_30_SDA_ite_2_student

# Run SSL model: 1_30_SDA_ite_3_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/1_30_SDA/FR_UNet_260530_043833/ite_3_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_1_30_SDA_ite_3_student

# Run SSL model: 1_60_ite_1_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/1_60/FR_UNet_260531_205724/ite_1_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_1_60_ite_1_student

# Run SSL model: 1_60_ite_1_teacher
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/1_60/FR_UNet_260531_205724/ite_1_teacher --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_1_60_ite_1_teacher

# Run SSL model: 1_60_ite_2_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/1_60/FR_UNet_260531_205724/ite_2_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_1_60_ite_2_student

# Run SSL model: 1_60_ite_3_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/1_60/FR_UNet_260531_205724/ite_3_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_1_60_ite_3_student

# Run SSL model: 1_60_SDA_ite_1_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/1_60_SDA/FR_UNet_260601_060734/ite_1_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_1_60_SDA_ite_1_student

# Run SSL model: 1_60_SDA_ite_1_teacher
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/1_60_SDA/FR_UNet_260601_060734/ite_1_teacher --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_1_60_SDA_ite_1_teacher

# Run SSL model: 1_60_SDA_ite_2_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/1_60_SDA/FR_UNet_260601_060734/ite_2_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_1_60_SDA_ite_2_student

# Run SSL model: 1_60_SDA_ite_3_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/1_60_SDA/FR_UNet_260601_060734/ite_3_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_1_60_SDA_ite_3_student

# Run SSL model: 3_30_SDA_ite_1_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/3_30_SDA/FR_UNet_260530_162356/ite_1_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_3_30_SDA_ite_1_student

# Run SSL model: 3_30_SDA_ite_1_teacher
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/3_30_SDA/FR_UNet_260530_162356/ite_1_teacher --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_3_30_SDA_ite_1_teacher

# Run SSL model: 3_30_SDA_ite_2_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/3_30_SDA/FR_UNet_260530_162356/ite_2_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_3_30_SDA_ite_2_student

# Run SSL model: 3_30_SDA_ite_3_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/3_30_SDA/FR_UNet_260530_162356/ite_3_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_3_30_SDA_ite_3_student

# Run SSL model: 3_60_ite_1_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/3_60/FR_UNet_260601_161037/ite_1_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_3_60_ite_1_student

# Run SSL model: 3_60_ite_1_teacher
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/3_60/FR_UNet_260601_161037/ite_1_teacher --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_3_60_ite_1_teacher

# Run SSL model: 3_60_ite_2_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/3_60/FR_UNet_260601_161037/ite_2_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_3_60_ite_2_student

# Run SSL model: 3_60_ite_3_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/3_60/FR_UNet_260601_161037/ite_3_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_3_60_ite_3_student

# Run SSL model: 3_60_SDA_ite_1_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/3_60_SDA/FR_UNet_260602_020818/ite_1_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_3_60_SDA_ite_1_student

# Run SSL model: 3_60_SDA_ite_1_teacher
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/3_60_SDA/FR_UNet_260602_020818/ite_1_teacher --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_3_60_SDA_ite_1_teacher

# Run SSL model: 3_60_SDA_ite_2_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/3_60_SDA/FR_UNet_260602_020818/ite_2_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_3_60_SDA_ite_2_student

# Run SSL model: 3_60_SDA_ite_3_student
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/semi_supervised/3_60_SDA/FR_UNet_260602_020818/ite_3_student --supervision ssl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/ssl_3_60_SDA_ite_3_student

# Run WSL model: RDFA_wsl_train_DMPLS
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/weakly_supervised/RDFA/wsl_train_DMPLS/UNet_CCT_RDFA_260606_185011 --supervision wsl --wsl_variant dmpls --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/wsl_RDFA_wsl_train_DMPLS

# Run WSL model: RDFA_wsl_train_EMA_sscr
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/weakly_supervised/RDFA/wsl_train_EMA_sscr/RDFA_UNet_260605_153256 --supervision wsl --wsl_variant ema --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/wsl_RDFA_wsl_train_EMA_sscr

# Run WSL model: RDFA_wsl_train_GatedCRFLoss
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/weakly_supervised/RDFA/wsl_train_GatedCRFLoss/UNet_RDFA_260603_223747 --supervision wsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/wsl_RDFA_wsl_train_GatedCRFLoss

# Run WSL model: RDFA_wsl_train_Inter&Intra_Class
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/weakly_supervised/RDFA/wsl_train_Inter&Intra_Class/UNet_RDFA_260604_001144 --supervision wsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/wsl_RDFA_wsl_train_Inter&Intra_Class

# Run WSL model: RDFA_wsl_train_entropy_mini
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/weakly_supervised/RDFA/wsl_train_entropy_mini/UNet_RDFA_260603_210412 --supervision wsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/wsl_RDFA_wsl_train_entropy_mini

# Run WSL model: RDFA_wsl_train_pcedice
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/weakly_supervised/RDFA/wsl_train_pcedice/UNet_RDFA_260603_193335 --supervision wsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/wsl_RDFA_wsl_train_pcedice

# Run WSL model: RDFA_wsl_train_sscr
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/weakly_supervised/RDFA/wsl_train_sscr/UNet_RDFA_260604_014249 --supervision wsl --wsl_variant double --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/wsl_RDFA_wsl_train_sscr

# Run WSL model: RDFA_wsl_train_sscr_ablation
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/weakly_supervised/RDFA/wsl_train_sscr_ablation/RDFA_UNet_260604_062445 --supervision wsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/wsl_RDFA_wsl_train_sscr_ablation

# Run WSL model: SALE_wsl_train_DMPLS
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/weakly_supervised/SALE/wsl_train_DMPLS/UNet_CCT_SALE_260606_230638 --supervision wsl --wsl_variant dmpls --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/wsl_SALE_wsl_train_DMPLS

# Run WSL model: SALE_wsl_train_EMA_sscr
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/weakly_supervised/SALE/wsl_train_EMA_sscr/SALE_UNet_260605_194816 --supervision wsl --wsl_variant ema --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/wsl_SALE_wsl_train_EMA_sscr

# Run WSL model: SALE_wsl_train_GatedCRFLoss
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/weakly_supervised/SALE/wsl_train_GatedCRFLoss/UNet_SALE_260604_141900 --supervision wsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/wsl_SALE_wsl_train_GatedCRFLoss

# Run WSL model: SALE_wsl_train_Inter&Intra_Class
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/weakly_supervised/SALE/wsl_train_Inter&Intra_Class/UNet_SALE_260604_160148 --supervision wsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/wsl_SALE_wsl_train_Inter&Intra_Class

# Run WSL model: SALE_wsl_train_entropy_mini
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/weakly_supervised/SALE/wsl_train_entropy_mini/UNet_SALE_260604_124157 --supervision wsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/wsl_SALE_wsl_train_entropy_mini

# Run WSL model: SALE_wsl_train_pcedice
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/weakly_supervised/SALE/wsl_train_pcedice/UNet_SALE_260604_110725 --supervision wsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/wsl_SALE_wsl_train_pcedice

# Run WSL model: SALE_wsl_train_sscr
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/weakly_supervised/SALE/wsl_train_sscr/UNet_SALE_260605_022144 --supervision wsl --wsl_variant double --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/wsl_SALE_wsl_train_sscr

# Run WSL model: SALE_wsl_train_sscr_ablation
python infer_ratdata_dias.py --dias_repo ~/projects/lab/DIAS --checkpoint_dir /home/ashmithandoo/projects/lab/saved_models/weakly_supervised/SALE/wsl_train_sscr_ablation/SALE_UNet_260605_051721 --supervision wsl --wsl_variant standard --data_dir data/medsam_preprocessed/RatData_NPZ --output_dir results/RatData_DIAS_Inference/wsl_SALE_wsl_train_sscr_ablation
