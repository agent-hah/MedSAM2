#!/bin/bash
set -e

# === Stage 1: DICOM → Raw NIfTI ===
echo "Converting RatData DICOMs to NIfTI..."
# python run_preprocessing.py dicom2nifti --dataset rat --input_dir data/raw/RatData --output_dir data/nifti/RatData --overwrite

echo "Converting HumanData DICOMs to NIfTI..."
# python run_preprocessing.py dicom2nifti --dataset human --input_dir data/raw/HumanData --output_dir data/nifti/HumanData --overwrite

# === Stage 2: Generate Hessian Masks ===
echo "Generating hessian masks for RatData..."
# python run_preprocessing.py generate-mask --input_dir data/nifti/RatData --overwrite --keep_all

echo "Generating hessian masks for HumanData..."
# python run_preprocessing.py generate-mask --input_dir data/nifti/HumanData --overwrite --keep_all

# === Stage 3: Preprocess NIfTI for training ===
echo "Preprocessing RatData NIfTI..."
python run_preprocessing.py preprocess-nifti --input_dir data/nifti/RatData --output_dir data/nifti_preprocessed/RatData --no_clahe --no_minmax --overwrite

echo "Preprocessing HumanData NIfTI..."
python run_preprocessing.py preprocess-nifti --input_dir data/nifti/HumanData --output_dir data/nifti_preprocessed/HumanData --no_clahe --no_minmax --overwrite

# # === DIAS stays on NPZ ===
# echo "Preprocessing DIAS dataset (NPZ)..."
# python run_preprocessing.py dias --skip-cropping --no_clahe --overwrite

echo "All datasets preprocessed successfully!"
