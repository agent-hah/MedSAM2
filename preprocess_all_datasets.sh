#!/bin/bash
set -e

# echo "Starting preprocessing for DIAS dataset..."
# python run_preprocessing.py dias

echo "Starting preprocessing for Rat dataset..."
python run_preprocessing.py rat --skip-cropping

# echo "Starting preprocessing for Human dataset..."
# python run_preprocessing.py human

echo "All datasets preprocessed successfully!"
