#!/bin/bash
cd data_preparation
set -e  # stop if any command fails

echo "=== Step 1: Data Preparation ==="

# 1. Download dataset into JSONL files
echo "[1/3] Downloading dataset..."
python download_dataset.py

# 2. Extract distinct contexts (from train split by default)
echo "[2/3] Extracting distinct contexts..."
python extract_distinct_context.py \
  --input data/uit-viquad_train.jsonl \
  --output data/processed/contexts.jsonl \
  --format jsonl

# 3. Preprocess data (replace contexts with context IDs)
echo "[3/3] Preprocessing splits..."
python preprocess_data.py \
  --input data/uit-viquad_train.jsonl \
  --contexts data/processed/contexts.jsonl \
  --output data/processed/train.jsonl

python preprocess_data.py \
  --input data/uit-viquad_val.jsonl \
  --contexts data/processed/contexts.jsonl \
  --output data/processed/val.jsonl

echo "Step 1 complete: data/processed contains preprocessed JSONL files"
