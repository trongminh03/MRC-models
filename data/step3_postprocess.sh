#!/bin/bash
cd post_processing
set -e  # stop on error

echo "=== Step 3: Post-processing & Conversion ==="

# Make sure output dirs exist
mkdir -p data/processed
mkdir -p data/final

##############################################
# 1. Merge questions with contexts
##############################################
echo "[1/2] Merging questions with contexts..."
python merge_file.py \
  --question data/processed/train.jsonl \
  --context data/transcripts/normalized/train_transcriptions_normalized.jsonl \
  --output data/processed/train_merged.jsonl

python merge_file.py \
  --question data/processed/val.jsonl \
  --context data/transcripts/normalized/val_transcriptions_normalized.jsonl \
  --output data/processed/val_merged.jsonl   


##############################################
# 2. Convert merged files into SQuAD format
##############################################
echo "[2/2] Converting to SQuAD format..."
python convert_to_squad.py \
  --input data/processed/train_merged.jsonl \
  --output data/final/train_squad.json \
  --version v1.0

python convert_to_squad.py \
  --input data/processed/val_merged.jsonl \
  --output data/final/val_squad.json \
  --version v1.0

echo "Step 3 complete: final SQuAD datasets are in data/final/"
