#!/bin/bash
cd speech_process
set -e  # stop on first error

echo "=== Step 2: Speech Processing ==="

# Make sure output directories exist
mkdir -p data/audio/contexts
mkdir -p data/transcripts/raw
mkdir -p data/transcripts/normalized

# Process splits (train + val only, skip test)
for SPLIT in train val; do
  echo ">>> Processing split: $SPLIT"

  ##############################################
  # 1. Generate audio with Google TTS
  ##############################################
  echo "[1/3] Generating TTS audio from contexts ($SPLIT)..."
  python batch_tts_processor.py \
    --input data/processed/${SPLIT}.jsonl \
    --contexts data/processed/contexts.jsonl \
    --output-dir data/audio/contexts/${SPLIT} \
    --metadata data/processed/contexts_tts_metadata_${SPLIT}.jsonl

  ##############################################
  # 2. Run ASR (Google Cloud Speech-to-Text V2)
  ##############################################
  echo "[2/3] Running ASR transcription ($SPLIT)..."
  python transcription_batch.py \
    --input-dir data/audio/contexts/${SPLIT} \
    --gcs-bucket <YOUR_GCS_BUCKET> \
    --project-id <YOUR_PROJECT_ID> \
    --pattern .wav \
    --output-dir data/transcripts/raw/${SPLIT} \
    --language vi-VN \
    --alt-language en-US \
    --workers 5 \
    --credentials <PATH_TO_CREDENTIALS_JSON>

  ##############################################
  # 3. Normalize transcriptions
  ##############################################
  echo "[3/3] Normalizing transcription JSONL ($SPLIT)..."
  python normalize_jsonl.py \
    --input data/transcripts/raw/${SPLIT}/transcriptions.jsonl \
    --output data/transcripts/normalized/${SPLIT}_transcriptions_normalized.jsonl \
    --field context

  echo "✅ Done split: $SPLIT"
done

echo "Step 2 complete: audio + normalized transcripts ready (train + val)"
