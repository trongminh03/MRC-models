# ViSQA
This repository provides a full pipeline to create **ASR-augmented question answering datasets** from [UIT-ViQuAD 2.0](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0).  
It downloads the dataset, generates audio and ASR transcriptions, merges them back into QA format, and outputs **SQuAD-style JSON** for model training and evaluation.


The pipeline has 3 main steps:  
1. **Data Preparation** → download + preprocess dataset  
2. **Speech Processing** → generate TTS audio + ASR transcription  
3. **Post-processing & Conversion** → merge with questions + export SQuAD format  

## Project Structure
```
.
data/
├── step1_data_prep.sh          # Step 1: Data preparation
├── step2_speech_proc.sh        # Step 2: Speech generation + ASR
├── step3_postprocess.sh        # Step 3: Merge + convert to SQuAD
│
├── data_preparation/
│   ├── download_dataset.py
│   ├── extract_distinct_context.py
│   └── preprocess_data.py
│
├── speech_process/
│   ├── batch_tts_processor.py
│   ├── transcription_batch.py
│   └── normalize_jsonl.py
│
└── post_processing/
    ├── merge_file.py
    └── convert_to_squad.py
```
## Requirements

- Python 3.8+
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)
- Google Cloud SDK & credentials (for TTS + Speech-to-Text V2)
- Install dependencies:
```bash
pip install -r requirements.txt
```

## Pipeline Steps
### Step 1: Data Preparation

```bash
bash step1_data_prep.sh
```
Downloads UIT-ViQuAD splits (train/validation/test).

Extracts distinct contexts and assigns IDs.

Replaces contexts with context IDs.

### Step 2: Speech Processing

```bash
bash step2_speech_proc.sh
```

Generate TTS audio for contexts.

Transcribe audio with Google Cloud Speech-to-Text V2.

Normalize ASR transcripts.

### Step 2.5: Add Noise to Audio
To simulate noisy speech conditions, environmental noise can be injected into the TTS audio before transcription. This can be done with the provided script addnoise.py, which mixes audio files from an external noise dataset (e.g., ESC-50
) into the generated speech. Users need to download and prepare their own noise dataset; any collection of WAV-format background sounds is compatible.

```bash
python addnoise.py \
  --input_folder "path/to/audio/" \
  --noise_folder "path/to/noise_wavs/" \
  --output_folder "path/to/output/addnoise/" \
  --output_json "path/to/output/json.json"
```

⚠️ If you add noise, you’ll need to re-run Step 2.2 (ASR transcription) on the noisy audio.

### Step 3: Post-processing & Conversion

```bash
bash step3_postprocess.sh
```

Merge normalized transcripts with QA questions.

Convert merged files to SQuAD JSON.

## Machine Reading Comprehension Models
### How to run the code  
For training: please run the train.py script.   
For evaluation: please run the test.py script.

Required params (for both train.py and test.py):  
--path: Path to the ViSQA dataset  
--type: Type of model (eg. bartpho, xlm-r, phobert, etc). Default is auto   
--model: the path to pre-trained model or slug as shown in the Hugging Face website (eg. xlm-r-base)   
--output_path: Path to the output directory   
--is_test: Pass this param in the test.py if you want to run the evaluation on the test set. If none, the code will run evaluation on the development (dev) set.   

Link dataset: doi.org/10.6084/m9.figshare.29493149