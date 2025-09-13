import os
import json
import random
import numpy as np
import librosa
import soundfile as sf
import torch
import torch_audiomentations as ta
import tempfile
from pydub import AudioSegment
from tqdm import tqdm
import argparse

# -------------------- ESC-50 ADD NOISE --------------------
def match_length(audio1, audio2):
    if len(audio2) > len(audio1):
        return audio2[:len(audio1)]
    elif len(audio2) < len(audio1):
        return np.pad(audio2, (0, len(audio1) - len(audio2)), 'constant')
    return audio2

def add_esc50_noise(original_audio, noise_audio, snr_db, noise_weight=1.0):
    noise_audio = match_length(original_audio, noise_audio)
    rms_original = np.sqrt(np.mean(original_audio**2))
    rms_noise = np.sqrt(np.mean(noise_audio**2))
    desired_rms_noise = rms_original / (10**(snr_db / 20))
    scaling_factor = desired_rms_noise / (rms_noise + 1e-9)
    noisy_audio = original_audio + noise_audio * scaling_factor * noise_weight
    return noisy_audio

# -------------------- TORCH AUGMENT --------------------
def get_dict_transforms(sample_rate=16000):
    transforms = {
        'ColouredNoise': ta.AddColoredNoise(p=1, output_type="dict"),
        'BandPassFilter': ta.BandPassFilter(p=1, output_type="dict"),
        'BandStopFilter': ta.BandStopFilter(p=1, output_type="dict"),
        'Gain': ta.Gain(min_gain_in_db=-8, max_gain_in_db=8, p=1, output_type="dict"),
        'LowPassFilter': ta.LowPassFilter(min_cutoff_freq=1000, max_cutoff_freq=2000, p=1, output_type="dict"),
        'Normalization': ta.PeakNormalization(p=1, output_type="dict"),
    }
    if sample_rate == 16000:
        transforms['PitchShift'] = ta.PitchShift(
            p=1,
            sample_rate=sample_rate,
            min_transpose_semitones=-8,
            max_transpose_semitones=8,
            output_type="dict"
        )
    return transforms

def apply_transform(ta_transform, wav_path):
    wav, sr = sf.read(wav_path)
    wav = wav.squeeze()
    wav_tensor = torch.from_numpy(wav[None, None, :]).to(torch.float32)
    result = ta_transform(wav_tensor, sample_rate=sr)
    wav_tensor = result.samples.to(torch.float32).flatten()
    return wav_tensor, sr

def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

def convert_wav_to_mp3(wav_path, mp3_path):
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3")

# -------------------- MAIN PIPELINE --------------------
def augment_dataset(input_folder, noise_folder, output_folder, output_json,
                    snr_db=10, min_weight=0.1, max_weight=1.5, ratio_esc50=0.33):
    os.makedirs(output_folder, exist_ok=True)

    all_audio_files = [f for f in os.listdir(input_folder) if f.endswith('.mp3')]
    random.shuffle(all_audio_files)

    n_esc50 = int(len(all_audio_files) * ratio_esc50)
    esc50_files = all_audio_files[:n_esc50]
    torch_files = all_audio_files[n_esc50:]

    noise_files = [f for f in os.listdir(noise_folder) if f.endswith('.wav')]
    if not noise_files:
        print("❌ No noise .wav files found!")
        return

    esc50_done, torch_done = [], []
    dict_transforms = None

    # --- 1. Add noise with ESC-50 ---
    print("\n=== ESC-50 Noise Augmentation ===")
    for file in tqdm(esc50_files):
        audio_path = os.path.join(input_folder, file)
        audio, sr = librosa.load(audio_path, sr=None)

        noise_file = random.choice(noise_files)
        noise_audio, _ = librosa.load(os.path.join(noise_folder, noise_file), sr=sr)
        weight = random.uniform(min_weight, max_weight)

        noisy_audio = add_esc50_noise(audio, noise_audio, snr_db, weight)

        output_path = os.path.join(output_folder, file)
        sf.write(output_path, noisy_audio, sr, format="MP3")
        esc50_done.append(file)

    # --- 2. Torch-audiomentations ---
    print("\n=== Torch-audiomentations Augmentation ===")
    for file in tqdm(torch_files):
        input_path = os.path.join(input_folder, file)

        # --- convert mp3 -> wav ---
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav_path = tmp_wav.name
        tmp_wav.close()
        convert_mp3_to_wav(input_path, tmp_wav_path)

        _, sr = sf.read(tmp_wav_path)
        if dict_transforms is None:
            dict_transforms = get_dict_transforms(sample_rate=sr)

        key, transform_fn = random.choice(list(dict_transforms.items()))
        augmented_tensor, _ = apply_transform(transform_fn, tmp_wav_path)

        # --- save augmented wav ---
        tmp_aug = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_aug_path = tmp_aug.name
        tmp_aug.close()
        sf.write(tmp_aug_path, augmented_tensor.numpy(), sr)

        output_path = os.path.join(output_folder, file)
        convert_wav_to_mp3(tmp_aug_path, output_path)
        torch_done.append(file)

        # cleanup
        os.remove(tmp_wav_path)
        os.remove(tmp_aug_path)

    # --- Save log JSON ---
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({
            "esc50_files": esc50_done,
            "torch_augmented_files": torch_done
        }, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 Done! ESC-50: {len(esc50_done)} | Torch: {len(torch_done)}")
    print(f"📄 Log saved at {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio augmentation pipeline")
    parser.add_argument("--input_folder", required=True, help="Folder containing original mp3 files")
    parser.add_argument("--noise_folder", required=True, help="Folder containing wav noise files (ESC-50)")
    parser.add_argument("--output_folder", required=True, help="Folder to save augmented files")
    parser.add_argument("--output_json", required=True, help="Path to save augmentation log JSON")
    args = parser.parse_args()

    augment_dataset(
        args.input_folder,
        args.noise_folder,
        args.output_folder,
        args.output_json,
        snr_db=10, min_weight=0.1, max_weight=1.5, ratio_esc50=0.33
    )
