#!/usr/bin/env python

"""
Batch TTS Processor Module
--------------------------
This module processes UIT-ViQuAD contexts in batch, converting them to speech 
using random Vietnamese TTS voices and saving:
1. Audio files to data/audio/contexts
2. Processing metadata to a JSONL file

The script reads from a processed JSONL file with context_ids and looks up
the full context text from a distinct contexts file.
"""

import os
import json
import random
from google.cloud import texttospeech
from dotenv import load_dotenv
import time
import argparse
import hashlib

def get_tts_client(credentials_path=None):
    """Initialize and return a Text-to-Speech client."""
    try:
        if credentials_path and os.path.exists(credentials_path):
            return texttospeech.TextToSpeechClient.from_service_account_json(credentials_path)
        else:
            return texttospeech.TextToSpeechClient()
    except Exception as e:
        print(f"Error initializing TTS client: {e}")
        print("\nTo fix authentication errors:")
        print("1. Ensure you have downloaded your Google Cloud service account key file (.json)")
        print("2. You can:")
        print("   a) Create a .env file with GOOGLE_APPLICATION_CREDENTIALS=path/to/your/key.json")
        print("   b) Set the environment variable: set GOOGLE_APPLICATION_CREDENTIALS=path\\to\\your\\key.json")
        print("   c) Or provide the credentials file path when prompted")
        return None

def get_vietnamese_voices(client):
    """Get all available Vietnamese voices."""
    try:
        response = client.list_voices(language_code="vi-VN")
        voices = response.voices
        
        # Log available voices
        print(f"Found {len(voices)} Vietnamese voices:")
        for idx, voice in enumerate(voices):
            gender = texttospeech.SsmlVoiceGender(voice.ssml_gender).name
            name = voice.name
            voice_type = "NEURAL" if any(term in name for term in ['Neural', 'Wavenet', 'HD']) else "STANDARD"
            print(f"{idx+1}. Name: {name}, Gender: {gender}, Type: {voice_type}")
        
        return voices
    except Exception as e:
        print(f"Error getting Vietnamese voices: {e}")
        return []

def text_to_speech(client, text, voice, output_file):
    """Convert text to speech using specified voice."""
    try:
        # Set the text input
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Build the voice parameters
        voice_params = texttospeech.VoiceSelectionParams(
            language_code="vi-VN",
            name=voice.name,
            ssml_gender=voice.ssml_gender
        )
        
        # Select the audio file type
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        # Perform the text-to-speech request
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config
        )
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the audio to a file
        with open(output_file, "wb") as out:
            out.write(response.audio_content)
        
        return True
    except Exception as e:
        print(f"Error in text_to_speech: {e}")
        return False

def read_jsonl(file_path):
    """Read data from JSONL file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))
        print(f"Read {len(data)} records from {file_path}")
        return data
    except Exception as e:
        print(f"Error reading JSONL file {file_path}: {e}")
        return []

def read_json(file_path):
    """Read data from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if the data is nested under 'contexts'
        if 'contexts' in data and isinstance(data['contexts'], list):
            print(f"Read {len(data['contexts'])} contexts from JSON file")
            return data['contexts']
        else:
            print(f"Read data from JSON file")
            return data
    except Exception as e:
        print(f"Error reading JSON file {file_path}: {e}")
        return []

def build_context_id_mapping(contexts_data):
    """Build a mapping from context_id to context text."""
    id_to_context = {}
    
    for item in contexts_data:
        if 'context_id' in item and 'context' in item:
            id_to_context[item['context_id']] = item['context']
    
    print(f"Built mapping for {len(id_to_context)} distinct contexts")
    return id_to_context

def collect_context_ids(data):
    """Collect all unique context_ids from the input data."""
    context_ids = set()
    
    for item in data:
        if 'context_id' in item:
            context_ids.add(item['context_id'])
    
    return context_ids

def process_batch(input_file, contexts_file, output_audio_dir, output_metadata_file, limit=None):
    """Process a batch of contexts from JSONL file based on context_ids."""
    # Load .env file if it exists
    if os.path.exists('.env'):
        load_dotenv()
    
    # Initialize client
    client = get_tts_client()
    if not client:
        print("Failed to initialize TTS client. Exiting...")
        return
    
    # Get Vietnamese voices
    voices = get_vietnamese_voices(client)
    if not voices:
        print("No Vietnamese voices available. Exiting...")
        return
    
    # Read input data with context_ids
    input_data = read_jsonl(input_file)
    if not input_data:
        print("No input data to process. Exiting...")
        return
    
    # Load distinct contexts
    if contexts_file.lower().endswith('.json'):
        contexts_data = read_json(contexts_file)
    else:
        contexts_data = read_jsonl(contexts_file)
    
    if not contexts_data:
        print("No context mapping data. Exiting...")
        return
    
    # Build context_id to context mapping
    id_to_context = build_context_id_mapping(contexts_data)
    
    # Get all unique context_ids from input data
    unique_context_ids = collect_context_ids(input_data)
    print(f"Found {len(unique_context_ids)} unique context IDs in input data")
    
    # Prepare data for processing
    contexts_to_process = []
    for context_id in unique_context_ids:
        if context_id in id_to_context:
            contexts_to_process.append({
                'context_id': context_id,
                'context': id_to_context[context_id]
            })
        else:
            print(f"Warning: context_id '{context_id}' not found in contexts file")
    
    # Limit the number of contexts to process if specified
    if limit and limit > 0:
        contexts_to_process = contexts_to_process[:limit]
        print(f"Processing limited to {limit} distinct contexts")
    
    # Create output directories
    os.makedirs(output_audio_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_metadata_file), exist_ok=True)
    
    # Process each context
    metadata = []
    success_count = 0
    
    for idx, item in enumerate(contexts_to_process):
        context_id = item['context_id']
        context_text = item['context']
        
        # Generate filename using context ID
        filename = f"{context_id}.mp3"
        output_file = os.path.join(output_audio_dir, filename)
        
        # Select a random Vietnamese voice
        voice = random.choice(voices)
        
        print(f"Processing context {idx+1}/{len(contexts_to_process)}: ID={context_id}")
        print(f"Text: '{context_text[:100]}...' ({len(context_text)} chars)")
        print(f"Voice: {voice.name}")
        
        # Convert text to speech
        success = text_to_speech(client, context_text, voice, output_file)
        
        if success:
            # Record metadata
            item_metadata = {
                "context_id": context_id,
                "audio_file": filename,
                "voice_name": voice.name,
                "voice_gender": texttospeech.SsmlVoiceGender(voice.ssml_gender).name,
                "voice_type": "NEURAL" if any(term in voice.name for term in ['Neural', 'Wavenet', 'HD']) else "STANDARD",
                "timestamp": time.time(),
                "context_length": len(context_text)
            }
            metadata.append(item_metadata)
            success_count += 1
        else:
            print(f"Failed to process context {context_id}")
    
    # Save metadata to JSONL file
    with open(output_metadata_file, 'w', encoding='utf-8') as f:
        for item in metadata:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\nProcessing complete: {success_count}/{len(contexts_to_process)} distinct contexts successfully processed")
    print(f"Audio files saved to: {output_audio_dir}")
    print(f"Metadata saved to: {output_metadata_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch TTS processor for UIT-ViQuAD contexts')
    parser.add_argument('--input', required=True,
                      help='Input JSONL file with context_ids')
    parser.add_argument('--contexts', required=True,
                      help='Distinct contexts file (JSONL or JSON)')
    parser.add_argument('--output-dir', default=os.path.join('data', 'audio', 'contexts'),
                      help='Output directory for audio files (default: data/audio/contexts)')
    parser.add_argument('--metadata', default=os.path.join('data', 'processed', 'contexts_tts_metadata.jsonl'),
                      help='Output metadata JSONL file (default: data/processed/contexts_tts_metadata.jsonl)')
    parser.add_argument('--limit', type=int, default=None,
                      help='Limit the number of distinct contexts to process (default: process all)')
    
    args = parser.parse_args()
    
    # Process the batch
    process_batch(args.input, args.contexts, args.output_dir, args.metadata, args.limit)

if __name__ == "__main__":
    main() 