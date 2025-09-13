#!/usr/bin/env python

"""
Google Cloud Storage Transcription Processor
------------------------------------------
This script handles transcription of audio files using Google Cloud Speech-to-Text API V2,
automatically using Google Cloud Storage for files that exceed direct API limits.

Features:
- Automatically detects files too large for direct processing
- Uploads large files to Google Cloud Storage
- Uses GCS URIs for processing
- Processes files in parallel using multiple workers
- Filters by context IDs

Usage:
    python gcs_transcription_processor.py --input-dir data/audio/contexts_wav
                                        --output-dir data/transcriptions
                                        --gcs-bucket your-gcs-bucket-name
                                        --workers 5
"""

import os
import json
import argparse
import time
import uuid
import wave
import contextlib
from google.cloud import speech_v2, storage
from google.oauth2 import service_account
from dotenv import load_dotenv
import concurrent.futures
from tqdm import tqdm
import logging
from datetime import datetime

# Try to import pydub for more accurate audio duration detection
try:
    from pydub import AudioSegment
    HAVE_PYDUB = True
except ImportError:
    HAVE_PYDUB = False
    print("pydub not installed. Will use basic duration detection for WAV files only.")
    print("For better duration detection with MP3 files, install pydub: pip install pydub")

class GCSTranscriptionProcessor:
    """
    Process transcription in parallel using Google Cloud Storage for large files.
    """
    
    # Constants for size limits
    MAX_INLINE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB size limit for direct API calls
    MAX_DURATION_SECONDS = 60  # 1 minute is a safe limit for direct API
    
    def __init__(
        self,
        output_dir,
        gcs_bucket,
        language_code="vi-VN",
        alternative_language_codes=["en-US"],
        max_workers=5,
        credentials_path=None,
        log_file=None,
        cleanup_gcs=True,
        force_gcs=False,
        project_id=None
    ):
        """
        Initialize the GCS transcription processor.
        
        Args:
            output_dir: Directory to save transcription results
            gcs_bucket: Google Cloud Storage bucket name
            language_code: Primary language code for transcription
            alternative_language_codes: List of alternative language codes
            max_workers: Maximum number of concurrent workers
            credentials_path: Path to GCP credentials JSON file
            log_file: Path to log file
            cleanup_gcs: Whether to delete GCS files after processing
            force_gcs: Whether to force using GCS for all files regardless of size/duration
            project_id: Google Cloud project ID (required for Speech V2)
        """
        self.output_dir = output_dir
        self.gcs_bucket = gcs_bucket
        self.language_code = language_code
        self.alternative_language_codes = alternative_language_codes
        self.max_workers = max_workers
        self.credentials_path = credentials_path
        self.cleanup_gcs = cleanup_gcs
        self.force_gcs = force_gcs
        self.project_id = project_id
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        if log_file is None:
            # Default log file in output directory
            self.log_file = os.path.join(output_dir, 'gcs_transcription_log.txt')
        else:
            # Use the provided log file path
            self.log_file = log_file
            # Make sure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
        
        self._setup_logging()
        self.logger.info(f"GCSTranscriptionProcessor initialized - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"GCS bucket: {gcs_bucket}")
        self.logger.info(f"Language: {language_code} (alternatives: {alternative_language_codes})")
        self.logger.info(f"Max workers: {max_workers}")
        self.logger.info(f"Cleanup GCS files: {cleanup_gcs}")
        self.logger.info(f"Force GCS for all files: {force_gcs}")
        self.logger.info(f"Project ID: {project_id}")
        
        # Load .env file if it exists
        if os.path.exists('.env'):
            load_dotenv()
        
        # Initialize clients
        self._initialize_clients()
        
        # Create a unique folder name for this run to avoid conflicts
        self.gcs_folder = f"speech_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.logger.info(f"Using GCS folder: {self.gcs_folder}")
    
    def _setup_logging(self):
        """Initialize logging to file and console"""
        self.logger = logging.getLogger('gcs_transcription')
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent propagation to root logger to avoid duplicate entries
        self.logger.propagate = False
        
        self.logger.info("=" * 80)
        self.logger.info("NEW GCS TRANSCRIPTION PROCESSOR SESSION STARTED")
        self.logger.info("=" * 80)
    
    def _initialize_clients(self):
        """Initialize Google Cloud clients"""
        try:
            # Check for project ID which is required for Speech V2
            if not self.project_id:
                # Try to get from environment
                self.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
                
            if not self.project_id:
                self.logger.error("Project ID is required for Speech V2 API")
                raise ValueError("Project ID is required for Speech V2 API. Provide it with the --project-id parameter or set GOOGLE_CLOUD_PROJECT environment variable.")
                
            # Use explicit credentials if provided
            if self.credentials_path and os.path.exists(self.credentials_path):
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                self.speech_client = speech_v2.SpeechClient(credentials=credentials)
                self.storage_client = storage.Client(credentials=credentials)
                self.logger.info(f"Initialized clients with credentials from: {self.credentials_path}")
            else:
                # Fall back to default credentials
                self.speech_client = speech_v2.SpeechClient()
                self.storage_client = storage.Client()
                self.logger.info("Initialized clients with default credentials")
            
            # Verify bucket exists
            self.bucket = self.storage_client.bucket(self.gcs_bucket)
            if not self.bucket.exists():
                self.logger.error(f"GCS bucket '{self.gcs_bucket}' does not exist")
                raise Exception(f"GCS bucket '{self.gcs_bucket}' does not exist")
            
            self.logger.info(f"Successfully connected to GCS bucket: {self.gcs_bucket}")
        except Exception as e:
            self.logger.error(f"Error initializing clients: {e}")
            print(f"Error initializing clients: {e}")
            print("\nTo fix authentication errors:")
            print("1. Ensure you have downloaded your Google Cloud service account key file (.json)")
            print("2. You can:")
            print("   a) Create a .env file with GOOGLE_APPLICATION_CREDENTIALS=path/to/your/key.json")
            print("   b) Set the environment variable: set GOOGLE_APPLICATION_CREDENTIALS=path\\to\\your\\key.json")
            print("   c) Or provide the credentials file path as a parameter")
            print("3. Make sure you've specified the correct Google Cloud project ID")
            raise
    
    def get_audio_duration(self, file_path):
        """
        Get the duration of an audio file in seconds.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Duration in seconds or None if unable to determine
        """
        try:
            # Try pydub first if available (handles mp3, wav, etc.)
            if HAVE_PYDUB:
                try:
                    audio = AudioSegment.from_file(file_path)
                    duration_seconds = len(audio) / 1000.0
                    return duration_seconds
                except Exception as e:
                    self.logger.warning(f"Unable to determine audio duration with pydub for {file_path}: {e}")
                    # Fall through to the next method
            
            # Fallback to basic wave module for WAV files
            if file_path.lower().endswith('.wav'):
                try:
                    with contextlib.closing(wave.open(file_path, 'r')) as f:
                        frames = f.getnframes()
                        rate = f.getframerate()
                        duration = frames / float(rate)
                        return duration
                except Exception as e:
                    self.logger.warning(f"Unable to determine audio duration with wave module for {file_path}: {e}")
                    return None
            
            # Unable to determine duration for non-WAV files without pydub
            return None
        
        except Exception as e:
            self.logger.warning(f"Unable to determine audio duration for {file_path}: {e}")
            return None
    
    def upload_to_gcs(self, audio_content, file_name):
        """
        Upload audio content to Google Cloud Storage.
        
        Args:
            audio_content: Binary audio content
            file_name: Base file name for the GCS object
            
        Returns:
            GCS URI for the uploaded file
        """
        try:
            # Create a unique blob name
            blob_name = f"{self.gcs_folder}/{file_name}"
            
            # Get blob reference
            blob = self.bucket.blob(blob_name)
            
            # Determine content type based on file extension
            content_type = 'audio/wav'
            if file_name.lower().endswith('.mp3'):
                content_type = 'audio/mpeg'
            
            # Upload the content
            blob.upload_from_string(audio_content, content_type=content_type)
            
            # Get GCS URI
            gcs_uri = f"gs://{self.gcs_bucket}/{blob_name}"
            
            self.logger.info(f"Uploaded audio to {gcs_uri}")
            return gcs_uri
        except Exception as e:
            self.logger.error(f"Error uploading to GCS: {e}")
            raise
    
    def delete_from_gcs(self, gcs_uri):
        """
        Delete a file from Google Cloud Storage.
        
        Args:
            gcs_uri: GCS URI of the file to delete
        """
        try:
            # Extract bucket and blob name from GCS URI
            # Format: gs://bucket-name/path/to/file
            uri_parts = gcs_uri.replace('gs://', '').split('/', 1)
            if len(uri_parts) != 2:
                self.logger.warning(f"Invalid GCS URI format: {gcs_uri}")
                return
            
            bucket_name, blob_name = uri_parts
            
            # Verify we're deleting from the correct bucket
            if bucket_name != self.gcs_bucket:
                self.logger.warning(f"URI bucket {bucket_name} doesn't match processor bucket {self.gcs_bucket}")
                return
            
            # Delete the blob
            blob = self.bucket.blob(blob_name)
            blob.delete()
            
            self.logger.debug(f"Deleted {gcs_uri}")
        except Exception as e:
            self.logger.warning(f"Error deleting {gcs_uri}: {e}")
    
    def cleanup_gcs_files(self):
        """Delete all files in the GCS folder for this run."""
        try:
            blobs = self.bucket.list_blobs(prefix=self.gcs_folder)
            for blob in blobs:
                blob.delete()
            
            self.logger.info(f"Cleaned up GCS folder: {self.gcs_folder}")
        except Exception as e:
            self.logger.error(f"Error cleaning up GCS folder: {e}")
    
    def transcribe_file(self, file_path):
        """
        Transcribe a single audio file, using GCS for large files.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary with transcription results
        """
        file_name = os.path.basename(file_path)
        context_id = os.path.splitext(file_name)[0]
        
        self.logger.info(f"Starting transcription for {file_name} (context_id: {context_id})")
        start_time = time.time()
        
        try:
            # Get file info for diagnostics
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            self.logger.info(f"{file_name}: File size: {file_size:.2f} MB")
            
            # Check audio duration if possible
            duration = self.get_audio_duration(file_path)
            if duration is not None:
                self.logger.info(f"{file_name}: Audio duration: {duration:.2f} seconds")
            else:
                self.logger.warning(f"{file_name}: Unable to determine audio duration")
            
            # Determine if we should use GCS based on file size, duration, or force flag
            use_gcs = self.force_gcs or \
                      file_size > (self.MAX_INLINE_SIZE_BYTES / (1024 * 1024)) or \
                      (duration is not None and duration > self.MAX_DURATION_SECONDS)
            
            # Read file content once to avoid multiple file reads
            with open(file_path, "rb") as audio_file:
                content = audio_file.read()
            
            # Create a recognition config for Speech v2
            # Create configuration objects directly instead of dictionaries
            features = speech_v2.RecognitionFeatures(
                enable_automatic_punctuation=True,
                max_alternatives=3
            )
            
            # Create decoding config
            # decoding_config = speech_v2.RecognitionConfig.AutoDetectDecodingConfig()
            decoding_config = speech_v2.types.cloud_speech.AutoDetectDecodingConfig()
            
            
            # Create main config
            config = speech_v2.RecognitionConfig(
                language_codes=[self.language_code] + self.alternative_language_codes,
                model="latest_long",
                features=features,
                auto_decoding_config=decoding_config
            )
            
            if file_path.lower().endswith('.mp3'):
                self.logger.info(f"{file_name}: Using MP3 with latest_long model")
            elif file_path.lower().endswith('.wav'):
                self.logger.info(f"{file_name}: Using WAV with latest_long model")
            else:
                self.logger.info(f"{file_name}: Using auto-detection with latest_long model")
            
            # Holds GCS URI if we upload the file
            gcs_uri = None
            
            # Function to process audio and handle the recognition
            def process_audio(use_gcs_method, is_retry=False):
                nonlocal gcs_uri
                
                if use_gcs_method:
                    if is_retry:
                        self.logger.info(f"{file_name}: Retrying with GCS after duration limit error")
                    else:
                        reason = "force_gcs=True" if self.force_gcs else (
                            f"file size {file_size:.2f}MB > 10MB" if file_size > (self.MAX_INLINE_SIZE_BYTES / (1024 * 1024)) else 
                            f"duration {duration:.2f}s > 60s" if duration is not None and duration > self.MAX_DURATION_SECONDS else
                            "as a precaution"
                        )
                        self.logger.info(f"{file_name}: Using GCS because {reason}")
                    
                    # Upload to GCS if we haven't already
                    if gcs_uri is None:
                        gcs_uri = self.upload_to_gcs(content, file_name)
                    
                    # Create a RecognizeRequest with GCS URI
                    request = speech_v2.RecognizeRequest(
                        recognizer=f"projects/{self.project_id}/locations/global/recognizers/_",
                        config=config,
                        uri=gcs_uri
                    )
                else:
                    self.logger.info(f"{file_name}: Using direct API with inline content")
                    # Create a RecognizeRequest with inline content
                    request = speech_v2.RecognizeRequest(
                        recognizer=f"projects/{self.project_id}/locations/global/recognizers/_",
                        config=config,
                        content=content
                    )
                
                # Process the audio with long running recognize (V2 API)
                self.logger.info(f"{file_name}: Sending request to Google Speech-to-Text API V2")
                print(f"Started transcription for {file_name}")
                
                # Make the API call
                response = self.speech_client.recognize(request=request, timeout=600)
                return response
            
            # First attempt - use GCS if determined earlier, otherwise try direct
            try:
                response = process_audio(use_gcs)
            except Exception as e:
                error_msg = str(e)
                # If we get a duration limit error and we didn't use GCS, retry with GCS
                if not use_gcs and "audio exceeds duration limit" in error_msg.lower():
                    self.logger.warning(f"{file_name}: {error_msg} - Retrying with GCS")
                    # Retry with GCS
                    response = process_audio(True, is_retry=True)
                else:
                    # Other error or already using GCS, re-raise
                    raise
            
            # Process results - Speech V2 API has a different response structure
            results_text = []
            all_alternatives = []
            
            if response.results:
                result_count = len(response.results)
                self.logger.info(f"{file_name}: Received {result_count} result segments")
                
                for i, result in enumerate(response.results):
                    segment_alternatives = []
                    alternative_count = len(result.alternatives)
                    
                    for j, alternative in enumerate(result.alternatives):
                        transcript = alternative.transcript
                        confidence = alternative.confidence if hasattr(alternative, 'confidence') else 0
                        
                        # Log only first alternative of first few segments to avoid huge logs
                        if j == 0 and i < 3:
                            self.logger.info(f"{file_name}: Segment {i+1}/{result_count}, confidence {confidence:.2f}")
                            
                        segment_alternatives.append({
                            "transcript": transcript,
                            "confidence": confidence
                        })
                        
                        # Add primary transcription pieces
                        if j == 0:
                            results_text.append(transcript)
                    
                    if segment_alternatives:
                        all_alternatives.append(segment_alternatives)
                
                final_text = " ".join(results_text)
                text_preview = final_text[:100] + "..." if len(final_text) > 100 else final_text
                self.logger.info(f"{file_name}: Transcription complete, text: {text_preview}")
            else:
                # Enhanced diagnostics for empty results
                self.logger.warning(f"{file_name}: No transcription results returned from API")
                self.logger.warning(f"{file_name}: This could be due to:")
                self.logger.warning(f"{file_name}: - Silent or corrupted audio")
                self.logger.warning(f"{file_name}: - Incompatible audio format")
                self.logger.warning(f"{file_name}: - Audio language mismatch")
                self.logger.warning(f"{file_name}: - Very poor audio quality")
                self.logger.warning(f"{file_name}: - Audio length exceeds API limits")
            
            # Clean up GCS file if we used it and cleanup is enabled
            if gcs_uri and self.cleanup_gcs:
                self.delete_from_gcs(gcs_uri)
            
            end_time = time.time()
            duration = end_time - start_time
            self.logger.info(f"{file_name}: Processing took {duration:.2f} seconds")
            
            # Return structured results, including empty transcription indicator if needed
            result = {
                "context_id": context_id,
                "transcription": " ".join(results_text) if results_text else "",
                "all_alternatives": all_alternatives,
                "file_path": file_path,
                "file_name": file_name,
                "processing_time": duration,
                "empty_result": len(results_text) == 0
            }
            
            # Log warning if transcription is empty but didn't error
            if len(results_text) == 0:
                self.logger.warning(f"{file_name}: Empty transcription result (API returned success but no text)")
                
            return result
        
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            error_msg = str(e)
            
            self.logger.error(f"{file_name}: Error after {duration:.2f} seconds - {error_msg}")
            print(f"Error transcribing {file_name}: {error_msg}")
            
            return {
                "context_id": context_id,
                "error": error_msg,
                "file_path": file_path,
                "file_name": file_name,
                "processing_time": duration,
                "empty_result": True
            }
    
    def process_batch(self, audio_files, filter_context_ids=None, limit=None, show_progress=True):
        """
        Process multiple audio files concurrently.
        
        Args:
            audio_files: List of paths to audio files
            filter_context_ids: Set of context IDs to filter by (None for all)
            limit: Maximum number of items to process
            show_progress: Whether to show a progress bar
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        # Filter audio files by context IDs if specified
        if filter_context_ids:
            self.logger.info(f"Filtering audio files to {len(filter_context_ids)} context IDs")
            filtered_files = []
            for file_path in audio_files:
                file_name = os.path.basename(file_path)
                context_id = os.path.splitext(file_name)[0]
                if context_id in filter_context_ids:
                    filtered_files.append(file_path)
            
            audio_files = filtered_files
            self.logger.info(f"Filtered to {len(audio_files)} audio files")
        
        # Apply limit if specified
        if limit and limit > 0:
            self.logger.info(f"Limiting to {limit} files")
            audio_files = audio_files[:limit]
        
        if not audio_files:
            self.logger.warning("No audio files to process")
            return {
                "success_count": 0,
                "failed_count": 0,
                "empty_count": 0,
                "total_files": 0,
                "total_time": 0,
                "results": {}
            }
        
        self.logger.info(f"Starting batch processing of {len(audio_files)} files with {self.max_workers} workers")
        
        # Prepare JSONL output file
        jsonl_path = os.path.join(self.output_dir, "transcriptions.jsonl")
        self.logger.info(f"Output will be written to: {jsonl_path}")
        
        jsonl_file = open(jsonl_path, 'w', encoding='utf-8')
        
        # Process files
        results = {}
        successful = 0
        failed = 0
        empty_results = 0
        
        try:
            # Use ThreadPoolExecutor for concurrent API requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all transcription tasks
                self.logger.info(f"Submitting {len(audio_files)} files to thread pool with {self.max_workers} workers")
                future_to_file = {
                    executor.submit(self.transcribe_file, file): file 
                    for file in audio_files
                }
                
                # Process results as they complete with progress bar
                if show_progress:
                    futures_iter = tqdm(
                        concurrent.futures.as_completed(future_to_file),
                        total=len(audio_files),
                        desc="Transcribing",
                        unit="file"
                    )
                else:
                    futures_iter = concurrent.futures.as_completed(future_to_file)
                
                for future in futures_iter:
                    file_path = future_to_file[future]
                    file_name = os.path.basename(file_path)
                    
                    try:
                        result = future.result()
                        results[file_name] = result
                        
                        # Write to JSONL
                        if "transcription" in result and not "error" in result:
                            if result.get("empty_result", False):
                                # Mark as empty but successful
                                jsonl_entry = {
                                    "context_id": result["context_id"],
                                    "context": "",
                                    "empty_result": True
                                }
                                jsonl_file.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")
                                self.logger.warning(f"Empty result: {result['context_id']} - API returned no transcription")
                                empty_results += 1
                            else:
                                # Normal successful result
                                jsonl_entry = {
                                    "context_id": result["context_id"],
                                    "context": result["transcription"]
                                }
                                jsonl_file.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")
                                successful += 1
                                self.logger.info(f"Success: {result['context_id']} - Length: {len(result['transcription'])} chars")
                        else:
                            # Log error in JSONL for tracking
                            error_msg = result.get("error", "Unknown error")
                            jsonl_entry = {
                                "context_id": result["context_id"],
                                "error": error_msg
                            }
                            jsonl_file.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")
                            failed += 1
                            self.logger.error(f"Failed: {result['context_id']} - Error: {error_msg}")
                            print(f"Error processing {result['context_id']}: {error_msg}")
                            
                    except Exception as e:
                        error_msg = str(e)
                        context_id = os.path.splitext(file_name)[0]
                        results[file_name] = {"error": error_msg, "file_path": file_path, "context_id": context_id}
                        failed += 1
                        
                        # Log unexpected errors in JSONL
                        jsonl_entry = {
                            "context_id": context_id,
                            "error": error_msg
                        }
                        jsonl_file.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")
                        self.logger.error(f"Exception: {context_id} - Error: {error_msg}")
                        print(f"Exception for {context_id}: {error_msg}")
            
            # Close JSONL file
            jsonl_file.close()
            
            # Calculate processing statistics
            end_time = time.time()
            total_time = end_time - start_time
            avg_time = total_time / len(audio_files) if audio_files else 0
            
            # Clean up GCS files if requested
            if self.cleanup_gcs:
                self.cleanup_gcs_files()
            
            # Log summary statistics
            self.logger.info("=" * 50)
            self.logger.info("BATCH PROCESSING COMPLETED")
            self.logger.info(f"Total files processed: {len(audio_files)}")
            self.logger.info(f"Successful: {successful}")
            self.logger.info(f"Empty results: {empty_results}")
            self.logger.info(f"Failed: {failed}")
            self.logger.info(f"Total time: {total_time:.2f} seconds")
            self.logger.info(f"Average time per file: {avg_time:.2f} seconds")
            self.logger.info(f"Output saved to: {jsonl_path}")
            self.logger.info("=" * 50)
            
            return {
                "results": results,
                "success_count": successful,
                "failed_count": failed,
                "empty_count": empty_results,
                "total_files": len(audio_files),
                "total_time": total_time,
                "avg_time_per_file": avg_time,
                "jsonl_path": jsonl_path
            }
        
        except Exception as e:
            # Make sure to close the file
            jsonl_file.close()
            
            # Clean up GCS files even on error
            if self.cleanup_gcs:
                self.cleanup_gcs_files()
            
            self.logger.error(f"Error in process_batch: {e}")
            raise

def find_audio_files(directory, pattern=None):
    """
    Find all audio files in a directory matching the pattern.
    
    Args:
        directory: Directory to search
        pattern: File extension pattern (e.g., '.wav', '.mp3')
        
    Returns:
        List of file paths
    """
    audio_files = []
    
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        return audio_files
    
    for root, _, files in os.walk(directory):
        for file in files:
            if pattern is None or file.lower().endswith(pattern.lower()):
                audio_files.append(os.path.join(root, file))
    
    return audio_files

def load_context_ids(file_path):
    """
    Load context IDs from a JSON or JSONL file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Set of context IDs
    """
    context_ids = set()
    
    try:
        if file_path.lower().endswith('.json'):
            # Load as JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Handle different formats
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'context_id' in item:
                            context_ids.add(item['context_id'])
                elif isinstance(data, dict):
                    if 'context_id' in data:
                        context_ids.add(data['context_id'])
                    elif 'contexts' in data and isinstance(data['contexts'], list):
                        for item in data['contexts']:
                            if isinstance(item, dict) and 'context_id' in item:
                                context_ids.add(item['context_id'])
        else:
            # Load as JSONL
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            if isinstance(item, dict) and 'context_id' in item:
                                context_ids.add(item['context_id'])
                        except json.JSONDecodeError:
                            pass
        
        print(f"Loaded {len(context_ids)} context IDs from {file_path}")
        return context_ids
    
    except Exception as e:
        print(f"Error loading context IDs from {file_path}: {e}")
        return context_ids

def load_audio_files_from_jsonl(file_path, audio_dir):
    """
    Load audio file paths from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        audio_dir: Base directory where audio files are located
        
    Returns:
        List of audio file paths
    """
    audio_files = []
    files_not_found = []
    
    if not audio_dir:
        print("Error: audio_dir must be specified when using input-jsonl")
        return audio_files
    
    # Make sure audio_dir is an absolute path
    audio_dir = os.path.abspath(audio_dir)
    print(f"Using audio directory: {audio_dir}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        item = json.loads(line)
                        if isinstance(item, dict):
                            # First try to get a full path if available
                            audio_path = None
                            for field in ['audio_path', 'file_path', 'path']:
                                if field in item and item[field]:
                                    audio_path = item[field]
                                    # If it's a relative path, make it absolute
                                    if not os.path.isabs(audio_path):
                                        audio_path = os.path.join(audio_dir, audio_path)
                                    break
                            
                            # If no path found, try to get just the filename
                            if not audio_path and 'audio_file' in item and item['audio_file']:
                                audio_file = item['audio_file']
                                audio_path = os.path.join(audio_dir, audio_file)
                            
                            # If we still don't have a path but have a context_id, try to construct filename
                            if not audio_path and 'context_id' in item and item['context_id']:
                                # Try both wav and mp3 extensions
                                for ext in ['.wav', '.mp3']:
                                    possible_path = os.path.join(audio_dir, f"{item['context_id']}{ext}")
                                    if os.path.exists(possible_path):
                                        audio_path = possible_path
                                        break
                            
                            # Check if the file exists
                            if audio_path:
                                if os.path.exists(audio_path):
                                    audio_files.append(audio_path)
                                else:
                                    # Try a case-insensitive search in the directory
                                    filename = os.path.basename(audio_path)
                                    dirname = os.path.dirname(audio_path)
                                    found = False
                                    
                                    if os.path.exists(dirname):
                                        for actual_file in os.listdir(dirname):
                                            if actual_file.lower() == filename.lower():
                                                correct_path = os.path.join(dirname, actual_file)
                                                audio_files.append(correct_path)
                                                found = True
                                                print(f"Found file with different case: {correct_path}")
                                                break
                                    
                                    if not found:
                                        context_id = item.get('context_id', f"Line {line_num}")
                                        files_not_found.append((context_id, audio_path))
                                        print(f"Warning: Audio file not found: {audio_path}")
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON at line {line_num}: {e}")
        
        # Print summary of not found files
        if files_not_found:
            print(f"\nWARNING: {len(files_not_found)} audio files not found:")
            for context_id, path in files_not_found[:10]:  # Show first 10
                print(f"  - {context_id}: {path}")
            if len(files_not_found) > 10:
                print(f"  ... and {len(files_not_found) - 10} more")
            
            print("\nCommon causes for missing files:")
            print("1. Incorrect input directory (--input-dir)")
            print("2. File extensions mismatch (.wav vs .mp3)")
            print("3. File naming doesn't match context IDs")
            print("4. Files are in subdirectories not searched")
        
        print(f"Loaded {len(audio_files)} audio file paths from {file_path}")
        return audio_files
    
    except Exception as e:
        print(f"Error loading audio files from {file_path}: {e}")
        return audio_files

def main():
    """Main function for CLI usage"""
    parser = argparse.ArgumentParser(description="Transcribe audio files using Google Cloud Speech-to-Text V2 with GCS support")
    parser.add_argument(
        "--input-dir", 
        help="Directory containing audio files to transcribe"
    )
    parser.add_argument(
        "--input-jsonl",
        help="JSONL file containing audio file references to transcribe"
    )
    parser.add_argument(
        "--gcs-bucket", required=True,
        help="Google Cloud Storage bucket name"
    )
    parser.add_argument(
        "--project-id", 
        help="Google Cloud project ID (required for Speech V2)"
    )
    parser.add_argument(
        "--pattern", default=".wav",
        help="File pattern to match (e.g. .mp3, .wav)"
    )
    parser.add_argument(
        "--output-dir", default="./transcriptions",
        help="Directory to save transcription results"
    )
    parser.add_argument(
        "--language", default="vi-VN",
        help="Primary language code for transcription"
    )
    parser.add_argument(
        "--alt-language", action="append", default=["en-US"],
        help="Alternative language code(s)"
    )
    parser.add_argument(
        "--workers", type=int, default=5,
        help="Maximum number of concurrent workers"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit the number of files to process (0 = no limit)"
    )
    parser.add_argument(
        "--filter-contexts", default=None,
        help="Optional JSON or JSONL file with context_ids to filter by"
    )
    parser.add_argument(
        "--credentials", default=None,
        help="Path to Google Cloud credentials JSON file"
    )
    parser.add_argument(
        "--log-file", default=None,
        help="Path to log file"
    )
    parser.add_argument(
        "--no-cleanup", action="store_true",
        help="Don't delete GCS files after processing"
    )
    parser.add_argument(
        "--force-gcs", action="store_true",
        help="Force using GCS for all files regardless of size/duration"
    )
    
    args = parser.parse_args()
    
    # Ensure we have either input-dir or input-jsonl
    if not args.input_dir and not args.input_jsonl:
        print("Error: Either --input-dir or --input-jsonl must be specified")
        return 1
    
    # Find audio files - either from directory or from JSONL
    audio_files = []
    if args.input_jsonl:
        audio_files = load_audio_files_from_jsonl(args.input_jsonl, args.input_dir)
    elif args.input_dir:
        audio_files = find_audio_files(args.input_dir, args.pattern)
    
    if not audio_files:
        print(f"No audio files found to process")
        return 1
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Load filter context IDs if specified
    filter_context_ids = None
    if args.filter_contexts and os.path.exists(args.filter_contexts):
        filter_context_ids = load_context_ids(args.filter_contexts)
    
    # Initialize processor
    try:
        processor = GCSTranscriptionProcessor(
            output_dir=args.output_dir,
            gcs_bucket=args.gcs_bucket,
            language_code=args.language,
            alternative_language_codes=args.alt_language,
            max_workers=args.workers,
            credentials_path=args.credentials,
            log_file=args.log_file,
            cleanup_gcs=not args.no_cleanup,
            force_gcs=args.force_gcs,
            project_id=args.project_id
        )
        
        # Process batch
        limit = args.limit if args.limit > 0 else None
        results = processor.process_batch(audio_files, filter_context_ids, limit)
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"  Total files: {results['total_files']}")
        print(f"  Successful: {results['success_count']}")
        print(f"  Empty results: {results['empty_count']}")
        print(f"  Failed: {results['failed_count']}")
        print(f"  Total time: {results['total_time']:.2f} seconds")
        print(f"  Average time per file: {results['avg_time_per_file']:.2f} seconds")
        print(f"  JSONL output saved to: {results['jsonl_path']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main() 