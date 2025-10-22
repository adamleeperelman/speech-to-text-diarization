#!/usr/bin/env python3
"""
Speech-to-Text with Speaker Diarization

This script processes .wav files from the callRecords directory,
performs speech-to-text conversion using OpenAI Whisper,
and includes speaker diarization using pyannote.audio.
"""

import os
import glob
import json
import whisper
import torch
from pyannote.audio import Pipeline
from pyannote.core import Segment
import librosa
import pandas as pd
from datetime import datetime
import argparse
from typing import Dict, List, Tuple

# Load environment variables from .env file if it exists
def load_env_file():
    """Load environment variables from .env file if it exists"""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env_file()

class SpeechProcessor:
    def __init__(self, 
                 whisper_model: str = "base",
                 hf_token: str = None,
                 output_dir: str = "transcriptions"):
        """
        Initialize the speech processor with Whisper and pyannote models.
        
        Args:
            whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            hf_token: HuggingFace token for pyannote.audio (required for diarization)
            output_dir: Directory to save transcription results
        """
        self.whisper_model_name = whisper_model
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load Whisper model
        print(f"Loading Whisper model: {whisper_model}")
        self.whisper_model = whisper.load_model(whisper_model)
        
        # Load diarization pipeline
        if hf_token:
            print("Loading speaker diarization model...")
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token
            )
        else:
            print("Warning: No HuggingFace token provided. Speaker diarization will be disabled.")
            self.diarization_pipeline = None
    
    def transcribe_audio(self, audio_path: str) -> Dict:
        """
        Transcribe audio file using Whisper.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing transcription results
        """
        print(f"Transcribing: {os.path.basename(audio_path)}")
        
        # Transcribe with Whisper
        result = self.whisper_model.transcribe(
            audio_path,
            word_timestamps=True,
            language=None  # Auto-detect language
        )
        
        return result
    
    def perform_diarization(self, audio_path: str) -> Dict:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing diarization results
        """
        if not self.diarization_pipeline:
            return {"speakers": [], "segments": []}
        
        print(f"Performing speaker diarization: {os.path.basename(audio_path)}")
        
        # Perform diarization
        diarization = self.diarization_pipeline(audio_path)
        
        # Extract speaker segments
        segments = []
        speakers = set()
        
        try:
            # Access the speaker_diarization attribute from DiarizeOutput
            diarization_result = diarization.speaker_diarization
            
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                    "duration": turn.end - turn.start
                })
                speakers.add(speaker)
        except Exception as e:
            print(f"Diarization processing error: {e}")
            # Continue without diarization data
            pass
        
        return {
            "speakers": sorted(list(speakers)),
            "segments": segments
        }
    
    def align_transcription_with_speakers(self, 
                                        transcription: Dict, 
                                        diarization: Dict) -> List[Dict]:
        """
        Align transcription segments with speaker diarization.
        
        Args:
            transcription: Whisper transcription results
            diarization: Speaker diarization results
            
        Returns:
            List of aligned segments with speaker labels
        """
        if not diarization["segments"]:
            # No diarization available, return transcription without speaker labels
            aligned_segments = []
            for segment in transcription.get("segments", []):
                aligned_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "speaker": "Unknown",
                    "words": segment.get("words", [])
                })
            return aligned_segments
        
        aligned_segments = []
        
        for trans_segment in transcription.get("segments", []):
            trans_start = trans_segment["start"]
            trans_end = trans_segment["end"]
            trans_mid = (trans_start + trans_end) / 2
            
            # Find the speaker segment that best overlaps with this transcription segment
            best_speaker = "Unknown"
            best_overlap = 0
            
            for dia_segment in diarization["segments"]:
                dia_start = dia_segment["start"]
                dia_end = dia_segment["end"]
                
                # Calculate overlap
                overlap_start = max(trans_start, dia_start)
                overlap_end = min(trans_end, dia_end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = dia_segment["speaker"]
            
            aligned_segments.append({
                "start": trans_start,
                "end": trans_end,
                "text": trans_segment["text"].strip(),
                "speaker": best_speaker,
                "words": trans_segment.get("words", [])
            })
        
        return aligned_segments
    
    def process_audio_file(self, audio_path: str) -> Dict:
        """
        Process a single audio file with transcription and diarization.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing all processing results
        """
        filename = os.path.basename(audio_path)
        print(f"\n{'='*50}")
        print(f"Processing: {filename}")
        print(f"{'='*50}")
        
        # Get audio info
        try:
            audio_data, sample_rate = librosa.load(audio_path, sr=None)
            duration = len(audio_data) / sample_rate
        except Exception as e:
            print(f"Error loading audio info: {e}")
            duration = 0
        
        # Perform transcription
        try:
            transcription = self.transcribe_audio(audio_path)
        except Exception as e:
            print(f"Error during transcription: {e}")
            transcription = {"segments": [], "text": ""}
        
        # Perform diarization
        try:
            diarization = self.perform_diarization(audio_path)
        except Exception as e:
            print(f"Error during diarization: {e}")
            diarization = {"speakers": [], "segments": []}
        
        # Align transcription with speakers
        try:
            aligned_segments = self.align_transcription_with_speakers(transcription, diarization)
        except Exception as e:
            print(f"Error during alignment: {e}")
            aligned_segments = []
        
        # Compile results
        results = {
            "filename": filename,
            "filepath": audio_path,
            "duration": duration,
            "processing_timestamp": datetime.now().isoformat(),
            "whisper_model": self.whisper_model_name,
            "full_transcription": transcription.get("text", ""),
            "language": transcription.get("language", "unknown"),
            "speakers": diarization["speakers"],
            "speaker_count": len(diarization["speakers"]),
            "segments": aligned_segments
        }
        
        return results
    
    def save_results(self, results: Dict, format: str = "json"):
        """
        Save processing results to file.
        
        Args:
            results: Processing results dictionary
            format: Output format ('json', 'txt', 'csv')
        """
        filename_base = os.path.splitext(results["filename"])[0]
        
        if format == "json":
            output_path = os.path.join(self.output_dir, f"{filename_base}_transcription.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        elif format == "txt":
            output_path = os.path.join(self.output_dir, f"{filename_base}_transcription.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"File: {results['filename']}\n")
                f.write(f"Duration: {results['duration']:.2f} seconds\n")
                f.write(f"Language: {results['language']}\n")
                f.write(f"Speakers: {', '.join(results['speakers'])}\n")
                f.write(f"Processing Date: {results['processing_timestamp']}\n")
                f.write("\n" + "="*50 + "\n")
                f.write("FULL TRANSCRIPTION:\n")
                f.write("="*50 + "\n")
                f.write(results['full_transcription'])
                f.write("\n\n" + "="*50 + "\n")
                f.write("SEGMENTED TRANSCRIPTION WITH SPEAKERS:\n")
                f.write("="*50 + "\n")
                
                for segment in results['segments']:
                    f.write(f"\n[{segment['start']:.2f}s - {segment['end']:.2f}s] "
                           f"{segment['speaker']}: {segment['text']}\n")
        
        elif format == "csv":
            output_path = os.path.join(self.output_dir, f"{filename_base}_segments.csv")
            df = pd.DataFrame(results['segments'])
            df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"Results saved to: {output_path}")
    
    def process_directory(self, 
                         directory_path: str, 
                         pattern: str = "*.wav",
                         save_formats: List[str] = ["json", "txt"]) -> List[Dict]:
        """
        Process all audio files in a directory.
        
        Args:
            directory_path: Path to directory containing audio files
            pattern: File pattern to match (default: "*.wav")
            save_formats: List of output formats to save
            
        Returns:
            List of processing results for all files
        """
        audio_files = glob.glob(os.path.join(directory_path, pattern))
        audio_files.sort()
        
        if not audio_files:
            print(f"No audio files found matching pattern: {pattern}")
            return []
        
        print(f"Found {len(audio_files)} audio files to process")
        
        all_results = []
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\nProgress: {i}/{len(audio_files)}")
            
            try:
                results = self.process_audio_file(audio_file)
                all_results.append(results)
                
                # Save individual results
                for format_type in save_formats:
                    self.save_results(results, format_type)
                    
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
        
        # Save summary results
        self.save_summary_results(all_results)
        
        return all_results
    
    def save_summary_results(self, all_results: List[Dict]):
        """Save summary of all processing results."""
        summary_path = os.path.join(self.output_dir, "processing_summary.json")
        
        summary = {
            "processing_timestamp": datetime.now().isoformat(),
            "total_files": len(all_results),
            "total_duration": sum(r["duration"] for r in all_results),
            "files": [
                {
                    "filename": r["filename"],
                    "duration": r["duration"],
                    "language": r["language"],
                    "speaker_count": r["speaker_count"],
                    "speakers": r["speakers"]
                }
                for r in all_results
            ]
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nProcessing summary saved to: {summary_path}")
        print(f"Total files processed: {len(all_results)}")
        print(f"Total duration: {summary['total_duration']:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description="Speech-to-Text with Speaker Diarization")
    parser.add_argument("--input-dir", 
                       default=os.getenv("INPUT_DIR", "callRecords"), 
                       help="Directory containing .wav files")
    parser.add_argument("--output-dir", 
                       default=os.getenv("OUTPUT_DIR", "transcriptions"), 
                       help="Directory to save results")
    parser.add_argument("--whisper-model", 
                       default=os.getenv("WHISPER_MODEL", "base"), 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size")
    parser.add_argument("--hf-token", 
                       default=os.getenv("HF_TOKEN"),
                       help="HuggingFace token for speaker diarization")
    parser.add_argument("--formats", 
                       nargs='+',
                       default=["json", "txt"],
                       choices=["json", "txt", "csv"],
                       help="Output formats")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = SpeechProcessor(
        whisper_model=args.whisper_model,
        hf_token=args.hf_token,
        output_dir=args.output_dir
    )
    
    # Process all files
    results = processor.process_directory(
        directory_path=args.input_dir,
        pattern="*.wav",
        save_formats=args.formats
    )
    
    print(f"\nProcessing complete! Check the '{args.output_dir}' directory for results.")

if __name__ == "__main__":
    main()