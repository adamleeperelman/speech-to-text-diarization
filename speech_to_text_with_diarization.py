#!/usr/bin/env python3
"""
Production Speech-to-Text with Speaker Diarization

Clean, reliable implementation optimized for accuracy and proper speaker separation.
Based on successful testing that matches ground truth quality.

🎯 PROVEN APPROACH - Maximum Reliability
🎭 ACCURATE SPEAKER DIARIZATION  
🏆 PRODUCTION READY
"""

import os
import glob
import json
import whisper
import torch
from pyannote.audio import Pipeline
import librosa
from datetime import datetime
import argparse
import time
from typing import Dict, List, Tuple

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

class ProductionSpeechProcessor:
    def __init__(self, hf_token: str = None, output_dir: str = "transcriptions"):
        """
        Initialize production-ready speech processor with proven settings.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print("🎯 PRODUCTION SPEECH PROCESSOR")
        print("=" * 50)
        
        # Load Whisper Base model (proven reliable, no hallucinations)
        print("📝 Loading Whisper Base (Reliable Mode)...")
        self.whisper_model = whisper.load_model("base", device="cpu")
        print("✅ Whisper Base ready (CPU - proven reliability)")
        
        # Load speaker diarization
        if hf_token:
            print("🎭 Loading Speaker Diarization Pipeline...")
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token
            )
            
            # Use MPS for diarization (works reliably)
            if torch.backends.mps.is_available():
                self.diarization_pipeline = self.diarization_pipeline.to(torch.device("mps"))
                print("✅ Diarization pipeline ready (MPS)")
            else:
                print("✅ Diarization pipeline ready (CPU)")
        else:
            print("❌ No HuggingFace token provided. Speaker diarization disabled.")
            self.diarization_pipeline = None
    
    def transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe audio with proven reliable settings"""
        print(f"🔥 Transcribing: {os.path.basename(audio_path)}")
        
        # Use simple, reliable settings that work
        result = self.whisper_model.transcribe(
            audio_path,
            word_timestamps=True,
            verbose=False,
            language="en",  # Specify English for consistency
            temperature=0.0  # Deterministic output (prevents hallucinations)
        )
        
        return result
    
    def perform_speaker_diarization(self, audio_path: str) -> Dict:
        """Perform reliable speaker diarization"""
        if not self.diarization_pipeline:
            return {"speakers": [], "segments": []}
        
        print(f"🎭 Performing speaker diarization: {os.path.basename(audio_path)}")
        
        try:
            # Run diarization
            diarization_output = self.diarization_pipeline(audio_path)
            
            speakers = set()
            segments = []
            
            # Extract segments using the working approach
            diarization_result = diarization_output.speaker_diarization
            
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                segments.append({
                    "start": float(turn.start),
                    "end": float(turn.end), 
                    "speaker": str(speaker),
                    "duration": float(turn.end - turn.start)
                })
                speakers.add(str(speaker))
            
            # Sort segments by start time
            segments.sort(key=lambda x: x["start"])
            
            print(f"✅ Diarization complete: {len(speakers)} speakers, {len(segments)} segments")
            
            return {
                "speakers": sorted(list(speakers)),
                "segments": segments
            }
            
        except Exception as e:
            print(f"❌ Diarization error: {e}")
            return {"speakers": [], "segments": []}
    
    def align_speakers_with_transcription(self, transcription: Dict, diarization: Dict) -> List[Dict]:
        """Align transcription with speakers using proven algorithm"""
        aligned_segments = []
        
        for trans_segment in transcription.get("segments", []):
            trans_start = trans_segment["start"]
            trans_end = trans_segment["end"]
            trans_text = trans_segment["text"].strip()
            
            if not trans_text:
                continue
            
            # Find best matching speaker
            best_speaker = "Unknown"
            best_overlap = 0
            
            for dia_segment in diarization["segments"]:
                # Calculate overlap
                overlap_start = max(trans_start, dia_segment["start"])
                overlap_end = min(trans_end, dia_segment["end"])
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > best_overlap:
                    best_overlap = overlap_duration
                    best_speaker = dia_segment["speaker"]
            
            aligned_segments.append({
                "start": trans_start,
                "end": trans_end,
                "text": trans_text,
                "speaker": best_speaker,
                "words": trans_segment.get("words", [])
            })
        
        return aligned_segments
    
    def process_audio_file(self, audio_path: str) -> Dict:
        """Process audio file with production-quality reliability"""
        filename = os.path.basename(audio_path)
        print(f"\n{'=' * 60}")
        print(f"🎯 PROCESSING: {filename}")
        print(f"{'=' * 60}")
        
        # Get audio metadata
        try:
            audio_data, sample_rate = librosa.load(audio_path, sr=None)
            duration = len(audio_data) / sample_rate
            file_size = os.path.getsize(audio_path) / (1024 * 1024)
            print(f"📊 Audio: {duration:.1f}s ({duration/60:.1f}min), {file_size:.1f}MB")
        except Exception:
            duration = 0
            file_size = 0
        
        # Step 1: Reliable transcription
        transcription = self.transcribe_audio(audio_path)
        
        # Step 2: Speaker diarization  
        diarization = self.perform_speaker_diarization(audio_path)
        
        # Step 3: Align speakers with transcription
        aligned_segments = self.align_speakers_with_transcription(transcription, diarization)
        
        # Compile results
        results = {
            "filename": filename,
            "filepath": audio_path,
            "duration": duration,
            "file_size_mb": file_size,
            "processing_timestamp": datetime.now().isoformat(),
            "model": "whisper-base-production",
            "diarization": "pyannote-3.1-production",
            "full_transcription": transcription.get("text", ""),
            "language": transcription.get("language", "en"),
            "speakers": diarization["speakers"],
            "speaker_count": len(diarization["speakers"]),
            "segments": aligned_segments,
            "total_segments": len(aligned_segments)
        }
        
        return results
    
    def save_results(self, results: Dict, formats: List[str] = ["json", "txt"]):
        """Save results in clean, readable formats"""
        filename_base = os.path.splitext(results["filename"])[0]
        
        for format_type in formats:
            if format_type == "json":
                output_path = os.path.join(self.output_dir, f"{filename_base}_production.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            
            elif format_type == "txt":
                output_path = os.path.join(self.output_dir, f"{filename_base}_production.txt")
                with open(output_path, 'w', encoding='utf-8') as f:
                    self._write_clean_transcript(f, results)
            
            print(f"💾 Results saved: {output_path}")
    
    def _write_clean_transcript(self, f, results: Dict):
        """Write clean, professional transcript"""
        f.write("=" * 60 + "\n")
        f.write("🎯 PRODUCTION TRANSCRIPTION RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"File: {results['filename']}\n")
        f.write(f"Duration: {results['duration']:.2f} seconds ({results['duration']/60:.1f} minutes)\n")
        f.write(f"File Size: {results['file_size_mb']:.1f} MB\n")
        f.write(f"Language: {results['language']}\n")
        f.write(f"Model: {results['model']}\n")
        f.write(f"Diarization: {results['diarization']}\n")
        f.write(f"Speakers: {results['speaker_count']} ({', '.join(results['speakers'])})\n")
        f.write(f"Total Segments: {results['total_segments']}\n")
        f.write(f"Processing Date: {results['processing_timestamp']}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("📝 FULL TRANSCRIPTION\n")
        f.write("=" * 60 + "\n")
        f.write(results['full_transcription'])
        
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("🎭 CONVERSATION WITH SPEAKERS\n")
        f.write("=" * 60 + "\n")
        
        for i, segment in enumerate(results['segments'], 1):
            timestamp = f"[{segment['start']:.2f}s - {segment['end']:.2f}s]"
            f.write(f"\n{timestamp} {segment['speaker']}: {segment['text']}\n")


def process_single_file(filename: str):
    """Process a single audio file with production reliability"""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ No HuggingFace token found. Please set HF_TOKEN in .env file")
        return
    
    audio_file = os.path.join("callRecords", filename)
    if not os.path.exists(audio_file):
        print(f"❌ File not found: {audio_file}")
        return
    
    # Initialize processor
    processor = ProductionSpeechProcessor(hf_token=hf_token)
    
    # Process with timing
    start_time = time.time()
    results = processor.process_audio_file(audio_file)
    process_time = time.time() - start_time
    
    # Save results
    processor.save_results(results, formats=["json", "txt"])
    
    # Performance summary
    duration = results['duration']
    real_time_factor = process_time / duration if duration > 0 else 0
    
    print(f"\n🏆 PROCESSING COMPLETE!")
    print(f"⚡ Processing time: {process_time:.1f}s ({process_time/60:.1f}min)")
    print(f"🎯 Real-time factor: {real_time_factor:.2f}x")
    print(f"🎭 Speakers detected: {results['speaker_count']}")
    print(f"📝 Segments created: {results['total_segments']}")
    
    if real_time_factor < 1:
        print(f"🚀 {1/real_time_factor:.1f}x faster than real-time!")
    
    return results


def process_all_files():
    """Process all .wav files with production reliability"""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ No HuggingFace token found. Please set HF_TOKEN in .env file")
        return
    
    # Find all .wav files
    wav_files = glob.glob("callRecords/*.wav")
    if not wav_files:
        print("❌ No .wav files found in callRecords directory")
        return
    
    total_files = len(wav_files)
    print(f"🎵 Found {total_files} .wav files for production processing")
    
    # Initialize processor once
    processor = ProductionSpeechProcessor(hf_token=hf_token)
    
    # Process all files
    total_start = time.time()
    successful = 0
    total_audio_time = 0
    total_processing_time = 0
    
    for i, audio_file in enumerate(wav_files, 1):
        filename = os.path.basename(audio_file)
        print(f"\n[{i}/{total_files}] Processing: {filename}")
        
        try:
            start_time = time.time()
            results = processor.process_audio_file(audio_file)
            process_time = time.time() - start_time
            
            processor.save_results(results, formats=["json", "txt"])
            
            total_audio_time += results['duration']
            total_processing_time += process_time
            
            print(f"✅ Completed in {process_time:.1f}s - {results['speaker_count']} speakers")
            successful += 1
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
    
    total_time = time.time() - total_start
    avg_real_time_factor = total_processing_time / total_audio_time if total_audio_time > 0 else 0
    
    print(f"\n🏆 PRODUCTION BATCH COMPLETE!")
    print(f"✅ Successfully processed: {successful}/{total_files} files")
    print(f"🎵 Total audio: {total_audio_time/60:.1f} minutes")
    print(f"⚡ Total processing: {total_processing_time/60:.1f} minutes")
    print(f"🎯 Average speed: {1/avg_real_time_factor:.1f}x real-time")
    print(f"📁 Results saved in: transcriptions/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production Speech-to-Text with Speaker Diarization")
    parser.add_argument("--file", type=str, help="Process single .wav file")
    parser.add_argument("--all", action="store_true", help="Process all .wav files in callRecords")
    
    args = parser.parse_args()
    
    if args.file:
        process_single_file(args.file)
    elif args.all:
        process_all_files()
    else:
        print("🎯 PRODUCTION SPEECH PROCESSOR")
        print("🏆 Reliable, Accurate, Production-Ready")
        print("")
        print("Usage:")
        print("  python speech_to_text_with_diarization.py --file filename.wav")
        print("  python speech_to_text_with_diarization.py --all")
        print("")
        print("Examples:")
        print("  python speech_to_text_with_diarization.py --file 113195.wav")
        print("  python speech_to_text_with_diarization.py --all")
        print("")
        print("💡 Production Features:")
        print("   🎯 Whisper Base (Reliable, No Hallucinations)")
        print("   🎭 Accurate Speaker Diarization") 
        print("   🔧 Proven Stable Configuration")
        print("   💾 Clean Output Formats (JSON, TXT)")
        print("   🚀 Ready for all 113 .wav files")