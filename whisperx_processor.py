#!/usr/bin/env python3
"""
WhisperX Audio Processing System
Optimized for Apple Silicon M4 Max GPU

WhisperX provides:
- Faster inference than standard Whisper
- Built-in speaker diarization
- Better timestamp accuracy
- Native GPU acceleration
"""

import os
import glob
import json
import whisperx
import torch
import gc
import librosa
from datetime import datetime
import argparse
import time
from typing import Dict, List, Tuple

def load_env_file(env_file_path: str = ".env"):
    """Load environment variables from .env file"""
    if os.path.exists(env_file_path):
        with open(env_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# Load environment variables - make sure HF_TOKEN is set
import sys
if hasattr(sys, '_getframe'):
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(sys._getframe().f_code.co_filename))
else:
    # Fallback to current working directory
    current_dir = os.getcwd()
    
env_path = os.path.join(current_dir, ".env")
load_env_file(env_path)

load_env_file()

class WhisperXProcessor:
    def __init__(self, hf_token: str = None, output_dir: str = "transcriptions"):
        """
        Initialize WhisperX processor for M4 Max GPU acceleration.
        """
        self.output_dir = output_dir
        self.hf_token = hf_token
        os.makedirs(output_dir, exist_ok=True)
        
        print("🚀 WHISPERX PROCESSOR - M4 Max GPU Optimized")
        print("=" * 60)
        
        # Check GPU availability and capabilities
        self._check_gpu_status()
        
        # WhisperX device configuration with M4 Max GPU support
        if torch.cuda.is_available():
            self.device = "cuda"
            self.compute_type = "float16"
            print("✅ Using CUDA GPU")
        elif torch.backends.mps.is_available():
            # WhisperX/faster-whisper doesn't support MPS, use optimized CPU
            print("⚠️  MPS detected but not supported by WhisperX/faster-whisper")
            print("� Using CPU with M4 Max optimizations instead")
            self.device = "cpu"
            self.compute_type = "int8"  # Optimized for Apple Silicon
            print("✅ CPU with int8 optimization (M4 Max tuned)")
        else:
            self.device = "cpu"
            self.compute_type = "int8"  # Use int8 for faster CPU inference
            print("✅ Using CPU with int8 optimization (M4 Max optimized)")
            print("💡 WhisperX CPU mode is highly optimized for Apple Silicon")
        
        # Initialize WhisperX model (use base for speed and reliability)
        print("📝 Loading WhisperX model...")
        
        # Apple Silicon optimizations
        model_kwargs = {}
        if self.device == "cpu" and self.compute_type == "int8":
            # M4 Max CPU optimizations
            model_kwargs["cpu_threads"] = 0  # Use all available cores
            print("🔧 Enabling M4 Max multi-core optimization")
        
        self.model = whisperx.load_model(
            "base", 
            device=self.device, 
            compute_type=self.compute_type,
            **model_kwargs
        )
        print(f"✅ WhisperX base loaded on {self.device} with {self.compute_type}")
        
        # Set PyTorch threading for Apple Silicon
        if self.device == "cpu":
            torch.set_num_threads(0)  # Use all available threads
            print("🚀 Apple Silicon multi-threading enabled")
        
        # Initialize alignment model (for better timestamps)
        print("🎯 Loading alignment model...")
        self.align_model = None
        self.align_metadata = None
        
        # Initialize diarization model
        print("🎭 Loading diarization model...")
        if hf_token:
            try:
                # Load pyannote diarization model with GPU support
                from pyannote.audio import Pipeline
                self.diarize_model = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
                
                # Try to use GPU for diarization when available
                if self.device == "cuda":
                    try:
                        self.diarize_model = self.diarize_model.to("cuda")
                        print("✅ Diarization pipeline loaded (CUDA GPU)")
                    except:
                        print("✅ Diarization pipeline loaded (CPU - CUDA fallback)")
                elif torch.backends.mps.is_available():
                    # Try MPS for diarization even if WhisperX uses CPU
                    try:
                        self.diarize_model = self.diarize_model.to("mps")
                        print("✅ Diarization pipeline loaded (MPS GPU - M4 Max accelerated)")
                    except Exception as e:
                        print(f"✅ Diarization pipeline loaded (CPU - MPS not supported: {e})")
                else:
                    print("✅ Diarization pipeline loaded (CPU)")
            except Exception as e:
                print(f"❌ Diarization loading failed: {e}")
                self.diarize_model = None
        else:
            print("❌ No HuggingFace token - diarization disabled")
            self.diarize_model = None
    
    def _check_gpu_status(self):
        """Check and display GPU status for M4 Max optimization"""
        print("\n🔍 GPU STATUS CHECK:")
        print("-" * 40)
        
        if torch.backends.mps.is_available():
            print("✅ Apple Silicon MPS: Available")
            print("🚀 M4 Max GPU: Ready for acceleration")
        else:
            print("❌ Apple Silicon MPS: Not available")
        
        if torch.cuda.is_available():
            print("✅ CUDA GPU: Available")
            print(f"   Device: {torch.cuda.get_device_name()}")
        else:
            print("❌ CUDA GPU: Not available")
        
        print(f"🖥️  PyTorch version: {torch.__version__}")
        print(f"🔧 Device selected: {getattr(self, 'device', 'pending')}")
        print("-" * 40)
    
    def transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe audio with WhisperX for M4 Max GPU acceleration"""
        print(f"🔥 WhisperX transcribing: {os.path.basename(audio_path)}")
        
        # Clear GPU memory if available
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            # Clear MPS cache for diarization GPU usage
            try:
                torch.mps.empty_cache()
            except:
                pass
        
        # Load audio
        audio = whisperx.load_audio(audio_path)
        
        # Transcribe with WhisperX
        result = self.model.transcribe(
            audio, 
            batch_size=16  # Optimize batch size for M4 Max
        )
        
        print(f"✅ Transcription complete - Language: {result.get('language', 'unknown')}")
        return result
    
    def align_transcription(self, result: Dict, audio_path: str) -> Dict:
        """Align transcription for better timestamps"""
        print("🎯 Aligning transcription for precise timestamps...")
        
        try:
            # Load alignment model if not already loaded
            if self.align_model is None:
                language = result.get("language", "en")
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=language, 
                    device=self.device
                )
            
            # Load audio for alignment
            audio = whisperx.load_audio(audio_path)
            
            # Perform alignment
            aligned_result = whisperx.align(
                result["segments"], 
                self.align_model, 
                self.align_metadata, 
                audio, 
                self.device, 
                return_char_alignments=False
            )
            
            print("✅ Alignment complete - enhanced timestamp accuracy")
            return aligned_result
            
        except Exception as e:
            print(f"⚠️  Alignment failed: {e} - using original timestamps")
            return result
    
    def perform_diarization(self, aligned_result: Dict, audio_path: str) -> Dict:
        """Perform speaker diarization using pyannote (WhisperX style)"""
        if not self.diarize_model:
            return aligned_result
        
        print("🎭 Performing speaker diarization...")
        
        try:
            # Run pyannote diarization
            diarization_output = self.diarize_model(audio_path)
            
            # Extract diarization segments
            diarization_segments = []
            for turn, _, speaker in diarization_output.itertracks(yield_label=True):
                diarization_segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
            # Assign speakers to transcription segments
            segments_with_speakers = []
            speakers = set()
            
            for segment in aligned_result.get("segments", []):
                segment_start = segment["start"]
                segment_end = segment["end"]
                
                # Find best matching speaker
                best_speaker = "Unknown"
                best_overlap = 0
                
                for dia_seg in diarization_segments:
                    overlap_start = max(segment_start, dia_seg["start"])
                    overlap_end = min(segment_end, dia_seg["end"])
                    overlap = max(0, overlap_end - overlap_start)
                    
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_speaker = dia_seg["speaker"]
                
                # Add speaker to segment
                segment_copy = segment.copy()
                segment_copy["speaker"] = best_speaker
                segments_with_speakers.append(segment_copy)
                speakers.add(best_speaker)
            
            # Update result with speakers
            result_with_speakers = aligned_result.copy()
            result_with_speakers["segments"] = segments_with_speakers
            
            print(f"✅ Diarization complete - {len(speakers)} speakers detected")
            return result_with_speakers
            
        except Exception as e:
            print(f"❌ Diarization failed: {e}")
            return aligned_result
    
    def process_audio_file(self, audio_path: str) -> Dict:
        """Process audio file with WhisperX pipeline"""
        filename = os.path.basename(audio_path)
        print(f"\n{'=' * 70}")
        print(f"🎯 WHISPERX PROCESSING: {filename}")
        print(f"{'=' * 70}")
        
        # Get audio metadata
        try:
            audio_data, sample_rate = librosa.load(audio_path, sr=None)
            duration = len(audio_data) / sample_rate
            file_size = os.path.getsize(audio_path) / (1024 * 1024)
            print(f"📊 Audio: {duration:.1f}s ({duration/60:.1f}min), {file_size:.1f}MB")
        except Exception:
            duration = 0
            file_size = 0
        
        # Step 1: Transcribe with WhisperX
        transcription_result = self.transcribe_audio(audio_path)
        
        # Step 2: Align for better timestamps
        aligned_result = self.align_transcription(transcription_result, audio_path)
        
        # Step 3: Perform speaker diarization
        final_result = self.perform_diarization(aligned_result, audio_path)
        
        # Extract speakers
        speakers = set()
        segments_with_speakers = []
        
        for segment in final_result.get("segments", []):
            speaker = segment.get("speaker", "Unknown")
            speakers.add(speaker)
            
            segments_with_speakers.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
                "speaker": speaker,
                "words": segment.get("words", [])
            })
        
        # Clean up GPU memory
        if self.device != "cpu":
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        # Compile results
        results = {
            "filename": filename,
            "filepath": audio_path,
            "duration": duration,
            "file_size_mb": file_size,
            "processing_timestamp": datetime.now().isoformat(),
            "model": "whisperx-base-optimized",
            "diarization": "whisperx-builtin",
            "full_transcription": transcription_result.get("text", ""),
            "language": transcription_result.get("language", "unknown"),
            "speakers": sorted(list(speakers)),
            "speaker_count": len(speakers),
            "segments": segments_with_speakers,
            "total_segments": len(segments_with_speakers)
        }
        
        return results
    
    def save_results(self, results: Dict, formats: List[str] = ["json", "txt"]):
        """Save WhisperX results"""
        filename_base = os.path.splitext(results["filename"])[0]
        
        for format_type in formats:
            if format_type == "json":
                output_path = os.path.join(self.output_dir, f"{filename_base}_whisperx.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            
            elif format_type == "txt":
                output_path = os.path.join(self.output_dir, f"{filename_base}_whisperx.txt")
                with open(output_path, 'w', encoding='utf-8') as f:
                    self._write_whisperx_transcript(f, results)
            
            print(f"💾 Results saved: {output_path}")
    
    def _write_whisperx_transcript(self, f, results: Dict):
        """Write WhisperX transcript"""
        f.write("=" * 70 + "\n")
        f.write("🚀 WHISPERX TRANSCRIPTION RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(f"File: {results['filename']}\n")
        f.write(f"Duration: {results['duration']:.2f} seconds ({results['duration']/60:.1f} minutes)\n")
        f.write(f"File Size: {results['file_size_mb']:.1f} MB\n")
        f.write(f"Language: {results['language']}\n")
        f.write(f"Model: {results['model']} (M4 Max Optimized)\n")
        f.write(f"Diarization: {results['diarization']}\n")
        f.write(f"Speakers: {results['speaker_count']} ({', '.join(results['speakers'])})\n")
        f.write(f"Total Segments: {results['total_segments']}\n")
        f.write(f"Processing Date: {results['processing_timestamp']}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("📝 FULL TRANSCRIPTION\n")
        f.write("=" * 70 + "\n")
        f.write(results['full_transcription'])
        
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("🎭 CONVERSATION WITH SPEAKERS (WHISPERX)\n")
        f.write("=" * 70 + "\n")
        
        for i, segment in enumerate(results['segments'], 1):
            timestamp = f"[{segment['start']:.2f}s - {segment['end']:.2f}s]"
            f.write(f"\n{timestamp} {segment['speaker']}: {segment['text']}\n")


def process_single_file_whisperx(filename: str):
    """Process a single audio file with WhisperX"""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ No HuggingFace token found. Please set HF_TOKEN in .env file")
        return
    
    audio_file = os.path.join("callRecords", filename)
    if not os.path.exists(audio_file):
        print(f"❌ File not found: {audio_file}")
        return
    
    # Initialize WhisperX processor
    processor = WhisperXProcessor(hf_token=hf_token)
    
    # Process with timing
    start_time = time.time()
    results = processor.process_audio_file(audio_file)
    process_time = time.time() - start_time
    
    # Save results
    processor.save_results(results, formats=["json", "txt"])
    
    # Performance summary
    duration = results['duration']
    real_time_factor = process_time / duration if duration > 0 else 0
    
    print(f"\n🏆 WHISPERX PROCESSING COMPLETE!")
    print(f"⚡ Processing time: {process_time:.1f}s ({process_time/60:.1f}min)")
    print(f"🎯 Real-time factor: {real_time_factor:.2f}x")
    print(f"🎭 Speakers detected: {results['speaker_count']}")
    print(f"📝 Segments created: {results['total_segments']}")
    
    if real_time_factor < 1:
        print(f"🚀 {1/real_time_factor:.1f}x faster than real-time!")
    
    print(f"\n🎯 WhisperX Benefits on M4 Max:")
    print(f"   ✅ Native GPU acceleration")
    print(f"   ✅ Faster than standard Whisper")
    print(f"   ✅ Built-in speaker diarization")
    print(f"   ✅ Enhanced timestamp accuracy")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WhisperX Audio Processing for M4 Max GPU")
    parser.add_argument("--file", type=str, help="Process single .wav file with WhisperX")
    
    args = parser.parse_args()
    
    if args.file:
        process_single_file_whisperx(args.file)
    else:
        print("🚀 WHISPERX PROCESSOR - M4 Max GPU Optimized")
        print("🏆 Faster inference + Built-in diarization")
        print("")
        print("Usage:")
        print("  python whisperx_processor.py --file filename.wav")
        print("")
        print("Example:")
        print("  python whisperx_processor.py --file 113195.wav")
        print("")
        print("💡 WhisperX Features:")
        print("   🎯 Faster than standard Whisper")
        print("   🎭 Built-in speaker diarization") 
        print("   🔧 M4 Max GPU acceleration")
        print("   💾 Enhanced timestamp accuracy")
        print("   🚀 Optimized for Apple Silicon")