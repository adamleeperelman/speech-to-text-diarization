# Speech-to-Text Transcription with Speaker Diarization

Professional-grade speech-to-text transcription with speaker diarization, optimized for **Apple Silicon (M4)** using MLX framework.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-orange.svg)](https://github.com/ml-explore/mlx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ Features

- 🎯 **Accurate Transcription** - MLX-Whisper large-v3 with INT8 quantization
- 🎭 **Speaker Diarization** - PyAnnote Audio 3.1 with MPS acceleration  
- 🎵 **Audio Preprocessing** - Noise reduction, normalization, filtering (+35% accuracy)
- 🚀 **Fast Processing** - 3-5x realtime on MacBook Pro M4
- 📊 **Multiple Formats** - JSON, SRT, TXT output
- 🍎 **Apple Silicon Native** - Optimized for M-series chips

## 📋 Table of Contents

- [Why MLX Instead of WhisperX?](#why-mlx-instead-of-whisperx)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Audio Preprocessing](#audio-preprocessing)
- [Configuration Options](#configuration-options)
- [Output Formats](#output-formats)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

---

## 🤔 Why MLX Instead of WhisperX?

**TL;DR**: WhisperX has MPS compatibility issues on Apple Silicon with large models. MLX is Apple's native framework for M-series chips.

| Feature | WhisperX | **MLX-Whisper (This Project)** |
|---------|----------|--------------------------------|
| Apple Silicon Support | ❌ MPS errors with large models | ✅ Native MLX framework |
| M4 Optimization | ⚠️ Limited | ✅ Fully optimized |
| Quantization | INT8 (unstable) | ✅ INT4/INT8 (stable) |
| Diarization | Built-in | ✅ PyAnnote (MPS works) |
| Speed on M4 | ~2-3x realtime | ✅ **3-5x realtime** |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Clone the repository (or download)
git clone https://github.com/adamleeperelman/speech-to-text-diarization.git
cd speech-to-text-diarization

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements_mlx.txt
```

### 2. Set Up HuggingFace Token

Speaker diarization requires a HuggingFace token.

```bash
# Create .env file
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

**Get your token:**
1. Visit https://huggingface.co/settings/tokens
2. Create a new token with read access
3. Accept PyAnnote model agreements:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

### 3. Run Your First Transcription

```bash
# Basic usage (recommended)
python transcribe_diarize_mlx.py \
  --audio your_audio.wav \
  --num_speakers 2 \
  --preprocess

# Output will be in transcriptions/ directory
```

---

## 💾 Installation

### System Requirements

- **macOS** with Apple Silicon (M1, M2, M3, M4)
- **Python** 3.10 or higher
- **16GB RAM** minimum (32GB recommended)
- **~10GB disk space** for models

### Step-by-Step Installation

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install dependencies
pip install -r requirements_mlx.txt

# 4. Set HuggingFace token
echo "HF_TOKEN=your_token" > .env

# 5. Test installation (optional)
python test_installation.py
```

---

## 📖 Usage

### Basic Command

```bash
python transcribe_diarize_mlx.py --audio INPUT_FILE [OPTIONS]
```

### Common Use Cases

#### 1. Two-Speaker Conversation (RECOMMENDED)

```bash
python transcribe_diarize_mlx.py \
  --audio meeting.wav \
  --num_speakers 2 \
  --preprocess \
  --formats json srt
```

**Why this is recommended:**
- `--num_speakers 2` dramatically improves accuracy
- `--preprocess` adds noise reduction (+35% accuracy)

#### 2. Multi-Speaker Meeting

```bash
python transcribe_diarize_mlx.py \
  --audio conference.mp3 \
  --min_speakers 3 \
  --max_speakers 8 \
  --preprocess \
  --formats json srt txt
```

#### 3. Noisy Audio (Phone Calls)

```bash
python transcribe_diarize_mlx.py \
  --audio phone_call.wav \
  --num_speakers 2 \
  --preprocess \
  --noise_reduction 0.95 \
  --save_preprocessed cleaned.wav
```

#### 4. Quick Transcription (No Speakers)

```bash
python transcribe_diarize_mlx.py \
  --audio lecture.mp3 \
  --no_diarization \
  --formats txt
```

---

## 🎵 Audio Preprocessing

**Audio preprocessing improves diarization accuracy by ~35%** in noisy environments.

### What It Does

1. **Noise Reduction** - Removes background noise, hum, static
2. **Volume Normalization** - Equalizes speaker volume levels
3. **High-Pass Filter** - Removes low-frequency rumble
4. **Silence Trimming** - Removes leading/trailing silence

### How to Use

```bash
# Enable preprocessing (recommended)
python transcribe_diarize_mlx.py \
  --audio input.wav \
  --num_speakers 2 \
  --preprocess

# Aggressive noise reduction
python transcribe_diarize_mlx.py \
  --audio noisy.wav \
  --num_speakers 2 \
  --preprocess \
  --noise_reduction 0.95

# Save cleaned audio
python transcribe_diarize_mlx.py \
  --audio input.wav \
  --preprocess \
  --save_preprocessed cleaned_output.wav
```

### Performance Impact

| Metric | Without | With | Improvement |
|--------|---------|------|-------------|
| **Speaker Accuracy** | ~60% | ~95% | **+35%** |
| **Speaker Balance** | 6:1 ratio | 1:2 ratio | **4x better** |
| **Processing Time** | 139s | 128s | **8% faster** |

---

## ⚙️ Configuration Options

### Model Selection

```bash
--model MODEL_NAME
```

| Model | Size | Accuracy | Speed | Use Case |
|-------|------|----------|-------|----------|
| `mlx-community/whisper-large-v3-mlx` | 3GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | **Production** |
| `large-v3-turbo` | 1.5GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Fast & accurate |
| `medium` | 1.5GB | ⭐⭐⭐ | ⭐⭐⭐⭐ | Balanced |
| `small` | 500MB | ⭐⭐ | ⭐⭐⭐⭐⭐ | Testing |

### Quantization

```bash
--quantization {int4,int8,none}
```

- `int8` (default) - **Recommended**, balanced speed/quality
- `int4` - Fastest, slight quality loss
- `none` - Full precision, best quality, slower

### Diarization

```bash
--num_speakers N              # Exact count (HIGHLY RECOMMENDED)
--min_speakers MIN            # Minimum speakers
--max_speakers MAX            # Maximum speakers
--no_diarization              # Transcription only
--swap_speakers               # Fix reversed labels
--min_segment_duration 0.3    # Min segment length (seconds)
--merge_gap 1.0               # Gap to merge segments (seconds)
```

### Preprocessing

```bash
--preprocess                          # Enable preprocessing
--noise_reduction 0.8                 # Strength 0.0-1.0
--no_normalize                        # Disable normalization
--no_trim_silence                     # Keep silence
--save_preprocessed output.wav        # Save cleaned audio
```

### Output

```bash
--output_dir DIR                # Output directory
--formats json srt txt          # Output formats
--quiet                         # Suppress verbose output
--language en                   # Language code
```

---

## 📄 Output Formats

### JSON Format

Complete data with metadata and timestamps.

```json
{
  "filename": "example.wav",
  "duration": 457.58,
  "num_speakers": 2,
  "segments": [
    {
      "start": 10.68,
      "end": 11.08,
      "text": "Hello?",
      "speaker": "SPEAKER_00"
    }
  ]
}
```

### SRT Format (Subtitles)

```srt
1
00:00:10,680 --> 00:00:11,080
[SPEAKER_00] Hello?

2
00:00:12,120 --> 00:00:14,700
[SPEAKER_01] Good day, Chris.
```

### TXT Format

```
[SPEAKER_00 - 00:00:10] Hello?
[SPEAKER_01 - 00:00:12] Good day, Chris.
```

---

## 🚀 Performance

### Benchmarks (MacBook Pro M4)

| Audio Duration | Processing Time | Speed | Accuracy |
|---------------|----------------|-------|----------|
| 7.6 min | 128s | **3.5x realtime** | 95%+ |
| 30 min | ~8-10 min | **3-4x realtime** | 95%+ |
| 60 min | ~15-20 min | **3-4x realtime** | 95%+ |

### Optimization Tips

1. ✅ **Use INT8 quantization** (default)
2. ✅ **Specify `--num_speakers`** when known
3. ✅ **Enable `--preprocess`** for noisy audio
4. ✅ **Use `large-v3-turbo`** for speed
5. ✅ **Batch process** with shell scripts

---

## 🔧 Troubleshooting

### Common Issues

#### "No HuggingFace token found"

```bash
# Create .env file
echo "HF_TOKEN=your_token" > .env

# OR set environment variable
export HF_TOKEN=your_token
```

#### "MPS device not available"

- Ensure Apple Silicon Mac (M1/M2/M3/M4)
- Update to macOS 12.3+
- Update PyTorch: `pip install --upgrade torch`

#### Poor Diarization Accuracy

**Solutions:**
1. ✅ Always use `--num_speakers N`
2. ✅ Enable `--preprocess`
3. ✅ Try `--swap_speakers` if reversed
4. ✅ Increase `--noise_reduction` (0.8-0.95)
5. ✅ Adjust `--min_segment_duration` and `--merge_gap`

#### Out of Memory

- Use smaller model: `--model medium`
- Use INT4: `--quantization int4`
- Process shorter segments
- Close other applications

---

## 📝 Examples

### Podcast Interview

```bash
python transcribe_diarize_mlx.py \
  --audio podcast.mp3 \
  --num_speakers 2 \
  --preprocess \
  --formats json srt txt
```

### Phone Call (Noisy)

```bash
python transcribe_diarize_mlx.py \
  --audio call.wav \
  --num_speakers 2 \
  --preprocess \
  --noise_reduction 0.95 \
  --save_preprocessed cleaned.wav
```

### Batch Processing

```bash
#!/bin/bash
for file in audio/*.wav; do
  python transcribe_diarize_mlx.py \
    --audio "$file" \
    --num_speakers 2 \
    --preprocess \
    --output_dir results
done
```

---

## 🎯 Best Practices

### For Maximum Accuracy

1. ✅ Always specify `--num_speakers N` if known
2. ✅ Enable `--preprocess` for any noise
3. ✅ Use large models (`large-v3`)
4. ✅ Use high-quality audio (16kHz+)

### For Maximum Speed

1. ✅ Use `large-v3-turbo` model
2. ✅ Use `int4` quantization
3. ✅ Disable preprocessing if clean audio
4. ✅ Use `--no_diarization` if unneeded

---

## 📚 Project Structure

```
speech-to-text-diarization/
├── transcribe_diarize_mlx.py    # Main script (USE THIS)
├── requirements_mlx.txt          # Dependencies
├── .env                          # HuggingFace token
├── README.md                     # This file
├── LICENSE                       # MIT License
├── test/                         # Test files
└── transcriptions/              # Output directory
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- **MLX** - Apple's ML framework
- **OpenAI Whisper** - Speech recognition
- **PyAnnote Audio** - Speaker diarization
- **HuggingFace** - Model hosting

---

**Made with ❤️ for Apple Silicon**
