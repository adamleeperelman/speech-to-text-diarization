#!/bin/bash
# Quick Start Script for Speech-to-Text Transcription
# Usage: ./quickstart.sh your_audio_file.wav

set -e

echo "🎯 Speech-to-Text Transcription - Quick Start"
echo "=============================================="

# Check if audio file provided
if [ $# -eq 0 ]; then
    echo "❌ Error: No audio file provided"
    echo ""
    echo "Usage: ./quickstart.sh your_audio_file.wav [num_speakers]"
    echo ""
    echo "Examples:"
    echo "  ./quickstart.sh interview.wav 2"
    echo "  ./quickstart.sh meeting.mp3 4"
    exit 1
fi

AUDIO_FILE=$1
NUM_SPEAKERS=${2:-2}

# Check if file exists
if [ ! -f "$AUDIO_FILE" ]; then
    echo "❌ Error: File not found: $AUDIO_FILE"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check if dependencies installed
if ! python -c "import mlx_whisper" 2>/dev/null; then
    echo "📥 Installing dependencies (this may take a few minutes)..."
    pip install -q --upgrade pip
    pip install -q -r requirements_mlx.txt
fi

# Check for HuggingFace token
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: No .env file found"
    echo "   Diarization requires a HuggingFace token"
    echo "   Create .env file with: HF_TOKEN=your_token_here"
    echo ""
    read -p "Do you have a HuggingFace token? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter your HuggingFace token: " TOKEN
        echo "HF_TOKEN=$TOKEN" > .env
        echo "✅ Token saved to .env"
    else
        echo "⚠️  Continuing without diarization..."
        echo "   To enable speaker labels later, get a token from:"
        echo "   https://huggingface.co/settings/tokens"
        NO_DIARIZATION="--no_diarization"
    fi
fi

# Run transcription
echo ""
echo "🎤 Starting transcription..."
echo "   Audio file: $AUDIO_FILE"
echo "   Speakers: $NUM_SPEAKERS"
echo "   Model: large-v3-turbo (fast & accurate)"
echo "   Preprocessing: ENABLED (noise reduction + normalization)"
echo ""

python transcribe_diarize_mlx.py \
    --audio "$AUDIO_FILE" \
    --num_speakers "$NUM_SPEAKERS" \
    --preprocess \
    --model large-v3-turbo \
    --formats json srt txt \
    $NO_DIARIZATION

echo ""
echo "✅ Transcription complete!"
echo "📁 Results saved to: transcriptions/"
echo ""
