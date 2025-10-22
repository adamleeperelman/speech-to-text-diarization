# Speech-to-Text with Speaker Diarization

This script processes .wav files from call recordings, performing speech-to-text transcription with speaker diarization (speaker identification).

## Files Included

1. **`speech_to_text_with_diarization.py`** - Main processing script
2. **`requirements.txt`** - Python package dependencies
3. **`callRecords/`** - Directory containing .wav files to process

## Setup

1. **Install Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment:**
   ```bash
   cp .env.template .env
   # Edit .env file with your HuggingFace token
   ```

3. **HuggingFace Setup:**
   - Create a free HuggingFace account at https://huggingface.co/
   - Go to Settings → Access Tokens
   - Create a new token with read permissions
   - Accept user conditions for:
     - https://huggingface.co/pyannote/speaker-diarization-3.1
     - https://huggingface.co/pyannote/speaker-diarization-community-1
   - Add your token to the `.env` file

## Usage

### Processing with Speaker Diarization

**Using environment variables (recommended):**
```bash
# Set up your .env file first, then simply run:
python speech_to_text_with_diarization.py
```

**Using command line arguments:**
```bash
python speech_to_text_with_diarization.py --hf-token YOUR_HUGGINGFACE_TOKEN
```

Additional options:
```bash
# Use a larger Whisper model for better accuracy
python speech_to_text_with_diarization.py --whisper-model medium --hf-token YOUR_TOKEN

# Specify custom input/output directories
python speech_to_text_with_diarization.py --input-dir /path/to/audio --output-dir /path/to/results --hf-token YOUR_TOKEN

# Save in different formats
python speech_to_text_with_diarization.py --formats json txt csv --hf-token YOUR_TOKEN
```

### Available Whisper Models

- `tiny` - Fastest, least accurate
- `base` - Good balance (default)
- `small` - Better accuracy
- `medium` - Even better accuracy
- `large` - Best accuracy, slowest

## Output Files

### For each audio file, you'll get:

1. **JSON file** (`filename_transcription.json`):
   - Complete transcription data
   - Segment timestamps
   - Speaker information (if diarization enabled)
   - Metadata

2. **Text file** (`filename_transcription.txt`):
   - Human-readable transcription
   - Segmented by speaker (if available)
   - Timestamps for each segment

3. **CSV file** (`filename_segments.csv`) - with diarization:
   - Tabular format of all segments
   - Useful for analysis

### Summary files:
- `processing_summary.json` - Overview of all processed files

## Example Output Structure

```
transcriptions/
├── 100630_transcription.json
├── 100630_transcription.txt
├── 101696_transcription.json
├── 101696_transcription.txt
├── ...
└── processing_summary.json
```

## Troubleshooting

1. **Import errors**: Make sure all packages are installed with `pip install -r requirements.txt`

2. **HuggingFace token issues**: 
   - Ensure you've accepted the model license at https://huggingface.co/pyannote/speaker-diarization-3.1
   - Check your token has read permissions

3. **Memory issues**: 
   - Use smaller Whisper models (`tiny` or `base`)
   - Process files in smaller batches

4. **Audio format issues**: 
   - Ensure files are valid .wav format
   - librosa can handle most audio formats automatically

## Performance Notes

- **Processing speed**: ~3-5x real-time (10 minute audio = 30-50 minutes processing)
- Processing time depends on:
  - Audio length
  - Whisper model size
  - Computer performance
  - Number of speakers

## Languages Supported

Whisper automatically detects and supports 99+ languages including:
- English
- Spanish
- French
- German
- Italian
- Portuguese
- Russian
- Japanese
- Chinese
- Korean
- Arabic
- And many more...