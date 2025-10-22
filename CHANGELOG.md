# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-10-22

### Added
- Initial release of speech-to-text processing with speaker diarization
- Support for OpenAI Whisper speech recognition models (tiny, base, small, medium, large)
- Speaker diarization using pyannote.audio
- Batch processing of .wav files
- Multiple output formats (JSON, TXT, CSV)
- Automatic language detection
- Word-level timestamps
- Speaker alignment with transcription segments
- Command-line interface with configurable options
- Comprehensive error handling and logging
- Progress tracking for batch processing
- Summary reports for processed files

### Features
- Process multiple .wav files in batch
- Identify and separate different speakers
- Generate human-readable transcriptions with speaker labels
- Export results in multiple formats
- Configurable Whisper model sizes for speed/accuracy tradeoffs
- Support for 99+ languages through Whisper
- Detailed processing statistics and summaries

### Dependencies
- openai-whisper: Speech-to-text conversion
- pyannote.audio: Speaker diarization
- torch/torchaudio: Deep learning framework
- librosa: Audio processing utilities
- pandas: Data manipulation and analysis
- soundfile: Audio file I/O