# Omni-Qwen: VoiceAssistant Dataset Processor

Process the [VoiceAssistant-400K](https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K) dataset to convert between audio codecs (SNAC → Audio → Mimi).

## Overview

Two processing pipelines:

### 1. SNAC → Audio (`process_snac_to_mimi.py`)
- Decodes `answer_snac` tokens to audio waveforms
- Stores audio in `answer_audio` column
- Uses [SNAC codec](https://github.com/hubertsiuzdak/snac) (24kHz)

### 2. Text → Mimi (`process_dataset.py`)
- Generates speech from `answer` text using [Kyutai TTS](https://huggingface.co/kyutai/tts-1.6b-en_fr)
- Stores Mimi tokens in `answer_mimi` column (replacing `answer_snac`)

## Requirements

- Python 3.12+
- CUDA (for GPU acceleration)
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
uv sync
```

This installs:
- `snac` - SNAC audio codec for decoding
- `moshi==0.2.11` - Kyutai's TTS with Mimi codec
- `torch`, `soundfile`, `pandas`, `pyarrow` - Processing tools

## Usage

### Step 1: SNAC to Audio

First, decode the existing SNAC tokens to audio:

```bash
# Inspect the SNAC data format first
uv run python process_snac_to_mimi.py --data-dir ./data --num-files 1 --inspect-only

# Process SNAC → Audio (stores in answer_audio column)
uv run python process_snac_to_mimi.py --data-dir ./data --num-files 1

# Also save audio as .wav files
uv run python process_snac_to_mimi.py --data-dir ./data --num-files 1 --save-audio-files
```

### Step 2: Text to Mimi (alternative)

Generate Mimi tokens from text using Kyutai TTS:

```bash
uv run python process_dataset.py --data-dir ./data --num-files 1
```

### All Options

```bash
# SNAC → Audio
uv run python process_snac_to_mimi.py --help

# Text → Mimi  
uv run python process_dataset.py --help
```

Common options:
- `--data-dir`: Directory for data (default: `./data`)
- `--output-dir`: Output directory (default: `data-dir/processed`)
- `--num-files`: Limit files to process
- `--device`: `cuda` or `cpu`
- `--skip-download`: Use existing files
- `--dry-run`: Test without processing

## Output

### From `process_snac_to_mimi.py`:
- `answer_audio`: Audio waveform as bytes (float32)
- Optional `.wav` files in `--audio-dir`

### From `process_dataset.py`:
- `answer_mimi`: Mimi codec tokens (32 codebook sequences)
- Removes `answer_snac` column

## Dataset Information

[VoiceAssistant-400K](https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K):
- 325 parquet files (~219GB total)
- ~470K voice assistant samples
- Columns: `split_name`, `index`, `round`, `question`, `question_audio`, `answer`, `answer_snac`

## Models

### SNAC (hubertsiuzdak/snac_24khz)
- Multi-Scale Neural Audio Codec
- 24kHz sample rate
- Used in the original dataset

### Kyutai TTS (kyutai/tts-1.6b-en_fr)
- 1.8B parameter streaming TTS
- Uses Mimi codec at 12.5 Hz
- 32 audio tokens per frame
- English and French

## License

- Code: Apache 2.0
- SNAC: MIT
- Kyutai TTS: CC-BY 4.0
- VoiceAssistant-400K: Apache 2.0
