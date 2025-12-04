# Omni-Qwen: VoiceAssistant Dataset Processor

Convert the [VoiceAssistant-400K](https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K) dataset between audio codecs (SNAC → Audio → Mimi tokens).

## Overview

**Main Script: `convert_snac_to_mimi.py`**

Converts each parquet file:
1. Decodes `answer_snac` → audio waveform (SNAC 24kHz codec)
2. Encodes audio → `answer_mimi` tokens (Mimi codec from Kyutai)
3. Saves both `answer_audio` and `answer_mimi` columns
4. **Resumable** - saves checkpoints to continue after interruption

## Requirements

- Python 3.12+
- CUDA GPU (recommended)
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
uv sync
```

Installs:
- `snac` - SNAC audio codec (24kHz)
- `moshi==0.2.11` - Kyutai TTS with Mimi codec
- `torch`, `soundfile`, `pandas`, `pyarrow`

## Usage

### Basic Conversion

```bash
# Process all files
uv run python convert_snac_to_mimi.py --data-dir ./data

# Test with first file only
uv run python convert_snac_to_mimi.py --data-dir ./data --num-files 1

# Skip dataset download (use existing files)
uv run python convert_snac_to_mimi.py --data-dir ./data --skip-download
```

### Resuming After Interruption

```bash
# If the script crashes or is interrupted, resume with:
uv run python convert_snac_to_mimi.py --data-dir ./data --resume

# Checkpoint is saved every 100 rows (configurable)
uv run python convert_snac_to_mimi.py --data-dir ./data --save-interval 50
```

### All Options

```bash
uv run python convert_snac_to_mimi.py --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | `./data` | Data directory |
| `--output-dir` | `data-dir/converted` | Output directory |
| `--num-files` | all | Limit files to process |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--skip-download` | false | Use existing files |
| `--resume` | false | Resume from checkpoint |
| `--checkpoint-file` | `conversion_checkpoint.pkl` | Checkpoint filename |
| `--save-interval` | 100 | Checkpoint every N rows |
| `--dry-run` | false | Test without processing |

## Output Format

Each processed parquet file contains:

| Column | Type | Description |
|--------|------|-------------|
| `answer_audio` | bytes | Audio waveform (float32, 24kHz) |
| `answer_mimi` | list[list[int]] | Mimi tokens (32 codebooks × time) |
| *(original columns)* | ... | All original data preserved |

### Reading Audio from Parquet

```python
import numpy as np
import pandas as pd

df = pd.read_parquet("data/converted/train-00000-of-00325.parquet")

# Convert bytes back to audio
audio_bytes = df["answer_audio"].iloc[0]
audio = np.frombuffer(audio_bytes, dtype=np.float32)

# Play or save audio (24kHz sample rate)
import soundfile as sf
sf.write("output.wav", audio, 24000)
```

### Reading Mimi Tokens

```python
# Mimi tokens: list of 32 lists (one per codebook)
mimi_tokens = df["answer_mimi"].iloc[0]
print(f"Codebooks: {len(mimi_tokens)}, Time steps: {len(mimi_tokens[0])}")
```

## Resumability

The script saves checkpoints to `conversion_checkpoint.pkl`:
- Tracks completed files
- Tracks current file and row
- Saves partial results periodically

If interrupted, use `--resume` to continue from where it stopped.

## Dataset Info

[VoiceAssistant-400K](https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K):
- 325 parquet files (~219GB)
- ~470K samples
- Original columns: `split_name`, `index`, `round`, `question`, `question_audio`, `answer`, `answer_snac`

## Models Used

| Model | Description |
|-------|-------------|
| [hubertsiuzdak/snac_24khz](https://huggingface.co/hubertsiuzdak/snac_24khz) | SNAC decoder (24kHz) |
| [kyutai/tts-1.6b-en_fr](https://huggingface.co/kyutai/tts-1.6b-en_fr) | Mimi encoder (from Kyutai TTS) |

## Other Scripts

- `process_snac_to_mimi.py` - SNAC → Audio only
- `process_dataset.py` - Text → Mimi (TTS generation)

## License

- Code: Apache 2.0
- SNAC: MIT
- Kyutai TTS/Mimi: CC-BY 4.0
- VoiceAssistant-400K: Apache 2.0
