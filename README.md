# Omni-Qwen: VoiceAssistant Dataset Processor

Process the [VoiceAssistant-400K](https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K) dataset with [Kyutai TTS](https://huggingface.co/kyutai/tts-1.6b-en_fr) to generate Mimi audio tokens.

## Overview

This tool:
1. Downloads the VoiceAssistant-400K dataset (parquet files)
2. Loads the Kyutai TTS model (1.6B parameters, uses Mimi codec)
3. For each row, generates Mimi tokens from the "answer" text column
4. Stores tokens in a new "answer_mimi" column (replacing "answer_snac")
5. Saves the processed data back to parquet files

## Requirements

- Python 3.12+
- CUDA (for GPU acceleration)
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

## Installation

```bash
uv sync
```

This will install all dependencies including:
- `moshi==0.2.11` - Kyutai's TTS package with Mimi codec
- `torch` - PyTorch (CUDA version on Linux)
- `datasets`, `pyarrow`, `pandas` - Data processing

## Usage

### Full processing

Process all parquet files from the dataset:

```bash
uv run python process_dataset.py --data-dir ./data
```

### Testing with a few files

Process only the first N files (useful for testing):

```bash
uv run python process_dataset.py --data-dir ./data --num-files 1
```

### Dry run (no token generation)

Test the pipeline without loading the model:

```bash
uv run python process_dataset.py --data-dir ./data --dry-run --num-files 1
```

### Use existing downloaded files

Skip the download step if you already have the dataset:

```bash
uv run python process_dataset.py --data-dir ./data --skip-download
```

### All options

```bash
uv run python process_dataset.py --help
```

Options:
- `--data-dir`: Directory to store downloaded data (default: `./data`)
- `--output-dir`: Directory for processed files (default: `data-dir/processed`)
- `--num-files`: Limit number of files to process
- `--device`: Device to run model on (`cuda` or `cpu`)
- `--skip-download`: Use existing downloaded files
- `--dry-run`: Process without generating tokens
- `--voice`: Voice from [tts-voices repo](https://huggingface.co/kyutai/tts-voices) (default: `expresso/ex03-ex01_happy_001_channel1_334s.wav`)
- `--n-q`: Number of quantizers/codebooks (1-32, default: 32)
- `--temp`: Sampling temperature (default: 0.6)

## Output

Processed parquet files are saved to `<data-dir>/processed/` with the same filenames as the originals.

Each file will have:
- All original columns preserved
- New `answer_mimi` column containing Mimi audio tokens (list of 32 codebook sequences)
- `answer_snac` column removed (if it existed)

## Model Information

The Kyutai TTS model ([kyutai/tts-1.6b-en_fr](https://huggingface.co/kyutai/tts-1.6b-en_fr)):
- 1.8B parameters (1B backbone + 600M depth transformer)
- Streaming text-to-speech
- Uses Mimi codec at 12.5 Hz frame rate
- 32 audio tokens per frame
- Supports English and French
- Voice conditioning via pre-computed embeddings

## Dataset Information

The VoiceAssistant-400K dataset contains:
- 325 parquet files (~219GB total)
- ~400K voice assistant conversation samples
- Columns: `split_name`, `index`, `round`, `question`, `question_audio`, `answer`, `answer_snac`

## License

- Code: Apache 2.0
- Kyutai TTS model: CC-BY 4.0
- VoiceAssistant-400K dataset: Apache 2.0
