# Omni-Qwen: VoiceAssistant Dataset Processor

Process the [VoiceAssistant-400K](https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K) dataset with [Dia2-1B](https://huggingface.co/nari-labs/Dia2-1B) to generate Mimi audio tokens.

## Overview

This tool:
1. Downloads the VoiceAssistant-400K dataset (parquet files)
2. Loads the Dia2-1B TTS model
3. For each row, generates Mimi tokens from the "answer" text column
4. Stores tokens in a new "answer_mimi" column (replacing "answer_snac")
5. Saves the processed data back to parquet files

## Requirements

- Python 3.12+
- CUDA 12.8+ (for GPU acceleration)
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

## Installation

### 1. Install base dependencies

```bash
uv sync
```

### 2. Install Dia2-1B

The Dia2 model needs to be installed from the HuggingFace repository:

```bash
# Clone the Dia2 repository
git clone https://huggingface.co/nari-labs/Dia2-1B
cd Dia2-1B
uv sync
cd ..

# Or install via pip if available
pip install dia2
```

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
- `--dtype`: Data type for model weights (`bfloat16`, `float16`, `float32`)
- `--skip-download`: Use existing downloaded files
- `--dry-run`: Process without generating tokens
- `--cfg-scale`: Classifier-free guidance scale (default: 2.0)
- `--temperature`: Sampling temperature (default: 0.8)

## Output

Processed parquet files are saved to `<data-dir>/processed/` with the same filenames as the originals.

Each file will have:
- All original columns preserved
- New `answer_mimi` column containing Mimi audio tokens
- `answer_snac` column removed (if it existed)

## Dataset Information

The VoiceAssistant-400K dataset contains:
- 325 parquet files
- ~400K voice assistant conversation samples
- Audio and text data for training speech models

## License

Apache 2.0

