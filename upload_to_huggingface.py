"""
Upload converted parquet files to HuggingFace Hub.

Usage:
    # First login to HuggingFace
    huggingface-cli login
    
    # Then upload
    uv run python upload_to_huggingface.py --repo-id YOUR_USERNAME/voice-assistant-mimi --data-dir ./data/converted
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Audio, Dataset
from huggingface_hub import HfApi, create_repo


DATASET_CARD = """---
license: apache-2.0
task_categories:
  - text-to-speech
  - automatic-speech-recognition
language:
  - en
tags:
  - audio
  - speech
  - mimi
  - snac
  - voice-assistant
pretty_name: VoiceAssistant-400K with Mimi Tokens
size_categories:
  - 100K<n<1M
---

# VoiceAssistant-400K with Mimi Tokens

This dataset is a processed version of [gpt-omni/VoiceAssistant-400K](https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K) with audio codec conversions.

## Processing

Each sample has been processed to add:
- **answer_audio**: Decoded audio waveform from SNAC tokens (24kHz `Audio` feature)
- **answer_mimi**: Re-encoded audio using [Kyutai's Mimi codec](https://huggingface.co/kyutai/tts-1.6b-en_fr) (32 codebooks)

## Columns

| Column | Type | Description |
|--------|------|-------------|
| `split_name` | string | Original split name |
| `index` | int | Sample index |
| `round` | int | Conversation round |
| `question` | string | User question text |
| `question_audio` | Audio | Question audio (original sample rate) |
| `answer` | string | Assistant answer text |
| `answer_snac` | string | Original SNAC tokens |
| `answer_audio` | Audio | Decoded audio (WAV, 24kHz) - playable in viewer |
| `answer_mimi` | list[list[int]] | Mimi tokens (32 codebooks × time) |

## Usage

```python
from datasets import load_dataset

# Load dataset
ds = load_dataset("YOUR_USERNAME/voice-assistant-mimi", split="train")

# Access a sample
sample = ds[0]

# Audio is stored as an Audio feature - can be played directly in Data Studio
answer_audio = sample["answer_audio"]
audio_array = answer_audio["array"]        # numpy array, shape (num_samples,)
sr = answer_audio["sampling_rate"]         # 24000

# Access Mimi tokens
mimi_tokens = sample["answer_mimi"]  # [32, time_steps]
```

## Models Used

- **SNAC Decoder**: [hubertsiuzdak/snac_24khz](https://huggingface.co/hubertsiuzdak/snac_24khz)
- **Mimi Encoder**: [kyutai/tts-1.6b-en_fr](https://huggingface.co/kyutai/tts-1.6b-en_fr)

## License

Apache 2.0 (same as original dataset)
"""


def upload_dataset(
    repo_id: str,
    data_dir: Path,
    private: bool = False,
    num_files: int | None = None,
):
    """
    Upload converted parquet files to HuggingFace Hub as a proper `datasets.Dataset`
    with all `_audio` columns cast to the `Audio` feature.
    
    This makes audio columns playable in the HuggingFace Data Studio viewer.
    
    Args:
        repo_id: HuggingFace repo ID (username/repo-name)
        data_dir: Directory containing converted parquet files
        private: Whether to make the repo private
        num_files: Limit number of files to upload (for testing)
    """
    api = HfApi()
    
    # Find parquet files
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if num_files:
        parquet_files = parquet_files[:num_files]
    
    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet files to include in dataset")
    
    # Build a single Dataset from all parquet shards
    print("\nLoading parquet files into a HuggingFace Dataset...")
    ds = Dataset.from_parquet([str(pf) for pf in parquet_files])
    print(f"Dataset loaded with {len(ds)} rows and columns: {ds.column_names}")
    
    # Cast all *_audio columns to Audio features (option A)
    audio_columns = [c for c in ds.column_names if c.endswith("_audio")]
    if not audio_columns:
        print("Warning: no columns ending with '_audio' found to cast to Audio feature")
    else:
        print(f"Casting columns to Audio feature: {audio_columns}")
        for col in audio_columns:
            # `answer_audio` is known to be 24 kHz; others keep default behavior
            if col == "answer_audio":
                ds = ds.cast_column(col, Audio(sampling_rate=24000))
            else:
                ds = ds.cast_column(col, Audio())
    
    # Create repository if it doesn't exist
    print(f"\nCreating/accessing repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        print(f"Repository ready: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Error creating repo: {e}")
        return
    
    # Push dataset (this will upload parquet shards + dataset_infos)
    print("\nPushing dataset with Audio features to HuggingFace Hub...")
    ds.push_to_hub(repo_id, private=private)
    
    # Upload README (dataset card)
    print("\nUploading dataset card (README.md)...")
    readme_content = DATASET_CARD.replace("YOUR_USERNAME/voice-assistant-mimi", repo_id)
    
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    
    print("\n" + "=" * 50)
    print("Upload complete!")
    print("=" * 50)
    print(f"Dataset URL: https://huggingface.co/datasets/{repo_id}")
    print(f"Data Studio: https://huggingface.co/datasets/{repo_id}/viewer")


def main():
    parser = argparse.ArgumentParser(
        description="Upload converted dataset to HuggingFace Hub"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., username/voice-assistant-mimi)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/converted",
        help="Directory containing converted parquet files",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=None,
        help="Limit number of files to upload (for testing)",
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    upload_dataset(
        repo_id=args.repo_id,
        data_dir=data_dir,
        private=args.private,
        num_files=args.num_files,
    )


if __name__ == "__main__":
    main()

