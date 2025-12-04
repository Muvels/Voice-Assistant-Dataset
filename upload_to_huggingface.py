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

from huggingface_hub import HfApi, create_repo, upload_file, upload_folder


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
- **answer_audio**: Decoded audio waveform from SNAC tokens (float32 bytes, 24kHz)
- **answer_mimi**: Re-encoded audio using [Kyutai's Mimi codec](https://huggingface.co/kyutai/tts-1.6b-en_fr) (32 codebooks)

## Columns

| Column | Type | Description |
|--------|------|-------------|
| `split_name` | string | Original split name |
| `index` | int | Sample index |
| `round` | int | Conversation round |
| `question` | string | User question text |
| `question_audio` | bytes | Question audio |
| `answer` | string | Assistant answer text |
| `answer_snac` | string | Original SNAC tokens |
| `answer_audio` | bytes | Decoded audio (float32, 24kHz) |
| `answer_mimi` | list[list[int]] | Mimi tokens (32 codebooks × time) |

## Usage

```python
import numpy as np
import pandas as pd
from datasets import load_dataset

# Load dataset
ds = load_dataset("YOUR_USERNAME/voice-assistant-mimi", split="train")

# Access a sample
sample = ds[0]

# Convert audio bytes back to waveform
audio_bytes = sample["answer_audio"]
audio = np.frombuffer(audio_bytes, dtype=np.float32)
# Sample rate: 24000 Hz

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
    Upload converted parquet files to HuggingFace Hub.
    
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
    
    print(f"Found {len(parquet_files)} parquet files to upload")
    
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
    
    # Upload README
    print("\nUploading dataset card (README.md)...")
    readme_content = DATASET_CARD.replace("YOUR_USERNAME/voice-assistant-mimi", repo_id)
    
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    
    # Upload parquet files to data/ folder
    print(f"\nUploading {len(parquet_files)} parquet files...")
    
    for i, pf in enumerate(parquet_files):
        print(f"  [{i+1}/{len(parquet_files)}] Uploading {pf.name}...")
        api.upload_file(
            path_or_fileobj=str(pf),
            path_in_repo=f"data/{pf.name}",
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

