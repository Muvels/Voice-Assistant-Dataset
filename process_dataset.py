"""
Process VoiceAssistant-400K dataset with Kyutai TTS to generate Mimi tokens.

This script:
1. Downloads the VoiceAssistant-400K dataset from HuggingFace
2. Loads the Kyutai TTS model (kyutai/tts-1.6b-en_fr)
3. For each row, generates Mimi tokens from the "answer" text
4. Stores tokens in "answer_mimi" column (replacing "answer_snac")
5. Saves back to parquet files
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

if TYPE_CHECKING:
    from moshi.models.tts import TTSModel

# Moshi/Kyutai TTS imports
MOSHI_AVAILABLE = False
MOSHI_IMPORT_ERROR = None

try:
    from moshi.models.loaders import CheckpointInfo
    from moshi.models.tts import DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO, TTSModel
    MOSHI_AVAILABLE = True
except ImportError as e:
    MOSHI_IMPORT_ERROR = str(e)
    print(f"Warning: moshi package not available. Error: {e}")
except Exception as e:
    MOSHI_IMPORT_ERROR = str(e)
    print(f"Warning: Failed to import moshi. Error: {e}")


def download_dataset(output_dir: str, num_files: int | None = None) -> list[Path]:
    """
    Download the VoiceAssistant-400K dataset parquet files.
    
    Args:
        output_dir: Directory to save downloaded files
        num_files: Optional limit on number of files to download (for testing)
    
    Returns:
        List of paths to downloaded parquet files
    """
    print("Downloading VoiceAssistant-400K dataset...")
    
    dataset_path = Path(output_dir) / "VoiceAssistant-400K"
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Download the dataset using huggingface_hub
    local_dir = snapshot_download(
        repo_id="gpt-omni/VoiceAssistant-400K",
        repo_type="dataset",
        local_dir=str(dataset_path),
        allow_patterns=["data/*.parquet"] if num_files is None else None,
    )
    
    # Find all parquet files
    parquet_files = sorted(Path(local_dir).glob("data/*.parquet"))
    
    if num_files is not None:
        parquet_files = parquet_files[:num_files]
    
    print(f"Found {len(parquet_files)} parquet files")
    return parquet_files


def load_tts_model(
    device: str = "cuda",
    n_q: int = 32,
    temp: float = 0.6,
) -> "TTSModel":
    """
    Load the Kyutai TTS model from HuggingFace.
    
    Uses kyutai/tts-1.6b-en_fr model with Mimi codec.
    Model weights are automatically downloaded on first use.
    
    Args:
        device: Device to load model on ("cuda" or "cpu")
        n_q: Number of quantizers (audio tokens per frame, max 32)
        temp: Sampling temperature
    
    Returns:
        Loaded TTSModel
    """
    if not MOSHI_AVAILABLE:
        error_msg = f"Import error: {MOSHI_IMPORT_ERROR}" if MOSHI_IMPORT_ERROR else "Unknown import error"
        raise ImportError(
            f"moshi package is not available. {error_msg}\n"
            "Install it with: pip install 'moshi==0.2.11'"
        )
    
    print(f"Loading Kyutai TTS model on {device}...")
    print(f"  Model: {DEFAULT_DSM_TTS_REPO}")
    print(f"  Voices: {DEFAULT_DSM_TTS_VOICE_REPO}")
    print("(Model weights will be downloaded from HuggingFace if not cached)")
    
    checkpoint_info = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
    tts_model = TTSModel.from_checkpoint_info(
        checkpoint_info,
        n_q=n_q,
        temp=temp,
        device=torch.device(device),
    )
    
    print("Model loaded successfully!")
    return tts_model


def generate_mimi_tokens(
    model: "TTSModel",
    text: str,
    voice: str = "expresso/ex03-ex01_happy_001_channel1_334s.wav",
    cfg_coef: float = 2.0,
) -> list[list[int]]:
    """
    Generate Mimi tokens from text using Kyutai TTS.
    
    Args:
        model: Loaded TTSModel
        text: Input text to convert to speech tokens
        voice: Voice identifier from tts-voices repository
        cfg_coef: CFG coefficient for generation
    
    Returns:
        List of Mimi audio token frames (each frame is a list of tokens)
    """
    # Prepare the script entries
    entries = model.prepare_script([text], padding_between=1)
    
    # Get voice conditioning
    voice_path = model.get_voice_path(voice)
    condition_attributes = model.make_condition_attributes([voice_path], cfg_coef=cfg_coef)
    
    # Collect Mimi tokens during generation
    mimi_tokens = []
    
    def on_frame(frame):
        """Callback to collect Mimi tokens from each generated frame."""
        if (frame != -1).all():
            # frame shape: [batch, n_q+1, time]
            # We take [:, 1:, :] to get the audio tokens (excluding semantic token at index 0)
            tokens = frame[:, 1:, :].cpu().numpy()
            mimi_tokens.append(tokens[0].tolist())  # [n_q, time] for this frame
    
    # Generate with streaming to collect tokens
    all_entries = [entries]
    all_condition_attributes = [condition_attributes]
    
    with model.mimi.streaming(len(all_entries)):
        model.generate(all_entries, all_condition_attributes, on_frame=on_frame)
    
    # Flatten tokens: each entry in mimi_tokens is [n_q, 1] for one timestep
    # We want to return as a list of token lists, one per codebook
    if not mimi_tokens:
        return []
    
    # Stack all frames: [num_frames, n_q, 1] -> [n_q, num_frames]
    stacked = np.concatenate(mimi_tokens, axis=-1)  # [n_q, total_time]
    
    # Return as list of lists (one list per codebook)
    return stacked.tolist()


def process_parquet_file(
    input_path: Path,
    output_path: Path,
    model: "TTSModel" | None = None,
    voice: str = "expresso/ex03-ex01_happy_001_channel1_334s.wav",
) -> None:
    """
    Process a single parquet file, generating Mimi tokens for each answer.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to save processed parquet file
        model: Loaded TTSModel (if None, will skip token generation)
        voice: Voice to use for TTS
    """
    print(f"\nProcessing: {input_path.name}")
    
    # Read the parquet file
    table = pq.read_table(input_path)
    df = table.to_pandas()
    
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Check if 'answer' column exists
    if "answer" not in df.columns:
        print("  Warning: 'answer' column not found, skipping file")
        return
    
    # Generate Mimi tokens for each answer
    mimi_tokens_list = []
    
    if model is not None:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Generating tokens"):
            answer_text = row["answer"]
            
            if pd.isna(answer_text) or not isinstance(answer_text, str) or len(answer_text.strip()) == 0:
                mimi_tokens_list.append(None)
                continue
            
            try:
                tokens = generate_mimi_tokens(model, answer_text, voice=voice)
                mimi_tokens_list.append(tokens)
            except Exception as e:
                print(f"  Error processing row {idx}: {e}")
                mimi_tokens_list.append(None)
    else:
        # If no model, create placeholder tokens
        print("  No model loaded, creating placeholder tokens")
        mimi_tokens_list = [None] * len(df)
    
    # Add the new column
    df["answer_mimi"] = mimi_tokens_list
    
    # Remove answer_snac column if it exists
    if "answer_snac" in df.columns:
        df = df.drop(columns=["answer_snac"])
        print("  Removed 'answer_snac' column")
    
    # Save to output parquet file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process VoiceAssistant-400K dataset with Kyutai TTS"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory to store downloaded data and processed files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for processed output files (default: data-dir/processed)",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=None,
        help="Limit number of parquet files to process (for testing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run model on",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading dataset (use existing files)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process files without generating tokens (for testing)",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="expresso/ex03-ex01_happy_001_channel1_334s.wav",
        help="Voice to use for TTS generation",
    )
    parser.add_argument(
        "--n-q",
        type=int,
        default=32,
        help="Number of quantizers (audio tokens per frame, 1-32)",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.6,
        help="Sampling temperature",
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "processed"
    
    # Download or find dataset files
    if args.skip_download:
        dataset_path = data_dir / "VoiceAssistant-400K" / "data"
        parquet_files = sorted(dataset_path.glob("*.parquet"))
        if args.num_files:
            parquet_files = parquet_files[:args.num_files]
        print(f"Using existing files: {len(parquet_files)} parquet files found")
    else:
        parquet_files = download_dataset(str(data_dir), args.num_files)
    
    if not parquet_files:
        print("No parquet files found!")
        return
    
    # Load model (unless dry run)
    model = None
    if not args.dry_run:
        try:
            model = load_tts_model(
                device=args.device,
                n_q=args.n_q,
                temp=args.temp,
            )
        except ImportError as e:
            print(f"Cannot load model: {e}")
            print("Running in dry-run mode instead...")
    
    # Process each parquet file
    print(f"\nProcessing {len(parquet_files)} files...")
    
    for parquet_file in parquet_files:
        output_file = output_dir / parquet_file.name
        process_parquet_file(
            input_path=parquet_file,
            output_path=output_file,
            model=model,
            voice=args.voice,
        )
    
    print("\n" + "=" * 50)
    print("Processing complete!")
    print(f"Output files saved to: {output_dir}")


if __name__ == "__main__":
    main()
