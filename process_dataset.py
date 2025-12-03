"""
Process VoiceAssistant-400K dataset with Dia2-1B to generate Mimi tokens.

This script:
1. Downloads the VoiceAssistant-400K dataset from HuggingFace
2. Loads the Dia2-1B model
3. For each row, generates Mimi tokens from the "answer" text
4. Stores tokens in "answer_mimi" column (replacing "answer_snac")
5. Saves back to parquet files
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from dia2 import Dia2

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm

# Dia2 imports - will be available after installing the dia2 package
try:
    from dia2 import Dia2, GenerationConfig, SamplingConfig
    DIA2_AVAILABLE = True
except ImportError:
    DIA2_AVAILABLE = False
    print("Warning: dia2 package not installed. Install it to enable token generation.")


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


def load_dia2_model(device: str = "cuda", dtype: str = "bfloat16") -> "Dia2":
    """
    Load the Dia2-1B model from HuggingFace.
    
    Args:
        device: Device to load model on ("cuda" or "cpu")
        dtype: Data type for model weights
    
    Returns:
        Loaded Dia2 model
    """
    if not DIA2_AVAILABLE:
        raise ImportError(
            "dia2 package is not installed. "
            "Clone the Dia2 repo and install it: "
            "git clone https://huggingface.co/nari-labs/Dia2-1B && cd Dia2-1B && uv sync"
        )
    
    print(f"Loading Dia2-1B model on {device} with {dtype}...")
    model = Dia2.from_repo("nari-labs/Dia2-1B", device=device, dtype=dtype)
    print("Model loaded successfully!")
    return model


def generate_mimi_tokens(
    model: "Dia2",
    text: str,
    cfg_scale: float = 2.0,
    temperature: float = 0.8,
    top_k: int = 50,
    use_cuda_graph: bool = True,
) -> list[int]:
    """
    Generate Mimi tokens from text using Dia2-1B.
    
    Args:
        model: Loaded Dia2 model
        text: Input text to convert to speech tokens
        cfg_scale: Classifier-free guidance scale
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        use_cuda_graph: Whether to use CUDA graphs for acceleration
    
    Returns:
        List of Mimi audio tokens
    """
    # Prepare text with speaker tag if not present
    if not text.startswith("[S1]") and not text.startswith("[S2]"):
        text = f"[S1] {text}"
    
    config = GenerationConfig(
        cfg_scale=cfg_scale,
        audio=SamplingConfig(temperature=temperature, top_k=top_k),
        use_cuda_graph=use_cuda_graph,
    )
    
    # Generate - we only want the audio tokens, not the waveform
    result = model.generate(text, config=config, verbose=False)
    
    # Extract audio tokens from result
    # The result contains audio_tokens which are the Mimi codec tokens
    audio_tokens = result.audio_tokens
    
    # Convert to list if it's a tensor
    if hasattr(audio_tokens, "tolist"):
        audio_tokens = audio_tokens.tolist()
    
    return audio_tokens


def process_parquet_file(
    input_path: Path,
    output_path: Path,
    model: "Dia2" | None = None,
    batch_size: int = 1,
) -> None:
    """
    Process a single parquet file, generating Mimi tokens for each answer.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to save processed parquet file
        model: Loaded Dia2 model (if None, will skip token generation)
        batch_size: Number of rows to process at once
    """
    print(f"\nProcessing: {input_path.name}")
    
    # Read the parquet file
    table = pq.read_table(input_path)
    df = table.to_pandas()
    
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Check if 'answer' column exists
    if "answer" not in df.columns:
        print(f"  Warning: 'answer' column not found, skipping file")
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
                tokens = generate_mimi_tokens(model, answer_text)
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
        description="Process VoiceAssistant-400K dataset with Dia2-1B"
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
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type for model weights",
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
        "--cfg-scale",
        type=float,
        default=2.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
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
            model = load_dia2_model(device=args.device, dtype=args.dtype)
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
        )
    
    print("\n" + "=" * 50)
    print("Processing complete!")
    print(f"Output files saved to: {output_dir}")


if __name__ == "__main__":
    main()


