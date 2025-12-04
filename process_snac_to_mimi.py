"""
Process VoiceAssistant-400K dataset: SNAC → Audio → Mimi tokens.

This script:
1. Downloads the VoiceAssistant-400K dataset from HuggingFace
2. Decodes SNAC tokens from "answer_snac" column to audio
3. Stores the audio in "answer_audio" column
4. (Optional) Encodes audio to Mimi tokens using Kyutai's Mimi codec
5. Saves back to parquet files
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

# SNAC codec imports
SNAC_AVAILABLE = False
SNAC_IMPORT_ERROR = None

try:
    from snac import SNAC
    SNAC_AVAILABLE = True
except ImportError as e:
    SNAC_IMPORT_ERROR = str(e)
    print(f"Warning: snac package not available. Error: {e}")
except Exception as e:
    SNAC_IMPORT_ERROR = str(e)
    print(f"Warning: Failed to import snac. Error: {e}")

# Moshi/Mimi codec imports for encoding
MIMI_AVAILABLE = False
MIMI_IMPORT_ERROR = None

try:
    from moshi.models.loaders import CheckpointInfo
    from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel
    MIMI_AVAILABLE = True
except ImportError as e:
    MIMI_IMPORT_ERROR = str(e)
except Exception as e:
    MIMI_IMPORT_ERROR = str(e)


def download_dataset(output_dir: str, num_files: int | None = None) -> list[Path]:
    """
    Download the VoiceAssistant-400K dataset parquet files.
    """
    print("Downloading VoiceAssistant-400K dataset...")
    
    dataset_path = Path(output_dir) / "VoiceAssistant-400K"
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    local_dir = snapshot_download(
        repo_id="gpt-omni/VoiceAssistant-400K",
        repo_type="dataset",
        local_dir=str(dataset_path),
        allow_patterns=["data/*.parquet"] if num_files is None else None,
    )
    
    parquet_files = sorted(Path(local_dir).glob("data/*.parquet"))
    
    if num_files is not None:
        parquet_files = parquet_files[:num_files]
    
    print(f"Found {len(parquet_files)} parquet files")
    return parquet_files


def load_snac_model(device: str = "cuda") -> "SNAC":
    """
    Load the SNAC model for decoding tokens to audio.
    
    SNAC (Multi-Scale Neural Audio Codec) is used in VoiceAssistant-400K.
    """
    if not SNAC_AVAILABLE:
        error_msg = f"Import error: {SNAC_IMPORT_ERROR}" if SNAC_IMPORT_ERROR else "Unknown error"
        raise ImportError(f"snac package not available. {error_msg}")
    
    print(f"Loading SNAC model on {device}...")
    
    # SNAC models: snac_24khz, snac_32khz, snac_44khz
    # VoiceAssistant-400K likely uses 24kHz
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    model = model.to(device)
    model.eval()
    
    print("SNAC model loaded successfully!")
    return model


def parse_snac_tokens(snac_tokens) -> list:
    """
    Parse SNAC tokens from various storage formats.
    
    The tokens might be stored as:
    - A list of lists (already parsed)
    - A JSON string
    - A numpy array
    - A string representation of a list
    """
    # If it's a string, try to parse it
    if isinstance(snac_tokens, str):
        # Try JSON first
        try:
            snac_tokens = json.loads(snac_tokens)
        except json.JSONDecodeError:
            # Try ast.literal_eval for Python list syntax
            try:
                snac_tokens = ast.literal_eval(snac_tokens)
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"Cannot parse SNAC tokens string: {e}")
    
    # If it's a numpy array, convert to list
    if isinstance(snac_tokens, np.ndarray):
        snac_tokens = snac_tokens.tolist()
    
    return snac_tokens


def decode_snac_to_audio(
    snac_model: "SNAC",
    snac_tokens,
    device: str = "cuda",
) -> np.ndarray:
    """
    Decode SNAC tokens to audio waveform.
    
    Args:
        snac_model: Loaded SNAC model
        snac_tokens: SNAC token data (can be list, string, or numpy array)
        device: Device to run on
    
    Returns:
        Audio waveform as numpy array
    """
    # Parse tokens if they're stored as strings
    snac_tokens = parse_snac_tokens(snac_tokens)
    
    # SNAC 24kHz uses 3 hierarchical layers with different frame rates
    # The tokens should be a list of 3 lists with different lengths
    # Layer 0: lowest resolution (longest temporal)
    # Layer 1: middle resolution
    # Layer 2: highest resolution (shortest temporal)
    
    if not isinstance(snac_tokens, list) or len(snac_tokens) == 0:
        raise ValueError(f"Invalid SNAC tokens format: {type(snac_tokens)}")
    
    # Check if it's already structured as layers
    if isinstance(snac_tokens[0], list):
        # Already structured as layers - convert each to tensor
        codes = []
        for layer in snac_tokens:
            layer_tensor = torch.tensor(layer, dtype=torch.long, device=device)
            # Add batch dimension: [seq_len] -> [1, seq_len]
            codes.append(layer_tensor.unsqueeze(0))
    else:
        # Flat list - this might be interleaved tokens
        # SNAC uses 3 layers with ratios roughly 1:2:4
        # Try to decode as a single layer first
        codes_tensor = torch.tensor(snac_tokens, dtype=torch.long, device=device)
        
        if codes_tensor.dim() == 1:
            # Single flat list - wrap as single layer
            codes = [codes_tensor.unsqueeze(0)]
        elif codes_tensor.dim() == 2:
            # 2D array - each row is a layer
            codes = [codes_tensor[i:i+1, :] for i in range(codes_tensor.shape[0])]
        else:
            raise ValueError(f"Unexpected SNAC tensor shape: {codes_tensor.shape}")
    
    with torch.no_grad():
        audio = snac_model.decode(codes)
    
    # Convert to numpy and squeeze
    audio_np = audio.cpu().numpy().squeeze()
    
    return audio_np


def inspect_snac_format(df: pd.DataFrame) -> dict:
    """
    Inspect the format of answer_snac column to understand the data structure.
    """
    if "answer_snac" not in df.columns:
        return {"error": "No answer_snac column found"}
    
    # Get first non-null entry
    sample = None
    for idx, val in enumerate(df["answer_snac"]):
        if val is not None and (not isinstance(val, float) or not pd.isna(val)):
            sample = val
            sample_idx = idx
            break
    
    if sample is None:
        return {"error": "No non-null answer_snac values found"}
    
    info = {
        "sample_index": sample_idx,
        "raw_type": type(sample).__name__,
    }
    
    # If it's a string, try to parse it
    if isinstance(sample, str):
        info["string_length"] = len(sample)
        info["string_preview"] = sample[:200] + "..." if len(sample) > 200 else sample
        
        # Try to parse
        try:
            parsed = parse_snac_tokens(sample)
            info["parsed_type"] = type(parsed).__name__
            if isinstance(parsed, list):
                info["parsed_length"] = len(parsed)
                if len(parsed) > 0:
                    info["first_elem_type"] = type(parsed[0]).__name__
                    if isinstance(parsed[0], list):
                        info["layer_lengths"] = [len(layer) for layer in parsed]
                    else:
                        info["sample_values"] = parsed[:20]
        except Exception as e:
            info["parse_error"] = str(e)
    
    elif isinstance(sample, (list, np.ndarray)):
        info["length"] = len(sample)
        if len(sample) > 0:
            first_elem = sample[0]
            info["first_element_type"] = type(first_elem).__name__
            if isinstance(first_elem, (list, np.ndarray)):
                info["first_element_length"] = len(first_elem)
                info["nested"] = True
                # Check all layer lengths
                info["layer_lengths"] = [len(layer) for layer in sample[:10]]
            else:
                info["nested"] = False
                info["sample_values"] = list(sample[:20])
    elif isinstance(sample, bytes):
        info["byte_length"] = len(sample)
    else:
        info["value"] = str(sample)[:100]
    
    return info


def process_parquet_file(
    input_path: Path,
    output_path: Path,
    snac_model: "SNAC" | None = None,
    save_audio_files: bool = False,
    audio_output_dir: Path | None = None,
    sample_rate: int = 24000,
) -> None:
    """
    Process a single parquet file, decoding SNAC tokens to audio.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to save processed parquet file
        snac_model: Loaded SNAC model
        save_audio_files: Whether to save audio as separate .wav files
        audio_output_dir: Directory for audio files (if save_audio_files=True)
        sample_rate: Sample rate for SNAC model (24000 for snac_24khz)
    """
    print(f"\nProcessing: {input_path.name}")
    
    # Read the parquet file
    table = pq.read_table(input_path)
    df = table.to_pandas()
    
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Inspect SNAC format
    snac_info = inspect_snac_format(df)
    print(f"  SNAC format: {snac_info}")
    
    if "answer_snac" not in df.columns:
        print("  Warning: 'answer_snac' column not found, skipping file")
        return
    
    # Decode SNAC tokens to audio
    audio_list = []
    
    if snac_model is not None:
        if save_audio_files and audio_output_dir:
            audio_output_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Decoding SNAC"):
            snac_tokens = row["answer_snac"]
            
            if snac_tokens is None or (isinstance(snac_tokens, float) and pd.isna(snac_tokens)):
                audio_list.append(None)
                continue
            
            try:
                audio = decode_snac_to_audio(snac_model, snac_tokens)
                
                # Store as bytes (more compact for parquet)
                audio_bytes = audio.astype(np.float32).tobytes()
                audio_list.append(audio_bytes)
                
                # Optionally save as wav file
                if save_audio_files and audio_output_dir:
                    wav_path = audio_output_dir / f"{input_path.stem}_{idx}.wav"
                    sf.write(wav_path, audio, sample_rate)
                    
            except Exception as e:
                print(f"  Error processing row {idx}: {e}")
                audio_list.append(None)
    else:
        print("  No SNAC model loaded, skipping audio decoding")
        audio_list = [None] * len(df)
    
    # Add the audio column
    df["answer_audio"] = audio_list
    
    # Save to output parquet file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process VoiceAssistant-400K: SNAC to Audio conversion"
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
        help="Inspect data format without processing",
    )
    parser.add_argument(
        "--save-audio-files",
        action="store_true",
        help="Also save audio as separate .wav files",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Directory for audio files (default: data-dir/audio)",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only inspect SNAC format without processing",
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "processed_audio"
    audio_dir = Path(args.audio_dir) if args.audio_dir else data_dir / "audio"
    
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
    
    # Inspect only mode
    if args.inspect_only:
        print("\n=== Inspecting SNAC data format ===")
        for parquet_file in parquet_files[:1]:  # Just first file
            table = pq.read_table(parquet_file)
            df = table.to_pandas()
            print(f"\nFile: {parquet_file.name}")
            print(f"Columns: {list(df.columns)}")
            print(f"Shape: {df.shape}")
            
            info = inspect_snac_format(df)
            print(f"SNAC format info: {info}")
            
            # Print first few rows of answer_snac
            if "answer_snac" in df.columns:
                print(f"\nFirst 3 answer_snac entries:")
                for i in range(min(3, len(df))):
                    val = df["answer_snac"].iloc[i]
                    if val is not None:
                        print(f"  [{i}]: type={type(val).__name__}, ", end="")
                        if hasattr(val, "__len__"):
                            print(f"len={len(val)}")
                        else:
                            print(f"value={val}")
        return
    
    # Load SNAC model (unless dry run)
    snac_model = None
    if not args.dry_run:
        try:
            snac_model = load_snac_model(device=args.device)
        except ImportError as e:
            print(f"Cannot load SNAC model: {e}")
            print("Running in dry-run mode instead...")
    
    # Process each parquet file
    print(f"\nProcessing {len(parquet_files)} files...")
    
    for parquet_file in parquet_files:
        output_file = output_dir / parquet_file.name
        process_parquet_file(
            input_path=parquet_file,
            output_path=output_file,
            snac_model=snac_model,
            save_audio_files=args.save_audio_files,
            audio_output_dir=audio_dir if args.save_audio_files else None,
        )
    
    print("\n" + "=" * 50)
    print("Processing complete!")
    print(f"Output files saved to: {output_dir}")
    if args.save_audio_files:
        print(f"Audio files saved to: {audio_dir}")


if __name__ == "__main__":
    main()

