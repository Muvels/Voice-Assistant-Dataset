"""
Convert VoiceAssistant-400K dataset: SNAC → Audio → Mimi tokens.

This script:
1. Takes each parquet file from the dataset
2. Decodes answer_snac tokens to audio (using SNAC codec)
3. Encodes audio to Mimi tokens (using Kyutai's Mimi codec)
4. Saves both answer_audio and answer_mimi columns
5. Supports resumable processing with checkpoints

Usage:
    uv run python convert_snac_to_mimi.py --data-dir ./data --num-files 1
    
Resume after interruption:
    uv run python convert_snac_to_mimi.py --data-dir ./data --resume
"""

from __future__ import annotations

import argparse
import ast
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

if TYPE_CHECKING:
    from snac import SNAC

# ============================================================================
# Import checks
# ============================================================================

SNAC_AVAILABLE = False
SNAC_IMPORT_ERROR = None
try:
    from snac import SNAC
    SNAC_AVAILABLE = True
except ImportError as e:
    SNAC_IMPORT_ERROR = str(e)

MIMI_AVAILABLE = False
MIMI_IMPORT_ERROR = None
try:
    from moshi.models.loaders import CheckpointInfo
    from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel
    MIMI_AVAILABLE = True
except ImportError as e:
    MIMI_IMPORT_ERROR = str(e)


# ============================================================================
# Checkpoint management for resumability
# ============================================================================

@dataclass
class ProcessingCheckpoint:
    """Tracks processing progress for resumability."""
    completed_files: list[str] = field(default_factory=list)
    current_file: str | None = None
    current_row: int = 0
    partial_results: dict = field(default_factory=dict)  # row_idx -> (audio, mimi)
    started_at: str = ""
    last_updated: str = ""
    
    def save(self, path: Path):
        """Save checkpoint to disk."""
        self.last_updated = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: Path) -> "ProcessingCheckpoint":
        """Load checkpoint from disk."""
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return cls(started_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    
    def mark_file_complete(self, filename: str):
        """Mark a file as fully processed."""
        if filename not in self.completed_files:
            self.completed_files.append(filename)
        self.current_file = None
        self.current_row = 0
        self.partial_results = {}
    
    def is_file_complete(self, filename: str) -> bool:
        """Check if a file was already processed."""
        return filename in self.completed_files


# ============================================================================
# SNAC and Mimi model loading
# ============================================================================

def load_snac_model(device: str = "cuda") -> "SNAC":
    """Load SNAC model for decoding tokens to audio."""
    if not SNAC_AVAILABLE:
        raise ImportError(f"snac not available: {SNAC_IMPORT_ERROR}")
    
    print(f"Loading SNAC model (hubertsiuzdak/snac_24khz) on {device}...")
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    model = model.to(device)
    model.eval()
    print("SNAC model loaded!")
    return model


def load_mimi_model(device: str = "cuda"):
    """Load Mimi codec from Kyutai TTS for encoding audio to tokens."""
    if not MIMI_AVAILABLE:
        raise ImportError(f"moshi not available: {MIMI_IMPORT_ERROR}")
    
    print(f"Loading Mimi codec on {device}...")
    print("(This will download model weights from HuggingFace if not cached)")
    
    checkpoint_info = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
    tts_model = TTSModel.from_checkpoint_info(
        checkpoint_info,
        n_q=32,
        temp=0.6,
        device=torch.device(device),
    )
    
    # We only need the mimi codec from the TTS model
    mimi = tts_model.mimi
    print("Mimi codec loaded!")
    return mimi


# ============================================================================
# Token parsing and conversion functions
# ============================================================================

def parse_snac_tokens(snac_tokens) -> list:
    """
    Parse SNAC tokens from various storage formats.
    
    The VoiceAssistant-400K dataset stores SNAC tokens as a string with format:
    "# t0_c0 t0_c1 t0_c2 t0_c3 t0_c4 t0_c5 t0_c6 # t1_c0 t1_c1 ..."
    
    Each # separated group has 7 tokens representing one timestep across 7 codebooks.
    We need to transpose this to get 7 layers (one per codebook).
    
    SNAC 24kHz uses 3 hierarchical codebooks but mini-omni flattens to 7 streams:
    - Codebook 0: 1 token per frame
    - Codebook 1: 2 tokens per frame  
    - Codebook 2: 4 tokens per frame
    Total: 1 + 2 + 4 = 7 tokens per frame
    """
    if isinstance(snac_tokens, str):
        # Check if it's the VoiceAssistant-400K format: "# tokens # tokens # ..."
        if snac_tokens.startswith('#') or ' # ' in snac_tokens:
            # Split by # and parse each timestep
            timesteps = []
            parts = snac_tokens.split('#')
            for part in parts:
                part = part.strip()
                if part:
                    # Parse space-separated integers for this timestep
                    tokens = [int(t) for t in part.split() if t.strip()]
                    if tokens:
                        timesteps.append(tokens)
            
            if not timesteps:
                raise ValueError("No valid timesteps found")
            
            # Each timestep should have 7 tokens (for mini-omni's SNAC format)
            # Transpose: from [num_timesteps, 7] to [7, num_timesteps]
            num_codebooks = len(timesteps[0])  # Usually 7
            
            # Transpose to get layers
            layers = []
            for cb_idx in range(num_codebooks):
                layer = [ts[cb_idx] for ts in timesteps if cb_idx < len(ts)]
                layers.append(layer)
            
            # SNAC expects 3 hierarchical layers with ratios 1:2:4
            # The 7 streams need to be reorganized:
            # Layer 0 (1 token/frame): indices 0
            # Layer 1 (2 tokens/frame): indices 1, 2
            # Layer 2 (4 tokens/frame): indices 3, 4, 5, 6
            
            if num_codebooks == 7:
                # Reorganize into proper SNAC format
                n_frames = len(layers[0])
                
                # Layer 0: 1 token per frame
                layer0 = layers[0]
                
                # Layer 1: 2 tokens per frame (interleaved from streams 1,2)
                layer1 = []
                for i in range(n_frames):
                    layer1.append(layers[1][i])
                    layer1.append(layers[2][i])
                
                # Layer 2: 4 tokens per frame (interleaved from streams 3,4,5,6)
                layer2 = []
                for i in range(n_frames):
                    layer2.append(layers[3][i])
                    layer2.append(layers[4][i])
                    layer2.append(layers[5][i])
                    layer2.append(layers[6][i])
                
                return [layer0, layer1, layer2]
            else:
                # Unknown format, return as-is
                return layers
        
        # Try JSON format
        try:
            snac_tokens = json.loads(snac_tokens)
        except json.JSONDecodeError:
            # Try Python literal format
            try:
                snac_tokens = ast.literal_eval(snac_tokens)
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"Cannot parse SNAC tokens: {e}")
    
    if isinstance(snac_tokens, np.ndarray):
        snac_tokens = snac_tokens.tolist()
    
    return snac_tokens


def decode_snac_to_audio(
    snac_model: "SNAC",
    snac_tokens,
    device: str = "cuda",
) -> tuple[np.ndarray, int]:
    """
    Decode SNAC tokens to audio waveform.
    
    Returns:
        Tuple of (audio_waveform, sample_rate)
    """
    snac_tokens = parse_snac_tokens(snac_tokens)
    
    if not isinstance(snac_tokens, list) or len(snac_tokens) == 0:
        raise ValueError(f"Invalid SNAC tokens: {type(snac_tokens)}")
    
    # Convert to tensors - SNAC expects list of tensors, one per layer
    if isinstance(snac_tokens[0], list):
        codes = [
            torch.tensor(layer, dtype=torch.long, device=device).unsqueeze(0)
            for layer in snac_tokens
        ]
    else:
        codes_tensor = torch.tensor(snac_tokens, dtype=torch.long, device=device)
        if codes_tensor.dim() == 1:
            codes = [codes_tensor.unsqueeze(0)]
        elif codes_tensor.dim() == 2:
            codes = [codes_tensor[i:i+1, :] for i in range(codes_tensor.shape[0])]
        else:
            raise ValueError(f"Unexpected tensor shape: {codes_tensor.shape}")
    
    with torch.no_grad():
        audio = snac_model.decode(codes)
    
    audio_np = audio.cpu().numpy().squeeze()
    sample_rate = 24000  # SNAC 24kHz
    
    return audio_np, sample_rate


def encode_audio_to_mimi(
    mimi_model,
    audio: np.ndarray,
    sample_rate: int = 24000,
    device: str = "cuda",
) -> list:
    """
    Encode audio waveform to Mimi tokens.
    
    Args:
        mimi_model: Loaded Mimi codec
        audio: Audio waveform as numpy array
        sample_rate: Input sample rate
        device: Device to run encoding on
    
    Returns:
        Mimi tokens as list of lists (one per codebook)
    """
    # Mimi expects 24kHz audio
    target_sample_rate = mimi_model.sample_rate  # Usually 24000
    
    # Resample if needed
    if sample_rate != target_sample_rate:
        import torch.nn.functional as F
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        ratio = target_sample_rate / sample_rate
        new_length = int(len(audio) * ratio)
        audio_tensor = F.interpolate(
            audio_tensor.unsqueeze(0).unsqueeze(0),
            size=new_length,
            mode='linear',
            align_corners=False
        ).squeeze()
        audio = audio_tensor.numpy()
    
    # Convert to tensor with proper shape: [batch, channels, samples]
    audio_tensor = torch.tensor(audio, dtype=torch.float32)
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
    
    audio_tensor = audio_tensor.to(device)
    
    # Encode with Mimi
    with torch.no_grad():
        codes = mimi_model.encode(audio_tensor)
    
    # codes shape: [batch, n_codebooks, time]
    # Convert to list format
    codes_np = codes.cpu().numpy()
    mimi_tokens = codes_np[0].tolist()  # [n_codebooks, time]
    
    return mimi_tokens


# ============================================================================
# Main processing functions
# ============================================================================

def download_dataset(output_dir: str, num_files: int | None = None) -> list[Path]:
    """Download VoiceAssistant-400K dataset parquet files."""
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


def process_single_file(
    input_path: Path,
    output_path: Path,
    snac_model,
    mimi_model,
    checkpoint: ProcessingCheckpoint,
    checkpoint_path: Path,
    save_interval: int = 100,
    device: str = "cuda",
) -> None:
    """
    Process a single parquet file with resumability.
    
    Args:
        input_path: Input parquet file path
        output_path: Output parquet file path
        snac_model: Loaded SNAC model
        mimi_model: Loaded Mimi codec
        checkpoint: Current checkpoint state
        checkpoint_path: Path to save checkpoints
        save_interval: Save checkpoint every N rows
        device: Device to use
    """
    filename = input_path.name
    print(f"\nProcessing: {filename}")
    
    # Check if already completed
    if checkpoint.is_file_complete(filename):
        print(f"  Skipping (already completed)")
        return
    
    # Read parquet file
    table = pq.read_table(input_path)
    df = table.to_pandas()
    
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    if "answer_snac" not in df.columns:
        print("  Warning: No answer_snac column, skipping")
        checkpoint.mark_file_complete(filename)
        checkpoint.save(checkpoint_path)
        return
    
    # Initialize or restore partial results
    if checkpoint.current_file == filename:
        start_row = checkpoint.current_row
        audio_results = checkpoint.partial_results.get("audio", [None] * len(df))
        mimi_results = checkpoint.partial_results.get("mimi", [None] * len(df))
        print(f"  Resuming from row {start_row}")
    else:
        start_row = 0
        audio_results = [None] * len(df)
        mimi_results = [None] * len(df)
        checkpoint.current_file = filename
        checkpoint.current_row = 0
        checkpoint.partial_results = {"audio": audio_results, "mimi": mimi_results}
    
    # Process each row
    errors = 0
    pbar = tqdm(
        range(start_row, len(df)),
        initial=start_row,
        total=len(df),
        desc="  Converting"
    )
    
    for idx in pbar:
        snac_tokens = df["answer_snac"].iloc[idx]
        
        # Skip if null
        if snac_tokens is None or (isinstance(snac_tokens, float) and pd.isna(snac_tokens)):
            continue
        
        try:
            # Step 1: SNAC → Audio
            audio, sample_rate = decode_snac_to_audio(snac_model, snac_tokens, device)
            
            # Store audio in HuggingFace Audio format
            # Format: {"array": numpy_array, "sampling_rate": int}
            audio_results[idx] = {"array": audio.astype(np.float32), "sampling_rate": sample_rate}
            
            # Step 2: Audio → Mimi tokens
            mimi_tokens = encode_audio_to_mimi(mimi_model, audio, sample_rate, device)
            mimi_results[idx] = mimi_tokens
            
        except Exception as e:
            errors += 1
            if errors <= 5:
                pbar.write(f"  Error row {idx}: {str(e)[:80]}")
            elif errors == 6:
                pbar.write(f"  ... suppressing further error messages")
        
        # Periodic checkpoint save
        if (idx + 1) % save_interval == 0:
            checkpoint.current_row = idx + 1
            checkpoint.partial_results = {"audio": audio_results, "mimi": mimi_results}
            checkpoint.save(checkpoint_path)
    
    if errors > 0:
        print(f"  Total errors: {errors}/{len(df)}")
    
    # Add new columns
    df["answer_audio"] = audio_results
    df["answer_mimi"] = mimi_results
    
    # Save using HuggingFace datasets library for proper Audio column handling
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    from datasets import Dataset, Audio, Features, Value, Sequence
    
    # Define features - answer_audio as Audio type for playback in Data Studio
    features = Features({
        'split_name': Value('string'),
        'index': Value('int64'),
        'round': Value('int64'),
        'question': Value('string'),
        'question_audio': Audio(sampling_rate=24000),
        'answer': Value('string'),
        'answer_snac': Value('string'),
        'answer_audio': Audio(sampling_rate=24000),
        'answer_mimi': Sequence(Sequence(Value('int64'))),
    })
    
    # Convert DataFrame to HuggingFace Dataset
    hf_dataset = Dataset.from_pandas(df, features=features)
    
    # Save as parquet with proper audio encoding
    hf_dataset.to_parquet(output_path)
    print(f"  Saved to: {output_path}")
    
    # Mark complete and save checkpoint
    checkpoint.mark_file_complete(filename)
    checkpoint.save(checkpoint_path)


def inspect_snac_data(parquet_files: list[Path], num_samples: int = 3):
    """Inspect the actual format of SNAC data in parquet files."""
    print("\n" + "=" * 50)
    print("Inspecting SNAC data format...")
    print("=" * 50)
    
    for pf in parquet_files[:1]:
        table = pq.read_table(pf)
        df = table.to_pandas()
        
        print(f"\nFile: {pf.name}")
        print(f"Columns: {list(df.columns)}")
        
        if "answer_snac" not in df.columns:
            print("  No answer_snac column!")
            continue
        
        print(f"\nanswer_snac column dtype: {df['answer_snac'].dtype}")
        
        for i in range(min(num_samples, len(df))):
            val = df["answer_snac"].iloc[i]
            print(f"\n--- Row {i} ---")
            print(f"  Python type: {type(val).__name__}")
            
            if val is None:
                print("  Value: None")
            elif isinstance(val, str):
                print(f"  String length: {len(val)}")
                print(f"  First 300 chars: {repr(val[:300])}")
                
                # Try to parse and show structure
                try:
                    parsed = parse_snac_tokens(val)
                    print(f"  Parsed successfully!")
                    print(f"  Number of layers: {len(parsed)}")
                    print(f"  Layer lengths: {[len(layer) for layer in parsed]}")
                    # SNAC expects ratios 1:2:4
                    if len(parsed) == 3:
                        l0, l1, l2 = len(parsed[0]), len(parsed[1]), len(parsed[2])
                        print(f"  Layer length ratios: 1:{l1//l0 if l0 else 0}:{l2//l0 if l0 else 0} (expected 1:2:4)")
                    print(f"  First layer first 10 tokens: {parsed[0][:10] if parsed else 'empty'}")
                except Exception as e:
                    print(f"  Parse error: {e}")
                    
            elif isinstance(val, bytes):
                print(f"  Bytes length: {len(val)}")
                print(f"  First 50 bytes: {val[:50]}")
            elif isinstance(val, (list, np.ndarray)):
                print(f"  Length: {len(val)}")
                if len(val) > 0:
                    print(f"  First element type: {type(val[0]).__name__}")
                    if isinstance(val[0], (list, np.ndarray)):
                        print(f"  Nested structure - layer lengths: {[len(x) for x in val[:5]]}")
                    else:
                        print(f"  First 10 values: {list(val[:10])}")
            else:
                print(f"  Value: {repr(val)[:200]}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert VoiceAssistant-400K: SNAC → Audio → Mimi tokens"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for data and processed files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data-dir/converted)",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=None,
        help="Limit number of files to process",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run models on",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading dataset",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="conversion_checkpoint.pkl",
        help="Checkpoint file name",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Save checkpoint every N rows",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test without loading models",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Only inspect SNAC data format, don't process",
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "converted"
    checkpoint_path = data_dir / args.checkpoint_file
    
    # Load or create checkpoint
    if args.resume and checkpoint_path.exists():
        checkpoint = ProcessingCheckpoint.load(checkpoint_path)
        print(f"Resuming from checkpoint (started: {checkpoint.started_at})")
        print(f"  Completed files: {len(checkpoint.completed_files)}")
        if checkpoint.current_file:
            print(f"  Current file: {checkpoint.current_file} (row {checkpoint.current_row})")
    else:
        checkpoint = ProcessingCheckpoint(started_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        if args.resume:
            print("No checkpoint found, starting fresh")
    
    # Download or find dataset files
    if args.skip_download:
        dataset_path = data_dir / "VoiceAssistant-400K" / "data"
        parquet_files = sorted(dataset_path.glob("*.parquet"))
        if args.num_files:
            parquet_files = parquet_files[:args.num_files]
        print(f"Using existing files: {len(parquet_files)} parquet files")
    else:
        parquet_files = download_dataset(str(data_dir), args.num_files)
    
    if not parquet_files:
        print("No parquet files found!")
        return
    
    # Inspect mode - just show data format
    if args.inspect:
        inspect_snac_data(parquet_files)
        return
    
    # Count remaining work
    remaining = sum(1 for f in parquet_files if not checkpoint.is_file_complete(f.name))
    print(f"Files to process: {remaining}/{len(parquet_files)}")
    
    if args.dry_run:
        print("\nDry run - not loading models or processing")
        return
    
    # Load models
    print("\n" + "=" * 50)
    print("Loading models...")
    print("=" * 50)
    
    snac_model = load_snac_model(device=args.device)
    mimi_model = load_mimi_model(device=args.device)
    
    # Process files
    print("\n" + "=" * 50)
    print(f"Processing {remaining} files...")
    print("=" * 50)
    
    for parquet_file in parquet_files:
        output_file = output_dir / parquet_file.name
        process_single_file(
            input_path=parquet_file,
            output_path=output_file,
            snac_model=snac_model,
            mimi_model=mimi_model,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            save_interval=args.save_interval,
            device=args.device,
        )
    
    # Final summary
    print("\n" + "=" * 50)
    print("Processing complete!")
    print("=" * 50)
    print(f"Output files: {output_dir}")
    print(f"Total completed: {len(checkpoint.completed_files)} files")
    
    # Clean up checkpoint on successful completion
    if remaining == 0 or len(checkpoint.completed_files) == len(parquet_files):
        print("All files processed - removing checkpoint file")
        if checkpoint_path.exists():
            checkpoint_path.unlink()


if __name__ == "__main__":
    main()

