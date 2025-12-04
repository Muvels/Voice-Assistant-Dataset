"""
Process VoiceAssistant-400K dataset: answer text → EdgeTTS audio → Mimi tokens.

This script:
1. Downloads the VoiceAssistant-400K dataset from HuggingFace
2. Uses Microsoft Edge TTS (edge-tts) to synthesize audio from the "answer" text
3. Stores the synthesized audio in an `answer_audio` column (WAV bytes, 24kHz)
4. Encodes the audio to Mimi tokens using Kyutai's Mimi codec
5. Stores Mimi tokens in an `answer_mimi` column
6. Saves back to parquet files

Usage:
    uv run python process_edgetts_to_mimi.py --data-dir ./data --num-files 1
"""

from __future__ import annotations

import argparse
import asyncio
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

# Optional: Edge TTS for text-to-speech
EDGE_TTS_AVAILABLE = False
EDGE_TTS_IMPORT_ERROR = None

try:
    import edge_tts

    EDGE_TTS_AVAILABLE = True
except ImportError as e:
    EDGE_TTS_IMPORT_ERROR = str(e)
except Exception as e:  # pragma: no cover - defensive
    EDGE_TTS_IMPORT_ERROR = str(e)

# Optional: Moshi / Mimi codec for encoding audio to Mimi tokens
MIMI_AVAILABLE = False
MIMI_IMPORT_ERROR = None

try:
    from moshi.models.loaders import CheckpointInfo
    from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel

    MIMI_AVAILABLE = True
except ImportError as e:
    MIMI_IMPORT_ERROR = str(e)
except Exception as e:  # pragma: no cover - defensive
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


def load_mimi_model(device: str = "cuda"):
    """
    Load Mimi codec from Kyutai TTS for encoding audio to tokens.
    """
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

    mimi = tts_model.mimi
    print("Mimi codec loaded!")
    return mimi


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
            mode="linear",
            align_corners=False,
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
    codes_np = codes.cpu().numpy()
    mimi_tokens = codes_np[0].tolist()  # [n_codebooks, time]

    return mimi_tokens


async def _edge_tts_synthesize_async(
    text: str,
    voice: str = "en-US-JennyNeural",
    output_format: str = "riff-24khz-16bit-mono-pcm",
) -> bytes:
    """
    Synthesize speech from text using Edge TTS and return WAV/PCM bytes.

    The default `output_format` produces a 24kHz mono PCM RIFF (WAV) file.
    """
    if not EDGE_TTS_AVAILABLE:
        raise ImportError(
            f"edge-tts package not available. Error: {EDGE_TTS_IMPORT_ERROR}"
        )

    communicate = edge_tts.Communicate(
        text,
        voice=voice,
        output_format=output_format,
    )

    audio_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]
    return audio_bytes


def edge_tts_synthesize(
    text: str,
    voice: str = "en-US-JennyNeural",
    output_format: str = "riff-24khz-16bit-mono-pcm",
) -> bytes:
    """
    Synchronous wrapper around Edge TTS synthesis.
    """
    return asyncio.run(_edge_tts_synthesize_async(text, voice=voice, output_format=output_format))


def process_parquet_file(
    input_path: Path,
    output_path: Path,
    mimi_model,
    device: str = "cuda",
    voice: str = "en-US-JennyNeural",
    save_audio_files: bool = False,
    audio_output_dir: Path | None = None,
) -> None:
    """
    Process a single parquet file, generating EdgeTTS audio and Mimi tokens.

    Args:
        input_path: Path to input parquet file
        output_path: Path to save processed parquet file
        mimi_model: Loaded Mimi codec (if None, Mimi tokens are not generated)
        device: Device for Mimi encoding ("cuda" or "cpu")
        voice: Edge TTS voice name (e.g., "en-US-JennyNeural")
        save_audio_files: Whether to save audio as separate .wav files
        audio_output_dir: Directory for audio files (if save_audio_files=True)
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

    if save_audio_files and audio_output_dir:
        audio_output_dir.mkdir(parents=True, exist_ok=True)

    audio_list: list | None = []
    mimi_list: list | None = [] if mimi_model is not None else None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  EdgeTTS + Mimi"):
        answer_text = row["answer"]

        if (
            answer_text is None
            or (isinstance(answer_text, float) and pd.isna(answer_text))
            or not isinstance(answer_text, str)
            or len(answer_text.strip()) == 0
        ):
            audio_list.append(None)
            if mimi_list is not None:
                mimi_list.append(None)
            continue

        try:
            # Step 1: TTS (answer text → WAV bytes)
            wav_bytes = edge_tts_synthesize(answer_text, voice=voice)

            # Decode to float32 waveform for Mimi
            audio_data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")

            # Ensure mono
            if audio_data.ndim == 2:
                # Average stereo channels if needed
                audio_data = audio_data.mean(axis=1)

            # Store audio as HF Audio-compatible dict
            audio_entry = {"bytes": wav_bytes, "path": None}
            audio_list.append(audio_entry)

            # Optionally save WAV file
            if save_audio_files and audio_output_dir:
                wav_path = audio_output_dir / f"{input_path.stem}_{idx}.wav"
                with open(wav_path, "wb") as f:
                    f.write(wav_bytes)

            # Step 2: Audio → Mimi tokens
            if mimi_list is not None:
                mimi_tokens = encode_audio_to_mimi(
                    mimi_model,
                    audio_data,
                    sample_rate=sr,
                    device=device,
                )
                mimi_list.append(mimi_tokens)

        except Exception as e:  # pragma: no cover - best-effort logging
            print(f"  Error processing row {idx}: {e}")
            audio_list.append(None)
            if mimi_list is not None:
                mimi_list.append(None)

    # Add new columns
    df["answer_audio"] = audio_list
    if mimi_list is not None:
        df["answer_mimi"] = mimi_list

    # Save to output parquet file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process VoiceAssistant-400K: EdgeTTS answer audio + Mimi tokens"
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
        help="Directory for processed output files (default: data-dir/processed_edgetts)",
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
        help="Device to run Mimi encoder on",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading dataset (use existing files)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without loading Mimi model (only generate audio)",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="en-US-JennyNeural",
        help="Edge TTS voice to use (e.g., en-US-JennyNeural)",
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
        help="Directory for audio files (default: data-dir/audio_edgetts)",
    )

    args = parser.parse_args()

    if not EDGE_TTS_AVAILABLE:
        raise ImportError(
            "edge-tts is required for this script but is not available. "
            f"Import error: {EDGE_TTS_IMPORT_ERROR}"
        )

    data_dir = Path(args.data_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else data_dir / "processed_edgetts_mimi"
    )
    audio_dir = (
        Path(args.audio_dir)
        if args.audio_dir
        else data_dir / "audio_edgetts"
    )

    # Download or find dataset files
    if args.skip_download:
        dataset_path = data_dir / "VoiceAssistant-400K" / "data"
        parquet_files = sorted(dataset_path.glob("*.parquet"))
        if args.num_files:
            parquet_files = parquet_files[: args.num_files]
        print(f"Using existing files: {len(parquet_files)} parquet files found")
    else:
        parquet_files = download_dataset(str(data_dir), args.num_files)

    if not parquet_files:
        print("No parquet files found!")
        return

    # Load Mimi model (unless dry run)
    mimi_model = None
    if not args.dry_run:
        try:
            mimi_model = load_mimi_model(device=args.device)
        except ImportError as e:
            print(f"Cannot load Mimi model: {e}")
            print("Running in audio-only mode (no Mimi tokens)...")

    # Process each parquet file
    print(f"\nProcessing {len(parquet_files)} files...")

    for parquet_file in parquet_files:
        output_file = output_dir / parquet_file.name
        process_parquet_file(
            input_path=parquet_file,
            output_path=output_file,
            mimi_model=mimi_model,
            device=args.device,
            voice=args.voice,
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


