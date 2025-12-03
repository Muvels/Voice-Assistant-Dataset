"""
Test script to verify the Dia2 model works correctly.
Downloads a single parquet file and processes one row.
"""

import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def test_download():
    """Download a single parquet file for testing."""
    print("Downloading a single parquet file for testing...")
    
    local_path = hf_hub_download(
        repo_id="gpt-omni/VoiceAssistant-400K",
        repo_type="dataset",
        filename="data/train-00000-of-00325.parquet",
        local_dir="./data/VoiceAssistant-400K",
    )
    
    print(f"Downloaded to: {local_path}")
    return Path(local_path)


def inspect_parquet(path: Path):
    """Inspect the structure of a parquet file."""
    print(f"\nInspecting: {path}")
    
    table = pq.read_table(path)
    df = table.to_pandas()
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nColumn dtypes:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    print(f"\nFirst row 'answer' text:")
    if "answer" in df.columns:
        first_answer = df["answer"].iloc[0]
        if isinstance(first_answer, str):
            print(f"  '{first_answer[:200]}...'")
        else:
            print(f"  {first_answer}")
    
    print(f"\nFirst row 'answer_snac' (if exists):")
    if "answer_snac" in df.columns:
        first_snac = df["answer_snac"].iloc[0]
        if hasattr(first_snac, "__len__"):
            print(f"  Type: {type(first_snac)}, Length: {len(first_snac)}")
        else:
            print(f"  {first_snac}")
    
    return df


def test_dia2():
    """Test Dia2 model loading and token generation."""
    try:
        from dia2 import Dia2, GenerationConfig, SamplingConfig
    except ImportError:
        print("\nDia2 not installed. To install:")
        print("  1. Clone: git clone https://huggingface.co/nari-labs/Dia2-1B")
        print("  2. cd Dia2-1B && uv sync")
        print("  3. Add Dia2-1B to your Python path or install it")
        return False
    
    print("\nLoading Dia2-1B model...")
    
    try:
        model = Dia2.from_repo("nari-labs/Dia2-1B", device="cuda", dtype="bfloat16")
    except Exception as e:
        print(f"Failed to load on CUDA: {e}")
        print("Trying CPU...")
        model = Dia2.from_repo("nari-labs/Dia2-1B", device="cpu", dtype="float32")
    
    print("Model loaded!")
    
    test_text = "[S1] Hello, this is a test of the Dia2 model."
    print(f"\nGenerating tokens for: '{test_text}'")
    
    config = GenerationConfig(
        cfg_scale=2.0,
        audio=SamplingConfig(temperature=0.8, top_k=50),
        use_cuda_graph=False,  # Disable for testing
    )
    
    result = model.generate(test_text, config=config, verbose=True)
    
    print(f"\nResult type: {type(result)}")
    print(f"Result attributes: {dir(result)}")
    
    if hasattr(result, "audio_tokens"):
        tokens = result.audio_tokens
        if hasattr(tokens, "shape"):
            print(f"Audio tokens shape: {tokens.shape}")
        if hasattr(tokens, "tolist"):
            token_list = tokens.tolist()
            print(f"First 20 tokens: {token_list[:20] if len(token_list) > 20 else token_list}")
    
    return True


def main():
    print("=" * 50)
    print("VoiceAssistant-400K + Dia2-1B Test")
    print("=" * 50)
    
    # Test 1: Download and inspect
    try:
        parquet_path = test_download()
        df = inspect_parquet(parquet_path)
    except Exception as e:
        print(f"Download/inspect failed: {e}")
        return 1
    
    # Test 2: Dia2 model
    print("\n" + "=" * 50)
    print("Testing Dia2 model...")
    print("=" * 50)
    
    if not test_dia2():
        print("\nDia2 test skipped (not installed)")
        print("Run the full script with --dry-run to test without Dia2")
        return 0
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())


