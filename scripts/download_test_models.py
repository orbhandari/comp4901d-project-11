#!/usr/bin/env python3
"""
Download small test models for CI/CD testing.

This script downloads small quantized models (< 1GB) for fast test execution
in CI/CD pipelines. Models are cached locally to avoid repeated downloads.
"""

import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path to import llm_benchmark modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import hf_hub_download

# Try to load environment variables if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, environment variables will be read directly
    pass

# Test models configuration
# Using small quantized models for fast CI testing
TEST_MODELS = {
    "tinyllama-1.1b-chat-v1.0.Q4_0.gguf": {
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
        "sha256": None,  # Optional: Add known checksums for verification
        "size_mb": 669,  # Approximate size
    },
}

# Cache directory for test models
CACHE_DIR = Path("./models/test_models")


def compute_sha256(file_path: Path) -> str:
    """
    Compute SHA256 checksum of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Hexadecimal SHA256 checksum string
    """
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()


def download_model(
    repo_id: str,
    filename: str,
    cache_dir: Path,
    expected_sha256: Optional[str] = None,
    hf_token: Optional[str] = None
) -> Path:
    """
    Download a model from Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face repository ID
        filename: Model filename
        cache_dir: Local cache directory
        expected_sha256: Expected SHA256 checksum (optional)
        hf_token: Hugging Face API token (optional)
        
    Returns:
        Path to downloaded model file
        
    Raises:
        ValueError: If checksum verification fails
    """
    print(f"Downloading {filename} from {repo_id}...")
    
    # Create cache directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download model using huggingface_hub
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(cache_dir),
            token=hf_token,
            resume_download=True,
        )
        
        model_path = Path(model_path)
        print(f"✓ Downloaded to {model_path}")
        
        # Compute and display checksum
        print(f"Computing SHA256 checksum...")
        actual_sha256 = compute_sha256(model_path)
        print(f"SHA256: {actual_sha256}")
        
        # Verify checksum if provided
        if expected_sha256:
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"Checksum mismatch for {filename}!\n"
                    f"Expected: {expected_sha256}\n"
                    f"Actual:   {actual_sha256}"
                )
            print(f"✓ Checksum verified")
        
        return model_path
        
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}", file=sys.stderr)
        raise


def main():
    """Download all test models."""
    print("=" * 70)
    print("Downloading test models for CI/CD")
    print("=" * 70)
    print()
    
    # Get HF token from environment
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not found in environment. Public models only.")
        print()
    
    # Track download results
    downloaded = []
    failed = []
    
    # Download each test model
    for model_name, config in TEST_MODELS.items():
        try:
            model_path = download_model(
                repo_id=config["repo_id"],
                filename=config["filename"],
                cache_dir=CACHE_DIR,
                expected_sha256=config.get("sha256"),
                hf_token=hf_token,
            )
            downloaded.append((model_name, model_path))
            print()
            
        except Exception as e:
            failed.append((model_name, str(e)))
            print()
    
    # Print summary
    print("=" * 70)
    print("Download Summary")
    print("=" * 70)
    print(f"Successfully downloaded: {len(downloaded)}")
    for model_name, model_path in downloaded:
        print(f"  ✓ {model_name}")
    
    if failed:
        print(f"\nFailed downloads: {len(failed)}")
        for model_name, error in failed:
            print(f"  ✗ {model_name}: {error}")
        sys.exit(1)
    else:
        print("\n✓ All test models downloaded successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
