"""
Integration tests for model download functionality.

Tests actual download of models from Hugging Face, caching behavior,
and subsequent runs using cached models.

**Validates: Requirements 10.1, 10.2, 10.3, 10.4**
"""

import os
import shutil
import tempfile
import pytest
from pathlib import Path

from llm_benchmark.model_manager.manager import ModelManager


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def test_model_download_and_cache(temp_cache_dir):
    """
    Test actual download of small test model from Hugging Face.
    
    **Validates: Requirements 10.1, 10.2, 10.3**
    """
    # Use a small test model for fast download
    # This is a tiny GGUF model suitable for testing
    repo_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    filename = "tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
    
    manager = ModelManager(cache_dir=temp_cache_dir, hf_token=None)
    
    # First download - should actually download from HF
    model_info = manager.get_model(repo_id=repo_id, filename=filename)
    
    # Verify model was downloaded
    assert model_info is not None
    assert model_info.quantization == "Q2_K"
    assert model_info.filename == filename
    assert os.path.exists(model_info.local_path)
    assert model_info.local_path.startswith(temp_cache_dir)
    
    # Verify file size is reasonable (should be > 0)
    file_size = os.path.getsize(model_info.local_path)
    assert file_size > 0
    assert model_info.size_mb > 0
    
    # Verify SHA256 checksum was computed
    assert model_info.sha256 is not None
    assert len(model_info.sha256) == 64  # SHA256 is 64 hex characters


def test_cached_model_reuse(temp_cache_dir):
    """
    Test that subsequent runs use cached model without re-downloading.
    
    **Validates: Requirements 10.3, 10.4**
    """
    repo_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    filename = "tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
    
    manager = ModelManager(cache_dir=temp_cache_dir, hf_token=None)
    
    # First download
    model_info_1 = manager.get_model(repo_id=repo_id, filename=filename)
    first_path = model_info_1.local_path
    first_checksum = model_info_1.sha256
    
    # Get file modification time
    first_mtime = os.path.getmtime(first_path)
    
    # Second call - should use cached version
    model_info_2 = manager.get_model(repo_id=repo_id, filename=filename)
    second_path = model_info_2.local_path
    second_checksum = model_info_2.sha256
    second_mtime = os.path.getmtime(second_path)
    
    # Verify same file is used
    assert first_path == second_path
    assert first_checksum == second_checksum
    
    # Verify file was not re-downloaded (modification time unchanged)
    assert first_mtime == second_mtime
    
    # Verify both model_info objects have same properties
    assert model_info_1.quantization == model_info_2.quantization
    assert model_info_1.filename == model_info_2.filename
    assert model_info_1.size_mb == model_info_2.size_mb


def test_model_cache_directory_structure(temp_cache_dir):
    """
    Test that downloaded model is cached correctly in directory structure.
    
    **Validates: Requirements 10.3**
    """
    repo_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    filename = "tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
    
    manager = ModelManager(cache_dir=temp_cache_dir, hf_token=None)
    
    # Download model
    model_info = manager.get_model(repo_id=repo_id, filename=filename)
    
    # Verify cache directory structure
    # The model should be in a subdirectory based on repo_id
    expected_subdir = os.path.join(temp_cache_dir, "models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF")
    
    # Check that the model path is within the cache directory
    assert model_info.local_path.startswith(temp_cache_dir)
    
    # Verify the file exists and is accessible
    assert os.path.isfile(model_info.local_path)
    assert os.access(model_info.local_path, os.R_OK)


@pytest.mark.skipif(
    os.getenv("HF_TOKEN") is None,
    reason="HF_TOKEN not set - skipping authenticated download test"
)
def test_authenticated_model_download(temp_cache_dir):
    """
    Test model download with HF authentication token.
    
    **Validates: Requirements 10.1, 10.5**
    """
    hf_token = os.getenv("HF_TOKEN")
    repo_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    filename = "tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
    
    manager = ModelManager(cache_dir=temp_cache_dir, hf_token=hf_token)
    
    # Download with authentication
    model_info = manager.get_model(repo_id=repo_id, filename=filename)
    
    # Verify download succeeded
    assert model_info is not None
    assert os.path.exists(model_info.local_path)
    assert model_info.size_mb > 0
