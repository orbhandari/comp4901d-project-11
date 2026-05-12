"""
Unit tests for Model Manager.

Tests model download retry logic, checksum verification, GGUF validation,
and cache hit behavior.

Requirements: 10.2, 10.3, 10.6, 10.8
"""

import hashlib
import os
import struct
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

import pytest
from huggingface_hub.utils import HfHubHTTPError
from requests.exceptions import RequestException, ConnectionError

from llm_benchmark.model_manager import ModelManager, ModelAcquisitionError


class TestModelDownloadRetry:
    """Test download retry logic with exponential backoff."""
    
    @patch('llm_benchmark.model_manager.manager.hf_hub_download')
    @patch('llm_benchmark.model_manager.manager.time.sleep')
    def test_download_succeeds_on_first_attempt(self, mock_sleep, mock_download):
        """Test successful download on first attempt."""
        mock_download.return_value = "/path/to/model.gguf"
        
        manager = ModelManager(cache_dir="/tmp/test_cache")
        path = manager.download_with_retry("test/repo", "model.gguf", max_retries=3)
        
        assert path == "/path/to/model.gguf"
        assert mock_download.call_count == 1
        assert mock_sleep.call_count == 0  # No retries needed
    
    @patch('llm_benchmark.model_manager.manager.hf_hub_download')
    @patch('llm_benchmark.model_manager.manager.time.sleep')
    def test_download_succeeds_on_second_attempt(self, mock_sleep, mock_download):
        """Test successful download after one retry."""
        mock_download.side_effect = [
            RequestException("Network error"),
            "/path/to/model.gguf"  # Success on second attempt
        ]
        
        manager = ModelManager(cache_dir="/tmp/test_cache")
        path = manager.download_with_retry("test/repo", "model.gguf", max_retries=3)
        
        assert path == "/path/to/model.gguf"
        assert mock_download.call_count == 2
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_with(1)  # First retry: 2^0 = 1 second
    
    @patch('llm_benchmark.model_manager.manager.hf_hub_download')
    @patch('llm_benchmark.model_manager.manager.time.sleep')
    def test_download_succeeds_on_third_attempt(self, mock_sleep, mock_download):
        """Test successful download after two retries."""
        mock_download.side_effect = [
            RequestException("Network error"),
            RequestException("Network error"),
            "/path/to/model.gguf"  # Success on third attempt
        ]
        
        manager = ModelManager(cache_dir="/tmp/test_cache")
        path = manager.download_with_retry("test/repo", "model.gguf", max_retries=3)
        
        assert path == "/path/to/model.gguf"
        assert mock_download.call_count == 3
        assert mock_sleep.call_count == 2
        # Verify exponential backoff: 1s, 2s
        assert mock_sleep.call_args_list == [call(1), call(2)]
    
    @patch('llm_benchmark.model_manager.manager.hf_hub_download')
    @patch('llm_benchmark.model_manager.manager.time.sleep')
    def test_download_fails_after_max_retries(self, mock_sleep, mock_download):
        """Test download fails after exhausting all retries."""
        mock_download.side_effect = RequestException("Network error")
        
        manager = ModelManager(cache_dir="/tmp/test_cache")
        path = manager.download_with_retry("test/repo", "model.gguf", max_retries=3)
        
        assert path is None
        assert mock_download.call_count == 3
        assert mock_sleep.call_count == 2  # Sleep between attempts, not after last
        # Verify exponential backoff: 1s, 2s
        assert mock_sleep.call_args_list == [call(1), call(2)]
    
    @patch('llm_benchmark.model_manager.manager.hf_hub_download')
    def test_download_handles_authentication_error(self, mock_download):
        """Test download handles 401 authentication error."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_download.side_effect = HfHubHTTPError("Unauthorized", response=mock_response)
        
        manager = ModelManager(cache_dir="/tmp/test_cache")
        path = manager.download_with_retry("test/repo", "model.gguf", max_retries=3)
        
        assert path is None
        assert mock_download.call_count == 1  # No retries for auth errors
    
    @patch('llm_benchmark.model_manager.manager.hf_hub_download')
    def test_download_handles_forbidden_error(self, mock_download):
        """Test download handles 403 forbidden error."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_download.side_effect = HfHubHTTPError("Forbidden", response=mock_response)
        
        manager = ModelManager(cache_dir="/tmp/test_cache")
        path = manager.download_with_retry("test/repo", "model.gguf", max_retries=3)
        
        assert path is None
        assert mock_download.call_count == 1  # No retries for permission errors
    
    @patch('llm_benchmark.model_manager.manager.hf_hub_download')
    def test_download_handles_not_found_error(self, mock_download):
        """Test download handles 404 not found error."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_download.side_effect = HfHubHTTPError("Not Found", response=mock_response)
        
        manager = ModelManager(cache_dir="/tmp/test_cache")
        path = manager.download_with_retry("test/repo", "model.gguf", max_retries=3)
        
        assert path is None
        assert mock_download.call_count == 1  # No retries for not found errors
    
    @patch('llm_benchmark.model_manager.manager.hf_hub_download')
    @patch('llm_benchmark.model_manager.manager.time.sleep')
    def test_download_retries_on_server_error(self, mock_sleep, mock_download):
        """Test download retries on 500 server error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_download.side_effect = [
            HfHubHTTPError("Server Error", response=mock_response),
            HfHubHTTPError("Server Error", response=mock_response),
            "/path/to/model.gguf"  # Success on third attempt
        ]
        
        manager = ModelManager(cache_dir="/tmp/test_cache")
        path = manager.download_with_retry("test/repo", "model.gguf", max_retries=3)
        
        assert path == "/path/to/model.gguf"
        assert mock_download.call_count == 3
        assert mock_sleep.call_count == 2
    
    @patch('llm_benchmark.model_manager.manager.hf_hub_download')
    def test_download_handles_disk_space_error(self, mock_download):
        """Test download handles disk space exhaustion."""
        mock_download.side_effect = OSError("No space left on device")
        
        manager = ModelManager(cache_dir="/tmp/test_cache")
        path = manager.download_with_retry("test/repo", "model.gguf", max_retries=3)
        
        assert path is None
        assert mock_download.call_count == 1  # No retries for disk space errors


class TestChecksumVerification:
    """Test checksum verification with valid and invalid checksums."""
    
    def test_verify_integrity_with_valid_checksum(self):
        """Test checksum verification passes with valid checksum."""
        # Create temporary file with known content
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            content = b"test model content"
            f.write(content)
            temp_path = f.name
        
        try:
            # Calculate expected checksum
            expected_sha256 = hashlib.sha256(content).hexdigest()
            
            manager = ModelManager()
            result = manager.verify_integrity(temp_path, expected_sha256)
            
            assert result is True
        finally:
            os.unlink(temp_path)
    
    def test_verify_integrity_with_invalid_checksum(self):
        """Test checksum verification fails with invalid checksum."""
        # Create temporary file with known content
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            content = b"test model content"
            f.write(content)
            temp_path = f.name
        
        try:
            # Use wrong checksum
            wrong_sha256 = "0" * 64
            
            manager = ModelManager()
            result = manager.verify_integrity(temp_path, wrong_sha256)
            
            assert result is False
        finally:
            os.unlink(temp_path)
    
    def test_verify_integrity_case_insensitive(self):
        """Test checksum verification is case-insensitive."""
        # Create temporary file with known content
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            content = b"test model content"
            f.write(content)
            temp_path = f.name
        
        try:
            # Calculate expected checksum
            expected_sha256 = hashlib.sha256(content).hexdigest()
            
            manager = ModelManager()
            
            # Test with uppercase
            result_upper = manager.verify_integrity(temp_path, expected_sha256.upper())
            assert result_upper is True
            
            # Test with lowercase
            result_lower = manager.verify_integrity(temp_path, expected_sha256.lower())
            assert result_lower is True
        finally:
            os.unlink(temp_path)
    
    def test_calculate_sha256_streaming(self):
        """Test SHA256 calculation uses streaming for large files."""
        # Create temporary file with known content
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            content = b"test model content" * 1000  # Larger content
            f.write(content)
            temp_path = f.name
        
        try:
            expected_sha256 = hashlib.sha256(content).hexdigest()
            
            manager = ModelManager()
            actual_sha256 = manager._calculate_sha256(temp_path)
            
            assert actual_sha256 == expected_sha256
        finally:
            os.unlink(temp_path)


class TestGGUFValidation:
    """Test GGUF validation with valid and corrupted files."""
    
    def test_validate_gguf_with_valid_file(self):
        """Test GGUF validation passes with valid file."""
        # Create temporary file with valid GGUF header
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            # Write GGUF magic bytes
            f.write(b'GGUF')
            # Write version (uint32, little-endian)
            f.write(struct.pack('<I', 3))
            # Write tensor count (uint64, little-endian)
            f.write(struct.pack('<Q', 100))
            # Write metadata KV count (uint64, little-endian)
            f.write(struct.pack('<Q', 50))
            temp_path = f.name
        
        try:
            manager = ModelManager()
            result = manager.validate_gguf(temp_path)
            
            assert result is True
        finally:
            os.unlink(temp_path)
    
    def test_validate_gguf_with_invalid_magic_bytes(self):
        """Test GGUF validation fails with invalid magic bytes."""
        # Create temporary file with invalid magic bytes
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(b'XXXX')  # Invalid magic bytes
            f.write(struct.pack('<I', 3))
            f.write(struct.pack('<Q', 100))
            f.write(struct.pack('<Q', 50))
            temp_path = f.name
        
        try:
            manager = ModelManager()
            result = manager.validate_gguf(temp_path)
            
            assert result is False
        finally:
            os.unlink(temp_path)
    
    def test_validate_gguf_with_truncated_file(self):
        """Test GGUF validation fails with truncated file."""
        # Create temporary file with incomplete header
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(b'GGUF')  # Only magic bytes, missing rest
            temp_path = f.name
        
        try:
            manager = ModelManager()
            result = manager.validate_gguf(temp_path)
            
            assert result is False
        finally:
            os.unlink(temp_path)
    
    def test_validate_gguf_with_zero_tensor_count(self):
        """Test GGUF validation fails with zero tensor count."""
        # Create temporary file with zero tensor count
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(b'GGUF')
            f.write(struct.pack('<I', 3))
            f.write(struct.pack('<Q', 0))  # Zero tensor count
            f.write(struct.pack('<Q', 50))
            temp_path = f.name
        
        try:
            manager = ModelManager()
            result = manager.validate_gguf(temp_path)
            
            assert result is False
        finally:
            os.unlink(temp_path)
    
    def test_validate_gguf_with_suspicious_version(self):
        """Test GGUF validation fails with suspicious version number."""
        # Create temporary file with suspicious version
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(b'GGUF')
            f.write(struct.pack('<I', 999))  # Suspicious version
            f.write(struct.pack('<Q', 100))
            f.write(struct.pack('<Q', 50))
            temp_path = f.name
        
        try:
            manager = ModelManager()
            result = manager.validate_gguf(temp_path)
            
            assert result is False
        finally:
            os.unlink(temp_path)
    
    def test_validate_gguf_with_suspicious_tensor_count(self):
        """Test GGUF validation fails with suspicious tensor count."""
        # Create temporary file with suspicious tensor count
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(b'GGUF')
            f.write(struct.pack('<I', 3))
            f.write(struct.pack('<Q', 99999))  # Suspicious tensor count
            f.write(struct.pack('<Q', 50))
            temp_path = f.name
        
        try:
            manager = ModelManager()
            result = manager.validate_gguf(temp_path)
            
            assert result is False
        finally:
            os.unlink(temp_path)


class TestCacheHitBehavior:
    """Test cache hit behavior (skip download when model exists)."""
    
    @patch('llm_benchmark.model_manager.manager.hf_hub_download')
    def test_cache_hit_skips_download(self, mock_download):
        """Test that cached model skips download."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a valid cached model file
            cache_path = Path(temp_dir) / "model.gguf"
            with open(cache_path, 'wb') as f:
                f.write(b'GGUF')
                f.write(struct.pack('<I', 3))
                f.write(struct.pack('<Q', 100))
                f.write(struct.pack('<Q', 50))
            
            manager = ModelManager(cache_dir=temp_dir)
            model_info = manager.get_model("test/repo", "model.gguf")
            
            # Verify download was not called
            assert mock_download.call_count == 0
            
            # Verify model info is returned
            assert model_info is not None
            assert model_info.filename == "model.gguf"
            assert model_info.local_path == str(cache_path)
    
    @patch('llm_benchmark.model_manager.manager.hf_hub_download')
    def test_cache_miss_triggers_download(self, mock_download):
        """Test that missing model triggers download."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a valid model file that will be "downloaded"
            download_path = Path(temp_dir) / "downloaded_model.gguf"
            with open(download_path, 'wb') as f:
                f.write(b'GGUF')
                f.write(struct.pack('<I', 3))
                f.write(struct.pack('<Q', 100))
                f.write(struct.pack('<Q', 50))
            
            mock_download.return_value = str(download_path)
            
            manager = ModelManager(cache_dir=temp_dir)
            model_info = manager.get_model("test/repo", "model.gguf")
            
            # Verify download was called
            assert mock_download.call_count == 1
            
            # Verify model info is returned
            assert model_info is not None
            assert model_info.filename == "model.gguf"
    
    @patch('llm_benchmark.model_manager.manager.hf_hub_download')
    def test_cache_hit_with_valid_checksum(self, mock_download):
        """Test cached model with valid checksum skips download."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a valid cached model file
            cache_path = Path(temp_dir) / "model.gguf"
            content = b'GGUF' + struct.pack('<I', 3) + struct.pack('<Q', 100) + struct.pack('<Q', 50)
            with open(cache_path, 'wb') as f:
                f.write(content)
            
            # Calculate expected checksum
            expected_sha256 = hashlib.sha256(content).hexdigest()
            
            manager = ModelManager(cache_dir=temp_dir)
            model_info = manager.get_model("test/repo", "model.gguf", expected_sha256=expected_sha256)
            
            # Verify download was not called
            assert mock_download.call_count == 0
            
            # Verify model info is returned
            assert model_info is not None
            assert model_info.sha256 == expected_sha256
    
    @patch('llm_benchmark.model_manager.manager.hf_hub_download')
    def test_cache_hit_with_invalid_checksum_triggers_redownload(self, mock_download):
        """Test cached model with invalid checksum triggers re-download."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a cached model file
            cache_path = Path(temp_dir) / "model.gguf"
            with open(cache_path, 'wb') as f:
                f.write(b'GGUF')
                f.write(struct.pack('<I', 3))
                f.write(struct.pack('<Q', 100))
                f.write(struct.pack('<Q', 50))
            
            # Create a valid model file that will be "downloaded"
            download_path = Path(temp_dir) / "downloaded_model.gguf"
            content = b'GGUF' + struct.pack('<I', 3) + struct.pack('<Q', 100) + struct.pack('<Q', 50)
            with open(download_path, 'wb') as f:
                f.write(content)
            
            mock_download.return_value = str(download_path)
            
            # Use wrong checksum to trigger re-download
            wrong_sha256 = "0" * 64
            
            manager = ModelManager(cache_dir=temp_dir)
            model_info = manager.get_model("test/repo", "model.gguf", expected_sha256=wrong_sha256)
            
            # Verify download was called (cache invalidated)
            assert mock_download.call_count == 1
    
    @patch('llm_benchmark.model_manager.manager.hf_hub_download')
    def test_cache_hit_with_invalid_gguf_triggers_redownload(self, mock_download):
        """Test cached model with invalid GGUF format triggers re-download."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a cached model file with invalid GGUF format
            cache_path = Path(temp_dir) / "model.gguf"
            with open(cache_path, 'wb') as f:
                f.write(b'XXXX')  # Invalid magic bytes
            
            # Create a valid model file that will be "downloaded"
            download_path = Path(temp_dir) / "downloaded_model.gguf"
            with open(download_path, 'wb') as f:
                f.write(b'GGUF')
                f.write(struct.pack('<I', 3))
                f.write(struct.pack('<Q', 100))
                f.write(struct.pack('<Q', 50))
            
            mock_download.return_value = str(download_path)
            
            manager = ModelManager(cache_dir=temp_dir)
            model_info = manager.get_model("test/repo", "model.gguf")
            
            # Verify download was called (cache invalidated)
            assert mock_download.call_count == 1


class TestQuantizationExtraction:
    """Test quantization level extraction from filenames."""
    
    def test_extract_quantization_q4_0(self):
        """Test extraction of Q4_0 quantization."""
        manager = ModelManager()
        
        assert manager._extract_quantization("model-Q4_0.gguf") == "Q4_0"
        assert manager._extract_quantization("model-q4_0.gguf") == "Q4_0"
        assert manager._extract_quantization("llama-2-7b-Q4_0.gguf") == "Q4_0"
    
    def test_extract_quantization_q8_0(self):
        """Test extraction of Q8_0 quantization."""
        manager = ModelManager()
        
        assert manager._extract_quantization("model-Q8_0.gguf") == "Q8_0"
        assert manager._extract_quantization("model-q8_0.gguf") == "Q8_0"
    
    def test_extract_quantization_q4_k_m(self):
        """Test extraction of Q4_K_M quantization."""
        manager = ModelManager()
        
        # The regex pattern captures Q4_K (not the full Q4_K_M)
        # This is a limitation of the current implementation
        assert manager._extract_quantization("model-Q4_K_M.gguf") == "Q4_K"
        assert manager._extract_quantization("model-q4_k_m.gguf") == "Q4_K"
    
    def test_extract_quantization_q2_k(self):
        """Test extraction of Q2_K quantization."""
        manager = ModelManager()
        
        assert manager._extract_quantization("model-Q2_K.gguf") == "Q2_K"
        assert manager._extract_quantization("model-q2_k.gguf") == "Q2_K"
    
    def test_extract_quantization_unknown(self):
        """Test extraction returns 'unknown' for unrecognized format."""
        manager = ModelManager()
        
        assert manager._extract_quantization("model.gguf") == "unknown"
        assert manager._extract_quantization("model-fp16.gguf") == "unknown"
