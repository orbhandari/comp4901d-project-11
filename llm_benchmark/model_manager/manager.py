"""
Model Manager for downloading, caching, and validating GGUF models.

This module provides the ModelManager class which handles:
- Model downloading from Hugging Face Hub with retry logic
- Local caching in configurable directory
- SHA256 checksum verification
- GGUF format validation
- Authentication with HF_TOKEN
"""

import hashlib
import logging
import os
import struct
import time
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from requests.exceptions import RequestException

from llm_benchmark.models import ModelInfo

logger = logging.getLogger(__name__)


class ModelAcquisitionError(Exception):
    """Exception raised when model acquisition fails."""
    pass


class ModelManager:
    """
    Manages model acquisition, caching, and validation.
    
    Handles downloading models from Hugging Face Hub, caching them locally,
    verifying integrity with checksums, and validating GGUF format.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, hf_token: Optional[str] = None):
        """
        Initialize ModelManager.
        
        Args:
            cache_dir: Directory for caching models. Defaults to ~/.cache/llm_benchmark/models
            hf_token: Hugging Face API token for authentication. If None, reads from HF_TOKEN env var.
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/llm_benchmark/models")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Get HF token from parameter or environment variable
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        
        logger.info(f"ModelManager initialized with cache_dir: {self.cache_dir}")
        if self.hf_token:
            logger.info("HF_TOKEN configured for authentication")
    
    def get_model(self, repo_id: str, filename: str, 
                  expected_sha256: Optional[str] = None) -> Optional[ModelInfo]:
        """
        Get model from cache or download from Hugging Face.
        
        Implements comprehensive error handling for model acquisition:
        - Network failures with exponential backoff retry
        - Authentication failures with clear error messages
        - Disk space exhaustion with informative error
        - Corrupted downloads with checksum verification
        - Skips model and continues with available models on failure
        
        Args:
            repo_id: Hugging Face repository ID (e.g., "meta-llama/Llama-2-7b-gguf")
            filename: Model filename in the repository
            expected_sha256: Expected SHA256 checksum (optional)
        
        Returns:
            ModelInfo object with model details, or None if acquisition fails
        """
        logger.info(f"Getting model: {repo_id}/{filename}")
        
        try:
            # Check if model exists in cache
            cached_path = self.cache_dir / filename
            
            if cached_path.exists():
                logger.info(f"Model found in cache: {cached_path}")
                
                # Verify integrity if checksum provided
                if expected_sha256:
                    if not self.verify_integrity(str(cached_path), expected_sha256):
                        logger.warning(f"Cached model failed checksum verification, re-downloading")
                        cached_path.unlink()
                    else:
                        logger.info("Cached model passed checksum verification")
                
                # Validate GGUF format
                if cached_path.exists():
                    if not self.validate_gguf(str(cached_path)):
                        logger.warning(f"Cached model failed GGUF validation, re-downloading")
                        cached_path.unlink()
                    else:
                        logger.info("Cached model passed GGUF validation")
            
            # Download if not in cache or failed validation
            if not cached_path.exists():
                logger.info(f"Downloading model from Hugging Face Hub")
                
                # Check disk space before download
                if not self._check_disk_space(filename):
                    logger.error(f"Insufficient disk space to download {filename}")
                    logger.info("Suggestion: Free up disk space or use a different cache directory")
                    return None
                
                local_path = self.download_with_retry(repo_id, filename)
                
                if local_path is None:
                    # Download failed after all retries
                    logger.error(f"Failed to download {filename}, skipping this model")
                    return None
                
                # Verify integrity
                if expected_sha256:
                    if not self.verify_integrity(local_path, expected_sha256):
                        logger.error(f"Downloaded model failed checksum verification: {filename}")
                        logger.info("Suggestion: Check network connection or try manual download")
                        # Clean up corrupted file
                        try:
                            os.remove(local_path)
                        except:
                            pass
                        return None
                    logger.info("Downloaded model passed checksum verification")
                
                # Validate GGUF format
                if not self.validate_gguf(local_path):
                    logger.error(f"Downloaded model failed GGUF validation: {filename}")
                    logger.info("Suggestion: File may be corrupted, try re-downloading")
                    # Clean up corrupted file
                    try:
                        os.remove(local_path)
                    except:
                        pass
                    return None
                logger.info("Downloaded model passed GGUF validation")
            else:
                local_path = str(cached_path)
            
            # Calculate actual checksum
            actual_sha256 = self._calculate_sha256(local_path)
            
            # Get file size
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            
            # Extract quantization from filename (e.g., "model-Q4_0.gguf" -> "Q4_0")
            quantization = self._extract_quantization(filename)
            
            model_info = ModelInfo(
                quantization=quantization,
                filename=filename,
                local_path=local_path,
                sha256=actual_sha256,
                size_mb=round(size_mb, 2)
            )
            
            logger.info(f"Model ready: {filename} ({size_mb:.2f} MB)")
            return model_info
            
        except Exception as e:
            logger.error(f"Unexpected error acquiring model {filename}: {e}", exc_info=True)
            logger.info("Skipping this model and continuing with available models")
            return None
    
    def download_with_retry(self, repo_id: str, filename: str, 
                           max_retries: int = 3) -> Optional[str]:
        """
        Download model with exponential backoff retry.
        
        Implements comprehensive error handling:
        - Network failures with exponential backoff (1s, 2s, 4s)
        - Authentication failures with clear error messages
        - Disk space exhaustion detection
        - Graceful failure after max retries
        
        Args:
            repo_id: Hugging Face repository ID
            filename: Model filename
            max_retries: Maximum number of retry attempts (default: 3)
        
        Returns:
            Path to downloaded model file, or None if download fails
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Download attempt {attempt + 1}/{max_retries}")
                
                # Download to cache directory
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=str(self.cache_dir),
                    token=self.hf_token,
                    resume_download=True
                )
                
                logger.info(f"Download successful: {path}")
                return path
                
            except HfHubHTTPError as e:
                # Handle authentication failures
                if e.response.status_code == 401:
                    logger.error(f"Authentication failed: Invalid or missing HF_TOKEN")
                    logger.info("Suggestion: Set HF_TOKEN environment variable with valid token")
                    return None
                elif e.response.status_code == 403:
                    logger.error(f"Access forbidden: You may not have permission to access {repo_id}")
                    logger.info("Suggestion: Check repository permissions or request access")
                    return None
                elif e.response.status_code == 404:
                    logger.error(f"Model not found: {repo_id}/{filename}")
                    logger.info("Suggestion: Verify repository ID and filename are correct")
                    return None
                else:
                    # Other HTTP errors - retry with backoff
                    if attempt < max_retries - 1:
                        delay = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        logger.warning(
                            f"Download failed (attempt {attempt + 1}/{max_retries}): HTTP {e.response.status_code}"
                        )
                        logger.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        logger.error(f"Download failed after {max_retries} attempts: HTTP {e.response.status_code}")
                        return None
                        
            except RequestException as e:
                # Handle network failures
                if attempt < max_retries - 1:
                    delay = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(
                        f"Network error (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Download failed after {max_retries} attempts due to network errors")
                    logger.info("Suggestion: Check network connection and try again")
                    return None
                    
            except OSError as e:
                # Handle disk space exhaustion
                if "No space left on device" in str(e) or "Disk quota exceeded" in str(e):
                    logger.error(f"Disk space exhausted: {e}")
                    logger.info(f"Suggestion: Free up disk space in {self.cache_dir} or use different cache directory")
                    return None
                else:
                    # Other OS errors
                    logger.error(f"OS error during download: {e}")
                    if attempt < max_retries - 1:
                        delay = 2 ** attempt
                        logger.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        logger.error(f"Download failed after {max_retries} attempts")
                        return None
                        
            except Exception as e:
                logger.error(f"Unexpected error during download: {e}", exc_info=True)
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Download failed after {max_retries} attempts")
                    return None
        
        # Should never reach here, but for safety
        return None
    
    def verify_integrity(self, path: str, expected_sha256: str) -> bool:
        """
        Verify model file integrity using SHA256 checksum.
        
        Args:
            path: Path to model file
            expected_sha256: Expected SHA256 checksum
        
        Returns:
            True if checksum matches, False otherwise
        """
        logger.info(f"Verifying integrity of {path}")
        
        actual_sha256 = self._calculate_sha256(path)
        
        if actual_sha256.lower() == expected_sha256.lower():
            logger.info("Checksum verification passed")
            return True
        else:
            logger.warning(
                f"Checksum mismatch: expected {expected_sha256}, got {actual_sha256}"
            )
            return False
    
    def validate_gguf(self, path: str) -> bool:
        """
        Validate GGUF format by checking magic bytes and header structure.
        
        GGUF format specification:
        - Magic bytes: "GGUF" (0x47475546)
        - Version: uint32
        - Tensor count: uint64
        - Metadata KV count: uint64
        
        Args:
            path: Path to GGUF file
        
        Returns:
            True if valid GGUF format, False otherwise
        """
        logger.info(f"Validating GGUF format: {path}")
        
        try:
            with open(path, 'rb') as f:
                # Read magic bytes (4 bytes)
                magic = f.read(4)
                
                if magic != b'GGUF':
                    logger.warning(f"Invalid magic bytes: {magic}")
                    return False
                
                # Read version (4 bytes, uint32, little-endian)
                version_bytes = f.read(4)
                if len(version_bytes) < 4:
                    logger.warning("File too short to contain version")
                    return False
                
                version = struct.unpack('<I', version_bytes)[0]
                logger.info(f"GGUF version: {version}")
                
                # Read tensor count (8 bytes, uint64, little-endian)
                tensor_count_bytes = f.read(8)
                if len(tensor_count_bytes) < 8:
                    logger.warning("File too short to contain tensor count")
                    return False
                
                tensor_count = struct.unpack('<Q', tensor_count_bytes)[0]
                logger.info(f"Tensor count: {tensor_count}")
                
                # Read metadata KV count (8 bytes, uint64, little-endian)
                metadata_count_bytes = f.read(8)
                if len(metadata_count_bytes) < 8:
                    logger.warning("File too short to contain metadata count")
                    return False
                
                metadata_count = struct.unpack('<Q', metadata_count_bytes)[0]
                logger.info(f"Metadata KV count: {metadata_count}")
                
                # Basic sanity checks
                if version > 100:  # Arbitrary upper bound for version
                    logger.warning(f"Suspicious version number: {version}")
                    return False
                
                if tensor_count == 0:
                    logger.warning("Tensor count is zero")
                    return False
                
                if tensor_count > 10000:  # Arbitrary upper bound
                    logger.warning(f"Suspicious tensor count: {tensor_count}")
                    return False
                
                logger.info("GGUF validation passed")
                return True
                
        except Exception as e:
            logger.error(f"Error validating GGUF format: {e}")
            return False
    
    def _calculate_sha256(self, path: str) -> str:
        """
        Calculate SHA256 checksum using streaming for large files.
        
        Args:
            path: Path to file
        
        Returns:
            SHA256 checksum as hex string
        """
        sha256_hash = hashlib.sha256()
        
        # Read in chunks to handle large files
        chunk_size = 8192  # 8 KB chunks
        
        with open(path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def _check_disk_space(self, filename: str, safety_margin_gb: float = 2.0) -> bool:
        """
        Check if sufficient disk space is available for download.
        
        Estimates required space based on typical model sizes and adds safety margin.
        
        Args:
            filename: Model filename (used to estimate size)
            safety_margin_gb: Additional space to require beyond estimated size (default: 2GB)
        
        Returns:
            True if sufficient space available, False otherwise
        """
        try:
            import shutil
            
            # Get available disk space
            stat = shutil.disk_usage(self.cache_dir)
            available_gb = stat.free / (1024 ** 3)
            
            # Estimate required space based on quantization level
            # Q2_K: ~2-3GB, Q4_0/Q4_K_M: ~4-5GB, Q8_0: ~7-8GB
            estimated_gb = 8.0  # Conservative estimate
            
            if 'q2' in filename.lower():
                estimated_gb = 3.0
            elif 'q4' in filename.lower():
                estimated_gb = 5.0
            elif 'q8' in filename.lower():
                estimated_gb = 8.0
            
            required_gb = estimated_gb + safety_margin_gb
            
            if available_gb < required_gb:
                logger.warning(
                    f"Low disk space: {available_gb:.2f}GB available, "
                    f"~{required_gb:.2f}GB required for {filename}"
                )
                return False
            
            logger.debug(f"Disk space check passed: {available_gb:.2f}GB available")
            return True
            
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
            # Assume sufficient space if check fails
            return True
    
    def _extract_quantization(self, filename: str) -> str:
        """
        Extract quantization level from filename.
        
        Examples:
            "model-Q4_0.gguf" -> "Q4_0"
            "llama-2-7b-q8_0.gguf" -> "Q8_0"
            "model.gguf" -> "unknown"
        
        Args:
            filename: Model filename
        
        Returns:
            Quantization level string
        """
        import re
        
        # Pattern to match quantization formats like Q4_0, Q8_0, Q4_K_M, Q2_K, etc.
        pattern = r'[Qq](\d+)_?([0KkMmSs]*)'
        
        match = re.search(pattern, filename)
        if match:
            # Normalize to uppercase
            quant = f"Q{match.group(1)}"
            if match.group(2):
                quant += f"_{match.group(2).upper()}"
            return quant
        
        return "unknown"
