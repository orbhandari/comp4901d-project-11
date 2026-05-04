"""
Hardware Abstraction Layer (HAL).

Provides unified interface for platform-specific operations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING

from llm_benchmark.models import HardwareInfo

if TYPE_CHECKING:
    from llm_benchmark.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class HardwareBackend(ABC):
    """Abstract base class for hardware backends."""
    
    def __init__(self, hw_info: HardwareInfo):
        """
        Initialize hardware backend.
        
        Args:
            hw_info: Hardware information from detector
        """
        self.hw_info = hw_info
    
    @abstractmethod
    def get_llama_config(self) -> Dict[str, Any]:
        """
        Return llama-cpp-python configuration for this platform.
        
        Returns:
            Dictionary of configuration parameters
        """
        pass
    
    @abstractmethod
    def get_metrics_collector(self) -> "MetricsCollector":
        """
        Return platform-specific metrics collector.
        
        Returns:
            MetricsCollector instance configured for this platform
        """
        pass
    
    @abstractmethod
    def optimize_for_inference(self) -> None:
        """Apply platform-specific optimizations."""
        pass


class X86Backend(HardwareBackend):
    """Hardware backend for x86 Linux systems."""
    
    def get_llama_config(self) -> Dict[str, Any]:
        """
        Return CPU-only configuration for x86.
        
        Returns:
            Configuration dictionary for llama-cpp-python
        """
        config = {
            "n_gpu_layers": 0,  # CPU-only
            "use_mlock": True,  # Lock model in RAM
            "n_threads": self.hw_info.cpu_cores,  # Use all cores
        }
        
        logger.info(f"X86 backend configuration: {config}")
        return config
    
    def get_metrics_collector(self) -> "MetricsCollector":
        """
        Return metrics collector for x86 platform.
        
        Returns:
            MetricsCollector instance configured for x86 (CPU-only metrics)
        """
        from llm_benchmark.metrics import MetricsCollector
        return MetricsCollector(self.hw_info)
    
    def load_model_safe(self, model_path: str, **kwargs) -> Any:
        """
        Load model with comprehensive error handling.
        
        Implements error handling for:
        - GGUF format validation before loading
        - Available memory checking before loading
        - Insufficient RAM with suggestions (reduce context_size, use smaller quantization)
        - Corrupted GGUF files with validation errors
        
        Args:
            model_path: Path to GGUF model file
            **kwargs: Additional arguments for Llama constructor
        
        Returns:
            Loaded Llama model instance, or None if loading fails
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            logger.error("llama-cpp-python not installed")
            logger.info("Suggestion: Install with 'pip install llama-cpp-python'")
            return None
        
        # Validate GGUF format before loading
        if not self._validate_gguf_format(model_path):
            logger.error(f"Invalid GGUF format: {model_path}")
            logger.info("Suggestion: File may be corrupted, try re-downloading")
            return None
        
        # Check available memory before loading
        if not self._check_available_memory(model_path, kwargs.get('n_ctx', 2048)):
            logger.error(f"Insufficient RAM to load model: {model_path}")
            logger.info("Suggestions:")
            logger.info("  - Reduce context_size (current: {})".format(kwargs.get('n_ctx', 2048)))
            logger.info("  - Use smaller quantization (Q4_0 or Q2_K)")
            logger.info("  - Close other applications to free memory")
            return None
        
        # Get initial configuration
        config = self.get_llama_config()
        
        # Merge with user-provided kwargs (user kwargs take precedence)
        config.update(kwargs)
        config["model_path"] = model_path
        
        # Try loading
        try:
            llm = Llama(**config)
            logger.info(f"Successfully loaded model: {model_path}")
            return llm
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            return None
    
    def _validate_gguf_format(self, model_path: str) -> bool:
        """
        Validate GGUF format by checking magic bytes and header structure.
        
        Args:
            model_path: Path to GGUF file
        
        Returns:
            True if valid GGUF format, False otherwise
        """
        try:
            import struct
            
            with open(model_path, 'rb') as f:
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
                
                # Basic sanity check
                if version > 100:  # Arbitrary upper bound
                    logger.warning(f"Suspicious version number: {version}")
                    return False
                
                logger.debug(f"GGUF validation passed (version {version})")
                return True
                
        except Exception as e:
            logger.error(f"Error validating GGUF format: {e}")
            return False
    
    def _check_available_memory(self, model_path: str, context_size: int) -> bool:
        """
        Check if sufficient RAM is available to load model.
        
        Estimates required memory based on model size and context size.
        
        Args:
            model_path: Path to model file
            context_size: Context size for model
        
        Returns:
            True if sufficient memory available, False otherwise
        """
        try:
            import psutil
            import os
            
            # Get model file size
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            
            # Estimate memory requirements:
            # - Model weights: file size
            # - Context buffer: ~context_size * 2 bytes per token * safety factor
            # - Overhead: ~20% of model size
            context_mb = (context_size * 2 * 1.5) / (1024 * 1024)  # 1.5x safety factor
            overhead_mb = model_size_mb * 0.2
            required_mb = model_size_mb + context_mb + overhead_mb
            
            # Get available memory
            mem = psutil.virtual_memory()
            available_mb = mem.available / (1024 * 1024)
            
            # Require 80% headroom (only use 80% of available memory)
            usable_mb = available_mb * 0.8
            
            if required_mb > usable_mb:
                logger.warning(
                    f"Insufficient RAM: need ~{required_mb:.0f}MB, "
                    f"have {available_mb:.0f}MB available (usable: {usable_mb:.0f}MB)"
                )
                return False
            
            logger.debug(
                f"Memory check passed: need ~{required_mb:.0f}MB, "
                f"have {available_mb:.0f}MB available"
            )
            return True
            
        except Exception as e:
            logger.warning(f"Could not check available memory: {e}")
            # Assume sufficient memory if check fails
            return True
    
    def optimize_for_inference(self) -> None:
        """Apply x86-specific optimizations."""
        # Could set CPU affinity, governor, etc.
        logger.info("Applying x86 optimizations...")
        pass


class JetsonBackend(HardwareBackend):
    """Hardware backend for NVIDIA Jetson Xavier NX."""
    
    def get_llama_config(self) -> Dict[str, Any]:
        """
        Return GPU-accelerated configuration for Jetson.
        
        Returns:
            Configuration dictionary for llama-cpp-python
        """
        gpu_layers = self._calculate_gpu_layers()
        
        config = {
            "n_gpu_layers": gpu_layers,
            "use_mlock": False,  # Limited RAM on Jetson
            "n_threads": 4,  # Leave cores for system
        }
        
        logger.info(f"Jetson backend configuration: {config}")
        return config
    
    def get_metrics_collector(self) -> "MetricsCollector":
        """
        Return metrics collector for Jetson platform.
        
        Returns:
            MetricsCollector instance configured for Jetson (includes GPU metrics)
        """
        from llm_benchmark.metrics import MetricsCollector
        return MetricsCollector(self.hw_info)
    
    def _calculate_gpu_layers(self) -> int:
        """
        Calculate optimal GPU layer count based on available GPU memory.
        
        Uses heuristic: ~100MB per layer for 8B model, 80% GPU memory utilization.
        
        Returns:
            Number of layers to offload to GPU
        """
        if not self.hw_info.has_gpu or self.hw_info.gpu_memory_gb is None:
            logger.warning("GPU not available, falling back to CPU-only")
            return 0
        
        # Heuristic: ~100MB per layer for 8B model
        # Use 80% of GPU memory to leave headroom
        available_mb = self.hw_info.gpu_memory_gb * 1024 * 0.8
        gpu_layers = int(available_mb / 100)
        
        logger.info(f"Calculated {gpu_layers} GPU layers for {self.hw_info.gpu_memory_gb:.2f} GB GPU memory")
        return gpu_layers
    
    def load_model_safe(self, model_path: str, **kwargs) -> Any:
        """
        Load model with comprehensive error handling.
        
        Implements error handling for:
        - GGUF format validation before loading
        - Available memory checking before loading
        - Insufficient RAM with suggestions (reduce context_size, use smaller quantization)
        - Corrupted GGUF files with validation errors
        - Missing CUDA libraries with fallback to CPU
        - GPU memory exhaustion with automatic fallback
        
        Args:
            model_path: Path to GGUF model file
            **kwargs: Additional arguments for Llama constructor
        
        Returns:
            Loaded Llama model instance, or None if loading fails
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            logger.error("llama-cpp-python not installed")
            logger.info("Suggestion: Install with 'pip install llama-cpp-python'")
            return None
        
        # Validate GGUF format before loading
        if not self._validate_gguf_format(model_path):
            logger.error(f"Invalid GGUF format: {model_path}")
            logger.info("Suggestion: File may be corrupted, try re-downloading")
            return None
        
        # Check available memory before loading
        if not self._check_available_memory(model_path, kwargs.get('n_ctx', 2048)):
            logger.error(f"Insufficient RAM to load model: {model_path}")
            logger.info("Suggestions:")
            logger.info("  - Reduce context_size (current: {})".format(kwargs.get('n_ctx', 2048)))
            logger.info("  - Use smaller quantization (Q4_0 or Q2_K)")
            logger.info("  - Close other applications to free memory")
            return None
        
        # Get initial configuration
        config = self.get_llama_config()
        
        # Merge with user-provided kwargs (user kwargs take precedence)
        config.update(kwargs)
        config["model_path"] = model_path
        
        # Try loading with GPU fallback
        try:
            llm = self.load_model_with_gpu_fallback(model_path, **kwargs)
            return llm
        except RuntimeError as e:
            # Check for missing CUDA libraries
            if "CUDA" in str(e) and "not found" in str(e).lower():
                logger.error(f"Missing CUDA libraries: {e}")
                logger.info("Suggestion: Install CUDA toolkit or use CPU-only mode")
                logger.info("Attempting CPU-only fallback...")
                
                try:
                    # Force CPU-only mode
                    config["n_gpu_layers"] = 0
                    llm = Llama(**config)
                    logger.info("Successfully loaded in CPU-only mode")
                    return llm
                except Exception as cpu_error:
                    logger.error(f"CPU-only fallback also failed: {cpu_error}")
                    return None
            else:
                logger.error(f"Failed to load model: {e}")
                return None
        except Exception as e:
            logger.error(f"Unexpected error loading model: {e}", exc_info=True)
            return None
    
    def _validate_gguf_format(self, model_path: str) -> bool:
        """
        Validate GGUF format by checking magic bytes and header structure.
        
        Args:
            model_path: Path to GGUF file
        
        Returns:
            True if valid GGUF format, False otherwise
        """
        try:
            import struct
            
            with open(model_path, 'rb') as f:
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
                
                # Basic sanity check
                if version > 100:  # Arbitrary upper bound
                    logger.warning(f"Suspicious version number: {version}")
                    return False
                
                logger.debug(f"GGUF validation passed (version {version})")
                return True
                
        except Exception as e:
            logger.error(f"Error validating GGUF format: {e}")
            return False
    
    def _check_available_memory(self, model_path: str, context_size: int) -> bool:
        """
        Check if sufficient RAM is available to load model.
        
        Estimates required memory based on model size and context size.
        
        Args:
            model_path: Path to model file
            context_size: Context size for model
        
        Returns:
            True if sufficient memory available, False otherwise
        """
        try:
            import psutil
            import os
            
            # Get model file size
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            
            # Estimate memory requirements:
            # - Model weights: file size
            # - Context buffer: ~context_size * 2 bytes per token * safety factor
            # - Overhead: ~20% of model size
            context_mb = (context_size * 2 * 1.5) / (1024 * 1024)  # 1.5x safety factor
            overhead_mb = model_size_mb * 0.2
            required_mb = model_size_mb + context_mb + overhead_mb
            
            # Get available memory
            mem = psutil.virtual_memory()
            available_mb = mem.available / (1024 * 1024)
            
            # Require 80% headroom (only use 80% of available memory)
            usable_mb = available_mb * 0.8
            
            if required_mb > usable_mb:
                logger.warning(
                    f"Insufficient RAM: need ~{required_mb:.0f}MB, "
                    f"have {available_mb:.0f}MB available (usable: {usable_mb:.0f}MB)"
                )
                return False
            
            logger.debug(
                f"Memory check passed: need ~{required_mb:.0f}MB, "
                f"have {available_mb:.0f}MB available"
            )
            return True
            
        except Exception as e:
            logger.warning(f"Could not check available memory: {e}")
            # Assume sufficient memory if check fails
            return True
    
    def load_model_with_gpu_fallback(self, model_path: str, **kwargs) -> Any:
        """
        Load model with GPU memory exhaustion handling and automatic fallback.
        
        Implements progressive layer reduction on GPU OOM errors, with final
        fallback to CPU-only mode. This ensures robust model loading even when
        GPU memory is constrained or fragmented.
        
        Args:
            model_path: Path to GGUF model file
            **kwargs: Additional arguments for Llama constructor
        
        Returns:
            Loaded Llama model instance
        
        Raises:
            RuntimeError: If model loading fails even in CPU-only mode
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError("llama-cpp-python not installed")
        
        # Get initial configuration
        config = self.get_llama_config()
        initial_gpu_layers = config["n_gpu_layers"]
        
        # Merge with user-provided kwargs (user kwargs take precedence)
        config.update(kwargs)
        config["model_path"] = model_path
        
        # Track current GPU layer count for fallback logic
        gpu_layers = initial_gpu_layers
        
        # Attempt to load with progressive fallback
        while gpu_layers >= 0:
            try:
                config["n_gpu_layers"] = gpu_layers
                
                logger.info(f"Attempting to load model with {gpu_layers} GPU layers...")
                llm = Llama(**config)
                
                if gpu_layers < initial_gpu_layers:
                    logger.warning(
                        f"Model loaded with reduced GPU layers: {gpu_layers} "
                        f"(initial: {initial_gpu_layers})"
                    )
                else:
                    logger.info(f"Model loaded successfully with {gpu_layers} GPU layers")
                
                return llm
                
            except RuntimeError as e:
                error_msg = str(e).lower()
                
                # Check for GPU memory errors
                if "cuda out of memory" in error_msg or "out of memory" in error_msg:
                    if gpu_layers > 0:
                        # Reduce GPU layers by 10 and retry
                        old_layers = gpu_layers
                        gpu_layers = max(0, gpu_layers - 10)
                        
                        logger.warning(
                            f"GPU OOM detected, reducing layers from {old_layers} "
                            f"to {gpu_layers} and retrying..."
                        )
                        
                        # Force garbage collection to free GPU memory
                        import gc
                        gc.collect()
                        
                        # Try CUDA memory cleanup if available
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                logger.debug("Cleared CUDA cache")
                        except ImportError:
                            pass
                        
                        continue
                    else:
                        # Already at CPU-only, re-raise the error
                        logger.error("Model loading failed even in CPU-only mode")
                        raise RuntimeError(
                            f"Failed to load model even with CPU-only mode: {e}"
                        ) from e
                else:
                    # Non-memory error, re-raise immediately
                    logger.error(f"Model loading failed with non-memory error: {e}")
                    raise
        
        # Should never reach here, but just in case
        raise RuntimeError("Model loading failed after all fallback attempts")
    
    def get_gpu_temperature(self) -> Optional[float]:
        """
        Get current GPU temperature in Celsius.
        
        Uses NVML to query GPU temperature sensor. This is useful for
        monitoring thermal conditions during inference.
        
        Returns:
            GPU temperature in Celsius, or None if unavailable
        """
        if not self.hw_info.has_gpu:
            return None
        
        try:
            import pynvml
            
            # Initialize NVML if not already done
            try:
                pynvml.nvmlInit()
            except pynvml.NVMLError:
                # Already initialized
                pass
            
            # Get GPU handle
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Query temperature
            temp = pynvml.nvmlDeviceGetTemperature(
                handle,
                pynvml.NVML_TEMPERATURE_GPU
            )
            
            return float(temp)
            
        except Exception as e:
            logger.debug(f"Failed to read GPU temperature: {e}")
            return None
    
    def get_gpu_power_consumption(self) -> Optional[float]:
        """
        Get current GPU power consumption in watts.
        
        Uses NVML to query GPU power draw. This is useful for monitoring
        energy efficiency during inference.
        
        Returns:
            Power consumption in watts, or None if unavailable
        """
        if not self.hw_info.has_gpu:
            return None
        
        try:
            import pynvml
            
            # Initialize NVML if not already done
            try:
                pynvml.nvmlInit()
            except pynvml.NVMLError:
                # Already initialized
                pass
            
            # Get GPU handle
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Query power usage (returns milliwatts)
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_watts = power_mw / 1000.0
            
            return power_watts
            
        except Exception as e:
            logger.debug(f"Failed to read GPU power consumption: {e}")
            return None
    
    def check_thermal_state(self, threshold_c: float = 85.0) -> tuple:
        """
        Check if GPU is thermally throttled.
        
        Compares current GPU temperature against a threshold to determine
        if the system is in a thermally constrained state.
        
        Args:
            threshold_c: Temperature threshold in Celsius for throttling detection
        
        Returns:
            Tuple of (is_throttled, current_temp) where is_throttled is True
            if temperature exceeds threshold, and current_temp is the GPU
            temperature in Celsius (or None if unavailable)
        """
        temp = self.get_gpu_temperature()
        
        if temp is None:
            return False, None
        
        is_throttled = temp > threshold_c
        
        if is_throttled:
            logger.warning(
                f"GPU temperature ({temp}°C) exceeds threshold ({threshold_c}°C)"
            )
        
        return is_throttled, temp
    
    def optimize_for_inference(self) -> None:
        """Apply Jetson-specific optimizations."""
        # Could set power mode, fan speed, etc.
        logger.info("Applying Jetson optimizations...")
        pass


def create_backend(hw_info: HardwareInfo) -> HardwareBackend:
    """
    Create appropriate hardware backend based on detected platform.
    
    Args:
        hw_info: Hardware information from detector
    
    Returns:
        Hardware backend instance
    """
    if hw_info.os_type == "jetson_xavier_nx":
        return JetsonBackend(hw_info)
    else:
        return X86Backend(hw_info)
