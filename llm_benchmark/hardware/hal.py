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


class AndroidBackend(HardwareBackend):
    """Hardware backend for Android smartphones."""
    
    def __init__(self, hw_info: HardwareInfo, android_config: Optional[Dict[str, Any]] = None):
        """Initialize AndroidBackend with hardware info and configuration."""
        super().__init__(hw_info)
        self._android_config = None
        self._config_loaded = False
        self._provided_config = android_config
    
    def _load_android_config(self) -> "AndroidConfig":
        """
        Load AndroidConfig from provided config, file, or create default configuration.
        
        Returns:
            AndroidConfig instance
        """
        if not self._config_loaded:
            try:
                from llm_benchmark.android_config import AndroidConfig, create_default_android_config
                
                # First, try to use provided configuration from BenchmarkConfig
                if self._provided_config is not None:
                    logger.info("Using Android configuration from BenchmarkConfig")
                    self._android_config = AndroidConfig.from_dict(self._provided_config)
                    self._config_loaded = True
                    return self._android_config
                
                # Fallback: Try to load from standard config file locations
                from pathlib import Path
                import json
                
                config_paths = [
                    Path("android_config.json"),
                    Path("config/android_config.json"),
                    Path(".config/android_config.json")
                ]
                
                for config_path in config_paths:
                    if config_path.exists():
                        logger.info(f"Loading Android configuration from {config_path}")
                        with open(config_path, 'r') as f:
                            config_dict = json.load(f)
                        self._android_config = AndroidConfig.from_dict(config_dict)
                        break
                
                if self._android_config is None:
                    logger.debug("No Android configuration found, using defaults")
                    self._android_config = create_default_android_config()
                
                self._config_loaded = True
                
            except Exception as e:
                logger.warning(f"Failed to load Android configuration: {e}")
                from llm_benchmark.android_config import create_default_android_config
                self._android_config = create_default_android_config()
                self._config_loaded = True
        
        return self._android_config
    
    def _detect_llama_server_availability(self) -> bool:
        """
        Detect if llama-server binary is available.
        
        Returns:
            True if llama-server binary exists and is executable, False otherwise
        """
        from pathlib import Path
        
        llama_server_paths = [
            Path("~/llama.cpp/build/bin/llama-server").expanduser(),
            Path("/usr/local/bin/llama-server"),
            Path("/usr/bin/llama-server"),
            Path("./llama-server")
        ]
        
        for server_path in llama_server_paths:
            if server_path.exists() and server_path.is_file():
                try:
                    # Check if file is executable
                    import os
                    if os.access(server_path, os.X_OK):
                        logger.debug(f"Found llama-server at {server_path}")
                        return True
                except Exception as e:
                    logger.debug(f"Error checking llama-server at {server_path}: {e}")
        
        logger.debug("llama-server binary not found in standard locations")
        return False
    
    def _should_use_llama_server(self, enable_ablation_studies: bool = False) -> bool:
        """
        Determine whether to use llama-server based on configuration and availability.
        
        Backend selection logic:
        1. Explicit configuration override (use_llama_server_for_ablation)
        2. Ablation mode preference (enable_ablation_studies + Android platform)
        3. Binary availability check
        4. Fallback to llama-cli
        
        Args:
            enable_ablation_studies: Whether ablation studies are enabled
        
        Returns:
            True if llama-server should be used, False for llama-cli
        """
        config = self._load_android_config()
        
        # 1. Explicit configuration override
        if config.use_llama_server_for_ablation is not None:
            use_server = config.use_llama_server_for_ablation
            logger.info(f"Using explicit llama-server configuration: {use_server}")
            
            if use_server and not self._detect_llama_server_availability():
                logger.error("llama-server explicitly requested but not available")
                self._log_llama_server_setup_instructions()
                return False
            
            return use_server
        
        # 2. Automatic selection based on ablation studies
        if enable_ablation_studies and self.hw_info.os_type == "android":
            if self._detect_llama_server_availability():
                logger.info("Using llama-server for Android ablation studies (automatic selection)")
                return True
            else:
                logger.warning("Ablation studies enabled but llama-server not available")
                self._log_ablation_limitation_warning()
                return False
        
        # 3. Default to llama-cli for non-ablation workloads
        logger.debug("Using llama-cli (default for non-ablation workloads)")
        return False
    
    def _log_llama_server_setup_instructions(self):
        """Log instructions for setting up llama-server on Android."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("LLAMA-SERVER SETUP REQUIRED")
        logger.info("=" * 70)
        logger.info("")
        logger.info("To use llama-server for ablation studies, build it alongside llama-cli:")
        logger.info("")
        logger.info("1. Navigate to your llama.cpp directory:")
        logger.info("   cd ~/llama.cpp")
        logger.info("")
        logger.info("2. Build llama-server (if not already built):")
        logger.info("   cmake --build build --config Release --target llama-server")
        logger.info("")
        logger.info("3. Verify the build:")
        logger.info("   ls -lh ~/llama.cpp/build/bin/llama-server")
        logger.info("")
        logger.info("4. Test the server:")
        logger.info("   ~/llama.cpp/build/bin/llama-server --help")
        logger.info("")
        logger.info("=" * 70)
        logger.info("")
    
    def _log_ablation_limitation_warning(self):
        """Log warning about ablation study limitations with llama-cli."""
        logger.warning("")
        logger.warning("=" * 70)
        logger.warning("ABLATION STUDY LIMITATION")
        logger.warning("=" * 70)
        logger.warning("")
        logger.warning("llama-cli cannot disable KV cache, which limits ablation study accuracy.")
        logger.warning("For true cache ablation studies, llama-server is required.")
        logger.warning("")
        logger.warning("Current limitations with llama-cli:")
        logger.warning("  - Cannot disable RAM-based KV cache (--cache-ram 0)")
        logger.warning("  - Cannot disable prompt cache per request")
        logger.warning("  - Control runs will still have some caching effects")
        logger.warning("")
        logger.warning("Results will show relative cache performance but not true 'no cache' baseline.")
        logger.warning("")
        logger.warning("To enable accurate ablation studies, build and configure llama-server.")
        logger.warning("")
        logger.warning("=" * 70)
        logger.warning("")
    
    def get_llama_config(self) -> Dict[str, Any]:
        """
        Return optimized configuration for Android.
        
        Android-specific optimizations:
        - Conservative thread count (leave cores for system)
        - No mlock (limited RAM on mobile)
        - Smaller batch size for memory efficiency
        
        Returns:
            Configuration dictionary for llama-cpp-python
        """
        # Use fewer threads on mobile to avoid thermal issues
        # Leave at least 2 cores for system
        thread_count = max(2, self.hw_info.cpu_cores - 2)
        
        config = {
            "n_gpu_layers": 0,  # CPU-only (most Android devices don't support GPU acceleration for llama.cpp)
            "use_mlock": False,  # Don't lock memory on mobile
            "n_threads": thread_count,  # Conservative thread count
            "n_batch": 256,  # Smaller batch size for memory efficiency
        }
        
        logger.info(f"Android backend configuration: {config}")
        logger.info(f"Using {thread_count}/{self.hw_info.cpu_cores} CPU cores")
        return config
    
    def get_metrics_collector(self) -> "MetricsCollector":
        """
        Return metrics collector for Android platform.
        
        Returns:
            MetricsCollector instance configured for Android (CPU-only metrics)
        """
        from llm_benchmark.metrics import MetricsCollector
        return MetricsCollector(self.hw_info)
    
    def load_model_safe(self, model_path: str, **kwargs) -> Any:
        """
        Load model with Android-specific error handling and backend selection.
        
        Implements automatic backend selection between llama-cli and llama-server
        based on configuration and ablation study requirements.
        
        Backend selection logic:
        1. Check explicit configuration (use_llama_server_for_ablation)
        2. Check if ablation studies are enabled (prefer llama-server)
        3. Check binary availability
        4. Fall back to llama-cli with appropriate warnings
        
        Args:
            model_path: Path to GGUF model file
            **kwargs: Additional arguments for model loading
                     enable_ablation_studies: Whether ablation studies are enabled
        
        Returns:
            Loaded model instance (NativeLlamaCpp or NativeLlamaServer), or None if loading fails
        """
        from pathlib import Path
        
        # Extract ablation studies flag from kwargs
        enable_ablation_studies = kwargs.get('enable_ablation_studies', False)
        
        # Determine which backend to use
        use_llama_server = self._should_use_llama_server(enable_ablation_studies)
        
        if use_llama_server:
            return self._load_model_with_llama_server(model_path, **kwargs)
        else:
            return self._load_model_with_llama_cli(model_path, **kwargs)
    
    def _load_model_with_llama_server(self, model_path: str, **kwargs) -> Any:
        """
        Load model using llama-server backend.
        
        Automatically configures cache settings based on ablation scenario when
        enable_ablation_studies is True. Uses ABLATION_CACHE_CONFIG mapping
        to ensure consistent cache configuration.
        
        Args:
            model_path: Path to GGUF model file
            **kwargs: Additional arguments for model loading
                     enable_ablation_studies: Whether ablation studies are enabled
                     ablation_scenario: Specific ablation scenario name (optional)
                     cache_mode: Explicit cache mode override (optional)
        
        Returns:
            NativeLlamaServer instance or None if loading fails
        """
        from pathlib import Path
        
        # Check if llama-server binary exists
        llama_server = Path("~/llama.cpp/build/bin/llama-server").expanduser()
        
        if not llama_server.exists():
            logger.error("llama-server binary not found at ~/llama.cpp/build/bin/llama-server")
            self._log_llama_server_setup_instructions()
            return None
        
        logger.info("Using llama-server backend (Android)")
        
        # Validate GGUF format before loading
        if not self._validate_gguf_format(model_path):
            logger.error(f"Invalid GGUF format: {model_path}")
            logger.info("Suggestion: File may be corrupted, try re-downloading")
            return None
        
        # Check available memory before loading (critical on mobile)
        if not self._check_available_memory(model_path, kwargs.get('n_ctx', 2048)):
            logger.error(f"Insufficient RAM to load model: {model_path}")
            logger.info("Suggestions for Android:")
            logger.info("  - Use smaller quantization (Q2_K or Q4_0)")
            logger.info("  - Reduce context_size to 512 or 1024")
            logger.info("  - Close background apps to free memory")
            logger.info("  - Use a smaller model (< 1B parameters)")
            return None
        
        # Get configuration
        config = self.get_llama_config()
        config.update(kwargs)
        
        # Load Android configuration for llama-server settings
        android_config = self._load_android_config()
        
        # Determine cache mode based on ablation scenario or explicit configuration
        cache_mode = self._determine_cache_mode_for_ablation(kwargs, android_config)
        
        try:
            from llm_benchmark.inference.native_llama_server import NativeLlamaServer
            
            logger.info("Loading model with llama-server (this may take a while)...")
            logger.info(f"Cache mode: {cache_mode}")
            
            llm = NativeLlamaServer(
                model_path=model_path,
                n_ctx=config.get('n_ctx', 2048),
                n_threads=config.get('n_threads', 4),
                n_batch=config.get('n_batch', 512),
                cache_mode=cache_mode,
                llama_server_path=str(llama_server),
                host=android_config.llama_server_host,
                port=android_config.llama_server_port
            )
            logger.info(f"Successfully loaded model with llama-server: {model_path}")
            
            # Store model instance for memory measurement (Android workaround)
            self._last_loaded_model = llm
            
            return llm
            
        except Exception as e:
            logger.error(f"Failed to load model with llama-server: {e}", exc_info=True)
            logger.info("llama-server troubleshooting:")
            logger.info("  - Ensure llama-server is built: cd ~/llama.cpp && ls build/bin/llama-server")
            logger.info("  - Check if port is available (default: 8080)")
            logger.info("  - Ensure you have enough free RAM (check with 'free -h')")
            logger.info("  - Try a smaller model or lower quantization")
            
            # Fall back to llama-cli if llama-server fails
            logger.info("Falling back to llama-cli...")
            return self._load_model_with_llama_cli(model_path, **kwargs)
    
    def _determine_cache_mode_for_ablation(self, kwargs: Dict[str, Any], android_config) -> str:
        """
        Determine cache mode based on ablation scenario configuration.
        
        Uses ABLATION_CACHE_CONFIG mapping when ablation studies are enabled
        and an ablation scenario is specified. Falls back to explicit cache_mode
        or default configuration.
        
        Args:
            kwargs: Keyword arguments passed to load_model_safe
            android_config: AndroidConfig instance
        
        Returns:
            Cache mode string ("none", "ram_only", "disk_only", "both")
        """
        # Check for explicit cache_mode override
        if 'cache_mode' in kwargs:
            cache_mode = kwargs['cache_mode']
            logger.debug(f"Using explicit cache_mode: {cache_mode}")
            return cache_mode
        
        # Check if ablation studies are enabled and scenario is specified
        enable_ablation_studies = kwargs.get('enable_ablation_studies', False)
        ablation_scenario = kwargs.get('ablation_scenario')
        
        if enable_ablation_studies and ablation_scenario:
            try:
                from llm_benchmark.inference.native_llama_server import ABLATION_CACHE_CONFIG
                
                if ablation_scenario in ABLATION_CACHE_CONFIG:
                    scenario_config = ABLATION_CACHE_CONFIG[ablation_scenario]
                    cache_mode = scenario_config["cache_mode"].value
                    
                    logger.info(f"Ablation scenario '{ablation_scenario}' detected")
                    logger.info(f"Using cache configuration: {scenario_config['description']}")
                    logger.info(f"Cache mode: {cache_mode}")
                    
                    return cache_mode
                else:
                    logger.warning(
                        f"Unknown ablation scenario '{ablation_scenario}', "
                        f"using default cache mode"
                    )
            except ImportError as e:
                logger.warning(f"Could not import ABLATION_CACHE_CONFIG: {e}")
        
        # Fall back to Android configuration default
        cache_mode = android_config.cache_mode.value
        logger.debug(f"Using default cache_mode from Android config: {cache_mode}")
        return cache_mode
    
    def _load_model_with_llama_cli(self, model_path: str, **kwargs) -> Any:
        """
        Load model using llama-cli backend (original implementation).
        
        Args:
            model_path: Path to GGUF model file
            **kwargs: Additional arguments for model loading
        
        Returns:
            NativeLlamaCpp instance or None if loading fails
        """
        from pathlib import Path
        
        # Check if native llama.cpp is available
        llama_cli = Path("~/llama.cpp/build/bin/llama-cli").expanduser()
        
        if llama_cli.exists():
            logger.info("Using native llama.cpp CLI (Android)")
            
            # Validate GGUF format before loading
            if not self._validate_gguf_format(model_path):
                logger.error(f"Invalid GGUF format: {model_path}")
                logger.info("Suggestion: File may be corrupted, try re-downloading")
                return None
            
            # Check available memory before loading (critical on mobile)
            if not self._check_available_memory(model_path, kwargs.get('n_ctx', 2048)):
                logger.error(f"Insufficient RAM to load model: {model_path}")
                logger.info("Suggestions for Android:")
                logger.info("  - Use smaller quantization (Q2_K or Q4_0)")
                logger.info("  - Reduce context_size to 512 or 1024")
                logger.info("  - Close background apps to free memory")
                logger.info("  - Use a smaller model (< 1B parameters)")
                return None
            
            # Get configuration
            config = self.get_llama_config()
            config.update(kwargs)
            
            try:
                from llm_benchmark.inference.native_llama import NativeLlamaCpp
                
                logger.info("Loading model with native llama.cpp (this may take a while)...")
                llm = NativeLlamaCpp(
                    model_path=model_path,
                    n_ctx=config.get('n_ctx', 2048),
                    n_threads=config.get('n_threads', 4),
                    n_batch=config.get('n_batch', 512),
                    llama_cli_path=str(llama_cli)
                )
                logger.info(f"Successfully loaded model: {model_path}")
                
                # Store model instance for memory measurement (Android workaround)
                self._last_loaded_model = llm
                
                return llm
                
            except Exception as e:
                logger.error(f"Failed to load model with native llama.cpp: {e}", exc_info=True)
                logger.info("Android troubleshooting:")
                logger.info("  - Ensure llama.cpp is built: cd ~/llama.cpp && ls build/bin/llama-cli")
                logger.info("  - Ensure you have enough free RAM (check with 'free -h')")
                logger.info("  - Try a smaller model or lower quantization")
                return None
        
        else:
            # Native llama.cpp not found, provide instructions
            logger.error("Native llama.cpp not found at ~/llama.cpp/build/bin/llama-cli")
            logger.info("")
            logger.info("=" * 70)
            logger.info("ANDROID SETUP REQUIRED")
            logger.info("=" * 70)
            logger.info("")
            logger.info("llama-cpp-python doesn't support Android. You need to build native llama.cpp:")
            logger.info("")
            logger.info("1. Install build dependencies:")
            logger.info("   pkg install git cmake clang")
            logger.info("")
            logger.info("2. Clone and build llama.cpp:")
            logger.info("   cd ~")
            logger.info("   git clone https://github.com/ggerganov/llama.cpp")
            logger.info("   cd llama.cpp")
            logger.info("   cmake -B build -DCMAKE_BUILD_TYPE=Release")
            logger.info("   cmake --build build --config Release -j4")
            logger.info("")
            logger.info("3. Verify the build:")
            logger.info("   ls -lh ~/llama.cpp/build/bin/llama-cli")
            logger.info("")
            logger.info("4. Test with a model:")
            logger.info("   ~/llama.cpp/build/bin/llama-cli -m <model.gguf> -p 'Hello' -n 10")
            logger.info("")
            logger.info("5. Then run the benchmark again")
            logger.info("")
            logger.info("=" * 70)
            logger.info("")
            
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
        
        More conservative on Android due to limited RAM and system overhead.
        
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
            
            # Estimate memory requirements (more conservative on mobile):
            # - Model weights: file size
            # - Context buffer: ~context_size * 2 bytes per token * safety factor
            # - Overhead: ~30% of model size (higher on mobile)
            context_mb = (context_size * 2 * 2.0) / (1024 * 1024)  # 2x safety factor
            overhead_mb = model_size_mb * 0.3  # 30% overhead
            required_mb = model_size_mb + context_mb + overhead_mb
            
            # Get available memory
            mem = psutil.virtual_memory()
            available_mb = mem.available / (1024 * 1024)
            
            # More conservative on mobile: require 70% headroom
            usable_mb = available_mb * 0.7
            
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
    
    def check_thermal_state(self, threshold_c: float = 70.0) -> tuple:
        """
        Check if device is thermally throttled.
        
        Reads from Android thermal zones to detect overheating.
        
        Args:
            threshold_c: Temperature threshold in Celsius for throttling detection
        
        Returns:
            Tuple of (is_throttled, current_temp) where is_throttled is True
            if temperature exceeds threshold, and current_temp is the highest
            temperature in Celsius (or None if unavailable)
        """
        try:
            from pathlib import Path
            
            thermal_dir = Path("/sys/class/thermal")
            if not thermal_dir.exists():
                return False, None
            
            max_temp = 0.0
            
            # Read all thermal zones
            for zone_dir in thermal_dir.glob("thermal_zone*"):
                temp_file = zone_dir / "temp"
                if temp_file.exists():
                    try:
                        with open(temp_file, 'r') as f:
                            # Temperature is in millidegrees Celsius
                            temp_millic = int(f.read().strip())
                            temp_c = temp_millic / 1000.0
                            max_temp = max(max_temp, temp_c)
                    except Exception as e:
                        logger.debug(f"Error reading {temp_file}: {e}")
            
            if max_temp == 0.0:
                return False, None
            
            is_throttled = max_temp > threshold_c
            
            if is_throttled:
                logger.warning(
                    f"Device temperature ({max_temp:.1f}°C) exceeds threshold ({threshold_c}°C)"
                )
            
            return is_throttled, max_temp
            
        except Exception as e:
            logger.debug(f"Failed to check thermal state: {e}")
            return False, None
    
    def optimize_for_inference(self) -> None:
        """Apply Android-specific optimizations."""
        logger.info("Applying Android optimizations...")
        
        # Log recommendations
        logger.info("Android optimization tips:")
        logger.info("  - Keep device plugged in during benchmarking")
        logger.info("  - Close background apps to free RAM")
        logger.info("  - Enable performance mode if available")
        logger.info("  - Keep device cool to avoid thermal throttling")


def create_backend(hw_info: HardwareInfo, android_config: Optional[Dict[str, Any]] = None) -> HardwareBackend:
    """
    Create appropriate hardware backend based on detected platform.
    
    Args:
        hw_info: Hardware information from detector
        android_config: Optional Android configuration dictionary
    
    Returns:
        Hardware backend instance
    """
    if hw_info.os_type == "jetson_xavier_nx":
        return JetsonBackend(hw_info)
    elif hw_info.os_type == "android":
        return AndroidBackend(hw_info, android_config)
    else:
        return X86Backend(hw_info)
