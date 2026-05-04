"""
Quantization Profiler for systematic performance measurement across quantization levels.

This module provides the QuantizationProfiler class which:
- Measures baseline memory before model load
- Times model loading
- Performs warmup inference before measurement
- Uses streaming inference to capture TTFT accurately
- Executes identical test prompts across all quantization levels
- Enforces garbage collection between quantization tests
- Generates comparison matrix showing all metrics across quantization levels
"""

import gc
import logging
import statistics
import time
from typing import Dict, List, Any

import psutil

from llm_benchmark.hardware.hal import HardwareBackend
from llm_benchmark.inference.native_llama import NativeLlamaCpp
from llm_benchmark.metrics.collector import MetricsCollector
from llm_benchmark.models import QuantizationResult

logger = logging.getLogger(__name__)


class QuantizationProfiler:
    """
    Systematically measures performance across quantization levels.
    
    Implements controlled testing with identical prompts, warmup runs,
    and garbage collection to ensure fair comparison between quantization levels.
    """
    
    def __init__(self, backend: HardwareBackend, metrics_collector: MetricsCollector):
        """
        Initialize quantization profiler.
        
        Args:
            backend: Hardware backend providing platform-specific configuration
            metrics_collector: Metrics collector for inference measurements
        """
        self.backend = backend
        self.metrics = metrics_collector
        self.process = psutil.Process()
        
        logger.info("QuantizationProfiler initialized")
    
    def _is_android_platform(self, llm: Any) -> bool:
        """
        Detect if the model instance is using Android native llama.cpp.
        
        Args:
            llm: Model instance to check
        
        Returns:
            True if using NativeLlamaCpp (Android), False otherwise
        """
        return isinstance(llm, NativeLlamaCpp)
    
    def _get_total_memory_mb(self) -> float:
        """
        Get total memory including subprocesses (for Android).
        
        For Android with NativeLlamaCpp, the native llama-cli runs as a subprocess
        and its memory must be included in the total. For other platforms using
        llama-cpp-python, the model runs in-process so only the parent process
        memory is needed.
        
        Returns:
            Total memory in MB (parent process + all child processes)
        """
        total_rss = self.process.memory_info().rss
        
        # Add subprocess memory (for Android native llama.cpp)
        try:
            for child in self.process.children(recursive=True):
                try:
                    total_rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Child process may have terminated or we don't have permission
                    pass
        except (psutil.AccessDenied, PermissionError):
            # On Android/Termux, psutil.children() fails due to /proc/stat restrictions
            # Use 'ps' command as fallback - it doesn't require root and works reliably
            logger.debug("psutil.children() failed due to permission restrictions. Using 'ps' command fallback...")
            
            # Try to get subprocess memory using ps command
            try:
                # Check if we have a NativeLlamaCpp instance with a subprocess PID
                if hasattr(self.backend, '_last_loaded_model'):
                    llm = self.backend._last_loaded_model
                    if hasattr(llm, 'last_subprocess_pid') and llm.last_subprocess_pid:
                        subprocess_memory = self._read_memory_from_ps(llm.last_subprocess_pid)
                        if subprocess_memory > 0:
                            total_rss += subprocess_memory
                            logger.debug(f"Added subprocess memory: {subprocess_memory / (1024 * 1024):.2f} MB (PID: {llm.last_subprocess_pid})")
                        else:
                            logger.debug(f"Subprocess PID {llm.last_subprocess_pid} not found in ps output")
            except Exception as e:
                logger.debug(f"ps command fallback failed: {e}")
            
            # Log warning only if we couldn't get subprocess memory
            if total_rss == self.process.memory_info().rss:
                logger.warning(
                    "Unable to measure subprocess memory. "
                    "Memory measurement will be incomplete for subprocess-based inference."
                )
        
        return total_rss / (1024 * 1024)
    
    def _read_memory_from_ps(self, pid: int) -> int:
        """
        Read memory usage using the 'ps' command.
        
        This is a reliable fallback for Android/Termux where psutil.children() fails
        due to /proc/stat permission restrictions. The 'ps' command doesn't require
        root access and works on all Unix-like systems.
        
        Args:
            pid: Process ID to read memory for
        
        Returns:
            RSS memory in bytes, or 0 if unable to read
        """
        try:
            import subprocess
            
            # Use ps to get RSS in KB for the specific PID
            # Format: ps -o rss= -p <pid>
            # The '=' removes the header, giving us just the value
            result = subprocess.run(
                ['ps', '-o', 'rss=', '-p', str(pid)],
                capture_output=True,
                text=True,
                timeout=1
            )
            
            if result.returncode == 0 and result.stdout.strip():
                # RSS is in KB, convert to bytes
                rss_kb = int(result.stdout.strip())
                return rss_kb * 1024
            else:
                logger.debug(f"ps command returned no data for PID {pid} (process may have terminated)")
                
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError, FileNotFoundError) as e:
            logger.debug(f"Failed to read memory using ps command for PID {pid}: {e}")
        
        return 0
    
    def _is_outlier(self, value: float, values: List[float], threshold: float = 10.0) -> bool:
        """
        Detect if a value is an outlier (>threshold * median).
        
        Args:
            value: Value to check
            values: List of all values for comparison
            threshold: Multiplier for median to determine outlier (default: 10.0)
        
        Returns:
            True if value is an outlier, False otherwise
        """
        if len(values) < 2:
            return False
        
        median = statistics.median(values)
        if median == 0:
            return False
        
        return value > threshold * median
    
    def profile_quantization(
        self,
        model_path: str,
        quant: str,
        prompt: str,
        max_tokens: int,
        warmup_tokens: int = 5,
        context_size: int = 2048
    ) -> QuantizationResult:
        """
        Profile single quantization level.
        
        Args:
            model_path: Path to GGUF model file
            quant: Quantization level identifier (e.g., "Q4_0", "Q8_0")
            prompt: Test prompt to use for inference
            max_tokens: Maximum tokens to generate
            warmup_tokens: Number of tokens to generate during warmup (default: 5)
            context_size: Context window size for model (default: 2048)
        
        Returns:
            QuantizationResult with all collected measurements
        
        Raises:
            Exception: If model loading or inference fails
        """
        logger.info(f"Profiling quantization level: {quant}")
        logger.info(f"Model path: {model_path}")
        
        # Measure baseline memory before model load
        baseline_memory_mb = self._get_total_memory_mb()
        logger.info(f"Baseline memory: {baseline_memory_mb:.2f} MB")
        
        # Get platform-specific configuration
        llama_config = self.backend.get_llama_config()
        
        # Set context size
        llama_config["n_ctx"] = context_size
        
        # Disable verbose output
        llama_config["verbose"] = False
        
        logger.info(f"Model config: {llama_config}")
        
        # Time model loading using backend's load_model_safe()
        # For non-Android: model loads during this call
        # For Android: only path validation happens here
        load_start = time.perf_counter()
        
        try:
            llm = self.backend.load_model_safe(model_path, **llama_config)
            if llm is None:
                raise RuntimeError(f"Failed to load model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise
        
        load_end = time.perf_counter()
        
        # Detect platform to determine load time measurement strategy
        is_android = self._is_android_platform(llm)
        
        if is_android:
            # For Android with NativeLlamaCpp, model loads during first inference
            logger.info("Android platform detected - measuring load time during first inference")
            
            # Measure load time during warmup inference (first inference call)
            load_start = time.perf_counter()
            try:
                _ = llm(prompt, max_tokens=warmup_tokens, stream=False)
            except Exception as e:
                logger.warning(f"Warmup inference failed: {e}")
                # Continue anyway - warmup failure shouldn't stop profiling
            load_end = time.perf_counter()
            load_time_s = load_end - load_start
            
            logger.info(f"Model loaded in {load_time_s:.2f} seconds (measured during first inference)")
        else:
            # For non-Android platforms, model loads during load_model_safe()
            load_time_s = load_end - load_start
            logger.info(f"Model loaded in {load_time_s:.2f} seconds")
        
        # Measure memory after model load
        post_load_memory_mb = self._get_total_memory_mb()
        ram_increase_mb = post_load_memory_mb - baseline_memory_mb
        
        logger.info(f"Memory after load: {post_load_memory_mb:.2f} MB")
        logger.info(f"RAM increase: {ram_increase_mb:.2f} MB")
        
        # Perform warmup inference (5 tokens) before measurement
        # For Android, we already did warmup during load time measurement
        if not is_android:
            logger.info(f"Performing warmup inference ({warmup_tokens} tokens)...")
            try:
                _ = llm(prompt, max_tokens=warmup_tokens, stream=False)
            except Exception as e:
                logger.warning(f"Warmup inference failed: {e}")
                # Continue anyway - warmup failure shouldn't stop profiling
        else:
            logger.info(f"Warmup already performed during load time measurement")
        
        # Measure memory after warmup
        post_warmup_memory_mb = self._get_total_memory_mb()
        logger.info(f"Memory after warmup: {post_warmup_memory_mb:.2f} MB")
        
        # Initialize peak memory tracker
        peak_ram_mb = post_warmup_memory_mb
        
        # Perform measurement inference with memory tracking during generation
        logger.info(f"Performing measurement inference ({max_tokens} tokens)...")
        
        try:
            # Use streaming inference and sample memory during generation
            stream = llm(
                prompt,
                max_tokens=max_tokens,
                stream=True,
                echo=False
            )
            
            # Track TTFT and tokens for metrics
            start_time = time.perf_counter()
            first_token_time = None
            output_tokens = 0
            
            # Sample memory during token generation
            for chunk in stream:
                current_time = time.perf_counter()
                
                # Capture TTFT
                if first_token_time is None:
                    first_token_time = current_time
                
                # Sample memory during inference (this captures peak during full context allocation)
                current_memory_mb = self._get_total_memory_mb()
                peak_ram_mb = max(peak_ram_mb, current_memory_mb)
                
                output_tokens += 1
            
            end_time = time.perf_counter()
            
            # Calculate metrics
            total_time_s = end_time - start_time
            
            if first_token_time is None:
                ttft_ms = 0.0
                ttft_s = 0.0
            else:
                ttft_s = first_token_time - start_time
                ttft_ms = ttft_s * 1000
            
            # Tokenize prompt to count tokens
            try:
                prompt_tokens = len(llm.tokenize(prompt.encode('utf-8')))
            except Exception as e:
                logger.warning(f"Failed to tokenize prompt: {e}")
                prompt_tokens = len(prompt) // 4
            
            # Calculate throughput
            if ttft_s > 0 and prompt_tokens > 0:
                prefill_tps = prompt_tokens / ttft_s
            else:
                prefill_tps = 0.0
            
            decode_duration = total_time_s - ttft_s
            if decode_duration > 0 and output_tokens > 1:
                decode_tps = (output_tokens - 1) / decode_duration
            else:
                decode_tps = 0.0
            
            # Create a simple metrics object
            class SimpleInferenceMetrics:
                def __init__(self, ttft_ms, prefill_tps, decode_tps, prompt_tokens, output_tokens):
                    self.ttft_ms = round(ttft_ms, 2)
                    self.prefill_tps = round(prefill_tps, 2)
                    self.decode_tps = round(decode_tps, 2)
                    self.prompt_tokens = prompt_tokens
                    self.output_tokens = output_tokens
                    self.gpu_memory_mb = None
                    self.gpu_utilization_pct = None
                    self.used_gpu_acceleration = False
            
            inference_metrics = SimpleInferenceMetrics(
                ttft_ms, prefill_tps, decode_tps, prompt_tokens, output_tokens
            )
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
        
        # Final memory check
        final_memory_mb = self._get_total_memory_mb()
        peak_ram_mb = max(peak_ram_mb, final_memory_mb)
        
        logger.info(f"Final memory: {final_memory_mb:.2f} MB")
        logger.info(f"Peak memory: {peak_ram_mb:.2f} MB")
        
        # Create quantization result
        result = QuantizationResult(
            quantization=quant,
            load_time_s=round(load_time_s, 2),
            peak_ram_mb=round(peak_ram_mb, 2),
            ram_increase_mb=round(ram_increase_mb, 2),
            ttft_ms=inference_metrics.ttft_ms,
            prefill_tps=inference_metrics.prefill_tps,
            decode_tps=inference_metrics.decode_tps,
            prompt_tokens=inference_metrics.prompt_tokens,
            output_tokens=inference_metrics.output_tokens,
            gpu_memory_mb=inference_metrics.gpu_memory_mb,
            gpu_utilization_pct=inference_metrics.gpu_utilization_pct,
            used_gpu_acceleration=inference_metrics.used_gpu_acceleration
        )
        
        logger.info(f"Quantization profiling complete for {quant}")
        logger.info(f"  Load time: {result.load_time_s}s")
        logger.info(f"  Peak RAM: {result.peak_ram_mb} MB")
        logger.info(f"  RAM increase: {result.ram_increase_mb} MB")
        logger.info(f"  TTFT: {result.ttft_ms} ms")
        logger.info(f"  Prefill: {result.prefill_tps} t/s")
        logger.info(f"  Decode: {result.decode_tps} t/s")
        
        return result
    
    def profile_all(
        self,
        models: Dict[str, str],
        prompt: str,
        max_tokens: int,
        warmup_tokens: int = 5,
        context_size: int = 2048
    ) -> List[QuantizationResult]:
        """
        Profile all quantization levels with identical prompt.
        
        Enforces garbage collection between quantization tests to ensure
        fair comparison and prevent memory contamination.
        
        Args:
            models: Dictionary mapping quantization level to model path
                   (e.g., {"Q4_0": "/path/to/model-q4_0.gguf", "Q8_0": "/path/to/model-q8_0.gguf"})
            prompt: Test prompt to use for all quantization levels
            max_tokens: Maximum tokens to generate
            warmup_tokens: Number of tokens to generate during warmup (default: 5)
            context_size: Context window size for model (default: 2048)
        
        Returns:
            List of QuantizationResult objects, one per quantization level
        
        Raises:
            Exception: If any quantization level fails to profile
        """
        logger.info(f"Profiling {len(models)} quantization levels")
        logger.info(f"Prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Prompt: {prompt}")
        logger.info(f"Max tokens: {max_tokens}")
        
        results: List[QuantizationResult] = []
        
        for quant, model_path in models.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Profiling quantization: {quant}")
            logger.info(f"{'='*60}")
            
            try:
                # Profile this quantization level
                result = self.profile_quantization(
                    model_path=model_path,
                    quant=quant,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    warmup_tokens=warmup_tokens,
                    context_size=context_size
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to profile {quant}: {e}")
                # Re-raise to let orchestrator handle the error
                raise
            
            finally:
                # Enforce garbage collection between quantization tests
                logger.info("Enforcing garbage collection...")
                gc.collect()
                
                # Log memory after GC
                post_gc_memory_mb = self._get_total_memory_mb()
                logger.info(f"Memory after GC: {post_gc_memory_mb:.2f} MB")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Quantization profiling complete")
        logger.info(f"{'='*60}")
        
        # Check for decode TPS outliers
        decode_tps_values = [result.decode_tps for result in results]
        for result in results:
            if self._is_outlier(result.decode_tps, decode_tps_values, threshold=10.0):
                median_tps = statistics.median(decode_tps_values)
                logger.warning(
                    f"Outlier detected in {result.quantization}: "
                    f"decode_tps = {result.decode_tps:.2f} t/s "
                    f"(median: {median_tps:.2f} t/s, "
                    f"ratio: {result.decode_tps / median_tps:.1f}x)"
                )
        
        # Generate comparison matrix
        self._log_comparison_matrix(results)
        
        return results
    
    def _log_comparison_matrix(self, results: List[QuantizationResult]) -> None:
        """
        Log comparison matrix showing all metrics across quantization levels.
        
        Args:
            results: List of QuantizationResult objects to compare
        """
        if not results:
            return
        
        logger.info("\nQuantization Comparison Matrix:")
        logger.info("=" * 110)
        
        # Header
        header = f"{'Quant':<10} {'Load(s)':<10} {'RAM(MB)':<12} {'TTFT(ms)':<12} {'Prefill(t/s)':<15} {'Decode(t/s)':<15}"
        if results[0].gpu_memory_mb is not None:
            header += f" {'GPU(MB)':<12} {'GPU(%)':<10} {'GPU Used':<10}"
        
        logger.info(header)
        logger.info("-" * 110)
        
        # Data rows
        for result in results:
            row = (
                f"{result.quantization:<10} "
                f"{result.load_time_s:<10.2f} "
                f"{result.peak_ram_mb:<12.2f} "
                f"{result.ttft_ms:<12.2f} "
                f"{result.prefill_tps:<15.2f} "
                f"{result.decode_tps:<15.2f}"
            )
            
            if result.gpu_memory_mb is not None:
                gpu_used_str = "Yes" if result.used_gpu_acceleration else "No"
                row += f" {result.gpu_memory_mb:<12.2f} {result.gpu_utilization_pct:<10.2f} {gpu_used_str:<10}"
            
            logger.info(row)
        
        logger.info("=" * 110)
