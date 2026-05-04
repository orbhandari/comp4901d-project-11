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
import time
from typing import Dict, List

import psutil
from llama_cpp import Llama

from llm_benchmark.hardware.hal import HardwareBackend
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
        baseline_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        logger.info(f"Baseline memory: {baseline_memory_mb:.2f} MB")
        
        # Get platform-specific configuration
        llama_config = self.backend.get_llama_config()
        
        # Set context size
        llama_config["n_ctx"] = context_size
        
        # Disable verbose output
        llama_config["verbose"] = False
        
        logger.info(f"Llama config: {llama_config}")
        
        # Time model loading
        load_start = time.perf_counter()
        
        try:
            llm = Llama(model_path=model_path, **llama_config)
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise
        
        load_end = time.perf_counter()
        load_time_s = load_end - load_start
        
        logger.info(f"Model loaded in {load_time_s:.2f} seconds")
        
        # Measure memory after model load
        post_load_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        ram_increase_mb = post_load_memory_mb - baseline_memory_mb
        
        logger.info(f"Memory after load: {post_load_memory_mb:.2f} MB")
        logger.info(f"RAM increase: {ram_increase_mb:.2f} MB")
        
        # Perform warmup inference (5 tokens) before measurement
        logger.info(f"Performing warmup inference ({warmup_tokens} tokens)...")
        try:
            _ = llm(prompt, max_tokens=warmup_tokens, stream=False)
        except Exception as e:
            logger.warning(f"Warmup inference failed: {e}")
            # Continue anyway - warmup failure shouldn't stop profiling
        
        # Measure peak memory after warmup
        peak_ram_mb = self.process.memory_info().rss / (1024 * 1024)
        logger.info(f"Peak memory after warmup: {peak_ram_mb:.2f} MB")
        
        # Perform measurement inference with streaming to capture TTFT accurately
        logger.info(f"Performing measurement inference ({max_tokens} tokens)...")
        
        try:
            inference_metrics = self.metrics.collect_inference_metrics(
                llm=llm,
                prompt=prompt,
                max_tokens=max_tokens,
                enable_background_monitoring=False  # Disable for quantization profiling
            )
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
        
        # Update peak memory if inference used more
        final_memory_mb = self.process.memory_info().rss / (1024 * 1024)
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
                post_gc_memory_mb = self.process.memory_info().rss / (1024 * 1024)
                logger.info(f"Memory after GC: {post_gc_memory_mb:.2f} MB")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Quantization profiling complete")
        logger.info(f"{'='*60}")
        
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
