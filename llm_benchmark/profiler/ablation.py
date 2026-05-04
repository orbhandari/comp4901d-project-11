"""
Ablation Engine for conducting controlled experiments isolating optimization effects.

This module provides the AblationEngine class which:
- Tests RAM-based and disk-based KV cache implementations
- Executes control runs without caching (baseline measurement)
- Executes cold runs with cache enabled but empty
- Executes warm runs with cache populated from previous prompt
- Uses prompts with substantial shared prefix (minimum 500 tokens)
- Measures TTFT improvement between cold and warm runs
- Measures cache memory overhead
- Ensures process isolation by creating fresh model instances
- Cleans up cache directories after ablation completion

IMPORTANT: Cache Control Strategies
===================================

The ability to disable RAM prompt caching depends on the backend:

1. **llama-cpp-python Backend (X86, Jetson)**:
   - Can disable KV cache by setting cache=False
   - Provides true "no cache" baseline for ablation studies
   - Recommended for accurate cache effect measurement

2. **Native llama-cli Backend (Android)**:
   - KV cache is ALWAYS enabled and cannot be disabled via CLI flags
   - Control runs will have KV cache active (not a true baseline)
   - This is a limitation of the llama-cli binary

3. **llama-server Backend (Recommended for Android)**:
   - Supports --cache-ram 0 and --no-cache-prompt flags
   - Can truly disable prompt caching for accurate measurements
   - Set cache_prompt=false in API requests
   - Provides the most control over caching behavior

How to Disable Prompt Caching:
-------------------------------

**Using llama-server:**
```bash
# Start server with caching disabled
llama-server --model model.gguf --cache-ram 0 --no-cache-prompt

# In API requests, set:
{
  "prompt": "...",
  "cache_prompt": false  # Ensure this is false
}
```

**Using llama-cli:**
```bash
# Avoid session flags to prevent prompt caching
# Do NOT use: --prompt-cache or --path_session
llama-cli --model model.gguf --prompt "..." --n-predict 100
```

**Using llama-cpp-python:**
```python
from llama_cpp import Llama
llm = Llama(model_path="model.gguf", cache=False)  # Disables KV cache
```

Limitations:
-----------
- llama-cli: Cannot disable KV cache (always active)
- Some llama-server versions may log "cache enabled" even with flags set (cosmetic bug)
- For true no-cache baseline on Android, use llama-server instead of llama-cli

References:
----------
[1] llama.cpp server documentation
[2] llama.cpp CLI flags reference
[3] Community discussions on cache control
[4] llama-cpp-python API documentation
"""

import gc
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import psutil

from llm_benchmark.hardware.hal import HardwareBackend
from llm_benchmark.metrics.collector import MetricsCollector
from llm_benchmark.models import AblationResult, InferenceMetrics

logger = logging.getLogger(__name__)


class AblationEngine:
    """
    Conducts controlled experiments isolating specific optimization effects.
    
    Implements KV cache ablation studies with process isolation, memory tracking,
    and automatic cleanup of cache directories.
    """
    
    def __init__(self, backend: HardwareBackend, metrics_collector: MetricsCollector, context_size: int = 2048, use_llama_server: bool = False):
        """
        Initialize ablation engine.
        
        Args:
            backend: Hardware backend providing platform-specific configuration
            metrics_collector: Metrics collector for inference measurements
            context_size: Context window size for model (default: 2048)
            use_llama_server: If True, use llama-server for better cache control (default: False)
        """
        self.backend = backend
        self.metrics = metrics_collector
        self.process = psutil.Process()
        self.context_size = context_size
        self.use_llama_server = use_llama_server
        
        # Temporary directories for cache testing
        self.temp_dirs: List[Path] = []
        
        logger.info(f"AblationEngine initialized (use_llama_server={use_llama_server})")
    
    def _load_model(self, model_path: str, **kwargs) -> Any:
        """
        Load model using backend's load_model_safe() method.
        
        This ensures compatibility with both llama-cpp-python and native llama.cpp.
        
        Args:
            model_path: Path to GGUF model file
            **kwargs: Additional arguments for model loading
        
        Returns:
            Loaded model instance
        
        Raises:
            RuntimeError: If model loading fails
        """
        llm = self.backend.load_model_safe(model_path, **kwargs)
        if llm is None:
            raise RuntimeError(f"Failed to load model: {model_path}")
        return llm
    
    def test_kv_cache_strategies(
        self,
        model_path: str,
        prompt_prefix: str,
        prompt_suffix: str,
        max_tokens: int = 50,
        cache_types: Optional[List[str]] = None
    ) -> List[AblationResult]:
        """
        Test RAM vs disk KV cache with cold/warm runs.
        
        Implements the following test scenarios:
        1. Control run: No caching (baseline)
        2. Cold run (RAM): Cache enabled but empty
        3. Warm run (RAM): Cache populated from previous prompt
        4. Cold run (Disk): Cache enabled but empty
        5. Warm run (Disk): Cache populated from previous prompt
        
        Args:
            model_path: Path to GGUF model file
            prompt_prefix: Shared prefix for cache effectiveness (minimum 500 tokens recommended)
            prompt_suffix: Unique suffix to append after prefix
            max_tokens: Maximum tokens to generate (default: 50)
            cache_types: List of cache types to test (default: ["ram", "disk"])
        
        Returns:
            List of AblationResult objects for each test scenario
        
        Raises:
            ValueError: If prompt_prefix is too short (< 100 tokens estimated)
        """
        if cache_types is None:
            cache_types = ["ram", "disk"]
        
        # Validate prompt prefix length (rough estimate: 4 chars per token)
        estimated_tokens = len(prompt_prefix) // 4
        if estimated_tokens < 100:
            logger.warning(
                f"Prompt prefix may be too short ({estimated_tokens} estimated tokens). "
                f"Recommend at least 500 tokens for effective cache testing."
            )
        
        logger.info("=" * 80)
        logger.info("Starting KV Cache Ablation Studies")
        logger.info("=" * 80)
        logger.info(f"Model: {model_path}")
        logger.info(f"Prompt prefix length: ~{estimated_tokens} tokens (estimated)")
        logger.info(f"Cache types to test: {cache_types}")
        logger.info(f"Max tokens: {max_tokens}")
        
        results: List[AblationResult] = []
        
        try:
            # 1. Control run: No caching (baseline)
            logger.info("\n" + "=" * 60)
            logger.info("Test 1: Control Run (No Caching)")
            logger.info("=" * 60)
            
            control_result = self._run_control(
                model_path=model_path,
                prompt=prompt_prefix + prompt_suffix,
                max_tokens=max_tokens
            )
            results.append(control_result)
            
            # Enforce garbage collection and process isolation
            self._ensure_process_isolation()
            
            # 2. RAM cache tests
            if "ram" in cache_types:
                logger.info("\n" + "=" * 60)
                logger.info("Test 2: RAM Cache - Cold Run")
                logger.info("=" * 60)
                
                ram_cold_result = self._run_cold_cache(
                    model_path=model_path,
                    prompt=prompt_prefix + prompt_suffix,
                    max_tokens=max_tokens,
                    cache_type="ram",
                    baseline_ttft=control_result.metrics.get("ttft_ms")
                )
                results.append(ram_cold_result)
                
                self._ensure_process_isolation()
                
                logger.info("\n" + "=" * 60)
                logger.info("Test 3: RAM Cache - Warm Run")
                logger.info("=" * 60)
                
                ram_warm_result = self._run_warm_cache(
                    model_path=model_path,
                    prompt_prefix=prompt_prefix,
                    prompt_suffix=prompt_suffix,
                    max_tokens=max_tokens,
                    cache_type="ram",
                    baseline_ttft=control_result.metrics.get("ttft_ms")
                )
                results.append(ram_warm_result)
                
                self._ensure_process_isolation()
            
            # 3. Disk cache tests
            if "disk" in cache_types:
                logger.info("\n" + "=" * 60)
                logger.info("Test 4: Disk Cache - Cold Run")
                logger.info("=" * 60)
                
                disk_cold_result = self._run_cold_cache(
                    model_path=model_path,
                    prompt=prompt_prefix + prompt_suffix,
                    max_tokens=max_tokens,
                    cache_type="disk",
                    baseline_ttft=control_result.metrics.get("ttft_ms")
                )
                results.append(disk_cold_result)
                
                self._ensure_process_isolation()
                
                logger.info("\n" + "=" * 60)
                logger.info("Test 5: Disk Cache - Warm Run")
                logger.info("=" * 60)
                
                disk_warm_result = self._run_warm_cache(
                    model_path=model_path,
                    prompt_prefix=prompt_prefix,
                    prompt_suffix=prompt_suffix,
                    max_tokens=max_tokens,
                    cache_type="disk",
                    baseline_ttft=control_result.metrics.get("ttft_ms")
                )
                results.append(disk_warm_result)
                
                self._ensure_process_isolation()
            
            logger.info("\n" + "=" * 80)
            logger.info("KV Cache Ablation Studies Complete")
            logger.info("=" * 80)
            
            # Log summary
            self._log_ablation_summary(results)
            
            return results
            
        finally:
            # Clean up cache directories
            self._cleanup_cache_directories()
    
    def _run_control(
        self,
        model_path: str,
        prompt: str,
        max_tokens: int
    ) -> AblationResult:
        """
        Execute control run without caching (baseline measurement).
        
        IMPORTANT: For llama-cpp-python, cache=False disables KV cache.
        For native llama.cpp (Android), KV cache is ALWAYS enabled and cannot be disabled.
        This is a limitation of the native llama.cpp CLI.
        
        To measure true "no cache" baseline with native llama.cpp, you would need to:
        - Use llama-server with --cache-ram 0 --no-cache-prompt flags
        - Or modify llama.cpp source code to disable KV cache
        
        Args:
            model_path: Path to GGUF model file
            prompt: Test prompt
            max_tokens: Maximum tokens to generate
        
        Returns:
            AblationResult with baseline measurements
        """
        logger.info("Running control (no cache) baseline measurement...")
        
        # Check if using native llama.cpp
        using_native = hasattr(self.backend, '__class__') and 'Android' in self.backend.__class__.__name__
        if using_native:
            logger.warning("=" * 80)
            logger.warning("LIMITATION: Native llama.cpp ALWAYS has KV cache enabled (RAM)")
            logger.warning("This 'control' run is NOT a true no-cache baseline!")
            logger.warning("KV cache cannot be disabled via llama-cli flags.")
            logger.warning("")
            logger.warning("To get true no-cache baseline, you would need to:")
            logger.warning("  1. Use llama-server with --cache-ram 0 --no-cache-prompt")
            logger.warning("  2. Or modify llama.cpp source to disable KV cache")
            logger.warning("=" * 80)
        
        # Measure baseline memory
        baseline_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        
        # Get platform-specific configuration (no cache)
        llama_config = self.backend.get_llama_config()
        
        # Explicitly disable caching (only works for llama-cpp-python)
        llama_config["cache"] = False
        
        # Set context size
        llama_config["n_ctx"] = self.context_size
        
        # Disable verbose output
        llama_config["verbose"] = False
        
        logger.info(f"Model config: {llama_config}")
        
        # Load model
        llm = self._load_model(model_path, **llama_config)
        
        # Measure memory after load
        post_load_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        memory_overhead_mb = post_load_memory_mb - baseline_memory_mb
        
        # Run inference
        metrics = self.metrics.collect_inference_metrics(
            llm=llm,
            prompt=prompt,
            max_tokens=max_tokens,
            enable_background_monitoring=False
        )
        
        # Measure peak memory
        peak_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        
        logger.info(f"Control run complete:")
        logger.info(f"  TTFT: {metrics.ttft_ms:.2f} ms")
        logger.info(f"  Memory overhead: {memory_overhead_mb:.2f} MB")
        logger.info(f"  Peak memory: {peak_memory_mb:.2f} MB")
        if using_native:
            logger.info(f"  Note: KV cache was ACTIVE (cannot be disabled)")
        
        return AblationResult(
            scenario="control_no_cache" if not using_native else "control_kv_cache_only",
            configuration={
                "cache_enabled": False if not using_native else True,
                "cache_type": None if not using_native else "ram_kv_only",
                "cache_state": "N/A" if not using_native else "KV cache active (cannot disable)",
                "true_no_cache_baseline": not using_native
            },
            metrics={
                "ttft_ms": metrics.ttft_ms,
                "prefill_tps": metrics.prefill_tps,
                "decode_tps": metrics.decode_tps,
                "memory_overhead_mb": round(memory_overhead_mb, 2),
                "peak_memory_mb": round(peak_memory_mb, 2),
                "prompt_tokens": metrics.prompt_tokens,
                "output_tokens": metrics.output_tokens
            },
            improvement_over_baseline=None  # This is the baseline
        )
    
    def _run_cold_cache(
        self,
        model_path: str,
        prompt: str,
        max_tokens: int,
        cache_type: str,
        baseline_ttft: Optional[float] = None
    ) -> AblationResult:
        """
        Execute cold run with cache enabled but empty.
        
        Args:
            model_path: Path to GGUF model file
            prompt: Test prompt
            max_tokens: Maximum tokens to generate
            cache_type: "ram" or "disk"
            baseline_ttft: Baseline TTFT for improvement calculation
        
        Returns:
            AblationResult with cold cache measurements
        """
        logger.info(f"Running cold {cache_type} cache test...")
        
        # Measure baseline memory
        baseline_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        
        # Get platform-specific configuration
        llama_config = self.backend.get_llama_config()
        
        # Configure cache
        if cache_type == "ram":
            # RAM cache is default in llama-cpp-python
            llama_config["cache"] = True
        elif cache_type == "disk":
            # Create temporary directory for disk cache
            cache_dir = self._create_temp_cache_dir()
            llama_config["cache"] = True
            llama_config["cache_type"] = "disk"
            llama_config["cache_dir"] = str(cache_dir)
        
        # Set context size
        llama_config["n_ctx"] = self.context_size
        
        # Disable verbose output
        llama_config["verbose"] = False
        
        logger.info(f"Model config: {llama_config}")
        
        # Load model with fresh cache
        llm = self._load_model(model_path, **llama_config)
        
        # Measure memory after load
        post_load_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        memory_overhead_mb = post_load_memory_mb - baseline_memory_mb
        
        # Run inference (cache is empty, so no benefit expected)
        metrics = self.metrics.collect_inference_metrics(
            llm=llm,
            prompt=prompt,
            max_tokens=max_tokens,
            enable_background_monitoring=False
        )
        
        # Measure peak memory
        peak_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        
        # Calculate improvement over baseline
        improvement = None
        if baseline_ttft is not None:
            improvement = ((baseline_ttft - metrics.ttft_ms) / baseline_ttft) * 100
        
        logger.info(f"Cold {cache_type} cache run complete:")
        logger.info(f"  TTFT: {metrics.ttft_ms:.2f} ms")
        logger.info(f"  Memory overhead: {memory_overhead_mb:.2f} MB")
        logger.info(f"  Peak memory: {peak_memory_mb:.2f} MB")
        if improvement is not None:
            logger.info(f"  Improvement over baseline: {improvement:.2f}%")
        
        return AblationResult(
            scenario=f"cold_{cache_type}_cache",
            configuration={
                "cache_enabled": True,
                "cache_type": cache_type,
                "cache_state": "empty"
            },
            metrics={
                "ttft_ms": metrics.ttft_ms,
                "prefill_tps": metrics.prefill_tps,
                "decode_tps": metrics.decode_tps,
                "memory_overhead_mb": round(memory_overhead_mb, 2),
                "peak_memory_mb": round(peak_memory_mb, 2),
                "prompt_tokens": metrics.prompt_tokens,
                "output_tokens": metrics.output_tokens
            },
            improvement_over_baseline=round(improvement, 2) if improvement is not None else None
        )
    
    def _run_warm_cache(
        self,
        model_path: str,
        prompt_prefix: str,
        prompt_suffix: str,
        max_tokens: int,
        cache_type: str,
        baseline_ttft: Optional[float] = None
    ) -> AblationResult:
        """
        Execute warm run with cache populated from previous prompt.
        
        Args:
            model_path: Path to GGUF model file
            prompt_prefix: Shared prefix to populate cache
            prompt_suffix: Unique suffix for second inference
            max_tokens: Maximum tokens to generate
            cache_type: "ram" or "disk"
            baseline_ttft: Baseline TTFT for improvement calculation
        
        Returns:
            AblationResult with warm cache measurements
        """
        logger.info(f"Running warm {cache_type} cache test...")
        
        # Measure baseline memory
        baseline_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        
        # Get platform-specific configuration
        llama_config = self.backend.get_llama_config()
        
        # Configure cache
        cache_dir = None
        if cache_type == "ram":
            llama_config["cache"] = True
        elif cache_type == "disk":
            cache_dir = self._create_temp_cache_dir()
            llama_config["cache"] = True
            llama_config["cache_type"] = "disk"
            llama_config["cache_dir"] = str(cache_dir)
        
        # Set context size
        llama_config["n_ctx"] = self.context_size
        
        # Disable verbose output
        llama_config["verbose"] = False
        
        logger.info(f"Model config: {llama_config}")
        
        # Load model
        llm = self._load_model(model_path, **llama_config)
        
        # Measure memory after load
        post_load_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        
        # First inference: Populate cache with prefix
        logger.info("Populating cache with prefix prompt...")
        first_prompt = prompt_prefix + " [warmup]"
        
        _ = self.metrics.collect_inference_metrics(
            llm=llm,
            prompt=first_prompt,
            max_tokens=10,  # Short generation just to populate cache
            enable_background_monitoring=False
        )
        
        # Measure memory after cache population
        post_cache_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        cache_memory_overhead_mb = post_cache_memory_mb - post_load_memory_mb
        
        logger.info(f"Cache populated. Memory overhead: {cache_memory_overhead_mb:.2f} MB")
        
        # Second inference: Use cached prefix with different suffix
        logger.info("Running inference with cached prefix...")
        second_prompt = prompt_prefix + prompt_suffix
        
        metrics = self.metrics.collect_inference_metrics(
            llm=llm,
            prompt=second_prompt,
            max_tokens=max_tokens,
            enable_background_monitoring=False
        )
        
        # Measure peak memory
        peak_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        total_memory_overhead_mb = peak_memory_mb - baseline_memory_mb
        
        # Calculate improvement over baseline
        improvement = None
        if baseline_ttft is not None:
            improvement = ((baseline_ttft - metrics.ttft_ms) / baseline_ttft) * 100
        
        logger.info(f"Warm {cache_type} cache run complete:")
        logger.info(f"  TTFT: {metrics.ttft_ms:.2f} ms")
        logger.info(f"  Cache memory overhead: {cache_memory_overhead_mb:.2f} MB")
        logger.info(f"  Total memory overhead: {total_memory_overhead_mb:.2f} MB")
        logger.info(f"  Peak memory: {peak_memory_mb:.2f} MB")
        if improvement is not None:
            logger.info(f"  Improvement over baseline: {improvement:.2f}%")
        
        return AblationResult(
            scenario=f"warm_{cache_type}_cache",
            configuration={
                "cache_enabled": True,
                "cache_type": cache_type,
                "cache_state": "populated"
            },
            metrics={
                "ttft_ms": metrics.ttft_ms,
                "prefill_tps": metrics.prefill_tps,
                "decode_tps": metrics.decode_tps,
                "cache_memory_overhead_mb": round(cache_memory_overhead_mb, 2),
                "total_memory_overhead_mb": round(total_memory_overhead_mb, 2),
                "peak_memory_mb": round(peak_memory_mb, 2),
                "prompt_tokens": metrics.prompt_tokens,
                "output_tokens": metrics.output_tokens
            },
            improvement_over_baseline=round(improvement, 2) if improvement is not None else None
        )
    
    def _ensure_process_isolation(self) -> None:
        """
        Force garbage collection and create fresh model instance.
        
        Ensures that each test run starts with a clean state to prevent
        memory contamination between tests.
        """
        logger.info("Enforcing process isolation...")
        
        # Force garbage collection
        gc.collect()
        
        # Log memory after GC
        post_gc_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        logger.info(f"Memory after GC: {post_gc_memory_mb:.2f} MB")
        
        # Small delay to allow system to stabilize
        time.sleep(1)
    
    def _create_temp_cache_dir(self) -> Path:
        """
        Create temporary directory for disk cache testing.
        
        Returns:
            Path to temporary cache directory
        """
        temp_dir = Path(tempfile.mkdtemp(prefix="llm_cache_"))
        self.temp_dirs.append(temp_dir)
        logger.info(f"Created temporary cache directory: {temp_dir}")
        return temp_dir
    
    def _cleanup_cache_directories(self) -> None:
        """
        Clean up all temporary cache directories created during testing.
        """
        if not self.temp_dirs:
            return
        
        logger.info("Cleaning up cache directories...")
        
        for temp_dir in self.temp_dirs:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    logger.info(f"Removed cache directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove cache directory {temp_dir}: {e}")
        
        self.temp_dirs.clear()
        logger.info("Cache cleanup complete")
    
    def test_prompt_caching(
        self,
        model_path: str,
        prefix_lengths: Optional[List[int]] = None,
        max_tokens: int = 50,
        cache_types: Optional[List[str]] = None
    ) -> List[AblationResult]:
        """
        Test prompt caching with varying shared prefix lengths.
        
        Measures:
        - Cache hit rate (percentage of tokens reused from cache)
        - Latency reduction from prompt caching (milliseconds)
        - Cache memory overhead (percentage of total model memory)
        - Disk I/O time for cache operations (disk cache only)
        - Cache file size (megabytes, disk cache only)
        
        Args:
            model_path: Path to GGUF model file
            prefix_lengths: List of prefix lengths to test in tokens (default: [100, 500, 1000])
            max_tokens: Maximum tokens to generate (default: 50)
            cache_types: List of cache types to test (default: ["ram", "disk"])
        
        Returns:
            List of AblationResult objects for each test scenario
        """
        if prefix_lengths is None:
            prefix_lengths = [100, 500, 1000]
        
        if cache_types is None:
            cache_types = ["ram", "disk"]
        
        logger.info("=" * 80)
        logger.info("Starting Prompt Caching Optimization Tests")
        logger.info("=" * 80)
        logger.info(f"Model: {model_path}")
        logger.info(f"Prefix lengths to test: {prefix_lengths} tokens")
        logger.info(f"Cache types to test: {cache_types}")
        logger.info(f"Max tokens: {max_tokens}")
        
        results: List[AblationResult] = []
        
        try:
            # Generate test prompts with varying prefix lengths
            test_prompts = self._generate_test_prompts(prefix_lengths)
            
            # Test each cache type
            for cache_type in cache_types:
                logger.info("\n" + "=" * 60)
                logger.info(f"Testing {cache_type.upper()} Cache")
                logger.info("=" * 60)
                
                # Test each prefix length
                for prefix_length in prefix_lengths:
                    logger.info(f"\nTesting prefix length: {prefix_length} tokens")
                    
                    prefix_prompt = test_prompts[prefix_length]["prefix"]
                    suffix_prompt = test_prompts[prefix_length]["suffix"]
                    
                    result = self._run_prompt_caching_test(
                        model_path=model_path,
                        prefix_prompt=prefix_prompt,
                        suffix_prompt=suffix_prompt,
                        prefix_length=prefix_length,
                        max_tokens=max_tokens,
                        cache_type=cache_type
                    )
                    
                    results.append(result)
                    
                    self._ensure_process_isolation()
            
            logger.info("\n" + "=" * 80)
            logger.info("Prompt Caching Optimization Tests Complete")
            logger.info("=" * 80)
            
            # Log summary
            self._log_prompt_caching_summary(results)
            
            return results
            
        finally:
            # Clean up cache directories
            self._cleanup_cache_directories()
    
    def test_prompt_caching_across_quantizations(
        self,
        model_paths: Dict[str, str],
        prefix_lengths: Optional[List[int]] = None,
        max_tokens: int = 50,
        cache_types: Optional[List[str]] = None
    ) -> List[AblationResult]:
        """
        Compare prompt caching effectiveness across different quantization levels.
        
        Tests prompt caching with multiple quantized models to compare
        cache effectiveness across quantization levels.
        
        Args:
            model_paths: Dictionary mapping quantization level to model path
                        e.g., {"Q8_0": "path/to/q8.gguf", "Q4_0": "path/to/q4.gguf"}
            prefix_lengths: List of prefix lengths to test in tokens (default: [100, 500, 1000])
            max_tokens: Maximum tokens to generate (default: 50)
            cache_types: List of cache types to test (default: ["ram", "disk"])
        
        Returns:
            List of AblationResult objects for each quantization/prefix/cache combination
        """
        if prefix_lengths is None:
            prefix_lengths = [100, 500, 1000]
        
        if cache_types is None:
            cache_types = ["ram", "disk"]
        
        logger.info("=" * 80)
        logger.info("Prompt Caching Across Quantization Levels")
        logger.info("=" * 80)
        logger.info(f"Quantization levels: {list(model_paths.keys())}")
        logger.info(f"Prefix lengths: {prefix_lengths} tokens")
        logger.info(f"Cache types: {cache_types}")
        
        all_results: List[AblationResult] = []
        
        try:
            # Test each quantization level
            for quant_level, model_path in model_paths.items():
                logger.info("\n" + "=" * 60)
                logger.info(f"Testing Quantization: {quant_level}")
                logger.info("=" * 60)
                
                # Run prompt caching tests for this quantization
                results = self.test_prompt_caching(
                    model_path=model_path,
                    prefix_lengths=prefix_lengths,
                    max_tokens=max_tokens,
                    cache_types=cache_types
                )
                
                # Add quantization level to results
                for result in results:
                    result.configuration["quantization"] = quant_level
                    result.scenario = f"{result.scenario}_{quant_level}"
                
                all_results.extend(results)
                
                self._ensure_process_isolation()
            
            logger.info("\n" + "=" * 80)
            logger.info("Quantization Comparison Complete")
            logger.info("=" * 80)
            
            # Log comparison summary
            self._log_quantization_comparison_summary(all_results)
            
            return all_results
            
        finally:
            # Cleanup is handled by test_prompt_caching
            pass
    
    def test_concurrent_prompt_caching(
        self,
        model_path: str,
        shared_prefix: str,
        unique_suffixes: List[str],
        max_tokens: int = 50,
        cache_type: str = "ram"
    ) -> List[AblationResult]:
        """
        Test cache behavior with multiple concurrent prompts sharing prefixes.
        
        Simulates multiple prompts that share a common prefix to test
        cache reuse effectiveness.
        
        Args:
            model_path: Path to GGUF model file
            shared_prefix: Common prefix shared by all prompts
            unique_suffixes: List of unique suffixes to append to shared prefix
            max_tokens: Maximum tokens to generate per prompt (default: 50)
            cache_type: Cache type to use (default: "ram")
        
        Returns:
            List of AblationResult objects for each prompt
        """
        logger.info("=" * 80)
        logger.info("Concurrent Prompt Caching Test")
        logger.info("=" * 80)
        logger.info(f"Model: {model_path}")
        logger.info(f"Cache type: {cache_type}")
        logger.info(f"Number of prompts: {len(unique_suffixes)}")
        logger.info(f"Shared prefix length: ~{len(shared_prefix) // 4} tokens (estimated)")
        
        results: List[AblationResult] = []
        
        try:
            # Measure baseline memory
            baseline_memory_mb = self.process.memory_info().rss / (1024 * 1024)
            
            # Get platform-specific configuration
            llama_config = self.backend.get_llama_config()
            
            # Configure cache
            cache_dir = None
            if cache_type == "ram":
                llama_config["cache"] = True
            elif cache_type == "disk":
                cache_dir = self._create_temp_cache_dir()
                llama_config["cache"] = True
                llama_config["cache_type"] = "disk"
                llama_config["cache_dir"] = str(cache_dir)
            
            # Set context size
            llama_config["n_ctx"] = self.context_size
            
            # Disable verbose output
            llama_config["verbose"] = False
            
            logger.info(f"Model config: {llama_config}")
            
            # Load model
            llm = self._load_model(model_path, **llama_config)
            
            # Measure memory after load
            post_load_memory_mb = self.process.memory_info().rss / (1024 * 1024)
            model_memory_mb = post_load_memory_mb - baseline_memory_mb
            
            # First inference: Populate cache with shared prefix
            logger.info("\nFirst inference: Populating cache with shared prefix...")
            warmup_prompt = shared_prefix + " [warmup]"
            
            warmup_metrics = self.metrics.collect_inference_metrics(
                llm=llm,
                prompt=warmup_prompt,
                max_tokens=10,
                enable_background_monitoring=False
            )
            
            # Measure cache memory overhead
            post_cache_memory_mb = self.process.memory_info().rss / (1024 * 1024)
            cache_memory_overhead_mb = post_cache_memory_mb - post_load_memory_mb
            
            logger.info(f"Cache populated. Memory overhead: {cache_memory_overhead_mb:.2f} MB")
            
            # Process each prompt with shared prefix
            for idx, suffix in enumerate(unique_suffixes, 1):
                logger.info(f"\nProcessing prompt {idx}/{len(unique_suffixes)}...")
                
                prompt = shared_prefix + suffix
                
                # Measure inference with cached prefix
                start_time = time.time()
                metrics = self.metrics.collect_inference_metrics(
                    llm=llm,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    enable_background_monitoring=False
                )
                end_time = time.time()
                
                # Calculate cache hit rate
                total_tokens = metrics.prompt_tokens
                estimated_prefix_tokens = len(shared_prefix) // 4
                estimated_cached_tokens = min(estimated_prefix_tokens, total_tokens)
                cache_hit_rate_pct = (estimated_cached_tokens / total_tokens) * 100 if total_tokens > 0 else 0
                
                # Calculate cache memory overhead percentage
                cache_memory_overhead_pct = (cache_memory_overhead_mb / model_memory_mb) * 100 if model_memory_mb > 0 else 0
                
                logger.info(f"  TTFT: {metrics.ttft_ms:.2f} ms")
                logger.info(f"  Cache hit rate: {cache_hit_rate_pct:.2f}%")
                
                result = AblationResult(
                    scenario=f"concurrent_prompt_{idx}_{cache_type}",
                    configuration={
                        "cache_type": cache_type,
                        "prompt_index": idx,
                        "total_prompts": len(unique_suffixes),
                        "shared_prefix": True
                    },
                    metrics={
                        "prompt_index": idx,
                        "ttft_ms": round(metrics.ttft_ms, 2),
                        "prefill_tps": round(metrics.prefill_tps, 2),
                        "decode_tps": round(metrics.decode_tps, 2),
                        "cache_hit_rate_pct": round(cache_hit_rate_pct, 2),
                        "cache_memory_overhead_mb": round(cache_memory_overhead_mb, 2),
                        "cache_memory_overhead_pct": round(cache_memory_overhead_pct, 2),
                        "prompt_tokens": metrics.prompt_tokens,
                        "output_tokens": metrics.output_tokens,
                        "total_time_s": round(end_time - start_time, 2)
                    },
                    improvement_over_baseline=None  # No baseline for concurrent test
                )
                
                results.append(result)
            
            logger.info("\n" + "=" * 80)
            logger.info("Concurrent Prompt Caching Test Complete")
            logger.info("=" * 80)
            
            # Log summary
            self._log_concurrent_caching_summary(results)
            
            return results
            
        finally:
            # Clean up cache directories
            self._cleanup_cache_directories()
    
    def _generate_test_prompts(self, prefix_lengths: List[int]) -> Dict[int, Dict[str, str]]:
        """
        Generate test prompts with specified prefix lengths.
        
        Uses a simple token estimation: ~4 characters per token.
        
        Args:
            prefix_lengths: List of desired prefix lengths in tokens
        
        Returns:
            Dictionary mapping prefix_length to {"prefix": str, "suffix": str}
        """
        prompts = {}
        
        # Base text for generating prompts (repeatable pattern)
        base_text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a test sentence for prompt caching evaluation. "
            "We need to generate prompts of varying lengths to test cache effectiveness. "
        )
        
        for length in prefix_lengths:
            # Estimate characters needed (~4 chars per token)
            chars_needed = length * 4
            
            # Repeat base text to reach desired length
            repetitions = (chars_needed // len(base_text)) + 1
            prefix = (base_text * repetitions)[:chars_needed]
            
            # Add unique suffix
            suffix = f" [Unique suffix for {length} token prefix test]"
            
            prompts[length] = {
                "prefix": prefix,
                "suffix": suffix
            }
        
        return prompts
    
    def _run_prompt_caching_test(
        self,
        model_path: str,
        prefix_prompt: str,
        suffix_prompt: str,
        prefix_length: int,
        max_tokens: int,
        cache_type: str
    ) -> AblationResult:
        """
        Run a single prompt caching test.
        
        Args:
            model_path: Path to GGUF model file
            prefix_prompt: Shared prefix prompt
            suffix_prompt: Unique suffix to append
            prefix_length: Length of prefix in tokens (for reporting)
            max_tokens: Maximum tokens to generate
            cache_type: "ram" or "disk"
        
        Returns:
            AblationResult with prompt caching metrics
        """
        logger.info(f"Running prompt caching test: {cache_type} cache, {prefix_length} token prefix")
        
        # Measure baseline memory
        baseline_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        
        # Get platform-specific configuration
        llama_config = self.backend.get_llama_config()
        
        # Configure cache
        cache_dir = None
        disk_io_start = None
        disk_io_end = None
        cache_file_size_mb = None
        
        if cache_type == "ram":
            llama_config["cache"] = True
        elif cache_type == "disk":
            cache_dir = self._create_temp_cache_dir()
            llama_config["cache"] = True
            llama_config["cache_type"] = "disk"
            llama_config["cache_dir"] = str(cache_dir)
        
        # Set context size
        llama_config["n_ctx"] = self.context_size
        
        # Disable verbose output
        llama_config["verbose"] = False
        
        logger.info(f"Model config: {llama_config}")
        
        # Load model
        llm = self._load_model(model_path, **llama_config)
        
        # Measure memory after load
        post_load_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        model_memory_mb = post_load_memory_mb - baseline_memory_mb
        
        # First inference: Populate cache with prefix
        logger.info("First inference: Populating cache with prefix...")
        first_prompt = prefix_prompt + " [first]"
        
        if cache_type == "disk":
            disk_io_start = time.time()
        
        first_metrics = self.metrics.collect_inference_metrics(
            llm=llm,
            prompt=first_prompt,
            max_tokens=10,  # Short generation to populate cache
            enable_background_monitoring=False
        )
        
        if cache_type == "disk":
            disk_io_end = time.time()
        
        # Measure memory after cache population
        post_cache_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        cache_memory_overhead_mb = post_cache_memory_mb - post_load_memory_mb
        
        # Calculate cache memory overhead as percentage of model memory
        cache_memory_overhead_pct = (cache_memory_overhead_mb / model_memory_mb) * 100 if model_memory_mb > 0 else 0
        
        # Measure cache file size for disk cache
        if cache_type == "disk" and cache_dir:
            cache_file_size_mb = self._measure_cache_directory_size(cache_dir)
        
        logger.info(f"Cache populated. Memory overhead: {cache_memory_overhead_mb:.2f} MB ({cache_memory_overhead_pct:.2f}%)")
        
        # Second inference: Use cached prefix with different suffix (cold - no cache hit expected)
        logger.info("Second inference: Testing with different suffix (measuring baseline)...")
        second_prompt_cold = prefix_prompt + " [different suffix for baseline]"
        
        second_metrics_cold = self.metrics.collect_inference_metrics(
            llm=llm,
            prompt=second_prompt_cold,
            max_tokens=max_tokens,
            enable_background_monitoring=False
        )
        
        # Third inference: Use cached prefix with target suffix (warm - cache hit expected)
        logger.info("Third inference: Testing with cached prefix (measuring cache benefit)...")
        third_prompt = prefix_prompt + suffix_prompt
        
        disk_io_time_ms = None
        if cache_type == "disk":
            disk_io_start = time.time()
        
        third_metrics = self.metrics.collect_inference_metrics(
            llm=llm,
            prompt=third_prompt,
            max_tokens=max_tokens,
            enable_background_monitoring=False
        )
        
        if cache_type == "disk":
            disk_io_end = time.time()
            disk_io_time_ms = (disk_io_end - disk_io_start) * 1000
        
        # Calculate cache hit rate
        # Estimate: prefix tokens should be cached, suffix tokens are new
        total_tokens = third_metrics.prompt_tokens
        estimated_cached_tokens = min(prefix_length, total_tokens)
        cache_hit_rate_pct = (estimated_cached_tokens / total_tokens) * 100 if total_tokens > 0 else 0
        
        # Calculate latency reduction
        baseline_ttft = second_metrics_cold.ttft_ms
        cached_ttft = third_metrics.ttft_ms
        latency_reduction_ms = baseline_ttft - cached_ttft
        latency_reduction_pct = (latency_reduction_ms / baseline_ttft) * 100 if baseline_ttft > 0 else 0
        
        # Measure peak memory
        peak_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        
        logger.info(f"Prompt caching test complete:")
        logger.info(f"  Cache hit rate: {cache_hit_rate_pct:.2f}%")
        logger.info(f"  Latency reduction: {latency_reduction_ms:.2f} ms ({latency_reduction_pct:.2f}%)")
        logger.info(f"  Cache memory overhead: {cache_memory_overhead_mb:.2f} MB ({cache_memory_overhead_pct:.2f}%)")
        if cache_type == "disk":
            logger.info(f"  Disk I/O time: {disk_io_time_ms:.2f} ms")
            logger.info(f"  Cache file size: {cache_file_size_mb:.2f} MB")
        
        # Build metrics dictionary
        metrics = {
            "prefix_length_tokens": prefix_length,
            "cache_hit_rate_pct": round(cache_hit_rate_pct, 2),
            "latency_reduction_ms": round(latency_reduction_ms, 2),
            "latency_reduction_pct": round(latency_reduction_pct, 2),
            "cache_memory_overhead_mb": round(cache_memory_overhead_mb, 2),
            "cache_memory_overhead_pct": round(cache_memory_overhead_pct, 2),
            "baseline_ttft_ms": round(baseline_ttft, 2),
            "cached_ttft_ms": round(cached_ttft, 2),
            "ttft_ms": round(cached_ttft, 2),
            "prefill_tps": round(third_metrics.prefill_tps, 2),
            "decode_tps": round(third_metrics.decode_tps, 2),
            "peak_memory_mb": round(peak_memory_mb, 2),
            "prompt_tokens": third_metrics.prompt_tokens,
            "output_tokens": third_metrics.output_tokens
        }
        
        if cache_type == "disk":
            metrics["disk_io_time_ms"] = round(disk_io_time_ms, 2)
            metrics["cache_file_size_mb"] = round(cache_file_size_mb, 2)
        
        return AblationResult(
            scenario=f"prompt_cache_{cache_type}_{prefix_length}tok",
            configuration={
                "cache_type": cache_type,
                "prefix_length_tokens": prefix_length,
                "cache_enabled": True
            },
            metrics=metrics,
            improvement_over_baseline=round(latency_reduction_pct, 2)
        )
    
    def _measure_cache_directory_size(self, cache_dir: Path) -> float:
        """
        Measure total size of cache directory in megabytes.
        
        Args:
            cache_dir: Path to cache directory
        
        Returns:
            Total size in megabytes
        """
        total_size = 0
        
        try:
            for file_path in cache_dir.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.warning(f"Failed to measure cache directory size: {e}")
            return 0.0
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def _log_prompt_caching_summary(self, results: List[AblationResult]) -> None:
        """
        Log summary of prompt caching test results.
        
        Args:
            results: List of AblationResult objects
        """
        if not results:
            return
        
        logger.info("\n" + "=" * 120)
        logger.info("Prompt Caching Test Summary")
        logger.info("=" * 120)
        
        # Header
        header = (
            f"{'Scenario':<30} {'Prefix (tok)':<12} {'Hit Rate %':<12} "
            f"{'Latency Δ (ms)':<15} {'Cache Mem %':<12} {'Improvement':<12}"
        )
        logger.info(header)
        logger.info("-" * 120)
        
        # Data rows
        for result in results:
            prefix_len = result.metrics.get("prefix_length_tokens", 0)
            hit_rate = result.metrics.get("cache_hit_rate_pct", 0)
            latency_reduction = result.metrics.get("latency_reduction_ms", 0)
            cache_mem_pct = result.metrics.get("cache_memory_overhead_pct", 0)
            improvement = result.improvement_over_baseline
            
            improvement_str = f"{improvement:+.2f}%" if improvement is not None else "N/A"
            
            row = (
                f"{result.scenario:<30} {prefix_len:<12} {hit_rate:<12.2f} "
                f"{latency_reduction:<15.2f} {cache_mem_pct:<12.2f} {improvement_str:<12}"
            )
            logger.info(row)
        
        logger.info("=" * 120)
    
    def _log_quantization_comparison_summary(self, results: List[AblationResult]) -> None:
        """
        Log summary comparing prompt caching across quantization levels.
        
        Args:
            results: List of AblationResult objects with quantization info
        """
        if not results:
            return
        
        logger.info("\n" + "=" * 130)
        logger.info("Prompt Caching Effectiveness Across Quantization Levels")
        logger.info("=" * 130)
        
        # Header
        header = (
            f"{'Quantization':<15} {'Cache Type':<12} {'Prefix (tok)':<12} "
            f"{'Hit Rate %':<12} {'Latency Δ (ms)':<15} {'Cache Mem %':<12} {'Improvement':<12}"
        )
        logger.info(header)
        logger.info("-" * 130)
        
        # Data rows
        for result in results:
            quant = result.configuration.get("quantization", "N/A")
            cache_type = result.configuration.get("cache_type", "N/A")
            prefix_len = result.metrics.get("prefix_length_tokens", 0)
            hit_rate = result.metrics.get("cache_hit_rate_pct", 0)
            latency_reduction = result.metrics.get("latency_reduction_ms", 0)
            cache_mem_pct = result.metrics.get("cache_memory_overhead_pct", 0)
            improvement = result.improvement_over_baseline
            
            improvement_str = f"{improvement:+.2f}%" if improvement is not None else "N/A"
            
            row = (
                f"{quant:<15} {cache_type:<12} {prefix_len:<12} "
                f"{hit_rate:<12.2f} {latency_reduction:<15.2f} {cache_mem_pct:<12.2f} {improvement_str:<12}"
            )
            logger.info(row)
        
        logger.info("=" * 130)
    
    def _log_concurrent_caching_summary(self, results: List[AblationResult]) -> None:
        """
        Log summary of concurrent prompt caching test results.
        
        Args:
            results: List of AblationResult objects for concurrent prompts
        """
        if not results:
            return
        
        logger.info("\n" + "=" * 110)
        logger.info("Concurrent Prompt Caching Summary")
        logger.info("=" * 110)
        
        # Header
        header = (
            f"{'Prompt #':<10} {'TTFT (ms)':<12} {'Prefill (t/s)':<15} "
            f"{'Decode (t/s)':<15} {'Hit Rate %':<12} {'Total Time (s)':<15}"
        )
        logger.info(header)
        logger.info("-" * 110)
        
        # Data rows
        for result in results:
            prompt_idx = result.metrics.get("prompt_index", 0)
            ttft = result.metrics.get("ttft_ms", 0)
            prefill = result.metrics.get("prefill_tps", 0)
            decode = result.metrics.get("decode_tps", 0)
            hit_rate = result.metrics.get("cache_hit_rate_pct", 0)
            total_time = result.metrics.get("total_time_s", 0)
            
            row = (
                f"{prompt_idx:<10} {ttft:<12.2f} {prefill:<15.2f} "
                f"{decode:<15.2f} {hit_rate:<12.2f} {total_time:<15.2f}"
            )
            logger.info(row)
        
        # Calculate and log averages
        if results:
            avg_ttft = sum(r.metrics.get("ttft_ms", 0) for r in results) / len(results)
            avg_hit_rate = sum(r.metrics.get("cache_hit_rate_pct", 0) for r in results) / len(results)
            avg_total_time = sum(r.metrics.get("total_time_s", 0) for r in results) / len(results)
            
            logger.info("-" * 110)
            logger.info(f"{'AVERAGE':<10} {avg_ttft:<12.2f} {'N/A':<15} {'N/A':<15} {avg_hit_rate:<12.2f} {avg_total_time:<15.2f}")
        
        logger.info("=" * 110)
    
    def test_batch_sizes(
        self,
        model_path: str,
        prompts: List[str],
        max_tokens: int = 50,
        batch_sizes: Optional[List[int]] = None
    ) -> List[AblationResult]:
        """
        Test throughput across different batch sizes.
        
        Measures:
        - Aggregate throughput in tokens per second for batch inference
        - Per-prompt latency distribution within batches
        - Memory scaling as batch size increases
        - Optimal batch size maximizing throughput without memory overflow
        - GPU utilization across different batch sizes (when GPU is enabled)
        - Throughput-latency tradeoff curves
        
        Args:
            model_path: Path to GGUF model file
            prompts: List of prompts to use for batch testing (should have at least 16 prompts)
            max_tokens: Maximum tokens to generate per prompt (default: 50)
            batch_sizes: List of batch sizes to test (default: [1, 2, 4, 8, 16])
        
        Returns:
            List of AblationResult objects for each batch size
        
        Raises:
            ValueError: If prompts list is too short for largest batch size
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16]
        
        max_batch_size = max(batch_sizes)
        if len(prompts) < max_batch_size:
            raise ValueError(
                f"Need at least {max_batch_size} prompts for batch size {max_batch_size}, "
                f"but only {len(prompts)} provided"
            )
        
        logger.info("=" * 80)
        logger.info("Starting Batch Processing Tests")
        logger.info("=" * 80)
        logger.info(f"Model: {model_path}")
        logger.info(f"Batch sizes to test: {batch_sizes}")
        logger.info(f"Max tokens per prompt: {max_tokens}")
        logger.info(f"Available prompts: {len(prompts)}")
        
        results: List[AblationResult] = []
        
        try:
            # Measure baseline memory
            baseline_memory_mb = self.process.memory_info().rss / (1024 * 1024)
            
            # Get platform-specific configuration
            llama_config = self.backend.get_llama_config()
            
            # Set context size
            llama_config["n_ctx"] = self.context_size
            
            # Disable verbose output
            llama_config["verbose"] = False
            
            logger.info(f"Model config: {llama_config}")
            
            # Load model once for all batch tests
            logger.info("Loading model...")
            llm = self._load_model(model_path, **llama_config)
            
            post_load_memory_mb = self.process.memory_info().rss / (1024 * 1024)
            model_memory_mb = post_load_memory_mb - baseline_memory_mb
            logger.info(f"Model loaded. Memory: {model_memory_mb:.2f} MB")
            
            # Test each batch size
            for batch_size in sorted(batch_sizes):
                logger.info("\n" + "=" * 60)
                logger.info(f"Testing Batch Size: {batch_size}")
                logger.info("=" * 60)
                
                result = self._run_batch_test(
                    llm=llm,
                    prompts=prompts[:batch_size],
                    max_tokens=max_tokens,
                    batch_size=batch_size,
                    baseline_memory_mb=post_load_memory_mb
                )
                
                results.append(result)
                
                # Small delay between batch tests
                time.sleep(1)
            
            logger.info("\n" + "=" * 80)
            logger.info("Batch Processing Tests Complete")
            logger.info("=" * 80)
            
            # Log summary
            self._log_batch_testing_summary(results)
            
            # Identify optimal batch size
            optimal_batch_size = self._identify_optimal_batch_size(results)
            logger.info(f"\nOptimal batch size: {optimal_batch_size}")
            
            return results
            
        finally:
            # Cleanup is handled by caller
            pass
    
    def _run_batch_test(
        self,
        llm: Any,
        prompts: List[str],
        max_tokens: int,
        batch_size: int,
        baseline_memory_mb: float
    ) -> AblationResult:
        """
        Run a single batch test.
        
        Args:
            llm: Loaded model instance (Llama or NativeLlamaCpp)
            prompts: List of prompts for this batch
            max_tokens: Maximum tokens to generate per prompt
            batch_size: Current batch size being tested
            baseline_memory_mb: Baseline memory after model load
        
        Returns:
            AblationResult with batch testing metrics
        """
        logger.info(f"Processing batch of {batch_size} prompts...")
        
        # Track per-prompt metrics
        per_prompt_latencies: List[float] = []
        per_prompt_ttfts: List[float] = []
        total_prompt_tokens = 0
        total_output_tokens = 0
        
        # Measure batch start time
        batch_start_time = time.time()
        
        # Process each prompt in the batch
        for idx, prompt in enumerate(prompts, 1):
            logger.info(f"  Processing prompt {idx}/{batch_size}...")
            
            prompt_start_time = time.time()
            
            # Collect metrics for this prompt
            metrics = self.metrics.collect_inference_metrics(
                llm=llm,
                prompt=prompt,
                max_tokens=max_tokens,
                enable_background_monitoring=False
            )
            
            prompt_end_time = time.time()
            prompt_latency_s = prompt_end_time - prompt_start_time
            
            # Track metrics
            per_prompt_latencies.append(prompt_latency_s * 1000)  # Convert to ms
            per_prompt_ttfts.append(metrics.ttft_ms)
            total_prompt_tokens += metrics.prompt_tokens
            total_output_tokens += metrics.output_tokens
        
        # Measure batch end time
        batch_end_time = time.time()
        batch_duration_s = batch_end_time - batch_start_time
        
        # Measure memory after batch
        post_batch_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        memory_increase_mb = post_batch_memory_mb - baseline_memory_mb
        
        # Calculate aggregate metrics
        total_tokens = total_prompt_tokens + total_output_tokens
        aggregate_throughput_tps = total_tokens / batch_duration_s if batch_duration_s > 0 else 0
        
        # Calculate latency statistics
        avg_latency_ms = sum(per_prompt_latencies) / len(per_prompt_latencies) if per_prompt_latencies else 0
        min_latency_ms = min(per_prompt_latencies) if per_prompt_latencies else 0
        max_latency_ms = max(per_prompt_latencies) if per_prompt_latencies else 0
        
        # Calculate latency standard deviation
        if len(per_prompt_latencies) > 1:
            mean_latency = avg_latency_ms
            variance = sum((x - mean_latency) ** 2 for x in per_prompt_latencies) / len(per_prompt_latencies)
            std_latency_ms = variance ** 0.5
        else:
            std_latency_ms = 0.0
        
        # Calculate TTFT statistics
        avg_ttft_ms = sum(per_prompt_ttfts) / len(per_prompt_ttfts) if per_prompt_ttfts else 0
        
        # Get GPU metrics if available
        gpu_utilization_pct = None
        gpu_memory_mb = None
        if hasattr(self.backend, 'hw_info') and self.backend.hw_info.has_gpu:
            # GPU metrics would be collected during inference
            # For now, we'll leave as None unless metrics collector provides them
            pass
        
        logger.info(f"Batch complete:")
        logger.info(f"  Aggregate throughput: {aggregate_throughput_tps:.2f} tokens/s")
        logger.info(f"  Avg latency per prompt: {avg_latency_ms:.2f} ms")
        logger.info(f"  Latency range: {min_latency_ms:.2f} - {max_latency_ms:.2f} ms")
        logger.info(f"  Memory increase: {memory_increase_mb:.2f} MB")
        
        return AblationResult(
            scenario=f"batch_size_{batch_size}",
            configuration={
                "batch_size": batch_size,
                "max_tokens": max_tokens,
                "num_prompts": len(prompts)
            },
            metrics={
                "batch_size": batch_size,
                "aggregate_throughput_tps": round(aggregate_throughput_tps, 2),
                "batch_duration_s": round(batch_duration_s, 2),
                "total_tokens": total_tokens,
                "total_prompt_tokens": total_prompt_tokens,
                "total_output_tokens": total_output_tokens,
                "avg_latency_per_prompt_ms": round(avg_latency_ms, 2),
                "min_latency_ms": round(min_latency_ms, 2),
                "max_latency_ms": round(max_latency_ms, 2),
                "std_latency_ms": round(std_latency_ms, 2),
                "avg_ttft_ms": round(avg_ttft_ms, 2),
                "memory_increase_mb": round(memory_increase_mb, 2),
                "peak_memory_mb": round(post_batch_memory_mb, 2),
                "gpu_utilization_pct": gpu_utilization_pct,
                "gpu_memory_mb": gpu_memory_mb
            },
            improvement_over_baseline=None  # Will be calculated relative to batch_size=1
        )
    
    def _identify_optimal_batch_size(self, results: List[AblationResult]) -> int:
        """
        Identify optimal batch size maximizing throughput without memory overflow.
        
        Strategy:
        - Find batch size with highest aggregate throughput
        - Ensure memory increase is reasonable (< 50% of baseline)
        - Ensure latency doesn't degrade too much (< 2x of batch_size=1)
        
        Args:
            results: List of AblationResult objects from batch testing
        
        Returns:
            Optimal batch size
        """
        if not results:
            return 1
        
        # Sort by batch size
        sorted_results = sorted(results, key=lambda r: r.configuration["batch_size"])
        
        # Get baseline (batch_size=1) metrics
        baseline_result = sorted_results[0]
        baseline_latency = baseline_result.metrics.get("avg_latency_per_prompt_ms", 0)
        
        # Find optimal batch size
        best_batch_size = 1
        best_throughput = 0.0
        
        for result in sorted_results:
            batch_size = result.configuration["batch_size"]
            throughput = result.metrics.get("aggregate_throughput_tps", 0)
            avg_latency = result.metrics.get("avg_latency_per_prompt_ms", 0)
            
            # Check if latency is acceptable (< 2x baseline)
            latency_acceptable = (avg_latency < baseline_latency * 2) if baseline_latency > 0 else True
            
            # Update best if throughput is higher and latency is acceptable
            if throughput > best_throughput and latency_acceptable:
                best_throughput = throughput
                best_batch_size = batch_size
        
        return best_batch_size
    
    def _log_batch_testing_summary(self, results: List[AblationResult]) -> None:
        """
        Log summary of batch testing results.
        
        Args:
            results: List of AblationResult objects
        """
        if not results:
            return
        
        logger.info("\n" + "=" * 130)
        logger.info("Batch Processing Test Summary")
        logger.info("=" * 130)
        
        # Header
        header = (
            f"{'Batch Size':<12} {'Throughput (t/s)':<18} {'Avg Latency (ms)':<18} "
            f"{'Latency Range (ms)':<25} {'Memory (MB)':<15} {'Duration (s)':<15}"
        )
        logger.info(header)
        logger.info("-" * 130)
        
        # Data rows
        for result in sorted(results, key=lambda r: r.configuration["batch_size"]):
            batch_size = result.configuration["batch_size"]
            throughput = result.metrics.get("aggregate_throughput_tps", 0)
            avg_latency = result.metrics.get("avg_latency_per_prompt_ms", 0)
            min_latency = result.metrics.get("min_latency_ms", 0)
            max_latency = result.metrics.get("max_latency_ms", 0)
            memory = result.metrics.get("memory_increase_mb", 0)
            duration = result.metrics.get("batch_duration_s", 0)
            
            latency_range = f"{min_latency:.2f} - {max_latency:.2f}"
            
            row = (
                f"{batch_size:<12} {throughput:<18.2f} {avg_latency:<18.2f} "
                f"{latency_range:<25} {memory:<15.2f} {duration:<15.2f}"
            )
            logger.info(row)
        
        logger.info("=" * 130)
    
    def _log_ablation_summary(self, results: List[AblationResult]) -> None:
        """
        Log summary of ablation study results.
        
        Args:
            results: List of AblationResult objects
        """
        if not results:
            return
        
        logger.info("\n" + "=" * 100)
        logger.info("Ablation Study Summary")
        logger.info("=" * 100)
        
        # Header
        header = (
            f"{'Scenario':<25} {'TTFT (ms)':<12} {'Prefill (t/s)':<15} "
            f"{'Decode (t/s)':<15} {'Memory (MB)':<15} {'Improvement':<12}"
        )
        logger.info(header)
        logger.info("-" * 100)
        
        # Data rows
        for result in results:
            ttft = result.metrics.get("ttft_ms", 0)
            prefill = result.metrics.get("prefill_tps", 0)
            decode = result.metrics.get("decode_tps", 0)
            memory = result.metrics.get("peak_memory_mb", 0)
            improvement = result.improvement_over_baseline
            
            improvement_str = f"{improvement:+.2f}%" if improvement is not None else "N/A"
            
            row = (
                f"{result.scenario:<25} {ttft:<12.2f} {prefill:<15.2f} "
                f"{decode:<15.2f} {memory:<15.2f} {improvement_str:<12}"
            )
            logger.info(row)
        
        logger.info("=" * 100)
