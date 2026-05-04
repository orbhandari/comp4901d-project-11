"""
Android-specific Ablation Engine for native llama.cpp CLI.

This module provides ablation testing for Android/Termux using native llama.cpp's
--prompt-cache feature instead of llama-cpp-python's cache API.

Key differences from standard AblationEngine:
- Uses --prompt-cache flag for disk-based caching
- Uses --prompt-cache-all for full prompt caching
- RAM cache is the default KV cache (always enabled, can't be disabled)
- Disk cache uses temporary files via --prompt-cache
- Control runs still have KV cache (can't be disabled in native llama.cpp)
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


class AndroidAblationEngine:
    """
    Android-specific ablation engine using native llama.cpp CLI.
    
    Implements caching tests using --prompt-cache flag instead of
    llama-cpp-python's cache API.
    """
    
    def __init__(self, backend: HardwareBackend, metrics_collector: MetricsCollector, context_size: int = 2048):
        """
        Initialize Android ablation engine.
        
        Args:
            backend: Hardware backend (must be AndroidBackend)
            metrics_collector: Metrics collector for inference measurements
            context_size: Context window size for model (default: 2048)
        """
        self.backend = backend
        self.metrics = metrics_collector
        self.process = psutil.Process()
        self.context_size = context_size
        
        # Temporary directories and files for cache testing
        self.temp_dirs: List[Path] = []
        self.temp_files: List[Path] = []
        
        logger.info("AndroidAblationEngine initialized")
        logger.info("Using native llama.cpp --prompt-cache for caching tests")
    
    def _load_model(self, model_path: str, **kwargs) -> Any:
        """
        Load model using backend's load_model_safe() method.
        
        Args:
            model_path: Path to GGUF model file
            **kwargs: Additional arguments for model loading
        
        Returns:
            Loaded model instance (NativeLlamaCpp)
        
        Raises:
            RuntimeError: If model loading fails
        """
        llm = self.backend.load_model_safe(model_path, **kwargs)
        if llm is None:
            raise RuntimeError(f"Failed to load model: {model_path}")
        return llm
    
    def test_prompt_cache_strategies(
        self,
        model_path: str,
        prompt_prefix: str,
        prompt_suffix: str,
        max_tokens: int = 50
    ) -> List[AblationResult]:
        """
        Test prompt caching using native llama.cpp's --prompt-cache feature.
        
        Implements the following test scenarios:
        1. Control run: No prompt cache file (baseline with KV cache)
        2. Cold run (Disk): Prompt cache file created but not used
        3. Warm run (Disk): Prompt cache file loaded from previous run
        
        Note: RAM cache (KV cache) is always enabled in native llama.cpp and
        cannot be disabled. The "control" run still has KV cache active.
        
        Args:
            model_path: Path to GGUF model file
            prompt_prefix: Shared prefix for cache effectiveness
            prompt_suffix: Unique suffix to append after prefix
            max_tokens: Maximum tokens to generate (default: 50)
        
        Returns:
            List of AblationResult objects for each test scenario
        """
        # Validate prompt prefix length
        estimated_tokens = len(prompt_prefix) // 4
        if estimated_tokens < 100:
            logger.warning(
                f"Prompt prefix may be too short ({estimated_tokens} estimated tokens). "
                f"Recommend at least 500 tokens for effective cache testing."
            )
        
        logger.info("=" * 80)
        logger.info("Starting Android Prompt Cache Ablation Studies")
        logger.info("=" * 80)
        logger.info(f"Model: {model_path}")
        logger.info(f"Prompt prefix length: ~{estimated_tokens} tokens (estimated)")
        logger.info(f"Max tokens: {max_tokens}")
        logger.info("")
        logger.info("Note: Native llama.cpp always has KV cache enabled (RAM cache)")
        logger.info("      Control run will have KV cache, not a true 'no cache' baseline")
        
        results: List[AblationResult] = []
        
        try:
            # 1. Control run: No prompt cache file (but KV cache is still active)
            logger.info("\n" + "=" * 60)
            logger.info("Test 1: Control Run (No Prompt Cache File)")
            logger.info("=" * 60)
            logger.info("Note: KV cache (RAM) is still active - cannot be disabled")
            
            control_result = self._run_control_android(
                model_path=model_path,
                prompt=prompt_prefix + prompt_suffix,
                max_tokens=max_tokens
            )
            results.append(control_result)
            
            self._ensure_process_isolation()
            
            # 2. Cold run: Create prompt cache file but don't use it
            logger.info("\n" + "=" * 60)
            logger.info("Test 2: Disk Cache - Cold Run (Create Cache)")
            logger.info("=" * 60)
            
            disk_cold_result = self._run_cold_cache_android(
                model_path=model_path,
                prompt=prompt_prefix + prompt_suffix,
                max_tokens=max_tokens,
                baseline_ttft=control_result.metrics.get("ttft_ms")
            )
            results.append(disk_cold_result)
            
            self._ensure_process_isolation()
            
            # 3. Warm run: Load prompt cache file from previous run
            logger.info("\n" + "=" * 60)
            logger.info("Test 3: Disk Cache - Warm Run (Load Cache)")
            logger.info("=" * 60)
            
            disk_warm_result = self._run_warm_cache_android(
                model_path=model_path,
                prompt_prefix=prompt_prefix,
                prompt_suffix=prompt_suffix,
                max_tokens=max_tokens,
                baseline_ttft=control_result.metrics.get("ttft_ms")
            )
            results.append(disk_warm_result)
            
            self._ensure_process_isolation()
            
            logger.info("\n" + "=" * 80)
            logger.info("Android Prompt Cache Ablation Studies Complete")
            logger.info("=" * 80)
            
            # Log summary
            self._log_ablation_summary(results)
            
            return results
            
        finally:
            # Clean up cache files and directories
            self._cleanup_cache_resources()
    
    def _run_control_android(
        self,
        model_path: str,
        prompt: str,
        max_tokens: int
    ) -> AblationResult:
        """
        Execute control run without prompt cache file.
        
        Note: KV cache (RAM) is still active - native llama.cpp cannot disable it.
        
        Args:
            model_path: Path to GGUF model file
            prompt: Test prompt
            max_tokens: Maximum tokens to generate
        
        Returns:
            AblationResult with baseline measurements
        """
        logger.info("Running control (no prompt cache file) baseline...")
        logger.info("Note: KV cache (RAM) is still active")
        
        # Measure baseline memory
        baseline_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        
        # Get platform-specific configuration
        llama_config = self.backend.get_llama_config()
        llama_config["n_ctx"] = self.context_size
        
        # Load model (NativeLlamaCpp)
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
        
        return AblationResult(
            scenario="control_no_prompt_cache",
            configuration={
                "prompt_cache_enabled": False,
                "kv_cache_enabled": True,  # Always true in native llama.cpp
                "cache_type": "ram_kv_only",
                "cache_state": "N/A"
            },
            metrics={
                "ttft_ms": metrics.ttft_ms,
                "prefill_tps": metrics.prefill_tps,
                "decode_tps": metrics.decode_tps,
                "memory_overhead_mb": round(memory_overhead_mb, 2),
                "peak_memory_mb": round(peak_memory_mb, 2),
            },
            improvement_over_baseline=None
        )
    
    def _run_cold_cache_android(
        self,
        model_path: str,
        prompt: str,
        max_tokens: int,
        baseline_ttft: Optional[float] = None
    ) -> AblationResult:
        """
        Execute cold run with prompt cache file creation.
        
        Creates a prompt cache file but doesn't load from it (first run).
        
        Args:
            model_path: Path to GGUF model file
            prompt: Test prompt
            max_tokens: Maximum tokens to generate
            baseline_ttft: Baseline TTFT for improvement calculation
        
        Returns:
            AblationResult with cold cache measurements
        """
        logger.info("Running cold cache (create prompt cache file)...")
        
        # Create temporary cache file
        cache_file = self._create_temp_cache_file()
        logger.info(f"Cache file: {cache_file}")
        
        # Measure baseline memory
        baseline_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        
        # Get configuration
        llama_config = self.backend.get_llama_config()
        llama_config["n_ctx"] = self.context_size
        
        # Add prompt cache parameters for NativeLlamaCpp
        # Note: We need to modify NativeLlamaCpp to support these
        llama_config["prompt_cache"] = str(cache_file)
        llama_config["prompt_cache_all"] = True
        
        # Load model
        llm = self._load_model(model_path, **llama_config)
        
        # Measure memory after load
        post_load_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        memory_overhead_mb = post_load_memory_mb - baseline_memory_mb
        
        # Run inference (this will CREATE the cache file)
        metrics = self.metrics.collect_inference_metrics(
            llm=llm,
            prompt=prompt,
            max_tokens=max_tokens,
            enable_background_monitoring=False
        )
        
        # Measure peak memory and cache file size
        peak_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        cache_file_size_mb = cache_file.stat().st_size / (1024 * 1024) if cache_file.exists() else 0
        
        # Calculate improvement
        improvement_pct = None
        if baseline_ttft and baseline_ttft > 0:
            improvement_pct = ((baseline_ttft - metrics.ttft_ms) / baseline_ttft) * 100
        
        logger.info(f"Cold cache run complete:")
        logger.info(f"  TTFT: {metrics.ttft_ms:.2f} ms")
        logger.info(f"  Cache file size: {cache_file_size_mb:.2f} MB")
        logger.info(f"  Memory overhead: {memory_overhead_mb:.2f} MB")
        logger.info(f"  Peak memory: {peak_memory_mb:.2f} MB")
        if improvement_pct is not None:
            logger.info(f"  Improvement over baseline: {improvement_pct:+.2f}%")
        
        return AblationResult(
            scenario="cold_cache_disk_create",
            configuration={
                "prompt_cache_enabled": True,
                "kv_cache_enabled": True,
                "cache_type": "disk",
                "cache_state": "cold_create",
                "cache_file": str(cache_file)
            },
            metrics={
                "ttft_ms": metrics.ttft_ms,
                "prefill_tps": metrics.prefill_tps,
                "decode_tps": metrics.decode_tps,
                "memory_overhead_mb": round(memory_overhead_mb, 2),
                "peak_memory_mb": round(peak_memory_mb, 2),
                "cache_file_size_mb": round(cache_file_size_mb, 2),
            },
            improvement_over_baseline=round(improvement_pct, 2) if improvement_pct is not None else None
        )
    
    def _run_warm_cache_android(
        self,
        model_path: str,
        prompt_prefix: str,
        prompt_suffix: str,
        max_tokens: int,
        baseline_ttft: Optional[float] = None
    ) -> AblationResult:
        """
        Execute warm run with prompt cache file loading.
        
        Loads prompt cache file from previous run (warm cache).
        
        Args:
            model_path: Path to GGUF model file
            prompt_prefix: Shared prefix (should be cached)
            prompt_suffix: Unique suffix
            max_tokens: Maximum tokens to generate
            baseline_ttft: Baseline TTFT for improvement calculation
        
        Returns:
            AblationResult with warm cache measurements
        """
        logger.info("Running warm cache (load prompt cache file)...")
        
        # Create temporary cache file
        cache_file = self._create_temp_cache_file()
        logger.info(f"Cache file: {cache_file}")
        
        # Measure baseline memory
        baseline_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        
        # Get configuration
        llama_config = self.backend.get_llama_config()
        llama_config["n_ctx"] = self.context_size
        llama_config["prompt_cache"] = str(cache_file)
        llama_config["prompt_cache_all"] = True
        
        # Load model
        llm = self._load_model(model_path, **llama_config)
        
        # Measure memory after load
        post_load_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        
        # First inference: Populate cache with prefix
        logger.info("Populating cache with prefix...")
        _ = self.metrics.collect_inference_metrics(
            llm=llm,
            prompt=prompt_prefix,
            max_tokens=10,  # Short generation just to populate cache
            enable_background_monitoring=False
        )
        
        # Measure memory after cache population
        post_cache_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        cache_memory_overhead_mb = post_cache_memory_mb - post_load_memory_mb
        cache_file_size_mb = cache_file.stat().st_size / (1024 * 1024) if cache_file.exists() else 0
        
        logger.info(f"Cache populated. File size: {cache_file_size_mb:.2f} MB")
        
        # Reload model to test cache loading
        del llm
        gc.collect()
        time.sleep(1)
        
        llm = self._load_model(model_path, **llama_config)
        
        # Second inference: Use cached prefix + new suffix
        logger.info("Running inference with cached prefix...")
        metrics = self.metrics.collect_inference_metrics(
            llm=llm,
            prompt=prompt_prefix + prompt_suffix,
            max_tokens=max_tokens,
            enable_background_monitoring=False
        )
        
        # Measure peak memory
        peak_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        
        # Calculate improvement
        improvement_pct = None
        if baseline_ttft and baseline_ttft > 0:
            improvement_pct = ((baseline_ttft - metrics.ttft_ms) / baseline_ttft) * 100
        
        logger.info(f"Warm cache run complete:")
        logger.info(f"  TTFT: {metrics.ttft_ms:.2f} ms")
        logger.info(f"  Cache file size: {cache_file_size_mb:.2f} MB")
        logger.info(f"  Cache memory overhead: {cache_memory_overhead_mb:.2f} MB")
        logger.info(f"  Peak memory: {peak_memory_mb:.2f} MB")
        if improvement_pct is not None:
            logger.info(f"  Improvement over baseline: {improvement_pct:+.2f}%")
        
        return AblationResult(
            scenario="warm_cache_disk_load",
            configuration={
                "prompt_cache_enabled": True,
                "kv_cache_enabled": True,
                "cache_type": "disk",
                "cache_state": "warm_load",
                "cache_file": str(cache_file)
            },
            metrics={
                "ttft_ms": metrics.ttft_ms,
                "prefill_tps": metrics.prefill_tps,
                "decode_tps": metrics.decode_tps,
                "cache_memory_overhead_mb": round(cache_memory_overhead_mb, 2),
                "peak_memory_mb": round(peak_memory_mb, 2),
                "cache_file_size_mb": round(cache_file_size_mb, 2),
            },
            improvement_over_baseline=round(improvement_pct, 2) if improvement_pct is not None else None
        )
    
    def _ensure_process_isolation(self) -> None:
        """Enforce process isolation between test runs."""
        gc.collect()
        time.sleep(2)  # Allow system to stabilize
    
    def _create_temp_cache_file(self) -> Path:
        """
        Create temporary cache file for prompt caching.
        
        Returns:
            Path to temporary cache file
        """
        fd, path = tempfile.mkstemp(suffix=".cache", prefix="llama_prompt_")
        os.close(fd)  # Close file descriptor
        cache_file = Path(path)
        self.temp_files.append(cache_file)
        return cache_file
    
    def _cleanup_cache_resources(self) -> None:
        """Clean up temporary cache files and directories."""
        logger.info("Cleaning up cache resources...")
        
        # Clean up temp files
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.debug(f"Removed temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_file}: {e}")
        
        # Clean up temp directories
        for temp_dir in self.temp_dirs:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Removed temp directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove temp directory {temp_dir}: {e}")
        
        self.temp_files.clear()
        self.temp_dirs.clear()
        
        logger.info("Cache cleanup complete")
    
    def _log_ablation_summary(self, results: List[AblationResult]) -> None:
        """
        Log summary of ablation study results.
        
        Args:
            results: List of ablation results
        """
        logger.info("\n" + "=" * 80)
        logger.info("ABLATION STUDY SUMMARY")
        logger.info("=" * 80)
        
        for result in results:
            logger.info(f"\nScenario: {result.scenario}")
            logger.info(f"  Configuration: {result.configuration}")
            logger.info(f"  Metrics:")
            for metric, value in result.metrics.items():
                logger.info(f"    {metric}: {value}")
            if result.improvement_over_baseline is not None:
                logger.info(f"  Improvement: {result.improvement_over_baseline:+.2f}%")
        
        logger.info("\n" + "=" * 80)
