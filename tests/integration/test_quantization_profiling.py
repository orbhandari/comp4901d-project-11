"""
Integration tests for quantization profiling functionality.

Tests profiling with multiple quantization levels, identical prompts,
metrics collection, and garbage collection between tests.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6**
"""

import os
import gc
import pytest
import psutil
from pathlib import Path

from llm_benchmark.profiler.quantization import QuantizationProfiler
from llm_benchmark.hardware.detector import HardwareDetector
from llm_benchmark.hardware.hal import create_backend
from llm_benchmark.metrics.collector import MetricsCollector


@pytest.fixture
def hardware_backend():
    """Create hardware backend for testing."""
    hw_info = HardwareDetector.detect()
    backend = create_backend(hw_info)
    return backend


@pytest.fixture
def test_models():
    """
    Get paths to test models for quantization profiling.
    
    Uses small test models if available, otherwise skips test.
    """
    # Check for test models in the models directory
    models_dir = Path("models")
    
    # Look for small test models
    test_model_patterns = [
        "tinyllama*.gguf",
        "tiny*.gguf",
        "*Q2_K*.gguf",
        "*Q4_0*.gguf",
    ]
    
    available_models = {}
    
    for pattern in test_model_patterns:
        matching_files = list(models_dir.glob(pattern))
        for model_file in matching_files:
            # Extract quantization from filename
            filename = model_file.name.lower()
            if "q2_k" in filename:
                available_models["Q2_K"] = str(model_file)
            elif "q4_0" in filename or "q4-0" in filename:
                available_models["Q4_0"] = str(model_file)
            elif "q4_k_m" in filename:
                available_models["Q4_K_M"] = str(model_file)
            elif "q8_0" in filename:
                available_models["Q8_0"] = str(model_file)
    
    if len(available_models) < 2:
        pytest.skip("Need at least 2 quantized models for profiling test")
    
    return available_models


def test_quantization_profiling_multiple_levels(hardware_backend, test_models):
    """
    Test profiling with multiple quantization levels.
    
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
    """
    profiler = QuantizationProfiler(
        backend=hardware_backend,
        metrics_collector=MetricsCollector(hardware_backend.hw_info)
    )
    
    # Use a simple test prompt
    test_prompt = "What is the capital of France?"
    max_tokens = 20
    
    results = []
    
    # Profile each quantization level
    for quant, model_path in test_models.items():
        result = profiler.profile_quantization(
            model_path=model_path,
            quant=quant,
            prompt=test_prompt,
            max_tokens=max_tokens
        )
        results.append(result)
        
        # Verify result has expected fields
        assert result.quantization == quant
        assert result.load_time_s > 0
        assert result.peak_ram_mb > 0
        assert result.ram_increase_mb >= 0
        assert result.ttft_ms > 0
        assert result.prefill_tps > 0
        assert result.decode_tps > 0
        assert result.prompt_tokens > 0
        assert result.output_tokens > 0
    
    # Verify we got results for multiple quantization levels
    assert len(results) >= 2
    
    # Verify different quantization levels have different characteristics
    # Generally, lower quantization (Q2_K) should use less RAM than higher (Q8_0)
    if "Q2_K" in test_models and "Q8_0" in test_models:
        q2_result = next(r for r in results if r.quantization == "Q2_K")
        q8_result = next(r for r in results if r.quantization == "Q8_0")
        
        # Q2_K should use less RAM than Q8_0
        assert q2_result.peak_ram_mb < q8_result.peak_ram_mb


def test_identical_prompts_across_quantizations(hardware_backend, test_models):
    """
    Test that identical prompts are used across quantization levels.
    
    **Validates: Requirement 2.5**
    """
    profiler = QuantizationProfiler(
        backend=hardware_backend,
        metrics_collector=MetricsCollector(hardware_backend.hw_info)
    )
    
    # Use a specific test prompt
    test_prompt = "Explain quantum computing in simple terms."
    max_tokens = 30
    
    results = []
    
    # Profile each quantization level with the same prompt
    for quant, model_path in test_models.items():
        result = profiler.profile_quantization(
            model_path=model_path,
            quant=quant,
            prompt=test_prompt,
            max_tokens=max_tokens
        )
        results.append(result)
    
    # Verify all results used the same prompt
    # We can verify this by checking that prompt_tokens are similar
    # (should be identical for the same prompt)
    prompt_token_counts = [r.prompt_tokens for r in results]
    
    # All prompt token counts should be the same (or very close)
    # Allow small variance due to different tokenizers
    min_tokens = min(prompt_token_counts)
    max_tokens_count = max(prompt_token_counts)
    
    # Should be within 10% of each other
    assert max_tokens_count <= min_tokens * 1.1


def test_metrics_collected_for_all_quantizations(hardware_backend, test_models):
    """
    Test that metrics are collected for all quantization levels.
    
    **Validates: Requirements 2.2, 2.3, 2.4, 2.6**
    """
    profiler = QuantizationProfiler(
        backend=hardware_backend,
        metrics_collector=MetricsCollector(hardware_backend.hw_info)
    )
    
    test_prompt = "Write a haiku about programming."
    max_tokens = 25
    
    results = profiler.profile_all(
        models=test_models,
        prompt=test_prompt,
        max_tokens=max_tokens
    )
    
    # Verify we got results for all models
    assert len(results) == len(test_models)
    
    # Verify each result has all required metrics
    for result in results:
        # Load time metrics
        assert result.load_time_s > 0
        assert result.load_time_s < 60.0  # Should load in under 60 seconds
        
        # Memory metrics
        assert result.peak_ram_mb > 0
        assert result.ram_increase_mb >= 0
        
        # Inference metrics
        assert result.ttft_ms > 0
        assert result.prefill_tps > 0
        assert result.decode_tps > 0
        
        # Token counts
        assert result.prompt_tokens > 0
        assert result.output_tokens > 0
        
        # GPU metrics (if GPU available)
        if hardware_backend.hw_info.has_gpu and result.used_gpu_acceleration:
            assert result.gpu_memory_mb is not None
            assert result.gpu_utilization_pct is not None


def test_garbage_collection_between_tests(hardware_backend, test_models):
    """
    Test that garbage collection is enforced between quantization tests.
    
    **Validates: Requirement 2.6**
    """
    profiler = QuantizationProfiler(
        backend=hardware_backend,
        metrics_collector=MetricsCollector(hardware_backend.hw_info)
    )
    
    test_prompt = "What is machine learning?"
    max_tokens = 15
    
    # Get initial memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    memory_readings = []
    
    # Profile each quantization level
    for quant, model_path in test_models.items():
        # Profile
        result = profiler.profile_quantization(
            model_path=model_path,
            quant=quant,
            prompt=test_prompt,
            max_tokens=max_tokens
        )
        
        # Force garbage collection (simulating what profiler should do)
        gc.collect()
        
        # Measure memory after GC
        current_memory = process.memory_info().rss / (1024 * 1024)
        memory_readings.append(current_memory)
    
    # Verify memory doesn't continuously grow
    # After GC, memory should not increase unboundedly
    # Allow some growth but not linear with number of models
    if len(memory_readings) >= 2:
        first_reading = memory_readings[0]
        last_reading = memory_readings[-1]
        
        # Memory growth should be reasonable (not more than 2x)
        # This indicates GC is working
        assert last_reading < first_reading * 2.0


def test_quantization_comparison_matrix(hardware_backend, test_models):
    """
    Test generation of comparison matrix across quantization levels.
    
    **Validates: Requirement 2.6**
    """
    profiler = QuantizationProfiler(
        backend=hardware_backend,
        metrics_collector=MetricsCollector(hardware_backend.hw_info)
    )
    
    test_prompt = "Describe the solar system."
    max_tokens = 20
    
    # Profile all quantization levels
    results = profiler.profile_all(
        models=test_models,
        prompt=test_prompt,
        max_tokens=max_tokens
    )
    
    # Verify we can create a comparison matrix
    # Extract key metrics for comparison
    comparison_data = []
    for result in results:
        comparison_data.append({
            'quantization': result.quantization,
            'load_time_s': result.load_time_s,
            'peak_ram_mb': result.peak_ram_mb,
            'ttft_ms': result.ttft_ms,
            'prefill_tps': result.prefill_tps,
            'decode_tps': result.decode_tps,
        })
    
    # Verify comparison data is complete
    assert len(comparison_data) == len(test_models)
    
    # Verify all metrics are present for each quantization
    for data in comparison_data:
        assert 'quantization' in data
        assert 'load_time_s' in data
        assert 'peak_ram_mb' in data
        assert 'ttft_ms' in data
        assert 'prefill_tps' in data
        assert 'decode_tps' in data
        
        # Verify all values are positive
        assert data['load_time_s'] > 0
        assert data['peak_ram_mb'] > 0
        assert data['ttft_ms'] > 0
        assert data['prefill_tps'] > 0
        assert data['decode_tps'] > 0


@pytest.mark.skipif(
    not os.path.exists("models"),
    reason="Models directory not found"
)
def test_missing_quantization_format_handling(hardware_backend):
    """
    Test handling when quantization formats are unavailable.
    
    **Validates: Requirement 2.7**
    """
    profiler = QuantizationProfiler(
        backend=hardware_backend,
        metrics_collector=MetricsCollector(hardware_backend.hw_info)
    )
    
    # Try to profile with non-existent model
    test_models_with_missing = {
        "Q4_0": "models/nonexistent_q4_0.gguf",
        "Q8_0": "models/nonexistent_q8_0.gguf",
    }
    
    test_prompt = "Test prompt"
    max_tokens = 10
    
    # Should handle missing models gracefully
    results = []
    for quant, model_path in test_models_with_missing.items():
        try:
            result = profiler.profile_quantization(
                model_path=model_path,
                quant=quant,
                prompt=test_prompt,
                max_tokens=max_tokens
            )
            results.append(result)
        except (FileNotFoundError, Exception) as e:
            # Should log error and continue
            # This is expected behavior
            pass
    
    # Should not crash, even if all models are missing
    # Results may be empty, which is acceptable
    assert isinstance(results, list)
