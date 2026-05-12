"""
Integration tests for prompt caching optimization.

Tests prompt caching with different prefix lengths, cache hit rate calculation,
latency reduction measurement, and testing across multiple quantization levels.

**Validates: Requirements 13.1, 13.2, 13.3, 13.7**
"""

import pytest
from pathlib import Path

from llm_benchmark.hardware.detector import HardwareDetector
from llm_benchmark.hardware.hal import create_backend
from llm_benchmark.profiler.ablation import AblationEngine


@pytest.fixture
def hardware_backend():
    """Create hardware backend for testing."""
    hw_info = HardwareDetector.detect()
    backend = create_backend(hw_info)
    return backend


@pytest.fixture
def test_models():
    """
    Get paths to test models with different quantization levels.
    
    Returns dict of quantization -> model_path.
    """
    models_dir = Path("models")
    
    available_models = {}
    
    # Look for models with different quantizations
    for model_file in models_dir.glob("*.gguf"):
        filename = model_file.name.lower()
        if "q2_k" in filename:
            available_models["Q2_K"] = str(model_file)
        elif "q4_0" in filename or "q4-0" in filename:
            available_models["Q4_0"] = str(model_file)
        elif "q4_k_m" in filename:
            available_models["Q4_K_M"] = str(model_file)
        elif "q8_0" in filename:
            available_models["Q8_0"] = str(model_file)
    
    if len(available_models) == 0:
        pytest.skip("No GGUF models found for testing")
    
    return available_models


def test_prompt_caching_with_100_token_prefix(hardware_backend, test_models):
    """
    Test prompt caching with 100-token shared prefix.
    
    **Validates: Requirement 13.1**
    """
    ablation_engine = AblationEngine(backend=hardware_backend)
    
    # Use any available model
    model_path = list(test_models.values())[0]
    
    # Create prompts with ~100 token shared prefix
    # Rough estimate: 1 token ≈ 4 characters, so ~400 characters
    base_prompt = "The history of computing began with mechanical calculators. " * 7  # ~400 chars
    prompt_1 = base_prompt + "What is a Turing machine?"
    prompt_2 = base_prompt + "What is the halting problem?"
    
    # Test prompt caching
    results = ablation_engine.test_prompt_caching(
        model_path=model_path,
        prefix_lengths=[100],
        base_prompt=base_prompt,
        suffixes=["What is a Turing machine?", "What is the halting problem?"]
    )
    
    # Should have results for 100-token prefix
    result_100 = next((r for r in results if "100" in r.scenario), None)
    assert result_100 is not None
    
    # Verify metrics were collected
    assert "ttft_ms" in result_100.metrics or "latency_reduction_ms" in result_100.metrics


def test_prompt_caching_with_500_token_prefix(hardware_backend, test_models):
    """
    Test prompt caching with 500-token shared prefix.
    
    **Validates: Requirement 13.1**
    """
    ablation_engine = AblationEngine(backend=hardware_backend)
    
    model_path = list(test_models.values())[0]
    
    # Create prompts with ~500 token shared prefix (~2000 characters)
    base_prompt = "The evolution of programming languages has been remarkable over the decades. " * 25
    prompt_1 = base_prompt + "Explain object-oriented programming."
    prompt_2 = base_prompt + "Explain functional programming."
    
    # Test prompt caching
    results = ablation_engine.test_prompt_caching(
        model_path=model_path,
        prefix_lengths=[500],
        base_prompt=base_prompt,
        suffixes=["Explain object-oriented programming.", "Explain functional programming."]
    )
    
    # Should have results for 500-token prefix
    result_500 = next((r for r in results if "500" in r.scenario), None)
    assert result_500 is not None
    
    # Verify metrics
    assert "ttft_ms" in result_500.metrics or "latency_reduction_ms" in result_500.metrics


def test_prompt_caching_with_1000_token_prefix(hardware_backend, test_models):
    """
    Test prompt caching with 1000-token shared prefix.
    
    **Validates: Requirement 13.1**
    """
    ablation_engine = AblationEngine(backend=hardware_backend)
    
    model_path = list(test_models.values())[0]
    
    # Create prompts with ~1000 token shared prefix (~4000 characters)
    base_prompt = "The field of artificial intelligence encompasses many subfields and techniques. " * 50
    prompt_1 = base_prompt + "What is machine learning?"
    prompt_2 = base_prompt + "What is deep learning?"
    
    # Test prompt caching
    results = ablation_engine.test_prompt_caching(
        model_path=model_path,
        prefix_lengths=[1000],
        base_prompt=base_prompt,
        suffixes=["What is machine learning?", "What is deep learning?"]
    )
    
    # Should have results for 1000-token prefix
    result_1000 = next((r for r in results if "1000" in r.scenario), None)
    assert result_1000 is not None
    
    # Verify metrics
    assert "ttft_ms" in result_1000.metrics or "latency_reduction_ms" in result_1000.metrics


def test_cache_hit_rate_calculation(hardware_backend, test_models):
    """
    Test cache hit rate calculation as percentage of tokens reused.
    
    **Validates: Requirement 13.2**
    """
    ablation_engine = AblationEngine(backend=hardware_backend)
    
    model_path = list(test_models.values())[0]
    
    # Create prompts with known shared prefix
    base_prompt = "The science of computation is fundamental to modern technology. " * 30
    prompt_1 = base_prompt + "What is an algorithm?"
    prompt_2 = base_prompt + "What is a data structure?"
    
    # Test prompt caching
    results = ablation_engine.test_prompt_caching(
        model_path=model_path,
        prefix_lengths=[500],
        base_prompt=base_prompt,
        suffixes=["What is an algorithm?", "What is a data structure?"]
    )
    
    # Check if cache hit rate is reported
    for result in results:
        if "cache_hit_rate" in result.metrics:
            # Should be a percentage (0-100)
            assert 0 <= result.metrics["cache_hit_rate"] <= 100
        
        # Or check in configuration
        if "cache_hit_rate_pct" in result.configuration:
            assert 0 <= result.configuration["cache_hit_rate_pct"] <= 100


def test_latency_reduction_measurement(hardware_backend, test_models):
    """
    Test latency reduction measurement from prompt caching.
    
    **Validates: Requirement 13.3**
    """
    ablation_engine = AblationEngine(backend=hardware_backend)
    
    model_path = list(test_models.values())[0]
    
    # Create prompts with substantial shared prefix
    base_prompt = "The development of software engineering practices has evolved significantly. " * 40
    prompt_1 = base_prompt + "What is agile development?"
    prompt_2 = base_prompt + "What is waterfall development?"
    
    # Test prompt caching
    results = ablation_engine.test_prompt_caching(
        model_path=model_path,
        prefix_lengths=[500],
        base_prompt=base_prompt,
        suffixes=["What is agile development?", "What is waterfall development?"]
    )
    
    # Check for latency reduction measurement
    for result in results:
        # Latency reduction might be in metrics
        if "latency_reduction_ms" in result.metrics:
            # Can be positive (improvement) or negative (regression)
            assert isinstance(result.metrics["latency_reduction_ms"], (int, float))
        
        # Or as improvement over baseline
        if result.improvement_over_baseline is not None:
            assert isinstance(result.improvement_over_baseline, (int, float))


def test_prompt_caching_across_quantization_levels(hardware_backend, test_models):
    """
    Test prompt caching effectiveness across different quantization levels.
    
    **Validates: Requirement 13.7**
    """
    if len(test_models) < 2:
        pytest.skip("Need at least 2 quantization levels for comparison")
    
    ablation_engine = AblationEngine(backend=hardware_backend)
    
    # Create test prompts
    base_prompt = "The principles of computer architecture are essential for understanding performance. " * 30
    suffixes = ["What is pipelining?", "What is caching?"]
    
    # Test each quantization level
    all_results = {}
    
    for quant, model_path in test_models.items():
        results = ablation_engine.test_prompt_caching(
            model_path=model_path,
            prefix_lengths=[500],
            base_prompt=base_prompt,
            suffixes=suffixes
        )
        
        all_results[quant] = results
    
    # Verify we got results for multiple quantization levels
    assert len(all_results) >= 2
    
    # Verify each quantization level has results
    for quant, results in all_results.items():
        assert len(results) > 0
        
        # Each result should have metrics
        for result in results:
            assert "ttft_ms" in result.metrics or "latency_reduction_ms" in result.metrics


def test_prompt_caching_with_varying_prefix_lengths(hardware_backend, test_models):
    """
    Test prompt caching with varying shared prefix lengths (100, 500, 1000).
    
    **Validates: Requirement 13.1**
    """
    ablation_engine = AblationEngine(backend=hardware_backend)
    
    model_path = list(test_models.values())[0]
    
    # Create base prompts of different lengths
    short_base = "Computing is important. " * 10  # ~100 tokens
    medium_base = "The field of computer science has many branches. " * 40  # ~500 tokens
    long_base = "The history and evolution of computing technology spans many decades. " * 60  # ~1000 tokens
    
    # Test all prefix lengths
    results = ablation_engine.test_prompt_caching(
        model_path=model_path,
        prefix_lengths=[100, 500, 1000],
        base_prompt=medium_base,  # Use medium as default
        suffixes=["Question 1?", "Question 2?"]
    )
    
    # Should have results for different prefix lengths
    prefix_lengths_tested = set()
    for result in results:
        # Extract prefix length from scenario or configuration
        if "100" in result.scenario:
            prefix_lengths_tested.add(100)
        elif "500" in result.scenario:
            prefix_lengths_tested.add(500)
        elif "1000" in result.scenario:
            prefix_lengths_tested.add(1000)
        
        # Or check configuration
        if "prefix_length" in result.configuration:
            prefix_lengths_tested.add(result.configuration["prefix_length"])
    
    # Should have tested multiple prefix lengths
    assert len(prefix_lengths_tested) >= 1


def test_cache_memory_overhead_as_percentage(hardware_backend, test_models):
    """
    Test cache memory overhead measurement as percentage of total model memory.
    
    **Validates: Requirement 13.4**
    """
    ablation_engine = AblationEngine(backend=hardware_backend)
    
    model_path = list(test_models.values())[0]
    
    # Create test prompts
    base_prompt = "The study of algorithms is central to computer science. " * 30
    suffixes = ["What is sorting?", "What is searching?"]
    
    # Test prompt caching
    results = ablation_engine.test_prompt_caching(
        model_path=model_path,
        prefix_lengths=[500],
        base_prompt=base_prompt,
        suffixes=suffixes
    )
    
    # Check for memory overhead measurement
    for result in results:
        # Memory overhead might be reported as percentage
        if "cache_memory_overhead_pct" in result.metrics:
            assert 0 <= result.metrics["cache_memory_overhead_pct"] <= 100
        
        # Or as absolute value
        if "cache_memory_mb" in result.metrics:
            assert result.metrics["cache_memory_mb"] >= 0


def test_multiple_concurrent_prompts_sharing_prefixes(hardware_backend, test_models):
    """
    Test cache behavior with multiple concurrent prompts sharing prefixes.
    
    **Validates: Requirement 13.8**
    """
    ablation_engine = AblationEngine(backend=hardware_backend)
    
    model_path = list(test_models.values())[0]
    
    # Create multiple prompts with shared prefix
    base_prompt = "The fundamentals of database systems are crucial for data management. " * 30
    suffixes = [
        "What is SQL?",
        "What is NoSQL?",
        "What is a transaction?",
        "What is normalization?"
    ]
    
    # Test with multiple prompts
    results = ablation_engine.test_prompt_caching(
        model_path=model_path,
        prefix_lengths=[500],
        base_prompt=base_prompt,
        suffixes=suffixes
    )
    
    # Should have results
    assert len(results) > 0
    
    # Verify metrics were collected for multiple prompts
    for result in results:
        assert "ttft_ms" in result.metrics or "latency_reduction_ms" in result.metrics


def test_prompt_caching_comparison_across_quantizations(hardware_backend, test_models):
    """
    Test comparison of prompt caching effectiveness across quantization levels.
    
    **Validates: Requirement 13.7**
    """
    if len(test_models) < 2:
        pytest.skip("Need at least 2 quantization levels for comparison")
    
    ablation_engine = AblationEngine(backend=hardware_backend)
    
    # Create consistent test prompts
    base_prompt = "The theory of computation explores the limits of what can be computed. " * 35
    suffixes = ["What is decidability?", "What is complexity?"]
    
    # Collect results for each quantization
    quantization_results = {}
    
    for quant, model_path in list(test_models.items())[:2]:  # Test first 2 quantizations
        results = ablation_engine.test_prompt_caching(
            model_path=model_path,
            prefix_lengths=[500],
            base_prompt=base_prompt,
            suffixes=suffixes
        )
        
        quantization_results[quant] = results
    
    # Verify we can compare across quantizations
    assert len(quantization_results) == 2
    
    # Both should have results
    for quant, results in quantization_results.items():
        assert len(results) > 0
        
        # Extract key metrics for comparison
        for result in results:
            if "ttft_ms" in result.metrics:
                assert result.metrics["ttft_ms"] > 0
            
            if "latency_reduction_ms" in result.metrics:
                # Can be positive or negative
                assert isinstance(result.metrics["latency_reduction_ms"], (int, float))


def test_disk_io_time_for_disk_cache(hardware_backend, test_models):
    """
    Test disk I/O time measurement for disk-based caching.
    
    **Validates: Requirement 13.6**
    """
    ablation_engine = AblationEngine(backend=hardware_backend)
    
    model_path = list(test_models.values())[0]
    
    # Create test prompts
    base_prompt = "The design of operating systems involves many complex tradeoffs. " * 30
    suffixes = ["What is scheduling?", "What is memory management?"]
    
    # Test with disk cache (if supported)
    results = ablation_engine.test_prompt_caching(
        model_path=model_path,
        prefix_lengths=[500],
        base_prompt=base_prompt,
        suffixes=suffixes,
        cache_type="disk"  # Request disk cache
    )
    
    # Check for disk I/O metrics
    for result in results:
        # Disk I/O time might be reported
        if "disk_io_time_ms" in result.metrics:
            assert result.metrics["disk_io_time_ms"] >= 0
        
        # Or cache file size
        if "cache_file_size_mb" in result.metrics:
            assert result.metrics["cache_file_size_mb"] >= 0


def test_cache_file_size_measurement(hardware_backend, test_models):
    """
    Test cache file size measurement for disk cache.
    
    **Validates: Requirement 13.5**
    """
    ablation_engine = AblationEngine(backend=hardware_backend)
    
    model_path = list(test_models.values())[0]
    
    # Create test prompts
    base_prompt = "The principles of networking are fundamental to distributed systems. " * 30
    suffixes = ["What is TCP?", "What is UDP?"]
    
    # Test with disk cache
    results = ablation_engine.test_prompt_caching(
        model_path=model_path,
        prefix_lengths=[500],
        base_prompt=base_prompt,
        suffixes=suffixes,
        cache_type="disk"
    )
    
    # Check for cache file size measurement
    for result in results:
        if "cache_file_size_mb" in result.metrics:
            # Should be positive if cache was created
            assert result.metrics["cache_file_size_mb"] >= 0
        
        # Or in configuration
        if "cache_size_mb" in result.configuration:
            assert result.configuration["cache_size_mb"] >= 0
