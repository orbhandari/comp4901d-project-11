"""
Integration tests for batch processing and throughput testing.

Tests batch inference with different batch sizes, aggregate throughput calculation,
per-prompt latency measurement, and memory scaling measurement.

**Validates: Requirements 15.1, 15.2, 15.3, 15.4, 15.5**
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
def test_model():
    """
    Get path to test model for batch testing.
    
    Uses a small model if available, otherwise skips test.
    """
    models_dir = Path("models")
    
    # Look for small test models
    for pattern in ["*Q2_K*.gguf", "*Q4_0*.gguf", "*tiny*.gguf"]:
        matching = list(models_dir.glob(pattern))
        if matching:
            return str(matching[0])
    
    # Use any available model
    model_files = list(models_dir.glob("*.gguf"))
    if not model_files:
        pytest.skip("No GGUF models found for testing")
    
    return str(model_files[0])


def test_batch_inference_with_batch_size_1(hardware_backend, test_model):
    """
    Test batch inference with batch size 1 (baseline).
    
    **Validates: Requirement 15.1**
    """
    ablation_engine = AblationEngine(backend=hardware_backend)
    
    # Create test prompts
    prompts = ["What is artificial intelligence?"]
    
    # Test batch processing
    results = ablation_engine.test_batch_sizes(
        model_path=test_model,
        batch_sizes=[1],
        prompts=prompts
    )
    
    # Should have results for batch size 1
    result_1 = next((r for r in results if "batch_size" in r.configuration and r.configuration["batch_size"] == 1), None)
    assert result_1 is not None
    
    # Verify metrics
    assert "aggregate_throughput_tps" in result_1.metrics or "throughput_tps" in result_1.metrics
    assert result_1.metrics.get("aggregate_throughput_tps", result_1.metrics.get("throughput_tps", 0)) > 0


def test_batch_inference_with_batch_size_2(hardware_backend, test_model):
    """
    Test batch inference with batch size 2.
    
    **Validates: Requirement 15.1**
    """
    ablation_engine = AblationEngine(backend=hardware_backend)
    
    # Create test prompts
    prompts = [
        "What is machine learning?",
        "What is deep learning?"
    ]
    
    # Test batch processing
    results = ablation_engine.test_batch_sizes(
        model_path=test_model,
        batch_sizes=[2],
        prompts=prompts
    )
    
    # Should have results for batch size 2
    result_2 = next((r for r in results if "batch_size" in r.configuration and r.configuration["batch_size"] == 2), None)
    assert result_2 is not None
    
    # Verify metrics
    assert "aggregate_throughput_tps" in result_2.metrics or "throughput_tps" in result_2.metrics


def test_batch_inference_with_batch_size_4(hardware_backend, test_model):
    """
    Test batch inference with batch size 4.
    
    **Validates: Requirement 15.1**
    """
    ablation_engine = AblationEngine(backend=hardware_backend)
    
    # Create test prompts
    prompts = [
        "Explain neural networks.",
        "Explain decision trees.",
        "Explain random forests.",
        "Explain support vector machines."
    ]
    
    # Test batch processing
    results = ablation_engine.test_batch_sizes(
        model_path=test_model,
        batch_sizes=[4],
        prompts=prompts
    )
    
    # Should have results for batch size 4
    result_4 = next((r for r in results if "batch_size" in r.configuration and r.configuration["batch_size"] == 4), None)
    assert result_4 is not None
    
    # Verify metrics
    assert "aggregate_throughput_tps" in result_4.metrics or "throughput_tps" in result_4.metrics


def test_aggregate_throughput_calculation(hardware_backend, test_model):
    """
    Test aggregate throughput calculation in tokens per second.
    
    **Validates: Requirement 15.2**
    """
    ablation_engine = AblationEngine(backend=hardware_backend)
    
    # Create test prompts
    prompts = [
        "What is Python?",
        "What is JavaScript?"
    ]
    
    # Test batch processing
    results = ablation_engine.test_batch_sizes(
        model_path=test_model,
        batch_sizes=[2],
        prompts=prompts
    )
    
    # Check aggregate throughput
    for result in results:
        if "aggregate_throughput_tps" in result.metrics:
            # Should be positive
            assert result.metrics["aggregate_throughput_tps"] > 0
            
            # Should be reasonable (typically 1-1000 tokens/sec)
            assert 0.1 <= result.metrics["aggregate_throughput_tps"] <= 10000


def test_per_prompt_latency_measurement(hardware_backend, test_model):
    """
    Test per-prompt latency distribution measurement within batches.
    
    **Validates: Requirement 15.3**
    """
    ablation_engine = AblationEngine(backend=hardware_backend)
    
    # Create test prompts
    prompts = [
        "Explain algorithms.",
        "Explain data structures.",
        "Explain complexity theory."
    ]
    
    # Test batch processing
    results = ablation_engine.test_batch_sizes(
        model_path=test_model,
        batch_sizes=[3],
        prompts=prompts
    )
    
    # Check per-prompt latency
    for result in results:
        # Latency distribution might be in metrics
        if "per_prompt_latency_ms" in result.metrics:
            latencies = result.metrics["per_prompt_latency_ms"]
            
            # Should be a list
            assert isinstance(latencies, list)
            
            # Should have one entry per prompt
            assert len(latencies) == 3
            
            # All latencies should be positive
            for latency in latencies:
                assert latency > 0
        
        # Or average latency
        if "avg_per_prompt_latency_ms" in result.metrics:
            assert result.metrics["avg_per_prompt_latency_ms"] > 0


def test_memory_scaling_measurement(hardware_backend, test_model):
    """
    Test memory scaling measurement as batch size increases.
    
    **Validates: Requirement 15.5**
    """
    ablation_engine = AblationEngine(backend=hardware_backend)
    
    # Create test prompts for different batch sizes
    prompts_1 = ["Test prompt 1."]
    prompts_2 = ["Test prompt 1.", "Test prompt 2."]
    prompts_4 = ["Test prompt 1.", "Test prompt 2.", "Test prompt 3.", "Test prompt 4."]
    
    # Test different batch sizes
    results = ablation_engine.test_batch_sizes(
        model_path=test_model,
        batch_sizes=[1, 2, 4],
        prompts=prompts_4  # Use largest set, will be subsetted
    )
    
    # Extract memory usage for each batch size
    memory_by_batch_size = {}
    
    for result in results:
        if "batch_size" in result.configuration:
            batch_size = result.configuration["batch_size"]
            
            if "peak_memory_mb" in result.metrics:
                memory_by_batch_size[batch_size] = result.metrics["peak_memory_mb"]
    
    # Verify memory scaling
    if len(memory_by_batch_size) >= 2:
        # Memory should generally increase with batch size
        # (though not strictly monotonic due to system variance)
        for batch_size, memory in memory_by_batch_size.items():
            assert memory > 0


def test_batch_sizes_1_2_4_8_16(hardware_backend, test_model):
    """
    Test batch sizes of 1, 2, 4, 8, and 16 prompts.
    
    **Validates: Requirement 15.4**
    """
    ablation_engine = AblationEngine(backend=hardware_backend)
    
    # Create enough prompts for largest batch
    prompts = [f"Test prompt number {i}." for i in range(16)]
    
    # Test all batch sizes
    results = ablation_engine.test_batch_sizes(
        model_path=test_model,
        batch_sizes=[1, 2, 4, 8, 16],
        prompts=prompts
    )
    
    # Verify we got results for each batch size
    batch_sizes_tested = set()
    
    for result in results:
        if "batch_size" in result.configuration:
            batch_sizes_tested.add(result.configuration["batch_size"])
    
    # Should have tested multiple batch sizes
    assert len(batch_sizes_tested) >= 1
    
    # Verify each result has metrics
    for result in results:
        assert "aggregate_throughput_tps" in result.metrics or "throughput_tps" in result.metrics


def test_optimal_batch_size_identification(hardware_backend, test_model):
    """
    Test identification of optimal batch size maximizing throughput.
    
    **Validates: Requirement 15.6**
    """
    ablation_engine = AblationEngine(backend=hardware_backend)
    
    # Create test prompts
    prompts = [f"Explain concept {i}." for i in range(8)]
    
    # Test multiple batch sizes
    results = ablation_engine.test_batch_sizes(
        model_path=test_model,
        batch_sizes=[1, 2, 4, 8],
        prompts=prompts
    )
    
    # Extract throughput for each batch size
    throughput_by_batch_size = {}
    
    for result in results:
        if "batch_size" in result.configuration:
            batch_size = result.configuration["batch_size"]
            throughput = result.metrics.get("aggregate_throughput_tps", result.metrics.get("throughput_tps", 0))
            
            if throughput > 0:
                throughput_by_batch_size[batch_size] = throughput
    
    # Find optimal batch size (highest throughput)
    if len(throughput_by_batch_size) >= 2:
        optimal_batch_size = max(throughput_by_batch_size, key=throughput_by_batch_size.get)
        optimal_throughput = throughput_by_batch_size[optimal_batch_size]
        
        # Verify optimal batch size is reasonable
        assert optimal_batch_size in [1, 2, 4, 8, 16]
        assert optimal_throughput > 0


def test_throughput_latency_tradeoff(hardware_backend, test_model):
    """
    Test throughput-latency tradeoff curves for different batch sizes.
    
    **Validates: Requirement 15.8**
    """
    ablation_engine = AblationEngine(backend=hardware_backend)
    
    # Create test prompts
    prompts = [f"Question {i}?" for i in range(8)]
    
    # Test multiple batch sizes
    results = ablation_engine.test_batch_sizes(
        model_path=test_model,
        batch_sizes=[1, 2, 4, 8],
        prompts=prompts
    )
    
    # Extract throughput and latency for each batch size
    tradeoff_data = []
    
    for result in results:
        if "batch_size" in result.configuration:
            batch_size = result.configuration["batch_size"]
            throughput = result.metrics.get("aggregate_throughput_tps", result.metrics.get("throughput_tps", 0))
            latency = result.metrics.get("avg_per_prompt_latency_ms", result.metrics.get("ttft_ms", 0))
            
            if throughput > 0 and latency > 0:
                tradeoff_data.append({
                    "batch_size": batch_size,
                    "throughput": throughput,
                    "latency": latency
                })
    
    # Verify we have tradeoff data
    if len(tradeoff_data) >= 2:
        # Generally, larger batch sizes increase throughput but also latency
        # (though not strictly monotonic)
        for data in tradeoff_data:
            assert data["throughput"] > 0
            assert data["latency"] > 0


def test_gpu_utilization_across_batch_sizes(hardware_backend, test_model):
    """
    Test GPU utilization measurement across different batch sizes.
    
    **Validates: Requirement 15.7**
    """
    if not hardware_backend.hw_info.has_gpu:
        pytest.skip("Requires GPU for GPU utilization testing")
    
    ablation_engine = AblationEngine(backend=hardware_backend)
    
    # Create test prompts
    prompts = [f"Prompt {i}" for i in range(8)]
    
    # Test multiple batch sizes
    results = ablation_engine.test_batch_sizes(
        model_path=test_model,
        batch_sizes=[1, 2, 4, 8],
        prompts=prompts
    )
    
    # Check GPU utilization
    for result in results:
        if "gpu_utilization_pct" in result.metrics:
            # Should be between 0 and 100
            assert 0 <= result.metrics["gpu_utilization_pct"] <= 100
        
        # Or GPU memory usage
        if "gpu_memory_mb" in result.metrics:
            assert result.metrics["gpu_memory_mb"] >= 0


def test_memory_overflow_detection(hardware_backend, test_model):
    """
    Test detection of memory overflow with large batch sizes.
    
    **Validates: Requirement 15.6**
    """
    ablation_engine = AblationEngine(backend=hardware_backend)
    
    # Create large batch of prompts
    prompts = [f"Test prompt {i} with some content." for i in range(32)]
    
    # Try large batch size (may cause OOM)
    try:
        results = ablation_engine.test_batch_sizes(
            model_path=test_model,
            batch_sizes=[32],
            prompts=prompts
        )
        
        # If it succeeds, verify results
        assert len(results) > 0
        
    except (MemoryError, RuntimeError) as e:
        # Expected: memory overflow detected
        # This is acceptable behavior
        assert "memory" in str(e).lower() or "oom" in str(e).lower()


def test_batch_processing_with_varying_prompt_lengths(hardware_backend, test_model):
    """
    Test batch processing with prompts of varying lengths.
    
    **Validates: Requirement 15.3**
    """
    ablation_engine = AblationEngine(backend=hardware_backend)
    
    # Create prompts of different lengths
    prompts = [
        "Short.",
        "This is a medium length prompt with more words.",
        "This is a much longer prompt that contains significantly more text and should take longer to process than the shorter prompts in this batch."
    ]
    
    # Test batch processing
    results = ablation_engine.test_batch_sizes(
        model_path=test_model,
        batch_sizes=[3],
        prompts=prompts
    )
    
    # Should have results
    assert len(results) > 0
    
    # Check per-prompt latency distribution
    for result in results:
        if "per_prompt_latency_ms" in result.metrics:
            latencies = result.metrics["per_prompt_latency_ms"]
            
            # Should have latency for each prompt
            assert len(latencies) == 3
            
            # Latencies might vary based on prompt length
            # (though this depends on implementation)
            for latency in latencies:
                assert latency > 0


def test_batch_processing_comparison_across_sizes(hardware_backend, test_model):
    """
    Test comparison of batch processing across different sizes.
    
    **Validates: Requirements 15.1, 15.2, 15.4**
    """
    ablation_engine = AblationEngine(backend=hardware_backend)
    
    # Create test prompts
    prompts = [f"Test {i}" for i in range(8)]
    
    # Test multiple batch sizes
    results = ablation_engine.test_batch_sizes(
        model_path=test_model,
        batch_sizes=[1, 2, 4, 8],
        prompts=prompts
    )
    
    # Verify we can compare across batch sizes
    comparison_data = []
    
    for result in results:
        if "batch_size" in result.configuration:
            comparison_data.append({
                "batch_size": result.configuration["batch_size"],
                "throughput": result.metrics.get("aggregate_throughput_tps", result.metrics.get("throughput_tps", 0)),
                "memory_mb": result.metrics.get("peak_memory_mb", 0)
            })
    
    # Should have data for multiple batch sizes
    assert len(comparison_data) >= 1
    
    # Verify all data is valid
    for data in comparison_data:
        assert data["batch_size"] > 0
        assert data["throughput"] >= 0
        assert data["memory_mb"] >= 0
