"""
Integration tests for GPU acceleration functionality.

These tests are designed to run on Jetson Xavier NX hardware with GPU support.
Tests GPU layer offloading, memory exhaustion handling, CPU fallback, and GPU metrics collection.

**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.7**

NOTE: These tests require actual Jetson hardware and should be run manually.
They are marked with @pytest.mark.jetson and will be skipped in CI/CD.
"""

import os
import pytest
from pathlib import Path

from llm_benchmark.hardware.detector import HardwareDetector
from llm_benchmark.hardware.hal import create_backend, JetsonBackend
from llm_benchmark.metrics.collector import MetricsCollector
from llm_benchmark.profiler.quantization import QuantizationProfiler


@pytest.fixture
def jetson_hardware():
    """
    Detect Jetson hardware and skip if not available.
    
    **Validates: Requirement 4.1**
    """
    hw_info = HardwareDetector.detect()
    
    if hw_info.os_type != "jetson_xavier_nx":
        pytest.skip("Requires Jetson Xavier NX hardware")
    
    if not hw_info.has_gpu:
        pytest.skip("Requires GPU to be available")
    
    return hw_info


@pytest.fixture
def jetson_backend(jetson_hardware):
    """Create Jetson backend for testing."""
    return create_backend(jetson_hardware)


@pytest.fixture
def test_model():
    """
    Get path to test model for GPU testing.
    
    Uses a small model if available, otherwise skips test.
    """
    models_dir = Path("models")
    
    # Look for any available GGUF model
    model_files = list(models_dir.glob("*.gguf"))
    
    if not model_files:
        pytest.skip("No GGUF models found for testing")
    
    # Prefer smaller models for faster testing
    for pattern in ["*Q2_K*.gguf", "*Q4_0*.gguf", "*tiny*.gguf"]:
        matching = list(models_dir.glob(pattern))
        if matching:
            return str(matching[0])
    
    # Use any available model
    return str(model_files[0])


@pytest.mark.jetson
@pytest.mark.manual
def test_gpu_layer_offloading(jetson_backend, test_model):
    """
    Test GPU layer offloading on Jetson Xavier NX.
    
    **Validates: Requirements 4.1, 4.2**
    """
    # Get llama config with GPU layers
    llama_config = jetson_backend.get_llama_config()
    
    # Verify GPU layers are configured
    assert "n_gpu_layers" in llama_config
    assert llama_config["n_gpu_layers"] > 0
    
    # GPU layers should be reasonable for Jetson Xavier NX
    # (typically 20-60 layers depending on model size and available memory)
    assert 10 <= llama_config["n_gpu_layers"] <= 100
    
    # Verify other GPU-related settings
    assert "use_mlock" in llama_config
    assert "n_threads" in llama_config
    
    # Try to load model with GPU layers
    try:
        from llama_cpp import Llama
        
        llm = Llama(
            model_path=test_model,
            **llama_config
        )
        
        # Verify model loaded successfully
        assert llm is not None
        
        # Run a simple inference to verify GPU is being used
        response = llm("Test prompt", max_tokens=5, stream=False)
        assert response is not None
        
    except Exception as e:
        pytest.fail(f"Failed to load model with GPU layers: {e}")


@pytest.mark.jetson
@pytest.mark.manual
def test_gpu_memory_exhaustion_and_fallback(jetson_backend, test_model):
    """
    Test GPU memory exhaustion and fallback behavior.
    
    **Validates: Requirements 4.2, 4.5**
    """
    from llama_cpp import Llama
    
    # Get initial GPU layer count
    initial_config = jetson_backend.get_llama_config()
    initial_gpu_layers = initial_config["n_gpu_layers"]
    
    # Try to load with excessive GPU layers (should trigger fallback)
    excessive_layers = 200  # Intentionally too many
    
    try:
        llm = Llama(
            model_path=test_model,
            n_gpu_layers=excessive_layers,
            n_ctx=2048
        )
        
        # If it succeeds, that's fine - model might be small enough
        assert llm is not None
        
    except RuntimeError as e:
        # Expected: GPU memory exhaustion
        assert "out of memory" in str(e).lower() or "cuda" in str(e).lower()
        
        # Now try with reduced layers (fallback behavior)
        reduced_layers = initial_gpu_layers // 2
        
        llm = Llama(
            model_path=test_model,
            n_gpu_layers=reduced_layers,
            n_ctx=2048
        )
        
        assert llm is not None
        
        # Verify it works with reduced layers
        response = llm("Test", max_tokens=5, stream=False)
        assert response is not None


@pytest.mark.jetson
@pytest.mark.manual
def test_cpu_only_fallback_when_gpu_unavailable(jetson_backend, test_model):
    """
    Test CPU-only fallback when GPU is unavailable or disabled.
    
    **Validates: Requirement 4.5**
    """
    from llama_cpp import Llama
    
    # Force CPU-only mode
    cpu_config = {
        "n_gpu_layers": 0,
        "n_ctx": 2048,
        "n_threads": 4
    }
    
    try:
        llm = Llama(
            model_path=test_model,
            **cpu_config
        )
        
        assert llm is not None
        
        # Run inference in CPU-only mode
        response = llm("Test prompt for CPU", max_tokens=10, stream=False)
        assert response is not None
        
        # Verify response is valid
        assert "choices" in response or isinstance(response, dict)
        
    except Exception as e:
        pytest.fail(f"CPU-only fallback failed: {e}")


@pytest.mark.jetson
@pytest.mark.manual
def test_gpu_metrics_collection(jetson_backend, test_model):
    """
    Test that GPU metrics are collected when GPU is used.
    
    **Validates: Requirements 4.3, 4.4, 4.7**
    """
    from llama_cpp import Llama
    
    # Create metrics collector
    metrics_collector = MetricsCollector(jetson_backend.hw_info)
    
    # Load model with GPU
    llama_config = jetson_backend.get_llama_config()
    
    llm = Llama(
        model_path=test_model,
        **llama_config
    )
    
    # Collect metrics during inference
    test_prompt = "Explain quantum computing in simple terms."
    max_tokens = 20
    
    metrics = metrics_collector.collect_inference_metrics(
        llm=llm,
        prompt=test_prompt,
        max_tokens=max_tokens
    )
    
    # Verify GPU metrics were collected
    assert metrics is not None
    
    # Check GPU-specific metrics
    if metrics.used_gpu_acceleration:
        # GPU memory should be reported
        assert metrics.gpu_memory_mb is not None
        assert metrics.gpu_memory_mb > 0
        
        # GPU utilization should be reported
        assert metrics.gpu_utilization_pct is not None
        assert 0 <= metrics.gpu_utilization_pct <= 100
        
        # GPU temperature might be available
        if metrics.gpu_temp_c is not None:
            assert 0 < metrics.gpu_temp_c < 150  # Reasonable temperature range
    
    # Verify standard metrics are also collected
    assert metrics.ttft_ms > 0
    assert metrics.prefill_tps > 0
    assert metrics.decode_tps > 0
    assert metrics.prompt_tokens > 0
    assert metrics.output_tokens > 0


@pytest.mark.jetson
@pytest.mark.manual
def test_gpu_acceleration_with_quantization_profiler(jetson_backend, test_model):
    """
    Test GPU acceleration integration with quantization profiler.
    
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
    """
    # Create profiler with Jetson backend
    profiler = QuantizationProfiler(
        backend=jetson_backend,
        metrics_collector=MetricsCollector(jetson_backend.hw_info)
    )
    
    # Profile with GPU acceleration
    test_prompt = "What is machine learning?"
    max_tokens = 15
    
    # Extract quantization from filename
    model_name = Path(test_model).name.lower()
    if "q2_k" in model_name:
        quant = "Q2_K"
    elif "q4_0" in model_name:
        quant = "Q4_0"
    elif "q4_k_m" in model_name:
        quant = "Q4_K_M"
    elif "q8_0" in model_name:
        quant = "Q8_0"
    else:
        quant = "UNKNOWN"
    
    result = profiler.profile_quantization(
        model_path=test_model,
        quant=quant,
        prompt=test_prompt,
        max_tokens=max_tokens
    )
    
    # Verify result
    assert result is not None
    assert result.quantization == quant
    
    # Verify GPU was used
    assert result.used_gpu_acceleration is True
    
    # Verify GPU metrics
    assert result.gpu_memory_mb is not None
    assert result.gpu_memory_mb > 0
    
    assert result.gpu_utilization_pct is not None
    assert result.gpu_utilization_pct >= 0
    
    # Verify standard metrics
    assert result.load_time_s > 0
    assert result.peak_ram_mb > 0
    assert result.ttft_ms > 0
    assert result.prefill_tps > 0
    assert result.decode_tps > 0


@pytest.mark.jetson
@pytest.mark.manual
def test_gpu_layer_calculation_heuristic(jetson_backend):
    """
    Test GPU layer calculation heuristic for Jetson.
    
    **Validates: Requirement 4.2**
    """
    # Verify backend is JetsonBackend
    assert isinstance(jetson_backend, JetsonBackend)
    
    # Get GPU memory info
    hw_info = jetson_backend.hw_info
    assert hw_info.has_gpu
    assert hw_info.gpu_memory_gb is not None
    assert hw_info.gpu_memory_gb > 0
    
    # Calculate GPU layers
    gpu_layers = jetson_backend._calculate_gpu_layers()
    
    # Verify calculation is reasonable
    # Heuristic: ~100MB per layer, 80% GPU memory utilization
    expected_layers = int((hw_info.gpu_memory_gb * 1024 * 0.8) / 100)
    
    # Allow some variance in the calculation
    assert abs(gpu_layers - expected_layers) <= 10
    
    # Verify layers are positive
    assert gpu_layers > 0
    
    # Verify layers are reasonable for Jetson Xavier NX (typically 20-60)
    assert 10 <= gpu_layers <= 100


@pytest.mark.jetson
@pytest.mark.manual
def test_gpu_acceleration_flag_in_results(jetson_backend, test_model):
    """
    Test that GPU acceleration flag is properly set in results.
    
    **Validates: Requirement 4.7**
    """
    from llama_cpp import Llama
    
    metrics_collector = MetricsCollector(jetson_backend.hw_info)
    
    # Test with GPU enabled
    gpu_config = jetson_backend.get_llama_config()
    llm_gpu = Llama(model_path=test_model, **gpu_config)
    
    metrics_gpu = metrics_collector.collect_inference_metrics(
        llm=llm_gpu,
        prompt="Test with GPU",
        max_tokens=10
    )
    
    # Should report GPU was used
    assert metrics_gpu.used_gpu_acceleration is True
    
    # Test with CPU only
    cpu_config = {"n_gpu_layers": 0, "n_ctx": 2048}
    llm_cpu = Llama(model_path=test_model, **cpu_config)
    
    metrics_cpu = metrics_collector.collect_inference_metrics(
        llm=llm_cpu,
        prompt="Test with CPU",
        max_tokens=10
    )
    
    # Should report GPU was not used
    assert metrics_cpu.used_gpu_acceleration is False
    
    # GPU metrics should be None for CPU-only
    assert metrics_cpu.gpu_memory_mb is None or metrics_cpu.gpu_memory_mb == 0
    assert metrics_cpu.gpu_utilization_pct is None or metrics_cpu.gpu_utilization_pct == 0
