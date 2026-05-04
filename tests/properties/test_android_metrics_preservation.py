"""
Preservation Property Tests - Non-Android Platform Behavior

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

This test suite verifies that non-Android platforms (X86Backend, JetsonBackend) using
llama-cpp-python continue to work correctly after the Android metrics fix is applied.

**CRITICAL**: These tests MUST PASS on unfixed code - passing confirms baseline behavior.
**IMPORTANT**: Follow observation-first methodology - observe behavior on UNFIXED code first.

**EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)

The tests capture the expected behavior for non-Android platforms:
1. Load time measurement around `backend.load_model_safe()` works correctly
2. RAM measurement using `self.process.memory_info().rss` works correctly
3. Warmup inference executes successfully
4. Memory tracking during token generation captures peak memory
5. TTFT, prefill TPS, and other metrics calculate correctly
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings, assume, HealthCheck
import statistics

from llm_benchmark.profiler.quantization import QuantizationProfiler
from llm_benchmark.hardware.hal import X86Backend, JetsonBackend
from llm_benchmark.models import HardwareInfo, QuantizationResult
from llm_benchmark.metrics.collector import MetricsCollector


@pytest.fixture
def x86_hardware_info():
    """Create hardware info for X86 platform."""
    return HardwareInfo(
        os_type="linux",
        cpu_model="Intel Core i7-9700K",
        cpu_cores=8,
        cpu_features=["avx2", "fma"],
        total_ram_gb=32.0,
        available_ram_gb=24.0,
        has_gpu=False,
        gpu_model=None,
        gpu_memory_gb=None,
        gpu_compute_capability=None,
        has_thermal_sensors=True,
        has_power_sensors=False,
    )


@pytest.fixture
def jetson_hardware_info():
    """Create hardware info for Jetson platform."""
    return HardwareInfo(
        os_type="jetson_xavier_nx",
        cpu_model="ARM Cortex-A78AE",
        cpu_cores=8,
        cpu_features=["neon", "fp16"],
        total_ram_gb=8.0,
        available_ram_gb=6.0,
        has_gpu=True,
        gpu_model="NVIDIA Xavier GPU",
        gpu_memory_gb=8.0,
        gpu_compute_capability="7.2",
        has_thermal_sensors=True,
        has_power_sensors=True,
    )


@pytest.fixture
def x86_backend(x86_hardware_info):
    """Create X86 backend for testing."""
    return X86Backend(x86_hardware_info)


@pytest.fixture
def jetson_backend(jetson_hardware_info):
    """Create Jetson backend for testing."""
    return JetsonBackend(jetson_hardware_info)


@pytest.fixture
def mock_llama_cpp():
    """Create a mock llama-cpp-python Llama instance (in-process inference)."""
    mock_llm = MagicMock()
    
    # Simulate tokenize method
    def mock_tokenize(text):
        if isinstance(text, bytes):
            char_count = len(text)
        else:
            char_count = len(text.encode('utf-8'))
        token_count = max(1, char_count // 4)
        return list(range(token_count))
    
    mock_llm.tokenize = mock_tokenize
    
    # Simulate streaming inference - return a generator
    def mock_call(prompt, max_tokens=100, stream=True, **kwargs):
        """Simulate inference with character-by-character streaming."""
        output = "This is a test response from the model. " * (max_tokens // 10)
        output = output[:max_tokens]
        
        # Yield characters
        for char in output:
            yield {
                'choices': [{
                    'text': char,
                    'finish_reason': None
                }]
            }
        
        # Final chunk
        yield {
            'choices': [{
                'text': '',
                'finish_reason': 'stop'
            }]
        }
    
    # Configure the mock to return the generator when called
    mock_llm.side_effect = mock_call
    
    return mock_llm


class TestPreservationX86Backend:
    """
    Property 2.1: Preservation - X86Backend Load Time Measurement
    
    **Validates: Requirements 3.1**
    
    Tests that load time measurement around `backend.load_model_safe()` continues
    to work correctly for X86Backend using llama-cpp-python (in-process inference).
    """
    
    def test_x86_load_time_measurement_works(self, x86_backend, mock_llama_cpp):
        """
        Test that load time measurement works for X86Backend.
        
        **Preservation Requirement**: Load time measured around backend.load_model_safe()
        must continue to work for llama-cpp-python platforms where the model is loaded
        during that call.
        
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (behavior preserved).
        """
        # Create profiler
        metrics_collector = MetricsCollector(x86_backend.hw_info)
        profiler = QuantizationProfiler(x86_backend, metrics_collector)
        
        # Mock backend.load_model_safe to return our mock Llama instance
        # Simulate some load time (llama-cpp-python loads model in constructor)
        def mock_load_with_delay(*args, **kwargs):
            import time
            time.sleep(0.1)  # Simulate 100ms load time
            return mock_llama_cpp
        
        with patch.object(x86_backend, 'load_model_safe', side_effect=mock_load_with_delay):
            # Profile quantization
            result = profiler.profile_quantization(
                model_path="/tmp/test-model.gguf",
                quant="Q4_0",
                prompt="What is the capital of France?",
                max_tokens=50,
                warmup_tokens=5,
                context_size=2048
            )
        
        # Preservation: Load time should be measured and > 0
        # For llama-cpp-python, model loads during load_model_safe() call
        assert result.load_time_s > 0, (
            f"Preservation violation: Load time is {result.load_time_s}s (expected > 0). "
            f"Load time measurement around backend.load_model_safe() must continue to work "
            f"for X86Backend with llama-cpp-python."
        )
        
        # Load time should be reasonable (not impossibly high)
        assert result.load_time_s < 60, (
            f"Preservation violation: Load time is {result.load_time_s}s (expected < 60s). "
            f"Load time measurement seems incorrect."
        )
    
    def test_x86_ram_measurement_works(self, x86_backend, mock_llama_cpp):
        """
        Test that RAM measurement works for X86Backend.
        
        **Preservation Requirement**: RAM measurement using self.process.memory_info().rss
        must continue to work for llama-cpp-python platforms where the model runs in-process.
        
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (behavior preserved).
        """
        # Create profiler
        metrics_collector = MetricsCollector(x86_backend.hw_info)
        profiler = QuantizationProfiler(x86_backend, metrics_collector)
        
        # Mock backend.load_model_safe to return our mock Llama instance
        with patch.object(x86_backend, 'load_model_safe', return_value=mock_llama_cpp):
            # Profile quantization
            result = profiler.profile_quantization(
                model_path="/tmp/test-model.gguf",
                quant="Q4_0",
                prompt="What is the capital of France?",
                max_tokens=50,
                warmup_tokens=5,
                context_size=2048
            )
        
        # Preservation: Peak RAM should be measured and > 0
        # For llama-cpp-python, model runs in-process so self.process.memory_info().rss works
        assert result.peak_ram_mb > 0, (
            f"Preservation violation: Peak RAM is {result.peak_ram_mb} MB (expected > 0). "
            f"RAM measurement using self.process.memory_info().rss must continue to work "
            f"for X86Backend with llama-cpp-python."
        )
        
        # Peak RAM should be reasonable (not impossibly high)
        assert result.peak_ram_mb < 100000, (
            f"Preservation violation: Peak RAM is {result.peak_ram_mb} MB (expected < 100000 MB). "
            f"RAM measurement seems incorrect."
        )
    
    def test_x86_warmup_inference_executes(self, x86_backend):
        """
        Test that warmup inference executes for X86Backend.
        
        **Preservation Requirement**: Warmup inference before measurement must continue
        to execute on all platforms.
        
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (behavior preserved).
        """
        # Create profiler
        metrics_collector = MetricsCollector(x86_backend.hw_info)
        profiler = QuantizationProfiler(x86_backend, metrics_collector)
        
        # Track all calls (both stream=True and stream=False)
        all_calls = []
        
        def mock_call_tracker(prompt, max_tokens=100, stream=True, **kwargs):
            """Track calls to llm() for warmup detection."""
            all_calls.append({'max_tokens': max_tokens, 'stream': stream})
            # Simulate inference
            output = "Test" * max_tokens
            if stream:
                # Return generator for streaming
                def gen():
                    for char in output:
                        yield {'choices': [{'text': char, 'finish_reason': None}]}
                    yield {'choices': [{'text': '', 'finish_reason': 'stop'}]}
                return gen()
            else:
                # Return dict for non-streaming
                return {'choices': [{'text': output, 'finish_reason': 'stop'}]}
        
        # Create a fresh mock with the tracker
        mock_llm = MagicMock()
        mock_llm.side_effect = mock_call_tracker
        mock_llm.tokenize = lambda text: list(range(max(1, len(text if isinstance(text, bytes) else text.encode('utf-8')) // 4)))
        
        # Mock backend.load_model_safe to return our mock Llama instance
        with patch.object(x86_backend, 'load_model_safe', return_value=mock_llm):
            # Profile quantization with warmup_tokens=5
            result = profiler.profile_quantization(
                model_path="/tmp/test-model.gguf",
                quant="Q4_0",
                prompt="What is the capital of France?",
                max_tokens=50,
                warmup_tokens=5,
                context_size=2048
            )
        
        # Preservation: Warmup inference should have been called
        # We expect at least 2 calls: warmup (5 tokens, stream=False) + measurement (50 tokens, stream=True)
        assert len(all_calls) >= 2, (
            f"Preservation violation: Expected at least 2 inference calls (warmup + measurement), "
            f"got {len(all_calls)}. Warmup inference must continue to execute. Calls: {all_calls}"
        )
        
        # First call should be warmup with 5 tokens and stream=False
        assert all_calls[0]['max_tokens'] == 5, (
            f"Preservation violation: First inference call should be warmup with 5 tokens, "
            f"got {all_calls[0]['max_tokens']} tokens."
        )
        
        assert all_calls[0]['stream'] == False, (
            f"Preservation violation: Warmup call should use stream=False, "
            f"got stream={all_calls[0]['stream']}."
        )
    
    def test_x86_metrics_calculate_correctly(self, x86_backend, mock_llama_cpp):
        """
        Test that TTFT, prefill TPS, and other metrics calculate correctly for X86Backend.
        
        **Preservation Requirement**: TTFT, prefill TPS, and other metrics calculation
        must continue using existing methodology on all platforms.
        
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (behavior preserved).
        """
        # Create profiler
        metrics_collector = MetricsCollector(x86_backend.hw_info)
        profiler = QuantizationProfiler(x86_backend, metrics_collector)
        
        # Mock backend.load_model_safe to return our mock Llama instance
        with patch.object(x86_backend, 'load_model_safe', return_value=mock_llama_cpp):
            # Profile quantization
            result = profiler.profile_quantization(
                model_path="/tmp/test-model.gguf",
                quant="Q4_0",
                prompt="What is the capital of France?",
                max_tokens=50,
                warmup_tokens=5,
                context_size=2048
            )
        
        # Preservation: All metrics should be calculated
        assert result.ttft_ms >= 0, (
            f"Preservation violation: TTFT is {result.ttft_ms} ms (expected >= 0). "
            f"TTFT calculation must continue to work."
        )
        
        assert result.prefill_tps >= 0, (
            f"Preservation violation: Prefill TPS is {result.prefill_tps} t/s (expected >= 0). "
            f"Prefill TPS calculation must continue to work."
        )
        
        assert result.decode_tps >= 0, (
            f"Preservation violation: Decode TPS is {result.decode_tps} t/s (expected >= 0). "
            f"Decode TPS calculation must continue to work."
        )
        
        assert result.prompt_tokens > 0, (
            f"Preservation violation: Prompt tokens is {result.prompt_tokens} (expected > 0). "
            f"Prompt tokenization must continue to work."
        )
        
        assert result.output_tokens > 0, (
            f"Preservation violation: Output tokens is {result.output_tokens} (expected > 0). "
            f"Output token counting must continue to work."
        )


class TestPreservationJetsonBackend:
    """
    Property 2.2: Preservation - JetsonBackend Load Time and RAM Measurement
    
    **Validates: Requirements 3.1, 3.2**
    
    Tests that load time and RAM measurement continue to work correctly for JetsonBackend
    using llama-cpp-python (in-process inference with GPU acceleration).
    """
    
    def test_jetson_load_time_measurement_works(self, jetson_backend, mock_llama_cpp):
        """
        Test that load time measurement works for JetsonBackend.
        
        **Preservation Requirement**: Load time measured around backend.load_model_safe()
        must continue to work for llama-cpp-python platforms.
        
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (behavior preserved).
        """
        # Create profiler
        metrics_collector = MetricsCollector(jetson_backend.hw_info)
        profiler = QuantizationProfiler(jetson_backend, metrics_collector)
        
        # Mock backend.load_model_safe to return our mock Llama instance
        def mock_load_with_delay(*args, **kwargs):
            import time
            time.sleep(0.15)  # Simulate 150ms load time (slightly longer for GPU)
            return mock_llama_cpp
        
        with patch.object(jetson_backend, 'load_model_safe', side_effect=mock_load_with_delay):
            # Profile quantization
            result = profiler.profile_quantization(
                model_path="/tmp/test-model.gguf",
                quant="Q4_0",
                prompt="What is the capital of France?",
                max_tokens=50,
                warmup_tokens=5,
                context_size=2048
            )
        
        # Preservation: Load time should be measured and > 0
        assert result.load_time_s > 0, (
            f"Preservation violation: Load time is {result.load_time_s}s (expected > 0). "
            f"Load time measurement must continue to work for JetsonBackend."
        )
        
        assert result.load_time_s < 60, (
            f"Preservation violation: Load time is {result.load_time_s}s (expected < 60s). "
            f"Load time measurement seems incorrect."
        )
    
    def test_jetson_ram_measurement_works(self, jetson_backend, mock_llama_cpp):
        """
        Test that RAM measurement works for JetsonBackend.
        
        **Preservation Requirement**: RAM measurement using self.process.memory_info().rss
        must continue to work for llama-cpp-python platforms.
        
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (behavior preserved).
        """
        # Create profiler
        metrics_collector = MetricsCollector(jetson_backend.hw_info)
        profiler = QuantizationProfiler(jetson_backend, metrics_collector)
        
        # Mock backend.load_model_safe to return our mock Llama instance
        with patch.object(jetson_backend, 'load_model_safe', return_value=mock_llama_cpp):
            # Profile quantization
            result = profiler.profile_quantization(
                model_path="/tmp/test-model.gguf",
                quant="Q4_0",
                prompt="What is the capital of France?",
                max_tokens=50,
                warmup_tokens=5,
                context_size=2048
            )
        
        # Preservation: Peak RAM should be measured and > 0
        assert result.peak_ram_mb > 0, (
            f"Preservation violation: Peak RAM is {result.peak_ram_mb} MB (expected > 0). "
            f"RAM measurement must continue to work for JetsonBackend."
        )
        
        assert result.peak_ram_mb < 100000, (
            f"Preservation violation: Peak RAM is {result.peak_ram_mb} MB (expected < 100000 MB). "
            f"RAM measurement seems incorrect."
        )


class TestPreservationPropertyBased:
    """
    Property-Based Preservation Tests
    
    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
    
    Uses property-based testing to generate many test cases and verify that
    non-Android platforms continue to work correctly across various configurations.
    """
    
    @given(
        st.integers(min_value=20, max_value=100),  # max_tokens
        st.integers(min_value=2, max_value=10),    # warmup_tokens
        st.integers(min_value=512, max_value=4096),  # context_size
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture]
    )
    def test_property_x86_load_time_always_positive(
        self, x86_backend, mock_llama_cpp, max_tokens, warmup_tokens, context_size
    ):
        """
        Property-based test: Load time is always positive for X86Backend.
        
        **Preservation Property**: For any configuration (max_tokens, warmup_tokens, context_size),
        load time measurement around backend.load_model_safe() produces positive values.
        
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (behavior preserved).
        """
        # Create profiler
        metrics_collector = MetricsCollector(x86_backend.hw_info)
        profiler = QuantizationProfiler(x86_backend, metrics_collector)
        
        # Mock backend.load_model_safe with simulated load time
        def mock_load_with_delay(*args, **kwargs):
            import time
            time.sleep(0.05)  # Simulate 50ms load time
            return mock_llama_cpp
        
        with patch.object(x86_backend, 'load_model_safe', side_effect=mock_load_with_delay):
            # Profile quantization
            result = profiler.profile_quantization(
                model_path="/tmp/test-model.gguf",
                quant="Q4_0",
                prompt="Test prompt",
                max_tokens=max_tokens,
                warmup_tokens=warmup_tokens,
                context_size=context_size
            )
        
        # Property: Load time should always be positive
        assert result.load_time_s > 0, (
            f"Property violation: Load time is {result.load_time_s}s for "
            f"max_tokens={max_tokens}, warmup_tokens={warmup_tokens}, context_size={context_size}. "
            f"Expected > 0 for X86Backend."
        )
    
    @given(
        st.integers(min_value=20, max_value=100),  # max_tokens
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_x86_ram_always_positive(
        self, x86_backend, mock_llama_cpp, max_tokens
    ):
        """
        Property-based test: Peak RAM is always positive for X86Backend.
        
        **Preservation Property**: For any max_tokens configuration, RAM measurement
        using self.process.memory_info().rss produces positive values.
        
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (behavior preserved).
        """
        # Create profiler
        metrics_collector = MetricsCollector(x86_backend.hw_info)
        profiler = QuantizationProfiler(x86_backend, metrics_collector)
        
        # Mock backend.load_model_safe
        with patch.object(x86_backend, 'load_model_safe', return_value=mock_llama_cpp):
            # Profile quantization
            result = profiler.profile_quantization(
                model_path="/tmp/test-model.gguf",
                quant="Q4_0",
                prompt="Test prompt",
                max_tokens=max_tokens,
                warmup_tokens=5,
                context_size=2048
            )
        
        # Property: Peak RAM should always be positive
        assert result.peak_ram_mb > 0, (
            f"Property violation: Peak RAM is {result.peak_ram_mb} MB for "
            f"max_tokens={max_tokens}. Expected > 0 for X86Backend."
        )
    
    @given(
        st.sampled_from(["Q2_K", "Q4_0", "Q8_0"]),  # quantization levels
    )
    @settings(
        max_examples=5,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_x86_metrics_always_valid(
        self, x86_backend, mock_llama_cpp, quant
    ):
        """
        Property-based test: All metrics are valid for X86Backend.
        
        **Preservation Property**: For any quantization level, all metrics
        (TTFT, prefill TPS, decode TPS, token counts) are calculated correctly.
        
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (behavior preserved).
        """
        # Create profiler
        metrics_collector = MetricsCollector(x86_backend.hw_info)
        profiler = QuantizationProfiler(x86_backend, metrics_collector)
        
        # Mock backend.load_model_safe
        with patch.object(x86_backend, 'load_model_safe', return_value=mock_llama_cpp):
            # Profile quantization
            result = profiler.profile_quantization(
                model_path=f"/tmp/test-model-{quant.lower()}.gguf",
                quant=quant,
                prompt="Test prompt",
                max_tokens=50,
                warmup_tokens=5,
                context_size=2048
            )
        
        # Property: All metrics should be valid (non-negative)
        assert result.ttft_ms >= 0, (
            f"Property violation: TTFT is {result.ttft_ms} ms for {quant}. Expected >= 0."
        )
        assert result.prefill_tps >= 0, (
            f"Property violation: Prefill TPS is {result.prefill_tps} t/s for {quant}. Expected >= 0."
        )
        assert result.decode_tps >= 0, (
            f"Property violation: Decode TPS is {result.decode_tps} t/s for {quant}. Expected >= 0."
        )
        assert result.prompt_tokens > 0, (
            f"Property violation: Prompt tokens is {result.prompt_tokens} for {quant}. Expected > 0."
        )
        assert result.output_tokens > 0, (
            f"Property violation: Output tokens is {result.output_tokens} for {quant}. Expected > 0."
        )
    
    @given(
        st.integers(min_value=20, max_value=100),  # max_tokens
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_jetson_load_time_always_positive(
        self, jetson_backend, mock_llama_cpp, max_tokens
    ):
        """
        Property-based test: Load time is always positive for JetsonBackend.
        
        **Preservation Property**: For any max_tokens configuration, load time
        measurement produces positive values for JetsonBackend.
        
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (behavior preserved).
        """
        # Create profiler
        metrics_collector = MetricsCollector(jetson_backend.hw_info)
        profiler = QuantizationProfiler(jetson_backend, metrics_collector)
        
        # Mock backend.load_model_safe with simulated load time
        def mock_load_with_delay(*args, **kwargs):
            import time
            time.sleep(0.08)  # Simulate 80ms load time
            return mock_llama_cpp
        
        with patch.object(jetson_backend, 'load_model_safe', side_effect=mock_load_with_delay):
            # Profile quantization
            result = profiler.profile_quantization(
                model_path="/tmp/test-model.gguf",
                quant="Q4_0",
                prompt="Test prompt",
                max_tokens=max_tokens,
                warmup_tokens=5,
                context_size=2048
            )
        
        # Property: Load time should always be positive
        assert result.load_time_s > 0, (
            f"Property violation: Load time is {result.load_time_s}s for "
            f"max_tokens={max_tokens}. Expected > 0 for JetsonBackend."
        )
    
    @given(
        st.integers(min_value=20, max_value=100),  # max_tokens
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_jetson_ram_always_positive(
        self, jetson_backend, mock_llama_cpp, max_tokens
    ):
        """
        Property-based test: Peak RAM is always positive for JetsonBackend.
        
        **Preservation Property**: For any max_tokens configuration, RAM measurement
        produces positive values for JetsonBackend.
        
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (behavior preserved).
        """
        # Create profiler
        metrics_collector = MetricsCollector(jetson_backend.hw_info)
        profiler = QuantizationProfiler(jetson_backend, metrics_collector)
        
        # Mock backend.load_model_safe
        with patch.object(jetson_backend, 'load_model_safe', return_value=mock_llama_cpp):
            # Profile quantization
            result = profiler.profile_quantization(
                model_path="/tmp/test-model.gguf",
                quant="Q4_0",
                prompt="Test prompt",
                max_tokens=max_tokens,
                warmup_tokens=5,
                context_size=2048
            )
        
        # Property: Peak RAM should always be positive
        assert result.peak_ram_mb > 0, (
            f"Property violation: Peak RAM is {result.peak_ram_mb} MB for "
            f"max_tokens={max_tokens}. Expected > 0 for JetsonBackend."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
