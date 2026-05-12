"""
Bug Condition Exploration Tests - Android Metrics Measurement Errors

**Validates: Requirements 1.1, 1.2, 1.3, 1.4**

This test explores the bug condition where Android profiling with NativeLlamaCpp produces:
1. Load time = 0.00s because measurement occurs during path validation, not actual loading
2. Peak RAM = 176 MB because only Python process is measured, excluding subprocess
3. Decode TPS = 45778.25 t/s on first iteration (impossibly high outlier)

**CRITICAL**: These tests MUST FAIL on unfixed code - failure confirms the bugs exist.
**DO NOT attempt to fix the tests or the code when they fail.**

**EXPECTED OUTCOME**: Tests FAIL (this is correct - it proves the bugs exist)

The tests encode the expected behavior - they will validate the fix when they pass after implementation.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings, assume, HealthCheck
import statistics

from llm_benchmark.profiler.quantization import QuantizationProfiler
from llm_benchmark.hardware.hal import AndroidBackend
from llm_benchmark.inference.native_llama import NativeLlamaCpp
from llm_benchmark.models import HardwareInfo, QuantizationResult
from llm_benchmark.metrics.collector import MetricsCollector


@pytest.fixture
def android_hardware_info():
    """Create hardware info for Android platform."""
    return HardwareInfo(
        os_type="android",
        cpu_model="Snapdragon 8 Gen 2",
        cpu_cores=8,
        cpu_features=["neon", "fp16"],
        total_ram_gb=12.0,
        available_ram_gb=8.0,
        has_gpu=False,
        gpu_model=None,
        gpu_memory_gb=None,
        gpu_compute_capability=None,
        has_thermal_sensors=True,
        has_power_sensors=False,
    )


@pytest.fixture
def android_backend(android_hardware_info):
    """Create Android backend for testing."""
    return AndroidBackend(android_hardware_info)


@pytest.fixture
def mock_native_llama():
    """Create a mock NativeLlamaCpp instance that simulates Android behavior."""
    mock_llm = MagicMock(spec=NativeLlamaCpp)
    
    # Simulate fast __init__ (path validation only, no model loading)
    mock_llm.model_path = Path("/data/local/tmp/tinyllama-q2_k.gguf")
    mock_llm.n_ctx = 2048
    mock_llm.n_threads = 4
    mock_llm.n_batch = 512
    
    # Simulate tokenize method (approximate)
    def mock_tokenize(text):
        if isinstance(text, bytes):
            char_count = len(text)
        else:
            char_count = len(text.encode('utf-8'))
        token_count = max(1, char_count // 4)
        return list(range(token_count))
    
    mock_llm.tokenize = mock_tokenize
    
    # Simulate streaming inference
    def mock_call(prompt, max_tokens=100, stream=True, **kwargs):
        """Simulate inference with character-by-character streaming."""
        import time
        
        # Simulate model loading time on first call (1-2 seconds for Android)
        # This simulates the actual model loading that happens during first inference
        time.sleep(1.0)
        
        output = "This is a test response from the model. " * (max_tokens // 10)
        output = output[:max_tokens]
        
        if stream:
            # Return generator for streaming
            def gen():
                for char in output:
                    yield {
                        'choices': [{
                            'text': char,
                            'finish_reason': None
                        }]
                    }
                yield {
                    'choices': [{
                        'text': '',
                        'finish_reason': 'stop'
                    }]
                }
            return gen()
        else:
            # Return dict for non-streaming
            return {
                'choices': [{
                    'text': output,
                    'finish_reason': 'stop'
                }]
            }
    
    # Use side_effect to make the mock callable
    mock_llm.side_effect = mock_call
    
    return mock_llm


class TestBugConditionAndroidMetrics:
    """
    Property 1: Bug Condition - Android Metrics Measurement Errors
    
    **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
    
    Tests that the bugs exist in unfixed code:
    - Load time measured around path validation (0.00s) instead of actual loading (1-5s)
    - Peak RAM only captures Python process (176 MB) instead of Python + subprocess (400-800 MB)
    - Decode TPS shows 20x outlier on first iteration (45778 t/s) instead of consistent ~2200 t/s
    """
    
    def test_load_time_measured_during_path_validation_not_actual_loading(
        self, android_backend, mock_native_llama
    ):
        """
        Test that load time is measured around path validation, not actual model loading.
        
        **Bug Condition**: Load time measured around backend.load_model_safe() which only
        validates paths in NativeLlamaCpp.__init__(), not during actual model loading.
        
        **Current Behavior (unfixed)**: Load time = 0.00s (path validation is instant)
        **Expected Behavior (fixed)**: Load time > 0.5s (actual model loading during first inference)
        
        **Expected on unfixed code**: This test FAILS because load time is 0.00s.
        **Expected after fix**: This test PASSES because load time is measured during first inference.
        """
        # Create profiler
        metrics_collector = MetricsCollector(android_backend.hw_info)
        profiler = QuantizationProfiler(android_backend, metrics_collector)
        
        # Mock backend.load_model_safe to return our mock NativeLlamaCpp instantly
        with patch.object(android_backend, 'load_model_safe', return_value=mock_native_llama):
            # Profile quantization
            result = profiler.profile_quantization(
                model_path="/data/local/tmp/tinyllama-q2_k.gguf",
                quant="Q2_K",
                prompt="What is the capital of France?",
                max_tokens=50,
                warmup_tokens=5,
                context_size=2048
            )
        
        # FIXED: Load time should be > 0.5s for actual model loading
        # After fix: load_time_s should be 1-5s (measured during first inference)
        # The mock simulates 1 second load time during first inference
        
        assert result.load_time_s > 0.5, (
            f"Fix verification failed: Load time is {result.load_time_s}s (expected > 0.5s). "
            f"Load time should be measured during first inference for Android. "
            f"Expected load time: 1-5s for TinyLlama models."
        )
    
    def test_peak_ram_only_captures_python_process_not_subprocess(
        self, android_backend, mock_native_llama
    ):
        """
        Test that peak RAM measurement only captures Python process, not subprocess.
        
        **Bug Condition**: RAM measured using self.process.memory_info().rss which only
        captures the Python process, excluding the native llama-cli subprocess.
        
        **Current Behavior (unfixed)**: Peak RAM = 176 MB (only Python process)
        **Expected Behavior (fixed)**: Peak RAM > 300 MB (Python + subprocess)
        
        **Expected on unfixed code**: This test FAILS because peak RAM is ~176 MB.
        **Expected after fix**: This test PASSES because subprocess memory is included.
        """
        # Create profiler
        metrics_collector = MetricsCollector(android_backend.hw_info)
        profiler = QuantizationProfiler(android_backend, metrics_collector)
        
        # Create a mock child process with memory
        mock_child = MagicMock()
        mock_child_memory_info = MagicMock()
        mock_child_memory_info.rss = 400 * 1024 * 1024  # 400 MB subprocess memory
        mock_child.memory_info.return_value = mock_child_memory_info
        
        # Patch psutil.Process().children() to return our mock child
        with patch.object(android_backend, 'load_model_safe', return_value=mock_native_llama), \
             patch.object(profiler.process, 'children', return_value=[mock_child]):
            # Profile quantization
            result = profiler.profile_quantization(
                model_path="/data/local/tmp/tinyllama-q2_k.gguf",
                quant="Q2_K",
                prompt="What is the capital of France?",
                max_tokens=50,
                warmup_tokens=5,
                context_size=2048
            )
        
        # FIXED: Peak RAM should be > 300 MB (Python + subprocess)
        # After fix: peak_ram_mb should be 400-600 MB for Q2_K (includes subprocess)
        # The mock simulates 400 MB subprocess memory
        
        assert result.peak_ram_mb > 300, (
            f"Fix verification failed: Peak RAM is {result.peak_ram_mb} MB (expected > 300 MB). "
            f"RAM measurement should include subprocess memory for Android. "
            f"Expected peak RAM: 400-600 MB for Q2_K quantization (Python + subprocess)."
        )
    
    def test_decode_tps_shows_impossible_outlier_on_first_iteration(
        self, android_backend, mock_native_llama
    ):
        """
        Test that decode TPS shows impossibly high outlier on first iteration.
        
        **Bug Condition**: First iteration decode TPS = 45778.25 t/s (impossibly high),
        while subsequent iterations show ~2200 t/s. This is a 20x outlier.
        
        **Current Behavior (unfixed)**: First iteration decode TPS > 10000 t/s (outlier)
        **Expected Behavior (fixed)**: All iterations decode TPS ~2200 t/s (consistent)
        
        **Expected on unfixed code**: This test FAILS because first iteration has outlier.
        **Expected after fix**: This test PASSES because decode TPS is consistent.
        """
        # Create profiler
        metrics_collector = MetricsCollector(android_backend.hw_info)
        profiler = QuantizationProfiler(android_backend, metrics_collector)
        
        # Mock backend.load_model_safe to return our mock NativeLlamaCpp
        with patch.object(android_backend, 'load_model_safe', return_value=mock_native_llama):
            # Profile multiple iterations
            results = []
            for i in range(3):
                result = profiler.profile_quantization(
                    model_path="/data/local/tmp/tinyllama-q2_k.gguf",
                    quant="Q2_K",
                    prompt="What is the capital of France?",
                    max_tokens=50,
                    warmup_tokens=5,
                    context_size=2048
                )
                results.append(result)
        
        # Extract decode TPS values
        decode_tps_values = [r.decode_tps for r in results]
        
        # Calculate median
        median_tps = statistics.median(decode_tps_values)
        
        # Check if any value is > 2x median (outlier detection)
        outliers = [tps for tps in decode_tps_values if tps > 2 * median_tps]
        
        # BUG: First iteration should NOT have 20x outlier
        # Current behavior: First iteration decode_tps = 45778.25 t/s (20x outlier)
        # After fix: All iterations should be within 2x of median (~2200 t/s)
        
        assert len(outliers) == 0, (
            f"Bug confirmed: Decode TPS outlier detected. "
            f"Values: {decode_tps_values}, Median: {median_tps:.2f} t/s, "
            f"Outliers (>2x median): {outliers}. "
            f"First iteration typically shows 45778.25 t/s (20x outlier) due to timing measurement issue. "
            f"Expected: All values within 2x of median (~2200 t/s for typical Android hardware)."
        )
    
    def test_statistical_summary_has_negative_confidence_interval_due_to_outlier(
        self, android_backend, mock_native_llama
    ):
        """
        Test that statistical summaries have negative confidence intervals due to outlier.
        
        **Bug Condition**: First iteration outlier (45778 t/s) propagates into statistical
        calculations, causing huge standard deviation and negative lower confidence bound.
        
        **Current Behavior (unfixed)**: Confidence interval lower bound < 0
        **Expected Behavior (fixed)**: Confidence interval lower bound >= 0
        
        **Expected on unfixed code**: This test FAILS because lower bound is negative.
        **Expected after fix**: This test PASSES because outlier is handled correctly.
        """
        # Create profiler
        metrics_collector = MetricsCollector(android_backend.hw_info)
        profiler = QuantizationProfiler(android_backend, metrics_collector)
        
        # Mock backend.load_model_safe to return our mock NativeLlamaCpp
        with patch.object(android_backend, 'load_model_safe', return_value=mock_native_llama):
            # Profile multiple iterations
            results = []
            for i in range(5):
                result = profiler.profile_quantization(
                    model_path="/data/local/tmp/tinyllama-q2_k.gguf",
                    quant="Q2_K",
                    prompt="What is the capital of France?",
                    max_tokens=50,
                    warmup_tokens=5,
                    context_size=2048
                )
                results.append(result)
        
        # Calculate statistical summary
        decode_tps_values = [r.decode_tps for r in results]
        mean_tps = statistics.mean(decode_tps_values)
        stdev_tps = statistics.stdev(decode_tps_values) if len(decode_tps_values) > 1 else 0
        
        # Calculate 95% confidence interval (assuming normal distribution)
        # CI = mean ± 1.96 * (stdev / sqrt(n))
        import math
        n = len(decode_tps_values)
        margin_of_error = 1.96 * (stdev_tps / math.sqrt(n))
        lower_bound = mean_tps - margin_of_error
        upper_bound = mean_tps + margin_of_error
        
        # BUG: Lower bound should NOT be negative
        # Current behavior: lower_bound = -11732.58 (negative due to outlier)
        # After fix: lower_bound should be >= 0 (reasonable confidence interval)
        
        assert lower_bound >= 0, (
            f"Bug confirmed: Confidence interval lower bound is negative ({lower_bound:.2f}). "
            f"Mean: {mean_tps:.2f} t/s, StdDev: {stdev_tps:.2f}, "
            f"CI: [{lower_bound:.2f}, {upper_bound:.2f}]. "
            f"This is caused by the first-iteration outlier (45778 t/s) propagating into "
            f"statistical calculations, inflating the standard deviation. "
            f"Expected: Non-negative confidence interval with reasonable bounds."
        )
    
    @given(
        st.integers(min_value=20, max_value=100),  # max_tokens
        st.integers(min_value=2, max_value=10),    # warmup_tokens
    )
    @settings(
        max_examples=10, 
        deadline=None, 
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture]
    )
    def test_property_load_time_always_near_zero_for_android(
        self, android_backend, mock_native_llama, max_tokens, warmup_tokens
    ):
        """
        Property-based test: Load time is always near zero for Android (bug condition).
        
        **Bug Condition**: For any max_tokens and warmup_tokens configuration,
        load time measured around backend.load_model_safe() is always ~0.00s.
        
        **Expected on unfixed code**: This test FAILS because load time is always ~0.00s.
        **Expected after fix**: This test PASSES because load time is measured during first inference.
        """
        # Create profiler
        metrics_collector = MetricsCollector(android_backend.hw_info)
        profiler = QuantizationProfiler(android_backend, metrics_collector)
        
        # Mock backend.load_model_safe to return our mock NativeLlamaCpp
        with patch.object(android_backend, 'load_model_safe', return_value=mock_native_llama):
            # Profile quantization
            result = profiler.profile_quantization(
                model_path="/data/local/tmp/tinyllama-q2_k.gguf",
                quant="Q2_K",
                prompt="Test prompt",
                max_tokens=max_tokens,
                warmup_tokens=warmup_tokens,
                context_size=2048
            )
        
        # Property: Load time should be > 0.5s for actual model loading
        # After fix: Load time is measured during first inference
        assert result.load_time_s > 0.5, (
            f"Property verification: Load time is {result.load_time_s}s for "
            f"max_tokens={max_tokens}, warmup_tokens={warmup_tokens}. "
            f"Expected > 0.5s for actual model loading."
        )
    
    @given(
        st.sampled_from(["Q2_K", "Q4_0", "Q8_0"]),  # quantization levels
    )
    @settings(
        max_examples=5, 
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_peak_ram_always_low_for_android(
        self, android_backend, mock_native_llama, quant
    ):
        """
        Property-based test: Peak RAM is always low for Android (bug condition).
        
        **Bug Condition**: For any quantization level, peak RAM only captures Python process,
        resulting in consistently low values (~176 MB) regardless of model size.
        
        **Expected on unfixed code**: This test FAILS because peak RAM is always ~176 MB.
        **Expected after fix**: This test PASSES because subprocess memory is included.
        """
        # Create profiler
        metrics_collector = MetricsCollector(android_backend.hw_info)
        profiler = QuantizationProfiler(android_backend, metrics_collector)
        
        # Create a mock child process with memory
        mock_child = MagicMock()
        mock_child_memory_info = MagicMock()
        mock_child_memory_info.rss = 400 * 1024 * 1024  # 400 MB subprocess memory
        mock_child.memory_info.return_value = mock_child_memory_info
        
        # Mock backend.load_model_safe to return our mock NativeLlamaCpp
        with patch.object(android_backend, 'load_model_safe', return_value=mock_native_llama), \
             patch.object(profiler.process, 'children', return_value=[mock_child]):
            # Profile quantization
            result = profiler.profile_quantization(
                model_path=f"/data/local/tmp/tinyllama-{quant.lower()}.gguf",
                quant=quant,
                prompt="Test prompt",
                max_tokens=50,
                warmup_tokens=5,
                context_size=2048
            )
        
        # Property: Peak RAM should be > 300 MB (Python + subprocess)
        # After fix: Peak RAM includes subprocess memory
        assert result.peak_ram_mb > 300, (
            f"Property verification: Peak RAM is {result.peak_ram_mb} MB for {quant} quantization. "
            f"Expected > 300 MB (Python + subprocess). "
            f"Subprocess memory should be included in measurement."
        )
    
    @given(
        st.integers(min_value=2, max_value=5),  # number of iterations
    )
    @settings(
        max_examples=5, 
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_property_decode_tps_has_outliers_for_android(
        self, android_backend, mock_native_llama, num_iterations
    ):
        """
        Property-based test: Decode TPS has outliers for Android (bug condition).
        
        **Bug Condition**: For any number of iterations, first iteration shows
        impossibly high decode TPS (>10000 t/s) while subsequent iterations are normal.
        
        **Expected on unfixed code**: This test FAILS because outliers are present.
        **Expected after fix**: This test PASSES because decode TPS is consistent.
        """
        # Create profiler
        metrics_collector = MetricsCollector(android_backend.hw_info)
        profiler = QuantizationProfiler(android_backend, metrics_collector)
        
        # Mock backend.load_model_safe to return our mock NativeLlamaCpp
        with patch.object(android_backend, 'load_model_safe', return_value=mock_native_llama):
            # Profile multiple iterations
            results = []
            for i in range(num_iterations):
                result = profiler.profile_quantization(
                    model_path="/data/local/tmp/tinyllama-q2_k.gguf",
                    quant="Q2_K",
                    prompt="Test prompt",
                    max_tokens=50,
                    warmup_tokens=5,
                    context_size=2048
                )
                results.append(result)
        
        # Extract decode TPS values
        decode_tps_values = [r.decode_tps for r in results]
        
        # Calculate median
        median_tps = statistics.median(decode_tps_values)
        
        # Property: All values should be within 2x of median (no outliers)
        # Bug: First iteration has 20x outlier
        for i, tps in enumerate(decode_tps_values):
            assert tps <= 2 * median_tps, (
                f"Property violation: Iteration {i} has decode TPS outlier. "
                f"Value: {tps:.2f} t/s, Median: {median_tps:.2f} t/s, "
                f"Ratio: {tps / median_tps:.1f}x. "
                f"Expected all values within 2x of median."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
