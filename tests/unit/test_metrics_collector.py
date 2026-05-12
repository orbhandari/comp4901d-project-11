"""
Unit tests for MetricsCollector.

Tests metrics collection functionality with mocked dependencies.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from llm_benchmark.metrics import MetricsCollector
from llm_benchmark.models import HardwareInfo


@pytest.fixture
def hw_info_cpu_only():
    """Hardware info for CPU-only system."""
    return HardwareInfo(
        os_type="linux_x86",
        cpu_model="Intel Core i7",
        cpu_cores=8,
        cpu_features=["avx2"],
        total_ram_gb=16.0,
        available_ram_gb=8.0,
        has_gpu=False,
        has_thermal_sensors=False,
        has_power_sensors=False
    )


@pytest.fixture
def hw_info_with_gpu():
    """Hardware info for system with GPU."""
    return HardwareInfo(
        os_type="jetson_xavier_nx",
        cpu_model="ARM Cortex-A57",
        cpu_cores=6,
        cpu_features=[],
        total_ram_gb=8.0,
        available_ram_gb=4.0,
        has_gpu=True,
        gpu_model="NVIDIA Tegra X1",
        gpu_memory_gb=8.0,
        gpu_compute_capability="7.2",
        has_thermal_sensors=True,
        has_power_sensors=True
    )


def test_metrics_collector_initialization_cpu_only(hw_info_cpu_only):
    """Test MetricsCollector initialization on CPU-only system."""
    collector = MetricsCollector(hw_info_cpu_only)
    
    assert collector.hw_info == hw_info_cpu_only
    assert collector.process is not None
    assert collector.nvml_initialized is False
    assert collector.gpu_handle is None
    assert collector.thermal_monitor is None
    assert collector.power_monitor is None


def test_metrics_collector_initialization_with_gpu(hw_info_with_gpu):
    """Test MetricsCollector initialization with GPU (mocked)."""
    # Patch pynvml before importing/creating MetricsCollector
    with patch.dict('sys.modules', {'pynvml': MagicMock()}):
        import sys
        mock_pynvml = sys.modules['pynvml']
        
        mock_handle = Mock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        
        with patch('llm_benchmark.metrics.collector.ThermalMonitor') as mock_thermal, \
             patch('llm_benchmark.metrics.collector.PowerMonitor') as mock_power:
            
            collector = MetricsCollector(hw_info_with_gpu)
            
            assert collector.hw_info == hw_info_with_gpu
            assert collector.nvml_initialized is True
            assert collector.gpu_handle == mock_handle
            mock_pynvml.nvmlInit.assert_called_once()
            
            # Should initialize thermal and power monitors
            mock_thermal.assert_called_once()
            mock_power.assert_called_once()


def test_collect_inference_metrics_basic(hw_info_cpu_only):
    """Test basic inference metrics collection."""
    collector = MetricsCollector(hw_info_cpu_only)
    
    # Mock llama model with timing
    mock_llm = Mock()
    
    def mock_stream_generator():
        import time
        time.sleep(0.01)  # Small delay to ensure measurable time
        yield {'choices': [{'text': 'token1'}], 'usage': {'prompt_tokens': 10}}
        time.sleep(0.01)
        yield {'choices': [{'text': 'token2'}]}
        time.sleep(0.01)
        yield {'choices': [{'text': 'token3'}]}
    
    mock_llm.return_value = mock_stream_generator()
    
    # Collect metrics (disable background monitoring for simpler test)
    metrics = collector.collect_inference_metrics(
        mock_llm,
        "Test prompt",
        max_tokens=10,
        enable_background_monitoring=False
    )
    
    # Verify metrics
    assert metrics.ttft_ms > 0
    assert metrics.total_time_s > 0
    assert metrics.prompt_tokens == 10
    assert metrics.output_tokens == 3
    assert metrics.peak_memory_mb > 0
    assert len(metrics.per_token_latency_ms) == 3
    
    # CPU-only system should not have GPU metrics
    assert metrics.gpu_memory_mb is None
    assert metrics.gpu_utilization_pct is None
    assert metrics.used_gpu_acceleration is False
    
    # Background monitoring disabled, so no aggregated stats
    assert metrics.cpu_temp_stats is None
    assert metrics.gpu_temp_stats is None
    assert metrics.power_stats is None
    assert metrics.thermal_throttled is False


def test_collect_inference_metrics_empty_prompt(hw_info_cpu_only):
    """Test that empty prompt raises ValueError."""
    collector = MetricsCollector(hw_info_cpu_only)
    mock_llm = Mock()
    
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        collector.collect_inference_metrics(mock_llm, "", max_tokens=10)


def test_collect_inference_metrics_with_gpu(hw_info_with_gpu):
    """Test metrics collection with GPU metrics."""
    # Patch pynvml before creating MetricsCollector
    with patch.dict('sys.modules', {'pynvml': MagicMock()}):
        import sys
        mock_pynvml = sys.modules['pynvml']
        
        # Setup GPU mocks
        mock_handle = Mock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        
        # Mock GPU memory info
        mock_mem_info = Mock()
        mock_mem_info.used = 2048 * 1024 * 1024  # 2048 MB
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem_info
        
        # Mock GPU utilization
        mock_utilization = Mock()
        mock_utilization.gpu = 75
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_utilization
        
        collector = MetricsCollector(hw_info_with_gpu)
        
        # Mock llama model
        mock_llm = Mock()
        mock_stream = [
            {'choices': [{'text': 'token1'}], 'usage': {'prompt_tokens': 10}},
            {'choices': [{'text': 'token2'}]},
        ]
        mock_llm.return_value = iter(mock_stream)
        
        # Collect metrics
        metrics = collector.collect_inference_metrics(
            mock_llm,
            "Test prompt",
            max_tokens=10
        )
        
        # Verify GPU metrics are collected
        assert metrics.gpu_memory_mb == 2048.0
        assert metrics.gpu_utilization_pct == 75.0
        assert metrics.used_gpu_acceleration is True


def test_gpu_metrics_collection_failure(hw_info_with_gpu):
    """Test that GPU metrics collection failure is handled gracefully."""
    with patch.dict('sys.modules', {'pynvml': MagicMock()}):
        import sys
        mock_pynvml = sys.modules['pynvml']
        
        # Setup GPU mocks
        mock_handle = Mock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        
        # Mock GPU metrics to raise exception
        mock_pynvml.nvmlDeviceGetMemoryInfo.side_effect = RuntimeError("GPU error")
        
        collector = MetricsCollector(hw_info_with_gpu)
        
        # Mock llama model
        mock_llm = Mock()
        mock_stream = [
            {'choices': [{'text': 'token1'}], 'usage': {'prompt_tokens': 10}},
        ]
        mock_llm.return_value = iter(mock_stream)
        
        # Collect metrics - should not raise exception
        metrics = collector.collect_inference_metrics(
            mock_llm,
            "Test prompt",
            max_tokens=10
        )
        
        # GPU metrics should be None when collection fails
        assert metrics.gpu_memory_mb is None
        assert metrics.gpu_utilization_pct is None
        assert metrics.used_gpu_acceleration is False


def test_prefill_throughput_calculation(hw_info_cpu_only):
    """Test prefill throughput calculation."""
    collector = MetricsCollector(hw_info_cpu_only)
    
    # Mock llama model with known timing
    mock_llm = Mock()
    
    # Create a stream that simulates timing
    def mock_stream_generator():
        import time
        # First token after 0.1 seconds (TTFT)
        time.sleep(0.1)
        yield {'choices': [{'text': 'token1'}], 'usage': {'prompt_tokens': 100}}
        # Subsequent tokens
        time.sleep(0.01)
        yield {'choices': [{'text': 'token2'}]}
    
    mock_llm.return_value = mock_stream_generator()
    
    metrics = collector.collect_inference_metrics(
        mock_llm,
        "Test prompt",
        max_tokens=10
    )
    
    # Prefill throughput should be approximately 100 tokens / 0.1s = 1000 t/s
    # Allow some tolerance for timing variations
    assert 800 <= metrics.prefill_tps <= 1200
    assert metrics.prompt_tokens == 100


def test_decode_throughput_calculation(hw_info_cpu_only):
    """Test decode throughput calculation."""
    collector = MetricsCollector(hw_info_cpu_only)
    
    # Mock llama model with known timing
    mock_llm = Mock()
    
    def mock_stream_generator():
        import time
        # First token (TTFT)
        time.sleep(0.1)
        yield {'choices': [{'text': 'token1'}], 'usage': {'prompt_tokens': 10}}
        # 4 more tokens, 0.01s each = 0.04s total decode time
        for i in range(4):
            time.sleep(0.01)
            yield {'choices': [{'text': f'token{i+2}'}]}
    
    mock_llm.return_value = mock_stream_generator()
    
    metrics = collector.collect_inference_metrics(
        mock_llm,
        "Test prompt",
        max_tokens=10
    )
    
    # Decode throughput: (5 - 1) tokens / 0.04s = 100 t/s
    # Allow tolerance for timing variations
    assert 80 <= metrics.decode_tps <= 120
    assert metrics.output_tokens == 5


def test_per_token_latency_tracking(hw_info_cpu_only):
    """Test per-token latency tracking."""
    collector = MetricsCollector(hw_info_cpu_only)
    
    mock_llm = Mock()
    
    def mock_stream_generator():
        import time
        time.sleep(0.01)  # Small delay to ensure measurable time
        yield {'choices': [{'text': 'token1'}], 'usage': {'prompt_tokens': 10}}
        time.sleep(0.01)
        yield {'choices': [{'text': 'token2'}]}
        time.sleep(0.01)
        yield {'choices': [{'text': 'token3'}]}
    
    mock_llm.return_value = mock_stream_generator()
    
    metrics = collector.collect_inference_metrics(
        mock_llm,
        "Test prompt",
        max_tokens=10
    )
    
    # Should have latency for each token
    assert len(metrics.per_token_latency_ms) == 3
    
    # All latencies should be positive
    for latency in metrics.per_token_latency_ms:
        assert latency > 0


def test_memory_measurement(hw_info_cpu_only):
    """Test memory usage measurement."""
    collector = MetricsCollector(hw_info_cpu_only)
    
    mock_llm = Mock()
    mock_stream = [
        {'choices': [{'text': 'token1'}], 'usage': {'prompt_tokens': 10}},
    ]
    mock_llm.return_value = iter(mock_stream)
    
    metrics = collector.collect_inference_metrics(
        mock_llm,
        "Test prompt",
        max_tokens=10
    )
    
    # Peak memory should be measured
    assert metrics.peak_memory_mb > 0
    
    # Should be a reasonable value (not negative, not absurdly large)
    assert 0 < metrics.peak_memory_mb < 100000


def test_thermal_metrics_collection(hw_info_with_gpu):
    """Test thermal metrics collection when sensors available."""
    with patch.dict('sys.modules', {'pynvml': MagicMock()}):
        import sys
        mock_pynvml = sys.modules['pynvml']
        
        # Setup GPU mocks
        mock_handle = Mock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 65.0
        mock_pynvml.NVML_TEMPERATURE_GPU = 0
        
        with patch('llm_benchmark.metrics.collector.psutil') as mock_psutil:
            # Setup CPU temperature mock
            mock_temp = Mock()
            mock_temp.current = 55.0
            mock_psutil.sensors_temperatures.return_value = {
                'coretemp': [mock_temp]
            }
            
            # Setup process mock
            mock_process = Mock()
            mock_mem_info = Mock()
            mock_mem_info.rss = 1024 * 1024 * 1024  # 1GB
            mock_process.memory_info.return_value = mock_mem_info
            mock_psutil.Process.return_value = mock_process
            
            collector = MetricsCollector(hw_info_with_gpu)
            
            # Mock llama model
            mock_llm = Mock()
            mock_stream = [
                {'choices': [{'text': 'token1'}], 'usage': {'prompt_tokens': 10}},
            ]
            mock_llm.return_value = iter(mock_stream)
            
            metrics = collector.collect_inference_metrics(
                mock_llm,
                "Test prompt",
                max_tokens=10
            )
            
            # Verify thermal metrics
            assert metrics.cpu_temp_c == 55.0
            assert metrics.gpu_temp_c == 65.0


def test_inference_failure_propagates(hw_info_cpu_only):
    """Test that inference failures are propagated."""
    collector = MetricsCollector(hw_info_cpu_only)
    
    # Mock llama model that raises exception
    mock_llm = Mock()
    mock_llm.side_effect = RuntimeError("Inference failed")
    
    with pytest.raises(RuntimeError, match="Inference failed"):
        collector.collect_inference_metrics(
            mock_llm,
            "Test prompt",
            max_tokens=10
        )


def test_no_tokens_generated(hw_info_cpu_only):
    """Test handling when no tokens are generated."""
    collector = MetricsCollector(hw_info_cpu_only)
    
    # Mock llama model that returns empty stream
    mock_llm = Mock()
    mock_llm.return_value = iter([])
    
    metrics = collector.collect_inference_metrics(
        mock_llm,
        "Test prompt",
        max_tokens=10
    )
    
    # Should handle gracefully
    assert metrics.ttft_ms == 0.0
    assert metrics.output_tokens == 0
    assert metrics.prefill_tps == 0.0
    assert metrics.decode_tps == 0.0


def test_rounding_precision(hw_info_cpu_only):
    """Test that metrics are rounded to 2 decimal places."""
    collector = MetricsCollector(hw_info_cpu_only)
    
    mock_llm = Mock()
    mock_stream = [
        {'choices': [{'text': 'token1'}], 'usage': {'prompt_tokens': 10}},
        {'choices': [{'text': 'token2'}]},
    ]
    mock_llm.return_value = iter(mock_stream)
    
    metrics = collector.collect_inference_metrics(
        mock_llm,
        "Test prompt",
        max_tokens=10,
        enable_background_monitoring=False
    )
    
    # Check that values are rounded to 2 decimal places
    assert metrics.ttft_ms == round(metrics.ttft_ms, 2)
    assert metrics.prefill_tps == round(metrics.prefill_tps, 2)
    assert metrics.decode_tps == round(metrics.decode_tps, 2)
    assert metrics.total_time_s == round(metrics.total_time_s, 2)
    assert metrics.peak_memory_mb == round(metrics.peak_memory_mb, 2)
    
    for latency in metrics.per_token_latency_ms:
        assert latency == round(latency, 2)


def test_background_monitoring_integration(hw_info_with_gpu):
    """Test background monitoring integration with inference."""
    with patch.dict('sys.modules', {'pynvml': MagicMock()}):
        import sys
        mock_pynvml = sys.modules['pynvml']
        
        # Setup GPU mocks
        mock_handle = Mock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        
        with patch('llm_benchmark.metrics.collector.ThermalMonitor') as mock_thermal_class, \
             patch('llm_benchmark.metrics.collector.PowerMonitor') as mock_power_class:
            
            # Setup thermal monitor mock
            mock_thermal = Mock()
            mock_thermal.stop_monitoring.return_value = (
                (50.0, 55.0, 60.0),  # CPU temp stats
                (60.0, 65.0, 70.0),  # GPU temp stats
                False  # Not throttled
            )
            mock_thermal_class.return_value = mock_thermal
            
            # Setup power monitor mock
            mock_power = Mock()
            mock_power.stop_monitoring.return_value = (5.0, 6.0, 7.0)  # Power stats
            mock_power_class.return_value = mock_power
            
            collector = MetricsCollector(hw_info_with_gpu)
            
            # Mock llama model
            mock_llm = Mock()
            mock_stream = [
                {'choices': [{'text': 'token1'}], 'usage': {'prompt_tokens': 10}},
                {'choices': [{'text': 'token2'}]},
            ]
            mock_llm.return_value = iter(mock_stream)
            
            # Collect metrics with background monitoring enabled
            metrics = collector.collect_inference_metrics(
                mock_llm,
                "Test prompt",
                max_tokens=10,
                enable_background_monitoring=True
            )
            
            # Verify background monitoring was started and stopped
            mock_thermal.start_monitoring.assert_called_once()
            mock_thermal.stop_monitoring.assert_called_once()
            mock_power.start_monitoring.assert_called_once()
            mock_power.stop_monitoring.assert_called_once()
            
            # Verify aggregated stats are present
            assert metrics.cpu_temp_stats == (50.0, 55.0, 60.0)
            assert metrics.gpu_temp_stats == (60.0, 65.0, 70.0)
            assert metrics.power_stats == (5.0, 6.0, 7.0)
            assert metrics.thermal_throttled is False


def test_background_monitoring_with_throttling(hw_info_with_gpu):
    """Test background monitoring detects thermal throttling."""
    with patch.dict('sys.modules', {'pynvml': MagicMock()}):
        import sys
        mock_pynvml = sys.modules['pynvml']
        
        # Setup GPU mocks
        mock_handle = Mock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        
        with patch('llm_benchmark.metrics.collector.ThermalMonitor') as mock_thermal_class, \
             patch('llm_benchmark.metrics.collector.PowerMonitor') as mock_power_class:
            
            # Setup thermal monitor mock with throttling
            mock_thermal = Mock()
            mock_thermal.stop_monitoring.return_value = (
                (80.0, 88.0, 95.0),  # CPU temp stats (high temps)
                (85.0, 90.0, 95.0),  # GPU temp stats (high temps)
                True  # Throttled!
            )
            mock_thermal_class.return_value = mock_thermal
            
            # Setup power monitor mock
            mock_power = Mock()
            mock_power.stop_monitoring.return_value = (5.0, 6.0, 7.0)
            mock_power_class.return_value = mock_power
            
            collector = MetricsCollector(hw_info_with_gpu)
            
            # Mock llama model
            mock_llm = Mock()
            mock_stream = [
                {'choices': [{'text': 'token1'}], 'usage': {'prompt_tokens': 10}},
            ]
            mock_llm.return_value = iter(mock_stream)
            
            # Collect metrics
            metrics = collector.collect_inference_metrics(
                mock_llm,
                "Test prompt",
                max_tokens=10,
                enable_background_monitoring=True
            )
            
            # Verify throttling was detected
            assert metrics.thermal_throttled is True
            assert metrics.cpu_temp_stats[2] >= 85.0  # Max CPU temp high
            assert metrics.gpu_temp_stats[2] >= 85.0  # Max GPU temp high


def test_start_stop_monitoring_methods(hw_info_with_gpu):
    """Test start_monitoring and stop_monitoring methods."""
    with patch.dict('sys.modules', {'pynvml': MagicMock()}):
        import sys
        mock_pynvml = sys.modules['pynvml']
        
        # Setup mocks
        mock_handle = Mock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        
        with patch('llm_benchmark.metrics.collector.ThermalMonitor') as mock_thermal_class, \
             patch('llm_benchmark.metrics.collector.PowerMonitor') as mock_power_class:
            
            mock_thermal = Mock()
            mock_thermal.stop_monitoring.return_value = (None, None, False)
            mock_thermal_class.return_value = mock_thermal
            
            mock_power = Mock()
            mock_power.stop_monitoring.return_value = None
            mock_power_class.return_value = mock_power
            
            collector = MetricsCollector(hw_info_with_gpu)
            
            # Test start_monitoring
            collector.start_monitoring(throttle_threshold_c=90.0)
            
            mock_thermal.start_monitoring.assert_called_once_with(90.0)
            mock_power.start_monitoring.assert_called_once()
            
            # Test stop_monitoring
            results = collector.stop_monitoring()
            
            mock_thermal.stop_monitoring.assert_called_once()
            mock_power.stop_monitoring.assert_called_once()
            
            assert 'cpu_temp_stats' in results
            assert 'gpu_temp_stats' in results
            assert 'thermal_throttled' in results
            assert 'power_stats' in results
