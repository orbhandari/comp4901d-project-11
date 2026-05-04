"""
Unit tests for Metrics Collector.

Tests memory measurement, GPU metrics collection, TTFT calculation,
and throughput calculations with mocked dependencies.

Requirements: 3.1, 3.2, 3.3, 3.4, 4.3, 4.4
"""

import time
from unittest.mock import Mock, patch, MagicMock, PropertyMock

import pytest

from llm_benchmark.metrics import MetricsCollector
from llm_benchmark.models import HardwareInfo


class TestMemoryMeasurement:
    """Test memory measurement using mocked psutil."""
    
    @patch('llm_benchmark.metrics.collector.psutil.Process')
    def test_memory_measurement_basic(self, mock_process_class):
        """Test basic memory measurement."""
        # Mock process memory info
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 100  # 100 MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_process_class.return_value = mock_process
        
        hw_info = HardwareInfo(
            os_type="linux_x86",
            cpu_model="Test CPU",
            cpu_cores=8,
            cpu_features=[],
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            has_gpu=False,
            gpu_model=None,
            gpu_memory_gb=None,
            gpu_compute_capability=None,
            has_thermal_sensors=False,
            has_power_sensors=False
        )
        
        collector = MetricsCollector(hw_info)
        
        # Verify memory is measured correctly
        memory_mb = collector.process.memory_info().rss / (1024 * 1024)
        assert memory_mb == 100.0
    
    @patch('llm_benchmark.metrics.collector.psutil.Process')
    def test_memory_measurement_with_different_values(self, mock_process_class):
        """Test memory measurement with various memory values."""
        test_cases = [
            (1024 * 1024 * 50, 50.0),    # 50 MB
            (1024 * 1024 * 200, 200.0),  # 200 MB
            (1024 * 1024 * 1024, 1024.0), # 1 GB
        ]
        
        for rss_bytes, expected_mb in test_cases:
            mock_process = Mock()
            mock_memory_info = Mock()
            mock_memory_info.rss = rss_bytes
            mock_process.memory_info.return_value = mock_memory_info
            mock_process_class.return_value = mock_process
            
            hw_info = HardwareInfo(
                os_type="linux_x86",
                cpu_model="Test CPU",
                cpu_cores=8,
                cpu_features=[],
                total_ram_gb=16.0,
                available_ram_gb=8.0,
                has_gpu=False,
                gpu_model=None,
                gpu_memory_gb=None,
                gpu_compute_capability=None,
                has_thermal_sensors=False,
                has_power_sensors=False
            )
            
            collector = MetricsCollector(hw_info)
            memory_mb = collector.process.memory_info().rss / (1024 * 1024)
            
            assert memory_mb == expected_mb, \
                f"Expected {expected_mb} MB, got {memory_mb} MB"


class TestGPUMetricsCollection:
    """Test GPU metrics collection using mocked pynvml."""
    
    @patch('llm_benchmark.metrics.collector.psutil.Process')
    def test_gpu_memory_measurement(self, mock_process_class):
        """Test GPU memory measurement."""
        # Mock process
        mock_process = Mock()
        mock_process_class.return_value = mock_process
        
        # Mock pynvml module
        with patch.dict('sys.modules', {'pynvml': MagicMock()}):
            import sys
            mock_pynvml = sys.modules['pynvml']
            
            # Mock NVML initialization
            mock_pynvml.nvmlInit.return_value = None
            mock_handle = Mock()
            mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
            
            # Mock memory info
            mock_mem_info = Mock()
            mock_mem_info.used = 1024 * 1024 * 500  # 500 MB
            mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem_info
            
            hw_info = HardwareInfo(
                os_type="jetson_xavier_nx",
                cpu_model="ARM Cortex-A57",
                cpu_cores=6,
                cpu_features=[],
                total_ram_gb=8.0,
                available_ram_gb=6.0,
                has_gpu=True,
                gpu_model="NVIDIA Tegra X1",
                gpu_memory_gb=8.0,
                gpu_compute_capability="5.3",
                has_thermal_sensors=True,
                has_power_sensors=True
            )
            
            collector = MetricsCollector(hw_info)
            gpu_memory_mb = collector._get_gpu_memory_mb()
            
            assert gpu_memory_mb == 500.0, \
                f"Expected 500 MB GPU memory, got {gpu_memory_mb} MB"
    
    @patch('llm_benchmark.metrics.collector.psutil.Process')
    def test_gpu_utilization_measurement(self, mock_process_class):
        """Test GPU utilization measurement."""
        # Mock process
        mock_process = Mock()
        mock_process_class.return_value = mock_process
        
        # Mock pynvml module
        with patch.dict('sys.modules', {'pynvml': MagicMock()}):
            import sys
            mock_pynvml = sys.modules['pynvml']
            
            # Mock NVML initialization
            mock_pynvml.nvmlInit.return_value = None
            mock_handle = Mock()
            mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
            
            # Mock utilization rates
            mock_utilization = Mock()
            mock_utilization.gpu = 75  # 75% utilization
            mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_utilization
            
            hw_info = HardwareInfo(
                os_type="jetson_xavier_nx",
                cpu_model="ARM Cortex-A57",
                cpu_cores=6,
                cpu_features=[],
                total_ram_gb=8.0,
                available_ram_gb=6.0,
                has_gpu=True,
                gpu_model="NVIDIA Tegra X1",
                gpu_memory_gb=8.0,
                gpu_compute_capability="5.3",
                has_thermal_sensors=True,
                has_power_sensors=True
            )
            
            collector = MetricsCollector(hw_info)
            gpu_utilization = collector._get_gpu_utilization_pct()
            
            assert gpu_utilization == 75.0, \
                f"Expected 75% GPU utilization, got {gpu_utilization}%"
    
    @patch('llm_benchmark.metrics.collector.psutil.Process')
    def test_gpu_metrics_unavailable_without_gpu(self, mock_process_class):
        """Test GPU metrics are None when GPU is unavailable."""
        # Mock process
        mock_process = Mock()
        mock_process_class.return_value = mock_process
        
        hw_info = HardwareInfo(
            os_type="linux_x86",
            cpu_model="Test CPU",
            cpu_cores=8,
            cpu_features=[],
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            has_gpu=False,
            gpu_model=None,
            gpu_memory_gb=None,
            gpu_compute_capability=None,
            has_thermal_sensors=False,
            has_power_sensors=False
        )
        
        collector = MetricsCollector(hw_info)
        
        # GPU metrics should not be initialized
        assert collector.nvml_initialized is False
        assert collector.gpu_handle is None


class TestTTFTCalculation:
    """Test TTFT calculation with simulated streaming."""
    
    @patch('llm_benchmark.metrics.collector.psutil.Process')
    @patch('llm_benchmark.metrics.collector.time.perf_counter')
    def test_ttft_calculation_basic(self, mock_time, mock_process_class):
        """Test basic TTFT calculation."""
        # Mock time progression
        time_sequence = [
            0.0,    # Start time
            0.1,    # First token (TTFT = 100ms)
            0.15,   # Second token
            0.2,    # Third token
            0.2     # End time
        ]
        mock_time.side_effect = time_sequence
        
        # Mock process memory
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 100
        mock_process.memory_info.return_value = mock_memory_info
        mock_process_class.return_value = mock_process
        
        # Mock LLM streaming
        mock_llm = Mock()
        mock_llm.return_value = [
            {'usage': {'prompt_tokens': 10}},  # First chunk with prompt tokens
            {},  # Second token
            {},  # Third token
        ]
        
        hw_info = HardwareInfo(
            os_type="linux_x86",
            cpu_model="Test CPU",
            cpu_cores=8,
            cpu_features=[],
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            has_gpu=False,
            gpu_model=None,
            gpu_memory_gb=None,
            gpu_compute_capability=None,
            has_thermal_sensors=False,
            has_power_sensors=False
        )
        
        collector = MetricsCollector(hw_info)
        metrics = collector._collect_inference_metrics_impl(
            mock_llm,
            "test prompt",
            max_tokens=10,
            enable_background_monitoring=False
        )
        
        # TTFT should be 100ms (0.1s)
        assert metrics.ttft_ms == 100.0, \
            f"Expected TTFT of 100ms, got {metrics.ttft_ms}ms"
    
    @patch('llm_benchmark.metrics.collector.psutil.Process')
    @patch('llm_benchmark.metrics.collector.time.perf_counter')
    def test_ttft_with_different_timings(self, mock_time, mock_process_class):
        """Test TTFT calculation with various timings."""
        test_cases = [
            ([0.0, 0.05, 0.1, 0.1], 50.0),   # 50ms TTFT
            ([0.0, 0.2, 0.4, 0.4], 200.0),   # 200ms TTFT
            ([0.0, 0.5, 1.0, 1.0], 500.0),   # 500ms TTFT
        ]
        
        for time_sequence, expected_ttft_ms in test_cases:
            mock_time.side_effect = time_sequence
            
            # Mock process memory
            mock_process = Mock()
            mock_memory_info = Mock()
            mock_memory_info.rss = 1024 * 1024 * 100
            mock_process.memory_info.return_value = mock_memory_info
            mock_process_class.return_value = mock_process
            
            # Mock LLM streaming
            mock_llm = Mock()
            mock_llm.return_value = [
                {'usage': {'prompt_tokens': 10}},
                {},
            ]
            
            hw_info = HardwareInfo(
                os_type="linux_x86",
                cpu_model="Test CPU",
                cpu_cores=8,
                cpu_features=[],
                total_ram_gb=16.0,
                available_ram_gb=8.0,
                has_gpu=False,
                gpu_model=None,
                gpu_memory_gb=None,
                gpu_compute_capability=None,
                has_thermal_sensors=False,
                has_power_sensors=False
            )
            
            collector = MetricsCollector(hw_info)
            metrics = collector._collect_inference_metrics_impl(
                mock_llm,
                "test prompt",
                max_tokens=10,
                enable_background_monitoring=False
            )
            
            assert metrics.ttft_ms == expected_ttft_ms, \
                f"Expected TTFT of {expected_ttft_ms}ms, got {metrics.ttft_ms}ms"


class TestThroughputCalculations:
    """Test throughput calculations with various token counts."""
    
    @patch('llm_benchmark.metrics.collector.psutil.Process')
    @patch('llm_benchmark.metrics.collector.time.perf_counter')
    def test_prefill_throughput_calculation(self, mock_time, mock_process_class):
        """Test prefill throughput calculation."""
        # Mock time: TTFT = 0.1s (100ms)
        time_sequence = [
            0.0,    # Start
            0.1,    # First token (TTFT)
            0.15,   # Second token
            0.15    # End
        ]
        mock_time.side_effect = time_sequence
        
        # Mock process memory
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 100
        mock_process.memory_info.return_value = mock_memory_info
        mock_process_class.return_value = mock_process
        
        # Mock LLM with 50 prompt tokens
        mock_llm = Mock()
        mock_llm.return_value = [
            {'usage': {'prompt_tokens': 50}},  # 50 prompt tokens
            {},
        ]
        
        hw_info = HardwareInfo(
            os_type="linux_x86",
            cpu_model="Test CPU",
            cpu_cores=8,
            cpu_features=[],
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            has_gpu=False,
            gpu_model=None,
            gpu_memory_gb=None,
            gpu_compute_capability=None,
            has_thermal_sensors=False,
            has_power_sensors=False
        )
        
        collector = MetricsCollector(hw_info)
        metrics = collector._collect_inference_metrics_impl(
            mock_llm,
            "test prompt",
            max_tokens=10,
            enable_background_monitoring=False
        )
        
        # Prefill TPS = 50 tokens / 0.1s = 500 t/s
        assert metrics.prefill_tps == 500.0, \
            f"Expected prefill TPS of 500, got {metrics.prefill_tps}"
    
    @patch('llm_benchmark.metrics.collector.psutil.Process')
    @patch('llm_benchmark.metrics.collector.time.perf_counter')
    def test_decode_throughput_calculation(self, mock_time, mock_process_class):
        """Test decode throughput calculation."""
        # Mock time: TTFT = 0.1s, total = 0.6s, decode = 0.5s
        time_sequence = [
            0.0,    # Start
            0.1,    # First token (TTFT)
            0.2,    # Second token
            0.3,    # Third token
            0.4,    # Fourth token
            0.5,    # Fifth token
            0.6,    # Sixth token
            0.6     # End
        ]
        mock_time.side_effect = time_sequence
        
        # Mock process memory
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 100
        mock_process.memory_info.return_value = mock_memory_info
        mock_process_class.return_value = mock_process
        
        # Mock LLM with 6 output tokens
        mock_llm = Mock()
        mock_llm.return_value = [
            {'usage': {'prompt_tokens': 10}},
            {},  # Token 2
            {},  # Token 3
            {},  # Token 4
            {},  # Token 5
            {},  # Token 6
        ]
        
        hw_info = HardwareInfo(
            os_type="linux_x86",
            cpu_model="Test CPU",
            cpu_cores=8,
            cpu_features=[],
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            has_gpu=False,
            gpu_model=None,
            gpu_memory_gb=None,
            gpu_compute_capability=None,
            has_thermal_sensors=False,
            has_power_sensors=False
        )
        
        collector = MetricsCollector(hw_info)
        metrics = collector._collect_inference_metrics_impl(
            mock_llm,
            "test prompt",
            max_tokens=10,
            enable_background_monitoring=False
        )
        
        # Decode TPS = (6 - 1) tokens / 0.5s = 10 t/s
        assert metrics.decode_tps == 10.0, \
            f"Expected decode TPS of 10, got {metrics.decode_tps}"
    
    @patch('llm_benchmark.metrics.collector.psutil.Process')
    @patch('llm_benchmark.metrics.collector.time.perf_counter')
    def test_throughput_with_various_token_counts(self, mock_time, mock_process_class):
        """Test throughput calculations with various token counts."""
        test_cases = [
            # (prompt_tokens, output_tokens, ttft_s, total_s, expected_prefill_tps, expected_decode_tps)
            (100, 10, 0.2, 1.0, 500.0, 11.25),  # 100/0.2=500, 9/0.8=11.25
            (50, 5, 0.1, 0.5, 500.0, 10.0),     # 50/0.1=500, 4/0.4=10
            (200, 20, 0.4, 2.0, 500.0, 11.88),  # 200/0.4=500, 19/1.6=11.875
        ]
        
        for prompt_tokens, output_tokens, ttft_s, total_s, expected_prefill, expected_decode in test_cases:
            # Build time sequence
            time_sequence = [0.0, ttft_s]  # Start and TTFT
            decode_duration = total_s - ttft_s
            time_per_token = decode_duration / (output_tokens - 1) if output_tokens > 1 else 0
            
            for i in range(1, output_tokens):
                time_sequence.append(ttft_s + i * time_per_token)
            time_sequence.append(total_s)  # End time
            
            mock_time.side_effect = time_sequence
            
            # Mock process memory
            mock_process = Mock()
            mock_memory_info = Mock()
            mock_memory_info.rss = 1024 * 1024 * 100
            mock_process.memory_info.return_value = mock_memory_info
            mock_process_class.return_value = mock_process
            
            # Mock LLM
            mock_llm = Mock()
            chunks = [{'usage': {'prompt_tokens': prompt_tokens}}]
            chunks.extend([{}] * (output_tokens - 1))
            mock_llm.return_value = chunks
            
            hw_info = HardwareInfo(
                os_type="linux_x86",
                cpu_model="Test CPU",
                cpu_cores=8,
                cpu_features=[],
                total_ram_gb=16.0,
                available_ram_gb=8.0,
                has_gpu=False,
                gpu_model=None,
                gpu_memory_gb=None,
                gpu_compute_capability=None,
                has_thermal_sensors=False,
                has_power_sensors=False
            )
            
            collector = MetricsCollector(hw_info)
            metrics = collector._collect_inference_metrics_impl(
                mock_llm,
                "test prompt",
                max_tokens=output_tokens,
                enable_background_monitoring=False
            )
            
            assert abs(metrics.prefill_tps - expected_prefill) < 0.1, \
                f"Expected prefill TPS ~{expected_prefill}, got {metrics.prefill_tps}"
            assert abs(metrics.decode_tps - expected_decode) < 0.1, \
                f"Expected decode TPS ~{expected_decode}, got {metrics.decode_tps}"


class TestEmptyPromptHandling:
    """Test error handling for empty prompts."""
    
    @patch('llm_benchmark.metrics.collector.psutil.Process')
    def test_empty_prompt_raises_error(self, mock_process_class):
        """Test that empty prompt raises ValueError."""
        mock_process = Mock()
        mock_process_class.return_value = mock_process
        
        hw_info = HardwareInfo(
            os_type="linux_x86",
            cpu_model="Test CPU",
            cpu_cores=8,
            cpu_features=[],
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            has_gpu=False,
            gpu_model=None,
            gpu_memory_gb=None,
            gpu_compute_capability=None,
            has_thermal_sensors=False,
            has_power_sensors=False
        )
        
        collector = MetricsCollector(hw_info)
        mock_llm = Mock()
        
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            collector._collect_inference_metrics_impl(
                mock_llm,
                "",  # Empty prompt
                max_tokens=10,
                enable_background_monitoring=False
            )
