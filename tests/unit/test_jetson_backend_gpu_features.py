"""
Unit tests for JetsonBackend GPU-specific features.

Tests GPU memory exhaustion handling, automatic fallback, and thermal/power monitoring.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from llm_benchmark.hardware.hal import JetsonBackend
from llm_benchmark.models import HardwareInfo


@pytest.fixture
def jetson_hw_info():
    """Create mock Jetson hardware info with GPU."""
    return HardwareInfo(
        os_type="jetson_xavier_nx",
        cpu_model="ARM Cortex-A78AE",
        cpu_cores=8,
        cpu_features=[],
        total_ram_gb=16.0,
        available_ram_gb=12.0,
        has_gpu=True,
        gpu_model="NVIDIA Xavier NX GPU",
        gpu_memory_gb=8.0,
        gpu_compute_capability="7.2",
        has_thermal_sensors=True,
        has_power_sensors=True
    )


@pytest.fixture
def jetson_backend(jetson_hw_info):
    """Create JetsonBackend instance."""
    return JetsonBackend(jetson_hw_info)


class TestGPULayerCalculation:
    """Test GPU layer calculation logic."""
    
    def test_calculate_gpu_layers_with_gpu(self, jetson_backend):
        """Test GPU layer calculation with available GPU memory."""
        # 8GB * 0.8 * 1024 / 100 = ~65 layers
        layers = jetson_backend._calculate_gpu_layers()
        assert 60 <= layers <= 70
    
    def test_calculate_gpu_layers_no_gpu(self):
        """Test GPU layer calculation falls back to 0 when no GPU."""
        hw_info = HardwareInfo(
            os_type="jetson_xavier_nx",
            cpu_model="ARM Cortex-A78AE",
            cpu_cores=8,
            cpu_features=[],
            total_ram_gb=16.0,
            available_ram_gb=12.0,
            has_gpu=False,
            has_thermal_sensors=False,
            has_power_sensors=False
        )
        backend = JetsonBackend(hw_info)
        layers = backend._calculate_gpu_layers()
        assert layers == 0
    
    def test_calculate_gpu_layers_none_memory(self):
        """Test GPU layer calculation with None GPU memory."""
        hw_info = HardwareInfo(
            os_type="jetson_xavier_nx",
            cpu_model="ARM Cortex-A78AE",
            cpu_cores=8,
            cpu_features=[],
            total_ram_gb=16.0,
            available_ram_gb=12.0,
            has_gpu=True,
            gpu_memory_gb=None,
            has_thermal_sensors=False,
            has_power_sensors=False
        )
        backend = JetsonBackend(hw_info)
        layers = backend._calculate_gpu_layers()
        assert layers == 0


class TestGPUMemoryExhaustionHandling:
    """Test GPU memory exhaustion handling with layer reduction fallback."""
    
    @patch('llama_cpp.Llama')
    def test_load_model_success_first_attempt(self, mock_llama, jetson_backend):
        """Test successful model loading on first attempt."""
        mock_instance = Mock()
        mock_llama.return_value = mock_instance
        
        result = jetson_backend.load_model_with_gpu_fallback("test_model.gguf")
        
        assert result == mock_instance
        assert mock_llama.call_count == 1
        # Verify GPU layers were used
        call_kwargs = mock_llama.call_args[1]
        assert call_kwargs["n_gpu_layers"] > 0
    
    @patch('llama_cpp.Llama')
    @patch('gc.collect')
    def test_load_model_oom_fallback(self, mock_gc, mock_llama, jetson_backend):
        """Test GPU OOM triggers layer reduction fallback."""
        # First call: OOM error
        # Second call: Success with reduced layers
        mock_instance = Mock()
        mock_llama.side_effect = [
            RuntimeError("CUDA out of memory"),
            mock_instance
        ]
        
        result = jetson_backend.load_model_with_gpu_fallback("test_model.gguf")
        
        assert result == mock_instance
        assert mock_llama.call_count == 2
        
        # Verify layer reduction
        first_call_layers = mock_llama.call_args_list[0][1]["n_gpu_layers"]
        second_call_layers = mock_llama.call_args_list[1][1]["n_gpu_layers"]
        assert second_call_layers == max(0, first_call_layers - 10)
        
        # Verify garbage collection was called
        assert mock_gc.call_count >= 1
    
    @patch('llama_cpp.Llama')
    @patch('gc.collect')
    def test_load_model_multiple_oom_fallbacks(self, mock_gc, mock_llama, jetson_backend):
        """Test multiple OOM errors trigger progressive layer reduction."""
        mock_instance = Mock()
        
        # Simulate multiple OOM errors before success
        mock_llama.side_effect = [
            RuntimeError("CUDA out of memory"),
            RuntimeError("out of memory"),
            RuntimeError("CUDA out of memory"),
            mock_instance
        ]
        
        result = jetson_backend.load_model_with_gpu_fallback("test_model.gguf")
        
        assert result == mock_instance
        assert mock_llama.call_count == 4
        
        # Verify progressive layer reduction
        layers = [call[1]["n_gpu_layers"] for call in mock_llama.call_args_list]
        for i in range(len(layers) - 1):
            assert layers[i+1] <= layers[i] - 10 or layers[i+1] == 0
    
    @patch('llama_cpp.Llama')
    def test_load_model_cpu_only_fallback(self, mock_llama, jetson_backend):
        """Test final fallback to CPU-only mode."""
        mock_instance = Mock()
        
        # Calculate how many OOM errors needed to reach CPU-only
        initial_layers = jetson_backend._calculate_gpu_layers()
        num_reductions = (initial_layers // 10) + 1
        
        # Simulate OOM errors until CPU-only, then success
        oom_errors = [RuntimeError("CUDA out of memory")] * num_reductions
        mock_llama.side_effect = oom_errors + [mock_instance]
        
        result = jetson_backend.load_model_with_gpu_fallback("test_model.gguf")
        
        assert result == mock_instance
        
        # Verify final call used CPU-only (0 GPU layers)
        final_call_layers = mock_llama.call_args_list[-1][1]["n_gpu_layers"]
        assert final_call_layers == 0
    
    @patch('llama_cpp.Llama')
    def test_load_model_cpu_only_failure(self, mock_llama, jetson_backend):
        """Test failure even in CPU-only mode raises error."""
        # All attempts fail, including CPU-only
        mock_llama.side_effect = RuntimeError("CUDA out of memory")
        
        with pytest.raises(RuntimeError, match="Failed to load model even with CPU-only mode"):
            jetson_backend.load_model_with_gpu_fallback("test_model.gguf")
    
    @patch('llama_cpp.Llama')
    def test_load_model_non_memory_error(self, mock_llama, jetson_backend):
        """Test non-memory errors are raised immediately without fallback."""
        mock_llama.side_effect = RuntimeError("Invalid model format")
        
        with pytest.raises(RuntimeError, match="Invalid model format"):
            jetson_backend.load_model_with_gpu_fallback("test_model.gguf")
        
        # Should only attempt once (no fallback for non-memory errors)
        assert mock_llama.call_count == 1
    
    @patch('llama_cpp.Llama')
    def test_load_model_with_custom_kwargs(self, mock_llama, jetson_backend):
        """Test custom kwargs are passed through to Llama."""
        mock_instance = Mock()
        mock_llama.return_value = mock_instance
        
        result = jetson_backend.load_model_with_gpu_fallback(
            "test_model.gguf",
            n_ctx=4096,
            n_batch=512
        )
        
        assert result == mock_instance
        call_kwargs = mock_llama.call_args[1]
        assert call_kwargs["n_ctx"] == 4096
        assert call_kwargs["n_batch"] == 512


class TestGPUTemperatureMonitoring:
    """Test GPU temperature monitoring."""
    
    def test_get_gpu_temperature_success(self, jetson_backend):
        """Test successful GPU temperature reading."""
        with patch.dict('sys.modules', {'pynvml': MagicMock()}):
            import sys
            mock_pynvml = sys.modules['pynvml']
            
            mock_handle = Mock()
            mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
            mock_pynvml.nvmlDeviceGetTemperature.return_value = 65
            mock_pynvml.NVML_TEMPERATURE_GPU = 0
            
            temp = jetson_backend.get_gpu_temperature()
            
            assert temp == 65.0
            mock_pynvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(0)
            mock_pynvml.nvmlDeviceGetTemperature.assert_called_once()
    
    def test_get_gpu_temperature_already_initialized(self, jetson_backend):
        """Test GPU temperature reading when NVML already initialized."""
        with patch.dict('sys.modules', {'pynvml': MagicMock()}):
            import sys
            mock_pynvml = sys.modules['pynvml']
            
            # Create a custom exception class for NVMLError
            class MockNVMLError(Exception):
                pass
            
            mock_pynvml.NVMLError = MockNVMLError
            mock_pynvml.nvmlInit.side_effect = MockNVMLError("Already initialized")
            mock_handle = Mock()
            mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
            mock_pynvml.nvmlDeviceGetTemperature.return_value = 70
            mock_pynvml.NVML_TEMPERATURE_GPU = 0
            
            temp = jetson_backend.get_gpu_temperature()
            
            assert temp == 70.0
    
    def test_get_gpu_temperature_no_gpu(self):
        """Test GPU temperature returns None when no GPU."""
        hw_info = HardwareInfo(
            os_type="jetson_xavier_nx",
            cpu_model="ARM Cortex-A78AE",
            cpu_cores=8,
            cpu_features=[],
            total_ram_gb=16.0,
            available_ram_gb=12.0,
            has_gpu=False,
            has_thermal_sensors=False,
            has_power_sensors=False
        )
        backend = JetsonBackend(hw_info)
        
        temp = backend.get_gpu_temperature()
        assert temp is None
    
    def test_get_gpu_temperature_error(self, jetson_backend):
        """Test GPU temperature returns None on error."""
        with patch.dict('sys.modules', {'pynvml': MagicMock()}):
            import sys
            mock_pynvml = sys.modules['pynvml']
            
            mock_pynvml.nvmlInit.side_effect = Exception("NVML error")
            temp = jetson_backend.get_gpu_temperature()
            assert temp is None


class TestGPUPowerMonitoring:
    """Test GPU power consumption monitoring."""
    
    def test_get_gpu_power_success(self, jetson_backend):
        """Test successful GPU power reading."""
        with patch.dict('sys.modules', {'pynvml': MagicMock()}):
            import sys
            mock_pynvml = sys.modules['pynvml']
            
            mock_handle = Mock()
            mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
            mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 15000  # 15W in milliwatts
            
            power = jetson_backend.get_gpu_power_consumption()
            
            assert power == 15.0
            mock_pynvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(0)
            mock_pynvml.nvmlDeviceGetPowerUsage.assert_called_once()
    
    def test_get_gpu_power_no_gpu(self):
        """Test GPU power returns None when no GPU."""
        hw_info = HardwareInfo(
            os_type="jetson_xavier_nx",
            cpu_model="ARM Cortex-A78AE",
            cpu_cores=8,
            cpu_features=[],
            total_ram_gb=16.0,
            available_ram_gb=12.0,
            has_gpu=False,
            has_thermal_sensors=False,
            has_power_sensors=False
        )
        backend = JetsonBackend(hw_info)
        
        power = backend.get_gpu_power_consumption()
        assert power is None
    
    def test_get_gpu_power_error(self, jetson_backend):
        """Test GPU power returns None on error."""
        with patch.dict('sys.modules', {'pynvml': MagicMock()}):
            import sys
            mock_pynvml = sys.modules['pynvml']
            
            mock_pynvml.nvmlInit.side_effect = Exception("NVML error")
            power = jetson_backend.get_gpu_power_consumption()
            assert power is None


class TestThermalStateChecking:
    """Test thermal state checking."""
    
    @patch.object(JetsonBackend, 'get_gpu_temperature')
    def test_check_thermal_state_normal(self, mock_get_temp, jetson_backend):
        """Test thermal state check when temperature is normal."""
        mock_get_temp.return_value = 60.0
        
        is_throttled, temp = jetson_backend.check_thermal_state(threshold_c=85.0)
        
        assert is_throttled is False
        assert temp == 60.0
    
    @patch.object(JetsonBackend, 'get_gpu_temperature')
    def test_check_thermal_state_throttled(self, mock_get_temp, jetson_backend):
        """Test thermal state check when temperature exceeds threshold."""
        mock_get_temp.return_value = 90.0
        
        is_throttled, temp = jetson_backend.check_thermal_state(threshold_c=85.0)
        
        assert is_throttled is True
        assert temp == 90.0
    
    @patch.object(JetsonBackend, 'get_gpu_temperature')
    def test_check_thermal_state_at_threshold(self, mock_get_temp, jetson_backend):
        """Test thermal state check when temperature equals threshold."""
        mock_get_temp.return_value = 85.0
        
        is_throttled, temp = jetson_backend.check_thermal_state(threshold_c=85.0)
        
        assert is_throttled is False
        assert temp == 85.0
    
    @patch.object(JetsonBackend, 'get_gpu_temperature')
    def test_check_thermal_state_no_temp(self, mock_get_temp, jetson_backend):
        """Test thermal state check when temperature unavailable."""
        mock_get_temp.return_value = None
        
        is_throttled, temp = jetson_backend.check_thermal_state(threshold_c=85.0)
        
        assert is_throttled is False
        assert temp is None
    
    @patch.object(JetsonBackend, 'get_gpu_temperature')
    def test_check_thermal_state_custom_threshold(self, mock_get_temp, jetson_backend):
        """Test thermal state check with custom threshold."""
        mock_get_temp.return_value = 75.0
        
        is_throttled, temp = jetson_backend.check_thermal_state(threshold_c=70.0)
        
        assert is_throttled is True
        assert temp == 75.0


class TestJetsonBackendConfiguration:
    """Test JetsonBackend configuration."""
    
    def test_get_llama_config(self, jetson_backend):
        """Test Jetson llama config has correct parameters."""
        config = jetson_backend.get_llama_config()
        
        assert "n_gpu_layers" in config
        assert "use_mlock" in config
        assert "n_threads" in config
        
        assert config["n_gpu_layers"] > 0  # Should use GPU
        assert config["use_mlock"] is False  # Limited RAM on Jetson
        assert config["n_threads"] == 4  # Leave cores for system
    
    def test_get_metrics_collector(self, jetson_backend):
        """Test Jetson backend returns metrics collector."""
        collector = jetson_backend.get_metrics_collector()
        
        assert collector is not None
        assert collector.hw_info == jetson_backend.hw_info
