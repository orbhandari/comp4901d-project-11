"""
Integration tests for AndroidBackend with backend selection.

Tests the complete integration of AndroidBackend with automatic backend
selection between llama-cli and llama-server based on configuration.
"""

import pytest
import tempfile
import json
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from llm_benchmark.hardware.hal import AndroidBackend, create_backend
from llm_benchmark.models import HardwareInfo
from llm_benchmark.android_config import AndroidConfig


class TestAndroidBackendIntegration:
    """Test AndroidBackend integration with configuration and backend selection."""
    
    @pytest.fixture
    def android_hw_info(self):
        """Create Android HardwareInfo for testing."""
        return HardwareInfo(
            os_type='android',
            cpu_model='Snapdragon 8 Gen 2',
            cpu_cores=8,
            cpu_features=['neon', 'fp16'],
            total_ram_gb=12.0,
            available_ram_gb=10.0,
            has_gpu=True,
            gpu_model='Adreno 740',
            gpu_memory_gb=0.0,
            has_thermal_sensors=True,
            has_power_sensors=False
        )
    
    def test_create_backend_returns_android_backend(self, android_hw_info):
        """Test that create_backend returns AndroidBackend for Android platform."""
        backend = create_backend(android_hw_info)
        
        assert isinstance(backend, AndroidBackend)
        assert backend.hw_info == android_hw_info
    
    def test_android_backend_loads_default_config(self, android_hw_info):
        """Test that AndroidBackend loads default configuration when no config file exists."""
        backend = AndroidBackend(android_hw_info)
        
        with patch('pathlib.Path.exists', return_value=False):
            config = backend._load_android_config()
            
            assert config.enable_ablation_studies is False
            assert config.use_llama_server_for_ablation is None
            assert config.llama_server_host == "127.0.0.1"
            assert config.llama_server_port == 8080
    
    def test_android_backend_loads_config_from_file(self, android_hw_info):
        """Test that AndroidBackend loads configuration from JSON file."""
        backend = AndroidBackend(android_hw_info)
        
        config_data = {
            "enable_ablation_studies": True,
            "use_llama_server_for_ablation": True,
            "llama_server_host": "192.168.1.100",
            "llama_server_port": 9090,
            "cache_mode": "none"
        }
        
        config_json = json.dumps(config_data)
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=config_json)):
            
            config = backend._load_android_config()
            
            assert config.enable_ablation_studies is True
            assert config.use_llama_server_for_ablation is True
            assert config.llama_server_host == "192.168.1.100"
            assert config.llama_server_port == 9090
    
    def test_backend_selection_with_explicit_config(self, android_hw_info):
        """Test backend selection with explicit configuration."""
        backend = AndroidBackend(android_hw_info)
        
        # Test explicit llama-server selection
        config_server = AndroidConfig(use_llama_server_for_ablation=True)
        with patch.object(backend, '_load_android_config', return_value=config_server), \
             patch.object(backend, '_detect_llama_server_availability', return_value=True):
            
            assert backend._should_use_llama_server(enable_ablation_studies=False) is True
        
        # Test explicit llama-cli selection
        config_cli = AndroidConfig(use_llama_server_for_ablation=False)
        with patch.object(backend, '_load_android_config', return_value=config_cli):
            assert backend._should_use_llama_server(enable_ablation_studies=True) is False
    
    def test_backend_selection_automatic_android_ablation(self, android_hw_info):
        """Test automatic backend selection for Android ablation studies."""
        backend = AndroidBackend(android_hw_info)
        
        config = AndroidConfig(enable_ablation_studies=True, use_llama_server_for_ablation=None)
        
        # Test with llama-server available
        with patch.object(backend, '_load_android_config', return_value=config), \
             patch.object(backend, '_detect_llama_server_availability', return_value=True):
            
            assert backend._should_use_llama_server(enable_ablation_studies=True) is True
        
        # Test with llama-server unavailable
        with patch.object(backend, '_load_android_config', return_value=config), \
             patch.object(backend, '_detect_llama_server_availability', return_value=False):
            
            assert backend._should_use_llama_server(enable_ablation_studies=True) is False
    
    def test_load_model_safe_integration(self, android_hw_info):
        """Test complete load_model_safe integration with backend selection."""
        backend = AndroidBackend(android_hw_info)
        
        # Mock successful llama-server loading
        mock_server = MagicMock()
        mock_server.__class__.__name__ = 'NativeLlamaServer'
        
        with patch.object(backend, '_should_use_llama_server', return_value=True), \
             patch.object(backend, '_load_model_with_llama_server', return_value=mock_server) as mock_load_server:
            
            result = backend.load_model_safe("/fake/model.gguf", enable_ablation_studies=True)
            
            mock_load_server.assert_called_once_with("/fake/model.gguf", enable_ablation_studies=True)
            assert result == mock_server
        
        # Mock successful llama-cli loading
        mock_cli = MagicMock()
        mock_cli.__class__.__name__ = 'NativeLlamaCpp'
        
        with patch.object(backend, '_should_use_llama_server', return_value=False), \
             patch.object(backend, '_load_model_with_llama_cli', return_value=mock_cli) as mock_load_cli:
            
            result = backend.load_model_safe("/fake/model.gguf", enable_ablation_studies=False)
            
            mock_load_cli.assert_called_once_with("/fake/model.gguf", enable_ablation_studies=False)
            assert result == mock_cli
    
    def test_llama_config_compatibility(self, android_hw_info):
        """Test that get_llama_config still works correctly."""
        backend = AndroidBackend(android_hw_info)
        
        config = backend.get_llama_config()
        
        # Verify Android-specific configuration
        assert config['n_gpu_layers'] == 0  # CPU-only
        assert config['use_mlock'] is False  # No memory locking on mobile
        assert config['n_threads'] == max(2, android_hw_info.cpu_cores - 2)  # Conservative thread count
        assert config['n_batch'] == 256  # Smaller batch size for memory efficiency
    
    def test_metrics_collector_compatibility(self, android_hw_info):
        """Test that get_metrics_collector still works correctly."""
        backend = AndroidBackend(android_hw_info)
        
        with patch('llm_benchmark.metrics.MetricsCollector') as mock_collector_class:
            mock_collector = MagicMock()
            mock_collector_class.return_value = mock_collector
            
            collector = backend.get_metrics_collector()
            
            mock_collector_class.assert_called_once_with(android_hw_info)
            assert collector == mock_collector


class TestAndroidBackendErrorHandling:
    """Test AndroidBackend error handling scenarios."""
    
    @pytest.fixture
    def android_backend(self):
        """Create AndroidBackend for testing."""
        hw_info = HardwareInfo(
            os_type='android',
            cpu_model='Test CPU',
            cpu_cores=4,
            cpu_features=[],
            total_ram_gb=4.0,
            available_ram_gb=3.0,
            has_gpu=False
        )
        return AndroidBackend(hw_info)
    
    def test_config_loading_error_fallback(self, android_backend):
        """Test that configuration loading errors fall back to defaults."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', side_effect=IOError("File read error")):
            
            config = android_backend._load_android_config()
            
            # Should fall back to default configuration
            assert config.enable_ablation_studies is False
            assert config.use_llama_server_for_ablation is None
    
    def test_binary_detection_error_handling(self, android_backend):
        """Test that binary detection handles errors gracefully."""
        with patch('pathlib.Path') as mock_path_class:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.is_file.return_value = True
            mock_path_class.return_value = mock_path
            
            with patch('os.access', side_effect=OSError("Permission denied")):
                result = android_backend._detect_llama_server_availability()
                
                # Should return False on error
                assert result is False
    
    def test_load_model_with_validation_failure(self, android_backend):
        """Test model loading with validation failures."""
        with patch('pathlib.Path') as mock_path_class, \
             patch.object(android_backend, '_validate_gguf_format', return_value=False):
            
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path_class.return_value = mock_path
            
            result = android_backend._load_model_with_llama_server("/fake/model.gguf")
            
            assert result is None
    
    def test_load_model_with_memory_check_failure(self, android_backend):
        """Test model loading with insufficient memory."""
        with patch('pathlib.Path') as mock_path_class, \
             patch.object(android_backend, '_validate_gguf_format', return_value=True), \
             patch.object(android_backend, '_check_available_memory', return_value=False):
            
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path_class.return_value = mock_path
            
            result = android_backend._load_model_with_llama_server("/fake/model.gguf")
            
            assert result is None