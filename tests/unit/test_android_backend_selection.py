"""
Unit tests for AndroidBackend backend selection logic.

Tests the automatic backend selection between llama-cli and llama-server
based on configuration and ablation study requirements.
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from llm_benchmark.hardware.hal import AndroidBackend
from llm_benchmark.models import HardwareInfo
from llm_benchmark.android_config import AndroidConfig, create_default_android_config


class TestAndroidBackendSelection:
    """Test AndroidBackend backend selection logic."""
    
    @pytest.fixture
    def android_hw_info(self):
        """Create mock HardwareInfo for Android."""
        return HardwareInfo(
            os_type='android',
            cpu_model='Snapdragon 8 Gen 2',
            cpu_cores=8,
            cpu_features=['neon'],
            total_ram_gb=8.0,
            available_ram_gb=6.0,
            has_gpu=True,
            gpu_model='Adreno 740',
            gpu_memory_gb=0.0,
            has_thermal_sensors=True,
            has_power_sensors=False
        )
    
    @pytest.fixture
    def android_backend(self, android_hw_info):
        """Create AndroidBackend instance."""
        return AndroidBackend(android_hw_info)
    
    def test_load_android_config_default(self, android_backend):
        """Test loading default AndroidConfig when no config file exists."""
        with patch('pathlib.Path.exists', return_value=False):
            config = android_backend._load_android_config()
            
            assert config.enable_ablation_studies is False
            assert config.use_llama_server_for_ablation is None
            assert config.llama_server_host == "127.0.0.1"
            assert config.llama_server_port == 8080
    
    def test_load_android_config_from_file(self, android_backend):
        """Test loading AndroidConfig from JSON file."""
        config_data = {
            "enable_ablation_studies": True,
            "use_llama_server_for_ablation": True,
            "llama_server_host": "192.168.1.100",
            "llama_server_port": 9090
        }
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data='{"enable_ablation_studies": true, "use_llama_server_for_ablation": true, "llama_server_host": "192.168.1.100", "llama_server_port": 9090}')):
            
            config = android_backend._load_android_config()
            
            assert config.enable_ablation_studies is True
            assert config.use_llama_server_for_ablation is True
            assert config.llama_server_host == "192.168.1.100"
            assert config.llama_server_port == 9090
    
    def test_detect_llama_server_availability_found(self, android_backend):
        """Test llama-server binary detection when available."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.is_file.return_value = True
        
        with patch('pathlib.Path') as mock_path_class, \
             patch('os.access', return_value=True):
            mock_path_class.return_value = mock_path
            
            result = android_backend._detect_llama_server_availability()
            assert result is True
    
    def test_detect_llama_server_availability_not_found(self, android_backend):
        """Test llama-server binary detection when not available."""
        with patch('pathlib.Path') as mock_path_class:
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            mock_path_class.return_value = mock_path
            
            result = android_backend._detect_llama_server_availability()
            assert result is False
    
    def test_should_use_llama_server_explicit_true(self, android_backend):
        """Test explicit llama-server configuration override (True)."""
        config = AndroidConfig(use_llama_server_for_ablation=True)
        
        with patch.object(android_backend, '_load_android_config', return_value=config), \
             patch.object(android_backend, '_detect_llama_server_availability', return_value=True):
            
            result = android_backend._should_use_llama_server(enable_ablation_studies=False)
            assert result is True
    
    def test_should_use_llama_server_explicit_false(self, android_backend):
        """Test explicit llama-server configuration override (False)."""
        config = AndroidConfig(use_llama_server_for_ablation=False)
        
        with patch.object(android_backend, '_load_android_config', return_value=config):
            result = android_backend._should_use_llama_server(enable_ablation_studies=True)
            assert result is False
    
    def test_should_use_llama_server_explicit_true_but_unavailable(self, android_backend):
        """Test explicit llama-server configuration when binary is unavailable."""
        config = AndroidConfig(use_llama_server_for_ablation=True)
        
        with patch.object(android_backend, '_load_android_config', return_value=config), \
             patch.object(android_backend, '_detect_llama_server_availability', return_value=False):
            
            result = android_backend._should_use_llama_server(enable_ablation_studies=False)
            assert result is False
    
    def test_should_use_llama_server_auto_android_ablation_available(self, android_backend):
        """Test automatic selection for Android ablation studies when llama-server is available."""
        config = AndroidConfig(enable_ablation_studies=True, use_llama_server_for_ablation=None)
        
        with patch.object(android_backend, '_load_android_config', return_value=config), \
             patch.object(android_backend, '_detect_llama_server_availability', return_value=True):
            
            result = android_backend._should_use_llama_server(enable_ablation_studies=True)
            assert result is True
    
    def test_should_use_llama_server_auto_android_ablation_unavailable(self, android_backend):
        """Test automatic selection for Android ablation studies when llama-server is unavailable."""
        config = AndroidConfig(enable_ablation_studies=True, use_llama_server_for_ablation=None)
        
        with patch.object(android_backend, '_load_android_config', return_value=config), \
             patch.object(android_backend, '_detect_llama_server_availability', return_value=False):
            
            result = android_backend._should_use_llama_server(enable_ablation_studies=True)
            assert result is False
    
    def test_should_use_llama_server_auto_no_ablation(self, android_backend):
        """Test automatic selection when ablation studies are disabled."""
        config = AndroidConfig(enable_ablation_studies=False, use_llama_server_for_ablation=None)
        
        with patch.object(android_backend, '_load_android_config', return_value=config):
            result = android_backend._should_use_llama_server(enable_ablation_studies=False)
            assert result is False
    
    def test_load_model_safe_uses_llama_server(self, android_backend):
        """Test that load_model_safe uses llama-server when selected."""
        with patch.object(android_backend, '_should_use_llama_server', return_value=True), \
             patch.object(android_backend, '_load_model_with_llama_server', return_value=MagicMock()) as mock_load_server:
            
            result = android_backend.load_model_safe("/fake/model.gguf", enable_ablation_studies=True)
            
            mock_load_server.assert_called_once_with("/fake/model.gguf", enable_ablation_studies=True)
            assert result is not None
    
    def test_load_model_safe_uses_llama_cli(self, android_backend):
        """Test that load_model_safe uses llama-cli when selected."""
        with patch.object(android_backend, '_should_use_llama_server', return_value=False), \
             patch.object(android_backend, '_load_model_with_llama_cli', return_value=MagicMock()) as mock_load_cli:
            
            result = android_backend.load_model_safe("/fake/model.gguf", enable_ablation_studies=False)
            
            mock_load_cli.assert_called_once_with("/fake/model.gguf", enable_ablation_studies=False)
            assert result is not None
    
    def test_load_model_with_llama_server_success(self, android_backend):
        """Test successful model loading with llama-server."""
        mock_server = MagicMock()
        
        with patch('pathlib.Path') as mock_path_class, \
             patch.object(android_backend, '_validate_gguf_format', return_value=True), \
             patch.object(android_backend, '_check_available_memory', return_value=True), \
             patch.object(android_backend, 'get_llama_config', return_value={'n_ctx': 2048}), \
             patch.object(android_backend, '_load_android_config') as mock_config, \
             patch('llm_benchmark.inference.native_llama_server.NativeLlamaServer', return_value=mock_server):
            
            # Mock path exists
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path_class.return_value = mock_path
            
            # Mock config
            config = create_default_android_config()
            mock_config.return_value = config
            
            result = android_backend._load_model_with_llama_server("/fake/model.gguf")
            
            assert result == mock_server
            assert android_backend._last_loaded_model == mock_server
    
    def test_load_model_with_llama_server_binary_not_found(self, android_backend):
        """Test model loading when llama-server binary is not found."""
        with patch('pathlib.Path') as mock_path_class:
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            mock_path_class.return_value = mock_path
            
            result = android_backend._load_model_with_llama_server("/fake/model.gguf")
            
            assert result is None
    
    def test_load_model_with_llama_server_fallback_to_cli(self, android_backend):
        """Test fallback to llama-cli when llama-server fails."""
        with patch('pathlib.Path') as mock_path_class, \
             patch.object(android_backend, '_validate_gguf_format', return_value=True), \
             patch.object(android_backend, '_check_available_memory', return_value=True), \
             patch.object(android_backend, 'get_llama_config', return_value={'n_ctx': 2048}), \
             patch.object(android_backend, '_load_android_config') as mock_config, \
             patch('llm_benchmark.inference.native_llama_server.NativeLlamaServer', side_effect=Exception("Server failed")), \
             patch.object(android_backend, '_load_model_with_llama_cli', return_value=MagicMock()) as mock_load_cli:
            
            # Mock path exists
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path_class.return_value = mock_path
            
            # Mock config
            config = create_default_android_config()
            mock_config.return_value = config
            
            result = android_backend._load_model_with_llama_server("/fake/model.gguf")
            
            mock_load_cli.assert_called_once()
            assert result is not None


class TestAndroidBackendLogging:
    """Test AndroidBackend logging functionality."""
    
    @pytest.fixture
    def android_backend(self):
        """Create AndroidBackend instance."""
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
    
    def test_log_llama_server_setup_instructions(self, android_backend, caplog):
        """Test that llama-server setup instructions are logged."""
        import logging
        caplog.set_level(logging.INFO)
        
        android_backend._log_llama_server_setup_instructions()
        
        assert "LLAMA-SERVER SETUP REQUIRED" in caplog.text
        assert "cmake --build build --config Release --target llama-server" in caplog.text
    
    def test_log_ablation_limitation_warning(self, android_backend, caplog):
        """Test that ablation limitation warning is logged."""
        import logging
        caplog.set_level(logging.WARNING)
        
        android_backend._log_ablation_limitation_warning()
        
        assert "ABLATION STUDY LIMITATION" in caplog.text
        assert "llama-cli cannot disable KV cache" in caplog.text
        assert "Cannot disable RAM-based KV cache" in caplog.text