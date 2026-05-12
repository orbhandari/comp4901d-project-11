"""
Unit tests for AndroidConfig data model.

Tests configuration validation, default values, cache control logic,
and backend selection behavior.
"""

import pytest
from unittest.mock import patch
from llm_benchmark.android_config import (
    AndroidConfig,
    CacheMode,
    ABLATION_CACHE_CONFIG,
    create_default_android_config,
    validate_android_config,
)


class TestCacheMode:
    """Test CacheMode enum functionality."""
    
    def test_cache_mode_values(self):
        """Test that CacheMode enum has correct values."""
        assert CacheMode.NONE.value == "none"
        assert CacheMode.RAM_ONLY.value == "ram_only"
        assert CacheMode.DISK_ONLY.value == "disk_only"
        assert CacheMode.BOTH.value == "both"
    
    def test_cache_mode_from_string(self):
        """Test creating CacheMode from string values."""
        assert CacheMode("none") == CacheMode.NONE
        assert CacheMode("ram_only") == CacheMode.RAM_ONLY
        assert CacheMode("disk_only") == CacheMode.DISK_ONLY
        assert CacheMode("both") == CacheMode.BOTH
    
    def test_invalid_cache_mode_string(self):
        """Test that invalid cache mode strings raise ValueError."""
        with pytest.raises(ValueError):
            CacheMode("invalid_mode")


class TestAndroidConfigInitialization:
    """Test AndroidConfig initialization and validation."""
    
    def test_default_initialization(self):
        """Test AndroidConfig with default values."""
        config = AndroidConfig()
        
        assert config.enable_ablation_studies is False
        assert config.use_llama_server_for_ablation is None
        assert config.llama_server_host == "127.0.0.1"
        assert config.llama_server_port == 8080
        assert config.llama_server_path is None
        assert config.llama_server_timeout == 600
        assert config.cache_mode == CacheMode.BOTH
        assert config.enable_prompt_cache_by_default is False
    
    def test_custom_initialization(self):
        """Test AndroidConfig with custom values."""
        config = AndroidConfig(
            enable_ablation_studies=True,
            use_llama_server_for_ablation=True,
            llama_server_host="192.168.1.100",
            llama_server_port=9090,
            llama_server_path="/custom/path/llama-server",
            llama_server_timeout=1200,
            cache_mode=CacheMode.NONE,
            enable_prompt_cache_by_default=True,
        )
        
        assert config.enable_ablation_studies is True
        assert config.use_llama_server_for_ablation is True
        assert config.llama_server_host == "192.168.1.100"
        assert config.llama_server_port == 9090
        assert config.llama_server_path == "/custom/path/llama-server"
        assert config.llama_server_timeout == 1200
        assert config.cache_mode == CacheMode.NONE
        assert config.enable_prompt_cache_by_default is True
    
    def test_cache_mode_string_conversion(self):
        """Test that cache_mode string is converted to enum."""
        config = AndroidConfig(cache_mode="none")
        assert config.cache_mode == CacheMode.NONE
        
        config = AndroidConfig(cache_mode="ram_only")
        assert config.cache_mode == CacheMode.RAM_ONLY


class TestAndroidConfigValidation:
    """Test AndroidConfig validation logic."""
    
    def test_invalid_host_empty(self):
        """Test validation fails for empty host."""
        with pytest.raises(ValueError, match="llama_server_host cannot be empty"):
            AndroidConfig(llama_server_host="")
    
    def test_invalid_host_type(self):
        """Test validation fails for non-string host."""
        with pytest.raises(ValueError, match="llama_server_host must be a string"):
            AndroidConfig(llama_server_host=123)
    
    def test_invalid_port_type(self):
        """Test validation fails for non-integer port."""
        with pytest.raises(ValueError, match="llama_server_port must be an integer"):
            AndroidConfig(llama_server_port="8080")
    
    def test_invalid_port_range_low(self):
        """Test validation fails for port below valid range."""
        with pytest.raises(ValueError, match="llama_server_port must be between 1 and 65535"):
            AndroidConfig(llama_server_port=0)
    
    def test_invalid_port_range_high(self):
        """Test validation fails for port above valid range."""
        with pytest.raises(ValueError, match="llama_server_port must be between 1 and 65535"):
            AndroidConfig(llama_server_port=65536)
    
    def test_invalid_timeout_type(self):
        """Test validation fails for non-integer timeout."""
        with pytest.raises(ValueError, match="llama_server_timeout must be an integer"):
            AndroidConfig(llama_server_timeout="600")
    
    def test_invalid_timeout_negative(self):
        """Test validation fails for negative timeout."""
        with pytest.raises(ValueError, match="llama_server_timeout must be positive"):
            AndroidConfig(llama_server_timeout=-1)
    
    def test_invalid_timeout_zero(self):
        """Test validation fails for zero timeout."""
        with pytest.raises(ValueError, match="llama_server_timeout must be positive"):
            AndroidConfig(llama_server_timeout=0)
    
    def test_invalid_cache_mode_string(self):
        """Test validation fails for invalid cache mode string."""
        with pytest.raises(ValueError, match="Invalid cache_mode 'invalid'"):
            AndroidConfig(cache_mode="invalid")
    
    def test_invalid_cache_mode_type(self):
        """Test validation fails for invalid cache mode type."""
        with pytest.raises(ValueError, match="cache_mode must be a CacheMode enum or string"):
            AndroidConfig(cache_mode=123)
    
    def test_invalid_enable_ablation_studies_type(self):
        """Test validation fails for non-boolean enable_ablation_studies."""
        with pytest.raises(ValueError, match="enable_ablation_studies must be a boolean"):
            AndroidConfig(enable_ablation_studies="true")
    
    def test_invalid_use_llama_server_for_ablation_type(self):
        """Test validation fails for invalid use_llama_server_for_ablation type."""
        with pytest.raises(ValueError, match="use_llama_server_for_ablation must be a boolean or None"):
            AndroidConfig(use_llama_server_for_ablation="true")
    
    def test_invalid_llama_server_path_type(self):
        """Test validation fails for non-string llama_server_path."""
        with pytest.raises(ValueError, match="llama_server_path must be a string or None"):
            AndroidConfig(llama_server_path=123)
    
    def test_invalid_llama_server_path_empty(self):
        """Test validation fails for empty llama_server_path."""
        with pytest.raises(ValueError, match="llama_server_path cannot be empty string"):
            AndroidConfig(llama_server_path="   ")
    
    def test_invalid_enable_prompt_cache_by_default_type(self):
        """Test validation fails for non-boolean enable_prompt_cache_by_default."""
        with pytest.raises(ValueError, match="enable_prompt_cache_by_default must be a boolean"):
            AndroidConfig(enable_prompt_cache_by_default="false")


class TestAndroidConfigWarnings:
    """Test AndroidConfig warning generation."""
    
    @patch('llm_benchmark.android_config.logger')
    def test_common_port_warning(self, mock_logger):
        """Test warning for commonly used ports."""
        AndroidConfig(llama_server_port=80)
        mock_logger.warning.assert_called_once()
        assert "commonly used by system services" in mock_logger.warning.call_args[0][0]
    
    @patch('llm_benchmark.android_config.logger')
    def test_short_timeout_warning(self, mock_logger):
        """Test warning for very short timeout."""
        AndroidConfig(llama_server_timeout=30)
        mock_logger.warning.assert_called_once()
        assert "very short" in mock_logger.warning.call_args[0][0]
    
    @patch('llm_benchmark.android_config.logger')
    def test_long_timeout_warning(self, mock_logger):
        """Test warning for very long timeout."""
        AndroidConfig(llama_server_timeout=7200)
        mock_logger.warning.assert_called_once()
        assert "very long" in mock_logger.warning.call_args[0][0]


class TestAndroidConfigMethods:
    """Test AndroidConfig utility methods."""
    
    def test_get_server_url(self):
        """Test server URL generation."""
        config = AndroidConfig(llama_server_host="192.168.1.100", llama_server_port=9090)
        assert config.get_server_url() == "http://192.168.1.100:9090"
    
    def test_get_health_url(self):
        """Test health check URL generation."""
        config = AndroidConfig()
        assert config.get_health_url() == "http://127.0.0.1:8080/health"
    
    def test_get_completion_url(self):
        """Test completion endpoint URL generation."""
        config = AndroidConfig()
        assert config.get_completion_url() == "http://127.0.0.1:8080/completion"
    
    def test_should_use_llama_server_explicit_true(self):
        """Test explicit llama-server selection."""
        config = AndroidConfig(use_llama_server_for_ablation=True)
        assert config.should_use_llama_server("android") is True
        assert config.should_use_llama_server("linux") is True
    
    def test_should_use_llama_server_explicit_false(self):
        """Test explicit llama-cli selection."""
        config = AndroidConfig(use_llama_server_for_ablation=False)
        assert config.should_use_llama_server("android") is False
        assert config.should_use_llama_server("linux") is False
    
    def test_should_use_llama_server_auto_android_ablation(self):
        """Test automatic llama-server selection for Android ablation."""
        config = AndroidConfig(enable_ablation_studies=True)
        assert config.should_use_llama_server("android") is True
    
    def test_should_use_llama_server_auto_android_no_ablation(self):
        """Test automatic llama-cli selection for Android without ablation."""
        config = AndroidConfig(enable_ablation_studies=False)
        assert config.should_use_llama_server("android") is False
    
    def test_should_use_llama_server_auto_other_platform(self):
        """Test automatic llama-cli selection for non-Android platforms."""
        config = AndroidConfig(enable_ablation_studies=True)
        assert config.should_use_llama_server("linux") is False
        assert config.should_use_llama_server("macos") is False
    
    def test_get_cache_flags_none(self):
        """Test cache flags for NONE mode."""
        config = AndroidConfig(cache_mode=CacheMode.NONE)
        flags = config.get_cache_flags()
        assert flags == ["--cache-ram", "0", "--no-cache-prompt"]
    
    def test_get_cache_flags_ram_only(self):
        """Test cache flags for RAM_ONLY mode."""
        config = AndroidConfig(cache_mode=CacheMode.RAM_ONLY)
        flags = config.get_cache_flags()
        assert flags == ["--no-cache-prompt"]
    
    def test_get_cache_flags_disk_only(self):
        """Test cache flags for DISK_ONLY mode."""
        config = AndroidConfig(cache_mode=CacheMode.DISK_ONLY)
        flags = config.get_cache_flags()
        assert flags == ["--cache-ram", "0"]
    
    def test_get_cache_flags_both(self):
        """Test cache flags for BOTH mode."""
        config = AndroidConfig(cache_mode=CacheMode.BOTH)
        flags = config.get_cache_flags()
        assert flags == []
    
    def test_get_request_cache_setting_override(self):
        """Test request cache setting with override."""
        config = AndroidConfig(enable_prompt_cache_by_default=False)
        assert config.get_request_cache_setting(True) is True
        assert config.get_request_cache_setting(False) is False
    
    def test_get_request_cache_setting_default(self):
        """Test request cache setting using default."""
        config = AndroidConfig(enable_prompt_cache_by_default=True)
        assert config.get_request_cache_setting() is True
        
        config = AndroidConfig(enable_prompt_cache_by_default=False)
        assert config.get_request_cache_setting() is False


class TestAndroidConfigSerialization:
    """Test AndroidConfig serialization and deserialization."""
    
    def test_to_dict(self):
        """Test converting AndroidConfig to dictionary."""
        config = AndroidConfig(
            enable_ablation_studies=True,
            cache_mode=CacheMode.NONE,
            llama_server_port=9090,
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["enable_ablation_studies"] is True
        assert config_dict["cache_mode"] == "none"
        assert config_dict["llama_server_port"] == 9090
        assert config_dict["llama_server_host"] == "127.0.0.1"
    
    def test_from_dict(self):
        """Test creating AndroidConfig from dictionary."""
        config_dict = {
            "enable_ablation_studies": True,
            "cache_mode": "none",
            "llama_server_port": 9090,
            "llama_server_timeout": 1200,
        }
        
        config = AndroidConfig.from_dict(config_dict)
        
        assert config.enable_ablation_studies is True
        assert config.cache_mode == CacheMode.NONE
        assert config.llama_server_port == 9090
        assert config.llama_server_timeout == 1200
    
    def test_roundtrip_serialization(self):
        """Test that to_dict/from_dict roundtrip preserves data."""
        original = AndroidConfig(
            enable_ablation_studies=True,
            use_llama_server_for_ablation=False,
            llama_server_host="192.168.1.100",
            llama_server_port=9090,
            cache_mode=CacheMode.RAM_ONLY,
            enable_prompt_cache_by_default=True,
        )
        
        config_dict = original.to_dict()
        restored = AndroidConfig.from_dict(config_dict)
        
        assert restored.enable_ablation_studies == original.enable_ablation_studies
        assert restored.use_llama_server_for_ablation == original.use_llama_server_for_ablation
        assert restored.llama_server_host == original.llama_server_host
        assert restored.llama_server_port == original.llama_server_port
        assert restored.cache_mode == original.cache_mode
        assert restored.enable_prompt_cache_by_default == original.enable_prompt_cache_by_default


class TestAblationCacheConfig:
    """Test ABLATION_CACHE_CONFIG mapping."""
    
    def test_ablation_cache_config_structure(self):
        """Test that ABLATION_CACHE_CONFIG has expected structure."""
        expected_scenarios = ["control", "cold_cache", "warm_cache", "ram_only", "disk_only"]
        
        assert set(ABLATION_CACHE_CONFIG.keys()) == set(expected_scenarios)
        
        for scenario, config in ABLATION_CACHE_CONFIG.items():
            assert "cache_mode" in config
            assert "enable_prompt_cache" in config
            assert "description" in config
            assert isinstance(config["cache_mode"], CacheMode)
            assert isinstance(config["enable_prompt_cache"], bool)
            assert isinstance(config["description"], str)
    
    def test_control_scenario_config(self):
        """Test control scenario has no caching."""
        control = ABLATION_CACHE_CONFIG["control"]
        assert control["cache_mode"] == CacheMode.NONE
        assert control["enable_prompt_cache"] is False
    
    def test_cache_scenarios_config(self):
        """Test cache scenarios have appropriate settings."""
        cold_cache = ABLATION_CACHE_CONFIG["cold_cache"]
        assert cold_cache["cache_mode"] == CacheMode.BOTH
        assert cold_cache["enable_prompt_cache"] is True
        
        warm_cache = ABLATION_CACHE_CONFIG["warm_cache"]
        assert warm_cache["cache_mode"] == CacheMode.BOTH
        assert warm_cache["enable_prompt_cache"] is True
        
        ram_only = ABLATION_CACHE_CONFIG["ram_only"]
        assert ram_only["cache_mode"] == CacheMode.RAM_ONLY
        assert ram_only["enable_prompt_cache"] is False
        
        disk_only = ABLATION_CACHE_CONFIG["disk_only"]
        assert disk_only["cache_mode"] == CacheMode.DISK_ONLY
        assert disk_only["enable_prompt_cache"] is True


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_default_android_config(self):
        """Test creating default AndroidConfig."""
        config = create_default_android_config()
        
        assert isinstance(config, AndroidConfig)
        assert config.enable_ablation_studies is False
        assert config.use_llama_server_for_ablation is None
        assert config.llama_server_host == "127.0.0.1"
        assert config.llama_server_port == 8080
        assert config.cache_mode == CacheMode.BOTH
    
    def test_validate_android_config_no_warnings(self):
        """Test validation with no warnings."""
        config = AndroidConfig()
        warnings = validate_android_config(config)
        assert warnings == []
    
    def test_validate_android_config_timeout_warning(self):
        """Test validation generates timeout warning."""
        config = AndroidConfig(
            enable_ablation_studies=True,
            llama_server_timeout=120
        )
        warnings = validate_android_config(config)
        assert len(warnings) == 1
        assert "may be too short" in warnings[0]
    
    def test_validate_android_config_cache_mode_warning(self):
        """Test validation generates cache mode warning."""
        config = AndroidConfig(
            enable_ablation_studies=True,
            cache_mode=CacheMode.NONE
        )
        warnings = validate_android_config(config)
        assert len(warnings) == 1
        assert "dynamic cache control" in warnings[0]
    
    def test_validate_android_config_host_warning(self):
        """Test validation generates host warning."""
        config = AndroidConfig(llama_server_host="192.168.1.100")
        warnings = validate_android_config(config)
        assert len(warnings) == 1
        assert "consider using localhost" in warnings[0]
    
    def test_validate_android_config_multiple_warnings(self):
        """Test validation generates multiple warnings."""
        config = AndroidConfig(
            enable_ablation_studies=True,
            llama_server_timeout=120,
            cache_mode=CacheMode.NONE,
            llama_server_host="192.168.1.100"
        )
        warnings = validate_android_config(config)
        assert len(warnings) == 3