"""
Integration tests for AndroidConfig usage in the benchmark framework.

Tests how AndroidConfig integrates with the existing configuration system
and demonstrates practical usage scenarios.
"""

import json
import tempfile
from pathlib import Path
import pytest

from llm_benchmark.android_config import (
    AndroidConfig,
    CacheMode,
    create_default_android_config,
    ABLATION_CACHE_CONFIG,
)


class TestAndroidConfigIntegration:
    """Test AndroidConfig integration with the benchmark framework."""
    
    def test_android_config_json_serialization(self):
        """Test that AndroidConfig can be serialized to/from JSON."""
        config = AndroidConfig(
            enable_ablation_studies=True,
            use_llama_server_for_ablation=True,
            llama_server_port=9090,
            cache_mode=CacheMode.NONE,
            llama_server_timeout=1200,
        )
        
        # Serialize to JSON
        config_dict = config.to_dict()
        json_str = json.dumps(config_dict, indent=2)
        
        # Deserialize from JSON
        loaded_dict = json.loads(json_str)
        restored_config = AndroidConfig.from_dict(loaded_dict)
        
        # Verify roundtrip preservation
        assert restored_config.enable_ablation_studies == config.enable_ablation_studies
        assert restored_config.use_llama_server_for_ablation == config.use_llama_server_for_ablation
        assert restored_config.llama_server_port == config.llama_server_port
        assert restored_config.cache_mode == config.cache_mode
        assert restored_config.llama_server_timeout == config.llama_server_timeout
    
    def test_android_config_file_persistence(self):
        """Test saving and loading AndroidConfig from file."""
        config = create_default_android_config()
        config.enable_ablation_studies = True
        config.llama_server_port = 8081
        config.cache_mode = CacheMode.RAM_ONLY
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config.to_dict(), f, indent=2)
            config_file = f.name
        
        try:
            # Load from file
            with open(config_file, 'r') as f:
                loaded_dict = json.load(f)
            
            restored_config = AndroidConfig.from_dict(loaded_dict)
            
            # Verify configuration was preserved
            assert restored_config.enable_ablation_studies is True
            assert restored_config.llama_server_port == 8081
            assert restored_config.cache_mode == CacheMode.RAM_ONLY
            
        finally:
            Path(config_file).unlink()
    
    def test_backend_selection_scenarios(self):
        """Test various backend selection scenarios."""
        # Scenario 1: Explicit llama-server selection
        config1 = AndroidConfig(use_llama_server_for_ablation=True)
        assert config1.should_use_llama_server("android") is True
        assert config1.should_use_llama_server("linux") is True
        
        # Scenario 2: Explicit llama-cli selection
        config2 = AndroidConfig(use_llama_server_for_ablation=False)
        assert config2.should_use_llama_server("android") is False
        assert config2.should_use_llama_server("linux") is False
        
        # Scenario 3: Automatic selection - Android with ablation
        config3 = AndroidConfig(enable_ablation_studies=True)
        assert config3.should_use_llama_server("android") is True
        assert config3.should_use_llama_server("linux") is False
        
        # Scenario 4: Automatic selection - Android without ablation
        config4 = AndroidConfig(enable_ablation_studies=False)
        assert config4.should_use_llama_server("android") is False
        assert config4.should_use_llama_server("linux") is False
    
    def test_ablation_scenario_configuration(self):
        """Test configuring AndroidConfig for different ablation scenarios."""
        base_config = AndroidConfig(
            enable_ablation_studies=True,
            llama_server_timeout=900,
        )
        
        # Test each ablation scenario
        for scenario_name, scenario_config in ABLATION_CACHE_CONFIG.items():
            # Create config for this scenario
            config = AndroidConfig(
                enable_ablation_studies=base_config.enable_ablation_studies,
                llama_server_timeout=base_config.llama_server_timeout,
                cache_mode=scenario_config["cache_mode"],
                enable_prompt_cache_by_default=scenario_config["enable_prompt_cache"],
            )
            
            # Verify cache flags are correct
            expected_flags = []
            if scenario_config["cache_mode"] == CacheMode.NONE:
                expected_flags = ["--cache-ram", "0", "--no-cache-prompt"]
            elif scenario_config["cache_mode"] == CacheMode.RAM_ONLY:
                expected_flags = ["--no-cache-prompt"]
            elif scenario_config["cache_mode"] == CacheMode.DISK_ONLY:
                expected_flags = ["--cache-ram", "0"]
            # CacheMode.BOTH has no flags
            
            assert config.get_cache_flags() == expected_flags
            
            # Verify request cache setting
            expected_cache_setting = scenario_config["enable_prompt_cache"]
            assert config.get_request_cache_setting() == expected_cache_setting
    
    def test_url_generation_for_different_configurations(self):
        """Test URL generation for various server configurations."""
        # Default configuration
        config1 = AndroidConfig()
        assert config1.get_server_url() == "http://127.0.0.1:8080"
        assert config1.get_health_url() == "http://127.0.0.1:8080/health"
        assert config1.get_completion_url() == "http://127.0.0.1:8080/completion"
        
        # Custom host and port
        config2 = AndroidConfig(
            llama_server_host="192.168.1.100",
            llama_server_port=9090
        )
        assert config2.get_server_url() == "http://192.168.1.100:9090"
        assert config2.get_health_url() == "http://192.168.1.100:9090/health"
        assert config2.get_completion_url() == "http://192.168.1.100:9090/completion"
    
    def test_configuration_validation_in_practice(self):
        """Test configuration validation with realistic scenarios."""
        # Valid configuration should not raise exceptions
        valid_config = AndroidConfig(
            enable_ablation_studies=True,
            llama_server_host="127.0.0.1",
            llama_server_port=8080,
            llama_server_timeout=600,
            cache_mode=CacheMode.BOTH,
        )
        
        # Should not raise any exceptions
        assert valid_config.enable_ablation_studies is True
        
        # Invalid configurations should raise appropriate exceptions
        with pytest.raises(ValueError):
            AndroidConfig(llama_server_port=0)  # Invalid port
        
        with pytest.raises(ValueError):
            AndroidConfig(llama_server_timeout=-1)  # Invalid timeout
        
        with pytest.raises(ValueError):
            AndroidConfig(cache_mode="invalid_mode")  # Invalid cache mode
    
    def test_practical_usage_example(self):
        """Test a practical usage example for Android ablation studies."""
        # Create configuration for Android ablation studies
        config = AndroidConfig(
            enable_ablation_studies=True,
            llama_server_host="127.0.0.1",
            llama_server_port=8080,
            llama_server_timeout=900,  # 15 minutes for mobile inference
            cache_mode=CacheMode.BOTH,  # Allow dynamic cache control
            enable_prompt_cache_by_default=False,  # Conservative default
        )
        
        # Verify backend selection logic
        assert config.should_use_llama_server("android") is True
        assert config.should_use_llama_server("linux") is False
        
        # Test configuration for control scenario (no cache)
        control_scenario = ABLATION_CACHE_CONFIG["control"]
        config.cache_mode = control_scenario["cache_mode"]
        
        assert config.get_cache_flags() == ["--cache-ram", "0", "--no-cache-prompt"]
        assert config.get_request_cache_setting(control_scenario["enable_prompt_cache"]) is False
        
        # Test configuration for warm cache scenario
        warm_scenario = ABLATION_CACHE_CONFIG["warm_cache"]
        config.cache_mode = warm_scenario["cache_mode"]
        
        assert config.get_cache_flags() == []  # No restrictions for BOTH mode
        assert config.get_request_cache_setting(warm_scenario["enable_prompt_cache"]) is True
        
        # Verify URLs are correctly generated
        assert config.get_completion_url() == "http://127.0.0.1:8080/completion"
        
        # Test serialization for persistence
        config_dict = config.to_dict()
        assert "enable_ablation_studies" in config_dict
        assert config_dict["enable_ablation_studies"] is True
        
        # Test deserialization
        restored_config = AndroidConfig.from_dict(config_dict)
        assert restored_config.enable_ablation_studies is True
        assert restored_config.cache_mode == config.cache_mode