"""
Unit tests for cache configuration models in NativeLlamaServer.

Tests the CacheMode enum, ABLATION_CACHE_CONFIG mapping, and validation functions
to ensure proper cache control for ablation studies.

Requirements tested: 4.1, 4.2, 4.3, 4.4
"""

import pytest
from typing import Dict, Any

from llm_benchmark.inference.native_llama_server import (
    CacheMode,
    ABLATION_CACHE_CONFIG,
    validate_cache_mode,
    get_ablation_cache_config,
    validate_ablation_scenario
)


class TestCacheMode:
    """Test CacheMode enum values and behavior."""
    
    def test_cache_mode_enum_values(self):
        """Test that CacheMode enum has all expected values."""
        # Test all enum values exist
        assert CacheMode.NONE.value == "none"
        assert CacheMode.RAM_ONLY.value == "ram_only"
        assert CacheMode.DISK_ONLY.value == "disk_only"
        assert CacheMode.BOTH.value == "both"
    
    def test_cache_mode_enum_count(self):
        """Test that CacheMode enum has exactly 4 values."""
        cache_modes = list(CacheMode)
        assert len(cache_modes) == 4
    
    def test_cache_mode_string_representation(self):
        """Test string representation of CacheMode enum values."""
        assert str(CacheMode.NONE) == "CacheMode.NONE"
        assert str(CacheMode.RAM_ONLY) == "CacheMode.RAM_ONLY"
        assert str(CacheMode.DISK_ONLY) == "CacheMode.DISK_ONLY"
        assert str(CacheMode.BOTH) == "CacheMode.BOTH"
    
    def test_cache_mode_equality(self):
        """Test CacheMode enum equality comparisons."""
        assert CacheMode.NONE == CacheMode.NONE
        assert CacheMode.NONE != CacheMode.RAM_ONLY
        assert CacheMode.RAM_ONLY != CacheMode.DISK_ONLY
        assert CacheMode.DISK_ONLY != CacheMode.BOTH


class TestValidateCacheMode:
    """Test validate_cache_mode function."""
    
    def test_validate_cache_mode_valid_strings(self):
        """Test validation of valid cache mode strings."""
        assert validate_cache_mode("none") == CacheMode.NONE
        assert validate_cache_mode("ram_only") == CacheMode.RAM_ONLY
        assert validate_cache_mode("disk_only") == CacheMode.DISK_ONLY
        assert validate_cache_mode("both") == CacheMode.BOTH
    
    def test_validate_cache_mode_invalid_strings(self):
        """Test validation of invalid cache mode strings raises ValueError."""
        invalid_modes = ["invalid", "cache", "memory", "disk", "", "None", "RAM_ONLY"]
        
        for invalid_mode in invalid_modes:
            with pytest.raises(ValueError) as exc_info:
                validate_cache_mode(invalid_mode)
            
            # Check error message contains the invalid mode and valid options
            error_msg = str(exc_info.value)
            assert invalid_mode in error_msg
            assert "Valid options:" in error_msg
            assert "none" in error_msg
            assert "ram_only" in error_msg
            assert "disk_only" in error_msg
            assert "both" in error_msg
    
    def test_validate_cache_mode_case_sensitivity(self):
        """Test that cache mode validation is case sensitive."""
        case_variants = ["NONE", "None", "RAM_ONLY", "Ram_Only", "DISK_ONLY", "Disk_Only", "BOTH", "Both"]
        
        for variant in case_variants:
            with pytest.raises(ValueError):
                validate_cache_mode(variant)
    
    def test_validate_cache_mode_whitespace(self):
        """Test that cache mode validation handles whitespace correctly."""
        whitespace_variants = [" none", "none ", " none ", "\tnone", "none\n"]
        
        for variant in whitespace_variants:
            with pytest.raises(ValueError):
                validate_cache_mode(variant)


class TestAblationCacheConfig:
    """Test ABLATION_CACHE_CONFIG mapping."""
    
    def test_ablation_cache_config_structure(self):
        """Test that ABLATION_CACHE_CONFIG has expected structure."""
        # Test all expected scenarios exist
        expected_scenarios = ["control", "cold_cache", "warm_cache", "ram_only", "disk_only"]
        assert set(ABLATION_CACHE_CONFIG.keys()) == set(expected_scenarios)
        
        # Test each scenario has required fields
        for scenario, config in ABLATION_CACHE_CONFIG.items():
            assert "cache_mode" in config
            assert "enable_prompt_cache" in config
            assert "description" in config
            assert isinstance(config["cache_mode"], CacheMode)
            assert isinstance(config["enable_prompt_cache"], bool)
            assert isinstance(config["description"], str)
    
    def test_control_scenario_config(self):
        """Test control scenario has correct cache configuration."""
        control_config = ABLATION_CACHE_CONFIG["control"]
        assert control_config["cache_mode"] == CacheMode.NONE
        assert control_config["enable_prompt_cache"] is False
        assert "baseline" in control_config["description"].lower()
    
    def test_cold_cache_scenario_config(self):
        """Test cold_cache scenario has correct cache configuration."""
        cold_config = ABLATION_CACHE_CONFIG["cold_cache"]
        assert cold_config["cache_mode"] == CacheMode.BOTH
        assert cold_config["enable_prompt_cache"] is True
        assert "empty" in cold_config["description"].lower() or "first" in cold_config["description"].lower()
    
    def test_warm_cache_scenario_config(self):
        """Test warm_cache scenario has correct cache configuration."""
        warm_config = ABLATION_CACHE_CONFIG["warm_cache"]
        assert warm_config["cache_mode"] == CacheMode.BOTH
        assert warm_config["enable_prompt_cache"] is True
        assert "reused" in warm_config["description"].lower() or "subsequent" in warm_config["description"].lower()
    
    def test_ram_only_scenario_config(self):
        """Test ram_only scenario has correct cache configuration."""
        ram_config = ABLATION_CACHE_CONFIG["ram_only"]
        assert ram_config["cache_mode"] == CacheMode.RAM_ONLY
        assert ram_config["enable_prompt_cache"] is False
        assert "ram" in ram_config["description"].lower()
    
    def test_disk_only_scenario_config(self):
        """Test disk_only scenario has correct cache configuration."""
        disk_config = ABLATION_CACHE_CONFIG["disk_only"]
        assert disk_config["cache_mode"] == CacheMode.DISK_ONLY
        assert disk_config["enable_prompt_cache"] is True
        assert "disk" in disk_config["description"].lower()
    
    def test_ablation_scenarios_logical_consistency(self):
        """Test that ablation scenarios are logically consistent."""
        # Control should disable all caching
        control = ABLATION_CACHE_CONFIG["control"]
        assert control["cache_mode"] == CacheMode.NONE
        assert control["enable_prompt_cache"] is False
        
        # Cold and warm cache should both use full caching but differ in prompt cache usage context
        cold = ABLATION_CACHE_CONFIG["cold_cache"]
        warm = ABLATION_CACHE_CONFIG["warm_cache"]
        assert cold["cache_mode"] == warm["cache_mode"] == CacheMode.BOTH
        assert cold["enable_prompt_cache"] == warm["enable_prompt_cache"] is True
        
        # RAM only should disable disk caching
        ram_only = ABLATION_CACHE_CONFIG["ram_only"]
        assert ram_only["cache_mode"] == CacheMode.RAM_ONLY
        assert ram_only["enable_prompt_cache"] is False  # Prompt cache is disk-based
        
        # Disk only should disable RAM caching but enable prompt cache
        disk_only = ABLATION_CACHE_CONFIG["disk_only"]
        assert disk_only["cache_mode"] == CacheMode.DISK_ONLY
        assert disk_only["enable_prompt_cache"] is True  # Prompt cache is disk-based


class TestGetAblationCacheConfig:
    """Test get_ablation_cache_config function."""
    
    def test_get_ablation_cache_config_valid_scenarios(self):
        """Test getting cache config for valid ablation scenarios."""
        valid_scenarios = ["control", "cold_cache", "warm_cache", "ram_only", "disk_only"]
        
        for scenario in valid_scenarios:
            config = get_ablation_cache_config(scenario)
            
            # Should return a copy of the original config
            assert config is not ABLATION_CACHE_CONFIG[scenario]
            assert config == ABLATION_CACHE_CONFIG[scenario]
            
            # Should have all required fields
            assert "cache_mode" in config
            assert "enable_prompt_cache" in config
            assert "description" in config
    
    def test_get_ablation_cache_config_returns_copy(self):
        """Test that get_ablation_cache_config returns a copy, not reference."""
        original_config = ABLATION_CACHE_CONFIG["control"]
        returned_config = get_ablation_cache_config("control")
        
        # Modify returned config
        returned_config["test_field"] = "test_value"
        
        # Original should be unchanged
        assert "test_field" not in original_config
        assert original_config == ABLATION_CACHE_CONFIG["control"]
    
    def test_get_ablation_cache_config_invalid_scenarios(self):
        """Test getting cache config for invalid ablation scenarios raises ValueError."""
        invalid_scenarios = ["invalid", "test", "", "Control", "CONTROL", "cold", "warm"]
        
        for invalid_scenario in invalid_scenarios:
            with pytest.raises(ValueError) as exc_info:
                get_ablation_cache_config(invalid_scenario)
            
            # Check error message contains the invalid scenario and valid options
            error_msg = str(exc_info.value)
            assert invalid_scenario in error_msg
            assert "Valid options:" in error_msg
            assert "control" in error_msg
            assert "cold_cache" in error_msg
            assert "warm_cache" in error_msg
            assert "ram_only" in error_msg
            assert "disk_only" in error_msg
    
    def test_get_ablation_cache_config_case_sensitivity(self):
        """Test that ablation scenario names are case sensitive."""
        case_variants = ["Control", "CONTROL", "Cold_Cache", "COLD_CACHE", "Warm_Cache", "WARM_CACHE"]
        
        for variant in case_variants:
            with pytest.raises(ValueError):
                get_ablation_cache_config(variant)


class TestValidateAblationScenario:
    """Test validate_ablation_scenario function."""
    
    def test_validate_ablation_scenario_valid_scenarios(self):
        """Test validation of valid ablation scenarios."""
        valid_scenarios = ["control", "cold_cache", "warm_cache", "ram_only", "disk_only"]
        
        for scenario in valid_scenarios:
            # Should return the same scenario string if valid
            result = validate_ablation_scenario(scenario)
            assert result == scenario
    
    def test_validate_ablation_scenario_invalid_scenarios(self):
        """Test validation of invalid ablation scenarios raises ValueError."""
        invalid_scenarios = ["invalid", "test", "", "Control", "CONTROL", "cold", "warm", "none", "both"]
        
        for invalid_scenario in invalid_scenarios:
            with pytest.raises(ValueError) as exc_info:
                validate_ablation_scenario(invalid_scenario)
            
            # Check error message contains the invalid scenario and valid options
            error_msg = str(exc_info.value)
            assert invalid_scenario in error_msg
            assert "Valid options:" in error_msg
            assert "control" in error_msg
            assert "cold_cache" in error_msg
            assert "warm_cache" in error_msg
            assert "ram_only" in error_msg
            assert "disk_only" in error_msg
    
    def test_validate_ablation_scenario_whitespace(self):
        """Test that ablation scenario validation handles whitespace correctly."""
        whitespace_variants = [" control", "control ", " control ", "\tcontrol", "control\n"]
        
        for variant in whitespace_variants:
            with pytest.raises(ValueError):
                validate_ablation_scenario(variant)


class TestCacheConfigurationIntegration:
    """Integration tests for cache configuration components."""
    
    def test_cache_mode_enum_matches_ablation_config(self):
        """Test that all CacheMode enum values are used in ABLATION_CACHE_CONFIG."""
        used_cache_modes = set()
        for config in ABLATION_CACHE_CONFIG.values():
            used_cache_modes.add(config["cache_mode"])
        
        all_cache_modes = set(CacheMode)
        assert used_cache_modes == all_cache_modes, "All CacheMode values should be used in ablation scenarios"
    
    def test_ablation_scenarios_cover_cache_combinations(self):
        """Test that ablation scenarios cover important cache combinations."""
        scenarios = ABLATION_CACHE_CONFIG
        
        # Should have a scenario with no caching (control)
        no_cache_scenarios = [s for s, c in scenarios.items() 
                             if c["cache_mode"] == CacheMode.NONE and not c["enable_prompt_cache"]]
        assert len(no_cache_scenarios) >= 1, "Should have at least one no-cache scenario"
        
        # Should have a scenario with full caching
        full_cache_scenarios = [s for s, c in scenarios.items() 
                               if c["cache_mode"] == CacheMode.BOTH and c["enable_prompt_cache"]]
        assert len(full_cache_scenarios) >= 1, "Should have at least one full-cache scenario"
        
        # Should have scenarios for each cache type individually
        ram_only_scenarios = [s for s, c in scenarios.items() 
                             if c["cache_mode"] == CacheMode.RAM_ONLY]
        assert len(ram_only_scenarios) >= 1, "Should have at least one RAM-only scenario"
        
        disk_only_scenarios = [s for s, c in scenarios.items() 
                              if c["cache_mode"] == CacheMode.DISK_ONLY]
        assert len(disk_only_scenarios) >= 1, "Should have at least one disk-only scenario"
    
    def test_validation_functions_consistency(self):
        """Test that validation functions are consistent with each other."""
        # All valid cache modes should work with validate_cache_mode
        for cache_mode in CacheMode:
            validated = validate_cache_mode(cache_mode.value)
            assert validated == cache_mode
        
        # All valid ablation scenarios should work with validate_ablation_scenario
        for scenario in ABLATION_CACHE_CONFIG.keys():
            validated = validate_ablation_scenario(scenario)
            assert validated == scenario
        
        # All ablation scenarios should return valid configs
        for scenario in ABLATION_CACHE_CONFIG.keys():
            config = get_ablation_cache_config(scenario)
            assert isinstance(config["cache_mode"], CacheMode)
            assert isinstance(config["enable_prompt_cache"], bool)
            assert isinstance(config["description"], str)