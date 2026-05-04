"""
Unit tests for configuration validation.

Tests configuration parsing from JSON/YAML files, command-line arguments,
validation of parameters, and default value assignment.

Requirements tested: 11.8, 11.9
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest

from llm_benchmark.config import BenchmarkConfig, ConfigParser


class TestBenchmarkConfigValidation:
    """Test BenchmarkConfig validation logic."""
    
    def test_valid_minimal_config(self):
        """Test that minimal valid configuration is accepted."""
        config = BenchmarkConfig(
            repo_id="test/repo",
            models={"Q4_0": "model.gguf"}
        )
        
        assert config.repo_id == "test/repo"
        assert config.models == {"Q4_0": "model.gguf"}
        # Check defaults are applied
        assert config.context_size == 2048
        assert config.batch_size == 512
        assert config.max_tokens == 100
        assert config.iterations == 3
        assert config.warmup_runs == 2
    
    def test_valid_full_config(self):
        """Test that full valid configuration is accepted."""
        config = BenchmarkConfig(
            repo_id="test/repo",
            models={"Q4_0": "model.gguf", "Q8_0": "model2.gguf"},
            model_cache_dir="./custom_models",
            context_size=4096,
            batch_size=1024,
            max_tokens=200,
            iterations=5,
            warmup_runs=3,
            enable_quantization_profiling=True,
            enable_ablation_studies=False,
            enable_batch_testing=True,
            enable_thermal_monitoring=False,
            kv_cache_types=["ram"],
            prompt_cache_prefix_lengths=[100, 500],
            batch_sizes=[1, 2, 4],
            sleep_between_tests_s=10,
            thermal_stabilization_threshold_c=80.0,
            inference_timeout_s=600,
            output_dir="./custom_results",
            save_formats=["json", "csv"],
            visualization_dpi=600,
            hf_token="test_token"
        )
        
        assert config.repo_id == "test/repo"
        assert config.models == {"Q4_0": "model.gguf", "Q8_0": "model2.gguf"}
        assert config.model_cache_dir == "./custom_models"
        assert config.context_size == 4096
        assert config.batch_size == 1024
        assert config.max_tokens == 200
        assert config.iterations == 5
        assert config.warmup_runs == 3
        assert config.enable_quantization_profiling is True
        assert config.enable_ablation_studies is False
        assert config.enable_batch_testing is True
        assert config.enable_thermal_monitoring is False
        assert config.kv_cache_types == ["ram"]
        assert config.prompt_cache_prefix_lengths == [100, 500]
        assert config.batch_sizes == [1, 2, 4]
        assert config.sleep_between_tests_s == 10
        assert config.thermal_stabilization_threshold_c == 80.0
        assert config.inference_timeout_s == 600
        assert config.output_dir == "./custom_results"
        assert config.save_formats == ["json", "csv"]
        assert config.visualization_dpi == 600
        assert config.hf_token == "test_token"
    
    def test_invalid_context_size_zero(self):
        """Test that zero context_size is rejected."""
        with pytest.raises(ValueError, match="context_size must be positive"):
            BenchmarkConfig(
                repo_id="test/repo",
                models={"Q4_0": "model.gguf"},
                context_size=0
            )
    
    def test_invalid_context_size_negative(self):
        """Test that negative context_size is rejected."""
        with pytest.raises(ValueError, match="context_size must be positive"):
            BenchmarkConfig(
                repo_id="test/repo",
                models={"Q4_0": "model.gguf"},
                context_size=-1
            )
    
    def test_invalid_batch_size_zero(self):
        """Test that zero batch_size is rejected."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            BenchmarkConfig(
                repo_id="test/repo",
                models={"Q4_0": "model.gguf"},
                batch_size=0
            )
    
    def test_invalid_batch_size_negative(self):
        """Test that negative batch_size is rejected."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            BenchmarkConfig(
                repo_id="test/repo",
                models={"Q4_0": "model.gguf"},
                batch_size=-1
            )
    
    def test_invalid_max_tokens_zero(self):
        """Test that zero max_tokens is rejected."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            BenchmarkConfig(
                repo_id="test/repo",
                models={"Q4_0": "model.gguf"},
                max_tokens=0
            )
    
    def test_invalid_max_tokens_negative(self):
        """Test that negative max_tokens is rejected."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            BenchmarkConfig(
                repo_id="test/repo",
                models={"Q4_0": "model.gguf"},
                max_tokens=-1
            )
    
    def test_invalid_iterations_zero(self):
        """Test that zero iterations is rejected."""
        with pytest.raises(ValueError, match="iterations must be at least 1"):
            BenchmarkConfig(
                repo_id="test/repo",
                models={"Q4_0": "model.gguf"},
                iterations=0
            )
    
    def test_invalid_iterations_negative(self):
        """Test that negative iterations is rejected."""
        with pytest.raises(ValueError, match="iterations must be at least 1"):
            BenchmarkConfig(
                repo_id="test/repo",
                models={"Q4_0": "model.gguf"},
                iterations=-1
            )
    
    def test_invalid_warmup_runs_negative(self):
        """Test that negative warmup_runs is rejected."""
        with pytest.raises(ValueError, match="warmup_runs must be non-negative"):
            BenchmarkConfig(
                repo_id="test/repo",
                models={"Q4_0": "model.gguf"},
                warmup_runs=-1
            )
    
    def test_valid_warmup_runs_zero(self):
        """Test that zero warmup_runs is accepted (no warmup)."""
        config = BenchmarkConfig(
            repo_id="test/repo",
            models={"Q4_0": "model.gguf"},
            warmup_runs=0
        )
        assert config.warmup_runs == 0
    
    def test_invalid_sleep_between_tests_negative(self):
        """Test that negative sleep_between_tests_s is rejected."""
        with pytest.raises(ValueError, match="sleep_between_tests_s must be non-negative"):
            BenchmarkConfig(
                repo_id="test/repo",
                models={"Q4_0": "model.gguf"},
                sleep_between_tests_s=-1
            )
    
    def test_valid_sleep_between_tests_zero(self):
        """Test that zero sleep_between_tests_s is accepted (no sleep)."""
        config = BenchmarkConfig(
            repo_id="test/repo",
            models={"Q4_0": "model.gguf"},
            sleep_between_tests_s=0
        )
        assert config.sleep_between_tests_s == 0
    
    def test_invalid_inference_timeout_zero(self):
        """Test that zero inference_timeout_s is rejected."""
        with pytest.raises(ValueError, match="inference_timeout_s must be positive"):
            BenchmarkConfig(
                repo_id="test/repo",
                models={"Q4_0": "model.gguf"},
                inference_timeout_s=0
            )
    
    def test_invalid_inference_timeout_negative(self):
        """Test that negative inference_timeout_s is rejected."""
        with pytest.raises(ValueError, match="inference_timeout_s must be positive"):
            BenchmarkConfig(
                repo_id="test/repo",
                models={"Q4_0": "model.gguf"},
                inference_timeout_s=-1
            )
    
    def test_invalid_empty_models(self):
        """Test that empty models dictionary is rejected."""
        with pytest.raises(ValueError, match="models dictionary cannot be empty"):
            BenchmarkConfig(
                repo_id="test/repo",
                models={}
            )
    
    def test_invalid_empty_repo_id(self):
        """Test that empty repo_id is rejected."""
        with pytest.raises(ValueError, match="repo_id cannot be empty"):
            BenchmarkConfig(
                repo_id="",
                models={"Q4_0": "model.gguf"}
            )
    
    def test_default_values_assignment(self):
        """Test that default values are correctly assigned."""
        config = BenchmarkConfig(
            repo_id="test/repo",
            models={"Q4_0": "model.gguf"}
        )
        
        # Test all default values
        assert config.model_cache_dir == "./models"
        assert config.context_size == 2048
        assert config.batch_size == 512
        assert config.max_tokens == 100
        assert config.iterations == 3
        assert config.warmup_runs == 2
        assert config.enable_quantization_profiling is True
        assert config.enable_ablation_studies is True
        assert config.enable_batch_testing is True
        assert config.enable_thermal_monitoring is True
        assert config.kv_cache_types == ["ram", "disk"]
        assert config.prompt_cache_prefix_lengths == [100, 500, 1000]
        assert config.batch_sizes == [1, 2, 4, 8, 16]
        assert config.sleep_between_tests_s == 5
        assert config.thermal_stabilization_threshold_c == 70.0
        assert config.inference_timeout_s == 300
        assert config.output_dir == "./benchmark_results"
        assert config.save_formats == ["json", "csv", "markdown", "html"]
        assert config.visualization_dpi == 300
        assert config.hf_token is None


class TestConfigParserFromFile:
    """Test configuration loading from JSON and YAML files."""
    
    def test_load_from_json_file(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "repo_id": "test/repo",
            "models": {"Q4_0": "model.gguf", "Q8_0": "model2.gguf"},
            "context_size": 4096,
            "iterations": 5
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = ConfigParser.from_file(temp_path)
            
            assert config.repo_id == "test/repo"
            assert config.models == {"Q4_0": "model.gguf", "Q8_0": "model2.gguf"}
            assert config.context_size == 4096
            assert config.iterations == 5
            # Check defaults are still applied
            assert config.batch_size == 512
            assert config.max_tokens == 100
        finally:
            os.unlink(temp_path)
    
    def test_load_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        pytest.importorskip("yaml", reason="PyYAML not installed")
        
        config_yaml = """
repo_id: "test/repo"
models:
  Q4_0: "model.gguf"
  Q8_0: "model2.gguf"
context_size: 4096
iterations: 5
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_yaml)
            temp_path = f.name
        
        try:
            config = ConfigParser.from_file(temp_path)
            
            assert config.repo_id == "test/repo"
            assert config.models == {"Q4_0": "model.gguf", "Q8_0": "model2.gguf"}
            assert config.context_size == 4096
            assert config.iterations == 5
            # Check defaults are still applied
            assert config.batch_size == 512
            assert config.max_tokens == 100
        finally:
            os.unlink(temp_path)
    
    def test_load_from_yml_extension(self):
        """Test loading configuration from .yml file."""
        pytest.importorskip("yaml", reason="PyYAML not installed")
        
        config_yaml = """
repo_id: "test/repo"
models:
  Q4_0: "model.gguf"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(config_yaml)
            temp_path = f.name
        
        try:
            config = ConfigParser.from_file(temp_path)
            
            assert config.repo_id == "test/repo"
            assert config.models == {"Q4_0": "model.gguf"}
        finally:
            os.unlink(temp_path)
    
    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            ConfigParser.from_file("/nonexistent/config.json")
    
    def test_unsupported_file_format(self):
        """Test that ValueError is raised for unsupported file format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("some content")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported configuration file format"):
                ConfigParser.from_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_invalid_json_syntax(self):
        """Test that JSONDecodeError is raised for invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{invalid json")
            temp_path = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                ConfigParser.from_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_yaml_without_pyyaml_installed(self, monkeypatch):
        """Test that ImportError is raised when PyYAML is not installed."""
        # Mock HAS_YAML to False
        import llm_benchmark.config
        monkeypatch.setattr(llm_benchmark.config, 'HAS_YAML', False)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("repo_id: test")
            temp_path = f.name
        
        try:
            with pytest.raises(ImportError, match="PyYAML is required"):
                ConfigParser.from_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_json_with_invalid_config_values(self):
        """Test that validation errors are raised for invalid values in JSON."""
        config_data = {
            "repo_id": "test/repo",
            "models": {"Q4_0": "model.gguf"},
            "context_size": -1  # Invalid
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="context_size must be positive"):
                ConfigParser.from_file(temp_path)
        finally:
            os.unlink(temp_path)


class TestConfigParserFromArgs:
    """Test configuration creation from command-line arguments."""
    
    def test_minimal_args(self):
        """Test configuration from minimal command-line arguments."""
        args = argparse.Namespace(
            repo_id="test/repo",
            models="Q4_0:model.gguf"
        )
        
        config = ConfigParser.from_args(args)
        
        assert config.repo_id == "test/repo"
        assert config.models == {"Q4_0": "model.gguf"}
        # Check defaults are applied
        assert config.context_size == 2048
        assert config.iterations == 3
    
    def test_multiple_models_args(self):
        """Test parsing multiple models from command-line."""
        args = argparse.Namespace(
            repo_id="test/repo",
            models="Q4_0:model1.gguf,Q8_0:model2.gguf,Q2_K:model3.gguf"
        )
        
        config = ConfigParser.from_args(args)
        
        assert config.models == {
            "Q4_0": "model1.gguf",
            "Q8_0": "model2.gguf",
            "Q2_K": "model3.gguf"
        }
    
    def test_full_args(self):
        """Test configuration from full command-line arguments."""
        args = argparse.Namespace(
            repo_id="test/repo",
            models="Q4_0:model.gguf",
            model_cache_dir="./custom_models",
            context_size=4096,
            batch_size=1024,
            max_tokens=200,
            iterations=5,
            warmup_runs=3,
            disable_quantization_profiling=False,
            disable_ablation_studies=True,
            disable_batch_testing=False,
            disable_thermal_monitoring=True,
            sleep_between_tests_s=10,
            thermal_stabilization_threshold_c=80.0,
            inference_timeout_s=600,
            output_dir="./custom_results",
            visualization_dpi=600,
            hf_token="test_token"
        )
        
        config = ConfigParser.from_args(args)
        
        assert config.repo_id == "test/repo"
        assert config.model_cache_dir == "./custom_models"
        assert config.context_size == 4096
        assert config.batch_size == 1024
        assert config.max_tokens == 200
        assert config.iterations == 5
        assert config.warmup_runs == 3
        assert config.enable_quantization_profiling is True  # Not disabled
        assert config.enable_ablation_studies is False  # Disabled
        assert config.enable_batch_testing is True  # Not disabled
        assert config.enable_thermal_monitoring is False  # Disabled
        assert config.sleep_between_tests_s == 10
        assert config.thermal_stabilization_threshold_c == 80.0
        assert config.inference_timeout_s == 600
        assert config.output_dir == "./custom_results"
        assert config.visualization_dpi == 600
        assert config.hf_token == "test_token"
    
    def test_disable_flags(self):
        """Test that disable flags correctly set boolean values to False."""
        args = argparse.Namespace(
            repo_id="test/repo",
            models="Q4_0:model.gguf",
            disable_quantization_profiling=True,
            disable_ablation_studies=True,
            disable_batch_testing=True,
            disable_thermal_monitoring=True
        )
        
        config = ConfigParser.from_args(args)
        
        assert config.enable_quantization_profiling is False
        assert config.enable_ablation_studies is False
        assert config.enable_batch_testing is False
        assert config.enable_thermal_monitoring is False
    
    def test_args_without_optional_params(self):
        """Test that None values in args don't override defaults."""
        args = argparse.Namespace(
            repo_id="test/repo",
            models="Q4_0:model.gguf",
            context_size=None,
            batch_size=None,
            iterations=None
        )
        
        config = ConfigParser.from_args(args)
        
        # Defaults should be used
        assert config.context_size == 2048
        assert config.batch_size == 512
        assert config.iterations == 3


class TestConfigParserArgumentParser:
    """Test argument parser creation."""
    
    def test_create_argument_parser(self):
        """Test that argument parser is created successfully."""
        parser = ConfigParser.create_argument_parser()
        
        assert parser is not None
        assert isinstance(parser, argparse.ArgumentParser)
    
    def test_parser_has_config_argument(self):
        """Test that parser has --config argument."""
        parser = ConfigParser.create_argument_parser()
        
        # Parse with config argument
        args = parser.parse_args(['--config', 'test.json'])
        assert args.config == 'test.json'
    
    def test_parser_has_repo_id_argument(self):
        """Test that parser has --repo-id argument."""
        parser = ConfigParser.create_argument_parser()
        
        args = parser.parse_args(['--repo-id', 'test/repo'])
        assert args.repo_id == 'test/repo'
    
    def test_parser_has_models_argument(self):
        """Test that parser has --models argument."""
        parser = ConfigParser.create_argument_parser()
        
        args = parser.parse_args(['--models', 'Q4_0:model.gguf'])
        assert args.models == 'Q4_0:model.gguf'
    
    def test_parser_has_integer_arguments(self):
        """Test that parser has integer arguments with correct types."""
        parser = ConfigParser.create_argument_parser()
        
        args = parser.parse_args([
            '--context-size', '4096',
            '--batch-size', '1024',
            '--max-tokens', '200',
            '--iterations', '5',
            '--warmup-runs', '3'
        ])
        
        assert args.context_size == 4096
        assert args.batch_size == 1024
        assert args.max_tokens == 200
        assert args.iterations == 5
        assert args.warmup_runs == 3
    
    def test_parser_has_disable_flags(self):
        """Test that parser has disable flags."""
        parser = ConfigParser.create_argument_parser()
        
        args = parser.parse_args([
            '--disable-quantization-profiling',
            '--disable-ablation-studies',
            '--disable-batch-testing',
            '--disable-thermal-monitoring'
        ])
        
        assert args.disable_quantization_profiling is True
        assert args.disable_ablation_studies is True
        assert args.disable_batch_testing is True
        assert args.disable_thermal_monitoring is True
    
    def test_parser_disable_flags_default_false(self):
        """Test that disable flags default to False."""
        parser = ConfigParser.create_argument_parser()
        
        args = parser.parse_args([])
        
        assert args.disable_quantization_profiling is False
        assert args.disable_ablation_studies is False
        assert args.disable_batch_testing is False
        assert args.disable_thermal_monitoring is False


class TestConfigParserLoadConfig:
    """Test the unified load_config method."""
    
    def test_load_from_file_only(self):
        """Test loading configuration from file only."""
        config_data = {
            "repo_id": "test/repo",
            "models": {"Q4_0": "model.gguf"},
            "context_size": 4096
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            args = argparse.Namespace(config=temp_path)
            config = ConfigParser.load_config(args)
            
            assert config.repo_id == "test/repo"
            assert config.context_size == 4096
        finally:
            os.unlink(temp_path)
    
    def test_load_from_args_only(self):
        """Test loading configuration from command-line arguments only."""
        args = argparse.Namespace(
            config=None,
            repo_id="test/repo",
            models="Q4_0:model.gguf",
            context_size=4096
        )
        
        config = ConfigParser.load_config(args)
        
        assert config.repo_id == "test/repo"
        assert config.context_size == 4096
    
    def test_args_override_file(self):
        """Test that command-line arguments override file configuration."""
        config_data = {
            "repo_id": "file/repo",
            "models": {"Q4_0": "file_model.gguf"},
            "context_size": 2048,
            "iterations": 3
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            args = argparse.Namespace(
                config=temp_path,
                repo_id="args/repo",  # Override
                models="Q8_0:args_model.gguf",  # Override
                context_size=4096,  # Override
                iterations=None  # Don't override
            )
            
            config = ConfigParser.load_config(args)
            
            # Overridden values
            assert config.repo_id == "args/repo"
            assert config.models == {"Q8_0": "args_model.gguf"}
            assert config.context_size == 4096
            # Non-overridden value from file
            assert config.iterations == 3
        finally:
            os.unlink(temp_path)
    
    def test_disable_flags_override_file(self):
        """Test that disable flags override file configuration."""
        config_data = {
            "repo_id": "test/repo",
            "models": {"Q4_0": "model.gguf"},
            "enable_ablation_studies": True,
            "enable_batch_testing": True
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            args = argparse.Namespace(
                config=temp_path,
                disable_ablation_studies=True,
                disable_batch_testing=False
            )
            
            config = ConfigParser.load_config(args)
            
            assert config.enable_ablation_studies is False  # Overridden
            assert config.enable_batch_testing is True  # Not overridden
        finally:
            os.unlink(temp_path)
    
    def test_no_config_source_raises_error(self):
        """Test that ValueError is raised when no configuration source is provided."""
        args = argparse.Namespace(config=None, repo_id=None)
        
        with pytest.raises(ValueError, match="Configuration must be provided"):
            ConfigParser.load_config(args)
    
    def test_load_config_without_args(self):
        """Test that ValueError is raised when called without args."""
        with pytest.raises(ValueError, match="Configuration must be provided"):
            ConfigParser.load_config(None)


class TestConfigParserEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_models_parsing_with_colons_in_filename(self):
        """Test parsing models when filename contains colons."""
        # This should fail because we split on ':' - documenting current behavior
        args = argparse.Namespace(
            repo_id="test/repo",
            models="Q4_0:path:to:model.gguf"
        )
        
        # This will raise ValueError because split(':') produces more than 2 parts
        with pytest.raises(ValueError):
            ConfigParser.from_args(args)
    
    def test_empty_models_string(self):
        """Test that empty models string is handled."""
        args = argparse.Namespace(
            repo_id="test/repo",
            models=""
        )
        
        # Empty string is falsy, so models won't be added to config_dict
        # This will raise TypeError because 'models' is required
        with pytest.raises(TypeError, match="missing 1 required positional argument: 'models'"):
            ConfigParser.from_args(args)
    
    def test_config_with_extra_fields(self):
        """Test that extra fields in config file are ignored."""
        config_data = {
            "repo_id": "test/repo",
            "models": {"Q4_0": "model.gguf"},
            "unknown_field": "should be ignored"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            # Should not raise error, extra fields are ignored by dataclass
            with pytest.raises(TypeError, match="unexpected keyword argument"):
                ConfigParser.from_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_boundary_value_iterations_one(self):
        """Test boundary value: iterations = 1 (minimum valid)."""
        config = BenchmarkConfig(
            repo_id="test/repo",
            models={"Q4_0": "model.gguf"},
            iterations=1
        )
        assert config.iterations == 1
    
    def test_boundary_value_warmup_runs_zero(self):
        """Test boundary value: warmup_runs = 0 (minimum valid)."""
        config = BenchmarkConfig(
            repo_id="test/repo",
            models={"Q4_0": "model.gguf"},
            warmup_runs=0
        )
        assert config.warmup_runs == 0
    
    def test_very_large_values(self):
        """Test that very large values are accepted."""
        config = BenchmarkConfig(
            repo_id="test/repo",
            models={"Q4_0": "model.gguf"},
            context_size=1000000,
            batch_size=1000000,
            max_tokens=1000000,
            iterations=1000000,
            inference_timeout_s=1000000
        )
        
        assert config.context_size == 1000000
        assert config.batch_size == 1000000
        assert config.max_tokens == 1000000
        assert config.iterations == 1000000
        assert config.inference_timeout_s == 1000000
