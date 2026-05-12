"""
Unit tests for configuration parsing with comment fields.

Tests that configuration files can contain comment fields (keys starting with underscore)
that are automatically filtered out during parsing.
"""

import pytest
import json
import tempfile
from pathlib import Path

from llm_benchmark.config import ConfigParser, BenchmarkConfig


class TestConfigCommentFields:
    """Test configuration parsing with comment fields."""
    
    def test_json_config_with_comment_fields(self):
        """
        Test that JSON config files with _comment fields are parsed correctly.
        
        Comment fields (keys starting with underscore) should be filtered out
        and not cause errors when creating BenchmarkConfig.
        """
        config_data = {
            "_comment": "This is a comment field that should be ignored",
            "repo_id": "test/repo",
            "models": {
                "Q4_0": "model.gguf"
            },
            "_note": "Another comment field",
            "context_size": 2048,
            "max_tokens": 100,
            "_description": "Test configuration"
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            # Should load successfully without errors
            config = ConfigParser.from_file(config_path)
            
            # Verify config values
            assert config.repo_id == "test/repo"
            assert config.models == {"Q4_0": "model.gguf"}
            assert config.context_size == 2048
            assert config.max_tokens == 100
            
            # Verify comment fields were filtered out (not accessible as attributes)
            assert not hasattr(config, '_comment')
            assert not hasattr(config, '_note')
            assert not hasattr(config, '_description')
            
        finally:
            # Clean up
            config_path.unlink()
    
    def test_config_without_comment_fields(self):
        """
        Test that config files without comment fields still work correctly.
        """
        config_data = {
            "repo_id": "test/repo",
            "models": {
                "Q4_0": "model.gguf"
            }
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            # Should load successfully
            config = ConfigParser.from_file(config_path)
            
            # Verify config values
            assert config.repo_id == "test/repo"
            assert config.models == {"Q4_0": "model.gguf"}
            
        finally:
            # Clean up
            config_path.unlink()
    
    def test_multiple_underscore_prefixed_fields(self):
        """
        Test that multiple underscore-prefixed fields are all filtered out.
        """
        config_data = {
            "_comment1": "First comment",
            "_comment2": "Second comment",
            "__private": "Private field",
            "___triple": "Triple underscore",
            "repo_id": "test/repo",
            "models": {"Q4_0": "model.gguf"},
            "_metadata": {"author": "test", "version": "1.0"}
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            # Should load successfully with all underscore fields filtered
            config = ConfigParser.from_file(config_path)
            
            # Verify only non-underscore fields are present
            assert config.repo_id == "test/repo"
            assert config.models == {"Q4_0": "model.gguf"}
            
        finally:
            # Clean up
            config_path.unlink()
    
    def test_x86_linux_example_config(self):
        """
        Test that the actual x86_linux_example.json config loads correctly.
        
        This config file contains _comment fields that should be filtered out.
        """
        config_path = Path("configs/x86_linux_example.json")
        
        if not config_path.exists():
            pytest.skip("x86_linux_example.json not found")
        
        # Should load successfully
        config = ConfigParser.from_file(config_path)
        
        # Verify it loaded correctly
        assert config.repo_id is not None
        assert len(config.models) > 0
        assert config.context_size > 0
