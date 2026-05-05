"""
Property Tests for NativeLlamaServer Initialization

**Validates: Requirements 2.2, 4.1, 4.2, 4.3, 4.4**

This test verifies that the NativeLlamaServer correctly constructs command-line arguments
for all valid initialization parameters, ensuring proper cache control flags are applied
based on the cache_mode setting.

Property 2: Command-Line Argument Construction
For any valid initialization parameters (model_path, n_ctx, n_threads, cache_mode),
the NativeLlamaServer should construct command-line arguments that include all parameters
and appropriate cache control flags based on the cache_mode.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings, assume

from llm_benchmark.inference.native_llama_server import NativeLlamaServer, CacheMode


class TestNativeLlamaServerInitializationProperties:
    """
    Property 2: Command-Line Argument Construction
    
    **Validates: Requirements 2.2, 4.1, 4.2, 4.3, 4.4**
    
    Tests that for any valid initialization parameters (model_path, n_ctx, n_threads, cache_mode),
    the NativeLlamaServer correctly constructs command-line arguments that include all parameters
    and appropriate cache control flags based on the cache_mode.
    """
    
    @given(
        model_path=st.text(min_size=1, max_size=100).filter(lambda x: not any(c in x for c in ['<', '>', ':', '"', '|', '?', '*'])),
        n_ctx=st.integers(min_value=512, max_value=8192),
        n_threads=st.integers(min_value=1, max_value=16),
        n_batch=st.integers(min_value=1, max_value=2048),
        cache_mode=st.sampled_from([CacheMode.NONE, CacheMode.RAM_ONLY, CacheMode.DISK_ONLY, CacheMode.BOTH]),
        host=st.sampled_from(["127.0.0.1", "localhost", "0.0.0.0"]),
        port=st.integers(min_value=1024, max_value=65535)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_command_line_argument_construction(
        self, model_path, n_ctx, n_threads, n_batch, cache_mode, host, port
    ):
        """
        Property 2: Command-Line Argument Construction
        
        **Validates: Requirements 2.2, 4.1, 4.2, 4.3, 4.4**
        
        For any valid initialization parameters, the NativeLlamaServer should construct
        command-line arguments that correctly include all parameters and appropriate
        cache control flags based on the cache_mode.
        """
        # Create temporary files for model and llama-server binary
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock model file
            model_file = temp_path / "model.gguf"
            model_file.write_text("mock model")
            
            # Create mock llama-server binary
            server_binary = temp_path / "llama-server"
            server_binary.write_text("#!/bin/bash\necho 'mock server'")
            server_binary.chmod(0o755)
            
            # Mock subprocess and HTTP operations to prevent actual server startup
            with patch('subprocess.Popen') as mock_popen, \
                 patch('requests.Session') as mock_session, \
                 patch.object(NativeLlamaServer, '_wait_for_health_check'), \
                 patch.object(NativeLlamaServer, '_start_memory_monitoring'):
                
                # Mock the process
                mock_process = MagicMock()
                mock_process.pid = 12345
                mock_process.poll.return_value = None  # Process is running
                mock_popen.return_value = mock_process
                
                # Create NativeLlamaServer instance
                server = NativeLlamaServer(
                    model_path=str(model_file),
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    n_batch=n_batch,
                    cache_mode=cache_mode.value,
                    llama_server_path=str(server_binary),
                    host=host,
                    port=port
                )
                
                # Get the command that would be built
                cmd = server._build_server_command()
                
                # Verify basic command structure
                assert isinstance(cmd, list), "Command should be a list"
                assert len(cmd) >= 8, "Command should have at least 8 basic arguments"
                
                # Verify binary path
                assert cmd[0] == str(server_binary), f"First argument should be server binary path: {server_binary}"
                
                # Verify model path
                model_idx = cmd.index("-m")
                assert cmd[model_idx + 1] == str(model_file), f"Model path should be {model_file}"
                
                # Verify context size
                ctx_idx = cmd.index("-c")
                assert cmd[ctx_idx + 1] == str(n_ctx), f"Context size should be {n_ctx}"
                
                # Verify thread count
                threads_idx = cmd.index("-t")
                assert cmd[threads_idx + 1] == str(n_threads), f"Thread count should be {n_threads}"
                
                # Verify batch size
                batch_idx = cmd.index("-b")
                assert cmd[batch_idx + 1] == str(n_batch), f"Batch size should be {n_batch}"
                
                # Verify host
                host_idx = cmd.index("--host")
                assert cmd[host_idx + 1] == host, f"Host should be {host}"
                
                # Verify port
                port_idx = cmd.index("--port")
                assert cmd[port_idx + 1] == str(port), f"Port should be {port}"
                
                # Verify cache control flags based on cache_mode
                if cache_mode == CacheMode.NONE:
                    # Should have both --cache-ram 0 and --no-cache-prompt
                    assert "--cache-ram" in cmd, "NONE mode should include --cache-ram flag"
                    cache_ram_idx = cmd.index("--cache-ram")
                    assert cmd[cache_ram_idx + 1] == "0", "NONE mode should set --cache-ram to 0"
                    assert "--no-cache-prompt" in cmd, "NONE mode should include --no-cache-prompt flag"
                    
                elif cache_mode == CacheMode.RAM_ONLY:
                    # Should have --no-cache-prompt but not --cache-ram 0
                    assert "--no-cache-prompt" in cmd, "RAM_ONLY mode should include --no-cache-prompt flag"
                    assert "--cache-ram" not in cmd, "RAM_ONLY mode should not include --cache-ram flag"
                    
                elif cache_mode == CacheMode.DISK_ONLY:
                    # Should have --cache-ram 0 but not --no-cache-prompt
                    assert "--cache-ram" in cmd, "DISK_ONLY mode should include --cache-ram flag"
                    cache_ram_idx = cmd.index("--cache-ram")
                    assert cmd[cache_ram_idx + 1] == "0", "DISK_ONLY mode should set --cache-ram to 0"
                    assert "--no-cache-prompt" not in cmd, "DISK_ONLY mode should not include --no-cache-prompt flag"
                    
                elif cache_mode == CacheMode.BOTH:
                    # Should have neither cache restriction flag
                    assert "--cache-ram" not in cmd, "BOTH mode should not include --cache-ram flag"
                    assert "--no-cache-prompt" not in cmd, "BOTH mode should not include --no-cache-prompt flag"
                
                # Clean up
                server.close()
    
    def test_cache_mode_none_includes_both_cache_flags(self):
        """
        Test that cache_mode "none" includes both --cache-ram 0 and --no-cache-prompt flags.
        
        **Validates: Requirements 4.1**
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock files
            model_file = temp_path / "model.gguf"
            model_file.write_text("mock model")
            server_binary = temp_path / "llama-server"
            server_binary.write_text("#!/bin/bash\necho 'mock server'")
            server_binary.chmod(0o755)
            
            # Mock subprocess and HTTP operations
            with patch('subprocess.Popen') as mock_popen, \
                 patch('requests.Session'), \
                 patch.object(NativeLlamaServer, '_wait_for_health_check'), \
                 patch.object(NativeLlamaServer, '_start_memory_monitoring'):
                
                mock_process = MagicMock()
                mock_process.pid = 12345
                mock_process.poll.return_value = None
                mock_popen.return_value = mock_process
                
                server = NativeLlamaServer(
                    model_path=str(model_file),
                    cache_mode="none",
                    llama_server_path=str(server_binary)
                )
                
                cmd = server._build_server_command()
                
                # Verify both cache flags are present
                assert "--cache-ram" in cmd
                assert "--no-cache-prompt" in cmd
                
                # Verify --cache-ram is set to 0
                cache_ram_idx = cmd.index("--cache-ram")
                assert cmd[cache_ram_idx + 1] == "0"
                
                server.close()
    
    def test_cache_mode_ram_only_includes_no_cache_prompt_only(self):
        """
        Test that cache_mode "ram_only" includes only --no-cache-prompt flag.
        
        **Validates: Requirements 4.2**
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock files
            model_file = temp_path / "model.gguf"
            model_file.write_text("mock model")
            server_binary = temp_path / "llama-server"
            server_binary.write_text("#!/bin/bash\necho 'mock server'")
            server_binary.chmod(0o755)
            
            # Mock subprocess and HTTP operations
            with patch('subprocess.Popen') as mock_popen, \
                 patch('requests.Session'), \
                 patch.object(NativeLlamaServer, '_wait_for_health_check'), \
                 patch.object(NativeLlamaServer, '_start_memory_monitoring'):
                
                mock_process = MagicMock()
                mock_process.pid = 12345
                mock_process.poll.return_value = None
                mock_popen.return_value = mock_process
                
                server = NativeLlamaServer(
                    model_path=str(model_file),
                    cache_mode="ram_only",
                    llama_server_path=str(server_binary)
                )
                
                cmd = server._build_server_command()
                
                # Verify only --no-cache-prompt is present
                assert "--no-cache-prompt" in cmd
                assert "--cache-ram" not in cmd
                
                server.close()
    
    def test_cache_mode_disk_only_includes_cache_ram_zero_only(self):
        """
        Test that cache_mode "disk_only" includes only --cache-ram 0 flag.
        
        **Validates: Requirements 4.3**
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock files
            model_file = temp_path / "model.gguf"
            model_file.write_text("mock model")
            server_binary = temp_path / "llama-server"
            server_binary.write_text("#!/bin/bash\necho 'mock server'")
            server_binary.chmod(0o755)
            
            # Mock subprocess and HTTP operations
            with patch('subprocess.Popen') as mock_popen, \
                 patch('requests.Session'), \
                 patch.object(NativeLlamaServer, '_wait_for_health_check'), \
                 patch.object(NativeLlamaServer, '_start_memory_monitoring'):
                
                mock_process = MagicMock()
                mock_process.pid = 12345
                mock_process.poll.return_value = None
                mock_popen.return_value = mock_process
                
                server = NativeLlamaServer(
                    model_path=str(model_file),
                    cache_mode="disk_only",
                    llama_server_path=str(server_binary)
                )
                
                cmd = server._build_server_command()
                
                # Verify only --cache-ram 0 is present
                assert "--cache-ram" in cmd
                assert "--no-cache-prompt" not in cmd
                
                # Verify --cache-ram is set to 0
                cache_ram_idx = cmd.index("--cache-ram")
                assert cmd[cache_ram_idx + 1] == "0"
                
                server.close()
    
    def test_cache_mode_both_includes_no_cache_flags(self):
        """
        Test that cache_mode "both" includes no cache restriction flags.
        
        **Validates: Requirements 4.4**
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock files
            model_file = temp_path / "model.gguf"
            model_file.write_text("mock model")
            server_binary = temp_path / "llama-server"
            server_binary.write_text("#!/bin/bash\necho 'mock server'")
            server_binary.chmod(0o755)
            
            # Mock subprocess and HTTP operations
            with patch('subprocess.Popen') as mock_popen, \
                 patch('requests.Session'), \
                 patch.object(NativeLlamaServer, '_wait_for_health_check'), \
                 patch.object(NativeLlamaServer, '_start_memory_monitoring'):
                
                mock_process = MagicMock()
                mock_process.pid = 12345
                mock_process.poll.return_value = None
                mock_popen.return_value = mock_process
                
                server = NativeLlamaServer(
                    model_path=str(model_file),
                    cache_mode="both",
                    llama_server_path=str(server_binary)
                )
                
                cmd = server._build_server_command()
                
                # Verify no cache restriction flags are present
                assert "--cache-ram" not in cmd
                assert "--no-cache-prompt" not in cmd
                
                server.close()
    
    def test_command_includes_all_required_parameters(self):
        """
        Test that the command includes all required parameters in correct format.
        
        **Validates: Requirements 2.2**
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock files
            model_file = temp_path / "model.gguf"
            model_file.write_text("mock model")
            server_binary = temp_path / "llama-server"
            server_binary.write_text("#!/bin/bash\necho 'mock server'")
            server_binary.chmod(0o755)
            
            # Mock subprocess and HTTP operations
            with patch('subprocess.Popen') as mock_popen, \
                 patch('requests.Session'), \
                 patch.object(NativeLlamaServer, '_wait_for_health_check'), \
                 patch.object(NativeLlamaServer, '_start_memory_monitoring'):
                
                mock_process = MagicMock()
                mock_process.pid = 12345
                mock_process.poll.return_value = None
                mock_popen.return_value = mock_process
                
                server = NativeLlamaServer(
                    model_path=str(model_file),
                    n_ctx=2048,
                    n_threads=4,
                    n_batch=512,
                    host="127.0.0.1",
                    port=8080,
                    llama_server_path=str(server_binary)
                )
                
                cmd = server._build_server_command()
                
                # Verify command structure
                expected_params = [
                    str(server_binary),
                    "-m", str(model_file),
                    "-c", "2048",
                    "-t", "4", 
                    "-b", "512",
                    "--host", "127.0.0.1",
                    "--port", "8080"
                ]
                
                # Check that all expected parameters are in the command
                for i in range(0, len(expected_params), 2):
                    if i + 1 < len(expected_params):
                        # Parameter with value
                        param = expected_params[i]
                        value = expected_params[i + 1]
                        assert param in cmd, f"Parameter {param} should be in command"
                        if param != str(server_binary):  # Skip binary path check
                            param_idx = cmd.index(param)
                            assert cmd[param_idx + 1] == value, f"Parameter {param} should have value {value}"
                    else:
                        # Single parameter
                        param = expected_params[i]
                        assert param in cmd, f"Parameter {param} should be in command"
                
                server.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])