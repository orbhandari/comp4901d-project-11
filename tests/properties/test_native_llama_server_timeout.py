"""
Property Tests for NativeLlamaServer Timeout Calculation

**Validates: Requirements 3.7**

This test verifies that the NativeLlamaServer correctly calculates request timeout
using the formula (max_tokens * 2 + 60) seconds, ensuring adequate time for
completion generation.

Property 4: Timeout Calculation
For any positive max_tokens value, the NativeLlamaServer should calculate request
timeout using the formula (max_tokens * 2 + 60) seconds, ensuring adequate time
for completion generation.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings, assume

from llm_benchmark.inference.native_llama_server import NativeLlamaServer, CacheMode


class TestNativeLlamaServerTimeoutProperties:
    """
    Property 4: Timeout Calculation
    
    **Validates: Requirements 3.7**
    
    Tests that for any positive max_tokens value, the NativeLlamaServer calculates
    request timeout using the formula (max_tokens * 2 + 60) seconds, ensuring
    adequate time for completion generation.
    """
    
    @given(
        max_tokens=st.integers(min_value=1, max_value=10000)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_timeout_calculation_formula(self, max_tokens):
        """
        Property 4: Timeout Calculation
        
        **Validates: Requirements 3.7**
        
        For any positive max_tokens value, the NativeLlamaServer should calculate
        request timeout using the formula (max_tokens * 2 + 60) seconds.
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
                 patch('requests.Session') as mock_session_class, \
                 patch.object(NativeLlamaServer, '_wait_for_health_check'), \
                 patch.object(NativeLlamaServer, '_start_memory_monitoring'):
                
                # Mock the process
                mock_process = MagicMock()
                mock_process.pid = 12345
                mock_process.poll.return_value = None  # Process is running
                mock_popen.return_value = mock_process
                
                # Mock the session and response
                mock_session = MagicMock()
                mock_session_class.return_value = mock_session
                
                # Create a mock response that simulates streaming
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.headers = {
                    'content-type': 'text/event-stream',
                    'transfer-encoding': 'chunked'
                }
                mock_response.iter_lines.return_value = [
                    'data: {"content": "test"}',
                    'data: [DONE]'
                ]
                mock_session.post.return_value = mock_response
                
                # Create NativeLlamaServer instance
                server = NativeLlamaServer(
                    model_path=str(model_file),
                    llama_server_path=str(server_binary)
                )
                
                # Call the method to trigger timeout calculation
                list(server(
                    prompt="test prompt",
                    max_tokens=max_tokens,
                    enable_prompt_cache=False
                ))
                
                # Verify that post was called
                assert mock_session.post.called, "HTTP POST request should be made"
                
                # Get the call arguments
                call_args, call_kwargs = mock_session.post.call_args
                
                # Verify timeout calculation using the formula: max_tokens * 2 + 60
                expected_timeout = max_tokens * 2 + 60
                actual_timeout = call_kwargs['timeout']
                
                assert actual_timeout == expected_timeout, \
                    f"For max_tokens={max_tokens}, timeout should be {expected_timeout} seconds " \
                    f"(formula: {max_tokens} * 2 + 60), but got {actual_timeout} seconds"
                
                # Verify timeout is always positive and reasonable
                assert actual_timeout > 0, "Timeout should always be positive"
                assert actual_timeout >= 62, "Timeout should be at least 62 seconds (1 * 2 + 60)"
                
                # Clean up
                server.close()
    
    def test_timeout_calculation_edge_cases(self):
        """
        Test timeout calculation for specific edge cases to ensure correctness.
        
        **Validates: Requirements 3.7**
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
                 patch('requests.Session') as mock_session_class, \
                 patch.object(NativeLlamaServer, '_wait_for_health_check'), \
                 patch.object(NativeLlamaServer, '_start_memory_monitoring'):
                
                mock_process = MagicMock()
                mock_process.pid = 12345
                mock_process.poll.return_value = None
                mock_popen.return_value = mock_process
                
                mock_session = MagicMock()
                mock_session_class.return_value = mock_session
                
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.headers = {'content-type': 'text/event-stream'}
                mock_response.iter_lines.return_value = ['data: [DONE]']
                mock_session.post.return_value = mock_response
                
                server = NativeLlamaServer(
                    model_path=str(model_file),
                    llama_server_path=str(server_binary)
                )
                
                # Test specific edge cases
                test_cases = [
                    (1, 62),      # Minimum: 1 * 2 + 60 = 62
                    (10, 80),     # Small: 10 * 2 + 60 = 80
                    (100, 260),   # Medium: 100 * 2 + 60 = 260
                    (500, 1060),  # Large: 500 * 2 + 60 = 1060
                    (1000, 2060), # Very large: 1000 * 2 + 60 = 2060
                    (2048, 4156), # Max typical: 2048 * 2 + 60 = 4156
                ]
                
                for max_tokens, expected_timeout in test_cases:
                    # Reset mock to clear previous calls
                    mock_session.post.reset_mock()
                    
                    list(server(
                        prompt="test",
                        max_tokens=max_tokens
                    ))
                    
                    # Verify timeout calculation
                    call_kwargs = mock_session.post.call_args[1]
                    actual_timeout = call_kwargs['timeout']
                    
                    assert actual_timeout == expected_timeout, \
                        f"For max_tokens={max_tokens}, expected timeout {expected_timeout}s, got {actual_timeout}s"
                
                server.close()
    
    def test_timeout_calculation_consistency(self):
        """
        Test that timeout calculation is consistent across multiple calls with same parameters.
        
        **Validates: Requirements 3.7**
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
                 patch('requests.Session') as mock_session_class, \
                 patch.object(NativeLlamaServer, '_wait_for_health_check'), \
                 patch.object(NativeLlamaServer, '_start_memory_monitoring'):
                
                mock_process = MagicMock()
                mock_process.pid = 12345
                mock_process.poll.return_value = None
                mock_popen.return_value = mock_process
                
                mock_session = MagicMock()
                mock_session_class.return_value = mock_session
                
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.headers = {'content-type': 'text/event-stream'}
                mock_response.iter_lines.return_value = ['data: [DONE]']
                mock_session.post.return_value = mock_response
                
                server = NativeLlamaServer(
                    model_path=str(model_file),
                    llama_server_path=str(server_binary)
                )
                
                max_tokens = 256
                expected_timeout = max_tokens * 2 + 60  # 256 * 2 + 60 = 572
                
                # Make multiple calls with same parameters
                timeouts = []
                for i in range(5):
                    mock_session.post.reset_mock()
                    
                    list(server(
                        prompt=f"test prompt {i}",
                        max_tokens=max_tokens
                    ))
                    
                    call_kwargs = mock_session.post.call_args[1]
                    timeouts.append(call_kwargs['timeout'])
                
                # Verify all timeouts are identical
                assert all(t == expected_timeout for t in timeouts), \
                    f"All timeouts should be {expected_timeout}, got {timeouts}"
                
                # Verify consistency
                assert len(set(timeouts)) == 1, "All timeout calculations should be identical"
                
                server.close()
    
    def test_timeout_calculation_independent_of_other_parameters(self):
        """
        Test that timeout calculation depends only on max_tokens, not other parameters.
        
        **Validates: Requirements 3.7**
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
                 patch('requests.Session') as mock_session_class, \
                 patch.object(NativeLlamaServer, '_wait_for_health_check'), \
                 patch.object(NativeLlamaServer, '_start_memory_monitoring'):
                
                mock_process = MagicMock()
                mock_process.pid = 12345
                mock_process.poll.return_value = None
                mock_popen.return_value = mock_process
                
                mock_session = MagicMock()
                mock_session_class.return_value = mock_session
                
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.headers = {'content-type': 'text/event-stream'}
                mock_response.iter_lines.return_value = ['data: [DONE]']
                mock_session.post.return_value = mock_response
                
                server = NativeLlamaServer(
                    model_path=str(model_file),
                    llama_server_path=str(server_binary)
                )
                
                max_tokens = 128
                expected_timeout = max_tokens * 2 + 60  # 128 * 2 + 60 = 316
                
                # Test with different combinations of other parameters
                test_scenarios = [
                    {
                        "prompt": "short",
                        "enable_prompt_cache": False
                    },
                    {
                        "prompt": "This is a much longer prompt that should not affect the timeout calculation at all",
                        "enable_prompt_cache": True
                    },
                    {
                        "prompt": "Another different prompt with special characters: !@#$%^&*()",
                        "enable_prompt_cache": False
                    }
                ]
                
                timeouts = []
                for scenario in test_scenarios:
                    mock_session.post.reset_mock()
                    
                    list(server(
                        max_tokens=max_tokens,
                        **scenario
                    ))
                    
                    call_kwargs = mock_session.post.call_args[1]
                    timeouts.append(call_kwargs['timeout'])
                
                # Verify all timeouts are identical regardless of other parameters
                assert all(t == expected_timeout for t in timeouts), \
                    f"All timeouts should be {expected_timeout} regardless of other parameters, got {timeouts}"
                
                server.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])