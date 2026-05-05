"""
Property Tests for NativeLlamaServer HTTP Request Construction

**Validates: Requirements 3.2, 5.1, 5.2**

This test verifies that the NativeLlamaServer correctly constructs HTTP request bodies
for completion requests, ensuring all required fields are included with correct values
and proper cache_prompt parameter mapping.

Property 3: HTTP Request Body Construction
For any valid completion parameters (prompt, max_tokens, enable_prompt_cache),
the NativeLlamaServer should construct HTTP request bodies that include all required
fields with correct values and proper cache_prompt parameter mapping.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings, assume

from llm_benchmark.inference.native_llama_server import NativeLlamaServer, CacheMode


class TestNativeLlamaServerHttpRequestProperties:
    """
    Property 3: HTTP Request Body Construction
    
    **Validates: Requirements 3.2, 5.1, 5.2**
    
    Tests that for any valid completion parameters (prompt, max_tokens, enable_prompt_cache),
    the NativeLlamaServer constructs HTTP request bodies that include all required fields
    with correct values and proper cache_prompt parameter mapping.
    """
    
    @given(
        prompt=st.text(min_size=1, max_size=1000),
        max_tokens=st.integers(min_value=1, max_value=2048),
        enable_prompt_cache=st.booleans()
    )
    @settings(max_examples=100, deadline=None)
    def test_property_http_request_body_construction(
        self, prompt, max_tokens, enable_prompt_cache
    ):
        """
        Property 3: HTTP Request Body Construction
        
        **Validates: Requirements 3.2, 5.1, 5.2**
        
        For any valid completion parameters, the NativeLlamaServer should construct
        HTTP request bodies that include all required fields with correct values
        and proper cache_prompt parameter mapping.
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
                
                # Call the method to trigger HTTP request construction
                list(server(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    enable_prompt_cache=enable_prompt_cache
                ))
                
                # Verify that post was called
                assert mock_session.post.called, "HTTP POST request should be made"
                
                # Get the call arguments
                call_args, call_kwargs = mock_session.post.call_args
                
                # Verify URL
                expected_url = f"{server.base_url}/completion"
                assert call_args[0] == expected_url, f"URL should be {expected_url}"
                
                # Verify request body structure
                assert 'json' in call_kwargs, "Request should include JSON body"
                request_body = call_kwargs['json']
                
                # Verify all required fields are present
                required_fields = ['prompt', 'n_predict', 'stream', 'cache_prompt', 'temperature', 'top_k', 'top_p']
                for field in required_fields:
                    assert field in request_body, f"Request body should include '{field}' field"
                
                # Verify field values
                assert request_body['prompt'] == prompt, f"Prompt should be '{prompt}'"
                assert request_body['n_predict'] == max_tokens, f"n_predict should be {max_tokens}"
                assert request_body['stream'] is True, "stream should always be True for compatibility"
                assert request_body['cache_prompt'] == enable_prompt_cache, f"cache_prompt should be {enable_prompt_cache}"
                
                # Verify default parameter values
                assert request_body['temperature'] == 0.8, "temperature should default to 0.8"
                assert request_body['top_k'] == 40, "top_k should default to 40"
                assert request_body['top_p'] == 0.9, "top_p should default to 0.9"
                
                # Verify timeout calculation
                expected_timeout = max_tokens * 2 + 60
                assert call_kwargs['timeout'] == expected_timeout, f"Timeout should be {expected_timeout}"
                
                # Verify streaming is enabled
                assert call_kwargs['stream'] is True, "HTTP request should use streaming"
                
                # Verify headers
                assert 'headers' in call_kwargs, "Request should include headers"
                headers = call_kwargs['headers']
                assert headers['Accept'] == 'text/event-stream', "Should request SSE format"
                assert headers['Cache-Control'] == 'no-cache', "Should prevent response caching"
                
                # Clean up
                server.close()
    
    def test_cache_prompt_false_when_enable_prompt_cache_false(self):
        """
        Test that cache_prompt is set to false when enable_prompt_cache is false.
        
        **Validates: Requirements 5.1**
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
                
                # Call with enable_prompt_cache=False
                list(server(
                    prompt="test prompt",
                    max_tokens=100,
                    enable_prompt_cache=False
                ))
                
                # Verify cache_prompt is false
                call_kwargs = mock_session.post.call_args[1]
                request_body = call_kwargs['json']
                assert request_body['cache_prompt'] is False, "cache_prompt should be False when enable_prompt_cache is False"
                
                server.close()
    
    def test_cache_prompt_true_when_enable_prompt_cache_true(self):
        """
        Test that cache_prompt is set to true when enable_prompt_cache is true.
        
        **Validates: Requirements 5.2**
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
                
                # Call with enable_prompt_cache=True
                list(server(
                    prompt="test prompt",
                    max_tokens=100,
                    enable_prompt_cache=True
                ))
                
                # Verify cache_prompt is true
                call_kwargs = mock_session.post.call_args[1]
                request_body = call_kwargs['json']
                assert request_body['cache_prompt'] is True, "cache_prompt should be True when enable_prompt_cache is True"
                
                server.close()
    
    def test_default_cache_prompt_false(self):
        """
        Test that cache_prompt defaults to false to prevent unintended caching.
        
        **Validates: Requirements 5.3**
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
                
                # Call without specifying enable_prompt_cache (should default to False)
                list(server(
                    prompt="test prompt",
                    max_tokens=100
                ))
                
                # Verify cache_prompt defaults to false
                call_kwargs = mock_session.post.call_args[1]
                request_body = call_kwargs['json']
                assert request_body['cache_prompt'] is False, "cache_prompt should default to False"
                
                server.close()
    
    def test_request_body_includes_all_required_fields(self):
        """
        Test that the request body includes all required fields for llama-server API.
        
        **Validates: Requirements 3.2**
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
                
                # Call with specific parameters
                list(server(
                    prompt="Hello, world!",
                    max_tokens=256,
                    enable_prompt_cache=True
                ))
                
                # Verify request body structure
                call_kwargs = mock_session.post.call_args[1]
                request_body = call_kwargs['json']
                
                # Check all required fields are present with correct types
                assert isinstance(request_body['prompt'], str), "prompt should be a string"
                assert isinstance(request_body['n_predict'], int), "n_predict should be an integer"
                assert isinstance(request_body['stream'], bool), "stream should be a boolean"
                assert isinstance(request_body['cache_prompt'], bool), "cache_prompt should be a boolean"
                assert isinstance(request_body['temperature'], (int, float)), "temperature should be numeric"
                assert isinstance(request_body['top_k'], int), "top_k should be an integer"
                assert isinstance(request_body['top_p'], (int, float)), "top_p should be numeric"
                
                # Verify specific values
                assert request_body['prompt'] == "Hello, world!"
                assert request_body['n_predict'] == 256
                assert request_body['stream'] is True
                assert request_body['cache_prompt'] is True
                
                server.close()
    
    def test_timeout_calculation_formula(self):
        """
        Test that timeout is calculated using the formula (max_tokens * 2 + 60) seconds.
        
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
                
                # Test with different max_tokens values
                test_cases = [1, 50, 100, 500, 1000, 2048]
                
                for max_tokens in test_cases:
                    # Reset mock to clear previous calls
                    mock_session.post.reset_mock()
                    
                    list(server(
                        prompt="test",
                        max_tokens=max_tokens
                    ))
                    
                    # Verify timeout calculation
                    call_kwargs = mock_session.post.call_args[1]
                    expected_timeout = max_tokens * 2 + 60
                    actual_timeout = call_kwargs['timeout']
                    
                    assert actual_timeout == expected_timeout, \
                        f"For max_tokens={max_tokens}, timeout should be {expected_timeout}, got {actual_timeout}"
                
                server.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])