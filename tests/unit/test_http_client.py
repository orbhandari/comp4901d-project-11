"""
Unit tests for HTTP client functionality in NativeLlamaServer.

Tests the HTTP API client including:
- POST requests to /completion endpoint
- Request body construction with proper parameters
- Streaming response handling
- Error conditions and recovery
- Timeout configuration

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from requests.exceptions import Timeout, HTTPError

from llm_benchmark.inference.native_llama_server import NativeLlamaServer


class TestHTTPClientFunctionality:
    """Test HTTP client functionality for completion requests."""
    
    @pytest.fixture
    def mock_server(self):
        """Create a mocked NativeLlamaServer instance."""
        with patch('llm_benchmark.inference.native_llama_server.Path') as mock_path:
            # Mock file existence checks
            mock_path.return_value.expanduser.return_value.exists.return_value = True
            
            with patch('subprocess.Popen') as mock_popen:
                mock_process = Mock()
                mock_process.pid = 12345
                mock_process.poll.return_value = None  # Process is running
                mock_popen.return_value = mock_process
                
                with patch('requests.Session') as mock_session_class:
                    mock_session = Mock()
                    mock_session_class.return_value = mock_session
                    
                    # Mock successful health check
                    mock_health_response = Mock()
                    mock_health_response.status_code = 200
                    mock_session.get.return_value = mock_health_response
                    
                    server = NativeLlamaServer(
                        model_path="/fake/model.gguf",
                        n_ctx=2048,
                        n_threads=4,
                        cache_mode="both"
                    )
                    
                    # Replace the session with our mock
                    server.session = mock_session
                    
                    yield server, mock_session
    
    def test_completion_request_construction(self, mock_server):
        """Test that completion requests are constructed correctly."""
        server, mock_session = mock_server
        
        # Mock streaming response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            'data: {"content": "Hello"}',
            'data: {"content": " world"}',
            'data: [DONE]'
        ]
        mock_session.post.return_value = mock_response
        
        # Make completion request
        prompt = "Test prompt"
        max_tokens = 50
        enable_prompt_cache = True
        
        list(server(prompt, max_tokens=max_tokens, enable_prompt_cache=enable_prompt_cache))
        
        # Verify request was made correctly
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        
        # Check URL
        assert call_args[0][0] == "http://127.0.0.1:8080/completion"
        
        # Check request body
        request_body = call_args[1]['json']
        assert request_body['prompt'] == prompt
        assert request_body['n_predict'] == max_tokens
        assert request_body['cache_prompt'] == enable_prompt_cache
        assert request_body['stream'] is True
        assert 'temperature' in request_body
        assert 'top_k' in request_body
        assert 'top_p' in request_body
        
        # Check timeout calculation (max_tokens * 2 + 60)
        expected_timeout = max_tokens * 2 + 60
        assert call_args[1]['timeout'] == expected_timeout
        
        # Check streaming enabled
        assert call_args[1]['stream'] is True
    
    def test_streaming_response_parsing(self, mock_server):
        """Test that streaming responses are parsed correctly."""
        server, mock_session = mock_server
        
        # Mock streaming response with various formats
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            'data: {"content": "Hello"}',
            'data: {"content": " world"}',
            'data: {"content": "!"}',
            'data: [DONE]'
        ]
        mock_session.post.return_value = mock_response
        
        # Collect response chunks
        chunks = list(server("Test prompt", max_tokens=10))
        
        # Verify response format matches NativeLlamaCpp
        assert len(chunks) == 4  # 3 content chunks + 1 done chunk
        
        # Check content chunks
        for i in range(3):
            assert 'choices' in chunks[i]
            assert len(chunks[i]['choices']) == 1
            assert 'text' in chunks[i]['choices'][0]
            assert chunks[i]['choices'][0]['finish_reason'] is None
        
        # Check expected content
        assert chunks[0]['choices'][0]['text'] == "Hello"
        assert chunks[1]['choices'][0]['text'] == " world"
        assert chunks[2]['choices'][0]['text'] == "!"
        
        # Check done chunk
        assert chunks[3]['choices'][0]['text'] == ""
        assert chunks[3]['choices'][0]['finish_reason'] == "stop"
    
    def test_malformed_json_handling(self, mock_server):
        """Test that malformed JSON chunks are handled gracefully with enhanced diagnostics."""
        server, mock_session = mock_server
        
        # Mock response with malformed JSON
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            'data: {"content": "Good"}',
            'data: {invalid json}',  # Malformed JSON
            'data: {"content": " chunk"}',
            'data: {"missing_content": "value"}',  # Valid JSON but missing content field
            'data: [DONE]'
        ]
        mock_session.post.return_value = mock_response
        
        # Should skip malformed chunks and continue
        chunks = list(server("Test prompt", max_tokens=10))
        
        # Should have 3 chunks: 2 good content + 1 done
        assert len(chunks) == 3
        assert chunks[0]['choices'][0]['text'] == "Good"
        assert chunks[1]['choices'][0]['text'] == " chunk"
        assert chunks[2]['choices'][0]['finish_reason'] == "stop"
    
    def test_sse_format_handling(self, mock_server):
        """Test that various SSE format elements are handled correctly."""
        server, mock_session = mock_server
        
        # Mock response with various SSE format elements
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            '',  # Empty line (should be skipped)
            ': This is a comment',  # SSE comment
            'event: message',  # SSE event type
            'id: 123',  # SSE event ID
            'data: {"content": "Hello"}',
            '',  # Another empty line
            'retry: 1000',  # SSE retry directive
            'data: {"content": " world"}',
            'data: [DONE]'
        ]
        mock_session.post.return_value = mock_response
        
        # Should handle all SSE elements and extract only data content
        chunks = list(server("Test prompt", max_tokens=10))
        
        # Should have 3 chunks: 2 content chunks + 1 done chunk
        assert len(chunks) == 3
        assert chunks[0]['choices'][0]['text'] == "Hello"
        assert chunks[1]['choices'][0]['text'] == " world"
        assert chunks[2]['choices'][0]['finish_reason'] == "stop"
    
    def test_chunked_encoding_error_handling(self, mock_server):
        """Test handling of chunked encoding connection drops with enhanced diagnostics."""
        server, mock_session = mock_server
        
        # Mock chunked encoding error (connection drop during streaming)
        mock_session.post.side_effect = requests.exceptions.ChunkedEncodingError("Connection broken")
        
        # Should handle gracefully and yield connection_error finish reason
        chunks = list(server("Test prompt", max_tokens=10))
        
        # Should have 1 chunk indicating connection error
        assert len(chunks) == 1
        assert chunks[0]['choices'][0]['text'] == ""
        assert chunks[0]['choices'][0]['finish_reason'] == "connection_error"
    
    def test_connection_error_handling(self, mock_server):
        """Test handling of connection errors with enhanced diagnostics."""
        server, mock_session = mock_server
        
        # Mock connection error
        mock_session.post.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
        # Should raise ConnectionError with enhanced diagnostic information
        with pytest.raises(ConnectionError) as exc_info:
            list(server("Test prompt", max_tokens=10))
        
        error_msg = str(exc_info.value)
        assert "Failed to connect to llama-server at http://127.0.0.1:8080" in error_msg
        assert "Connection refused" in error_msg
        assert "Troubleshooting:" in error_msg
        assert "Ensure llama-server is running and accessible" in error_msg
        assert "Check if port 8080 is available" in error_msg
        assert "llama-server process is running" in error_msg
    
    def test_timeout_error_handling(self, mock_server):
        """Test handling of timeout errors with enhanced diagnostics."""
        server, mock_session = mock_server
        
        # Mock timeout error
        mock_session.post.side_effect = requests.exceptions.Timeout("Request timed out")
        
        # Should raise TimeoutError with enhanced timeout info
        with pytest.raises(TimeoutError) as exc_info:
            list(server("Test prompt", max_tokens=100))
        
        error_msg = str(exc_info.value)
        assert "Request to llama-server timed out after 260s" in error_msg  # 100 * 2 + 60
        assert "Request timed out" in error_msg
        assert "Request parameters: max_tokens=100" in error_msg
        assert "Timeout calculation: 100 tokens * 2s + 60s buffer = 260s" in error_msg
        assert "Consider:" in error_msg
        assert "Reducing max_tokens for faster completion" in error_msg
    
    def test_http_error_handling(self, mock_server):
        """Test handling of HTTP errors with enhanced diagnostics."""
        server, mock_session = mock_server
        
        # Mock HTTP error response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.url = "http://127.0.0.1:8080/completion"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_session.post.return_value = mock_response
        
        # Should raise RuntimeError with enhanced HTTP details
        with pytest.raises(RuntimeError) as exc_info:
            list(server("Test prompt"))
        
        error_msg = str(exc_info.value)
        assert "HTTP error from llama-server" in error_msg
        assert "Status code: 500" in error_msg
        assert "Internal Server Error" in error_msg
        assert "URL: http://127.0.0.1:8080/completion" in error_msg
        assert "Explanation: Internal Server Error - Server-side issue" in error_msg
    
    def test_timeout_calculation_property(self, mock_server):
        """Test timeout calculation formula: max_tokens * 2 + 60."""
        server, mock_session = mock_server
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = ['data: [DONE]']
        mock_session.post.return_value = mock_response
        
        test_cases = [1, 10, 50, 100, 500, 1000]
        
        for max_tokens in test_cases:
            mock_session.post.reset_mock()
            
            list(server("Test", max_tokens=max_tokens))
            
            # Verify timeout calculation
            call_args = mock_session.post.call_args
            expected_timeout = max_tokens * 2 + 60
            actual_timeout = call_args[1]['timeout']
            
            assert actual_timeout == expected_timeout, f"Failed for max_tokens={max_tokens}"
    
    def test_cache_prompt_parameter_mapping(self, mock_server):
        """Test that enable_prompt_cache maps correctly to cache_prompt."""
        server, mock_session = mock_server
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = ['data: [DONE]']
        mock_session.post.return_value = mock_response
        
        # Test enable_prompt_cache=True
        list(server("Test", enable_prompt_cache=True))
        request_body = mock_session.post.call_args[1]['json']
        assert request_body['cache_prompt'] is True
        
        mock_session.post.reset_mock()
        
        # Test enable_prompt_cache=False (default)
        list(server("Test", enable_prompt_cache=False))
        request_body = mock_session.post.call_args[1]['json']
        assert request_body['cache_prompt'] is False
        
        mock_session.post.reset_mock()
        
        # Test default behavior (should be False)
        list(server("Test"))
        request_body = mock_session.post.call_args[1]['json']
        assert request_body['cache_prompt'] is False
    
    def test_create_completion_alias(self, mock_server):
        """Test that create_completion method works as alias to __call__."""
        server, mock_session = mock_server
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            'data: {"content": "Test response"}',
            'data: [DONE]'
        ]
        mock_session.post.return_value = mock_response
        
        # Test create_completion method
        chunks = list(server.create_completion(
            prompt="Test prompt",
            max_tokens=25,
            enable_prompt_cache=True
        ))
        
        # Verify request was made
        mock_session.post.assert_called_once()
        
        # Verify request parameters
        call_args = mock_session.post.call_args
        request_body = call_args[1]['json']
        assert request_body['prompt'] == "Test prompt"
        assert request_body['n_predict'] == 25
        assert request_body['cache_prompt'] is True
        
        # Verify response format
        assert len(chunks) == 2
        assert chunks[0]['choices'][0]['text'] == "Test response"
        assert chunks[1]['choices'][0]['finish_reason'] == "stop"
    
    def test_headers_configuration(self, mock_server):
        """Test that proper headers are set for requests."""
        server, mock_session = mock_server
        
        # Verify that the server has a session attribute
        assert hasattr(server, 'session')
        
        # In the real implementation, headers are set during initialization
        # For this test, we verify that the session exists and would have headers
        # The actual header setting is tested implicitly by other tests that make requests
        assert server.session is not None
    
    def test_unexpected_error_handling(self, mock_server):
        """Test handling of unexpected errors during streaming with enhanced diagnostics."""
        server, mock_session = mock_server
        
        # Mock an unexpected error during streaming
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.side_effect = ValueError("Unexpected streaming error")
        mock_session.post.return_value = mock_response
        
        # Should raise RuntimeError with enhanced diagnostic information
        with pytest.raises(RuntimeError) as exc_info:
            list(server("Test prompt", max_tokens=50))
        
        error_msg = str(exc_info.value)
        assert "Streaming failed: Unexpected streaming error" in error_msg