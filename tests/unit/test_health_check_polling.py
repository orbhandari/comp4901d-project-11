"""
Unit tests for health check polling system in NativeLlamaServer.

Tests the health check polling functionality including:
- Successful health checks
- Timeout handling
- Process status verification
- Exponential backoff retry strategy
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import requests
from llm_benchmark.inference.native_llama_server import NativeLlamaServer


class TestHealthCheckPolling:
    """Test health check polling system functionality."""
    
    def test_successful_health_check(self):
        """Test successful health check completes quickly."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session') as mock_session_class, \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch('pathlib.Path.exists', return_value=True):
            
            # Setup mocks
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_process.poll.return_value = None  # Process is running
            mock_popen.return_value = mock_process
            
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            
            # Mock successful health check response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_session.get.return_value = mock_response
            
            # Create server instance (this will call _wait_for_health_check)
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            )
            
            # Verify health check was called
            mock_session.get.assert_called_with(
                "http://127.0.0.1:8080/health",
                timeout=1.0
            )
    
    def test_health_check_timeout_raises_error(self):
        """Test health check timeout raises TimeoutError with diagnostic info."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session') as mock_session_class, \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch.object(NativeLlamaServer, '_cleanup_process'), \
             patch('pathlib.Path.exists', return_value=True):
            
            # Setup mocks
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_process.poll.return_value = None  # Process is running
            mock_popen.return_value = mock_process
            
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            
            # Mock health check always fails (connection error)
            mock_session.get.side_effect = requests.exceptions.ConnectionError("Connection refused")
            
            # Mock time to speed up test
            with patch('time.time') as mock_time, \
                 patch('time.sleep'):
                # Simulate timeout after 30 seconds
                mock_time.side_effect = [0, 31]  # Start time, then timeout
                
                with pytest.raises(RuntimeError) as exc_info:
                    NativeLlamaServer(
                        model_path="/fake/model.gguf",
                        llama_server_path="/fake/llama-server"
                    )
                
                # Verify error message contains diagnostic information
                error_msg = str(exc_info.value)
                assert "llama-server startup failed" in error_msg
                assert "health check timed out after 30s" in error_msg
    
    def test_process_termination_during_health_check(self):
        """Test process termination detection during health check."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session') as mock_session_class, \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch.object(NativeLlamaServer, '_cleanup_process'), \
             patch('pathlib.Path.exists', return_value=True):
            
            # Setup mocks
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_process.returncode = 1
            mock_process.communicate.return_value = ("stdout", "stderr output")
            mock_popen.return_value = mock_process
            
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            
            # First call: process running, second call: process terminated
            mock_process.poll.side_effect = [None, 1]
            
            with pytest.raises(RuntimeError) as exc_info:
                NativeLlamaServer(
                    model_path="/fake/model.gguf",
                    llama_server_path="/fake/llama-server"
                )
            
            # Verify error message contains process information
            error_msg = str(exc_info.value)
            assert "llama-server process terminated during startup" in error_msg
            assert "Return code: 1" in error_msg
            assert "stderr: stderr output" in error_msg
    
    def test_exponential_backoff_retry_strategy(self):
        """Test exponential backoff retry strategy is implemented."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session') as mock_session_class, \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch('pathlib.Path.exists', return_value=True):
            
            # Setup mocks
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_process.poll.return_value = None  # Process is running
            mock_popen.return_value = mock_process
            
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            
            # Mock health check fails twice, then succeeds
            mock_response_success = MagicMock()
            mock_response_success.status_code = 200
            
            mock_session.get.side_effect = [
                requests.exceptions.ConnectionError("Connection refused"),
                requests.exceptions.ConnectionError("Connection refused"),
                mock_response_success
            ]
            
            # Track sleep calls to verify exponential backoff
            sleep_calls = []
            with patch('time.sleep') as mock_sleep:
                mock_sleep.side_effect = lambda x: sleep_calls.append(x)
                
                # Create server instance
                server = NativeLlamaServer(
                    model_path="/fake/model.gguf",
                    llama_server_path="/fake/llama-server"
                )
                
                # Verify exponential backoff pattern
                assert len(sleep_calls) >= 2
                # First retry should be 0.1s, second should be approximately 0.15s (0.1 * 1.5)
                assert sleep_calls[0] == 0.1
                assert abs(sleep_calls[1] - 0.15) < 0.001  # Allow for floating point precision
    
    def test_health_check_with_custom_timeout(self):
        """Test health check with custom timeout value."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session') as mock_session_class, \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch.object(NativeLlamaServer, '_cleanup_process'):
            
            # Setup mocks
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process
            
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.get.side_effect = requests.exceptions.ConnectionError("Connection refused")
            
            # Test with custom timeout by directly calling _wait_for_health_check
            server = object.__new__(NativeLlamaServer)  # Create without __init__
            server.process = mock_process
            server.session = mock_session
            server.base_url = "http://127.0.0.1:8080"
            server._cleanup_process = Mock()
            
            # Mock time to simulate custom timeout
            with patch('time.time') as mock_time, \
                 patch('time.sleep'):
                mock_time.side_effect = [0, 16]  # Start time, then 15s timeout + 1s
                
                with pytest.raises(TimeoutError) as exc_info:
                    server._wait_for_health_check(timeout=15)
                
                # Verify custom timeout is reflected in error message
                error_msg = str(exc_info.value)
                assert "timed out after 15s" in error_msg