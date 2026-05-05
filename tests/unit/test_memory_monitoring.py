"""
Unit tests for memory monitoring functionality in NativeLlamaServer.

Tests memory monitoring background thread behavior, compatibility attributes,
and tokenize method accuracy as specified in task 5.3.

Requirements: 14.1, 14.2, 14.3, 14.4
"""

import pytest
import threading
import time
import subprocess
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path

from llm_benchmark.inference.native_llama_server import NativeLlamaServer, CacheMode


class TestMemoryMonitoring:
    """Test memory monitoring functionality."""
    
    def test_memory_monitoring_thread_initialization(self):
        """Test memory monitoring thread is properly initialized."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('threading.Thread') as mock_thread_class:
            
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            mock_thread = MagicMock()
            mock_thread_class.return_value = mock_thread
            
            # Create server instance
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            )
            
            # Verify thread was created with correct parameters
            mock_thread_class.assert_called_once()
            args, kwargs = mock_thread_class.call_args
            
            # Check thread configuration
            assert 'target' in kwargs
            assert kwargs['daemon'] is True
            
            # Verify thread was started
            mock_thread.start.assert_called_once()
            
            # Verify memory monitoring attributes are initialized
            assert hasattr(server, '_memory_monitor_thread')
            assert hasattr(server, '_stop_memory_monitoring')
            assert isinstance(server._stop_memory_monitoring, threading.Event)
    
    def test_memory_monitoring_samples_every_50ms(self):
        """Test memory monitoring samples process memory every 50ms."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('subprocess.run') as mock_run, \
             patch('time.sleep') as mock_sleep:
            
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            # Mock ps command output with varying memory usage
            mock_results = [
                MagicMock(returncode=0, stdout="1024\n"),  # 1024 KB
                MagicMock(returncode=0, stdout="2048\n"),  # 2048 KB
                MagicMock(returncode=0, stdout="1536\n"),  # 1536 KB
            ]
            mock_run.side_effect = mock_results
            
            # Create server instance
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            )
            
            # Let monitoring run for a few cycles
            time.sleep(0.2)
            
            # Stop memory monitoring
            server._stop_memory_monitoring.set()
            if server._memory_monitor_thread:
                server._memory_monitor_thread.join(timeout=1)
            
            # Verify ps command was called multiple times
            assert mock_run.call_count >= 1
            
            # Verify sleep was called with 0.05 seconds (50ms)
            sleep_calls = [call for call in mock_sleep.call_args_list if call[0][0] == 0.05]
            assert len(sleep_calls) >= 1
            
            # Verify ps command format
            for call_args in mock_run.call_args_list:
                cmd = call_args[0][0]
                assert cmd == ['ps', '-o', 'rss=', '-p', '12345']
                
                # Verify command options
                kwargs = call_args[1]
                assert kwargs['capture_output'] is True
                assert kwargs['text'] is True
                assert kwargs['timeout'] == 0.1
    
    def test_memory_monitoring_tracks_peak_memory(self):
        """Test memory monitoring correctly tracks peak memory usage."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('subprocess.run') as mock_run:
            
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            # Mock ps command output with increasing then decreasing memory
            memory_values = ["1024", "2048", "3072", "1536", "2560"]
            mock_results = [
                MagicMock(returncode=0, stdout=f"{val}\n") 
                for val in memory_values
            ]
            mock_run.side_effect = mock_results
            
            # Create server instance
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            )
            
            # Let monitoring run and collect samples
            time.sleep(0.3)
            
            # Stop memory monitoring
            server._stop_memory_monitoring.set()
            if server._memory_monitor_thread:
                server._memory_monitor_thread.join(timeout=1)
            
            # Peak should be the maximum value seen (3072 KB)
            assert server.subprocess_peak_memory_kb >= 3072
    
    def test_memory_monitoring_handles_process_not_found(self):
        """Test memory monitoring handles process termination gracefully."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('subprocess.run') as mock_run:
            
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            # Mock ps command failure (process not found)
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            
            # Create server instance
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            )
            
            # Let monitoring run
            time.sleep(0.1)
            
            # Stop memory monitoring
            server._stop_memory_monitoring.set()
            if server._memory_monitor_thread:
                server._memory_monitor_thread.join(timeout=1)
            
            # Should not raise exception and memory should remain at initial value
            assert server.subprocess_peak_memory_kb == 0
    
    def test_memory_monitoring_handles_ps_command_exception(self):
        """Test memory monitoring handles ps command exceptions gracefully."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('subprocess.run') as mock_run:
            
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            # Mock ps command exception
            mock_run.side_effect = subprocess.TimeoutExpired("ps", 0.1)
            
            # Create server instance
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            )
            
            # Let monitoring run
            time.sleep(0.1)
            
            # Stop memory monitoring
            server._stop_memory_monitoring.set()
            if server._memory_monitor_thread:
                server._memory_monitor_thread.join(timeout=1)
            
            # Should not raise exception and memory should remain at initial value
            assert server.subprocess_peak_memory_kb == 0
    
    def test_memory_monitoring_stops_when_subprocess_not_running(self):
        """Test memory monitoring stops when subprocess_is_running is False."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('subprocess.run') as mock_run:
            
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            mock_run.return_value = MagicMock(returncode=0, stdout="1024\n")
            
            # Create server instance
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            )
            
            # Let monitoring start
            time.sleep(0.05)
            
            # Set subprocess_is_running to False
            server.subprocess_is_running = False
            
            # Wait for monitoring to stop
            time.sleep(0.1)
            
            # Verify monitoring thread stops
            if server._memory_monitor_thread:
                server._memory_monitor_thread.join(timeout=1)
                assert not server._memory_monitor_thread.is_alive()
    
    def test_memory_monitoring_stops_on_stop_event(self):
        """Test memory monitoring stops when stop event is set."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('subprocess.run') as mock_run:
            
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            mock_run.return_value = MagicMock(returncode=0, stdout="1024\n")
            
            # Create server instance
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            )
            
            # Let monitoring start
            time.sleep(0.05)
            
            # Set stop event
            server._stop_memory_monitoring.set()
            
            # Wait for monitoring to stop
            if server._memory_monitor_thread:
                server._memory_monitor_thread.join(timeout=1)
                assert not server._memory_monitor_thread.is_alive()


class TestCompatibilityAttributes:
    """Test compatibility attributes for memory measurement."""
    
    def test_last_subprocess_pid_attribute(self):
        """Test last_subprocess_pid attribute is set correctly."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True):
            
            mock_process = MagicMock()
            mock_process.pid = 54321
            mock_popen.return_value = mock_process
            
            # Create server instance
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            )
            
            # Verify PID is stored correctly
            assert server.last_subprocess_pid == 54321
            assert hasattr(server, 'last_subprocess_pid')
    
    def test_subprocess_is_running_flag(self):
        """Test subprocess_is_running flag is managed correctly."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True):
            
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            # Create server instance
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            )
            
            # Initially should be running
            assert server.subprocess_is_running is True
            
            # After close, should be False
            server.subprocess_is_running = False
            assert server.subprocess_is_running is False
    
    def test_subprocess_peak_memory_kb_attribute(self):
        """Test subprocess_peak_memory_kb attribute is initialized and updated."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True):
            
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            # Create server instance
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            )
            
            # Initially should be 0
            assert server.subprocess_peak_memory_kb == 0
            assert hasattr(server, 'subprocess_peak_memory_kb')
            
            # Should be updatable
            server.subprocess_peak_memory_kb = 2048
            assert server.subprocess_peak_memory_kb == 2048
    
    def test_compatibility_with_native_llama_cpp_interface(self):
        """Test compatibility attributes match NativeLlamaCpp interface."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True):
            
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            # Create server instance
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            )
            
            # Verify all required compatibility attributes exist
            required_attributes = [
                'last_subprocess_pid',
                'subprocess_is_running', 
                'subprocess_peak_memory_kb'
            ]
            
            for attr in required_attributes:
                assert hasattr(server, attr), f"Missing compatibility attribute: {attr}"
    
    def test_memory_monitoring_updates_peak_memory_attribute(self):
        """Test memory monitoring updates subprocess_peak_memory_kb correctly."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('subprocess.run') as mock_run:
            
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            # Mock ps command with specific memory value
            mock_run.return_value = MagicMock(returncode=0, stdout="4096\n")
            
            # Create server instance
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            )
            
            # Let monitoring run
            time.sleep(0.1)
            
            # Stop memory monitoring
            server._stop_memory_monitoring.set()
            if server._memory_monitor_thread:
                server._memory_monitor_thread.join(timeout=1)
            
            # Verify peak memory was updated
            assert server.subprocess_peak_memory_kb >= 4096


class TestTokenizeMethod:
    """Test tokenize method accuracy."""
    
    def test_tokenize_string_input(self):
        """Test tokenize method with string input."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True):
            
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            # Create server instance
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            )
            
            # Test with simple string
            text = "Hello world"
            tokens = server.tokenize(text)
            
            # Should return list of integers
            assert isinstance(tokens, list)
            assert all(isinstance(token, int) for token in tokens)
            
            # Length should be approximately char_count / 4
            expected_length = max(1, len(text) // 4)
            assert len(tokens) == expected_length
            
            # Tokens should be sequential integers starting from 0
            assert tokens == list(range(len(tokens)))
    
    def test_tokenize_bytes_input(self):
        """Test tokenize method with bytes input."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True):
            
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            # Create server instance
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            )
            
            # Test with bytes input
            text = b"Hello world"
            tokens = server.tokenize(text)
            
            # Should return list of integers
            assert isinstance(tokens, list)
            assert all(isinstance(token, int) for token in tokens)
            
            # Length should be approximately byte_count / 4
            expected_length = max(1, len(text) // 4)
            assert len(tokens) == expected_length
    
    def test_tokenize_empty_string(self):
        """Test tokenize method with empty string."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True):
            
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            # Create server instance
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            )
            
            # Test with empty string
            tokens = server.tokenize("")
            
            # Should return at least one token (minimum is 1)
            assert isinstance(tokens, list)
            assert len(tokens) == 1
            assert tokens == [0]
    
    def test_tokenize_long_text(self):
        """Test tokenize method with long text."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True):
            
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            # Create server instance
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            )
            
            # Test with long text (100 characters)
            text = "A" * 100
            tokens = server.tokenize(text)
            
            # Should return list with length = char_count / 4
            expected_length = 100 // 4  # 25 tokens
            assert len(tokens) == expected_length
            assert tokens == list(range(expected_length))
    
    def test_tokenize_unicode_text(self):
        """Test tokenize method with Unicode text."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True):
            
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            # Create server instance
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            )
            
            # Test with Unicode text
            text = "Hello 世界 🌍"
            tokens = server.tokenize(text)
            
            # Should handle UTF-8 encoding correctly
            byte_count = len(text.encode('utf-8'))
            expected_length = max(1, byte_count // 4)
            
            assert isinstance(tokens, list)
            assert len(tokens) == expected_length
            assert all(isinstance(token, int) for token in tokens)
    
    def test_tokenize_approximation_accuracy(self):
        """Test tokenize method approximation is reasonably accurate."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True):
            
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            # Create server instance
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            )
            
            # Test various text lengths
            test_cases = [
                ("Hi", 1),      # 2 chars -> 0 tokens -> min 1 token
                ("Hello", 1),   # 5 chars -> 1 token
                ("Hello world", 2),  # 11 chars -> 2 tokens
                ("This is a longer sentence", 6),  # 26 chars -> 6 tokens
                ("A" * 20, 5),  # 20 chars -> 5 tokens
            ]
            
            for text, expected_tokens in test_cases:
                tokens = server.tokenize(text)
                assert len(tokens) == expected_tokens, \
                    f"Text '{text}' ({len(text)} chars) should produce {expected_tokens} tokens, got {len(tokens)}"
    
    def test_tokenize_return_format(self):
        """Test tokenize method returns correct format."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True):
            
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            # Create server instance
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            )
            
            # Test return format
            text = "Test tokenization"
            tokens = server.tokenize(text)
            
            # Should return List[int] as specified in interface
            assert isinstance(tokens, list)
            assert all(isinstance(token, int) for token in tokens)
            
            # Tokens should be non-negative integers
            assert all(token >= 0 for token in tokens)
            
            # Should be sequential starting from 0
            assert tokens == list(range(len(tokens)))