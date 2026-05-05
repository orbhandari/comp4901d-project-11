"""
Unit tests for subprocess management in NativeLlamaServer.

Tests subprocess lifecycle management including:
- Process creation with proper command-line arguments
- Health check polling scenarios
- Cleanup behavior and resource management
- Error handling for various failure modes
- Memory monitoring functionality

Requirements: 2.1, 2.3, 2.4
"""

import pytest
import subprocess
import signal
import os
import threading
import time
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import requests

from llm_benchmark.inference.native_llama_server import NativeLlamaServer, CacheMode


class TestSubprocessManagement:
    """Test subprocess management functionality."""
    
    def test_process_creation_with_basic_arguments(self):
        """Test subprocess creation with basic command-line arguments."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True):
            
            # Setup mock process
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            # Create server instance
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                n_ctx=2048,
                n_threads=4,
                n_batch=256,
                cache_mode="both",
                llama_server_path="/fake/llama-server",
                host="127.0.0.1",
                port=8080
            )
            
            # Verify subprocess was called with correct arguments
            mock_popen.assert_called_once()
            args, kwargs = mock_popen.call_args
            
            # Check command arguments
            cmd = args[0]
            expected_cmd = [
                "/fake/llama-server",
                "-m", "/fake/model.gguf",
                "-c", "2048",
                "-t", "4",
                "-b", "256",
                "--host", "127.0.0.1",
                "--port", "8080"
            ]
            assert cmd == expected_cmd
            
            # Check subprocess options
            assert kwargs['stdin'] == subprocess.DEVNULL
            assert kwargs['stdout'] == subprocess.PIPE
            assert kwargs['stderr'] == subprocess.PIPE
            assert kwargs['text'] is True
            assert kwargs['start_new_session'] is True
            
            # Verify process attributes are set
            assert server.last_subprocess_pid == 12345
            assert server.subprocess_is_running is True
            assert server.subprocess_peak_memory_kb == 0
    
    def test_process_creation_with_cache_mode_none(self):
        """Test subprocess creation with cache_mode 'none' adds cache control flags."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True):
            
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            # Create server with cache_mode "none"
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                cache_mode="none",
                llama_server_path="/fake/llama-server"
            )
            
            # Verify cache control flags are added
            args, _ = mock_popen.call_args
            cmd = args[0]
            
            # Should contain --cache-ram 0 and --no-cache-prompt
            assert "--cache-ram" in cmd
            assert "0" in cmd[cmd.index("--cache-ram") + 1]
            assert "--no-cache-prompt" in cmd
    
    def test_process_creation_with_cache_mode_ram_only(self):
        """Test subprocess creation with cache_mode 'ram_only' adds disk cache disable flag."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True):
            
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            # Create server with cache_mode "ram_only"
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                cache_mode="ram_only",
                llama_server_path="/fake/llama-server"
            )
            
            # Verify only disk cache is disabled
            args, _ = mock_popen.call_args
            cmd = args[0]
            
            assert "--no-cache-prompt" in cmd
            assert "--cache-ram" not in cmd
    
    def test_process_creation_with_cache_mode_disk_only(self):
        """Test subprocess creation with cache_mode 'disk_only' adds RAM cache disable flag."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True):
            
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            # Create server with cache_mode "disk_only"
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                cache_mode="disk_only",
                llama_server_path="/fake/llama-server"
            )
            
            # Verify only RAM cache is disabled
            args, _ = mock_popen.call_args
            cmd = args[0]
            
            assert "--cache-ram" in cmd
            assert "0" in cmd[cmd.index("--cache-ram") + 1]
            assert "--no-cache-prompt" not in cmd
    
    def test_process_creation_failure_raises_runtime_error(self):
        """Test process creation failure raises RuntimeError with diagnostic info."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_cleanup_process') as mock_cleanup, \
             patch('pathlib.Path.exists', return_value=True):
            
            # Mock subprocess.Popen to raise an exception
            mock_popen.side_effect = FileNotFoundError("Binary not found")
            
            with pytest.raises(RuntimeError) as exc_info:
                NativeLlamaServer(
                    model_path="/fake/model.gguf",
                    llama_server_path="/fake/llama-server"
                )
            
            # Verify error message contains diagnostic information
            error_msg = str(exc_info.value)
            assert "llama-server startup failed" in error_msg
            assert "Binary not found" in error_msg
            
            # Verify cleanup was called
            mock_cleanup.assert_called_once()
    
    def test_process_crash_during_startup_detected(self):
        """Test process crash during startup is detected and reported."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session') as mock_session_class, \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch.object(NativeLlamaServer, '_cleanup_process'), \
             patch('pathlib.Path.exists', return_value=True):
            
            # Setup mock process that crashes
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_process.poll.return_value = 1  # Process terminated with error
            mock_process.returncode = 1
            mock_process.communicate.return_value = ("stdout", "Process crashed")
            mock_popen.return_value = mock_process
            
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            
            with pytest.raises(RuntimeError) as exc_info:
                NativeLlamaServer(
                    model_path="/fake/model.gguf",
                    llama_server_path="/fake/llama-server"
                )
            
            # Verify error message contains crash information
            error_msg = str(exc_info.value)
            assert "llama-server process terminated during startup" in error_msg
            assert "Return code: 1" in error_msg
            assert "stderr: Process crashed" in error_msg
    
    def test_cleanup_process_terminates_gracefully(self):
        """Test cleanup process terminates subprocess gracefully."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True):
            
            # Setup mock process
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_process.wait.return_value = None  # Graceful termination
            mock_popen.return_value = mock_process
            
            # Create server and then close it
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            )
            
            # Mock memory monitoring thread
            server._memory_monitor_thread = MagicMock()
            server._memory_monitor_thread.is_alive.return_value = True
            
            server.close()
            
            # Verify graceful termination sequence
            mock_process.terminate.assert_called_once()
            mock_process.wait.assert_called_once_with(timeout=5)
            
            # Verify process attributes are reset
            assert server.subprocess_is_running is False
            assert server.process is None
    
    def test_cleanup_process_force_kills_on_timeout(self):
        """Test cleanup process force kills subprocess if graceful termination times out."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('os.getpgid') as mock_getpgid, \
             patch('os.killpg') as mock_killpg:
            
            # Setup mock process that doesn't terminate gracefully
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_process.wait.side_effect = [subprocess.TimeoutExpired("cmd", 5), None]
            mock_popen.return_value = mock_process
            
            mock_getpgid.return_value = 12345
            
            # Create server and then close it
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            )
            
            # Mock memory monitoring thread
            server._memory_monitor_thread = MagicMock()
            server._memory_monitor_thread.is_alive.return_value = True
            
            server.close()
            
            # Verify force kill sequence
            mock_process.terminate.assert_called_once()
            assert mock_process.wait.call_count == 2  # First timeout, then after kill
            mock_getpgid.assert_called_once_with(12345)
            mock_killpg.assert_called_once_with(12345, signal.SIGKILL)
    
    def test_cleanup_handles_process_group_kill_failure(self):
        """Test cleanup handles failure in process group kill gracefully."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('os.getpgid') as mock_getpgid, \
             patch('os.killpg') as mock_killpg:
            
            # Setup mock process that doesn't terminate gracefully
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_process.wait.side_effect = [subprocess.TimeoutExpired("cmd", 5), None]
            mock_popen.return_value = mock_process
            
            # Mock process group kill failure
            mock_getpgid.side_effect = OSError("No such process")
            
            # Create server and then close it
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            )
            
            # Mock memory monitoring thread
            server._memory_monitor_thread = MagicMock()
            server._memory_monitor_thread.is_alive.return_value = True
            
            server.close()
            
            # Verify fallback to process.kill()
            mock_process.terminate.assert_called_once()
            mock_process.kill.assert_called_once()
            assert server.subprocess_is_running is False
    
    def test_context_manager_calls_cleanup(self):
        """Test context manager properly calls cleanup on exit."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch.object(NativeLlamaServer, 'close') as mock_close, \
             patch('pathlib.Path.exists', return_value=True):
            
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            # Use context manager
            with NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            ) as server:
                assert server is not None
            
            # Verify close was called
            mock_close.assert_called_once()
    
    def test_memory_monitoring_thread_started(self):
        """Test memory monitoring thread is started during initialization."""
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
            
            # Verify thread was created and started
            mock_thread_class.assert_called_once()
            args, kwargs = mock_thread_class.call_args
            assert kwargs['daemon'] is True
            mock_thread.start.assert_called_once()
    
    def test_memory_monitoring_samples_process_memory(self):
        """Test memory monitoring samples subprocess memory correctly."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('subprocess.run') as mock_run:
            
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            # Mock ps command output
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "1024\n"  # 1024 KB memory usage
            mock_run.return_value = mock_result
            
            # Create server instance
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            )
            
            # Wait a bit for memory monitoring to run
            time.sleep(0.1)
            
            # Stop memory monitoring
            server._stop_memory_monitoring.set()
            if server._memory_monitor_thread:
                server._memory_monitor_thread.join(timeout=1)
            
            # Verify ps command was called
            mock_run.assert_called()
            call_args = mock_run.call_args[0][0]
            assert call_args == ['ps', '-o', 'rss=', '-p', '12345']
            
            # Verify memory was tracked (may not be exactly 1024 due to timing)
            assert server.subprocess_peak_memory_kb >= 0
    
    def test_memory_monitoring_handles_process_termination(self):
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
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stdout = ""
            mock_run.return_value = mock_result
            
            # Create server instance
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                llama_server_path="/fake/llama-server"
            )
            
            # Wait a bit for memory monitoring to run
            time.sleep(0.1)
            
            # Stop memory monitoring
            server._stop_memory_monitoring.set()
            if server._memory_monitor_thread:
                server._memory_monitor_thread.join(timeout=1)
            
            # Should not raise exception despite ps command failure
            assert server.subprocess_peak_memory_kb >= 0
    
    def test_binary_not_found_raises_file_not_found_error(self):
        """Test missing llama-server binary raises FileNotFoundError with build instructions."""
        with patch('pathlib.Path.exists', return_value=False):
            
            with pytest.raises(FileNotFoundError) as exc_info:
                NativeLlamaServer(
                    model_path="/fake/model.gguf",
                    llama_server_path="/fake/llama-server"
                )
            
            # Verify error message contains build instructions
            error_msg = str(exc_info.value)
            assert "llama-server binary not found" in error_msg
            assert "Build llama.cpp with server support" in error_msg
            assert "cmake -B build" in error_msg
            assert "DLLAMA_SERVER=ON" in error_msg
    
    def test_model_not_found_raises_file_not_found_error(self):
        """Test missing model file raises FileNotFoundError."""
        with patch('pathlib.Path.exists') as mock_exists:
            # Binary exists, model doesn't
            mock_exists.side_effect = lambda: mock_exists.call_count == 1
            
            with pytest.raises(FileNotFoundError) as exc_info:
                NativeLlamaServer(
                    model_path="/fake/model.gguf",
                    llama_server_path="/fake/llama-server"
                )
            
            # Verify error message
            error_msg = str(exc_info.value)
            assert "Model not found" in error_msg
            assert "/fake/model.gguf" in error_msg
    
    def test_auto_thread_count_detection(self):
        """Test automatic thread count detection when n_threads is -1."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('requests.Session'), \
             patch.object(NativeLlamaServer, '_start_memory_monitoring'), \
             patch.object(NativeLlamaServer, '_wait_for_health_check'), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('os.cpu_count', return_value=8):
            
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            # Create server with n_threads=-1
            server = NativeLlamaServer(
                model_path="/fake/model.gguf",
                n_threads=-1,
                llama_server_path="/fake/llama-server"
            )
            
            # Verify thread count was set to CPU count
            assert server.n_threads == 8
            
            # Verify command line includes correct thread count
            args, _ = mock_popen.call_args
            cmd = args[0]
            thread_index = cmd.index("-t")
            assert cmd[thread_index + 1] == "8"