"""
Native llama-server wrapper for Android ablation studies.

Uses subprocess to manage llama-server binary and HTTP API client for inference.
This enables proper cache control for accurate ablation measurements.
"""

import logging
import os
import signal
import subprocess
import sys
import time
import threading
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, List
import requests
import json

logger = logging.getLogger(__name__)


class CacheMode(Enum):
    """Cache mode configuration for llama-server."""
    NONE = "none"          # --cache-ram 0 --no-cache-prompt
    RAM_ONLY = "ram_only"  # --no-cache-prompt
    DISK_ONLY = "disk_only" # --cache-ram 0
    BOTH = "both"          # No restrictions


# Ablation scenario cache configuration mapping
# Maps ablation scenarios to appropriate cache settings for accurate measurements
ABLATION_CACHE_CONFIG = {
    "control": {
        "cache_mode": CacheMode.NONE,
        "enable_prompt_cache": False,
        "description": "No caching - true baseline measurement"
    },
    "cold_cache": {
        "cache_mode": CacheMode.BOTH,
        "enable_prompt_cache": True,
        "description": "Cache enabled but empty (first request)"
    },
    "warm_cache": {
        "cache_mode": CacheMode.BOTH,
        "enable_prompt_cache": True,
        "description": "Cache populated and reused (subsequent requests)"
    },
    "ram_only": {
        "cache_mode": CacheMode.RAM_ONLY,
        "enable_prompt_cache": False,
        "description": "RAM-based KV cache only, no disk caching"
    },
    "disk_only": {
        "cache_mode": CacheMode.DISK_ONLY,
        "enable_prompt_cache": True,
        "description": "Disk-based prompt cache only, no RAM KV cache"
    }
}


def validate_cache_mode(cache_mode: str) -> CacheMode:
    """
    Validate and convert cache mode string to CacheMode enum.
    
    Args:
        cache_mode: Cache mode string to validate
        
    Returns:
        CacheMode enum value
        
    Raises:
        ValueError: If cache_mode is not a valid cache mode
    """
    try:
        return CacheMode(cache_mode)
    except ValueError:
        valid_modes = [mode.value for mode in CacheMode]
        raise ValueError(
            f"Invalid cache_mode '{cache_mode}'. "
            f"Valid options: {valid_modes}"
        )


def get_ablation_cache_config(scenario: str) -> Dict[str, Any]:
    """
    Get cache configuration for a specific ablation scenario.
    
    Args:
        scenario: Ablation scenario name
        
    Returns:
        Dictionary containing cache_mode and enable_prompt_cache settings
        
    Raises:
        ValueError: If scenario is not a valid ablation scenario
    """
    if scenario not in ABLATION_CACHE_CONFIG:
        valid_scenarios = list(ABLATION_CACHE_CONFIG.keys())
        raise ValueError(
            f"Invalid ablation scenario '{scenario}'. "
            f"Valid options: {valid_scenarios}"
        )
    
    return ABLATION_CACHE_CONFIG[scenario].copy()


def validate_ablation_scenario(scenario: str) -> str:
    """
    Validate ablation scenario name.
    
    Args:
        scenario: Scenario name to validate
        
    Returns:
        Validated scenario name
        
    Raises:
        ValueError: If scenario is not valid
    """
    if scenario not in ABLATION_CACHE_CONFIG:
        valid_scenarios = list(ABLATION_CACHE_CONFIG.keys())
        raise ValueError(
            f"Invalid ablation scenario '{scenario}'. "
            f"Valid options: {valid_scenarios}"
        )
    
    return scenario


class NativeLlamaServer:
    """Wrapper around native llama-server for Android ablation studies."""
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = -1,
        n_batch: int = 512,
        cache_mode: str = "both",
        llama_server_path: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        **kwargs
    ):
        """
        Initialize native llama-server wrapper.
        
        Args:
            model_path: Path to GGUF model file
            n_ctx: Context size
            n_threads: Number of threads (-1 for auto)
            n_batch: Batch size for prompt processing
            cache_mode: Cache configuration ("none", "ram_only", "disk_only", "both")
            llama_server_path: Path to llama-server binary
            host: Server host address
            port: Server port number
            **kwargs: Additional arguments (ignored, for compatibility)
        
        Note on Cache Control:
        ----------------------
        - cache_mode "none": Disables both RAM and disk caching (--cache-ram 0 --no-cache-prompt)
        - cache_mode "ram_only": Disables disk caching only (--no-cache-prompt)
        - cache_mode "disk_only": Disables RAM caching only (--cache-ram 0)
        - cache_mode "both": Enables all caching (default)
        - Per-request cache control available via enable_prompt_cache parameter
        """
        self.model_path = Path(model_path).expanduser()
        self.n_ctx = n_ctx
        self.n_threads = n_threads if n_threads > 0 else os.cpu_count()
        self.n_batch = n_batch
        self.cache_mode = validate_cache_mode(cache_mode)
        self.host = host
        self.port = port
        
        # Subprocess management attributes
        self.process = None
        self.last_subprocess_pid = None  # Track subprocess PID for memory measurement
        self.subprocess_is_running = False  # Track if subprocess is currently active
        self.subprocess_peak_memory_kb = 0  # Track peak memory during subprocess execution
        
        # HTTP client setup
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Memory monitoring thread
        self._memory_monitor_thread = None
        self._stop_memory_monitoring = threading.Event()
        
        # Find llama-server binary
        if llama_server_path is None:
            llama_server_path = "~/llama.cpp/build/bin/llama-server"
        
        self.llama_server_path = Path(llama_server_path).expanduser()
        
        if not self.llama_server_path.exists():
            raise FileNotFoundError(
                f"llama-server binary not found: {self.llama_server_path}\n"
                "Build llama.cpp with server support:\n"
                "  cd ~/llama.cpp\n"
                "  cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_SERVER=ON\n"
                "  cmake --build build --config Release -j4"
            )
        
        # Verify model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        logger.info(f"Initialized NativeLlamaServer with binary: {self.llama_server_path}")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Configuration: n_ctx={n_ctx}, n_threads={self.n_threads}, n_batch={n_batch}")
        logger.info(f"Cache mode: {self.cache_mode.value}")
        logger.info(f"Server endpoint: {self.base_url}")
        
        # Start llama-server subprocess
        self._start_server()
    
    def _build_server_command(self) -> List[str]:
        """Build command-line arguments for llama-server."""
        cmd = [
            str(self.llama_server_path),
            "-m", str(self.model_path),
            "-c", str(self.n_ctx),
            "-t", str(self.n_threads),
            "-b", str(self.n_batch),
            "--host", self.host,
            "--port", str(self.port),
        ]
        
        # Add cache control flags based on cache_mode
        cache_flags = []
        if self.cache_mode == CacheMode.NONE:
            cache_flags.extend(["--cache-ram", "0", "--no-cache-prompt"])
            logger.info("Cache mode: NONE - disabling both RAM and disk caching")
        elif self.cache_mode == CacheMode.RAM_ONLY:
            cache_flags.append("--no-cache-prompt")
            logger.info("Cache mode: RAM_ONLY - disabling disk caching only")
        elif self.cache_mode == CacheMode.DISK_ONLY:
            cache_flags.extend(["--cache-ram", "0"])
            logger.info("Cache mode: DISK_ONLY - disabling RAM caching only")
        elif self.cache_mode == CacheMode.BOTH:
            logger.info("Cache mode: BOTH - enabling all caching")
        
        cmd.extend(cache_flags)
        
        # Log the complete command for debugging
        logger.info(f"llama-server command: {' '.join(cmd)}")
        if cache_flags:
            logger.info(f"Cache control flags: {' '.join(cache_flags)}")
        else:
            logger.info("No cache control flags (default caching enabled)")
        
        return cmd
    
    def _start_server(self):
        """Start llama-server subprocess."""
        cmd = self._build_server_command()
        
        logger.debug(f"Starting llama-server: {' '.join(cmd)}")
        
        try:
            # Start subprocess with new session for process group management
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True
            )
            
            # Store subprocess information
            self.last_subprocess_pid = self.process.pid
            self.subprocess_is_running = True
            self.subprocess_peak_memory_kb = 0
            
            logger.info(f"Started llama-server subprocess PID: {self.last_subprocess_pid}")
            
            # Start memory monitoring thread
            self._start_memory_monitoring()
            
            # Wait for server to be ready
            self._wait_for_health_check()
            
        except Exception as e:
            logger.error(f"Failed to start llama-server: {e}")
            self._cleanup_process()
            raise RuntimeError(f"llama-server startup failed: {e}")
    
    def _start_memory_monitoring(self):
        """Start background thread to monitor subprocess memory."""
        def monitor_memory():
            """Monitor subprocess memory in background."""
            import psutil
            
            # Get psutil process object for more reliable monitoring
            try:
                psutil_process = psutil.Process(self.last_subprocess_pid)
                logger.debug(f"Memory monitoring started for PID {self.last_subprocess_pid}")
            except psutil.NoSuchProcess:
                logger.warning(f"Process {self.last_subprocess_pid} not found for memory monitoring")
                return
            
            while not self._stop_memory_monitoring.is_set() and self.subprocess_is_running:
                try:
                    if psutil_process.is_running():
                        # Get memory info using psutil (more reliable than ps command)
                        memory_info = psutil_process.memory_info()
                        rss_kb = memory_info.rss // 1024  # Convert bytes to KB
                        
                        # Update peak memory
                        old_peak = self.subprocess_peak_memory_kb
                        self.subprocess_peak_memory_kb = max(self.subprocess_peak_memory_kb, rss_kb)
                        
                        # Log significant memory changes for debugging
                        if rss_kb > old_peak + 10240:  # Log if memory increased by >10MB
                            logger.debug(f"Memory increased: {old_peak//1024}MB -> {rss_kb//1024}MB")
                    else:
                        # Process has terminated
                        logger.debug("llama-server process terminated, stopping memory monitoring")
                        self.subprocess_is_running = False
                        break
                        
                except psutil.NoSuchProcess:
                    # Process terminated
                    logger.debug("llama-server process no longer exists")
                    self.subprocess_is_running = False
                    break
                except Exception as e:
                    logger.debug(f"Memory monitoring error: {e}")
                    # Continue monitoring despite errors
                
                time.sleep(0.05)  # Sample every 50ms
            
            logger.debug(f"Memory monitoring stopped. Peak memory: {self.subprocess_peak_memory_kb//1024}MB")
        
        self._memory_monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        self._memory_monitor_thread.start()
    
    def _wait_for_health_check(self, timeout: int = 30):
        """Wait for llama-server to be ready by polling health endpoint."""
        start_time = time.time()
        retry_delay = 0.1
        max_retry_delay = 2.0
        
        while time.time() - start_time < timeout:
            try:
                # Check if process is still running
                if self.process.poll() is not None:
                    stdout, stderr = self.process.communicate()
                    raise RuntimeError(
                        f"llama-server process terminated during startup\n"
                        f"Return code: {self.process.returncode}\n"
                        f"stderr: {stderr}"
                    )
                
                # Try health check
                response = self.session.get(
                    f"{self.base_url}/health",
                    timeout=1.0
                )
                
                if response.status_code == 200:
                    logger.info("llama-server health check passed - server is ready")
                    return
                
            except requests.exceptions.RequestException:
                # Server not ready yet, continue waiting
                pass
            
            # Exponential backoff with jitter
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 1.5, max_retry_delay)
        
        # Timeout reached
        self._cleanup_process()
        raise TimeoutError(
            f"llama-server health check timed out after {timeout}s\n"
            f"Server may have failed to start or is not responding at {self.base_url}/health"
        )
    
    def _cleanup_process(self):
        """Clean up llama-server subprocess and resources."""
        # Stop memory monitoring
        self._stop_memory_monitoring.set()
        if self._memory_monitor_thread and self._memory_monitor_thread.is_alive():
            self._memory_monitor_thread.join(timeout=1)
        
        # Terminate subprocess
        if self.process:
            try:
                # Try graceful termination first
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful termination fails
                    logger.warning("Graceful termination failed, force killing llama-server")
                    try:
                        pgid = os.getpgid(self.process.pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except Exception:
                        self.process.kill()
                    self.process.wait()
                
                logger.info(f"Terminated llama-server subprocess PID: {self.last_subprocess_pid}")
                
            except Exception as e:
                logger.warning(f"Error during subprocess cleanup: {e}")
            
            finally:
                self.subprocess_is_running = False
                self.process = None
    
    def __call__(
        self,
        prompt: str,
        max_tokens: int = 128,
        stream: bool = True,
        enable_prompt_cache: bool = False,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """
        Generate text using llama-server HTTP API.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            stream: Whether to stream output (always True for compatibility)
            enable_prompt_cache: Whether to enable prompt caching for this request
            **kwargs: Additional arguments (ignored)
        
        Yields:
            Dictionary with 'choices' containing generated text chunks
        """
        # Build request body
        request_body = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "stream": True,  # Always stream for compatibility
            "cache_prompt": enable_prompt_cache,
            "temperature": 0.8,
            "top_k": 40,
            "top_p": 0.9
        }
        
        # Calculate timeout: 2 seconds per token + 60s buffer
        timeout_seconds = max_tokens * 2 + 60
        
        # Enhanced logging for debugging cache behavior
        logger.info(f"=== llama-server HTTP Request ===")
        logger.info(f"Endpoint: {self.base_url}/completion")
        logger.info(f"Cache mode (server): {self.cache_mode.value}")
        logger.info(f"Cache prompt (request): {enable_prompt_cache}")
        logger.info(f"Prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)")
        logger.info(f"Max tokens: {max_tokens}")
        logger.info(f"Timeout: {timeout_seconds}s")
        logger.info(f"Request body cache_prompt: {request_body['cache_prompt']}")
        
        # Log cache configuration summary
        cache_summary = []
        if self.cache_mode == CacheMode.NONE:
            cache_summary.append("RAM cache: DISABLED (--cache-ram 0)")
            cache_summary.append("Disk cache: DISABLED (--no-cache-prompt)")
        elif self.cache_mode == CacheMode.RAM_ONLY:
            cache_summary.append("RAM cache: ENABLED")
            cache_summary.append("Disk cache: DISABLED (--no-cache-prompt)")
        elif self.cache_mode == CacheMode.DISK_ONLY:
            cache_summary.append("RAM cache: DISABLED (--cache-ram 0)")
            cache_summary.append("Disk cache: ENABLED")
        elif self.cache_mode == CacheMode.BOTH:
            cache_summary.append("RAM cache: ENABLED")
            cache_summary.append("Disk cache: ENABLED")
        
        cache_summary.append(f"Request cache_prompt: {enable_prompt_cache}")
        logger.info("Cache configuration: " + ", ".join(cache_summary))
        
        try:
            response = self.session.post(
                f"{self.base_url}/completion",
                json=request_body,
                timeout=timeout_seconds,
                stream=True,
                headers={
                    'Accept': 'text/event-stream',  # Explicitly request SSE format
                    'Cache-Control': 'no-cache'     # Prevent caching of streaming responses
                }
            )
            
            response.raise_for_status()
            
            # Verify we got the expected content type for SSE
            content_type = response.headers.get('content-type', '')
            # Handle both real strings and Mock objects in tests
            if hasattr(content_type, '__contains__'):
                if 'text/event-stream' not in content_type and 'text/plain' not in content_type:
                    logger.warning(f"Unexpected content type for streaming: {content_type}")
            
            # Check for chunked transfer encoding
            transfer_encoding = response.headers.get('transfer-encoding', '')
            # Handle both real strings and Mock objects in tests
            if hasattr(transfer_encoding, 'lower') and hasattr(transfer_encoding, '__contains__'):
                if 'chunked' in transfer_encoding.lower():
                    logger.debug("Using chunked transfer encoding for streaming response")
            
            # Parse streaming response (Server-Sent Events format)
            # Handle chunked transfer encoding by processing line by line
            for line in response.iter_lines(decode_unicode=True, chunk_size=1024):
                # Skip empty lines (common in SSE format)
                if not line.strip():
                    continue
                
                # Handle SSE format: data:, event:, id:, retry:, or comments
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    
                    # Check for end of stream marker
                    if data_str.strip() == "[DONE]":
                        # End of stream
                        yield {
                            'choices': [{
                                'text': '',
                                'finish_reason': 'stop'
                            }]
                        }
                        break
                    
                    # Parse JSON data chunk
                    try:
                        data = json.loads(data_str)
                        if "content" in data:
                            # Extract text content and yield in compatible format
                            yield {
                                'choices': [{
                                    'text': data["content"],
                                    'finish_reason': None
                                }]
                            }
                        else:
                            # Log unexpected JSON structure but continue
                            logger.debug(f"JSON chunk missing 'content' field: {data_str[:100]}...")
                    except json.JSONDecodeError as e:
                        # Enhanced JSON parsing error recovery with diagnostic information
                        logger.debug(
                            f"JSON parsing error in streaming response:\n"
                            f"  Chunk: {data_str[:200]}{'...' if len(data_str) > 200 else ''}\n"
                            f"  Error: {e}\n"
                            f"  Position: line {e.lineno}, column {e.colno}\n"
                            f"  Continuing with next chunk..."
                        )
                        continue
                
                elif line.startswith("event: "):
                    # Handle SSE event type (currently not used by llama-server)
                    event_type = line[7:]
                    logger.debug(f"Received SSE event: {event_type}")
                    
                elif line.startswith("id: "):
                    # Handle SSE event ID (currently not used by llama-server)
                    event_id = line[4:]
                    logger.debug(f"Received SSE event ID: {event_id}")
                    
                elif line.startswith("retry: "):
                    # Handle SSE retry directive (currently not used by llama-server)
                    retry_ms = line[7:]
                    logger.debug(f"Received SSE retry directive: {retry_ms}ms")
                    
                elif line.startswith(": "):
                    # Handle SSE comments (lines starting with colon)
                    logger.debug(f"Received SSE comment: {line[2:]}")
                    
                else:
                    # Unknown SSE line format - log but continue
                    logger.debug(f"Unknown SSE line format: {line[:50]}...")
                    continue
        
        except requests.exceptions.ConnectionError as e:
            # Enhanced connection error with more diagnostic information
            error_details = []
            error_details.append(f"Failed to connect to llama-server at {self.base_url}")
            error_details.append(f"Connection error: {e}")
            
            # Add process status information if available
            if self.process:
                if self.process.poll() is not None:
                    error_details.append(f"llama-server process has terminated (exit code: {self.process.returncode})")
                else:
                    error_details.append(f"llama-server process is running (PID: {self.last_subprocess_pid})")
            else:
                error_details.append("llama-server process not started")
            
            error_details.append("Troubleshooting:")
            error_details.append("- Ensure llama-server is running and accessible")
            error_details.append(f"- Check if port {self.port} is available")
            error_details.append(f"- Verify server is listening on {self.host}:{self.port}")
            
            raise ConnectionError("\n".join(error_details))
        
        except requests.exceptions.Timeout as e:
            # Enhanced timeout error with configuration details
            error_details = []
            error_details.append(f"Request to llama-server timed out after {timeout_seconds}s")
            error_details.append(f"Timeout error: {e}")
            error_details.append(f"Request parameters: max_tokens={max_tokens}, prompt_length={len(prompt)}")
            error_details.append(f"Timeout calculation: {max_tokens} tokens * 2s + 60s buffer = {timeout_seconds}s")
            error_details.append("Consider:")
            error_details.append("- Reducing max_tokens for faster completion")
            error_details.append("- Checking server performance and load")
            error_details.append("- Verifying model size and hardware capabilities")
            
            raise TimeoutError("\n".join(error_details))
        
        except requests.exceptions.HTTPError as e:
            # Enhanced HTTP error with response details
            error_details = []
            error_details.append(f"HTTP error from llama-server: {e}")
            error_details.append(f"Status code: {response.status_code}")
            error_details.append(f"URL: {response.url}")
            
            # Include response body if available and not too large
            response_text = getattr(response, 'text', 'No response body')
            if len(response_text) > 1000:
                response_text = response_text[:1000] + "... (truncated)"
            error_details.append(f"Response body: {response_text}")
            
            # Add common HTTP status code explanations
            status_explanations = {
                400: "Bad Request - Check request parameters",
                401: "Unauthorized - Authentication required",
                403: "Forbidden - Access denied",
                404: "Not Found - Endpoint not available",
                405: "Method Not Allowed - Check HTTP method",
                413: "Payload Too Large - Reduce prompt size",
                429: "Too Many Requests - Server overloaded",
                500: "Internal Server Error - Server-side issue",
                502: "Bad Gateway - Server connectivity issue",
                503: "Service Unavailable - Server temporarily down",
                504: "Gateway Timeout - Server response timeout"
            }
            
            if response.status_code in status_explanations:
                error_details.append(f"Explanation: {status_explanations[response.status_code]}")
            
            raise RuntimeError("\n".join(error_details))
        
        except requests.exceptions.ChunkedEncodingError as e:
            # Enhanced connection drop handling with diagnostic information
            logger.warning(
                f"Connection dropped during streaming response:\n"
                f"  Error: {e}\n"
                f"  Server: {self.base_url}\n"
                f"  Request: max_tokens={max_tokens}, prompt_length={len(prompt)}\n"
                f"  This may indicate server overload or network issues"
            )
            # Yield final chunk to indicate completion with connection error
            yield {
                'choices': [{
                    'text': '',
                    'finish_reason': 'connection_error'
                }]
            }
        
        except Exception as e:
            # Enhanced general error handling with context information
            error_details = []
            error_details.append(f"Unexpected error during streaming: {type(e).__name__}: {e}")
            error_details.append(f"Server: {self.base_url}")
            error_details.append(f"Request parameters: max_tokens={max_tokens}, prompt_length={len(prompt)}")
            
            # Add process status if available
            if self.process and self.process.poll() is not None:
                error_details.append(f"llama-server process terminated unexpectedly (exit code: {self.process.returncode})")
            
            # Include traceback for debugging
            import traceback
            error_details.append(f"Traceback: {traceback.format_exc()}")
            
            logger.error("\n".join(error_details))
            raise RuntimeError(f"Streaming failed: {e}")
    
    def create_completion(
        self,
        prompt: str,
        max_tokens: int = 128,
        stream: bool = True,
        enable_prompt_cache: bool = False,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """
        Alternative interface for compatibility.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            stream: Whether to stream output
            enable_prompt_cache: Whether to enable prompt caching for this request
            **kwargs: Additional arguments
        
        Yields:
            Dictionary with 'choices' containing generated text chunks
        """
        return self(
            prompt=prompt,
            max_tokens=max_tokens,
            stream=stream,
            enable_prompt_cache=enable_prompt_cache,
            **kwargs
        )
    
    def tokenize(self, text) -> List[int]:
        """
        Tokenize text (approximate).
        
        Note: This is an approximation since we don't have direct access
        to the tokenizer. For accurate token counts, use the llama-server
        /tokenize endpoint separately.
        
        Args:
            text: Text to tokenize (string or bytes)
        
        Returns:
            List of token IDs (approximated as character count / 4)
        """
        # Handle both string and bytes input for compatibility
        if isinstance(text, bytes):
            char_count = len(text)
        else:
            char_count = len(text.encode('utf-8'))
        
        # Rough approximation: 1 token ≈ 4 characters
        token_count = max(1, char_count // 4)
        
        # Return dummy token IDs (framework only needs the count)
        return list(range(token_count))
    
    def close(self):
        """Terminate llama-server subprocess and clean up resources."""
        logger.info("Closing NativeLlamaServer")
        self._cleanup_process()
        
        # Close HTTP session
        if hasattr(self, 'session'):
            self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_native_llama_server(model_path: str, **kwargs) -> NativeLlamaServer:
    """
    Factory function to create NativeLlamaServer instance.
    
    Args:
        model_path: Path to GGUF model
        **kwargs: Additional arguments for NativeLlamaServer
    
    Returns:
        NativeLlamaServer instance
    """
    return NativeLlamaServer(model_path, **kwargs)