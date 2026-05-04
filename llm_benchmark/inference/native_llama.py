"""
Native llama.cpp wrapper for Android.

Uses subprocess to call the native llama-cli binary instead of llama-cpp-python.
This bypasses the llama-cpp-python "unsupported platform" issue on Android.
"""

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator, Optional, Dict, Any

logger = logging.getLogger(__name__)


class NativeLlamaCpp:
    """Wrapper around native llama.cpp CLI for Android."""
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 4,
        n_batch: int = 512,
        llama_cli_path: str = "~/llama.cpp/build/bin/llama-cli",
        **kwargs
    ):
        """
        Initialize native llama.cpp wrapper.
        
        Args:
            model_path: Path to GGUF model file
            n_ctx: Context size
            n_threads: Number of threads
            n_batch: Batch size for prompt processing
            llama_cli_path: Path to llama-cli binary
            **kwargs: Additional arguments (ignored, for compatibility)
        """
        self.model_path = Path(model_path).expanduser()
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.last_subprocess_pid = None  # Track subprocess PID for memory measurement
        
        # Try to find the best binary to use
        # Priority: main -> llama-simple -> llama-cli
        # main is preferred because it's older and doesn't have conversation mode
        llama_cli_path_expanded = Path(llama_cli_path).expanduser()
        main_path = llama_cli_path_expanded.parent / "main"
        simple_path = llama_cli_path_expanded.parent / "llama-simple"
        
        binaries_to_try = [
            ("main", main_path),
            ("llama-simple", simple_path),
            ("llama-cli", llama_cli_path_expanded),
        ]
        
        for binary_type, binary_path in binaries_to_try:
            if binary_path.exists():
                self.llama_cli_path = binary_path
                self.binary_type = binary_type
                logger.info(f"Using '{binary_type}' binary: {binary_path}")
                break
        else:
            raise FileNotFoundError(
                f"No llama.cpp binary found. Tried:\n"
                f"  - {main_path}\n"
                f"  - {simple_path}\n"
                f"  - {llama_cli_path_expanded}\n"
                "Build llama.cpp first:\n"
                "  cd ~/llama.cpp\n"
                "  cmake -B build -DCMAKE_BUILD_TYPE=Release\n"
                "  cmake --build build --config Release -j4"
            )
        
        # Verify model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        logger.info(f"Initialized NativeLlamaCpp with binary: {self.llama_cli_path} (type: {self.binary_type})")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Configuration: n_ctx={n_ctx}, n_threads={n_threads}, n_batch={n_batch}")
    
    def __call__(
        self,
        prompt: str,
        max_tokens: int = 100,
        stream: bool = True,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """
        Generate text using native llama.cpp.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            stream: Whether to stream output (always True for timing)
            **kwargs: Additional arguments (ignored)
        
        Yields:
            Dictionary with 'choices' containing generated text chunks
        """
        # Build command with minimal flags to avoid triggering conversation mode
        # No timeout wrapper - we handle timeout at Python level
        cmd = [
            str(self.llama_cli_path),
            "-m", str(self.model_path),
            "-c", str(self.n_ctx),
            "-t", str(self.n_threads),
            "-b", str(self.n_batch),
            "-n", str(max_tokens),
            "-p", prompt,
        ]
        
        # Calculate timeout: 2 seconds per token + 60s buffer
        timeout_seconds = max_tokens * 2 + 60
        
        logger.debug(f"Running {self.binary_type}: {' '.join(cmd)}")
        logger.debug(f"Timeout: {timeout_seconds}s")
        
        # Run binary with process group creation for aggressive killing
        try:
            start_time = time.time()
            
            # Create process with new session (process group)
            # This allows us to kill all child processes
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,  # Completely closed stdin
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True  # Create new process group
            )
            
            # Store subprocess PID for memory measurement
            self.last_subprocess_pid = process.pid
            logger.debug(f"Subprocess PID: {self.last_subprocess_pid}")
            
            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
                
            except subprocess.TimeoutExpired:
                logger.error(f"Process timed out after {timeout_seconds}s")
                logger.error(f"Binary: {self.binary_type} at {self.llama_cli_path}")
                logger.error(f"Command: {' '.join(cmd)}")
                
                # Kill entire process group with SIGKILL
                try:
                    pgid = os.getpgid(process.pid)
                    os.killpg(pgid, signal.SIGKILL)
                    logger.info(f"Killed process group {pgid} with SIGKILL")
                except Exception as e:
                    logger.warning(f"Failed to kill process group: {e}")
                    # Fallback: kill just the parent process
                    process.kill()
                    logger.info("Killed parent process with SIGKILL (fallback)")
                
                # Wait for process to die
                stdout, stderr = process.communicate()
                
                raise TimeoutError(
                    f"{self.binary_type} timed out after {timeout_seconds}s. "
                    f"It likely entered conversation mode (infinite > loop). "
                    f"Try checking for alternative binaries: ls ~/llama.cpp/build/bin/ "
                    f"If 'main' binary exists, it will be used automatically on next run."
                )
            
            if process.returncode != 0:
                logger.error(f"{self.binary_type} failed with return code {process.returncode}")
                logger.error(f"stderr: {stderr}")
                raise RuntimeError(f"{self.binary_type} failed: {stderr}")
            
            # Clean the output - remove prompt markers and extra whitespace
            output = stdout.strip()
            
            # Remove common prompt markers that might appear
            for marker in ['>', '>>>', '> ', 'prompt:', 'Prompt:']:
                output = output.replace(marker, '')
            
            # Remove the original prompt if it was echoed
            if output.startswith(prompt):
                output = output[len(prompt):].lstrip()
            
            output = output.strip()
            
            # Remove any remaining whitespace-only lines
            lines = [line for line in output.split('\n') if line.strip()]
            output = '\n'.join(lines)
            
            if not output:
                logger.warning("No output generated - this might indicate an issue")
                logger.warning(f"stdout was: {repr(stdout[:200])}")
                logger.warning(f"stderr was: {repr(stderr[:200])}")
            
            logger.debug(f"Generated output: {output[:100]}..." if len(output) > 100 else f"Generated output: {output}")
            
            # Simulate streaming by yielding character by character
            # This maintains compatibility with the profiler's TTFT measurement
            for i, char in enumerate(output):
                yield {
                    'choices': [{
                        'text': char,
                        'finish_reason': None
                    }]
                }
            
            # Final chunk with finish reason
            yield {
                'choices': [{
                    'text': '',
                    'finish_reason': 'stop'
                }]
            }
            
            total_time = time.time() - start_time
            logger.debug(f"Generated {len(output)} characters in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Native llama.cpp execution failed: {e}")
            raise
    
    def tokenize(self, text: bytes) -> list:
        """
        Tokenize text (approximate).
        
        Note: This is an approximation since we don't have direct access
        to the tokenizer. For accurate token counts, use the llama-cli
        --tokenize flag separately.
        
        Args:
            text: Text to tokenize (as bytes)
        
        Returns:
            List of token IDs (approximated as character count / 4)
        """
        # Rough approximation: 1 token ≈ 4 characters
        # This is used by the framework for prompt token estimation
        char_count = len(text) if isinstance(text, bytes) else len(text.encode('utf-8'))
        token_count = max(1, char_count // 4)
        
        # Return dummy token IDs (framework only needs the count)
        return list(range(token_count))
    
    def create_completion(
        self,
        prompt: str,
        max_tokens: int = 100,
        stream: bool = True,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """
        Alternative interface for compatibility.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            stream: Whether to stream output
            **kwargs: Additional arguments
        
        Yields:
            Dictionary with 'choices' containing generated text chunks
        """
        return self(prompt=prompt, max_tokens=max_tokens, stream=stream, **kwargs)


def create_native_llama(model_path: str, **kwargs) -> NativeLlamaCpp:
    """
    Factory function to create NativeLlamaCpp instance.
    
    Args:
        model_path: Path to GGUF model
        **kwargs: Additional arguments for NativeLlamaCpp
    
    Returns:
        NativeLlamaCpp instance
    """
    return NativeLlamaCpp(model_path, **kwargs)
