"""
Native llama.cpp wrapper for Android.

Uses subprocess to call the native llama-cli binary instead of llama-cpp-python.
This bypasses the llama-cpp-python "unsupported platform" issue on Android.
"""

import logging
import select
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
        
        # Try to find the best binary to use
        # Priority: llama-cli -> main -> llama-simple
        llama_cli_path_expanded = Path(llama_cli_path).expanduser()
        main_path = llama_cli_path_expanded.parent / "main"
        simple_path = llama_cli_path_expanded.parent / "llama-simple"
        
        if llama_cli_path_expanded.exists():
            self.llama_cli_path = llama_cli_path_expanded
            self.binary_type = "llama-cli"
        elif main_path.exists():
            self.llama_cli_path = main_path
            self.binary_type = "main"
            logger.info(f"Using 'main' binary instead of 'llama-cli': {main_path}")
        elif simple_path.exists():
            self.llama_cli_path = simple_path
            self.binary_type = "llama-simple"
            logger.info(f"Using 'llama-simple' binary: {simple_path}")
        else:
            raise FileNotFoundError(
                f"No llama.cpp binary found. Tried:\n"
                f"  - {llama_cli_path_expanded}\n"
                f"  - {main_path}\n"
                f"  - {simple_path}\n"
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
        # Build command based on binary type
        # Keep it simple - only use flags that are universally supported
        # Wrap with timeout command to force kill if it hangs
        base_cmd = [
            str(self.llama_cli_path),
            "-m", str(self.model_path),
            "-c", str(self.n_ctx),
            "-t", str(self.n_threads),
            "-b", str(self.n_batch),
            "-n", str(max_tokens),
            "-p", prompt,
        ]
        
        # Add binary-specific flags (only well-supported ones)
        if self.binary_type == "llama-cli":
            # llama-cli specific flags - only use widely supported ones
            base_cmd.extend([
                "--log-disable",  # Disable logging
                "-ngl", "0",  # Disable GPU (already warned, but needed)
            ])
        elif self.binary_type == "main":
            # main binary (older) - usually doesn't have conversation mode
            base_cmd.extend([
                "--log-disable",  # Disable logging
                "-ngl", "0",  # Disable GPU
            ])
        
        # Wrap with timeout command (60 seconds should be enough for 50 tokens)
        # This will kill the process if it hangs in conversation mode
        timeout_seconds = max(60, max_tokens * 2)  # 2 seconds per token, minimum 60s
        cmd = ["timeout", f"{timeout_seconds}s"] + base_cmd
        
        logger.debug(f"Running: {' '.join(cmd)}")
        
        # Run binary - use DEVNULL for stdin (don't send EOF, just close it completely)
        try:
            start_time = time.time()
            
            # Use DEVNULL for stdin - completely closed, no EOF signal
            # This is cleaner than sending EOF
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,  # Completely closed stdin
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds + 10)  # Extra 10s buffer
            except subprocess.TimeoutExpired:
                logger.error(f"Binary timed out after {timeout_seconds + 10} seconds")
                logger.error("This usually means the binary entered interactive/conversation mode")
                logger.error("The 'timeout' command should have killed it, but communicate() also timed out")
                process.kill()
                stdout, stderr = process.communicate()
                raise TimeoutError(
                    f"{self.binary_type} timed out. "
                    "It likely entered conversation mode. "
                    "Try checking for alternative binaries with: ls ~/llama.cpp/build/bin/"
                )
            
            # Check if timeout command killed the process (exit code 124)
            if process.returncode == 124:
                logger.error(f"Process was killed by timeout command after {timeout_seconds}s")
                logger.error("This means llama-cli entered conversation mode and didn't exit")
                logger.error(f"stdout: {stdout[:500]}")
                logger.error(f"stderr: {stderr[:500]}")
                raise TimeoutError(
                    f"{self.binary_type} was killed by timeout after {timeout_seconds}s. "
                    "It entered conversation mode (infinite > loop). "
                    "Your llama-cli version may not support non-interactive mode. "
                    "Check for 'main' binary: ls ~/llama.cpp/build/bin/main"
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
