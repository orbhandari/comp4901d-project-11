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
        self.llama_cli_path = Path(llama_cli_path).expanduser()
        
        # Verify binary exists
        if not self.llama_cli_path.exists():
            raise FileNotFoundError(
                f"llama-cli not found at {self.llama_cli_path}. "
                "Build llama.cpp first:\n"
                "  cd ~/llama.cpp\n"
                "  cmake -B build -DCMAKE_BUILD_TYPE=Release\n"
                "  cmake --build build --config Release -j4"
            )
        
        # Verify model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        logger.info(f"Initialized NativeLlamaCpp with model: {self.model_path}")
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
        # Build command - use simple one-shot generation
        cmd = [
            str(self.llama_cli_path),
            "-m", str(self.model_path),
            "-c", str(self.n_ctx),
            "-t", str(self.n_threads),
            "-b", str(self.n_batch),
            "-n", str(max_tokens),
            "-p", prompt,
            "--log-disable",  # Disable logging to stderr
            "-ngl", "0",  # Disable GPU offloading
        ]
        
        logger.debug(f"Running: {' '.join(cmd)}")
        
        # Run llama-cli and wait for completion
        try:
            start_time = time.time()
            
            # Use communicate() to get all output at once (simpler and more reliable)
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=300)  # 5 minute timeout
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                raise TimeoutError("llama-cli timed out after 300 seconds")
            
            if process.returncode != 0:
                logger.error(f"llama-cli failed with return code {process.returncode}")
                logger.error(f"stderr: {stderr}")
                raise RuntimeError(f"llama-cli failed: {stderr}")
            
            # Clean the output - remove prompt markers and extra whitespace
            output = stdout.strip()
            
            # Remove common prompt markers that might appear
            for marker in ['>', '>>>', 'prompt:', 'Prompt:', '> ']:
                output = output.replace(marker, '')
            
            # Remove the original prompt if it was echoed
            if output.startswith(prompt):
                output = output[len(prompt):].lstrip()
            
            output = output.strip()
            
            logger.debug(f"Generated output: {output[:100]}..." if len(output) > 100 else f"Generated output: {output}")
            
            # Simulate streaming by yielding character by character
            # This maintains compatibility with the profiler's TTFT measurement
            first_token_time = time.time()
            
            for i, char in enumerate(output):
                # Yield first character immediately to capture TTFT
                if i == 0:
                    first_token_time = time.time()
                
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
