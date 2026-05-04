# Android Native llama.cpp Solution

## The Solution: Build llama.cpp Directly in Termux

Instead of using `llama-cpp-python` (which doesn't support Android), we'll:
1. Build native llama.cpp in Termux
2. Create a Python wrapper around the llama.cpp CLI
3. Integrate with the existing benchmark framework

This approach is **officially supported** by llama.cpp!

## Step-by-Step Implementation

### Phase 1: Build llama.cpp in Termux (30 minutes)

```bash
# 1. Install build dependencies
pkg update && pkg upgrade -y
pkg install git cmake clang

# 2. Clone llama.cpp
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# 3. Build with CMake
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4

# 4. Verify build
ls -lh build/bin/
# Should see: llama-cli, llama-simple, etc.

# 5. Test with a model (if you have one)
./build/bin/llama-cli -m ~/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf -c 512 -p "Hello" -n 10
```

### Phase 2: Create Python Wrapper (1-2 hours)

Create a Python wrapper that calls the native llama.cpp CLI:

**File: `llm_benchmark/inference/native_llama.py`**

```python
"""
Native llama.cpp wrapper for Android.

Uses subprocess to call the native llama-cli binary instead of llama-cpp-python.
"""

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


class NativeLlamaCpp:
    """Wrapper around native llama.cpp CLI for Android."""
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 4,
        llama_cli_path: str = "~/llama.cpp/build/bin/llama-cli"
    ):
        """
        Initialize native llama.cpp wrapper.
        
        Args:
            model_path: Path to GGUF model file
            n_ctx: Context size
            n_threads: Number of threads
            llama_cli_path: Path to llama-cli binary
        """
        self.model_path = Path(model_path).expanduser()
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.llama_cli_path = Path(llama_cli_path).expanduser()
        
        # Verify binary exists
        if not self.llama_cli_path.exists():
            raise FileNotFoundError(
                f"llama-cli not found at {self.llama_cli_path}. "
                "Build llama.cpp first: cd ~/llama.cpp && cmake -B build && cmake --build build"
            )
        
        # Verify model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        logger.info(f"Initialized NativeLlamaCpp with model: {self.model_path}")
    
    def __call__(
        self,
        prompt: str,
        max_tokens: int = 100,
        stream: bool = False,
        **kwargs
    ) -> Iterator[dict]:
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
        # Build command
        cmd = [
            str(self.llama_cli_path),
            "-m", str(self.model_path),
            "-c", str(self.n_ctx),
            "-t", str(self.n_threads),
            "-n", str(max_tokens),
            "-p", prompt,
            "--no-display-prompt",  # Don't echo prompt
            "--log-disable",  # Disable logging
        ]
        
        logger.debug(f"Running: {' '.join(cmd)}")
        
        # Run llama-cli and capture output
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Stream output token by token
            for line in process.stdout:
                if line.strip():
                    # Yield in llama-cpp-python compatible format
                    yield {
                        'choices': [{
                            'text': line,
                            'finish_reason': None
                        }]
                    }
            
            # Wait for completion
            process.wait()
            
            if process.returncode != 0:
                stderr = process.stderr.read()
                raise RuntimeError(f"llama-cli failed: {stderr}")
            
            # Final chunk with finish reason
            yield {
                'choices': [{
                    'text': '',
                    'finish_reason': 'stop'
                }]
            }
            
        except Exception as e:
            logger.error(f"Native llama.cpp execution failed: {e}")
            raise
    
    def tokenize(self, text: bytes) -> list:
        """
        Tokenize text (approximate).
        
        Args:
            text: Text to tokenize (as bytes)
        
        Returns:
            List of token IDs (approximated as character count / 4)
        """
        # Rough approximation: 1 token ≈ 4 characters
        return list(range(len(text) // 4))


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
```

### Phase 3: Modify Framework to Use Native llama.cpp (30 minutes)

**File: `llm_benchmark/hardware/hal.py`**

Add detection for native llama.cpp:

```python
def _should_use_native_llama(self) -> bool:
    """Check if we should use native llama.cpp instead of llama-cpp-python."""
    # On Android, prefer native llama.cpp if available
    if self.hw_info.os_type == "android":
        llama_cli = Path("~/llama.cpp/build/bin/llama-cli").expanduser()
        if llama_cli.exists():
            logger.info("Using native llama.cpp (Android)")
            return True
    return False
```

Modify `AndroidBackend.load_model_safe()`:

```python
def load_model_safe(self, model_path: str, **kwargs) -> Any:
    """Load model using native llama.cpp on Android."""
    
    # Check if native llama.cpp is available
    llama_cli = Path("~/llama.cpp/build/bin/llama-cli").expanduser()
    
    if llama_cli.exists():
        logger.info("Using native llama.cpp CLI")
        from llm_benchmark.inference.native_llama import NativeLlamaCpp
        
        return NativeLlamaCpp(
            model_path=model_path,
            n_ctx=kwargs.get('n_ctx', 2048),
            n_threads=kwargs.get('n_threads', 4)
        )
    else:
        # Fall back to llama-cpp-python (will likely fail)
        logger.warning("Native llama.cpp not found, trying llama-cpp-python")
        try:
            from llama_cpp import Llama
            return Llama(model_path=model_path, **kwargs)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Build native llama.cpp:")
            logger.info("  cd ~/llama.cpp")
            logger.info("  cmake -B build && cmake --build build")
            raise
```

### Phase 4: Test the Integration (15 minutes)

```bash
# 1. Ensure llama.cpp is built
cd ~/llama.cpp
ls build/bin/llama-cli  # Should exist

# 2. Download a small model
cd ~
mkdir -p models
cd models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf

# 3. Test native llama.cpp directly
~/llama.cpp/build/bin/llama-cli \
  -m ~/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf \
  -c 512 \
  -n 20 \
  -p "Hello, how are you?"

# 4. Test with framework
cd ~/comp4901d-project-11
python -m llm_benchmark --config configs/android_example.json
```

## Complete Android Setup (Revised)

### 1. Install Termux Packages

```bash
pkg update && pkg upgrade -y
pkg install python python-pip git cmake clang binutils rust
pkg install python-numpy python-psutil python-pandas python-matplotlib python-scipy python-pyyaml
pip install --upgrade pip
```

### 2. Build Native llama.cpp

```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4
```

### 3. Install Python Dependencies

```bash
pip install seaborn jinja2 python-dotenv huggingface-hub
# Skip llama-cpp-python - we're using native llama.cpp!
```

### 4. Setup Benchmark Framework

```bash
cd ~
git clone <repo-url> comp4901d-project-11
cd comp4901d-project-11

# Create native_llama.py wrapper (copy code from Phase 2 above)
# Modify hal.py (copy code from Phase 3 above)
```

### 5. Download Model

```bash
mkdir -p ~/storage/shared/models
cd ~/storage/shared/models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf
```

### 6. Run Benchmark

```bash
cd ~/comp4901d-project-11
python -m llm_benchmark --config configs/android_example.json
```

## Advantages of This Approach

✅ **Uses official llama.cpp Android support**
✅ **No llama-cpp-python dependency**
✅ **Native performance**
✅ **Officially supported by llama.cpp team**
✅ **Simpler than JNI/NDK cross-compilation**
✅ **Reuses existing Python framework**
✅ **Easy to debug**

## Implementation Checklist

- [ ] Build llama.cpp in Termux
- [ ] Create `native_llama.py` wrapper
- [ ] Modify `AndroidBackend` to use native llama.cpp
- [ ] Test with small model
- [ ] Run full benchmark
- [ ] Verify metrics collection works
- [ ] Generate HTML report

## Estimated Time

- Build llama.cpp: 30 minutes
- Create wrapper: 1-2 hours
- Integration: 30 minutes
- Testing: 30 minutes
- **Total: 3-4 hours**

Much faster than rewriting in Swift or complex NDK setup!

## Next Steps

Would you like me to:
1. Create the complete `native_llama.py` file?
2. Modify the `AndroidBackend` class?
3. Create a test script to verify the integration?
4. Update the Android setup guide with these instructions?

Let me know and I'll implement the solution! 🚀
