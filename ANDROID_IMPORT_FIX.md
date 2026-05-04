# Android Import Fix - Removing llama-cpp-python Dependency

## Problem

The framework was importing `llama_cpp` at the module level in profilers, causing "unsupported platform" errors on Android before the native llama.cpp detection could even run.

**Error:**
```
RuntimeError: Unsupported platform
  at ctypes_extensions.py
```

## Root Cause

Two files were importing `llama-cpp-python` at the top of the file:

1. **`llm_benchmark/profiler/quantization.py`**:
   ```python
   from llama_cpp import Llama  # ❌ Imported at module level
   ```

2. **`llm_benchmark/profiler/ablation.py`**:
   ```python
   from llama_cpp import Llama  # ❌ Imported at module level
   ```

This meant that even before the `AndroidBackend` could detect and use native llama.cpp, Python would try to import `llama-cpp-python` and fail with "unsupported platform" on Android.

## Solution

### 1. Removed Module-Level Imports

**quantization.py:**
```python
# Before
from llama_cpp import Llama

# After
# No import - use backend.load_model_safe() instead
```

**ablation.py:**
```python
# Before
from llama_cpp import Llama

# After
# No import - use backend.load_model_safe() instead
```

### 2. Use Backend's load_model_safe() Method

Instead of directly instantiating `Llama()`, the profilers now use the backend's `load_model_safe()` method, which handles platform detection and chooses the appropriate loader.

**quantization.py:**
```python
# Before
llm = Llama(model_path=model_path, **llama_config)

# After
llm = self.backend.load_model_safe(model_path, **llama_config)
if llm is None:
    raise RuntimeError(f"Failed to load model: {model_path}")
```

**ablation.py:**
```python
# Added helper method
def _load_model(self, model_path: str, **kwargs) -> Any:
    """Load model using backend's load_model_safe() method."""
    llm = self.backend.load_model_safe(model_path, **kwargs)
    if llm is None:
        raise RuntimeError(f"Failed to load model: {model_path}")
    return llm

# Before (6 occurrences)
llm = Llama(model_path=model_path, **llama_config)

# After (6 occurrences)
llm = self._load_model(model_path, **llama_config)
```

## Files Modified

1. **`llm_benchmark/profiler/quantization.py`**
   - Removed `from llama_cpp import Llama`
   - Changed `Llama(...)` to `self.backend.load_model_safe(...)`
   - Added null check and error handling

2. **`llm_benchmark/profiler/ablation.py`**
   - Removed `from llama_cpp import Llama`
   - Added `_load_model()` helper method
   - Replaced 6 occurrences of `Llama(...)` with `self._load_model(...)`
   - Updated log messages from "Llama config" to "Model config"

## Verification

Created `test_android_imports.py` to verify all imports work without llama-cpp-python:

```bash
$ python test_android_imports.py
Testing imports without llama-cpp-python...
======================================================================

1. Testing hardware detector...
   ✅ HardwareDetector imported successfully

2. Testing hardware backends...
   ✅ Hardware backends imported successfully

3. Testing native llama wrapper...
   ✅ NativeLlamaCpp imported successfully

4. Testing quantization profiler...
   ✅ QuantizationProfiler imported successfully

5. Testing ablation engine...
   ✅ AblationEngine imported successfully

6. Testing metrics collector...
   ✅ MetricsCollector imported successfully

7. Testing orchestrator...
   ✅ TestOrchestrator imported successfully

======================================================================
✅ ALL IMPORTS SUCCESSFUL!
======================================================================
```

## How It Works Now

### Execution Flow on Android

1. **User runs benchmark:**
   ```bash
   python -m llm_benchmark --config configs/android_example.json
   ```

2. **Hardware detection:**
   - `HardwareDetector` identifies Android platform
   - Returns `os_type="android"`

3. **Backend creation:**
   - `create_backend()` returns `AndroidBackend` instance

4. **Profiler initialization:**
   - `QuantizationProfiler` and `AblationEngine` initialize
   - **No llama-cpp-python import** - just store backend reference

5. **Model loading:**
   - Profiler calls `backend.load_model_safe(model_path, **config)`
   - `AndroidBackend.load_model_safe()` checks for native llama.cpp
   - If found: Returns `NativeLlamaCpp` instance
   - If not found: Shows setup instructions

6. **Inference:**
   - Profiler uses model instance (works with both `Llama` and `NativeLlamaCpp`)
   - Metrics collection works identically
   - Results generated normally

### Compatibility

The fix maintains compatibility with all platforms:

- **x86 Linux**: Uses llama-cpp-python (imported in `X86Backend.load_model_safe()`)
- **Jetson**: Uses llama-cpp-python (imported in `JetsonBackend.load_model_safe()`)
- **Android**: Uses native llama.cpp (no llama-cpp-python import)

The imports are now **lazy** - only imported when actually needed by the specific backend.

## Benefits

1. ✅ **No premature imports** - llama-cpp-python only imported when needed
2. ✅ **Platform-agnostic profilers** - work with any backend
3. ✅ **Clean separation** - backend handles platform-specific loading
4. ✅ **Better error messages** - Android users see setup instructions, not cryptic import errors
5. ✅ **Maintainable** - single point of model loading logic (backend)

## Testing on Android

Now when you run on Android:

```bash
# 1. Build llama.cpp (one-time setup)
cd ~/llama.cpp
cmake -B build && cmake --build build -j4

# 2. Run benchmark
cd ~/comp4901d-project-11
python -m llm_benchmark --config configs/android_example.json
```

The framework will:
- ✅ Import successfully (no llama-cpp-python errors)
- ✅ Detect Android platform
- ✅ Check for native llama.cpp
- ✅ Use `NativeLlamaCpp` wrapper
- ✅ Run all benchmarks normally

If native llama.cpp is not found, you'll see clear setup instructions instead of a cryptic "unsupported platform" error.

## Summary

The fix ensures that:
- **Imports are lazy** - only when needed by specific backend
- **Profilers are platform-agnostic** - delegate to backend for loading
- **Android works out of the box** - no llama-cpp-python dependency
- **Other platforms unchanged** - still use llama-cpp-python
- **Error messages are helpful** - guide users to correct setup

The framework can now run on Android without any llama-cpp-python dependency!

