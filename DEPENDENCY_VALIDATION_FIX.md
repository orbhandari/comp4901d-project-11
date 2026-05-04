# Dependency Validation Fix for Android

## Problem

The `validate_dependencies()` function in `main.py` was trying to import `llama_cpp` on all platforms, causing:

```
RuntimeError: Unsupported platform
  at llama_cpp/_ctypes_extensions.py
```

This happened during the dependency validation step, before the framework could even detect Android and use native llama.cpp.

## Root Cause

The validation function had:

```python
required_packages = {
    'llama_cpp': 'llama-cpp-python',  # ❌ Required on ALL platforms
    'psutil': 'psutil',
    # ...
}

for module_name, package_name in required_packages.items():
    __import__(module_name)  # ❌ Fails on Android
```

This tried to import `llama_cpp` on Android, which fails with "Unsupported platform" before we can use native llama.cpp.

## Solution

Made `llama-cpp-python` optional on Android:

```python
# Detect if we're on Android
from llm_benchmark.hardware.detector import HardwareDetector
hw_info = HardwareDetector.detect()
is_android = hw_info.os_type == "android"

# Check required packages
required_packages = {
    'psutil': 'psutil',
    'pandas': 'pandas',
    # ... other packages
}

# llama-cpp-python is only required on non-Android platforms
if not is_android:
    required_packages['llama_cpp'] = 'llama-cpp-python'
```

## Additional Improvements

### 1. Android-Specific Installation Instructions

When dependencies are missing on Android, show Termux-specific commands:

```python
if is_android:
    logger.info("On Android/Termux:")
    logger.info("  pkg install python-numpy python-psutil python-pandas ...")
    logger.info("  pip install seaborn huggingface-hub")
```

### 2. Native llama.cpp Check

After validating dependencies on Android, check if native llama.cpp is available:

```python
if is_android:
    llama_cli = Path("~/llama.cpp/build/bin/llama-cli").expanduser()
    if llama_cli.exists():
        logger.info("✅ Native llama.cpp found")
    else:
        logger.warning("⚠️  Native llama.cpp not found")
        logger.warning("   Build it with:")
        logger.warning("     cd ~/llama.cpp")
        logger.warning("     cmake -B build && cmake --build build -j4")
```

This gives users immediate feedback if they forgot to build llama.cpp.

### 3. llama-cpp-python as Optional on Android

Moved `llama_cpp` to optional packages on Android:

```python
if is_android:
    optional_packages['llama_cpp'] = 'llama-cpp-python (optional - using native llama.cpp)'
```

This allows the framework to run even if llama-cpp-python is installed but broken on Android.

## Behavior by Platform

### x86 Linux / Jetson
- ✅ `llama-cpp-python` is **required**
- ✅ Import validation runs
- ✅ Fails if llama-cpp-python is missing

### Android
- ✅ `llama-cpp-python` is **optional**
- ✅ Import validation skipped for llama_cpp
- ✅ Checks for native llama.cpp instead
- ✅ Shows build instructions if missing

## Files Modified

**`llm_benchmark/main.py`:**
- Modified `validate_dependencies()` function
- Added Android detection
- Made llama-cpp-python conditional
- Added native llama.cpp check
- Added Android-specific installation instructions

## Testing

### On Development Machine (x86 Linux)
```bash
$ python -c "from llm_benchmark.main import validate_dependencies; validate_dependencies()"
All required dependencies are installed
```

### On Android (Expected Output)
```bash
$ python -c "from llm_benchmark.main import validate_dependencies; validate_dependencies()"
Optional package not installed: llama-cpp-python (optional - using native llama.cpp)
All required dependencies are installed
✅ Native llama.cpp found at ~/llama.cpp/build/bin/llama-cli
```

### On Android Without Native llama.cpp
```bash
$ python -c "from llm_benchmark.main import validate_dependencies; validate_dependencies()"
All required dependencies are installed
⚠️  Native llama.cpp not found at ~/llama.cpp/build/bin/llama-cli
   Build it with:
     cd ~/llama.cpp
     cmake -B build && cmake --build build -j4
```

## For Android Users

After pulling this fix:

```bash
cd ~/comp4901d-project-11
git pull

# Now run the benchmark
python -m llm_benchmark --config configs/android_example.json
```

The dependency validation will:
1. ✅ Skip llama-cpp-python import (no error!)
2. ✅ Check other dependencies
3. ✅ Check for native llama.cpp
4. ✅ Proceed to run the benchmark

## Summary

The fix ensures that:
- ✅ Dependency validation doesn't fail on Android
- ✅ llama-cpp-python is optional on Android
- ✅ Native llama.cpp availability is checked
- ✅ Clear instructions shown if setup incomplete
- ✅ Other platforms unchanged (still require llama-cpp-python)

The framework can now pass dependency validation on Android and proceed to use native llama.cpp!

