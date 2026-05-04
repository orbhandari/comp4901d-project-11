# Android Setup Guide - Native llama.cpp Solution

This guide explains how to set up and run the LLM Benchmark Framework on Android devices using Termux with **native llama.cpp**.

## Overview

On Android, we use **native llama.cpp** instead of llama-cpp-python because llama-cpp-python doesn't support Android. This approach:
- ✅ Uses official llama.cpp Android support
- ✅ Provides native performance  
- ✅ Bypasses Python binding limitations
- ✅ Is officially supported by the llama.cpp team
- ✅ Actually works on Android!

## Prerequisites

### Device Requirements

- **Android Version**: Android 7.0 (Nougat) or higher (tested on Android 15)
- **RAM**: Minimum 6GB, recommended 8GB+ (tested on 12GB)
- **Storage**: At least 10GB free space for llama.cpp, models, and results
- **CPU**: ARM64 architecture (most modern Android devices)

### Tested Configuration

- **Device**: Xiaomi 13T (model 2306EPN60G)
- **OS**: HyperOS (Android 15 AP3A)
- **RAM**: 12GB
- **CPU**: MediaTek Dimensity (ARM64)

## Installation Steps

### Step 1: Install Termux

Termux is a terminal emulator for Android that provides a Linux environment.

1. **Download Termux** from F-Droid (recommended) or GitHub releases
   - F-Droid: https://f-droid.org/packages/com.termux/
   - GitHub: https://github.com/termux/termux-app/releases

2. **Do NOT use Google Play Store version** (it's outdated and incompatible)

3. **Install Termux** and open it

### Step 2: Update Termux Packages

```bash
pkg update && pkg upgrade -y
```

### Step 3: Install Build Dependencies

```bash
# Install Python and build tools
pkg install python python-pip git cmake clang binutils

# Install Rust (required for huggingface-hub)
pkg install rust
```

### Step 4: Install Python Packages

**IMPORTANT**: Packages with C extensions MUST be installed via `pkg install python-*`, NOT pip!

```bash
# Install Python packages with C extensions via pkg (NOT pip)
pkg install python-numpy python-psutil python-pandas python-matplotlib python-scipy python-pyyaml

# Upgrade pip
pip install --upgrade pip

# Install pure Python packages via pip
pip install seaborn jinja2 python-dotenv huggingface-hub

# DO NOT install llama-cpp-python - we use native llama.cpp instead!
```

**Why this matters:**
- ✅ Packages with C extensions (numpy, psutil, etc.) are pre-compiled for Android ARM64 via pkg
- ❌ Installing them via pip will fail or produce incompatible binaries
- ✅ Pure Python packages (seaborn, jinja2, etc.) can be installed via pip

### Step 5: Build Native llama.cpp

This is the key step that makes Android support work:

```bash
# Clone llama.cpp
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with CMake (this will take ~30 minutes)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4

# Verify build succeeded
ls -lh build/bin/llama-cli
# Should show the llama-cli binary (~50-100MB)
```

**Build time**: ~30 minutes on most Android devices.

**Troubleshooting build issues:**
```bash
# If build fails, try with fewer parallel jobs
cmake --build build --config Release -j2

# Or build without parallelism
cmake --build build --config Release
```

### Step 6: Test Native llama.cpp (Optional)

Before integrating with the framework, test that llama.cpp works:

```bash
# Download a small test model
cd ~
mkdir -p models
cd models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf

# Test llama-cli directly
~/llama.cpp/build/bin/llama-cli \
  -m ~/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf \
  -c 512 \
  -n 20 \
  -p "Hello, how are you?"

# You should see text generation output
```

If this works, you're ready to proceed!

### Step 7: Clone the Benchmark Repository

```bash
cd ~
git clone <repository-url> comp4901d-project-11
cd comp4901d-project-11
```

### Step 8: Setup Hugging Face Token

```bash
# Create .env file
echo "HF_TOKEN=your_token_here" > .env

# Or export directly
export HF_TOKEN=your_token_here

# Add to .bashrc for persistence
echo 'export HF_TOKEN="your_token"' >> ~/.bashrc
```

### Step 9: Download Models

```bash
# Create models directory in shared storage
mkdir -p ~/storage/shared/models
cd ~/storage/shared/models

# Download TinyLlama models (recommended for testing)
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf
```

### Step 10: Run Benchmark

```bash
cd ~/comp4901d-project-11
python -m llm_benchmark --config configs/android_example.json
```

The framework will automatically detect that you're on Android and use the native llama.cpp binary at `~/llama.cpp/build/bin/llama-cli`.

## How It Works

The framework includes a Python wrapper (`llm_benchmark/inference/native_llama.py`) that:

1. Detects if you're running on Android
2. Checks if `~/llama.cpp/build/bin/llama-cli` exists
3. If found, uses native llama.cpp via subprocess instead of llama-cpp-python
4. Provides the same interface as llama-cpp-python for compatibility
5. Streams output token-by-token for accurate timing measurements

This approach bypasses the llama-cpp-python "unsupported platform" issue entirely!

## Configuration Example

Create an Android-specific configuration:

```json
{
  "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
  "models": {
    "Q2_K": "tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
    "Q4_0": "tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
    "Q8_0": "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
  },
  "model_cache_dir": "~/storage/shared/models",
  "context_size": 512,
  "batch_size": 256,
  "max_tokens": 50,
  "iterations": 3,
  "warmup_runs": 1,
  "enable_quantization_profiling": true,
  "enable_ablation_studies": false,
  "enable_batch_testing": false,
  "enable_thermal_monitoring": true,
  "sleep_between_tests_s": 10,
  "thermal_stabilization_threshold_c": 70.0,
  "inference_timeout_s": 600,
  "output_dir": "~/storage/shared/benchmark_results",
  "save_formats": ["json", "csv", "html"],
  "visualization_dpi": 150
}
```

## Optimization Tips

### 1. Memory Management

**Use Smaller Models**:
- Start with Q2_K quantization (~600MB RAM)
- Move to Q4_0 if you have 8GB+ RAM (~900MB RAM)
- Avoid Q8_0 unless you have 12GB+ RAM (~1.2GB RAM)

**Reduce Context Size**:
```json
{
  "context_size": 512,  // Instead of 2048
  "batch_size": 256     // Instead of 512
}
```

**Close Background Apps**:
```bash
# Check available memory
free -h

# If low on memory, close other apps
```

### 2. Thermal Management

**Monitor Temperature**:
```bash
# Check thermal zones
for zone in /sys/class/thermal/thermal_zone*/temp; do
    temp=$(cat $zone)
    temp_c=$((temp / 1000))
    echo "$(basename $(dirname $zone)): ${temp_c}°C"
done
```

**Prevent Overheating**:
- Keep device plugged in (but remove case for better cooling)
- Run benchmarks in a cool environment
- Use a phone cooler or fan if available
- Increase sleep time between tests:
  ```json
  {
    "sleep_between_tests_s": 30
  }
  ```

### 3. Performance Tuning

**Thread Count**:
The framework automatically uses `cpu_cores - 2` threads to leave cores for the system. You can override this:

```json
{
  "n_threads": 4  // Adjust based on your device
}
```

**Batch Size**:
Smaller batch sizes use less memory but may be slower:

```json
{
  "batch_size": 128  // Default is 256 on Android
}
```

### 4. Storage Optimization

**Use Shared Storage** for easy access from file manager:
```json
{
  "model_cache_dir": "~/storage/shared/models",
  "output_dir": "~/storage/shared/benchmark_results"
}
```

**Clean Up Old Results**:
```bash
# Remove old benchmark runs
rm -rf ~/storage/shared/benchmark_results/run_*
```

## Troubleshooting

### Issue: "llama-cli not found"

**Error Message:**
```
FileNotFoundError: llama-cli not found at ~/llama.cpp/build/bin/llama-cli
```

**Solution:**
Build llama.cpp first (see Step 5):

```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4

# Verify build
ls -lh build/bin/llama-cli
```

### Issue: llama.cpp build fails

**Common Errors:**
- "cmake: command not found"
- "clang: command not found"
- "ninja: command not found"

**Solution:**
Install build dependencies:

```bash
pkg install cmake clang binutils
```

Then retry the build.

### Issue: "Insufficient RAM to load model"

**Solutions**:
1. Use smaller quantization (Q2_K instead of Q4_0)
2. Reduce context size to 256 or 512
3. Close all background apps
4. Restart Termux to free memory
5. Use a smaller model (< 1B parameters)

**Check Memory**:
```bash
# View memory usage
free -h

# View process memory
top -n 1
```

### Issue: "Device overheating / thermal throttling"

**Solutions**:
1. Let device cool down before continuing
2. Remove phone case
3. Use a fan or phone cooler
4. Increase sleep time between tests
5. Run fewer iterations

### Issue: Compilation errors when installing Python packages

**Error Examples:**
```
error: command 'clang' failed
fatal error: 'Python.h' file not found
```

**Solution:**
Use pkg instead of pip for packages with C extensions:

```bash
# These MUST be installed via pkg (NOT pip):
pkg install python-numpy
pkg install python-psutil
pkg install python-pandas
pkg install python-matplotlib
pkg install python-scipy
pkg install python-pyyaml
```

### Issue: "error: can't find Rust compiler"

**Solution:**
Install rust via Termux:

```bash
pkg install rust
pip install huggingface-hub
```

### Issue: Benchmark is very slow

**Expected Behavior**: Mobile CPUs are slower than desktop CPUs
- TinyLlama Q2_K: ~5-10 tokens/second
- TinyLlama Q4_0: ~3-7 tokens/second

**Optimizations**:
1. Reduce max_tokens to 20-30
2. Use Q2_K quantization
3. Reduce context_size to 256
4. Disable ablation studies and batch testing

### Issue: "Permission denied" errors

**Solution:**
Grant storage permissions to Termux:

```bash
termux-setup-storage
# Accept the permission request
```

## Performance Expectations

### Xiaomi 13T (12GB RAM, MediaTek Dimensity)

**TinyLlama 1.1B Q2_K**:
- Load Time: ~3-5 seconds
- Memory Usage: ~600-800 MB
- TTFT: ~200-400 ms
- Decode Speed: ~5-10 tokens/second

**TinyLlama 1.1B Q4_0**:
- Load Time: ~4-6 seconds
- Memory Usage: ~900-1200 MB
- TTFT: ~300-500 ms
- Decode Speed: ~3-7 tokens/second

**TinyLlama 1.1B Q8_0**:
- Load Time: ~5-7 seconds
- Memory Usage: ~1200-1500 MB
- TTFT: ~400-600 ms
- Decode Speed: ~2-5 tokens/second

### Comparison to Desktop

Mobile performance is typically:
- **3-5x slower** than desktop CPUs
- **Higher memory overhead** due to Android system
- **More thermal throttling** during sustained workloads

## Best Practices

### 1. Start Small

Begin with minimal configuration:
```json
{
  "iterations": 1,
  "max_tokens": 20,
  "context_size": 256,
  "enable_ablation_studies": false,
  "enable_batch_testing": false
}
```

### 2. Monitor Resources

```bash
# Watch memory and CPU during benchmark
watch -n 1 'free -h && echo && top -n 1 -b | head -20'
```

### 3. Use Shared Storage

Always use `~/storage/shared/` for easy access from file manager:
```json
{
  "model_cache_dir": "~/storage/shared/models",
  "output_dir": "~/storage/shared/benchmark_results"
}
```

### 4. Keep Device Charged

- Plug in during benchmarking
- Some devices throttle CPU when battery is low
- Performance mode may require charging

### 5. Regular Breaks

- Let device cool between benchmark runs
- Don't run continuous benchmarks for hours
- Monitor temperature regularly

## Example: Complete Benchmark Session

```bash
# 1. Check available memory
free -h

# 2. Check temperature
cat /sys/class/thermal/thermal_zone0/temp

# 3. Navigate to benchmark directory
cd ~/comp4901d-project-11

# 4. Run benchmark
python -m llm_benchmark --config configs/android_example.json

# 5. Monitor progress (in another Termux session)
tail -f ~/storage/shared/benchmark_results/benchmark.log

# 6. After completion, view results
cd ~/storage/shared/benchmark_results/run_*
cat results.md

# 7. Copy HTML report to view in browser
cp benchmark_report.html ~/storage/shared/
# Open in browser: /storage/emulated/0/benchmark_report.html
```

## Accessing Results

Results are saved to shared storage for easy access:

**From Termux:**
```bash
cd ~/storage/shared/benchmark_results
ls -lh
```

**From File Manager:**
Navigate to: `/storage/emulated/0/benchmark_results/`

**View HTML Report:**
1. Copy `benchmark_report.html` to shared storage
2. Open with any browser on your phone
3. View interactive charts and detailed metrics

## Additional Resources

- **Termux Wiki**: https://wiki.termux.com/
- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **llama.cpp Android**: https://github.com/ggerganov/llama.cpp/tree/master/examples/llama.android
- **Hugging Face Models**: https://huggingface.co/models?library=gguf

## Status

### ✅ Fully Supported

**What Works:**
- ✅ Native llama.cpp builds successfully on Android
- ✅ All dependencies install correctly
- ✅ Hardware detection identifies Android
- ✅ AndroidBackend properly configured
- ✅ Python wrapper provides llama-cpp-python compatible interface
- ✅ Thermal and memory monitoring functional
- ✅ All benchmark modes work (quantization profiling, ablation studies, etc.)
- ✅ HTML reports generate correctly

**Tested On:**
- Xiaomi 13T (Android 15, 12GB RAM)
- Termux (latest from F-Droid)
- llama.cpp (latest from GitHub)

### Implementation Details

The framework automatically detects Android and uses native llama.cpp:

1. **Detection**: `HardwareDetector` identifies Android via Termux markers
2. **Backend**: `AndroidBackend` checks for `~/llama.cpp/build/bin/llama-cli`
3. **Wrapper**: `NativeLlamaCpp` class wraps llama-cli via subprocess
4. **Interface**: Provides same API as llama-cpp-python for compatibility
5. **Metrics**: All timing and memory metrics work correctly

No code changes needed - just build llama.cpp and run!

## Conclusion

Running LLM benchmarks on Android is now fully supported using native llama.cpp! This approach:

- ✅ Works reliably on Android
- ✅ Provides native performance
- ✅ Requires minimal setup (just build llama.cpp)
- ✅ Integrates seamlessly with the framework
- ✅ Supports all benchmark features

Start with small models and conservative settings, then scale up based on your device's capabilities!

