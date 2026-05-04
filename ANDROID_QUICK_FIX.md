# Android Setup - Quick Fix

## The Error You're Seeing

```
ERROR: file:///data/data/com.termux/files/home/comp4901d-project-11 does not appear to be a Python project: neither 'setup.py' nor 'pyproject.toml' found.
```

## Why This Happens

The repository doesn't have a `setup.py` or `pyproject.toml` file because it's designed to run directly without installation as a package.

## The Fix

**Don't use `pip install -e .`** - Instead, just install the dependencies:

```bash
# Navigate to the repository
cd ~/comp4901d-project-11

# Install dependencies only
pip install -r requirements.txt
```

## Complete Android Setup (Corrected)

### 1. Install Termux Packages

```bash
pkg update
pkg upgrade

# Install Python, build tools, and rust
pkg install python python-pip git clang cmake binutils rust

# Install ALL Python packages with C extensions via pkg (NOT pip)
pkg install python-numpy python-psutil python-pandas python-matplotlib python-scipy python-pyyaml

pip install --upgrade pip
```

**Critical Notes:**
- ✅ `rust` is REQUIRED for `huggingface-hub` installation
- ✅ ALL packages with C extensions MUST use pkg:
  - `numpy`, `psutil`, `pandas`, `matplotlib`, `scipy`, `pyyaml`
- ❌ Do NOT try to install these via pip - they will fail to compile
- ✅ pkg provides pre-compiled ARM64 binaries

### 2. Install Python Dependencies

```bash
# Install pure Python packages via pip
pip install seaborn jinja2 python-dotenv

# Hugging Face Hub (requires rust from Step 1)
pip install huggingface-hub

# llama-cpp-python (compiles from source, takes 10-15 minutes)
pip install llama-cpp-python
```

**Important:**
- ❌ Do NOT pip install: numpy, psutil, pandas, matplotlib, scipy, pyyaml
- ✅ These are already installed via pkg in Step 1
- ✅ Only install pure Python packages via pip
- ✅ Skip pynvml (no NVIDIA GPU on Android)
- ✅ Skip pytest packages (not needed for benchmarks)

### 3. Clone and Setup Repository

```bash
cd ~
git clone <your-repo-url> comp4901d-project-11
cd comp4901d-project-11

# Install dependencies (NOT the package itself)
pip install -r requirements.txt
```

### 4. Set Hugging Face Token (Optional)

```bash
export HF_TOKEN="your_token_here"
echo 'export HF_TOKEN="your_token"' >> ~/.bashrc
```

### 5. Create Android Config

```bash
# Create config directory if it doesn't exist
mkdir -p configs

# Create Android-specific config
cat > configs/android_config.json << 'EOF'
{
  "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
  "models": {
    "Q2_K": "tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
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
  "save_formats": ["json", "csv", "markdown"],
  "visualization_dpi": 150
}
EOF
```

### 6. Grant Storage Permissions

```bash
termux-setup-storage
# Accept the permission prompt
```

### 7. Run Benchmark

```bash
# Make sure you're in the repository directory
cd ~/comp4901d-project-11

# Run the benchmark
python -m llm_benchmark --config configs/android_config.json
```

## Verification

Check that everything is installed:

```bash
# Check Python version (should be 3.10+)
python --version

# Check if llama-cpp-python is installed
python -c "import llama_cpp; print('llama-cpp-python:', llama_cpp.__version__)"

# Check if framework can be imported
python -c "from llm_benchmark.hardware.detector import HardwareDetector; print('Framework: OK')"

# Check available memory
free -h
```

## Expected Output

When you run the benchmark, you should see:

```
[INFO] Detecting hardware platform...
[INFO] Detected platform: android
[INFO] CPU: <your CPU model>
[INFO] RAM: <your RAM> GB total, <available> GB available
[INFO] GPU: Not detected
[INFO] Starting benchmark...
```

## Common Issues

### Issue: "error: can't find Rust compiler"

**When**: Installing `huggingface-hub`

**Solution**:
```bash
pkg install rust
pip install huggingface-hub
```

### Issue: "No module named 'psutil'" or psutil errors

**Solution**:
```bash
# Remove pip version if installed
pip uninstall psutil

# Install via pkg (pre-compiled for Android)
pkg install python-psutil
```

**Why**: The pip version doesn't compile correctly on Termux. Always use the pkg version.

### Issue: "No module named 'llama_cpp'"

```bash
pip install llama-cpp-python
```

### Issue: "Insufficient RAM to load model"

Use smaller model or quantization:
```json
{
  "models": {
    "Q2_K": "tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
  },
  "context_size": 256,
  "max_tokens": 20
}
```

### Issue: "Permission denied" accessing storage

```bash
termux-setup-storage
# Accept the permission prompt
```

### Issue: Device overheating

- Remove phone case
- Use a fan
- Increase sleep time: `"sleep_between_tests_s": 30`
- Reduce iterations: `"iterations": 1`

## Performance Expectations (Xiaomi 13T)

**TinyLlama 1.1B Q2_K:**
- Load Time: ~3-5 seconds
- Memory Usage: ~600-800 MB
- TTFT: ~200-400 ms
- Decode Speed: ~5-10 tokens/second

This is **3-5x slower** than desktop, which is normal for mobile CPUs!

## Need Help?

Check the full guide: `docs/ANDROID_SETUP.md`
