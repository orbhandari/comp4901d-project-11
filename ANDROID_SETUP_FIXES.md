# Android Setup - Critical Fixes

## Issues Discovered During Real Android Testing

### Issue 1: Rust Compiler Required for huggingface-hub

**Error Message:**
```
error: can't find Rust compiler
```

**Root Cause:**
The `huggingface-hub` package includes `hf_xet` which requires Rust to compile native extensions.

**Fix:**
```bash
pkg install rust
```

**Updated Installation Order:**
```bash
# Install rust BEFORE installing huggingface-hub
pkg install python python-pip git clang cmake binutils rust
pip install huggingface-hub
```

### Issue 2: psutil Must Be Installed via pkg, Not pip

**Error Message:**
```
No module named 'psutil'
# OR
ImportError: cannot import name 'psutil'
# OR
Compilation errors when installing psutil via pip
```

**Root Cause:**
The pip version of `psutil` tries to compile from source on Android/Termux, which often fails or produces incompatible binaries. Termux provides a pre-compiled version via `pkg`.

**Fix:**
```bash
# Remove pip version if installed
pip uninstall psutil

# Install pre-compiled version via pkg
pkg install python-psutil
```

**Why This Matters:**
- The pkg version is pre-compiled for Android ARM64 architecture
- The pip version tries to compile C extensions which often fail on Termux
- The framework requires psutil for memory monitoring

### Issue 3: numpy Should Also Use pkg

**Recommendation:**
While numpy CAN be installed via pip, the pkg version is more reliable:

```bash
pkg install python-numpy
```

**Benefits:**
- Pre-compiled for Android
- Faster installation
- Better compatibility with other Termux packages

## Corrected Installation Sequence

### Complete Working Installation

```bash
# 1. Update Termux
pkg update && pkg upgrade

# 2. Install ALL required packages (including rust)
pkg install python python-pip git clang cmake binutils rust

# 3. Install system packages via pkg (NOT pip)
pkg install python-numpy python-psutil

# 4. Update pip
pip install --upgrade pip

# 5. Install Python packages via pip (NOT psutil or numpy)
pip install pandas matplotlib seaborn scipy jinja2

# 6. Install huggingface-hub (rust is now available)
pip install huggingface-hub

# 7. Install llama-cpp-python (takes 10-15 minutes)
pip install llama-cpp-python

# 8. Clone repository
cd ~
git clone <repo-url> comp4901d-project-11
cd comp4901d-project-11

# 9. Install remaining dependencies
pip install -r requirements.txt

# 10. Grant storage permissions
termux-setup-storage

# 11. Run benchmark
python -m llm_benchmark --config configs/android_example.json
```

## Key Takeaways

### ✅ DO:
- Install `rust` via pkg before installing huggingface-hub
- Install `psutil` via `pkg install python-psutil`
- Install `numpy` via `pkg install python-numpy`
- Use pip for pure Python packages (pandas, matplotlib, etc.)

### ❌ DON'T:
- Try to `pip install psutil` on Android
- Skip installing rust
- Use `pip install -e .` (repository has no setup.py)
- Install numpy via pip (pkg version is better)

## Verification Commands

After installation, verify everything works:

```bash
# Check rust
rustc --version

# Check psutil (should import without errors)
python -c "import psutil; print('psutil:', psutil.__version__)"

# Check numpy
python -c "import numpy; print('numpy:', numpy.__version__)"

# Check huggingface-hub
python -c "import huggingface_hub; print('huggingface-hub:', huggingface_hub.__version__)"

# Check llama-cpp-python
python -c "import llama_cpp; print('llama-cpp-python:', llama_cpp.__version__)"

# Check framework
python -c "from llm_benchmark.hardware.detector import HardwareDetector; print('Framework: OK')"
```

## Documentation Updated

The following files have been updated with these fixes:

1. ✅ `docs/ANDROID_SETUP.md` - Complete setup guide
2. ✅ `ANDROID_QUICK_FIX.md` - Quick reference
3. ✅ `ANDROID_SETUP_FIXES.md` - This document

## Testing Status

- ✅ Tested on: Xiaomi 13T (HyperOS, Android 15, 12GB RAM)
- ✅ All dependencies install successfully
- ✅ Framework imports without errors
- ⏳ Benchmark execution: In progress

## Next Steps

After successful installation:

1. Create Android config: `configs/android_config.json`
2. Grant storage permissions: `termux-setup-storage`
3. Run benchmark: `python -m llm_benchmark --config configs/android_config.json`
4. Monitor temperature: `cat /sys/class/thermal/thermal_zone0/temp`
5. Check memory: `free -h`

## Support

If you encounter other issues:

1. Check Termux logs: `logcat | grep termux`
2. Verify Python version: `python --version` (should be 3.10+)
3. Check available memory: `free -h`
4. Verify all packages: Run verification commands above
5. Report with device model, Android version, and error logs
