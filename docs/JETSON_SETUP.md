# NVIDIA Jetson Setup Guide

This guide helps you set up GPU monitoring on NVIDIA Jetson devices (Xavier NX, Nano, Orin, etc.).

## Quick Diagnosis

Run this diagnostic script on your Jetson device:

```bash
python scripts/test_jetson_gpu.py
```

This will test all GPU detection methods and provide specific recommendations.

## Common Issues and Solutions

### Issue 1: "No GPU detected" on Jetson

**Symptoms:**
```
Has GPU: False
No GPU detected
```

**Diagnosis Steps:**

1. **Check if nvidia-smi works:**
   ```bash
   nvidia-smi
   ```
   
   If this fails, your NVIDIA drivers aren't working properly.

2. **Check JetPack installation:**
   ```bash
   dpkg -l | grep nvidia-jetpack
   ```

3. **Test NVML libraries:**
   ```bash
   python -c "
   try:
       import pynvml
       pynvml.nvmlInit()
       print('✅ pynvml works')
   except Exception as e:
       print(f'❌ pynvml failed: {e}')
   
   try:
       import nvidia_ml_py3 as nvml
       nvml.nvmlInit()
       print('✅ nvidia-ml-py3 works')
   except Exception as e:
       print(f'❌ nvidia-ml-py3 failed: {e}')
   "
   ```

### Issue 2: pynvml works but nvidia-ml-py3 doesn't

**This is common on Jetson devices due to ARM architecture and older CUDA versions.**

**Solution 1: Use pynvml (recommended for Jetson)**
```bash
pip install pynvml
```

**Solution 2: Try nvidia-ml-py3 with specific version**
```bash
pip install nvidia-ml-py3==7.352.0
```

**The benchmark framework automatically prefers pynvml on Jetson devices.**

### Issue 3: Both libraries fail

**Symptoms:**
```
❌ pynvml failed: NVML Shared Library Not Found
❌ nvidia-ml-py3 failed: NVML Shared Library Not Found
```

**Solutions:**

1. **Reinstall JetPack:**
   ```bash
   sudo apt update
   sudo apt install nvidia-jetpack
   ```

2. **Check CUDA installation:**
   ```bash
   nvcc --version
   ls /usr/local/cuda*/lib64/libnvidia-ml.so*
   ```

3. **Set library path (if needed):**
   ```bash
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

4. **Reboot after driver installation:**
   ```bash
   sudo reboot
   ```

## Jetson-Specific Optimizations

The benchmark framework includes Jetson-specific optimizations:

### 1. **Library Priority**
- On Jetson: Tries `pynvml` first (more reliable)
- On other systems: Tries `nvidia-ml-py3` first

### 2. **nvidia-smi Fallback**
- If NVML libraries fail, uses `nvidia-smi` as fallback
- Extracts GPU name and memory from nvidia-smi output

### 3. **Device Detection**
- Automatically detects Jetson devices via `/proc/device-tree/model`
- Applies Jetson-specific GPU detection logic

## Installation Recommendations

### For Jetson Xavier NX / Nano / Orin:

1. **Install JetPack (includes NVIDIA drivers):**
   ```bash
   sudo apt update
   sudo apt install nvidia-jetpack
   ```

2. **Install Python NVML library:**
   ```bash
   # Option 1: pynvml (most reliable on Jetson)
   pip install pynvml
   
   # Option 2: nvidia-ml-py3 (if you prefer the modern library)
   pip install nvidia-ml-py3
   ```

3. **Verify installation:**
   ```bash
   python scripts/test_jetson_gpu.py
   ```

4. **Run benchmark:**
   ```bash
   python -m llm_benchmark --config configs/jetson_config.json
   ```

## Expected GPU Detection Results

### Jetson Xavier NX:
```
✅ Hardware detection completed!
Platform: jetson_xavier_nx
CPU: ARM Cortex-A78AE (8 cores)
RAM: 8.00 GB
Has GPU: True
GPU: NVIDIA Tegra X1
GPU Memory: 8.00 GB (shared with system)
```

### Jetson Nano:
```
✅ Hardware detection completed!
Platform: jetson_xavier_nx  # (detected as same family)
CPU: ARM Cortex-A57 (4 cores)
RAM: 4.00 GB
Has GPU: True
GPU: NVIDIA Tegra X1
GPU Memory: 4.00 GB (shared with system)
```

### Jetson Orin:
```
✅ Hardware detection completed!
Platform: jetson_xavier_nx  # (detected as same family)
CPU: ARM Cortex-A78AE (12 cores)
RAM: 32.00 GB
Has GPU: True
GPU: NVIDIA Ampere
GPU Memory: 32.00 GB (shared with system)
```

## Troubleshooting Commands

### Check Jetson Model:
```bash
cat /proc/device-tree/model
```

### Check CUDA Version:
```bash
nvcc --version
cat /usr/local/cuda/version.txt  # if exists
```

### Check JetPack Version:
```bash
dpkg -l | grep nvidia-jetpack
sudo apt show nvidia-jetpack
```

### Check GPU Memory:
```bash
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
```

### Check GPU Temperature:
```bash
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits
```

### Monitor GPU in Real-time:
```bash
watch -n 1 nvidia-smi
```

## Performance Notes

- **Shared Memory**: Jetson devices use unified memory (GPU shares system RAM)
- **Thermal Throttling**: Monitor temperature during benchmarks
- **Power Modes**: Use `sudo nvpmodel -m 0` for maximum performance
- **Fan Control**: Ensure adequate cooling during intensive benchmarks

## Getting Help

If GPU detection still fails after following this guide:

1. **Run the diagnostic script:**
   ```bash
   python scripts/test_jetson_gpu.py > jetson_diagnostic.txt
   ```

2. **Include this information in bug reports:**
   - Jetson model (from `/proc/device-tree/model`)
   - JetPack version (from `dpkg -l | grep nvidia-jetpack`)
   - Output of `nvidia-smi`
   - Output of diagnostic script
   - Python version and installed packages (`pip list | grep -i nvidia`)

The benchmark framework is designed to work reliably on Jetson devices with proper GPU detection and monitoring.