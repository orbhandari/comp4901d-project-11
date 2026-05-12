# GPU Monitoring Setup

This document explains how to set up GPU monitoring for the LLM benchmark framework.

## Overview

The benchmark framework supports GPU monitoring on NVIDIA GPUs using either:
- **nvidia-ml-py3** (recommended, modern)
- **pynvml** (legacy, for backward compatibility)

## Installation

### Option 1: nvidia-ml-py3 (Recommended)

```bash
pip install nvidia-ml-py3
```

### Option 2: pynvml (Legacy)

```bash
pip install pynvml
```

## Platform Support

### ✅ **Supported Platforms:**
- **Jetson Xavier NX**: Full GPU monitoring (temperature, power, memory, utilization)
- **Desktop/Server with NVIDIA GPU**: Full GPU monitoring
- **Linux x86 with NVIDIA GPU**: Full GPU monitoring

### ⚠️ **Limited/No Support:**
- **Android devices**: Most Android devices don't have NVIDIA GPUs
- **Intel/AMD integrated graphics**: Not supported by NVIDIA ML libraries
- **Apple Silicon (M1/M2)**: Not supported by NVIDIA ML libraries

## Testing GPU Detection

Test if GPU monitoring is working:

```bash
python -c "
try:
    import nvidia_ml_py3 as nvml
    nvml.nvmlInit()
    count = nvml.nvmlDeviceGetCount()
    print(f'✅ nvidia-ml-py3: Detected {count} NVIDIA GPU(s)')
    for i in range(count):
        handle = nvml.nvmlDeviceGetHandleByIndex(i)
        name = nvml.nvmlDeviceGetName(handle)
        print(f'  GPU {i}: {name}')
    nvml.nvmlShutdown()
except ImportError:
    try:
        import pynvml as nvml
        nvml.nvmlInit()
        count = nvml.nvmlDeviceGetCount()
        print(f'⚠️ pynvml: Detected {count} NVIDIA GPU(s) (consider upgrading to nvidia-ml-py3)')
        nvml.nvmlShutdown()
    except ImportError:
        print('❌ Neither nvidia-ml-py3 nor pynvml available')
    except Exception as e:
        print(f'❌ GPU detection failed: {e}')
except Exception as e:
    print(f'❌ GPU detection failed: {e}')
"
```

## Expected Behavior by Platform

### Android Devices
```
Has GPU: False
No GPU detected (expected for most Android devices)
```

### Jetson Xavier NX
```
Has GPU: True
GPU: NVIDIA Tegra X1 (or similar)
GPU Memory: 8.00 GB
```

### Desktop with NVIDIA GPU
```
Has GPU: True  
GPU: NVIDIA GeForce RTX 4090 (or similar)
GPU Memory: 24.00 GB
```

### Systems without NVIDIA GPU
```
Has GPU: False
No GPU detected (expected for non-NVIDIA systems)
```

## Troubleshooting

### "No GPU detected" on system with NVIDIA GPU

1. **Check NVIDIA drivers:**
   ```bash
   nvidia-smi
   ```

2. **Install CUDA toolkit** (if needed):
   ```bash
   # Ubuntu/Debian
   sudo apt install nvidia-cuda-toolkit
   
   # Or download from NVIDIA website
   ```

3. **Install nvidia-ml-py3:**
   ```bash
   pip install nvidia-ml-py3
   ```

4. **Test detection:**
   ```bash
   python -c "import nvidia_ml_py3 as nvml; nvml.nvmlInit(); print('GPU detected')"
   ```

### "ImportError: No module named nvidia_ml_py3"

```bash
pip install nvidia-ml-py3
```

### "NVML library not found"

This usually means NVIDIA drivers are not installed or not working:

1. **Install NVIDIA drivers:**
   ```bash
   # Ubuntu/Debian
   sudo apt install nvidia-driver-535  # or latest version
   
   # Reboot after installation
   sudo reboot
   ```

2. **Verify drivers:**
   ```bash
   nvidia-smi
   ```

## Migration from pynvml to nvidia-ml-py3

The benchmark framework automatically detects and uses nvidia-ml-py3 if available, falling back to pynvml for compatibility.

**To upgrade:**
```bash
pip uninstall pynvml
pip install nvidia-ml-py3
```

**API compatibility:** nvidia-ml-py3 provides the same API as pynvml, so no code changes are needed.

## Android-Specific Notes

Most Android devices use ARM-based SoCs (Snapdragon, Exynos, etc.) that don't have NVIDIA GPUs. The benchmark framework correctly detects this and runs in CPU-only mode.

**Expected behavior on Android:**
- GPU monitoring: Disabled
- GPU acceleration: Not available  
- CPU-only inference: ✅ Supported
- Thermal monitoring: ✅ Supported via `/sys/class/thermal`

This is normal and expected behavior for Android devices.