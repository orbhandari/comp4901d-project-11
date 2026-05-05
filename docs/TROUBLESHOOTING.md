# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the LLM Benchmark Framework.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Model Download Issues](#model-download-issues)
3. [Memory Issues](#memory-issues)
4. [GPU Issues](#gpu-issues)
5. [Performance Issues](#performance-issues)
6. [Platform-Specific Issues](#platform-specific-issues)
7. [Diagnostic Commands](#diagnostic-commands)

## Installation Issues

### Missing Dependencies

**Symptom**: Error message "Missing required dependencies" on startup.

**Diagnosis**:
```bash
python -m llm_benchmark --help
```

**Solution**:
```bash
# Install all required dependencies
pip install -r requirements.txt

# Or install individually
pip install llama-cpp-python psutil pandas matplotlib seaborn numpy scipy huggingface-hub
```

**For GPU support**:
```bash
pip install pynvml
```

### llama-cpp-python Installation Fails

**Symptom**: Error during `pip install llama-cpp-python`.

**Common Causes**:
- Missing C++ compiler
- Missing CMake
- CUDA version mismatch (for GPU support)

**Solution for CPU-only**:
```bash
pip install llama-cpp-python --no-cache-dir
```

**Solution for GPU support**:
```bash
# Ensure CUDA is installed
nvcc --version

# Install with CUDA support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --no-cache-dir
```

**On Ubuntu/Debian**:
```bash
# Install build dependencies
sudo apt-get update
sudo apt-get install build-essential cmake

# Then install llama-cpp-python
pip install llama-cpp-python
```

### Import Errors

**Symptom**: `ModuleNotFoundError` or `ImportError`.

**Diagnosis**:
```bash
python -c "import llm_benchmark"
```

**Solution**:
```bash
# Ensure you're in the correct directory
cd /path/to/llm-benchmark

# Ensure virtual environment is activated
source .venv/bin/activate  # Linux/Mac

# Reinstall in development mode
pip install -e .
```

## Model Download Issues

### Network Timeout

**Symptom**: "Failed to download model" after timeout.

**Diagnosis**:
```bash
# Test Hugging Face connectivity
curl -I https://huggingface.co
```

**Solution**:
- Check internet connection
- Try again (framework has automatic retry with exponential backoff)
- Download model manually:

```bash
# Download manually using huggingface-cli
huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf \
  --local-dir ~/.cache/llm_benchmark/models

# Then run benchmark with local path
python -m llm_benchmark \
  --repo-id "TheBloke/Llama-2-7B-GGUF" \
  --models "Q4_K_M:~/.cache/llm_benchmark/models/llama-2-7b.Q4_K_M.gguf"
```

### Authentication Failed

**Symptom**: "Authentication failed" or "401 Unauthorized".

**Diagnosis**:
```bash
# Check if HF_TOKEN is set
echo $HF_TOKEN
```

**Solution**:
```bash
# Set Hugging Face token
export HF_TOKEN="your_huggingface_token"

# Or add to configuration file
{
  "hf_token": "your_huggingface_token",
  ...
}
```

**Get token from**: https://huggingface.co/settings/tokens

### Disk Space Exhausted

**Symptom**: "No space left on device" during download.

**Diagnosis**:
```bash
# Check available disk space
df -h ~/.cache/llm_benchmark/models
```

**Solution**:
```bash
# Clean up old models
rm -rf ~/.cache/llm_benchmark/models/*

# Or change cache directory
python -m llm_benchmark \
  --model-cache-dir /path/to/larger/disk \
  ...
```

### Corrupted Download

**Symptom**: "Checksum verification failed" or "Invalid GGUF format".

**Diagnosis**:
```bash
# Check file size
ls -lh ~/.cache/llm_benchmark/models/

# Verify GGUF magic bytes
head -c 4 ~/.cache/llm_benchmark/models/model.gguf | xxd
# Should show: 47 47 55 46 (GGUF)
```

**Solution**:
```bash
# Delete corrupted file
rm ~/.cache/llm_benchmark/models/corrupted_model.gguf

# Re-run benchmark (will re-download)
python -m llm_benchmark --config config.json
```

## Memory Issues

### Insufficient RAM

**Symptom**: "Insufficient RAM" error or system freezing.

**Diagnosis**:
```bash
# Check available memory
free -h

# Check memory usage during benchmark
watch -n 1 free -h
```

**Solutions**:

**1. Use smaller quantization**:
```bash
# Use Q4_0 or Q2_K instead of Q8_0
python -m llm_benchmark \
  --models "Q4_0:model.Q4_0.gguf" \
  ...
```

**2. Reduce context size**:
```json
{
  "context_size": 1024,  // Instead of 2048 or 4096
  ...
}
```

**3. Close other applications**:
```bash
# Check memory-hungry processes
ps aux --sort=-%mem | head -n 10

# Kill unnecessary processes
kill <PID>
```

**4. Use swap space** (not recommended for performance):
```bash
# Check swap
swapon --show

# Add swap if needed (temporary)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Memory Leak

**Symptom**: Memory usage increases over time, eventually causing OOM.

**Diagnosis**:
```bash
# Monitor memory during benchmark
watch -n 1 'ps aux | grep python'
```

**Solution**:
- Framework automatically calls `gc.collect()` between tests
- If issue persists, reduce `iterations` and run multiple separate benchmarks
- Report issue on GitHub with logs

### Memory Fragmentation

**Symptom**: "Cannot allocate memory" despite sufficient free RAM.

**Diagnosis**:
```bash
# Check memory fragmentation
cat /proc/buddyinfo
```

**Solution**:
```bash
# Restart system to defragment memory
sudo reboot

# Or use smaller models/quantizations
```

## GPU Issues

### GPU Not Detected

**Symptom**: "GPU: Not Available" in hardware info, or CPU-only inference.

**Diagnosis**:
```bash
# Check if NVIDIA GPU is detected
nvidia-smi

# Check if nvidia-ml-py3 or pynvml is installed
python -c "
try:
    import nvidia_ml_py3 as nvml
    nvml.nvmlInit()
    print('GPU detected with nvidia-ml-py3')
except ImportError:
    try:
        import pynvml as nvml
        nvml.nvmlInit()
        print('GPU detected with pynvml (consider upgrading to nvidia-ml-py3)')
    except ImportError:
        print('Neither nvidia-ml-py3 nor pynvml available')
    except Exception as e:
        print(f'GPU detection failed: {e}')
except Exception as e:
    print(f'GPU detection failed: {e}')
"

# Check CUDA version
nvcc --version
```

**Solutions**:

**1. Install NVIDIA drivers**:
```bash
# Ubuntu/Debian
sudo apt-get install nvidia-driver-535  # Or latest version

# Check installation
nvidia-smi
```

**2. Install pynvml**:
```bash
pip install pynvml
```

**3. Reinstall llama-cpp-python with CUDA support**:
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --no-cache-dir --force-reinstall
```

### GPU Out of Memory

**Symptom**: "CUDA out of memory" error.

**Diagnosis**:
```bash
# Check GPU memory usage
nvidia-smi

# Monitor GPU memory during benchmark
watch -n 1 nvidia-smi
```

**Solutions**:

**1. Framework automatic handling**:
- Framework automatically reduces GPU layers and retries
- Falls back to CPU if GPU completely exhausted

**2. Manual configuration**:
```json
{
  "n_gpu_layers": 20,  // Reduce from default
  ...
}
```

**3. Reduce context size**:
```json
{
  "context_size": 1024,  // Instead of 2048
  ...
}
```

**4. Use smaller quantization**:
```bash
# Q4_0 uses ~50% less GPU memory than Q8_0
python -m llm_benchmark \
  --models "Q4_0:model.Q4_0.gguf" \
  ...
```

**5. Close other GPU applications**:
```bash
# Check GPU processes
nvidia-smi

# Kill GPU process
sudo kill -9 <PID>
```

### GPU Utilization Low

**Symptom**: GPU utilization < 50% during inference.

**Diagnosis**:
```bash
# Monitor GPU utilization
nvidia-smi dmon -s u
```

**Possible Causes**:
- Not enough GPU layers offloaded
- Batch size too small
- CPU bottleneck (data transfer)

**Solutions**:

**1. Increase GPU layers**:
```json
{
  "n_gpu_layers": 40,  // Increase from default
  ...
}
```

**2. Increase batch size**:
```json
{
  "batch_size": 1024,  // Increase from 512
  ...
}
```

**3. Check CPU bottleneck**:
```bash
# Monitor CPU usage
htop
```

### CUDA Version Mismatch

**Symptom**: "CUDA driver version is insufficient" or similar.

**Diagnosis**:
```bash
# Check CUDA driver version
nvidia-smi

# Check CUDA toolkit version
nvcc --version

# Check llama-cpp-python CUDA version
python -c "import llama_cpp; print(llama_cpp.__version__)"
```

**Solution**:
```bash
# Reinstall llama-cpp-python matching your CUDA version
# For CUDA 11.8:
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --no-cache-dir --force-reinstall

# Or update NVIDIA drivers
sudo apt-get update
sudo apt-get install nvidia-driver-535
```

## Performance Issues

### Slow Inference

**Symptom**: Decode throughput < 10 tokens/second.

**Diagnosis**:
```bash
# Check CPU usage
htop

# Check if thermal throttling
sensors  # Install lm-sensors if needed

# Check system load
uptime
```

**Solutions**:

**1. Use GPU acceleration**:
- Ensure GPU is detected and enabled
- Check GPU utilization with `nvidia-smi`

**2. Use faster quantization**:
```bash
# Q4_0 is faster than Q8_0
python -m llm_benchmark \
  --models "Q4_0:model.Q4_0.gguf" \
  ...
```

**3. Reduce context size**:
```json
{
  "context_size": 1024,  // Smaller context = faster
  ...
}
```

**4. Check thermal throttling**:
```bash
# Monitor CPU temperature
watch -n 1 sensors

# If throttling, improve cooling or reduce workload
```

**5. Close background applications**:
```bash
# Check CPU-intensive processes
ps aux --sort=-%cpu | head -n 10
```

### Inference Timeout

**Symptom**: "Inference timed out after 300s".

**Diagnosis**:
```bash
# Check if system is overloaded
uptime

# Check if thermal throttling
sensors
```

**Solutions**:

**1. Increase timeout**:
```json
{
  "inference_timeout_s": 600,  // Increase from 300
  ...
}
```

**2. Reduce max_tokens**:
```json
{
  "max_tokens": 50,  // Reduce from 100
  ...
}
```

**3. Use faster quantization**:
```bash
python -m llm_benchmark \
  --models "Q4_0:model.Q4_0.gguf" \
  ...
```

### Thermal Throttling

**Symptom**: "Thermal throttling detected" warning, or performance degradation over time.

**Diagnosis**:
```bash
# Monitor CPU temperature
watch -n 1 sensors

# Check throttling status (Intel)
cat /sys/devices/system/cpu/cpu*/thermal_throttle/core_throttle_count

# Check throttling status (Jetson)
cat /sys/devices/virtual/thermal/thermal_zone*/temp
```

**Solutions**:

**1. Framework automatic handling**:
- Framework waits for temperature to drop below threshold
- Increases sleep delays between tests

**2. Improve cooling**:
- Clean dust from fans and heatsinks
- Improve airflow around system
- Use external cooling (fan, cooling pad)

**3. Reduce workload**:
```json
{
  "iterations": 3,  // Reduce from 10
  "sleep_between_tests_s": 10,  // Increase from 5
  "enable_ablation_studies": false,  // Disable optional tests
  ...
}
```

**4. Lower thermal threshold**:
```json
{
  "thermal_stabilization_threshold_c": 60.0,  // Lower from 70.0
  ...
}
```

## Platform-Specific Issues

### Linux x86

#### AVX/AVX2 Not Detected

**Symptom**: Slow CPU inference, AVX not in CPU features.

**Diagnosis**:
```bash
# Check CPU features
cat /proc/cpuinfo | grep flags | head -n 1
```

**Solution**:
- Ensure CPU supports AVX/AVX2 (most modern CPUs do)
- Update BIOS if AVX is disabled
- Reinstall llama-cpp-python to detect CPU features

#### Permission Denied for Sensors

**Symptom**: "Permission denied" when accessing `/sys/class/thermal`.

**Solution**:
```bash
# Run with sudo (not recommended)
sudo python -m llm_benchmark ...

# Or add user to appropriate group
sudo usermod -a -G video $USER
# Log out and log back in
```

### NVIDIA Jetson Xavier NX

#### JetPack Version Issues

**Symptom**: CUDA errors or GPU not detected.

**Diagnosis**:
```bash
# Check JetPack version
cat /etc/nv_tegra_release

# Check CUDA version
nvcc --version
```

**Solution**:
- Ensure JetPack 4.6+ or 5.0+ is installed
- Update JetPack if needed: https://developer.nvidia.com/embedded/jetpack

#### Power Mode Not Optimal

**Symptom**: Lower performance than expected.

**Diagnosis**:
```bash
# Check current power mode
sudo nvpmodel -q
```

**Solution**:
```bash
# Set to maximum performance mode (MAXN)
sudo nvpmodel -m 0

# Enable jetson_clocks for maximum clocks
sudo jetson_clocks
```

#### Thermal Throttling on Jetson

**Symptom**: Frequent thermal throttling warnings.

**Diagnosis**:
```bash
# Monitor Jetson temperatures
watch -n 1 cat /sys/devices/virtual/thermal/thermal_zone*/temp
```

**Solution**:
- Add heatsink and fan to Jetson module
- Reduce power mode: `sudo nvpmodel -m 2` (15W mode)
- Reduce workload (fewer iterations, disable ablation)

## Diagnostic Commands

### System Information

```bash
# OS and kernel
uname -a

# CPU information
lscpu

# Memory information
free -h

# Disk space
df -h

# GPU information (NVIDIA)
nvidia-smi

# Jetson information
cat /etc/nv_tegra_release
```

### Python Environment

```bash
# Python version
python --version

# Installed packages
pip list

# Check specific package
pip show llama-cpp-python

# Check import
python -c "import llm_benchmark; print('OK')"
```

### Hardware Detection

```bash
# Test hardware detection
python -c "
from llm_benchmark.hardware import HardwareDetector
hw_info = HardwareDetector.detect()
print(f'Platform: {hw_info.os_type}')
print(f'CPU: {hw_info.cpu_model} ({hw_info.cpu_cores} cores)')
print(f'RAM: {hw_info.total_ram_gb:.2f} GB')
print(f'GPU: {hw_info.gpu_model if hw_info.has_gpu else \"Not available\"}')
"
```

### Model Validation

```bash
# Check GGUF file
python -c "
import struct
with open('model.gguf', 'rb') as f:
    magic = f.read(4)
    print(f'Magic: {magic}')
    print(f'Valid GGUF: {magic == b\"GGUF\"}')
"

# Check file size
ls -lh model.gguf
```

### Benchmark Logs

```bash
# View recent logs
tail -f benchmark_results/run_*/logs/benchmark.log

# Search for errors
grep -i error benchmark_results/run_*/logs/benchmark.log

# Search for warnings
grep -i warning benchmark_results/run_*/logs/benchmark.log
```

## Getting More Help

If you're still experiencing issues:

1. **Check logs**: Review `benchmark_results/run_*/logs/benchmark.log`
2. **Enable debug logging**: Set `LOG_LEVEL=DEBUG`
3. **Test minimal configuration**: Single model, 1 iteration, no ablation
4. **Check GitHub issues**: https://github.com/yourusername/llm-benchmark/issues
5. **Create new issue**: Include logs, hardware info, and configuration
6. **Join Discord**: https://discord.gg/llm-benchmark

### Information to Include in Bug Reports

When reporting issues, please include:

1. **Hardware information**:
   ```bash
   python -c "from llm_benchmark.hardware import HardwareDetector; print(HardwareDetector.detect())"
   ```

2. **Software versions**:
   ```bash
   python --version
   pip list | grep -E "(llama-cpp|psutil|pandas|numpy)"
   ```

3. **Configuration used**:
   ```bash
   cat config.json
   ```

4. **Error logs**:
   ```bash
   cat benchmark_results/run_*/logs/benchmark.log
   ```

5. **Steps to reproduce**: Exact commands used to trigger the issue
