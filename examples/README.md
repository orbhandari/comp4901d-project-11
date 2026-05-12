# LLM Benchmark Framework - Examples

This directory contains example scripts demonstrating how to use the LLM Benchmark Framework for various benchmarking scenarios.

## Available Examples

### 1. Basic Quantization Profiling (`basic_quantization_profiling.py`)

Demonstrates how to compare performance across different quantization levels (Q8_0, Q4_0, Q2_K).

**Use Case**: Finding the optimal quantization level for your hardware and use case.

**Usage**:
```bash
python examples/basic_quantization_profiling.py
```

**What it does**:
- Downloads and caches models (first run only)
- Profiles each quantization level
- Measures load time, memory usage, TTFT, and throughput
- Provides recommendations for best speed, memory, and latency

**Requirements**:
- Internet connection (first run)
- At least 8GB RAM
- ~5-10 minutes runtime

### 2. Ablation Studies (`ablation_studies.py`)

Demonstrates how to measure the impact of optimization strategies like KV cache and prompt caching.

**Use Case**: Understanding which optimizations provide the most benefit for your workload.

**Usage**:
```bash
python examples/ablation_studies.py
```

**What it does**:
- Tests KV cache strategies (RAM vs Disk, Cold vs Warm)
- Tests prompt caching with different prefix lengths
- Measures TTFT improvements and cache hit rates
- Quantifies memory overhead of caching

**Requirements**:
- Internet connection (first run)
- At least 8GB RAM
- ~10-15 minutes runtime

### 3. Batch Processing Tests (`batch_processing.py`)

Demonstrates how to find the optimal batch size for throughput-oriented workloads.

**Use Case**: Optimizing for high-throughput scenarios with multiple concurrent requests.

**Usage**:
```bash
python examples/batch_processing.py
```

**What it does**:
- Tests different batch sizes (1, 2, 4, 8)
- Measures aggregate throughput and per-prompt latency
- Tracks memory scaling with batch size
- Calculates throughput scaling efficiency

**Requirements**:
- Internet connection (first run)
- At least 8GB RAM
- ~10-15 minutes runtime

## Example Configurations

The `configs/` directory contains platform-specific example configurations:

### x86 Linux Configuration (`configs/x86_linux_example.json`)

Optimized for x86 Linux systems with CPU-only inference:
- Multiple quantization levels (Q8_0, Q4_0, Q2_K)
- Full ablation studies enabled
- Batch testing with sizes up to 16
- Thermal monitoring enabled
- 5 iterations for statistical validity

**Recommended for**:
- Desktop/server systems with 16GB+ RAM
- CPU-only inference
- Comprehensive benchmarking

**Usage**:
```bash
python -m llm_benchmark --config configs/x86_linux_example.json
```

### Jetson Xavier NX Configuration (`configs/jetson_xavier_nx_example.json`)

Optimized for NVIDIA Jetson Xavier NX with GPU acceleration:
- Smaller quantization levels (Q4_0, Q2_K) for limited RAM
- Reduced context size (1024) and batch size (256)
- GPU layer offloading automatically configured
- Lower thermal threshold (65°C) for passive cooling
- Increased sleep between tests (10s) for thermal stabilization
- RAM-only KV cache (limited disk I/O)

**Recommended for**:
- Jetson Xavier NX edge devices
- GPU-accelerated inference
- Thermal-constrained environments

**Usage**:
```bash
python -m llm_benchmark --config configs/jetson_xavier_nx_example.json
```

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run a simple example**:
   ```bash
   python examples/basic_quantization_profiling.py
   ```

3. **Run with custom configuration**:
   ```bash
   python -m llm_benchmark --config configs/x86_linux_example.json
   ```

4. **Override configuration parameters**:
   ```bash
   python -m llm_benchmark --config configs/x86_linux_example.json \
     --iterations 10 \
     --disable-ablation-studies
   ```

## Tips

### For Fast Testing
- Use `iterations=1` and `warmup_runs=0`
- Disable ablation studies and batch testing
- Use smaller models (TinyLlama-1.1B)
- Reduce `max_tokens` to 10-20

### For Production Benchmarks
- Use `iterations=5` or more for statistical validity
- Enable all test types
- Use realistic models for your use case
- Increase `max_tokens` to match your workload

### For Memory-Constrained Systems
- Reduce `context_size` (try 512 or 1024)
- Reduce `batch_size` (try 128 or 256)
- Use smaller quantization levels (Q4_0, Q2_K)
- Disable batch testing or use smaller batch sizes

### For Thermal-Constrained Systems
- Increase `sleep_between_tests_s` (try 10-15s)
- Lower `thermal_stabilization_threshold_c` (try 60-65°C)
- Reduce `iterations` to minimize heat buildup
- Run tests during cooler times of day

## Troubleshooting

### Model Download Fails
- Check internet connection
- Verify Hugging Face repository exists
- Set `HF_TOKEN` environment variable for private models
- Check disk space in model cache directory

### Out of Memory Errors
- Reduce `context_size` and `batch_size`
- Use smaller quantization levels
- Close other applications
- Disable batch testing

### Inference Timeout
- Increase `inference_timeout_s`
- Reduce `max_tokens`
- Use faster quantization levels (Q4_0 instead of Q8_0)
- Check system load

### Thermal Throttling
- Increase `sleep_between_tests_s`
- Lower `thermal_stabilization_threshold_c`
- Improve cooling (add fans, clean dust)
- Reduce workload intensity

## Next Steps

- Read the [User Guide](../docs/USER_GUIDE.md) for detailed documentation
- Check the [Troubleshooting Guide](../docs/TROUBLESHOOTING.md) for common issues
- Explore the [CI/CD Setup](../docs/CI_CD_SETUP.md) for automated testing
- Review the main [README](../README.md) for project overview

## Contributing

Found a bug or have a suggestion? Please open an issue on GitHub!

Want to add a new example? Pull requests are welcome!
