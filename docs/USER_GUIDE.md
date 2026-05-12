# LLM Benchmark Framework - User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Command-Line Interface](#command-line-interface)
6. [Understanding Results](#understanding-results)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)
9. [Platform-Specific Notes](#platform-specific-notes)

## Introduction

The LLM Benchmark Framework is a comprehensive tool for evaluating large language model (LLM) inference performance across different hardware platforms, quantization levels, and optimization strategies.

### Key Features

- **Cross-Platform Support**: Works on Linux x86 and NVIDIA Jetson Xavier NX
- **Automatic Hardware Detection**: Detects CPU, GPU, memory, and sensors
- **Multi-Level Quantization Profiling**: Compare Q8_0, Q4_0, Q4_K_M, Q2_K formats
- **GPU Acceleration**: Automatic GPU offloading with fallback to CPU
- **Ablation Studies**: Measure KV cache and prompt caching effectiveness
- **Batch Processing**: Test throughput optimization with different batch sizes
- **Statistical Validation**: Confidence intervals, t-tests, outlier detection
- **Interactive Reports**: Professional HTML reports with embedded visualizations

### What Gets Measured

- **Time to First Token (TTFT)**: Latency from prompt submission to first output
- **Prefill Throughput**: Tokens per second during prompt processing
- **Decode Throughput**: Tokens per second during generation
- **Memory Usage**: Peak RAM and GPU memory consumption
- **Model Load Time**: Time to load model into memory
- **Thermal Metrics**: CPU/GPU temperature during inference (if sensors available)
- **Power Consumption**: Power usage during inference (if sensors available)

## Installation

### Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Linux (x86_64 or ARM64 for Jetson)
- **RAM**: Minimum 8GB (16GB+ recommended)
- **Disk Space**: 10GB+ for models and results

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-benchmark.git
cd llm-benchmark

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m llm_benchmark --help
```

### Required Dependencies

The framework will check for these dependencies on startup:

- `llama-cpp-python` - LLM inference engine
- `psutil` - System monitoring
- `pandas` - Data processing
- `matplotlib` - Visualization
- `seaborn` - Enhanced plotting
- `numpy` - Numerical computing
- `scipy` - Statistical analysis
- `huggingface-hub` - Model downloading

### Optional Dependencies

- `pynvml` - GPU monitoring (for NVIDIA GPUs)
- `pyyaml` - YAML configuration files

## Quick Start

### Basic Usage

Run a benchmark with default settings:

```bash
python -m llm_benchmark \
  --repo-id "TheBloke/Llama-2-7B-GGUF" \
  --models "Q8_0:llama-2-7b.Q8_0.gguf,Q4_0:llama-2-7b.Q4_0.gguf" \
  --iterations 3 \
  --output-dir ./results
```

This will:
1. Download the specified models from Hugging Face
2. Detect your hardware platform
3. Run quantization profiling for Q8_0 and Q4_0
4. Generate visualizations and reports
5. Save results to `./results/`

### Using a Configuration File

Create a configuration file `config.json`:

```json
{
  "repo_id": "TheBloke/Llama-2-7B-GGUF",
  "models": {
    "Q8_0": "llama-2-7b.Q8_0.gguf",
    "Q4_K_M": "llama-2-7b.Q4_K_M.gguf",
    "Q4_0": "llama-2-7b.Q4_0.gguf"
  },
  "context_size": 2048,
  "max_tokens": 100,
  "iterations": 5,
  "warmup_runs": 2,
  "enable_ablation_studies": true,
  "enable_batch_testing": true,
  "output_dir": "./benchmark_results"
}
```

Run with configuration file:

```bash
python -m llm_benchmark --config config.json
```

## Configuration

### Configuration Parameters

#### Model Configuration

- `repo_id` (required): Hugging Face repository ID
- `models` (required): Dictionary mapping quantization level to filename
- `model_cache_dir` (default: `~/.cache/llm_benchmark/models`): Model cache directory

#### Test Parameters

- `context_size` (default: 2048): Maximum context window size
- `batch_size` (default: 512): Batch size for inference
- `max_tokens` (default: 100): Maximum tokens to generate per test
- `iterations` (default: 3): Number of benchmark iterations
- `warmup_runs` (default: 2): Number of warmup runs before measurement

#### Test Selection

- `enable_quantization_profiling` (default: true): Run quantization tests
- `enable_ablation_studies` (default: true): Run KV cache ablation studies
- `enable_batch_testing` (default: true): Run batch processing tests
- `enable_thermal_monitoring` (default: true): Monitor temperature and power

#### Ablation Configuration

- `kv_cache_types` (default: `["ram", "disk"]`): Cache types to test
- `prompt_cache_prefix_lengths` (default: `[100, 500, 1000]`): Prefix lengths for prompt caching
- `batch_sizes` (default: `[1, 2, 4, 8, 16]`): Batch sizes to test

#### Orchestration

- `sleep_between_tests_s` (default: 5): Delay between tests for thermal stabilization
- `thermal_stabilization_threshold_c` (default: 70.0): Temperature threshold for throttling
- `inference_timeout_s` (default: 300): Timeout for inference operations

#### Output

- `output_dir` (default: `./benchmark_results`): Output directory for results
- `save_formats` (default: `["json", "csv", "markdown", "html"]`): Output formats
- `visualization_dpi` (default: 300): Resolution for saved visualizations

#### Authentication

- `hf_token` (optional): Hugging Face API token (or set `HF_TOKEN` environment variable)

### Configuration Priority

Configuration values are applied in this order (highest priority first):

1. Command-line arguments
2. Configuration file
3. Default values

Example:

```bash
# Override config file iterations with CLI argument
python -m llm_benchmark --config config.json --iterations 10
```

## Command-Line Interface

### Basic Commands

```bash
# Show help
python -m llm_benchmark --help

# Run with minimal configuration
python -m llm_benchmark \
  --repo-id "TheBloke/Llama-2-7B-GGUF" \
  --models "Q8_0:llama-2-7b.Q8_0.gguf"

# Run with configuration file
python -m llm_benchmark --config config.json

# Override config file settings
python -m llm_benchmark --config config.json --iterations 10 --output-dir ./custom_results
```

### Common Options

```
--config PATH              Path to JSON/YAML configuration file
--repo-id REPO_ID          Hugging Face repository ID
--models MODELS            Comma-separated list of "quant:filename" pairs
--context-size SIZE        Context window size (default: 2048)
--max-tokens N             Maximum tokens to generate (default: 100)
--iterations N             Number of benchmark iterations (default: 3)
--output-dir DIR           Output directory (default: ./benchmark_results)
--model-cache-dir DIR      Model cache directory
--enable-ablation          Enable ablation studies (default: true)
--disable-ablation         Disable ablation studies
--enable-batch-testing     Enable batch processing tests (default: true)
--disable-batch-testing    Disable batch processing tests
```

### Examples

#### Test Multiple Quantization Levels

```bash
python -m llm_benchmark \
  --repo-id "TheBloke/Llama-2-7B-GGUF" \
  --models "Q8_0:llama-2-7b.Q8_0.gguf,Q4_K_M:llama-2-7b.Q4_K_M.gguf,Q4_0:llama-2-7b.Q4_0.gguf,Q2_K:llama-2-7b.Q2_K.gguf" \
  --iterations 5 \
  --output-dir ./results/multi_quant
```

#### Quick Test (Minimal Configuration)

```bash
python -m llm_benchmark \
  --repo-id "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" \
  --models "Q4_K_M:tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
  --iterations 1 \
  --max-tokens 50 \
  --disable-ablation \
  --disable-batch-testing
```

#### Comprehensive Benchmark

```bash
python -m llm_benchmark \
  --repo-id "TheBloke/Llama-2-13B-GGUF" \
  --models "Q8_0:llama-2-13b.Q8_0.gguf,Q4_K_M:llama-2-13b.Q4_K_M.gguf" \
  --context-size 4096 \
  --max-tokens 200 \
  --iterations 10 \
  --warmup-runs 3 \
  --enable-ablation \
  --enable-batch-testing \
  --output-dir ./results/comprehensive
```

## Understanding Results

### Output Structure

After running a benchmark, you'll find these files in the output directory:

```
benchmark_results/
├── run_20240115_143022/
│   ├── config.json                    # Test configuration
│   ├── hardware_info.json             # Hardware detection results
│   ├── results.json                   # Complete results (machine-readable)
│   ├── results.csv                    # Tabular results (spreadsheet-friendly)
│   ├── results.md                     # Human-readable report
│   ├── benchmark_report.html          # Interactive HTML report
│   ├── visualizations/
│   │   ├── quantization_comparison.png
│   │   ├── memory_vs_speed_tradeoff.png
│   │   ├── throughput_over_time.png
│   │   └── ablation_comparison.png
│   └── logs/
│       └── benchmark.log              # Detailed execution log
```

### Key Metrics Explained

#### Time to First Token (TTFT)

- **What it measures**: Latency from prompt submission to first output token
- **Lower is better**: Indicates faster response time
- **Typical values**: 50-500ms depending on model size and hardware
- **Affected by**: Prompt length, model size, quantization, GPU acceleration

#### Prefill Throughput

- **What it measures**: Speed of processing input prompt tokens
- **Higher is better**: Indicates faster prompt processing
- **Typical values**: 100-500 tokens/second
- **Affected by**: Hardware, quantization, context size

#### Decode Throughput

- **What it measures**: Speed of generating output tokens
- **Higher is better**: Indicates faster text generation
- **Typical values**: 20-100 tokens/second
- **Affected by**: Hardware, quantization, batch size

#### Memory Usage

- **Peak RAM**: Maximum memory used during inference
- **RAM Increase**: Memory increase from baseline (model loading)
- **GPU Memory**: VRAM used when GPU acceleration is enabled
- **Typical values**: 2-8GB for 7B models, varies by quantization

### Interpreting Quantization Results

**Q8_0 (8-bit quantization)**:
- Highest quality, closest to original model
- Highest memory usage
- Slower inference
- Best for accuracy-critical applications

**Q4_K_M (4-bit K-quant, medium)**:
- Good balance of quality and performance
- ~50% memory reduction vs Q8_0
- Faster inference
- Recommended for most use cases

**Q4_0 (4-bit quantization)**:
- Lower quality than Q4_K_M
- Similar memory usage to Q4_K_M
- Slightly faster inference
- Good for resource-constrained environments

**Q2_K (2-bit K-quant)**:
- Lowest quality
- Lowest memory usage (~75% reduction vs Q8_0)
- Fastest inference
- Only for extreme resource constraints

### Interpreting Ablation Results

**KV Cache Ablation**:
- **Control (no cache)**: Baseline performance
- **Cold cache**: Cache enabled but empty (first run)
- **Warm cache**: Cache populated from previous prompt
- **Improvement**: TTFT reduction with warm cache (typically 30-60%)

**Prompt Caching**:
- **Cache hit rate**: Percentage of tokens reused from cache
- **Latency reduction**: Time saved by reusing cached tokens
- **Memory overhead**: Additional memory used by cache

**Batch Processing**:
- **Aggregate throughput**: Total tokens/second for all prompts in batch
- **Per-prompt latency**: Average latency per prompt in batch
- **Optimal batch size**: Batch size maximizing throughput without OOM

## Advanced Usage

### Custom Test Prompts

Create a configuration file with custom prompts:

```json
{
  "repo_id": "TheBloke/Llama-2-7B-GGUF",
  "models": {
    "Q4_K_M": "llama-2-7b.Q4_K_M.gguf"
  },
  "test_prompts": [
    "Explain quantum computing in simple terms.",
    "Write a Python function to calculate fibonacci numbers.",
    "Summarize the key points of machine learning."
  ]
}
```

### Running on Specific Hardware

The framework automatically detects your hardware, but you can verify detection:

```bash
# Check hardware detection
python -c "from llm_benchmark.hardware import HardwareDetector; print(HardwareDetector.detect())"
```

### Using Hugging Face Authentication

For private models or to avoid rate limits:

```bash
# Set environment variable
export HF_TOKEN="your_huggingface_token"

# Or in configuration file
{
  "hf_token": "your_huggingface_token",
  ...
}
```

### Comparing Multiple Runs

To compare results from different runs:

1. Run benchmarks with different configurations
2. Save results to different directories
3. Use the JSON results for programmatic comparison

```python
import json

# Load results from two runs
with open('results/run1/results.json') as f:
    run1 = json.load(f)

with open('results/run2/results.json') as f:
    run2 = json.load(f)

# Compare decode throughput
for r1, r2 in zip(run1['quantization_results'], run2['quantization_results']):
    print(f"{r1['quantization']}: {r1['decode_tps']:.2f} vs {r2['decode_tps']:.2f}")
```

## Troubleshooting

### Common Issues

#### "Missing required dependencies"

**Problem**: Required Python packages are not installed.

**Solution**:
```bash
pip install -r requirements.txt
```

#### "Failed to download model"

**Problem**: Network issues or authentication failure.

**Solutions**:
- Check internet connection
- Verify Hugging Face repository ID is correct
- Set `HF_TOKEN` environment variable for private models
- Try downloading model manually and specify local path

#### "Insufficient RAM"

**Problem**: Not enough memory to load model.

**Solutions**:
- Use smaller quantization level (Q4_0 or Q2_K instead of Q8_0)
- Reduce `context_size` (e.g., 1024 instead of 2048)
- Close other applications to free memory
- Use a smaller model (7B instead of 13B)

#### "GPU out of memory"

**Problem**: GPU VRAM exhausted.

**Solutions**:
- Framework will automatically reduce GPU layers and retry
- If still failing, use CPU-only mode (framework will fallback automatically)
- Reduce `context_size` or `batch_size`

#### "Inference timeout"

**Problem**: Inference taking longer than 300 seconds.

**Solutions**:
- Reduce `max_tokens` (generate fewer tokens)
- Reduce `context_size` (smaller context window)
- Increase `inference_timeout_s` in configuration
- Use faster quantization level (Q4_0 instead of Q8_0)

#### "Thermal throttling detected"

**Problem**: CPU/GPU temperature too high.

**Solutions**:
- Framework will automatically wait for cooldown
- Increase `sleep_between_tests_s` for longer cooling periods
- Improve system cooling (clean fans, better ventilation)
- Reduce workload (fewer iterations, disable ablation studies)

### Debug Mode

Enable detailed logging:

```bash
# Set log level to DEBUG
export LOG_LEVEL=DEBUG
python -m llm_benchmark --config config.json
```

Check log files in output directory:
```bash
cat benchmark_results/run_*/logs/benchmark.log
```

### Getting Help

If you encounter issues:

1. Check the log files in `output_dir/logs/`
2. Verify hardware detection: `python -c "from llm_benchmark.hardware import HardwareDetector; print(HardwareDetector.detect())"`
3. Test with minimal configuration (single model, 1 iteration, no ablation)
4. Check GitHub issues: https://github.com/yourusername/llm-benchmark/issues

## Platform-Specific Notes

### Linux x86

**Supported**:
- CPU-only inference
- GPU acceleration (NVIDIA GPUs with CUDA)
- Thermal monitoring (if sensors available)
- All quantization levels

**Optimizations**:
- Automatic SIMD detection (AVX, AVX2, AVX512)
- Multi-threading (uses all CPU cores)
- Memory locking for better performance

**Requirements**:
- Linux kernel 4.0+
- glibc 2.17+
- CUDA 11.0+ (for GPU acceleration)

### NVIDIA Jetson Xavier NX

**Supported**:
- GPU acceleration (automatic layer offloading)
- Thermal monitoring (built-in sensors)
- Power monitoring (built-in sensors)
- All quantization levels

**Optimizations**:
- Automatic GPU layer calculation based on available VRAM
- Conservative thread count (leaves cores for system)
- Thermal throttling protection

**Requirements**:
- JetPack 4.6+ or 5.0+
- CUDA 10.2+ (included in JetPack)
- 8GB RAM minimum

**Notes**:
- GPU acceleration is automatically enabled
- Framework will fallback to CPU if GPU memory exhausted
- Thermal monitoring is always enabled
- Power monitoring provides detailed energy metrics

### macOS (Experimental)

**Status**: Not officially supported yet

**Potential Support**:
- CPU-only inference (ARM64 on M1/M2/M3)
- Metal GPU acceleration (future)

## Best Practices

### For Accurate Benchmarks

1. **Close other applications**: Minimize background processes
2. **Use multiple iterations**: Run at least 3-5 iterations for statistical validity
3. **Enable warmup runs**: Use 2-3 warmup runs to stabilize system state
4. **Monitor thermal state**: Ensure system is not thermally throttled
5. **Use consistent prompts**: Test with same prompts across runs for fair comparison

### For Reproducibility

1. **Record hardware info**: Save `hardware_info.json` with results
2. **Record software versions**: Save `config.json` with results
3. **Record model checksums**: Framework automatically records SHA256 checksums
4. **Use fixed random seeds**: Set `PYTHONHASHSEED=0` for deterministic behavior
5. **Document environment**: Note any system-specific configurations

### For Performance

1. **Use appropriate quantization**: Q4_K_M is usually the best balance
2. **Enable GPU acceleration**: Significant speedup on supported hardware
3. **Optimize batch size**: Test different batch sizes to find optimal throughput
4. **Use prompt caching**: Enable for scenarios with repeated prefixes
5. **Monitor resource usage**: Ensure no memory or thermal bottlenecks

## Next Steps

- Read the [API Documentation](API.md) for programmatic usage
- Check [Examples](../examples/) for sample configurations
- See [Contributing Guide](CONTRIBUTING.md) to contribute to the project
- Join our [Discord](https://discord.gg/llm-benchmark) for community support
