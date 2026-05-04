# LLM Benchmark Framework

A comprehensive, cross-platform framework for benchmarking large language model (LLM) inference performance across different hardware platforms, quantization levels, and optimization strategies.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🖥️ **Cross-Platform Support**: Linux x86 and NVIDIA Jetson Xavier NX
- 🔍 **Automatic Hardware Detection**: CPU, GPU, memory, and sensors
- 📊 **Multi-Level Quantization Profiling**: Q8_0, Q4_0, Q4_K_M, Q2_K
- ⚡ **GPU Acceleration**: Automatic GPU offloading with CPU fallback
- 🧪 **Ablation Studies**: KV cache and prompt caching effectiveness
- 📈 **Batch Processing**: Throughput optimization testing (planned feature)
- 📉 **Statistical Validation**: Confidence intervals, t-tests, outlier detection
- 📱 **Interactive Reports**: Professional HTML reports with embedded visualizations
- 🔄 **Reproducible Results**: Complete environment capture and checksums

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-benchmark.git
cd llm-benchmark

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set Hugging Face token (required for downloading models)
export HF_TOKEN="your_huggingface_token_here"
# On Windows: set HF_TOKEN=your_huggingface_token_here
```

### Basic Usage

```bash
# Run a basic benchmark
python -m llm_benchmark \
  --repo-id "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" \
  --models "Q8_0:tinyllama-1.1b-chat-v1.0.Q8_0.gguf,Q4_0:tinyllama-1.1b-chat-v1.0.Q4_0.gguf" \
  --iterations 3 \
  --output-dir ./benchmark_results
```

### Using Configuration File

Use one of the provided example configurations:

```bash
# For x86 Linux systems
python -m llm_benchmark --config configs/x86_linux_example.json

# For Jetson Xavier NX
python -m llm_benchmark --config configs/jetson_xavier_nx_example.json
```

Or create your own `config.json`:

```json
{
  "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
  "models": {
    "Q8_0": "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
    "Q4_K_M": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    "Q4_0": "tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
  },
  "context_size": 2048,
  "max_tokens": 100,
  "iterations": 5,
  "enable_ablation_studies": true,
  "enable_batch_testing": false,
  "output_dir": "./benchmark_results"
}
```

Run with your configuration:

```bash
python -m llm_benchmark --config config.json
```

## What Gets Measured

### Core Metrics

- **Time to First Token (TTFT)**: Latency from prompt to first output
- **Prefill Throughput**: Tokens/second during prompt processing
- **Decode Throughput**: Tokens/second during generation
- **Memory Usage**: Peak RAM and GPU memory consumption
- **Model Load Time**: Time to load model into memory

### Optional Metrics

- **Thermal Metrics**: CPU/GPU temperature during inference
- **Power Consumption**: Power usage during inference (Jetson)
- **Cache Effectiveness**: KV cache and prompt caching improvements
- **Batch Throughput**: Aggregate throughput for batch inference (planned feature)

## Output

After running a benchmark, you'll find:

```
benchmark_results/
├── run_20260504_163307/
│   ├── results.json                   # Machine-readable results
│   ├── results.csv                    # Spreadsheet-friendly format
│   ├── results.md                     # Human-readable report
│   ├── visualizations/
│   │   ├── quantization_comparison.png
│   │   ├── memory_vs_speed_tradeoff.png
│   │   └── throughput_over_time.png
│   └── logs/
│       └── benchmark.log              # Detailed execution log
├── benchmark_report.html              # Interactive HTML report (latest run)
└── visualizations/                    # Latest visualizations
```

## Documentation

- **[User Guide](docs/USER_GUIDE.md)**: Comprehensive usage guide
- **[Troubleshooting](docs/TROUBLESHOOTING.md)**: Common issues and solutions
- **[CI/CD Setup](docs/CI_CD_SETUP.md)**: Continuous integration configuration
- **[Examples](examples/)**: Sample scripts and configurations

## Supported Platforms

### Linux x86

- ✅ CPU-only inference
- ✅ GPU acceleration (NVIDIA with CUDA)
- ✅ Thermal monitoring
- ✅ All quantization levels
- ✅ SIMD optimizations (AVX, AVX2, AVX512)

### NVIDIA Jetson Xavier NX

- ✅ GPU acceleration (automatic layer offloading)
- ✅ Thermal monitoring (built-in sensors)
- ✅ Power monitoring (built-in sensors)
- ✅ All quantization levels
- ✅ Optimized for ARM64 architecture

### macOS (Experimental)

- ⚠️ CPU-only inference (ARM64 on M1/M2/M3)
- ⚠️ Metal GPU acceleration (future)

## Requirements

### Minimum

- Python 3.8+
- 8GB RAM
- 10GB disk space

### Recommended

- Python 3.10+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM (for GPU acceleration)
- 20GB+ disk space

## Dependencies

### Required

- `llama-cpp-python` - LLM inference engine
- `psutil` - System monitoring
- `pandas` - Data processing
- `matplotlib` - Visualization
- `seaborn` - Enhanced plotting
- `numpy` - Numerical computing
- `scipy` - Statistical analysis
- `huggingface-hub` - Model downloading
- `jinja2` - HTML report generation

### Optional

- `pynvml` - GPU monitoring (for NVIDIA GPUs)
- `pyyaml` - YAML configuration files

## Examples

### Test Multiple Quantization Levels

```bash
python -m llm_benchmark \
  --repo-id "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" \
  --models "Q8_0:tinyllama-1.1b-chat-v1.0.Q8_0.gguf,Q4_K_M:tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf,Q4_0:tinyllama-1.1b-chat-v1.0.Q4_0.gguf,Q2_K:tinyllama-1.1b-chat-v1.0.Q2_K.gguf" \
  --iterations 5 \
  --output-dir ./benchmark_results
```

### Quick Test (Minimal Configuration)

```bash
python -m llm_benchmark \
  --repo-id "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" \
  --models "Q4_K_M:tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
  --iterations 1 \
  --max-tokens 50 \
  --disable-ablation-studies \
  --disable-batch-testing
```

### Comprehensive Benchmark (Larger Model)

```bash
python -m llm_benchmark \
  --repo-id "TheBloke/Llama-2-7B-GGUF" \
  --models "Q8_0:llama-2-7b.Q8_0.gguf,Q4_K_M:llama-2-7b.Q4_K_M.gguf" \
  --context-size 4096 \
  --max-tokens 200 \
  --iterations 10 \
  --warmup-runs 3 \
  --output-dir ./benchmark_results
```

### Using Example Scripts

The `examples/` directory contains ready-to-use scripts:

```bash
# Basic quantization profiling
python examples/basic_quantization_profiling.py

# Ablation studies
python examples/ablation_studies.py

# Batch processing (when implemented)
python examples/batch_processing.py
```

See [examples/README.md](examples/README.md) for more details.

## Troubleshooting

### Common Issues

**Missing dependencies**:
```bash
pip install -r requirements.txt
```

**GPU not detected**:
```bash
# Install pynvml
pip install pynvml

# Reinstall llama-cpp-python with CUDA support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --no-cache-dir --force-reinstall
```

**Insufficient RAM**:
- Use smaller quantization (Q4_0 or Q2_K)
- Reduce context size (`--context-size 1024`)
- Close other applications

**Inference timeout**:
- Reduce max tokens (`--max-tokens 50`)
- Increase timeout in config (`"inference_timeout_s": 600`)
- Use faster quantization (Q4_0)

**Hugging Face authentication**:
```bash
# Set token as environment variable
export HF_TOKEN="your_token_here"

# Or pass directly to command
python -m llm_benchmark --hf-token "your_token_here" ...
```

See [Troubleshooting Guide](docs/TROUBLESHOOTING.md) for more solutions.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

Please ensure:
- All tests pass (420 tests in the test suite)
- Code follows the existing style
- New features include tests
- Documentation is updated

See [docs/CI_CD_SETUP.md](docs/CI_CD_SETUP.md) for CI/CD configuration details.

## Project Structure

```
llm-benchmark/
├── llm_benchmark/           # Core framework code
│   ├── hardware/           # Hardware detection and HAL
│   ├── models/             # Model management
│   ├── metrics/            # Metrics collection
│   ├── quantization/       # Quantization profiling
│   ├── ablation/           # Ablation studies
│   ├── statistical/        # Statistical validation
│   ├── visualization/      # Report generation
│   └── orchestrator/       # Test orchestration
├── tests/                  # Test suite (420 tests)
├── configs/                # Example configurations
├── examples/               # Example scripts
├── docs/                   # Documentation
└── scripts/                # Utility scripts
```

## Testing

The framework includes a comprehensive test suite with 420 tests:

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/              # Unit tests
pytest tests/integration/       # Integration tests
pytest tests/property/          # Property-based tests

# Run with coverage
pytest tests/ --cov=llm_benchmark --cov-report=html
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{llm_benchmark_framework,
  title = {LLM Benchmark Framework},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/llm-benchmark}
}
```

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) for the inference engine
- [Hugging Face](https://huggingface.co) for model hosting
- [TheBloke](https://huggingface.co/TheBloke) for GGUF model conversions

## Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/llm-benchmark/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/llm-benchmark/discussions)

## Status

✅ **Production Ready** - All 420 tests passing
- Core quantization profiling: Complete
- Hardware detection and HAL: Complete
- Ablation studies: Complete
- Statistical validation: Complete
- Visualization and reporting: Complete
- CI/CD integration: Complete

🚧 **Planned Features**
- Batch processing optimization
- Additional model formats (ONNX, TensorRT)
- Web-based dashboard
- Distributed benchmarking
