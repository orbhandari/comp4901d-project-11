# Test Fixtures and Baseline Data

This directory contains test fixtures, baseline results, and expected metric ranges for reproducible benchmarking and regression detection.

## Files

### `test_prompts.py`

Fixed test prompts for reproducible benchmarking across different runs and environments.

**Prompts included:**
- `SHORT_PROMPT`: < 50 tokens, for quick sanity checks
- `MEDIUM_PROMPT`: ~100 tokens, for standard testing
- `LONG_PROMPT`: ~500 tokens, for comprehensive testing
- `CACHE_PREFIX`: Shared prefix for cache effectiveness testing
- `CACHE_TEST_PROMPT_1`, `CACHE_TEST_PROMPT_2`: Prompts with shared prefix
- `BATCH_PROMPTS`: Set of similar-length prompts for batch testing
- `UNICODE_PROMPT`, `SPECIAL_CHARS_PROMPT`: Edge case testing

**Usage:**
```python
from tests.fixtures.test_prompts import SHORT_PROMPT, MEDIUM_PROMPT

# Use in tests
result = run_inference(model, SHORT_PROMPT, max_tokens=10)
```

### `baseline_results.py`

Baseline performance metrics and expected ranges for different hardware configurations.

**Baselines included:**
- `BASELINE_X86_CPU`: x86_64 CPU-only (typical CI/CD runner)
- `BASELINE_JETSON_XAVIER_NX_GPU`: Jetson Xavier NX with GPU acceleration
- `BASELINE_JETSON_XAVIER_NX_CPU`: Jetson Xavier NX CPU-only fallback

**Metrics tracked:**
- Load time (seconds)
- Peak RAM usage (MB)
- GPU memory usage (MB, when applicable)
- Time to first token (milliseconds)
- Prefill throughput (tokens/second)
- Decode throughput (tokens/second)
- GPU utilization (percentage, when applicable)
- Total inference time for 100 tokens (seconds)

**Usage:**
```python
from tests.fixtures.baseline_results import validate_results, detect_regression

# Validate results against baseline
results = {"ttft_ms": 150, "decode_tps": 25}
validation = validate_results(results, baseline_key="x86_cpu")

for metric, (is_valid, message) in validation.items():
    print(message)

# Detect regressions
is_regression, message = detect_regression("ttft_ms", current_value=200, baseline_value=150)
if is_regression:
    print(f"⚠️ {message}")
```

## Test Models

Test models are downloaded using `scripts/download_test_models.py`.

**Current test models:**
- **TinyLlama-1.1B-Chat-v1.0 Q4_0** (~669 MB)
  - Repository: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`
  - Filename: `tinyllama-1.1b-chat-v1.0.Q4_0.gguf`
  - Use case: Fast CI/CD testing, small enough for quick downloads

### Downloading Test Models

```bash
# Download all test models
python scripts/download_test_models.py

# With Hugging Face token (for private models)
HF_TOKEN=your_token_here python scripts/download_test_models.py
```

Models are cached in `./models/test_models/` to avoid repeated downloads.

### Model Checksums

All downloaded models have their SHA256 checksums computed and recorded. This satisfies **Requirement 8.7**: "Compute and record SHA256 checksums for all model files tested."

The checksums are:
- Displayed during download
- Used for integrity verification
- Recorded in benchmark results

## Regression Detection

The baseline results include regression thresholds for automatic detection of performance degradation:

**Thresholds:**
- **Performance metrics** (TTFT, throughput): 20% degradation triggers alert
- **Memory metrics** (RAM, GPU memory): 20% increase triggers alert
- **Load time**: 20% increase triggers alert

**Tolerance:**
- Baseline ranges include 20% tolerance to account for normal variability
- Values within tolerance are considered acceptable
- Values outside tolerance trigger warnings

## Expected Metric Ranges

### x86 CPU (CI/CD Runner)

Typical 4-core x86_64 CPU with 8GB RAM:

| Metric | Min | Max | Notes |
|--------|-----|-----|-------|
| Load time | 0.5s | 5.0s | Model loading |
| Peak RAM | 500 MB | 1500 MB | Memory usage |
| TTFT | 50 ms | 500 ms | First token latency |
| Prefill TPS | 50 | 500 | Prompt processing |
| Decode TPS | 5 | 50 | Token generation |
| Total time (100 tokens) | 2s | 30s | Complete inference |

### Jetson Xavier NX (GPU)

6-core ARM CPU with GPU acceleration:

| Metric | Min | Max | Notes |
|--------|-----|-----|-------|
| Load time | 1.0s | 10.0s | Model loading |
| Peak RAM | 600 MB | 2000 MB | System memory |
| GPU memory | 400 MB | 1200 MB | GPU memory |
| TTFT | 20 ms | 200 ms | First token latency |
| Prefill TPS | 100 | 1000 | Prompt processing |
| Decode TPS | 10 | 100 | Token generation |
| GPU utilization | 30% | 100% | GPU usage |
| Total time (100 tokens) | 1s | 15s | Complete inference |

### Jetson Xavier NX (CPU-only)

6-core ARM CPU without GPU (fallback mode):

| Metric | Min | Max | Notes |
|--------|-----|-----|-------|
| Load time | 0.5s | 8.0s | Model loading |
| Peak RAM | 500 MB | 1800 MB | Memory usage |
| TTFT | 100 ms | 800 ms | First token latency |
| Prefill TPS | 20 | 200 | Prompt processing |
| Decode TPS | 3 | 30 | Token generation |
| Total time (100 tokens) | 3s | 40s | Complete inference |

## Usage in CI/CD

The GitHub Actions workflow (`.github/workflows/test.yml`) uses these fixtures:

1. **Download test models** using `scripts/download_test_models.py`
2. **Run tests** with fixed prompts from `test_prompts.py`
3. **Validate results** against baselines from `baseline_results.py`
4. **Detect regressions** using threshold checks
5. **Report coverage** to Codecov

## Updating Baselines

When intentional performance improvements are made:

1. Run benchmarks on reference hardware
2. Update metric ranges in `baseline_results.py`
3. Document the change in git commit message
4. Update this README if hardware configurations change

## Adding New Test Models

To add a new test model:

1. Edit `scripts/download_test_models.py`
2. Add model configuration to `TEST_MODELS` dictionary:
   ```python
   "model-name.gguf": {
       "repo_id": "author/repo-name",
       "filename": "model-name.gguf",
       "sha256": None,  # Optional: add known checksum
       "size_mb": 500,  # Approximate size
   }
   ```
3. Run download script to verify
4. Update baselines if needed

## Requirements Satisfied

This test fixture setup satisfies the following requirements:

- **Requirement 8.5**: Generate requirements file with exact versions (via `requirements.txt`)
- **Requirement 8.7**: Compute and record SHA256 checksums for all model files tested
- **Requirement 10.3**: Cache downloaded models in configurable directory
- **Requirement 10.4**: Skip downloading when model exists locally

## Notes

- Test models are kept small (< 1GB) for fast CI/CD execution
- Prompts are fixed for reproducibility across runs
- Baseline ranges account for hardware variability
- Regression thresholds are conservative (20%) to avoid false positives
- All metrics are validated against expected ranges
