# CI/CD Pipeline Setup

This document describes the CI/CD pipeline configuration for the LLM Benchmark Framework.

## Overview

The CI/CD pipeline automatically runs tests on every push and pull request to ensure code quality and detect performance regressions. The pipeline is configured using GitHub Actions.

## Requirements Satisfied

This CI/CD setup satisfies the following requirements:

- **Requirement 8.5**: Generate requirements file with exact versions (`requirements-frozen.txt`)
- **Requirement 8.7**: Compute and record SHA256 checksums for all model files tested

## Pipeline Configuration

### Workflow File

Location: `.github/workflows/test.yml`

### Trigger Events

The pipeline runs on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

### Pipeline Steps

1. **Checkout code**: Clone the repository
2. **Set up Python**: Install Python 3.10
3. **Cache dependencies**: Cache pip packages for faster builds
4. **Install dependencies**: Install from `requirements-frozen.txt`
5. **Download test models**: Download small test models (< 1GB)
6. **Run unit tests**: Execute unit tests with coverage
7. **Run integration tests**: Execute integration tests (excluding Jetson-specific)
8. **Upload coverage**: Send coverage reports to Codecov
9. **Generate coverage report**: Display coverage summary

### Test Execution

```bash
# Unit tests
pytest tests/unit/ -v --cov=llm_benchmark --cov-report=xml --cov-report=term

# Integration tests (excluding Jetson-specific)
pytest tests/integration/ -v -m "not jetson" --cov=llm_benchmark --cov-append --cov-report=xml --cov-report=term
```

### Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.jetson`: Requires Jetson Xavier NX hardware (excluded from CI)
- `@pytest.mark.slow`: Slow-running tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.property`: Property-based tests

To exclude Jetson tests locally:
```bash
pytest tests/ -m "not jetson"
```

## Test Models

### Small Test Models

The pipeline uses small quantized models (< 1GB) for fast execution:

- **TinyLlama-1.1B-Chat-v1.0 Q4_0** (~669 MB)
  - Repository: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`
  - Filename: `tinyllama-1.1b-chat-v1.0.Q4_0.gguf`

### Model Download

Models are downloaded using `scripts/download_test_models.py`:

```bash
python scripts/download_test_models.py
```

The script:
- Downloads models from Hugging Face Hub
- Caches models locally to avoid repeated downloads
- Computes SHA256 checksums for integrity verification
- Displays checksums for documentation

### Hugging Face Token

For private models, set the `HF_TOKEN` secret in GitHub repository settings:

1. Go to repository Settings → Secrets and variables → Actions
2. Add new repository secret: `HF_TOKEN`
3. Set value to your Hugging Face API token

## Test Fixtures

### Fixed Test Prompts

Location: `tests/fixtures/test_prompts.py`

Provides standardized prompts for reproducible testing:
- `SHORT_PROMPT`: < 50 tokens
- `MEDIUM_PROMPT`: ~100 tokens
- `LONG_PROMPT`: ~500 tokens
- `CACHE_TEST_PROMPT_1`, `CACHE_TEST_PROMPT_2`: For cache testing
- `BATCH_PROMPTS`: For batch testing

### Baseline Results

Location: `tests/fixtures/baseline_results.py`

Defines expected performance ranges for different hardware:
- `BASELINE_X86_CPU`: x86_64 CPU-only (CI/CD runner)
- `BASELINE_JETSON_XAVIER_NX_GPU`: Jetson with GPU
- `BASELINE_JETSON_XAVIER_NX_CPU`: Jetson CPU-only

### Regression Detection

The baseline module includes regression detection:

```python
from tests.fixtures import detect_regression

is_regression, message = detect_regression(
    "ttft_ms",
    current_value=200,
    baseline_value=150
)

if is_regression:
    print(f"⚠️ {message}")
```

**Regression Thresholds:**
- Performance metrics (TTFT, throughput): 20% degradation
- Memory metrics (RAM, GPU memory): 20% increase
- Load time: 20% increase

## Dependencies

### Requirements Files

1. **`requirements.txt`**: Minimum version constraints for development
2. **`requirements-frozen.txt`**: Exact versions for CI/CD reproducibility

### Frozen Requirements

The frozen requirements file (`requirements-frozen.txt`) contains exact versions of all dependencies, satisfying **Requirement 8.5**.

To update frozen requirements:
```bash
pip freeze > requirements-frozen.txt
```

### Key Dependencies

- `llama-cpp-python`: LLM inference engine
- `pytest`: Testing framework
- `pytest-cov`: Coverage reporting
- `pytest-mock`: Mocking utilities
- `hypothesis`: Property-based testing
- `huggingface-hub`: Model downloads
- `psutil`: System metrics
- `pandas`, `matplotlib`, `seaborn`: Data analysis and visualization
- `scipy`, `numpy`: Statistical analysis

## Coverage Reporting

### Codecov Integration

Coverage reports are automatically uploaded to Codecov after test execution.

To set up Codecov:
1. Sign up at [codecov.io](https://codecov.io)
2. Add your repository
3. No additional secrets needed (GitHub Actions integration is automatic)

### Coverage Configuration

Location: `pytest.ini`

```ini
[coverage:run]
source = llm_benchmark
omit =
    */tests/*
    */test_*.py
    */__pycache__/*
    */site-packages/*

[coverage:report]
precision = 2
show_missing = True
skip_covered = False
```

### Coverage Goals

- **Unit tests**: 80% code coverage minimum
- **Integration tests**: Cover all major workflows
- **Property tests**: Cover all pure functions

## Local Testing

### Run All Tests

```bash
# All tests except Jetson-specific
pytest tests/ -m "not jetson"

# With coverage
pytest tests/ -m "not jetson" --cov=llm_benchmark --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Property-based tests only
pytest tests/ -m "property" -v

# Slow tests only
pytest tests/ -m "slow" -v
```

### Run with Different Verbosity

```bash
# Minimal output
pytest tests/ -q

# Verbose output
pytest tests/ -v

# Very verbose output (show all test details)
pytest tests/ -vv
```

## Manual Testing on Jetson

Jetson-specific tests require actual hardware and must be run manually:

```bash
# SSH into Jetson Xavier NX
ssh jetson@<jetson-ip>

# Clone repository
git clone <repo-url>
cd llm-benchmark

# Install dependencies
pip install -r requirements-frozen.txt

# Run Jetson-specific tests
pytest tests/ -m "jetson" -v

# Run full integration test
python benchmark.py --config configs/jetson_test.json
```

## Troubleshooting

### Test Failures

1. **Model download fails**:
   - Check internet connectivity
   - Verify HF_TOKEN is set correctly
   - Check Hugging Face Hub status

2. **Coverage upload fails**:
   - Check Codecov service status
   - Verify repository is added to Codecov
   - Check GitHub Actions logs for details

3. **Tests timeout**:
   - Increase timeout in workflow file
   - Use smaller test models
   - Reduce number of test iterations

### Performance Issues

1. **Slow test execution**:
   - Use test model caching
   - Run tests in parallel: `pytest -n auto`
   - Skip slow tests: `pytest -m "not slow"`

2. **High memory usage**:
   - Use smaller test models
   - Reduce batch sizes in tests
   - Run fewer tests in parallel

## Continuous Improvement

### Adding New Tests

1. Write test in appropriate directory (`tests/unit/`, `tests/integration/`)
2. Add appropriate markers (`@pytest.mark.jetson`, etc.)
3. Run locally to verify
4. Push to trigger CI/CD pipeline

### Updating Baselines

When intentional performance improvements are made:

1. Run benchmarks on reference hardware
2. Update `tests/fixtures/baseline_results.py`
3. Document changes in commit message
4. Update this documentation if needed

### Adding New Test Models

1. Edit `scripts/download_test_models.py`
2. Add model to `TEST_MODELS` dictionary
3. Test download locally
4. Update baselines if needed
5. Update documentation

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Pytest Documentation](https://docs.pytest.org/)
- [Codecov Documentation](https://docs.codecov.com/)
- [Hugging Face Hub Documentation](https://huggingface.co/docs/huggingface_hub/)

## Support

For issues with the CI/CD pipeline:
1. Check GitHub Actions logs
2. Review this documentation
3. Check test output for specific errors
4. Open an issue with detailed error information
