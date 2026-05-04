# Implementation Plan: Comprehensive LLM Benchmark Framework

## Overview

This implementation plan transforms the existing `benchmark.py` script into a comprehensive, cross-platform LLM inference benchmarking framework. The plan follows an 8-week roadmap, building from core infrastructure through advanced features like GPU acceleration, ablation studies, statistical validation, and automated reporting. The implementation will support both Linux x86 and NVIDIA Jetson Xavier NX platforms with hardware-specific optimizations.

## Tasks

- [x] 1. Set up project structure and core infrastructure
  - Create modular package structure with separate modules for hardware detection, metrics collection, model management, and orchestration
  - Define data models and configuration schema using Python dataclasses
  - Implement configuration parsing from JSON/YAML files and command-line arguments
  - Set up logging infrastructure with file and console handlers
  - Create requirements.txt with all dependencies (llama-cpp-python, psutil, pandas, matplotlib, seaborn, pynvml, huggingface-hub, pytest, hypothesis)
  - _Requirements: 8.1, 8.2, 8.3, 11.6, 11.8, 11.9_

- [x] 1.1 Write unit tests for configuration validation
  - Test valid and invalid configuration parameters
  - Test command-line argument parsing
  - Test configuration file loading (JSON/YAML)
  - Test default value assignment
  - _Requirements: 11.8, 11.9_

- [x] 2. Implement Hardware Detection and HAL
  - [x] 2.1 Create HardwareInfo dataclass and HardwareDetector
    - Detect OS type (Linux x86 vs Jetson Xavier NX) using platform module and device tree
    - Parse CPU information from /proc/cpuinfo (model, cores, SIMD features)
    - Measure RAM using psutil (total and available)
    - Detect GPU availability using pynvml for NVIDIA GPUs
    - Probe thermal sensors in /sys/class/thermal
    - Probe power sensors in /sys/class/hwmon and Jetson power rails
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [x] 2.2 Write unit tests for hardware detection
    - Mock system calls to test x86 detection
    - Mock system calls to test Jetson detection
    - Test GPU detection with and without GPU
    - Test sensor probing with and without sensors
    - _Requirements: 1.1, 1.2, 1.3, 1.6_

  - [x] 2.3 Create Hardware Abstraction Layer (HAL)
    - Define HardwareBackend abstract base class with get_llama_config(), get_metrics_collector(), and optimize_for_inference() methods
    - Implement X86Backend with CPU-only configuration (n_gpu_layers=0, use_mlock=True, optimal thread count)
    - Implement JetsonBackend with GPU layer calculation heuristic (100MB per layer, 80% GPU memory utilization)
    - Implement platform-specific optimization methods
    - _Requirements: 1.1, 1.2, 4.1, 4.2, 4.5_

  - [x] 2.4 Write unit tests for HAL backends
    - Test X86Backend configuration generation
    - Test JetsonBackend GPU layer calculation with different memory sizes
    - Test fallback behavior when GPU is unavailable
    - _Requirements: 4.1, 4.2, 4.5_

- [x] 3. Checkpoint - Verify hardware detection and HAL
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement Model Manager
  - [x] 4.1 Create ModelInfo dataclass and ModelManager class
    - Implement model caching in configurable directory (default: ~/.cache/llm_benchmark/models)
    - Implement model download from Hugging Face Hub using hf_hub_download with progress callbacks
    - Implement exponential backoff retry logic (1s, 2s, 4s delays, max 3 retries)
    - Implement SHA256 checksum verification using streaming for large files
    - Implement GGUF format validation (check magic bytes and header structure)
    - Support HF_TOKEN from environment variable for authentication
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8_

  - [x] 4.2 Write unit tests for model manager
    - Mock hf_hub_download to test download retry logic
    - Test checksum verification with valid and invalid checksums
    - Test GGUF validation with valid and corrupted files
    - Test cache hit behavior (skip download when model exists)
    - _Requirements: 10.2, 10.3, 10.6, 10.8_

  - [x] 4.3 Write integration test for model download
    - Test actual download of small test model from Hugging Face
    - Verify downloaded model is cached correctly
    - Verify subsequent runs use cached model
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [x] 5. Implement Metrics Collector
  - [x] 5.1 Create InferenceMetrics dataclass and MetricsCollector class
    - Implement high-resolution timing using time.perf_counter()
    - Capture TTFT by measuring time to first chunk in streaming inference
    - Track per-token latency by timestamping each chunk
    - Calculate prefill throughput as prompt_tokens / ttft_s
    - Calculate decode throughput as (output_tokens - 1) / decode_duration
    - Measure memory usage using psutil.Process().memory_info()
    - Implement GPU metrics collection using pynvml (memory, utilization)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.3, 4.4_

  - [x] 5.2 Implement thermal and power monitoring
    - Create ThermalMonitor class reading from /sys/class/thermal
    - Create PowerMonitor class reading from /sys/class/hwmon and Jetson power rails
    - Run monitoring in background thread at 1 Hz frequency
    - Aggregate thermal/power as (min, avg, max) over inference duration
    - Detect thermal throttling by monitoring temperature thresholds
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.8_

  - [x] 5.3 Write unit tests for metrics collector
    - Mock psutil to test memory measurement
    - Mock pynvml to test GPU metrics collection
    - Test TTFT calculation with simulated streaming
    - Test throughput calculations with various token counts
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 4.3, 4.4_

  - [x] 5.4 Write unit tests for thermal and power monitoring
    - Mock /sys/class/thermal reads to test thermal monitoring
    - Mock power rail reads to test power monitoring
    - Test aggregation (min, avg, max) calculations
    - Test throttling detection logic
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6_

- [x] 6. Checkpoint - Verify metrics collection
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement Quantization Profiler
  - [x] 7.1 Create QuantizationResult dataclass and QuantizationProfiler class
    - Measure baseline memory before model load using psutil
    - Time model loading with time.perf_counter()
    - Perform warmup inference (5 tokens) before measurement
    - Use streaming inference to capture TTFT accurately
    - Execute identical test prompts across all quantization levels
    - Enforce garbage collection between quantization tests
    - Generate comparison matrix showing all metrics across quantization levels
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

  - [x] 7.2 Write integration test for quantization profiling
    - Test profiling with multiple quantization levels (Q8_0, Q4_0)
    - Verify identical prompts are used across quantizations
    - Verify metrics are collected for all quantization levels
    - Verify garbage collection between tests
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [x] 8. Implement Test Orchestrator
  - [x] 8.1 Create TestConfig and TestOrchestrator classes
    - Load configuration from JSON/YAML or command-line args
    - Implement warmup runs (default 2) before measurement runs
    - Enforce gc.collect() between test cases
    - Implement configurable sleep delays (default 5s) for thermal stabilization
    - Implement thermal stabilization delay checking temperature thresholds
    - Catch exceptions per test case, log error, continue with remaining tests
    - Save intermediate results after each test case to checkpoint.json
    - Generate final summary report with pass/fail status
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8_

  - [x] 8.2 Write unit tests for test orchestrator
    - Test warmup run execution
    - Test garbage collection enforcement
    - Test thermal stabilization delay logic
    - Test checkpoint saving and recovery
    - Test error handling (continue on failure)
    - _Requirements: 7.3, 7.4, 7.5, 7.6, 7.8_

- [x] 9. Implement result persistence and basic reporting
  - [x] 9.1 Create BenchmarkRun dataclass and result persistence
    - Implement JSON serialization for complete results
    - Implement CSV export for tabular data
    - Implement Markdown report generation with human-readable formatting
    - Record hardware info, software versions, config, and model checksums
    - Record timestamp and duration for each benchmark run
    - Create organized directory structure (results/run_TIMESTAMP/)
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.6, 8.7_

  - [x] 9.2 Write unit tests for result persistence
    - Test JSON serialization and deserialization
    - Test CSV export format
    - Test Markdown report generation
    - Test directory structure creation
    - _Requirements: 8.6, 8.7_

- [x] 10. Checkpoint - Verify core benchmarking workflow
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Implement GPU acceleration for Jetson Xavier NX
  - [x] 11.1 Enhance JetsonBackend with GPU-specific features
    - Implement GPU memory exhaustion handling with layer reduction fallback
    - Implement automatic fallback to CPU-only when GPU fails
    - Add GPU temperature and power monitoring for Jetson
    - Optimize thread count for Jetson (leave cores for system)
    - _Requirements: 4.1, 4.2, 4.5, 4.6, 4.7, 14.2, 14.3_

  - [x] 11.2 Implement GPU metrics collection in MetricsCollector
    - Measure GPU memory usage during inference using pynvml
    - Measure GPU utilization percentage during inference using pynvml
    - Report whether GPU acceleration was used in results
    - _Requirements: 4.3, 4.4, 4.7_

  - [x] 11.3 Write integration tests for GPU acceleration (manual on Jetson)
    - Test GPU layer offloading on Jetson Xavier NX
    - Test GPU memory exhaustion and fallback behavior
    - Test CPU-only fallback when GPU unavailable
    - Verify GPU metrics are collected when GPU is used
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.7_

- [x] 12. Implement KV Cache Ablation Studies
  - [x] 12.1 Create AblationResult dataclass and AblationEngine class
    - Implement control run without caching (baseline measurement)
    - Implement cold run with RAM cache enabled but empty
    - Implement warm run with RAM cache populated from previous prompt
    - Implement cold run with disk cache enabled but empty
    - Implement warm run with disk cache populated from previous prompt
    - Use prompts with substantial shared prefix (minimum 500 tokens)
    - Measure TTFT improvement between cold and warm runs
    - Measure cache memory overhead
    - Ensure process isolation by creating fresh model instances
    - Clean up cache directories after ablation completion
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9_

  - [x] 12.2 Write integration tests for KV cache ablation
    - Test RAM cache cold and warm runs
    - Test disk cache cold and warm runs
    - Verify process isolation between runs
    - Verify cache cleanup after completion
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.9_

- [x] 13. Implement Prompt Caching Optimization Testing
  - [x] 13.1 Extend AblationEngine with prompt caching tests
    - Test prompt caching with varying shared prefix lengths (100, 500, 1000 tokens)
    - Measure cache hit rate as percentage of tokens reused from cache
    - Measure latency reduction from prompt caching in milliseconds
    - Measure cache memory overhead as percentage of total model memory
    - Measure disk I/O time for cache operations when using disk cache
    - Measure cache file size in megabytes for disk cache
    - Compare prompt caching effectiveness across different quantization levels
    - Test cache behavior with multiple concurrent prompts sharing prefixes
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8_

  - [x] 13.2 Write integration tests for prompt caching
    - Test prompt caching with different prefix lengths
    - Verify cache hit rate calculation
    - Verify latency reduction measurement
    - Test across multiple quantization levels
    - _Requirements: 13.1, 13.2, 13.3, 13.7_

- [x] 14. Checkpoint - Verify ablation studies
  - Ensure all tests pass, ask the user if questions arise.

- [x] 15. Implement Batch Processing and Throughput Testing
  - [x] 15.1 Extend AblationEngine with batch testing
    - Test batch sizes of 1, 2, 4, 8, and 16 prompts
    - Measure aggregate throughput in tokens per second for batch inference
    - Measure per-prompt latency distribution within batches
    - Measure memory scaling as batch size increases
    - Identify optimal batch size maximizing throughput without memory overflow
    - Measure GPU utilization across different batch sizes when GPU is enabled
    - Generate throughput-latency tradeoff curves for different batch sizes
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8_

  - [x] 15.2 Write integration tests for batch processing
    - Test batch inference with different batch sizes
    - Verify aggregate throughput calculation
    - Verify per-prompt latency measurement
    - Verify memory scaling measurement
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5_

- [x] 16. Implement Statistical Validator
  - [x] 16.1 Create StatisticalSummary and ComparisonResult dataclasses
    - Calculate mean and standard deviation for each metric across multiple runs
    - Calculate 95% confidence intervals using mean ± 1.96 * (std / sqrt(n))
    - Perform paired t-tests comparing two configurations using scipy.stats.ttest_rel
    - Report p-values for all statistical comparisons
    - Mark differences as statistically significant when p < 0.05
    - Detect outliers using IQR method (values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR])
    - Require minimum 3 runs per configuration for statistical validity
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8_

  - [x] 16.2 Write property tests for statistical calculations
    - **Property 1: Confidence interval contains mean**
    - **Validates: Requirements 6.1, 6.2**
    - Test that 95% CI always contains the mean value
    - Use hypothesis to generate random metric lists

  - [x] 16.3 Write property tests for outlier detection
    - **Property 2: Outlier detection symmetry**
    - **Validates: Requirements 6.8**
    - Test that outliers are at distribution extremes
    - Use hypothesis to generate random value lists

  - [x] 16.4 Write unit tests for statistical validator
    - Test mean and standard deviation calculations
    - Test confidence interval calculations with known values
    - Test paired t-test with known distributions
    - Test outlier detection with known outliers
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.8_

- [x] 17. Checkpoint - Verify statistical validation
  - Ensure all tests pass, ask the user if questions arise.

- [x] 18. Implement Visualization Generator
  - [x] 18.1 Create VisualizationGenerator class
    - Create bar charts comparing metrics across quantization levels using matplotlib
    - Create line plots showing throughput over time during inference
    - Create scatter plots showing memory usage versus inference speed tradeoffs
    - Create heatmaps showing performance across hardware and quantization combinations using seaborn
    - Create before/after comparison charts for ablation studies
    - Include error bars representing confidence intervals on all charts
    - Save all visualizations as PNG files with minimum 300 DPI resolution
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_

  - [x] 18.2 Generate interactive HTML report
    - Create HTML report template using jinja2
    - Embed all visualizations in HTML report
    - Add interactive tooltips using mpld3 or plotly
    - Include complete environment documentation in report
    - Include summary statistics and comparison tables
    - _Requirements: 9.8_

  - [x] 18.3 Write integration tests for visualization generation
    - Test bar chart generation with sample data
    - Test line plot generation with sample data
    - Test scatter plot generation with sample data
    - Test heatmap generation with sample data
    - Test HTML report generation
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.8_

- [x] 19. Implement comprehensive error handling
  - [x] 19.1 Add error handling for model acquisition
    - Handle network failures during download with exponential backoff retry
    - Handle authentication failures with clear error messages
    - Handle disk space exhaustion with informative error
    - Handle corrupted downloads with checksum verification
    - Skip model and continue with available models on failure
    - _Requirements: 12.1, 10.6_

  - [x] 19.2 Add error handling for model loading
    - Validate GGUF format before loading
    - Check available memory before loading model
    - Handle insufficient RAM with suggestions (reduce context_size, use smaller quantization)
    - Handle corrupted GGUF files with validation errors
    - Handle missing CUDA libraries on Jetson with fallback to CPU
    - Skip model and continue with remaining models on failure
    - _Requirements: 12.1, 12.2, 12.4_

  - [x] 19.3 Add error handling for GPU memory exhaustion
    - Catch CUDA out of memory errors
    - Retry with reduced GPU layer count (reduce by 10 layers)
    - Fall back to CPU-only if GPU completely exhausted
    - Log GPU memory usage for diagnostics
    - Continue benchmark with reduced GPU utilization
    - _Requirements: 12.3_

  - [x] 19.4 Add error handling for inference timeout
    - Implement timeout protection using signal.alarm (default 300s)
    - Terminate inference after timeout
    - Mark test case as failed with timeout flag
    - Log timeout for diagnostics with suggestions
    - Continue with remaining test cases
    - _Requirements: 12.5, 12.6_

  - [x] 19.5 Add error handling for thermal throttling
    - Check thermal state before running tests
    - Wait for temperature to drop below threshold if throttled
    - Detect throttling during test execution
    - Flag results as thermally throttled if detected
    - Increase sleep delays between tests if needed
    - _Requirements: 14.6, 14.7_

  - [x] 19.6 Add dependency validation
    - Check for required dependencies (llama-cpp-python, pandas, matplotlib, etc.)
    - Check for CUDA libraries on Jetson
    - Report missing packages with installation instructions
    - Exit gracefully with error code 1 if dependencies missing
    - _Requirements: 12.7, 12.8_

  - [x] 19.7 Write unit tests for error handling
    - Test model download retry logic with mocked failures
    - Test model loading with insufficient memory
    - Test GPU memory exhaustion and fallback
    - Test inference timeout handling
    - Test thermal throttling detection and waiting
    - Test dependency validation
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8_

- [x] 20. Checkpoint - Verify error handling
  - Ensure all tests pass, ask the user if questions arise.

- [x] 21. Implement command-line interface and configuration
  - [x] 21.1 Create CLI using argparse
    - Add arguments for model paths, quantization levels, context size, batch size
    - Add arguments for number of iterations, output directory
    - Add arguments for enabling/disabling ablation studies, batch testing, thermal monitoring
    - Support configuration file path argument (JSON/YAML)
    - Prioritize command-line arguments over configuration file
    - Validate all configuration parameters and report errors for invalid values
    - Provide default values for all optional parameters
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9_

  - [x] 21.2 Write unit tests for CLI
    - Test argument parsing with various combinations
    - Test configuration file loading
    - Test command-line argument priority over config file
    - Test validation of invalid parameters
    - Test default value assignment
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9_

- [x] 22. Integrate all components and create main entry point
  - [x] 22.1 Create main benchmark script
    - Validate dependencies before starting
    - Parse configuration from CLI and config file
    - Detect hardware and create appropriate backend
    - Initialize model manager, metrics collector, and orchestrator
    - Run quantization profiling tests
    - Run ablation studies if enabled
    - Run batch processing tests if enabled
    - Perform statistical validation on results
    - Generate visualizations and reports
    - Handle errors gracefully with informative messages
    - _Requirements: 7.1, 7.2, 7.6, 7.7, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 12.7_

  - [x] 22.2 Write end-to-end integration test
    - Test complete benchmark run on x86 Linux with small test model
    - Verify all components are integrated correctly
    - Verify results are saved in correct format
    - Verify visualizations are generated
    - Verify HTML report is created
    - _Requirements: 7.1, 7.2, 8.6, 9.8_

- [x] 23. Create documentation and examples
  - [x] 23.1 Write user documentation
    - Create README with installation instructions
    - Document command-line arguments and configuration options
    - Provide example configuration files for common scenarios
    - Document hardware requirements and platform support
    - Create troubleshooting guide for common issues
    - _Requirements: 8.5, 11.6, 11.9_

  - [x] 23.2 Create example scripts
    - Create example for basic quantization profiling
    - Create example for ablation studies
    - Create example for batch processing tests
    - Create example configuration for x86 Linux
    - Create example configuration for Jetson Xavier NX
    - _Requirements: 11.6_

- [x] 24. Set up CI/CD pipeline
  - [x] 24.1 Create GitHub Actions workflow
    - Set up Python environment with required dependencies
    - Download small test models for CI testing
    - Run unit tests with coverage reporting
    - Run integration tests (excluding Jetson-specific tests)
    - Upload coverage reports to codecov
    - Run on push and pull request events
    - _Requirements: 8.5_

  - [x] 24.2 Create test data and fixtures
    - Download small test models (< 1GB) for fast test execution
    - Create fixed test prompts for reproducibility
    - Create baseline results for regression detection
    - Document expected metric ranges for different hardware
    - _Requirements: 8.7_

- [x] 25. Final checkpoint - Complete system verification
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at major milestones
- The implementation follows the 8-week roadmap from the design document:
  - Phase 1 (Week 1-2): Tasks 1-3 (Core Infrastructure)
  - Phase 2 (Week 2-3): Tasks 4-10 (Quantization Profiling)
  - Phase 3 (Week 3-4): Task 11 (GPU Acceleration)
  - Phase 4 (Week 4-5): Tasks 12-14 (Ablation Studies)
  - Phase 5 (Week 5-6): Tasks 15-17 (Statistical Validation)
  - Phase 6 (Week 6-7): Task 18 (Visualization and Reporting)
  - Phase 7 (Week 7-8): Tasks 19-25 (Error Handling, CLI, Documentation, CI/CD)
- Property tests validate statistical calculations (confidence intervals, outlier detection)
- Integration tests validate end-to-end workflows with real models and hardware
- Unit tests validate component logic in isolation with mocked dependencies
- Manual testing on Jetson Xavier NX is required for GPU-specific features
