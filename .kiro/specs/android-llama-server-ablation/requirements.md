# Requirements Document: Android llama-server Ablation Support

## Introduction

This document specifies requirements for implementing llama-server support in the Android ablation engine to enable accurate cache ablation studies. Currently, the AndroidAblationEngine uses llama-cli which cannot disable KV cache, preventing true "no cache" baseline measurements. This feature will add llama-server integration with proper cache control flags (--cache-ram 0, --no-cache-prompt) to enable rigorous ablation studies when `enable_ablation_studies: true` is configured for Android.

The implementation will maintain backward compatibility with existing llama-cli behavior while providing automatic backend selection based on ablation configuration and llama-server availability.

## Glossary

- **AndroidAblationEngine**: The ablation testing engine for Android/Termux that measures cache performance
- **llama-cli**: Command-line inference binary from llama.cpp that always has KV cache enabled
- **llama-server**: HTTP server binary from llama.cpp that provides cache control via API and command-line flags
- **KV_Cache**: Key-Value cache stored in RAM that accelerates inference by caching attention computations
- **Prompt_Cache**: Disk-based cache that stores processed prompt embeddings for reuse across sessions
- **NativeLlamaCpp**: Python wrapper class for llama-cli subprocess execution
- **NativeLlamaServer**: New Python wrapper class for llama-server subprocess and HTTP API client
- **Ablation_Study**: Controlled experiment that isolates and measures the effect of individual system components
- **Control_Run**: Baseline measurement with no caching (requires --cache-ram 0 flag)
- **Cold_Cache_Run**: Measurement with cache enabled but empty (first use)
- **Warm_Cache_Run**: Measurement with cache populated and reused
- **Backend_Selection**: Process of choosing between llama-cli and llama-server based on configuration
- **AndroidBackend**: Hardware abstraction layer for Android platform in the HAL module
- **MetricsCollector**: Component that collects inference performance metrics (TTFT, TPS, memory)

## Requirements

### Requirement 1: Automatic Backend Selection for Ablation Studies

**User Story:** As a researcher, I want the system to automatically use llama-server when ablation studies are enabled on Android, so that I get accurate cache measurements without manual configuration.

#### Acceptance Criteria

1. WHEN `enable_ablation_studies` is true AND the platform is Android, THE AndroidBackend SHALL use llama-server for inference
2. WHEN `enable_ablation_studies` is false AND the platform is Android, THE AndroidBackend SHALL use llama-cli for inference
3. WHEN llama-server binary is not available AND `enable_ablation_studies` is true, THE AndroidBackend SHALL fall back to llama-cli and log a warning message
4. THE AndroidBackend SHALL detect llama-server availability by checking for the binary at ~/llama.cpp/build/bin/llama-server
5. WHERE `use_llama_server_for_ablation` configuration option is provided, THE AndroidBackend SHALL respect the explicit configuration value

### Requirement 2: llama-server Subprocess Management

**User Story:** As a developer, I want the system to manage llama-server lifecycle automatically, so that I don't need to manually start and stop the server process.

#### Acceptance Criteria

1. THE NativeLlamaServer SHALL start llama-server as a subprocess when initialized
2. WHEN NativeLlamaServer is initialized, THE NativeLlamaServer SHALL pass model path, context size, thread count, and cache control flags to llama-server
3. THE NativeLlamaServer SHALL verify llama-server is ready by polling the /health endpoint with a timeout of 30 seconds
4. WHEN NativeLlamaServer is destroyed or explicitly closed, THE NativeLlamaServer SHALL terminate the llama-server subprocess
5. IF llama-server subprocess crashes, THEN THE NativeLlamaServer SHALL detect the failure and raise an appropriate exception
6. THE NativeLlamaServer SHALL capture llama-server stdout and stderr for debugging purposes

### Requirement 3: HTTP API Client for Inference

**User Story:** As a developer, I want to send inference requests to llama-server via HTTP API, so that I can control cache behavior per request.

#### Acceptance Criteria

1. THE NativeLlamaServer SHALL send POST requests to the /completion endpoint for text generation
2. WHEN generating text, THE NativeLlamaServer SHALL include prompt, n_predict (max tokens), and cache_prompt parameters in the request body
3. THE NativeLlamaServer SHALL stream responses from llama-server using chunked transfer encoding
4. WHEN streaming responses, THE NativeLlamaServer SHALL yield text chunks in the same format as NativeLlamaCpp for compatibility
5. IF the HTTP request fails with a connection error, THEN THE NativeLlamaServer SHALL raise a ConnectionError with diagnostic information
6. IF the HTTP request times out, THEN THE NativeLlamaServer SHALL raise a TimeoutError with the configured timeout value
7. THE NativeLlamaServer SHALL set request timeout to (max_tokens * 2 + 60) seconds

### Requirement 4: Cache Control via Command-Line Flags

**User Story:** As a researcher, I want to disable all caching for control runs, so that I can establish a true baseline without any cache effects.

#### Acceptance Criteria

1. WHEN cache_mode is "none", THE NativeLlamaServer SHALL start llama-server with --cache-ram 0 and --no-cache-prompt flags
2. WHEN cache_mode is "ram_only", THE NativeLlamaServer SHALL start llama-server with --no-cache-prompt flag only
3. WHEN cache_mode is "disk_only", THE NativeLlamaServer SHALL start llama-server with --cache-ram 0 flag only
4. WHEN cache_mode is "both" or not specified, THE NativeLlamaServer SHALL start llama-server without cache restriction flags
5. THE NativeLlamaServer SHALL log the cache configuration flags used when starting llama-server

### Requirement 5: Per-Request Cache Control

**User Story:** As a researcher, I want to control prompt caching on a per-request basis, so that I can test cold vs warm cache scenarios without restarting the server.

#### Acceptance Criteria

1. WHEN enable_prompt_cache is false, THE NativeLlamaServer SHALL set cache_prompt to false in the API request
2. WHEN enable_prompt_cache is true, THE NativeLlamaServer SHALL set cache_prompt to true in the API request
3. THE NativeLlamaServer SHALL default cache_prompt to false to prevent unintended caching
4. THE NativeLlamaServer SHALL accept enable_prompt_cache as a parameter to the __call__ method

### Requirement 6: True Cache Ablation Studies

**User Story:** As a researcher, I want to measure the pure effect of each cache type independently, so that I can understand which caching strategy is most effective for my workload.

#### Acceptance Criteria

1. WHEN running control ablation, THE AndroidAblationEngine SHALL configure NativeLlamaServer with cache_mode "none" and enable_prompt_cache false
2. WHEN running cold cache ablation, THE AndroidAblationEngine SHALL configure NativeLlamaServer with cache_mode "both" and enable_prompt_cache true on first request
3. WHEN running warm cache ablation, THE AndroidAblationEngine SHALL configure NativeLlamaServer with cache_mode "both" and enable_prompt_cache true on subsequent requests
4. THE AndroidAblationEngine SHALL verify control runs have zero cache activity by checking llama-server logs for cache-related messages
5. THE AndroidAblationEngine SHALL measure and report the isolated effect of RAM cache by comparing control vs ram_only scenarios
6. THE AndroidAblationEngine SHALL measure and report the isolated effect of disk cache by comparing control vs disk_only scenarios

### Requirement 7: Backward Compatibility with llama-cli

**User Story:** As a user, I want existing llama-cli functionality to continue working unchanged, so that my current workflows are not disrupted.

#### Acceptance Criteria

1. WHEN `enable_ablation_studies` is false, THE AndroidBackend SHALL use NativeLlamaCpp (llama-cli wrapper) as before
2. THE NativeLlamaCpp class SHALL remain unchanged in its public interface
3. WHEN using NativeLlamaCpp, THE system SHALL produce the same metrics and results as before this feature
4. THE existing unit tests for NativeLlamaCpp SHALL continue to pass without modification
5. THE existing integration tests for AndroidAblationEngine with llama-cli SHALL continue to pass

### Requirement 8: Configuration Options

**User Story:** As a user, I want to configure llama-server behavior through the configuration file, so that I can customize the setup for my environment.

#### Acceptance Criteria

1. THE system SHALL support a `use_llama_server_for_ablation` configuration option in the Android config file
2. WHEN `use_llama_server_for_ablation` is true, THE AndroidBackend SHALL use llama-server regardless of ablation settings
3. WHEN `use_llama_server_for_ablation` is false, THE AndroidBackend SHALL use llama-cli regardless of ablation settings
4. WHEN `use_llama_server_for_ablation` is not specified, THE AndroidBackend SHALL default to true on Android when ablation is enabled
5. THE system SHALL support a `llama_server_port` configuration option with default value 8080
6. THE system SHALL support a `llama_server_host` configuration option with default value "127.0.0.1"

### Requirement 9: Error Handling and Diagnostics

**User Story:** As a user, I want clear error messages when llama-server is not available or fails, so that I can troubleshoot issues quickly.

#### Acceptance Criteria

1. WHEN llama-server binary is not found, THE AndroidBackend SHALL log an error message with the expected binary path
2. WHEN llama-server binary is not found AND ablation is enabled, THE AndroidBackend SHALL log instructions for building llama-server
3. WHEN llama-server fails to start, THE NativeLlamaServer SHALL raise an exception with stderr output from llama-server
4. WHEN llama-server health check times out, THE NativeLlamaServer SHALL raise a TimeoutError with diagnostic information
5. WHEN falling back to llama-cli, THE AndroidBackend SHALL log a warning explaining the limitation of llama-cli for ablation studies
6. THE NativeLlamaServer SHALL include the llama-server process ID in log messages for debugging

### Requirement 10: NativeLlamaServer Class Implementation

**User Story:** As a developer, I want a NativeLlamaServer class with the same interface as NativeLlamaCpp, so that I can use them interchangeably in the codebase.

#### Acceptance Criteria

1. THE NativeLlamaServer SHALL implement __init__ method accepting model_path, n_ctx, n_threads, n_batch, cache_mode, and llama_server_path parameters
2. THE NativeLlamaServer SHALL implement __call__ method accepting prompt, max_tokens, stream, and enable_prompt_cache parameters
3. THE NativeLlamaServer SHALL implement create_completion method as an alias to __call__ for compatibility
4. THE NativeLlamaServer SHALL implement tokenize method that returns approximate token count (character count divided by 4)
5. THE NativeLlamaServer SHALL implement close method that terminates the llama-server subprocess
6. THE NativeLlamaServer SHALL implement __enter__ and __exit__ methods for context manager support
7. THE NativeLlamaServer SHALL store subprocess PID in last_subprocess_pid attribute for memory measurement compatibility

### Requirement 11: Integration with AndroidAblationEngine

**User Story:** As a researcher, I want AndroidAblationEngine to automatically use llama-server when available, so that I get accurate ablation results without code changes.

#### Acceptance Criteria

1. THE AndroidAblationEngine SHALL detect whether the loaded model is NativeLlamaServer or NativeLlamaCpp
2. WHEN using NativeLlamaServer, THE AndroidAblationEngine SHALL configure cache_mode for each ablation scenario
3. WHEN using NativeLlamaCpp, THE AndroidAblationEngine SHALL use the existing prompt_cache file approach
4. THE AndroidAblationEngine SHALL update result configuration metadata to indicate which backend was used
5. THE AndroidAblationEngine SHALL log a warning when using NativeLlamaCpp for ablation studies explaining the limitation

### Requirement 12: Testing and Validation

**User Story:** As a developer, I want comprehensive tests for llama-server integration, so that I can verify correctness and prevent regressions.

#### Acceptance Criteria

1. THE test suite SHALL include unit tests for NativeLlamaServer subprocess management
2. THE test suite SHALL include unit tests for NativeLlamaServer HTTP API client
3. THE test suite SHALL include integration tests for AndroidAblationEngine with llama-server
4. THE test suite SHALL include comparison tests verifying llama-cli and llama-server produce similar results for the same prompt
5. THE test suite SHALL include tests for fallback behavior when llama-server is not available
6. THE test suite SHALL include tests for error handling (server crash, timeout, connection failure)
7. THE test suite SHALL mock HTTP requests to avoid requiring actual llama-server binary in CI

### Requirement 13: Documentation Updates

**User Story:** As a user, I want updated documentation explaining llama-server setup and usage, so that I can configure my environment correctly.

#### Acceptance Criteria

1. THE documentation SHALL include instructions for building llama-server on Android/Termux
2. THE documentation SHALL explain the difference between llama-cli and llama-server for ablation studies
3. THE documentation SHALL provide example configurations for enabling llama-server
4. THE documentation SHALL update ANDROID_ABLATION_LIMITATIONS.md to reflect that limitations are resolved with llama-server
5. THE documentation SHALL include troubleshooting steps for common llama-server issues
6. THE documentation SHALL document the cache_mode parameter and its valid values

### Requirement 14: Memory Measurement Compatibility

**User Story:** As a researcher, I want memory measurements to work correctly with llama-server, so that I can track memory usage during ablation studies.

#### Acceptance Criteria

1. THE NativeLlamaServer SHALL track subprocess PID in last_subprocess_pid attribute
2. THE NativeLlamaServer SHALL set subprocess_is_running flag to true when server is active
3. THE NativeLlamaServer SHALL track peak memory usage in subprocess_peak_memory_kb attribute
4. THE NativeLlamaServer SHALL use a background thread to monitor subprocess memory every 50ms
5. THE MetricsCollector SHALL use last_subprocess_pid to measure llama-server memory usage
6. THE memory measurement approach SHALL be consistent between NativeLlamaCpp and NativeLlamaServer

### Requirement 15: Performance Parity with llama-cli

**User Story:** As a user, I want llama-server to have similar performance to llama-cli, so that switching backends doesn't significantly impact benchmark results.

#### Acceptance Criteria

1. WHEN comparing llama-cli and llama-server with the same cache configuration, THE difference in TTFT SHALL be less than 10%
2. WHEN comparing llama-cli and llama-server with the same cache configuration, THE difference in decode TPS SHALL be less than 5%
3. WHEN comparing llama-cli and llama-server with the same cache configuration, THE difference in memory usage SHALL be less than 50MB
4. THE NativeLlamaServer SHALL use the same thread count and batch size as NativeLlamaCpp by default
5. THE NativeLlamaServer SHALL minimize HTTP overhead by using keep-alive connections

