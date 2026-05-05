# Implementation Plan: Android llama-server Ablation Support

## Overview

This implementation plan converts the feature design into a series of coding tasks for implementing llama-server integration in the Android ablation engine. The tasks build incrementally from core infrastructure through backend selection, HTTP client implementation, and finally integration with the existing ablation engine.

## Tasks

- [ ] 1. Set up core infrastructure and data models
  - [x] 1.1 Create NativeLlamaServer class structure and initialization
    - Create `native_llama_server.py` file with class definition
    - Implement `__init__` method with all required parameters (model_path, n_ctx, n_threads, n_batch, cache_mode, llama_server_path, host, port)
    - Add subprocess management attributes and HTTP client setup
    - _Requirements: 10.1, 2.1, 2.2_

  - [x] 1.2 Write property test for NativeLlamaServer initialization
    - **Property 2: Command-Line Argument Construction**
    - **Validates: Requirements 2.2, 4.1, 4.2, 4.3, 4.4**

  - [x] 1.3 Create cache configuration data models
    - Define `CacheMode` enum with all cache modes (none, ram_only, disk_only, both)
    - Create `ABLATION_CACHE_CONFIG` mapping for ablation scenarios
    - Add cache mode validation logic
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 6.1, 6.2, 6.3_

  - [x] 1.4 Write unit tests for cache configuration models
    - Test enum values and validation
    - Test ablation scenario mappings
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 2. Implement subprocess management and health monitoring
  - [x] 2.1 Implement llama-server subprocess startup
    - Build command-line arguments based on cache_mode and parameters
    - Start subprocess with `subprocess.Popen`
    - Capture stdout/stderr for debugging
    - _Requirements: 2.1, 2.2, 2.6, 4.5_

  - [x] 2.2 Implement health check polling system
    - Create `/health` endpoint polling with 30-second timeout
    - Implement exponential backoff retry strategy
    - Add process status verification via `poll()`
    - _Requirements: 2.3, 9.4_

  - [x] 2.3 Write property test for command-line argument construction
    - **Property 2: Command-Line Argument Construction**
    - **Validates: Requirements 2.2, 4.1, 4.2, 4.3, 4.4**

  - [x] 2.4 Implement subprocess cleanup and termination
    - Add `close()` method to terminate subprocess
    - Implement context manager methods (`__enter__`, `__exit__`)
    - Handle graceful shutdown and forced termination
    - _Requirements: 2.4, 10.5, 10.6_

  - [x] 2.5 Write unit tests for subprocess management
    - Mock `subprocess.Popen` and test process creation
    - Test health check polling scenarios
    - Test cleanup behavior
    - _Requirements: 2.1, 2.3, 2.4_

- [ ] 3. Checkpoint - Ensure subprocess management tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Implement HTTP API client for inference
  - [x] 4.1 Create HTTP client for completion requests
    - Implement POST requests to `/completion` endpoint
    - Build request body with prompt, n_predict, cache_prompt parameters
    - Add proper headers and timeout configuration
    - _Requirements: 3.1, 3.2, 3.7_

  - [x] 4.2 Implement streaming response handling
    - Parse Server-Sent Events (SSE) format responses
    - Handle chunked transfer encoding
    - Extract and yield text chunks for compatibility
    - _Requirements: 3.3, 3.4_

  - [x] 4.3 Write property test for HTTP request construction
    - **Property 3: HTTP Request Body Construction**
    - **Validates: Requirements 3.2, 5.1, 5.2**

  - [x] 4.4 Implement error handling for HTTP requests
    - Handle connection errors with diagnostic information
    - Handle timeout errors with configured timeout values
    - Add HTTP status code validation and JSON parsing error recovery
    - _Requirements: 3.5, 3.6, 9.3, 9.4_

  - [x] 4.5 Write property test for timeout calculation
    - **Property 4: Timeout Calculation**
    - **Validates: Requirements 3.7**

  - [x] 4.6 Implement `__call__` and `create_completion` methods
    - Create main inference interface matching NativeLlamaCpp
    - Add streaming and non-streaming response handling
    - Implement per-request cache control via enable_prompt_cache parameter
    - _Requirements: 10.2, 10.3, 5.1, 5.2, 5.3, 5.4_

  - [x] 4.7 Write unit tests for HTTP client functionality
    - Mock `requests` library for API interactions
    - Test streaming response parsing
    - Test error conditions and recovery
    - _Requirements: 3.1, 3.3, 3.5, 3.6_

- [ ] 5. Implement memory monitoring and compatibility features
  - [x] 5.1 Add memory monitoring background thread
    - Sample memory usage every 50ms
    - Track peak memory in `subprocess_peak_memory_kb`
    - Update `subprocess_is_running` flag
    - _Requirements: 14.1, 14.2, 14.3, 14.4_

  - [x] 5.2 Implement compatibility attributes and methods
    - Add `last_subprocess_pid` attribute for external monitoring
    - Implement `tokenize` method with approximate token count (char_count / 4)
    - Ensure interface compatibility with NativeLlamaCpp
    - _Requirements: 10.4, 10.7, 14.5, 14.6_

  - [x] 5.3 Write unit tests for memory monitoring
    - Test background thread behavior
    - Test compatibility attributes
    - Test tokenize method accuracy
    - _Requirements: 14.1, 14.2, 14.3, 14.4_

- [ ] 6. Checkpoint - Ensure HTTP client and monitoring tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Implement backend selection logic
  - [x] 7.1 Create AndroidConfig data model
    - Define configuration schema with all llama-server options
    - Add validation for configuration parameters
    - Set appropriate default values
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

  - [-] 7.2 Implement backend selection in AndroidBackend
    - Add logic for automatic backend selection based on ablation settings
    - Implement binary availability detection
    - Add explicit configuration override support
    - _Requirements: 1.1, 1.2, 1.4, 1.5_

  - [ ] 7.3 Write property test for backend selection logic
    - **Property 1: Backend Selection Logic**
    - **Validates: Requirements 1.1, 1.2, 1.5**

  - [ ] 7.4 Implement fallback behavior and logging
    - Add fallback to llama-cli when llama-server unavailable
    - Implement warning messages for ablation study limitations
    - Add diagnostic error messages for troubleshooting
    - _Requirements: 1.3, 9.1, 9.2, 9.5, 11.5_

  - [ ] 7.5 Write property test for fallback behavior
    - **Property 6: Fallback Behavior**
    - **Validates: Requirements 1.3**

  - [ ] 7.6 Write unit tests for backend selection
    - Test configuration parsing and validation
    - Test binary availability detection
    - Test fallback logic scenarios
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 8. Integrate with AndroidAblationEngine
  - [ ] 8.1 Update AndroidAblationEngine to detect backend type
    - Add logic to detect NativeLlamaServer vs NativeLlamaCpp
    - Implement cache_mode configuration for ablation scenarios
    - Maintain existing prompt_cache file approach for llama-cli
    - _Requirements: 11.1, 11.2, 11.3_

  - [ ] 8.2 Implement ablation scenario cache configuration
    - Map control runs to cache_mode "none" and enable_prompt_cache false
    - Map cold cache runs to cache_mode "both" with proper prompt cache handling
    - Map warm cache runs to cache_mode "both" with cache reuse
    - _Requirements: 6.1, 6.2, 6.3_

  - [ ] 8.3 Update result metadata and logging
    - Add backend type to result configuration metadata
    - Log warnings when using llama-cli for ablation studies
    - Add cache activity verification for control runs
    - _Requirements: 11.4, 11.5, 6.4_

  - [ ] 8.4 Write integration tests for AndroidAblationEngine
    - Test ablation engine with both backends
    - Test cache configuration scenarios
    - Test result metadata generation
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

- [ ] 9. Implement comprehensive error handling
  - [ ] 9.1 Add detailed error messages and diagnostics
    - Implement binary not found error with expected path
    - Add llama-server build instructions in error messages
    - Include process ID in log messages for debugging
    - _Requirements: 9.1, 9.2, 9.6_

  - [ ] 9.2 Implement subprocess failure detection
    - Add llama-server crash detection and exception raising
    - Include stderr output in failure exceptions
    - Handle health check timeout with diagnostic information
    - _Requirements: 2.5, 9.3, 9.4_

  - [ ] 9.3 Write unit tests for error handling
    - Test various failure scenarios
    - Test error message content and formatting
    - Test exception types and diagnostic information
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

- [ ] 10. Final integration and compatibility verification
  - [ ] 10.1 Ensure backward compatibility with existing code
    - Verify NativeLlamaCpp class remains unchanged
    - Test existing unit tests continue to pass
    - Verify existing integration tests work with llama-cli
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [ ] 10.2 Write property test for output format compatibility
    - **Property 5: Output Format Compatibility**
    - **Validates: Requirements 3.4**

  - [ ] 10.3 Add comprehensive integration tests
    - Test complete ablation study execution with mocked llama-server
    - Test performance comparison between backends (mocked measurements)
    - Test configuration file parsing and application
    - _Requirements: 12.3, 12.4, 12.5_

  - [ ] 10.4 Write comparison tests for backend parity
    - Test llama-cli and llama-server produce similar results
    - Test performance parity within acceptable thresholds
    - Test memory usage compatibility
    - _Requirements: 12.4, 15.1, 15.2, 15.3, 15.4, 15.5_

- [ ] 11. Final checkpoint - Ensure all tests pass and integration works
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation throughout development
- Property tests validate universal correctness properties from the design
- Unit tests validate specific examples and edge cases
- Integration tests verify end-to-end functionality
- The implementation maintains strict backward compatibility with existing llama-cli workflows