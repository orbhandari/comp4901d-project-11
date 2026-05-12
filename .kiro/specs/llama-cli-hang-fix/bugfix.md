# Bugfix Requirements Document

## Introduction

The Android native llama.cpp implementation has a critical bug where the `llama-cli` binary enters conversation/interactive mode and hangs indefinitely, preventing the benchmark from completing. The binary prints infinite `>` characters and never exits on its own, even when Ctrl-C is pressed. This makes the benchmark unusable on Android/Termux, forcing users to disconnect from Termux completely to regain control.

The bug occurs because the user's `llama-cli` version defaults to conversation/interactive mode, and there is no flag available to disable it. The binary ignores stdin state (even when set to DEVNULL or closed) and enters conversation mode regardless of the flags provided.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN `llama-cli` is invoked with a prompt and token count THEN the system enters conversation/interactive mode and prints infinite `>` characters

1.2 WHEN `llama-cli` enters conversation mode THEN the system never exits on its own, causing the benchmark to hang indefinitely

1.3 WHEN `llama-cli` is running in conversation mode THEN the system ignores Ctrl-C signals and cannot be killed normally

1.4 WHEN `llama-cli` hangs in conversation mode THEN the system forces the user to disconnect from Termux completely to regain control

1.5 WHEN `stdin=DEVNULL` is used THEN the system still enters conversation mode (stdin state is ignored)

1.6 WHEN `--simple-io` flag is used THEN the system still prints `>` infinitely (flag doesn't prevent conversation mode)

1.7 WHEN `timeout` command wrapper is used THEN the system process continues running after timeout expires (timeout doesn't work reliably)

### Expected Behavior (Correct)

2.1 WHEN `llama-cli` is invoked with a prompt and token count THEN the system SHALL generate the requested tokens and exit cleanly without entering conversation mode

2.2 WHEN text generation completes THEN the system SHALL exit with return code 0 and allow the benchmark to continue

2.3 WHEN the process needs to be terminated THEN the system SHALL respond to Ctrl-C (SIGINT) or SIGTERM signals and exit gracefully

2.4 WHEN the process exceeds a reasonable timeout THEN the system SHALL be forcibly killed (SIGKILL) and the benchmark SHALL handle the timeout error appropriately

2.5 WHEN alternative llama.cpp binaries are available (e.g., `main`, `llama-simple`) THEN the system SHALL attempt to use them as fallback options if `llama-cli` is problematic

2.6 WHEN no working binary is found THEN the system SHALL provide clear error messages with troubleshooting steps for the user

### Unchanged Behavior (Regression Prevention)

3.1 WHEN text generation succeeds THEN the system SHALL CONTINUE TO return generated text in the same format (character-by-character streaming)

3.2 WHEN the model is loaded successfully THEN the system SHALL CONTINUE TO use the same configuration parameters (n_ctx, n_threads, n_batch)

3.3 WHEN running on non-Android platforms (x86, Jetson) THEN the system SHALL CONTINUE TO use llama-cpp-python without changes

3.4 WHEN tokenization is requested THEN the system SHALL CONTINUE TO provide approximate token counts using the same heuristic (1 token ≈ 4 characters)

3.5 WHEN errors occur during model loading or execution THEN the system SHALL CONTINUE TO log appropriate error messages and raise exceptions

3.6 WHEN the benchmark measures TTFT (Time To First Token) THEN the system SHALL CONTINUE TO support character-by-character streaming for accurate timing
