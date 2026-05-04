# llama-cli Hang Bugfix Design

## Overview

The Android native llama.cpp implementation has a critical bug where the `llama-cli` binary enters conversation/interactive mode and hangs indefinitely. The binary prints infinite `>` characters and never exits, even when stdin is set to DEVNULL or when timeout commands are used. This makes the benchmark completely unusable on Android/Termux.

The fix strategy involves multiple layers of defense:
1. **Binary Selection**: Try alternative binaries (`main`, `llama-simple`) that don't have conversation mode
2. **Process Termination**: Use aggressive process group killing with SIGKILL
3. **Timeout Enforcement**: Implement Python-level timeout with forced termination
4. **Flag Optimization**: Use minimal, universally-supported flags to avoid triggering conversation mode
5. **Streaming Compatibility**: Maintain character-by-character streaming for TTFT measurement

## Glossary

- **Bug_Condition (C)**: The condition that triggers the bug - when `llama-cli` enters conversation mode and hangs
- **Property (P)**: The desired behavior - binary generates tokens and exits cleanly without hanging
- **Preservation**: Existing streaming behavior and TTFT measurement accuracy that must remain unchanged
- **llama-cli**: The primary llama.cpp binary (newer, has conversation mode issues)
- **main**: The legacy llama.cpp binary (older, usually doesn't have conversation mode)
- **llama-simple**: Alternative simplified binary (if available)
- **Conversation Mode**: Interactive mode where binary prints `>` prompts and waits for user input
- **SIGKILL**: Forceful process termination signal that cannot be ignored
- **Process Group**: Parent process and all child processes that should be killed together

## Bug Details

### Bug Condition

The bug manifests when `llama-cli` is invoked with a prompt and token count. The binary enters conversation/interactive mode instead of generating tokens and exiting. This happens because:
1. The user's `llama-cli` version defaults to conversation mode
2. There is no flag to reliably disable conversation mode
3. The binary ignores stdin state (DEVNULL, closed, EOF)
4. Standard timeout mechanisms don't work reliably

**Formal Specification:**
```
FUNCTION isBugCondition(execution)
  INPUT: execution of type ProcessExecution
  OUTPUT: boolean
  
  RETURN execution.binary == "llama-cli"
         AND execution.stdout CONTAINS ">"
         AND execution.process_state == "running"
         AND execution.elapsed_time > expected_generation_time
         AND NOT execution.has_exited
END FUNCTION
```

### Examples

- **Example 1**: Running `llama-cli -m model.gguf -p "Hello" -n 50` prints `>` infinitely and never exits
- **Example 2**: Using `stdin=DEVNULL` still results in conversation mode (stdin state ignored)
- **Example 3**: Using `--simple-io` flag still prints `>` infinitely (flag doesn't prevent conversation mode)
- **Example 4**: Using `timeout 60s llama-cli ...` continues running after 60 seconds (timeout doesn't work)
- **Edge Case**: Pressing Ctrl-C doesn't kill the process (SIGINT ignored in conversation mode)

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Character-by-character streaming must continue to work for TTFT measurement
- Model loading configuration (n_ctx, n_threads, n_batch) must remain unchanged
- Non-Android platforms (x86, Jetson) must continue using llama-cpp-python
- Tokenization approximation (1 token ≈ 4 characters) must remain unchanged
- Error logging and exception handling must remain unchanged

**Scope:**
All inputs that do NOT involve Android native llama.cpp execution should be completely unaffected by this fix. This includes:
- llama-cpp-python usage on x86 and Jetson platforms
- Model loading and validation logic
- Metrics collection and hardware detection
- All other benchmark functionality

## Hypothesized Root Cause

Based on the bug description and code analysis, the most likely issues are:

1. **Binary Version Issue**: The user's `llama-cli` binary is a newer version that defaults to conversation/interactive mode, and there is no flag to disable it reliably. The binary may have been compiled with conversation mode as the default behavior.

2. **Stdin Handling Bug**: The binary ignores stdin state (DEVNULL, closed, EOF) and enters conversation mode regardless. This is a bug in the binary itself, not in our code.

3. **Timeout Command Limitation**: The `timeout` command wrapper doesn't work reliably on Android/Termux. The process may be running in a way that makes it immune to the timeout signal, or the timeout command itself may not be functioning correctly.

4. **Process Group Issue**: The process may be spawning child processes or detaching from the parent, making it difficult to kill. Standard SIGTERM may not reach all processes in the group.

5. **Flag Incompatibility**: Some flags used in the command (like `--log-disable`, `-ngl`) may be triggering conversation mode as a fallback behavior when the binary doesn't recognize them.

## Correctness Properties

Property 1: Bug Condition - Clean Exit After Generation

_For any_ execution where `llama-cli` is invoked with a prompt and token count, the fixed implementation SHALL generate the requested tokens and exit cleanly within a reasonable timeout (max_tokens * 2 seconds + 60s buffer), without entering conversation mode or hanging indefinitely.

**Validates: Requirements 2.1, 2.2, 2.4**

Property 2: Preservation - Streaming and TTFT Measurement

_For any_ execution that successfully generates text, the fixed implementation SHALL produce exactly the same character-by-character streaming behavior as the original implementation, preserving TTFT measurement accuracy and output format.

**Validates: Requirements 3.1, 3.6**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct, we need to make the following changes:

**File**: `llm_benchmark/inference/native_llama.py`

**Function**: `NativeLlamaCpp.__call__`

**Specific Changes**:

1. **Binary Selection Enhancement**: Improve the binary selection logic to prefer `main` over `llama-cli` when both are available, since `main` is less likely to have conversation mode issues.
   - Check for `main` binary first
   - Fall back to `llama-cli` only if `main` is not available
   - Add `llama-simple` as a third fallback option
   - Log which binary is being used for debugging

2. **Remove Timeout Command Wrapper**: Remove the `timeout` command wrapper since it doesn't work reliably on Android/Termux.
   - Remove `["timeout", f"{timeout_seconds}s"]` from command construction
   - Implement timeout at Python level instead

3. **Implement Python-Level Timeout**: Use `subprocess.communicate(timeout=...)` with aggressive process killing on timeout.
   - Set timeout to `max_tokens * 2 + 60` seconds (2 seconds per token + 60s buffer)
   - On timeout, kill the entire process group with SIGKILL
   - Use `os.killpg()` to kill process group, not just parent process
   - Log timeout errors with troubleshooting information

4. **Simplify Command Flags**: Remove potentially problematic flags that might trigger conversation mode.
   - Remove `--log-disable` flag (may not be supported in all versions)
   - Remove `-ngl` flag (GPU is already disabled by default)
   - Keep only essential flags: `-m`, `-c`, `-t`, `-b`, `-n`, `-p`
   - Add `--no-display-prompt` flag if available (check binary version)

5. **Add Process Group Creation**: Create a new process group for the subprocess to enable killing all child processes.
   - Use `start_new_session=True` in Popen to create new process group
   - On timeout or error, kill entire process group with `os.killpg(os.getpgid(process.pid), signal.SIGKILL)`
   - Handle errors gracefully if process group killing fails

6. **Improve Error Messages**: Provide clearer error messages when timeout occurs.
   - Log which binary was used
   - Log the exact command that was run
   - Suggest trying alternative binaries
   - Provide instructions for checking available binaries

7. **Add Binary Capability Detection**: Detect which flags are supported by the binary before using them.
   - Run `llama-cli --help` to check available flags
   - Cache the results to avoid repeated checks
   - Only use flags that are confirmed to be supported

### Implementation Details

**Binary Selection Priority**:
```python
# Priority order: main -> llama-simple -> llama-cli
# main is preferred because it's older and doesn't have conversation mode
binaries_to_try = [
    ("main", llama_cli_path_expanded.parent / "main"),
    ("llama-simple", llama_cli_path_expanded.parent / "llama-simple"),
    ("llama-cli", llama_cli_path_expanded),
]

for binary_type, binary_path in binaries_to_try:
    if binary_path.exists():
        self.llama_cli_path = binary_path
        self.binary_type = binary_type
        logger.info(f"Using '{binary_type}' binary: {binary_path}")
        break
else:
    raise FileNotFoundError(...)
```

**Process Group Killing**:
```python
import os
import signal

# Create process with new session (process group)
process = subprocess.Popen(
    cmd,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    start_new_session=True  # Create new process group
)

# On timeout, kill entire process group
try:
    stdout, stderr = process.communicate(timeout=timeout_seconds)
except subprocess.TimeoutExpired:
    logger.error(f"Process timed out after {timeout_seconds}s")
    
    # Kill entire process group with SIGKILL
    try:
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal.SIGKILL)
        logger.info(f"Killed process group {pgid}")
    except Exception as e:
        logger.warning(f"Failed to kill process group: {e}")
        # Fallback: kill just the parent process
        process.kill()
    
    # Wait for process to die
    stdout, stderr = process.communicate()
    
    raise TimeoutError(...)
```

**Simplified Command Construction**:
```python
# Minimal flags - only use universally supported ones
cmd = [
    str(self.llama_cli_path),
    "-m", str(self.model_path),
    "-c", str(self.n_ctx),
    "-t", str(self.n_threads),
    "-b", str(self.n_batch),
    "-n", str(max_tokens),
    "-p", prompt,
]

# No timeout wrapper, no extra flags
# Keep it simple to avoid triggering conversation mode
```

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code, then verify the fix works correctly and preserves existing behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Write tests that invoke `llama-cli` with a simple prompt and verify that it hangs in conversation mode. Run these tests on the UNFIXED code to observe failures and understand the root cause. Use a short timeout (10 seconds) to make the test fail quickly.

**Test Cases**:
1. **Basic Hang Test**: Invoke `llama-cli` with prompt "Hello" and 10 tokens, verify it times out after 10 seconds (will fail on unfixed code - process hangs)
2. **Conversation Mode Detection**: Check if stdout contains `>` character after 5 seconds (will fail on unfixed code - `>` is present)
3. **Process Termination Test**: Try to kill the process with SIGTERM, verify it doesn't exit (will fail on unfixed code - process ignores SIGTERM)
4. **Timeout Command Test**: Use `timeout 10s llama-cli ...` and verify it kills the process (may fail on unfixed code - timeout doesn't work)

**Expected Counterexamples**:
- Process hangs indefinitely and prints `>` characters
- Process ignores SIGTERM and SIGINT signals
- Timeout command doesn't kill the process reliably
- Possible causes: conversation mode default, stdin handling bug, process group issue

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds (llama-cli execution), the fixed function produces the expected behavior (clean exit within timeout).

**Pseudocode:**
```
FOR ALL execution WHERE isBugCondition_potential(execution) DO
  result := execute_with_fixed_implementation(execution)
  ASSERT result.exited_cleanly == True
  ASSERT result.elapsed_time < timeout_threshold
  ASSERT result.stdout NOT CONTAINS ">"
  ASSERT result.generated_text.length > 0
END FOR
```

**Test Cases**:
1. **Clean Exit Test**: Verify process exits with return code 0 after generating tokens
2. **Timeout Compliance Test**: Verify process completes within expected timeout (max_tokens * 2 + 60s)
3. **No Conversation Mode Test**: Verify stdout doesn't contain `>` characters
4. **Output Generation Test**: Verify generated text is non-empty and reasonable
5. **Binary Fallback Test**: If `llama-cli` hangs, verify system tries `main` binary
6. **Process Group Kill Test**: Verify timeout kills entire process group, not just parent

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold (non-Android platforms, successful executions), the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL execution WHERE NOT isBugCondition(execution) DO
  ASSERT fixed_implementation(execution) = original_implementation(execution)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain
- It catches edge cases that manual unit tests might miss
- It provides strong guarantees that behavior is unchanged for all non-buggy inputs

**Test Plan**: Observe behavior on UNFIXED code first for successful executions, then write property-based tests capturing that behavior.

**Test Cases**:
1. **Streaming Preservation**: Verify character-by-character streaming produces same output format
2. **TTFT Measurement Preservation**: Verify timing measurements are accurate (within 10ms tolerance)
3. **Configuration Preservation**: Verify n_ctx, n_threads, n_batch are used correctly
4. **Tokenization Preservation**: Verify token count approximation is unchanged
5. **Error Handling Preservation**: Verify error messages and exceptions are unchanged
6. **Non-Android Platform Preservation**: Verify x86 and Jetson platforms use llama-cpp-python unchanged

### Unit Tests

- Test binary selection logic (prefer `main` over `llama-cli`)
- Test process group creation and killing
- Test timeout enforcement at Python level
- Test command construction with minimal flags
- Test error message generation on timeout
- Test output cleaning (remove `>` markers)

### Property-Based Tests

- Generate random prompts and token counts, verify clean exit within timeout
- Generate random binary configurations, verify correct binary is selected
- Generate random timeout scenarios, verify process is killed correctly
- Test that streaming output format is preserved across many executions

### Integration Tests

- Test full benchmark flow on Android with real model
- Test binary fallback when `llama-cli` hangs
- Test that TTFT measurement is accurate after fix
- Test that timeout errors are handled gracefully
- Test that process group killing works in real scenarios
