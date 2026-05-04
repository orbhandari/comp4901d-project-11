# Android Streaming Fix - Inference Hanging Issue

## Problem

The benchmark was hanging at "performing measurement inference (50 tokens)" when using native llama.cpp on Android. The issue was in how we were reading output from the `llama-cli` subprocess.

## Root Cause

The previous implementation used `readline()` which blocks waiting for a newline character. However, `llama-cli` outputs tokens character-by-character or word-by-word without necessarily adding newlines, causing the read to block indefinitely.

## Solution

Updated `llm_benchmark/inference/native_llama.py` with:

### 1. Non-Blocking I/O
- Uses `fcntl` to set stdout to non-blocking mode on Unix-like systems (including Android)
- Prevents the read operation from blocking forever

### 2. Character-by-Character Reading
- Reads one character at a time instead of waiting for lines
- Properly handles the streaming output from llama-cli

### 3. Timeout Protection
- Adds a 5-minute timeout to detect if the process is truly hung
- Kills the process if no output is received within the timeout period

### 4. Better Error Handling
- Properly cleans up the subprocess if an error occurs
- Catches and handles IOError/OSError from non-blocking reads

## Changes Made

### File: `llm_benchmark/inference/native_llama.py`

**Added imports:**
```python
import select
import sys
import fcntl  # For non-blocking I/O
import os
```

**Updated `__call__()` method:**
- Set stdout to non-blocking mode using `fcntl`
- Read character-by-character with timeout protection
- Poll process status to detect completion
- Sleep briefly (0.01s) when no data is available
- Properly handle remaining output after process finishes

## Testing

### Quick Test
Run the diagnostic script to test llama-cli directly:

```bash
cd ~/comp4901d-project-11
python test_llama_cli.py
```

This will run three tests:
1. Simple generation with `--simple-io` flag
2. Generation without `--simple-io` flag
3. Streaming with Popen (character-by-character)

### Full Benchmark Test
Run the full benchmark:

```bash
cd ~/comp4901d-project-11
python -m llm_benchmark --config configs/android_config.json
```

## Expected Behavior

After the fix:
1. Model loads successfully
2. Warmup inference completes (5 tokens)
3. Measurement inference streams output character-by-character
4. Progress is visible in the logs
5. Benchmark completes without hanging

## Troubleshooting

### If still hanging:

1. **Check llama-cli output format:**
   ```bash
   ~/llama.cpp/build/bin/llama-cli \
     -m ~/storage/shared/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf \
     -n 20 \
     -p "Hello" \
     --no-display-prompt \
     --log-disable
   ```
   
   Observe how the output appears (character-by-character, word-by-word, or all at once).

2. **Check for stderr output:**
   The `--log-disable` flag should suppress llama.cpp's logging, but some versions might still output to stderr. Check if stderr is blocking.

3. **Try with --simple-io:**
   Add `--simple-io` flag to the command in `native_llama.py` if the output format is causing issues.

4. **Increase timeout:**
   If generation is just slow, increase the `timeout_seconds` value in `native_llama.py` (currently 300 seconds = 5 minutes).

5. **Check for interactive prompts:**
   Some llama-cli versions might wait for user input. Try adding `-i` or `--interactive` flags (or their negation).

## Alternative Approaches

If the current fix doesn't work, we can try:

1. **Use `--simple-io` flag:**
   This makes llama-cli output in a simpler format that might be easier to parse.

2. **Use `communicate()` instead of streaming:**
   Wait for the entire output to complete, then parse it. This loses streaming capability but ensures completion.

3. **Use a temporary file:**
   Have llama-cli write to a file, then read the file. This is slower but more reliable.

4. **Use `select.select()` for I/O multiplexing:**
   More sophisticated approach to handle both stdout and stderr without blocking.

## Files Modified

- `llm_benchmark/inference/native_llama.py` - Updated streaming logic with non-blocking I/O

## Files Created

- `test_llama_cli.py` - Diagnostic script to test llama-cli behavior
- `ANDROID_STREAMING_FIX.md` - This documentation

## Next Steps

1. Pull the latest changes
2. Run `test_llama_cli.py` to verify llama-cli works
3. Run the full benchmark with `python -m llm_benchmark --config configs/android_config.json`
4. Report results (success or specific error messages)

## Technical Details

### Non-Blocking I/O on Unix

```python
import fcntl
import os

# Get file descriptor
fd = process.stdout.fileno()

# Get current flags
flags = fcntl.fcntl(fd, fcntl.F_GETFL)

# Add O_NONBLOCK flag
fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
```

This makes `read()` return immediately with whatever data is available, or raise `IOError`/`OSError` if no data is available, instead of blocking.

### Read Loop Logic

```python
while True:
    # Check if process finished
    if process.poll() is not None:
        # Read remaining output and break
        break
    
    # Check timeout
    if time.time() - last_output_time > timeout:
        raise TimeoutError()
    
    try:
        # Try to read (non-blocking)
        char = process.stdout.read(1)
        if char:
            # Got data, yield it
            yield char
        else:
            # No data yet, sleep briefly
            time.sleep(0.01)
    except (IOError, OSError):
        # No data available (expected with non-blocking I/O)
        time.sleep(0.01)
```

This ensures we:
- Don't block forever waiting for data
- Detect when the process finishes
- Handle timeouts gracefully
- Sleep briefly when no data is available (prevents busy-waiting)
