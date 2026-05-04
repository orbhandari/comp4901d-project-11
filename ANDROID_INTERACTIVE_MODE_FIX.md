# Android Interactive Mode Fix

## Problem
llama-cli was entering interactive prompt mode and waiting for user input instead of automatically completing generation.

## Solution
Added three critical fixes to `llm_benchmark/inference/native_llama.py`:

### 1. Non-Interactive Mode Flag (`-e`)
```python
"-e",  # Process prompt and exit (non-interactive)
```
This tells llama-cli to exit after generation instead of waiting for more input.

### 2. Disable GPU Offloading (`-ngl 0`)
```python
"-ngl", "0",  # Disable GPU offloading (can cause interactive issues)
```
Some systems enter interactive mode when GPU is enabled.

### 3. Close stdin (`stdin=subprocess.DEVNULL`)
```python
process = subprocess.Popen(
    cmd,
    stdin=subprocess.DEVNULL,  # Don't read from stdin (prevents interactive mode)
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)
```
This prevents llama-cli from trying to read user input.

## Testing

### Quick Test
```bash
cd ~/comp4901d-project-11
python test_llama_cli.py
```

This should complete all three tests without entering interactive mode.

### Manual Test
```bash
~/llama.cpp/build/bin/llama-cli \
  -m ~/storage/shared/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf \
  -n 20 \
  -p "Hello, my name is" \
  -e \
  --no-display-prompt \
  --log-disable
```

Expected: Generates 20 tokens and exits automatically.

### Full Benchmark
```bash
cd ~/comp4901d-project-11
python -m llm_benchmark --config configs/android_config.json
```

Expected: Completes without entering interactive mode.

## What Changed

**File: `llm_benchmark/inference/native_llama.py`**
- Added `-e` flag to command
- Added `-ngl 0` flag to command
- Added `stdin=subprocess.DEVNULL` to Popen call

**File: `test_llama_cli.py`**
- Added `-e` flag to all test commands

## If Still Having Issues

1. **Check llama-cli help:**
   ```bash
   ~/llama.cpp/build/bin/llama-cli --help | grep -i interactive
   ```

2. **Check llama-cli version:**
   ```bash
   ~/llama.cpp/build/bin/llama-cli --version
   ```

3. **Try alternative flags:**
   - `--one-shot` (if available)
   - `--run-once` (if available)
   - `-ins` with negation (if available)

4. **Check if `-e` flag exists:**
   ```bash
   ~/llama.cpp/build/bin/llama-cli --help | grep -E "^\s+-e"
   ```

If the `-e` flag doesn't exist in your version, we may need to use a different approach (like piping input or using a different binary).
