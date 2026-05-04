# Android llama-cli Conversation Mode Issue

## Problem
llama-cli keeps printing `>` characters infinitely and never exits. This indicates it's stuck in **conversation mode** (interactive chat mode).

## Root Cause
Your version of llama-cli is defaulting to conversation/chat mode, which:
1. Waits for user input after each response
2. Prints `>` as a prompt marker
3. Never exits on its own

## Diagnosis Steps

**FIRST: Run the diagnostic script to understand your llama-cli version:**

```bash
cd ~/comp4901d-project-11
chmod +x diagnose_llama_cli.sh
bash diagnose_llama_cli.sh
```

This will:
- Check llama-cli version
- List available flags
- Check for alternative binaries (`main`, `llama-simple`)
- Test if llama-cli exits properly

## Solutions (Try in Order)

### Solution 1: Use the `main` Binary (Recommended)

The older `main` binary (from older llama.cpp versions) doesn't have conversation mode and works more reliably for non-interactive use.

**Check if you have it:**
```bash
ls -lh ~/llama.cpp/build/bin/main
```

**If it exists**, the code will automatically use it! Just run the benchmark again:
```bash
python -m llm_benchmark --config configs/android_config.json
```

The updated code now tries binaries in this order:
1. `llama-cli` (if it works)
2. `main` (fallback, more reliable)
3. `llama-simple` (if available)

### Solution 2: Find the Right Flag for llama-cli

**Check available flags:**
```bash
~/llama.cpp/build/bin/llama-cli --help | grep -i conversation
~/llama.cpp/build/bin/llama-cli --help | grep -i interactive
~/llama.cpp/build/bin/llama-cli --help | grep -i chat
```

**Look for flags like:**
- `--no-cnv` or `--no-conversation` - Disable conversation mode
- `--no-interactive` - Disable interactive mode
- `--one-shot` - Single generation and exit
- `-cnv` (with value 0 or false) - Disable conversation

**If you find the right flag**, let me know and I'll update the code.

### Solution 3: Rebuild llama.cpp Without Conversation Mode

Some llama.cpp builds default to conversation mode. Try rebuilding:

```bash
cd ~/llama.cpp
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build --config Release -j4
```

This should build both `llama-cli` and `main` binaries.

### Solution 4: Use llama-server Instead

If nothing else works, we can use `llama-server` (HTTP API) instead:

```bash
# Check if llama-server exists
ls -lh ~/llama.cpp/build/bin/llama-server
```

If it exists, we can modify the code to use HTTP requests instead of subprocess calls.

## What the Code Does Now

### 1. Tries Multiple Binaries
```python
# Priority order:
1. llama-cli (if it works)
2. main (fallback - more reliable)
3. llama-simple (if available)
```

### 2. Sends EOF to stdin
```python
process.stdin.close()  # Send EOF to prevent waiting for input
```

### 3. Uses Timeout
```python
stdout, stderr = process.communicate(timeout=300)  # Kill after 5 minutes
```

### 4. Adds --no-cnv Flag
```python
cmd.extend(["--no-cnv"])  # Try to disable conversation mode
```

## Testing

### Test 1: Check Which Binary Will Be Used
```bash
cd ~/comp4901d-project-11
python -c "
from llm_benchmark.inference.native_llama import NativeLlamaCpp
model = '~/storage/shared/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf'
llm = NativeLlamaCpp(model)
print(f'Using binary: {llm.llama_cli_path}')
print(f'Binary type: {llm.binary_type}')
"
```

### Test 2: Run Diagnostic
```bash
bash diagnose_llama_cli.sh
```

### Test 3: Manual Test with Timeout
```bash
# This should kill after 5 seconds if it hangs
timeout 5s ~/llama.cpp/build/bin/llama-cli \
  -m ~/storage/shared/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf \
  -n 10 \
  -p "Hello" \
  --log-disable \
  -ngl 0 \
  --no-cnv
```

**Expected:** Either generates text and exits, OR times out after 5 seconds.

### Test 4: Try the `main` Binary
```bash
# If main binary exists, test it
timeout 5s ~/llama.cpp/build/bin/main \
  -m ~/storage/shared/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf \
  -n 10 \
  -p "Hello" \
  --log-disable \
  -ngl 0
```

**Expected:** Should generate text and exit cleanly.

## Next Steps

1. **Run the diagnostic script:**
   ```bash
   bash diagnose_llama_cli.sh
   ```

2. **Share the output** so I can see:
   - Which binaries you have
   - What flags are available
   - Whether `main` binary exists

3. **Try the benchmark again:**
   ```bash
   python -m llm_benchmark --config configs/android_config.json
   ```
   
   The code will now automatically try the `main` binary if `llama-cli` doesn't work.

## Expected Behavior

✅ Code detects available binaries  
✅ Uses `main` binary if available (more reliable)  
✅ Sends EOF to prevent interactive mode  
✅ Times out after 5 minutes if stuck  
✅ Provides helpful error messages  

## If Still Stuck

If it still times out, we have two options:

1. **Use llama-server** (HTTP API approach)
2. **Use Python bindings** (compile llama-cpp-python from source for Android)

Let me know the results of the diagnostic script and we'll proceed from there!
