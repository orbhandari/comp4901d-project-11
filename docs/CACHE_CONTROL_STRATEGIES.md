# Cache Control Strategies for Ablation Studies

## Overview

This document explains how to properly control caching behavior in llama.cpp for accurate ablation studies. The ability to disable caching varies significantly across different backends.

## The Problem

**In ablation studies, we need to measure the effect of caching by comparing:**
1. **Control run**: No caching (true baseline)
2. **Cold cache run**: Cache enabled but empty
3. **Warm cache run**: Cache populated and reused

However, **native llama-cli cannot disable KV cache** - it's always active in RAM. This means we cannot get a true "no cache" baseline using llama-cli alone.

## Cache Types in llama.cpp

### 1. KV Cache (RAM-based)
- Stores key-value tensors from attention mechanism
- Always active in llama-cli (cannot be disabled)
- Can be disabled in llama-cpp-python with `cache=False`
- Significantly speeds up inference by avoiding recomputation

### 2. Prompt Cache (Disk-based)
- Stores processed prompt state to disk
- Controlled via `--prompt-cache` flag in llama-cli
- Can be disabled by not providing the flag
- Useful for reusing common prompt prefixes across sessions

## Solutions by Backend

### Option 1: llama-server (✅ Recommended for Android)

**Advantages:**
- Full control over both KV cache and prompt cache
- Can truly disable caching for accurate baselines
- RESTful API for easy integration

**How to use:**

```bash
# Start server with caching DISABLED
llama-server \
  --model model.gguf \
  --cache-ram 0 \
  --no-cache-prompt \
  --port 8080 \
  --ctx-size 2048
```

**API Request (Python):**

```python
import requests

response = requests.post(
    "http://localhost:8080/completion",
    json={
        "prompt": "Your prompt here",
        "n_predict": 100,
        "cache_prompt": False,  # CRITICAL: Must be False
        "temperature": 0.7
    }
)
```

**Important Notes:**
- `cache_prompt` defaults to `true` if omitted - always set it explicitly
- Some versions may log "cache enabled" even with flags set (cosmetic bug)
- The flags effectively disable the functional impact

### Option 2: llama-cli (⚠️ Limited)

**Limitations:**
- KV cache is ALWAYS enabled (cannot be disabled)
- Can only control prompt cache (disk-based)
- Control runs will NOT be true "no cache" baselines

**How to use:**

```bash
# Control run (no prompt cache, but KV cache still active)
llama-cli \
  --model model.gguf \
  --prompt "Your prompt" \
  --n-predict 100 \
  --ctx-size 2048
  # Do NOT use --prompt-cache or --path_session

# Warm cache run (with prompt cache)
llama-cli \
  --model model.gguf \
  --prompt "Your prompt" \
  --n-predict 100 \
  --ctx-size 2048 \
  --prompt-cache /path/to/cache.bin
```

**What this means for ablation studies:**
- Your "control" run will have KV cache active
- You're only measuring the effect of prompt cache, not total cache effect
- Results will underestimate the true impact of caching

### Option 3: llama-cpp-python (✅ Best for X86/Jetson)

**Advantages:**
- Full control over KV cache via Python API
- Can set `cache=False` to disable KV cache
- True "no cache" baseline possible

**How to use:**

```python
from llama_cpp import Llama

# Control run (no cache)
llm_no_cache = Llama(
    model_path="model.gguf",
    n_ctx=2048,
    cache=False  # Disables KV cache
)

# Warm cache run (with cache)
llm_with_cache = Llama(
    model_path="model.gguf",
    n_ctx=2048,
    cache=True  # Enables KV cache (default)
)
```

## Implementation in This Benchmark

### Current Behavior

The `AblationEngine` class automatically detects which backend is being used:

1. **llama-cpp-python (X86, Jetson)**:
   - Sets `cache=False` for control runs
   - Provides true "no cache" baseline
   - Results are accurate

2. **Native llama-cli (Android)**:
   - Cannot disable KV cache
   - Logs warning about limitation
   - Marks results as `"true_no_cache_baseline": false`
   - Scenario name changes to `"control_kv_cache_only"`

### Example Output

```
==================================================================================
LIMITATION: Native llama.cpp ALWAYS has KV cache enabled (RAM)
This 'control' run is NOT a true no-cache baseline!
KV cache cannot be disabled via llama-cli flags.

To get true no-cache baseline, you would need to:
  1. Use llama-server with --cache-ram 0 --no-cache-prompt
  2. Or modify llama.cpp source to disable KV cache
==================================================================================
```

## Recommendations

### For Android Users

**Best approach:**
1. Use `llama-server` instead of `llama-cli` for ablation studies
2. Start server with `--cache-ram 0 --no-cache-prompt`
3. Make API requests with `cache_prompt: false`

**Alternative (if llama-server not available):**
1. Use `llama-cli` but understand the limitations
2. Interpret results as "prompt cache effect only"
3. Note that KV cache is always active in your reports

### For X86/Jetson Users

**Best approach:**
1. Use `llama-cpp-python` backend (default)
2. Let `AblationEngine` handle cache control automatically
3. Results will be accurate with true "no cache" baseline

## Verifying Cache Behavior

### Check if cache is truly disabled:

```bash
# Run with verbose logging
llama-server --model model.gguf --cache-ram 0 --no-cache-prompt --verbose

# Look for log messages about cache initialization
# Should NOT see: "cache initialized" or "cache loaded"
```

### Monitor memory usage:

```bash
# Watch memory during inference
watch -n 0.5 'ps aux | grep llama'

# With cache disabled: memory should be stable
# With cache enabled: memory increases as cache fills
```

## Common Issues

### Issue 1: "Cache enabled" log message with --no-cache-prompt

**Symptom:** Server logs "cache enabled" even with flags set

**Cause:** Cosmetic bug in some llama.cpp versions

**Solution:** Ignore the log message - the flags work functionally

### Issue 2: cache_prompt defaults to true

**Symptom:** Cache is used even though you didn't enable it

**Cause:** API defaults `cache_prompt` to `true` if omitted

**Solution:** Always explicitly set `cache_prompt: false` in requests

### Issue 3: Inconsistent results with llama-cli

**Symptom:** "Control" and "warm" runs show similar performance

**Cause:** KV cache is active in both runs

**Solution:** Switch to llama-server or llama-cpp-python

## References

1. [llama.cpp server documentation](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)
2. [llama.cpp CLI flags reference](https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md)
3. [llama-cpp-python API documentation](https://llama-cpp-python.readthedocs.io/)
4. Community discussions on cache control in llama.cpp issues

## Summary

| Backend | KV Cache Control | Prompt Cache Control | True Baseline | Recommended |
|---------|------------------|---------------------|---------------|-------------|
| llama-server | ✅ Yes (`--cache-ram 0`) | ✅ Yes (`--no-cache-prompt`) | ✅ Yes | ✅ Best for Android |
| llama-cli | ❌ No (always on) | ✅ Yes (omit `--prompt-cache`) | ❌ No | ⚠️ Limited |
| llama-cpp-python | ✅ Yes (`cache=False`) | ✅ Yes (via API) | ✅ Yes | ✅ Best for X86/Jetson |

**For accurate ablation studies, use llama-server or llama-cpp-python, not llama-cli.**
