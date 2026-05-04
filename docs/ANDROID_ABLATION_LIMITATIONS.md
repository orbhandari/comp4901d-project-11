# Android Ablation Engine Limitations

## Summary

**The AndroidAblationEngine does NOT measure pure RAM vs Disk caching effects.**

This is due to a fundamental limitation: **llama-cli cannot disable KV cache (RAM)**, which is always active.

## What It Actually Measures

### Test Scenarios

| Scenario | KV Cache (RAM) | Prompt Cache (Disk) | What It Represents |
|----------|----------------|---------------------|-------------------|
| **Control** | ✅ ACTIVE | ❌ None | Baseline with RAM cache only |
| **Cold** | ✅ ACTIVE | ✅ Creating | RAM + Disk (creating file) |
| **Warm** | ✅ ACTIVE | ✅ Loaded | RAM + Disk (loaded from file) |

### Comparisons

| Comparison | What It Shows | What It Does NOT Show |
|------------|---------------|----------------------|
| Control vs Cold | Overhead of **creating** disk cache on top of RAM cache | Pure disk cache effect |
| Control vs Warm | Benefit of **adding** disk cache to existing RAM cache | RAM vs Disk comparison |
| Cold vs Warm | Benefit of **loading** vs **creating** disk cache | Independent cache effects |

## What You CANNOT Determine

❌ **Pure RAM cache effect** - No true "no cache" baseline exists  
❌ **RAM vs Disk comparison** - Both are active simultaneously  
❌ **Independent cache overhead** - Cannot isolate each cache type  
❌ **Which cache type is more effective** - They're always combined  

## What You CAN Determine

✅ **Incremental benefit of disk caching** - When added to existing RAM cache  
✅ **Disk cache file size** - Storage overhead  
✅ **Disk I/O overhead** - Time to create/load cache files  
✅ **Combined cache effectiveness** - Total speedup with both caches  

## Why This Limitation Exists

### Technical Reason

```bash
# llama-cli has no flag to disable KV cache
llama-cli --model model.gguf --prompt "..." --n-predict 100
# KV cache is ALWAYS active internally

# Only prompt cache can be controlled
llama-cli --model model.gguf --prompt "..." --n-predict 100 \
  --prompt-cache cache.bin  # ← Only this is optional
```

The KV cache is hardcoded into llama-cli's inference loop and cannot be disabled via command-line flags.

### Code Evidence

From `android_ablation.py`:

```python
return AblationResult(
    scenario="control_no_prompt_cache",
    configuration={
        "prompt_cache_enabled": False,
        "kv_cache_enabled": True,  # ← ALWAYS TRUE!
        "cache_type": "ram_kv_only",
        "cache_state": "N/A"
    },
    # ...
)
```

The "control" run is labeled as having only RAM cache, but there's no comparison against a true "no cache" baseline.

## How to Interpret Results

### Example Results

```
Control (KV only):     TTFT = 150ms
Cold (KV + Disk):      TTFT = 160ms  (+10ms overhead from creating cache)
Warm (KV + Disk):      TTFT = 120ms  (-30ms benefit from loaded cache)
```

### Correct Interpretation

✅ "Adding disk cache provides 20% speedup over RAM cache alone"  
✅ "Creating disk cache adds 7% overhead"  
✅ "Loading disk cache is 25% faster than creating it"  

### Incorrect Interpretation

❌ "RAM cache provides X% speedup" (no baseline to compare against)  
❌ "Disk cache is faster than RAM cache" (both active simultaneously)  
❌ "RAM cache overhead is Y MB" (cannot isolate RAM cache)  

## Workarounds for Accurate Measurement

### Option 1: Use llama-server (Recommended)

```bash
# Start server with NO caching
llama-server --model model.gguf --cache-ram 0 --no-cache-prompt --port 8080

# Make API requests
curl http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Your prompt",
    "n_predict": 100,
    "cache_prompt": false
  }'
```

This gives you true control:
- `--cache-ram 0`: Disables RAM cache
- `--no-cache-prompt`: Disables prompt cache
- `cache_prompt: false`: Per-request cache control

### Option 2: Use llama-cpp-python (X86/Jetson)

```python
from llama_cpp import Llama

# True no-cache baseline
llm_no_cache = Llama(model_path="model.gguf", cache=False)

# With RAM cache
llm_with_cache = Llama(model_path="model.gguf", cache=True)
```

### Option 3: Modify llama.cpp Source

Edit the llama.cpp source code to add a flag for disabling KV cache, then recompile.

**Not recommended** - requires maintaining a fork.

## Recommendations

### For Research/Benchmarking

If you need accurate cache ablation studies:
1. **Use llama-server** instead of llama-cli
2. Document that you're using llama-server in your methodology
3. Test all cache combinations: none, RAM only, Disk only, Both

### For Production Use

If you're just optimizing inference on Android:
1. **Use AndroidAblationEngine as-is** - it's fine for practical optimization
2. Understand you're measuring "incremental benefit of disk cache"
3. Focus on the warm cache results (most relevant for production)

### For Documentation

When reporting results:
1. **Clearly state the limitation** - "Control run has RAM cache active"
2. **Use accurate terminology** - "Incremental benefit" not "pure effect"
3. **Provide context** - Explain what comparisons are valid

## Future Work

To properly support cache ablation on Android, we would need to:

1. **Implement NativeLlamaServer class**
   - Start/stop llama-server subprocess
   - HTTP API client for inference
   - Support for cache control flags

2. **Update AndroidBackend**
   - Add `use_llama_server` configuration option
   - Conditionally use llama-server vs llama-cli
   - Pass cache control flags appropriately

3. **Update AndroidAblationEngine**
   - Detect if llama-server is available
   - Use proper cache control when available
   - Fall back to current behavior with warnings

This would require significant implementation work but would enable true cache ablation studies on Android.

## Conclusion

The AndroidAblationEngine is **useful for practical optimization** but **not suitable for rigorous cache research** due to the inability to disable KV cache in llama-cli.

**Key Takeaway:** It measures the incremental benefit of adding disk-based prompt caching on top of the always-active RAM KV cache, not a pure comparison of cache types.

For accurate cache ablation studies, use llama-server or llama-cpp-python instead.
