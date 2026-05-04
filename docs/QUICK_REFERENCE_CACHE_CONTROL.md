# Quick Reference: Cache Control for Ablation Studies

## TL;DR

**Problem:** llama-cli cannot disable KV cache → no true "no cache" baseline

**Solution:** Use llama-server with proper flags

## Commands

### ✅ llama-server (Recommended)

```bash
# Disable ALL caching
llama-server --model model.gguf --cache-ram 0 --no-cache-prompt --port 8080

# API request
curl http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Your prompt",
    "n_predict": 100,
    "cache_prompt": false
  }'
```

### ⚠️ llama-cli (Limited - KV cache always on)

```bash
# "Control" run (KV cache still active!)
llama-cli --model model.gguf --prompt "..." --n-predict 100

# Warm cache run
llama-cli --model model.gguf --prompt "..." --n-predict 100 \
  --prompt-cache cache.bin
```

### ✅ llama-cpp-python

```python
from llama_cpp import Llama

# No cache
llm = Llama(model_path="model.gguf", cache=False)

# With cache
llm = Llama(model_path="model.gguf", cache=True)
```

## Cache Prefix Used in Tests

```python
CACHE_PREFIX = """You are a helpful AI assistant tasked with explaining complex 
technical concepts. Please provide a comprehensive explanation of how large 
language models work, covering the following topics:

1. The transformer architecture and attention mechanisms
2. How models are trained on large text corpora
3. The role of tokenization in processing text
4. How inference works when generating responses"""
```

**Length:** ~100 tokens  
**Location:** `tests/fixtures/test_prompts.py`  
**Configurable:** 100, 500, 1000 tokens

## Comparison Table

| Backend | KV Cache Control | Prompt Cache Control | True Baseline |
|---------|------------------|---------------------|---------------|
| llama-server | ✅ `--cache-ram 0` | ✅ `--no-cache-prompt` | ✅ Yes |
| llama-cli | ❌ Always on | ✅ Omit `--prompt-cache` | ❌ No |
| llama-cpp-python | ✅ `cache=False` | ✅ API control | ✅ Yes |

## Common Mistakes

### ❌ Mistake 1: Forgetting cache_prompt in API

```json
{
  "prompt": "...",
  "n_predict": 100
  // Missing: "cache_prompt": false
  // Result: Cache is used (defaults to true!)
}
```

### ✅ Correct:

```json
{
  "prompt": "...",
  "n_predict": 100,
  "cache_prompt": false  // Explicitly disable
}
```

### ❌ Mistake 2: Using llama-cli for ablation studies

```bash
# This does NOT give you a true baseline!
llama-cli --model model.gguf --prompt "..."
# KV cache is still active
```

### ✅ Correct:

```bash
# Use llama-server instead
llama-server --model model.gguf --cache-ram 0 --no-cache-prompt
```

## Verification

### Check if cache is disabled:

```bash
# Monitor memory during inference
watch -n 0.5 'ps aux | grep llama'

# No cache: memory stable
# With cache: memory increases
```

### Check server logs:

```bash
llama-server --model model.gguf --cache-ram 0 --no-cache-prompt --verbose

# Should NOT see:
# - "cache initialized"
# - "cache loaded"
```

## Files Updated

1. `llm_benchmark/profiler/ablation.py` - Added cache control detection and warnings
2. `llm_benchmark/inference/native_llama.py` - Added `no_cache_prompt` parameter
3. `configs/android_config_with_ablation.json` - Added cache control documentation
4. `docs/CACHE_CONTROL_STRATEGIES.md` - Comprehensive guide
5. `docs/ABLATION_CACHE_PREFIX.md` - Answers your specific question

## Need More Details?

See [CACHE_CONTROL_STRATEGIES.md](./CACHE_CONTROL_STRATEGIES.md) for the full guide.
