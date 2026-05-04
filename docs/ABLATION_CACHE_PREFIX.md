# Ablation Study Cache Prefix

## Question: What "prefix" is the ablation studies using to test the caching effect?

### Answer

The ablation studies use the following **shared prefix** for testing cache effectiveness:

```python
CACHE_PREFIX = """You are a helpful AI assistant tasked with explaining complex 
technical concepts. Please provide a comprehensive explanation of how large 
language models work, covering the following topics:

1. The transformer architecture and attention mechanisms
2. How models are trained on large text corpora
3. The role of tokenization in processing text
4. How inference works when generating responses"""
```

**Source:** `tests/fixtures/test_prompts.py`

### Prefix Characteristics

- **Length**: Approximately **100 tokens** (first portion of LONG_PROMPT)
- **Purpose**: Shared across multiple test prompts to measure cache reuse
- **Usage**: Combined with different suffixes to test cache hit rates

### Test Prompts Using This Prefix

1. **CACHE_TEST_PROMPT_1**: Prefix + focus on attention mechanism
2. **CACHE_TEST_PROMPT_2**: Prefix + focus on training process

### Configurable Prefix Lengths

The ablation engine supports testing with different prefix lengths:

```python
# From llm_benchmark/config.py
prompt_cache_prefix_lengths: List[int] = [100, 500, 1000]
```

This allows measuring how cache effectiveness varies with prefix size.

### How It's Used in Tests

```python
# First inference: Populate cache with prefix
first_prompt = CACHE_PREFIX + " [warmup]"
llm(first_prompt, max_tokens=10)

# Second inference: Reuse cached prefix with different suffix
second_prompt = CACHE_PREFIX + " Focus on attention mechanisms..."
llm(second_prompt, max_tokens=50)

# Measure: TTFT improvement from cache reuse
```

### Cache Hit Rate Calculation

```python
# Estimate cached tokens
total_tokens = metrics.prompt_tokens
estimated_prefix_tokens = len(CACHE_PREFIX) // 4  # ~100 tokens
estimated_cached_tokens = min(estimated_prefix_tokens, total_tokens)

# Calculate hit rate
cache_hit_rate_pct = (estimated_cached_tokens / total_tokens) * 100
```

## Important Limitation: Cannot Disable RAM Prompt Caching

### The Problem

**You are correct** - in ablation studies, we **cannot turn off RAM prompt caching** when using native llama-cli. This is a critical limitation that affects the accuracy of ablation studies.

### Why This Matters

For proper ablation studies, we need:
1. **Control run**: No caching (true baseline)
2. **Cold cache run**: Cache enabled but empty
3. **Warm cache run**: Cache populated and reused

However, **llama-cli always has KV cache active** - we cannot get a true "no cache" baseline.

### Solutions

#### ✅ Solution 1: Use llama-server (Recommended for Android)

```bash
# Start server with caching DISABLED
llama-server \
  --model model.gguf \
  --cache-ram 0 \
  --no-cache-prompt \
  --port 8080
```

**API Request:**
```python
import requests

response = requests.post(
    "http://localhost:8080/completion",
    json={
        "prompt": "Your prompt",
        "n_predict": 100,
        "cache_prompt": False  # MUST be False (defaults to true!)
    }
)
```

**Key Points:**
- `--cache-ram 0`: Disables RAM cache
- `--no-cache-prompt`: Disables prompt caching
- `cache_prompt: false`: Must be set in API requests (defaults to true)

#### ✅ Solution 2: Use llama-cpp-python (X86/Jetson)

```python
from llama_cpp import Llama

# Control run (no cache)
llm = Llama(model_path="model.gguf", cache=False)
```

#### ⚠️ Solution 3: llama-cli (Limited)

```bash
# Control run (KV cache still active!)
llama-cli --model model.gguf --prompt "..." --n-predict 100
# Do NOT use --prompt-cache or --path_session

# Warm cache run
llama-cli --model model.gguf --prompt "..." --n-predict 100 \
  --prompt-cache /path/to/cache.bin
```

**Limitation:** KV cache is always active - not a true "no cache" baseline

### What the Benchmark Does

The `AblationEngine` automatically detects the backend and warns users:

```python
# From llm_benchmark/profiler/ablation.py
if using_native:
    logger.warning("=" * 80)
    logger.warning("LIMITATION: Native llama.cpp ALWAYS has KV cache enabled (RAM)")
    logger.warning("This 'control' run is NOT a true no-cache baseline!")
    logger.warning("KV cache cannot be disabled via llama-cli flags.")
    logger.warning("")
    logger.warning("To get true no-cache baseline, you would need to:")
    logger.warning("  1. Use llama-server with --cache-ram 0 --no-cache-prompt")
    logger.warning("  2. Or modify llama.cpp source to disable KV cache")
    logger.warning("=" * 80)
```

### Updated Configuration

The `android_config_with_ablation.json` now includes:

```json
{
  "ablation_config": {
    "_IMPORTANT_CACHE_CONTROL": {
      "limitation": "llama-cli CANNOT disable KV cache (always active in RAM)",
      "workaround_1": "Use llama-server with --cache-ram 0 --no-cache-prompt flags",
      "workaround_2": "Use llama-cpp-python on supported platforms (can set cache=False)",
      "impact": "Control runs will have KV cache active, not a true 'no cache' baseline",
      "recommendation": "For accurate ablation studies, use llama-server instead of llama-cli"
    },
    
    "_llama_server_example": {
      "start_server": "llama-server --model model.gguf --cache-ram 0 --no-cache-prompt --port 8080",
      "api_request": {
        "prompt": "Your prompt here",
        "max_tokens": 100,
        "cache_prompt": false
      },
      "note": "cache_prompt must be false in API requests, defaults to true if omitted"
    }
  }
}
```

## Summary

### Cache Prefix
- **Text**: Technical explanation prompt about LLMs
- **Length**: ~100 tokens (configurable: 100, 500, 1000)
- **Location**: `tests/fixtures/test_prompts.py`

### Cache Control
- **llama-cli**: ❌ Cannot disable KV cache (always active)
- **llama-server**: ✅ Can disable with `--cache-ram 0 --no-cache-prompt`
- **llama-cpp-python**: ✅ Can disable with `cache=False`

### Recommendation
**For accurate ablation studies on Android, use llama-server instead of llama-cli.**

## See Also

- [CACHE_CONTROL_STRATEGIES.md](./CACHE_CONTROL_STRATEGIES.md) - Comprehensive guide
- `tests/fixtures/test_prompts.py` - Test prompt definitions
- `llm_benchmark/profiler/ablation.py` - Ablation engine implementation
- `configs/android_config_with_ablation.json` - Configuration with cache control notes
