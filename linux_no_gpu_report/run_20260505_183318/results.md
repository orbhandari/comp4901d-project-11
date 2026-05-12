# Benchmark Run Report: 20260505_183318

**Timestamp:** 2026-05-05T18:33:22.324305
**Duration:** 111.67 seconds

## Hardware Information

- **Platform:** linux_x86
- **CPU:** AMD Ryzen AI 7 PRO 360 w/ Radeon 880M (8 cores)
- **CPU Features:** avx, avx2, avx512f, sse, sse2, ssse3, sse4_1, sse4_2
- **RAM:** 30.49 GB total, 23.02 GB available
- **GPU:** Not available
- **Thermal Sensors:** Available
- **Power Sensors:** Available

## Software Versions

- **python:** 3.14.4
- **llama-cpp-python:** 0.3.21

## Configuration

```json
{
  "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
  "models": {
    "Q8_0": "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
    "Q4_0": "tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
    "Q2_K": "tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
  },
  "model_cache_dir": "./models",
  "context_size": 2048,
  "batch_size": 512,
  "max_tokens": 100,
  "iterations": 5,
  "warmup_runs": 2,
  "enable_quantization_profiling": true,
  "enable_ablation_studies": true,
  "enable_batch_testing": true,
  "enable_thermal_monitoring": true,
  "kv_cache_types": [
    "ram",
    "disk"
  ],
  "prompt_cache_prefix_lengths": [
    100,
    500,
    1000
  ],
  "batch_sizes": [
    1,
    2,
    4,
    8,
    16
  ],
  "sleep_between_tests_s": 5,
  "thermal_stabilization_threshold_c": 70.0,
  "inference_timeout_s": 300,
  "output_dir": "./benchmark_results",
  "save_formats": [
    "json",
    "csv",
    "markdown",
    "html"
  ],
  "visualization_dpi": 300,
  "hf_token": null,
  "android_config": null
}
```

## Model Checksums (SHA256)

- **Q8_0:** `a4c9bb1dbaa372f6381a035fa5c02ef087aaa1ff1f843a56a22328114f03fc59`
- **Q4_0:** `da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556`
- **Q2_K:** `030a469a63576d59f601ef5608846b7718eaa884dd820e9aa7493efec1788afa`

## Quantization Results

| Quantization | Load Time (s) | Peak RAM (MB) | TTFT (ms) | Prefill TPS | Decode TPS | Tokens (in/out) |
|--------------|---------------|---------------|-----------|-------------|------------|-----------------|
| Q8_0 | 0.06 | 1395.65 | 32.67 | 581.59 | 27.67 | 19/60 |
| Q4_0 | 0.13 | 1408.18 | 21.62 | 878.69 | 32.41 | 19/16 |
| Q2_K | 0.06 | 774.04 | 19.24 | 987.59 | 40.05 | 19/101 |
| Q8_0 | 0.06 | 1397.59 | 31.16 | 609.80 | 26.84 | 19/60 |
| Q4_0 | 0.13 | 1407.12 | 18.33 | 1036.38 | 39.13 | 19/16 |
| Q2_K | 0.13 | 774.45 | 19.43 | 977.95 | 38.37 | 19/101 |
| Q8_0 | 0.06 | 1397.77 | 32.09 | 592.14 | 27.42 | 19/60 |
| Q4_0 | 0.13 | 1409.12 | 16.84 | 1128.35 | 43.68 | 19/16 |
| Q2_K | 0.06 | 775.30 | 21.78 | 872.44 | 40.36 | 19/101 |
| Q8_0 | 0.06 | 1400.62 | 30.02 | 632.98 | 27.33 | 19/60 |
| Q4_0 | 0.13 | 1411.98 | 14.97 | 1269.14 | 43.66 | 19/16 |
| Q2_K | 0.06 | 775.32 | 19.45 | 976.67 | 40.31 | 19/101 |
| Q8_0 | 0.06 | 1400.62 | 33.38 | 569.16 | 25.58 | 19/60 |
| Q4_0 | 0.13 | 1411.98 | 16.29 | 1166.54 | 42.95 | 19/16 |
| Q2_K | 0.06 | 779.32 | 17.60 | 1079.31 | 39.65 | 19/101 |

## Ablation Study Results

### control_no_cache

**Configuration:**
```json
{
  "cache_enabled": false,
  "cache_type": null,
  "cache_state": "N/A",
  "true_no_cache_baseline": true
}
```

**Metrics:**
- ttft_ms: 7722.32
- prefill_tps: 115.64
- decode_tps: 28.05
- memory_overhead_mb: 1160.93
- peak_memory_mb: 1476.15
- prompt_tokens: 893.00
- output_tokens: 40.00

### cold_ram_cache

**Configuration:**
```json
{
  "cache_enabled": true,
  "cache_type": "ram",
  "cache_state": "empty"
}
```

**Metrics:**
- ttft_ms: 7469.44
- prefill_tps: 119.55
- decode_tps: 28.71
- memory_overhead_mb: 1141.04
- peak_memory_mb: 1482.15
- prompt_tokens: 893.00
- output_tokens: 40.00
- **Improvement over baseline:** 3.27%

### warm_ram_cache

**Configuration:**
```json
{
  "cache_enabled": true,
  "cache_type": "ram",
  "cache_state": "populated"
}
```

**Metrics:**
- ttft_ms: 146.48
- prefill_tps: 6096.45
- decode_tps: 28.39
- cache_memory_overhead_mb: 91.38
- total_memory_overhead_mb: 1252.36
- peak_memory_mb: 1470.93
- prompt_tokens: 893.00
- output_tokens: 47.00
- **Improvement over baseline:** 98.10%

### cold_disk_cache

**Configuration:**
```json
{
  "cache_enabled": true,
  "cache_type": "disk",
  "cache_state": "empty"
}
```

**Metrics:**
- ttft_ms: 7201.51
- prefill_tps: 124.00
- decode_tps: 28.27
- memory_overhead_mb: 1144.05
- peak_memory_mb: 1481.95
- prompt_tokens: 893.00
- output_tokens: 40.00
- **Improvement over baseline:** 6.74%

### warm_disk_cache

**Configuration:**
```json
{
  "cache_enabled": true,
  "cache_type": "disk",
  "cache_state": "populated"
}
```

**Metrics:**
- ttft_ms: 280.66
- prefill_tps: 3181.82
- decode_tps: 28.13
- cache_memory_overhead_mb: 91.38
- total_memory_overhead_mb: 1253.86
- peak_memory_mb: 1472.44
- prompt_tokens: 893.00
- output_tokens: 47.00
- **Improvement over baseline:** 96.37%

## Statistical Summaries

| Metric | Mean | Std Dev | 95% CI | Outliers |
|--------|------|---------|--------|----------|
| ttft_ms | 31.86 | 1.31 | [30.71, 33.01] | None |
| prefill_tps | 597.13 | 24.97 | [575.24, 619.02] | None |
| decode_tps | 26.97 | 0.83 | [26.24, 27.70] | 1 detected |
| load_time_s | 0.06 | 0.00 | [0.06, 0.06] | None |
| peak_ram_mb | 1398.45 | 2.15 | [1396.57, 1400.33] | None |
| ram_increase_mb | 1182.15 | 7.08 | [1175.94, 1188.36] | 1 detected |
| ttft_ms | 17.61 | 2.54 | [15.38, 19.84] | 1 detected |
| prefill_tps | 1095.82 | 147.29 | [966.71, 1224.93] | None |
| decode_tps | 40.37 | 4.83 | [36.13, 44.60] | None |
| load_time_s | 0.13 | 0.00 | [0.13, 0.13] | None |
| peak_ram_mb | 1409.68 | 2.22 | [1407.73, 1411.62] | None |
| ram_increase_mb | 1190.56 | 1.81 | [1188.98, 1192.14] | 2 detected |
| ttft_ms | 19.50 | 1.49 | [18.19, 20.81] | 2 detected |
| prefill_tps | 978.79 | 73.31 | [914.53, 1043.05] | 2 detected |
| decode_tps | 39.75 | 0.82 | [39.03, 40.47] | 1 detected |
| load_time_s | 0.07 | 0.03 | [0.05, 0.10] | 1 detected |
| peak_ram_mb | 775.69 | 2.11 | [773.84, 777.53] | 1 detected |
| ram_increase_mb | 555.90 | 1.47 | [554.62, 557.19] | 1 detected |

## Visualizations

- [quantization_comparison.png](benchmark_results/run_20260505_183318/visualizations/quantization_comparison.png)
- [memory_vs_speed_tradeoff.png](benchmark_results/run_20260505_183318/visualizations/memory_vs_speed_tradeoff.png)
- [ablation_comparison.png](benchmark_results/run_20260505_183318/visualizations/ablation_comparison.png)
