# Benchmark Run Report: 20260505_160613

**Timestamp:** 2026-05-05T16:06:32.689486
**Duration:** 857.43 seconds

## Hardware Information

- **Platform:** jetson_xavier_nx
- **CPU:** ARMv8 Processor rev 0 (v8l) (6 cores)
- **CPU Features:** 
- **RAM:** 14.54 GB total, 11.02 GB available
- **GPU:** Not available
- **Thermal Sensors:** Available
- **Power Sensors:** Available

## Software Versions

- **python:** 3.9.5
- **llama-cpp-python:** 0.3.22

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
  "hf_token": null
}
```

## Model Checksums (SHA256)

- **Q8_0:** `a4c9bb1dbaa372f6381a035fa5c02ef087aaa1ff1f843a56a22328114f03fc59`
- **Q4_0:** `da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556`
- **Q2_K:** `030a469a63576d59f601ef5608846b7718eaa884dd820e9aa7493efec1788afa`

## Quantization Results

| Quantization | Load Time (s) | Peak RAM (MB) | TTFT (ms) | Prefill TPS | Decode TPS | Tokens (in/out) |
|--------------|---------------|---------------|-----------|-------------|------------|-----------------|
| Q8_0 | 0.85 | 1639.65 | 147.86 | 128.50 | 5.14 | 19/59 |
| Q4_0 | 0.40 | 1131.58 | 150.78 | 126.01 | 4.67 | 19/17 |
| Q2_K | 0.37 | 986.33 | 180.25 | 105.41 | 4.18 | 19/51 |
| Q8_0 | 0.46 | 1638.14 | 138.20 | 137.48 | 4.99 | 19/59 |
| Q4_0 | 0.41 | 1136.58 | 150.34 | 126.38 | 4.62 | 19/17 |
| Q2_K | 0.36 | 991.16 | 177.38 | 107.11 | 4.16 | 19/51 |
| Q8_0 | 0.46 | 1642.97 | 141.91 | 133.88 | 5.08 | 19/59 |
| Q4_0 | 0.39 | 1136.59 | 159.07 | 119.45 | 4.61 | 19/17 |
| Q2_K | 0.37 | 991.39 | 178.20 | 106.62 | 4.19 | 19/51 |
| Q8_0 | 0.45 | 1643.20 | 140.42 | 135.31 | 5.17 | 19/59 |
| Q4_0 | 0.39 | 1136.81 | 167.79 | 113.24 | 3.70 | 19/17 |
| Q2_K | 0.39 | 991.39 | 220.39 | 86.21 | 4.05 | 19/51 |
| Q8_0 | 0.47 | 1643.20 | 139.52 | 136.18 | 5.09 | 19/59 |
| Q4_0 | 0.39 | 1136.81 | 156.55 | 121.37 | 4.63 | 19/17 |
| Q2_K | 0.36 | 991.39 | 179.52 | 105.84 | 4.22 | 19/51 |

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
- ttft_ms: 91293.04
- prefill_tps: 12.51
- decode_tps: 6.29
- memory_overhead_mb: 1158.89
- peak_memory_mb: 1718.17
- prompt_tokens: 1142.00
- output_tokens: 26.00

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
- ttft_ms: 91169.21
- prefill_tps: 12.53
- decode_tps: 6.33
- memory_overhead_mb: 1152.92
- peak_memory_mb: 1718.02
- prompt_tokens: 1142.00
- output_tokens: 26.00
- **Improvement over baseline:** 0.14%

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
- ttft_ms: 1258.65
- prefill_tps: 907.32
- decode_tps: 6.47
- cache_memory_overhead_mb: 94.16
- total_memory_overhead_mb: 1205.32
- peak_memory_mb: 1718.00
- prompt_tokens: 1142.00
- output_tokens: 51.00
- **Improvement over baseline:** 98.62%

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
- ttft_ms: 91568.45
- prefill_tps: 12.47
- decode_tps: 6.04
- memory_overhead_mb: 1153.18
- peak_memory_mb: 1718.07
- prompt_tokens: 1142.00
- output_tokens: 26.00
- **Improvement over baseline:** -0.30%

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
- ttft_ms: 1243.60
- prefill_tps: 918.30
- decode_tps: 6.37
- cache_memory_overhead_mb: 94.03
- total_memory_overhead_mb: 1205.07
- peak_memory_mb: 1717.77
- prompt_tokens: 1142.00
- output_tokens: 51.00
- **Improvement over baseline:** 98.64%

## Statistical Summaries

| Metric | Mean | Std Dev | 95% CI | Outliers |
|--------|------|---------|--------|----------|
| ttft_ms | 141.58 | 3.76 | [138.29, 144.88] | 1 detected |
| prefill_tps | 134.27 | 3.48 | [131.22, 137.32] | 1 detected |
| decode_tps | 5.09 | 0.07 | [5.03, 5.15] | None |
| load_time_s | 0.54 | 0.17 | [0.38, 0.69] | 1 detected |
| peak_ram_mb | 1641.43 | 2.38 | [1639.35, 1643.52] | None |
| ram_increase_mb | 1177.89 | 2.23 | [1175.93, 1179.84] | 1 detected |
| ttft_ms | 156.91 | 7.14 | [150.65, 163.16] | None |
| prefill_tps | 121.29 | 5.39 | [116.56, 126.02] | None |
| decode_tps | 4.45 | 0.42 | [4.08, 4.81] | 2 detected |
| load_time_s | 0.40 | 0.01 | [0.39, 0.40] | None |
| peak_ram_mb | 1135.67 | 2.29 | [1133.67, 1137.68] | 1 detected |
| ram_increase_mb | 671.53 | 2.13 | [669.66, 673.39] | 1 detected |
| ttft_ms | 187.15 | 18.62 | [170.83, 203.47] | 1 detected |
| prefill_tps | 102.24 | 8.98 | [94.36, 110.11] | 1 detected |
| decode_tps | 4.16 | 0.07 | [4.10, 4.22] | 1 detected |
| load_time_s | 0.37 | 0.01 | [0.36, 0.38] | 1 detected |
| peak_ram_mb | 990.33 | 2.24 | [988.37, 992.29] | 1 detected |
| ram_increase_mb | 525.13 | 0.10 | [525.05, 525.22] | 1 detected |

## Visualizations

- [quantization_comparison.png](benchmark_results/run_20260505_160613/visualizations/quantization_comparison.png)
- [memory_vs_speed_tradeoff.png](benchmark_results/run_20260505_160613/visualizations/memory_vs_speed_tradeoff.png)
- [ablation_comparison.png](benchmark_results/run_20260505_160613/visualizations/ablation_comparison.png)
