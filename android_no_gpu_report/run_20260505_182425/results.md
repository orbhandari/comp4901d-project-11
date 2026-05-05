# Benchmark Run Report: 20260505_182425

**Timestamp:** 2026-05-05T18:24:28.831514
**Duration:** 202.35 seconds

## Hardware Information

- **Platform:** android
- **CPU:** Unknown CPU (8 cores)
- **CPU Features:** 
- **RAM:** 11.17 GB total, 3.34 GB available
- **GPU:** Not available
- **Thermal Sensors:** Not available
- **Power Sensors:** Not available

## Software Versions

- **python:** 3.13.13

## Configuration

```json
{
  "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
  "models": {
    "Q2_K": "tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
    "Q4_0": "tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
  },
  "model_cache_dir": "~/storage/shared/models",
  "context_size": 2048,
  "batch_size": 256,
  "max_tokens": 100,
  "iterations": 5,
  "warmup_runs": 2,
  "enable_quantization_profiling": true,
  "enable_ablation_studies": true,
  "enable_batch_testing": false,
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
  "sleep_between_tests_s": 20,
  "thermal_stabilization_threshold_c": 70.0,
  "inference_timeout_s": 900,
  "output_dir": "~/storage/shared/benchmark_results",
  "save_formats": [
    "json",
    "csv",
    "markdown"
  ],
  "visualization_dpi": 150,
  "hf_token": null,
  "android_config": {
    "enable_ablation_studies": true,
    "use_llama_server_for_ablation": null,
    "llama_server_host": "127.0.0.1",
    "llama_server_port": 8080,
    "llama_server_path": null,
    "llama_server_timeout": 900,
    "cache_mode": "both",
    "enable_prompt_cache_by_default": false
  }
}
```

## Model Checksums (SHA256)

- **Q2_K:** `030a469a63576d59f601ef5608846b7718eaa884dd820e9aa7493efec1788afa`
- **Q4_0:** `da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556`

## Quantization Results

| Quantization | Load Time (s) | Peak RAM (MB) | TTFT (ms) | Prefill TPS | Decode TPS | Tokens (in/out) |
|--------------|---------------|---------------|-----------|-------------|------------|-----------------|
| Q2_K | 1.37 | 660.82 | 1403.31 | 17.81 | 490.89 | 25/137 |
| Q4_0 | 0.96 | 1341.74 | 984.40 | 25.40 | 479.00 | 25/137 |
| Q2_K | 1.35 | 663.82 | 1424.83 | 17.55 | 490.16 | 25/137 |
| Q4_0 | 0.95 | 1352.61 | 891.32 | 28.05 | 510.91 | 25/137 |
| Q2_K | 1.35 | 675.32 | 1479.81 | 16.89 | 493.74 | 25/137 |
| Q4_0 | 0.90 | 1356.25 | 920.49 | 27.16 | 489.50 | 25/137 |
| Q2_K | 1.36 | 687.34 | 1400.31 | 17.85 | 484.94 | 25/137 |
| Q4_0 | 0.91 | 1328.10 | 908.67 | 27.51 | 486.67 | 25/137 |
| Q2_K | 1.40 | 701.08 | 1361.93 | 18.36 | 618.13 | 25/137 |
| Q4_0 | 0.98 | 1328.26 | 914.86 | 27.33 | 496.96 | 25/137 |

## Ablation Study Results

### control

**Configuration:**
```json
{
  "backend_type": "llama-server",
  "cache_mode": "none",
  "prompt_cache_enabled": false,
  "cache_state": "disabled",
  "is_warm_run": false,
  "scenario_description": "No caching - true baseline measurement",
  "cache_activity_verified": true
}
```

**Metrics:**
- ttft_ms: 22551.09
- prefill_tps: 45.36
- decode_tps: 14.73
- memory_overhead_mb: 0.04
- peak_memory_mb: 177.51

### cold_cache

**Configuration:**
```json
{
  "backend_type": "llama-server",
  "cache_mode": "both",
  "prompt_cache_enabled": true,
  "cache_state": "cold_create",
  "is_warm_run": false,
  "scenario_description": "Cache enabled but empty (first request)",
  "cache_activity_verified": true
}
```

**Metrics:**
- ttft_ms: 22348.71
- prefill_tps: 45.77
- decode_tps: 14.51
- memory_overhead_mb: 25.50
- peak_memory_mb: 551.54
- **Improvement over baseline:** 0.90%

### warm_cache

**Configuration:**
```json
{
  "backend_type": "llama-server",
  "cache_mode": "both",
  "prompt_cache_enabled": true,
  "cache_state": "warm_reuse",
  "is_warm_run": true,
  "scenario_description": "Cache populated and reused (subsequent requests)",
  "cache_activity_verified": true
}
```

**Metrics:**
- ttft_ms: 11630.68
- prefill_tps: 87.96
- decode_tps: 14.64
- memory_overhead_mb: 25.72
- peak_memory_mb: 551.71
- **Improvement over baseline:** 48.43%

## Statistical Summaries

| Metric | Mean | Std Dev | 95% CI | Outliers |
|--------|------|---------|--------|----------|
| ttft_ms | 1414.04 | 43.19 | [1376.18, 1451.89] | 2 detected |
| prefill_tps | 17.69 | 0.54 | [17.22, 18.16] | 2 detected |
| decode_tps | 515.57 | 57.42 | [465.24, 565.90] | 1 detected |
| load_time_s | 1.37 | 0.02 | [1.35, 1.38] | None |
| peak_ram_mb | 677.68 | 16.75 | [662.99, 692.36] | None |
| ram_increase_mb | -430.63 | 511.76 | [-879.21, 17.95] | 1 detected |
| ttft_ms | 923.95 | 35.52 | [892.81, 955.08] | 1 detected |
| prefill_tps | 27.09 | 1.00 | [26.21, 27.97] | 2 detected |
| decode_tps | 492.61 | 12.08 | [482.02, 503.20] | None |
| load_time_s | 0.94 | 0.03 | [0.91, 0.97] | None |
| peak_ram_mb | 1341.39 | 13.19 | [1329.83, 1352.95] | None |
| ram_increase_mb | 663.74 | 27.78 | [639.39, 688.10] | None |

## Visualizations

- [quantization_comparison.png](/data/data/com.termux/files/home/storage/shared/benchmark_results/run_20260505_182425/visualizations/quantization_comparison.png)
- [memory_vs_speed_tradeoff.png](/data/data/com.termux/files/home/storage/shared/benchmark_results/run_20260505_182425/visualizations/memory_vs_speed_tradeoff.png)
- [ablation_comparison.png](/data/data/com.termux/files/home/storage/shared/benchmark_results/run_20260505_182425/visualizations/ablation_comparison.png)
