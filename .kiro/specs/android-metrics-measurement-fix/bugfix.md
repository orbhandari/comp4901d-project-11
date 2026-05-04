# Bugfix Requirements Document

## Introduction

The Android benchmark produces three categories of incorrect metrics when using native llama.cpp subprocess-based inference: (1) model load time is always 0.00s because measurement occurs during path validation rather than actual model loading, (2) peak RAM shows only 176 MB because only the Python process is measured while the native llama.cpp subprocess memory is excluded, and (3) decode throughput shows wildly inconsistent values with the first iteration producing impossibly high speeds (45778 t/s) while subsequent runs show ~2200 t/s. These measurement errors make the benchmark results unreliable for Android performance evaluation and prevent accurate comparison between quantization levels.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN profiling quantization on Android with native llama.cpp THEN the system measures load time around `backend.load_model_safe()` which only validates paths in `NativeLlamaCpp.__init__()`, resulting in 0.00s load time

1.2 WHEN profiling quantization on Android with native llama.cpp THEN the system measures peak RAM using `self.process.memory_info().rss` which only captures the Python process memory (176 MB), excluding the native llama.cpp subprocess memory

1.3 WHEN profiling quantization on Android with native llama.cpp THEN the system produces decode TPS of 45778.25 t/s on the first Q2_K iteration, which is impossibly high and inconsistent with subsequent runs (~2200 t/s)

1.4 WHEN calculating statistical summaries for decode TPS THEN the system produces huge standard deviation (25154.48) and negative lower confidence interval bound (-11732.58) due to the first-iteration outlier

1.5 WHEN calculating RAM increase (Peak RAM - Baseline RAM) THEN the system produces negative values (-767.03 MB for Q2_K, -492.98 MB for Q4_0) because the subprocess memory tracking fix is applied inconsistently between baseline and peak measurements

### Expected Behavior (Correct)

2.1 WHEN profiling quantization on Android with native llama.cpp THEN the system SHALL measure the actual model load time during the first inference call when the subprocess loads the model, producing load times of 1-5 seconds for TinyLlama models

2.2 WHEN profiling quantization on Android with native llama.cpp THEN the system SHALL measure peak RAM including both the Python process and the native llama.cpp subprocess memory, producing values of 400-600 MB for Q2_K and 600-800 MB for Q4_0 quantization

2.3 WHEN profiling quantization on Android with native llama.cpp THEN the system SHALL produce consistent decode TPS measurements across all iterations without 20x outliers, with values around 2200 t/s for typical Android hardware

2.4 WHEN calculating statistical summaries for decode TPS THEN the system SHALL produce reasonable standard deviations and confidence intervals with non-negative lower bounds

2.5 WHEN calculating RAM increase (Peak RAM - Baseline RAM) THEN the system SHALL produce positive values representing the actual memory increase from model loading, with values around 450 MB for Q2_K and 1100 MB for Q4_0

### Unchanged Behavior (Regression Prevention)

3.1 WHEN profiling quantization on non-Android platforms using llama-cpp-python THEN the system SHALL CONTINUE TO measure load time around `backend.load_model_safe()` as the model is loaded during that call

3.2 WHEN profiling quantization on non-Android platforms using llama-cpp-python THEN the system SHALL CONTINUE TO measure peak RAM using `self.process.memory_info().rss` as the model runs in-process

3.3 WHEN profiling quantization on any platform THEN the system SHALL CONTINUE TO perform warmup inference before measurement to stabilize timing

3.4 WHEN profiling quantization on any platform THEN the system SHALL CONTINUE TO track memory during token generation to capture peak memory usage

3.5 WHEN profiling quantization on any platform THEN the system SHALL CONTINUE TO calculate TTFT, prefill TPS, and other metrics using the existing methodology
