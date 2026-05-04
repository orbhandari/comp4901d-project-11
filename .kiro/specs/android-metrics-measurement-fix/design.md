# Android Metrics Measurement Fix - Bugfix Design

## Overview

This bugfix addresses four systematic measurement errors in the Android benchmark when using native llama.cpp subprocess-based inference. The issues stem from the fundamental architectural difference between Android (subprocess-based) and other platforms (in-process): (1) load time is measured around path validation in `__init__()` rather than actual model loading during first inference, (2) RAM measurement only captures the Python process while excluding the native llama.cpp subprocess, (3) decode throughput shows a 20x outlier on the first iteration due to timing measurement issues, and (4) RAM increase calculation produces negative values (-767.03 MB for Q2_K, -492.98 MB for Q4_0) because the subprocess memory tracking fix is applied inconsistently between baseline and peak measurements. The fix strategy involves: measuring load time during first inference for Android, tracking subprocess memory using psutil's children() API consistently for both baseline and peak measurements, and either improving timing precision or adding outlier detection for decode TPS.

## Glossary

- **Bug_Condition (C)**: The condition that triggers measurement errors - when using NativeLlamaCpp (Android) instead of llama-cpp-python (other platforms)
- **Property (P)**: The desired behavior - accurate load time (1-5s), complete RAM measurement (400-800 MB), consistent decode TPS (~2200 t/s), and positive RAM increase (~450-1100 MB)
- **Preservation**: Existing measurement behavior for llama-cpp-python platforms that must remain unchanged
- **NativeLlamaCpp**: The Android-specific wrapper in `llm_benchmark/inference/native_llama.py` that uses subprocess to call native llama-cli binary
- **QuantizationProfiler**: The profiler in `llm_benchmark/profiler/quantization.py` that measures load time, RAM, and inference metrics
- **AndroidBackend**: The hardware backend in `llm_benchmark/hardware/hal.py` that provides Android-specific configuration and model loading
- **load_model_safe()**: The backend method that loads models - for Android, it instantiates NativeLlamaCpp which only validates paths in `__init__()`
- **profile_quantization()**: The profiler method that measures metrics around model loading and inference
- **Subprocess Memory**: The memory used by the native llama-cli process that runs separately from the Python process
- **Baseline RAM**: Memory measurement taken BEFORE model loading (Python process only, ~200 MB)
- **Peak RAM**: Memory measurement taken DURING inference (Python + subprocess, ~650-1305 MB)
- **RAM Increase**: Calculated as (Peak RAM - Baseline RAM), should always be positive for Android

## Bug Details

### Bug Condition

The bug manifests when profiling quantization on Android with native llama.cpp subprocess-based inference. The `QuantizationProfiler.profile_quantization()` method measures load time around `backend.load_model_safe()`, which for Android only instantiates `NativeLlamaCpp.__init__()` that validates paths without loading the model. The actual model loading happens during the first `llm()` call when the subprocess starts. Additionally, RAM measurement uses `self.process.memory_info().rss` which only captures the Python process memory, excluding the native llama-cli subprocess. Finally, decode TPS shows a 20x outlier on the first iteration, likely due to timing measurement issues or subprocess startup overhead.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type (backend: HardwareBackend, model_instance: Any)
  OUTPUT: boolean
  
  RETURN isinstance(backend, AndroidBackend)
         AND isinstance(model_instance, NativeLlamaCpp)
         AND model_instance uses subprocess for inference
END FUNCTION
```

### Examples

**Issue 1: Load Time Measurement**
- **Buggy Input**: Android platform with NativeLlamaCpp, measuring load time around `backend.load_model_safe()`
- **Current Behavior**: Load time = 0.00s (only path validation in `__init__()`)
- **Expected Behavior**: Load time = 1-5s (actual model loading during first inference)

**Issue 2: RAM Measurement**
- **Buggy Input**: Android platform with NativeLlamaCpp, measuring RAM with `self.process.memory_info().rss`
- **Current Behavior**: Peak RAM = 176 MB (only Python process)
- **Expected Behavior**: Peak RAM = 400-600 MB for Q2_K, 600-800 MB for Q4_0 (Python + subprocess)

**Issue 3: Decode TPS Outlier**
- **Buggy Input**: Android platform with NativeLlamaCpp, first Q2_K iteration
- **Current Behavior**: Decode TPS = 45778.25 t/s (impossibly high outlier)
- **Expected Behavior**: Decode TPS = ~2200 t/s (consistent with subsequent runs)

**Issue 4: Negative RAM Increase**
- **Buggy Input**: Android platform with NativeLlamaCpp, calculating RAM increase (Peak RAM - Baseline RAM)
- **Current Behavior**: RAM increase = -767.03 MB for Q2_K, -492.98 MB for Q4_0 (negative values)
- **Expected Behavior**: RAM increase = ~450 MB for Q2_K, ~1100 MB for Q4_0 (positive values representing actual memory increase from model loading)

**Edge Case: Non-Android Platforms**
- **Input**: X86Backend or JetsonBackend with llama-cpp-python
- **Expected Behavior**: All measurements continue to work correctly (preservation requirement)

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Load time measurement around `backend.load_model_safe()` must continue to work for llama-cpp-python platforms (X86Backend, JetsonBackend) where the model is loaded during that call
- RAM measurement using `self.process.memory_info().rss` must continue to work for llama-cpp-python platforms where the model runs in-process
- Warmup inference before measurement must continue to execute
- Memory tracking during token generation must continue to capture peak memory
- TTFT, prefill TPS, and other metrics calculation must continue using existing methodology

**Scope:**
All inputs that do NOT involve AndroidBackend with NativeLlamaCpp should be completely unaffected by this fix. This includes:
- X86Backend with llama-cpp-python (in-process inference)
- JetsonBackend with llama-cpp-python (in-process inference)
- Any platform using llama-cpp-python instead of NativeLlamaCpp

## Hypothesized Root Cause

Based on the bug description and code analysis, the root causes are:

### Issue 1: Load Time Measurement (0.00s)

**Root Cause**: Architectural mismatch between measurement point and actual loading

The `QuantizationProfiler.profile_quantization()` method measures load time around `backend.load_model_safe()`:

```python
load_start = time.perf_counter()
llm = self.backend.load_model_safe(model_path, **llama_config)
load_end = time.perf_counter()
load_time_s = load_end - load_start
```

For Android, `AndroidBackend.load_model_safe()` instantiates `NativeLlamaCpp`, which only validates paths and finds the binary in `__init__()`:

```python
def __init__(self, model_path: str, ...):
    self.model_path = Path(model_path).expanduser()
    # Find binary (fast)
    # Verify model exists (fast)
    # No actual model loading happens here
```

The actual model loading happens during the first `llm()` call when the subprocess starts and loads the model into memory. This is fundamentally different from llama-cpp-python where `Llama()` constructor loads the model immediately.

### Issue 2: RAM Measurement (176 MB)

**Root Cause**: Process memory measurement excludes subprocess

The profiler measures RAM using `self.process.memory_info().rss`:

```python
self.process = psutil.Process()  # Current Python process
peak_ram_mb = self.process.memory_info().rss / (1024 * 1024)
```

For Android, the native llama-cli runs as a separate subprocess spawned by `NativeLlamaCpp.__call__()`:

```python
process = subprocess.Popen(cmd, ...)
```

The subprocess loads the model and performs inference, consuming 400-800 MB of RAM, but this memory is not captured by `self.process.memory_info().rss` which only measures the parent Python process.

**Solution Approach**: Use `psutil.Process().children(recursive=True)` to find subprocess and sum memory across all processes.

### Issue 3: Decode TPS Outlier (45778.25 t/s)

**Root Cause**: Timing measurement issue or subprocess startup overhead

The first iteration shows decode TPS of 45778.25 t/s while subsequent runs show ~2200 t/s. Possible causes:

1. **Timing Precision Issue**: The decode duration calculation may be incorrect on the first run:
   ```python
   decode_duration = total_time_s - ttft_s
   decode_tps = (output_tokens - 1) / decode_duration
   ```
   If `decode_duration` is very small due to timing measurement error, `decode_tps` becomes very large.

2. **Subprocess Startup Overhead**: The first inference includes subprocess startup time in TTFT but not in decode time, causing timing skew.

3. **Character-by-Character Streaming**: `NativeLlamaCpp` simulates streaming by yielding character-by-character after the subprocess completes, which may cause timing measurement issues.

**Solution Approaches**:
- Improve timing measurement precision for first inference
- Add outlier detection to filter out impossible values (>10x median)
- Separate subprocess startup time from actual inference time

### Issue 4: Statistical Summary Errors

**Root Cause**: Outlier propagation into statistical calculations

The first-iteration outlier (45778.25 t/s) propagates into statistical summaries:
- Mean is inflated
- Standard deviation is huge (25154.48)
- Confidence interval has negative lower bound (-11732.58)

This is a consequence of Issue 3 - fixing the outlier will fix the statistical summaries.

### Issue 5: Negative RAM Increase Values

**Root Cause**: Inconsistent application of subprocess memory tracking between baseline and peak measurements

The RAM increase calculation produces negative values:
- Q2_K: ram_increase_mb = -767.03 MB (mean)
- Q4_0: ram_increase_mb = -492.98 MB (mean)

RAM increase should measure (Peak RAM during inference - Baseline RAM before loading), which should always be positive. The negative values indicate that the subprocess memory tracking fix (`_get_total_memory_mb()`) is being applied inconsistently:

**Possible Scenarios:**
1. **Baseline measured AFTER subprocess exists**: If baseline RAM is measured after the subprocess has already started, it includes subprocess memory (~650 MB for Q2_K), inflating the baseline. Then peak RAM is measured correctly (~653 MB), resulting in a small or negative increase.

2. **Peak measured BEFORE subprocess exists**: If peak RAM is measured before the subprocess starts or after it terminates, it only captures Python process memory (~200 MB), deflating the peak. Then baseline is measured correctly (~200 MB), resulting in a small or negative increase.

3. **Measurement timing reversed**: The baseline and peak measurements may be swapped or measured at the wrong points in the profiling lifecycle.

**Expected Behavior:**
- **Baseline RAM**: Python process only (~200 MB) - measured BEFORE model loading
- **Peak RAM**: Python + subprocess (~650 MB for Q2_K, ~1305 MB for Q4_0) - measured DURING inference
- **RAM Increase**: Should be POSITIVE (~450 MB for Q2_K, ~1100 MB for Q4_0)

**Solution Approach**: Ensure `_get_total_memory_mb()` is called consistently at the correct measurement points:
- Baseline: Measure BEFORE calling `backend.load_model_safe()` or first inference
- Peak: Measure DURING inference when subprocess is active
- Verify subprocess exists during peak measurement using `self.process.children()`

## Correctness Properties

Property 1: Bug Condition - Accurate Load Time Measurement

_For any_ model loading on Android with NativeLlamaCpp, the fixed profiler SHALL measure the actual model load time during the first inference call, producing load times of 1-5 seconds for TinyLlama models instead of 0.00s.

**Validates: Requirements 2.1**

Property 2: Bug Condition - Complete RAM Measurement

_For any_ inference on Android with NativeLlamaCpp, the fixed profiler SHALL measure peak RAM including both the Python process and the native llama-cli subprocess memory, producing values of 400-600 MB for Q2_K and 600-800 MB for Q4_0 quantization instead of 176 MB.

**Validates: Requirements 2.2**

Property 3: Bug Condition - Consistent Decode TPS

_For any_ inference on Android with NativeLlamaCpp, the fixed profiler SHALL produce consistent decode TPS measurements across all iterations without 20x outliers, with values around 2200 t/s for typical Android hardware instead of 45778.25 t/s on first iteration.

**Validates: Requirements 2.3, 2.4**

Property 4: Bug Condition - Positive RAM Increase

_For any_ inference on Android with NativeLlamaCpp, the fixed profiler SHALL calculate RAM increase as (Peak RAM - Baseline RAM) producing positive values representing the actual memory increase from model loading, with values around 450 MB for Q2_K and 1100 MB for Q4_0 instead of negative values.

**Validates: Requirements 2.5**

Property 5: Preservation - Non-Android Load Time Measurement

_For any_ model loading on non-Android platforms (X86Backend, JetsonBackend) using llama-cpp-python, the fixed profiler SHALL continue to measure load time around `backend.load_model_safe()` exactly as before, preserving existing behavior.

**Validates: Requirements 3.1**

Property 6: Preservation - Non-Android RAM Measurement

_For any_ inference on non-Android platforms using llama-cpp-python, the fixed profiler SHALL continue to measure peak RAM using `self.process.memory_info().rss` exactly as before, preserving existing behavior.

**Validates: Requirements 3.2**

Property 7: Preservation - Warmup and Other Metrics

_For any_ inference on any platform, the fixed profiler SHALL continue to perform warmup inference, track memory during token generation, and calculate TTFT, prefill TPS, and other metrics using the existing methodology, preserving all existing behavior.

**Validates: Requirements 3.3, 3.4, 3.5**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct, we need to modify `QuantizationProfiler.profile_quantization()` in `llm_benchmark/profiler/quantization.py` to handle Android-specific measurement requirements.

**File**: `llm_benchmark/profiler/quantization.py`

**Function**: `profile_quantization()`

**Specific Changes**:

#### 1. **Detect Android Platform**: Add platform detection to identify when using NativeLlamaCpp
   ```python
   from llm_benchmark.inference.native_llama import NativeLlamaCpp
   is_android = isinstance(llm, NativeLlamaCpp)
   ```

#### 2. **Fix Load Time Measurement**: For Android, measure load time during first inference instead of around `load_model_safe()`
   ```python
   if is_android:
       # For Android, model loads during first inference
       load_start = time.perf_counter()
       # Perform first inference (warmup)
       _ = llm(prompt, max_tokens=warmup_tokens, stream=False)
       load_end = time.perf_counter()
       load_time_s = load_end - load_start
   else:
       # For other platforms, model loads during load_model_safe()
       load_start = time.perf_counter()
       llm = self.backend.load_model_safe(model_path, **llama_config)
       load_end = time.perf_counter()
       load_time_s = load_end - load_start
   ```

#### 3. **Fix RAM Measurement**: For Android, track subprocess memory in addition to Python process memory
   ```python
   def _get_total_memory_mb(self) -> float:
       """Get total memory including subprocesses (for Android)."""
       total_rss = self.process.memory_info().rss
       
       # Add subprocess memory (for Android native llama.cpp)
       for child in self.process.children(recursive=True):
           try:
               total_rss += child.memory_info().rss
           except (psutil.NoSuchProcess, psutil.AccessDenied):
               pass
       
       return total_rss / (1024 * 1024)
   
   # Use in profiling
   peak_ram_mb = self._get_total_memory_mb()
   ```

#### 4. **Fix Decode TPS Outlier**: Add outlier detection or improve timing measurement
   
   **Option A: Outlier Detection** (simpler, more robust)
   ```python
   def _is_outlier(self, value: float, values: List[float], threshold: float = 10.0) -> bool:
       """Detect if value is an outlier (>threshold * median)."""
       if len(values) < 2:
           return False
       median = statistics.median(values)
       return value > threshold * median
   
   # In profile_all(), track decode_tps values and filter outliers
   decode_tps_values = [result.decode_tps for result in results]
   for result in results:
       if self._is_outlier(result.decode_tps, decode_tps_values):
           logger.warning(f"Outlier detected: {result.decode_tps} t/s (median: {statistics.median(decode_tps_values)})")
           # Mark as outlier or recalculate
   ```
   
   **Option B: Improved Timing** (more complex, addresses root cause)
   ```python
   if is_android:
       # For Android, separate subprocess startup from inference timing
       # First call includes startup overhead
       _ = llm(prompt, max_tokens=1, stream=False)  # Startup
       
       # Second call measures actual inference
       start_time = time.perf_counter()
       stream = llm(prompt, max_tokens=max_tokens, stream=True)
       # ... rest of timing logic
   ```

#### 5. **Fix RAM Increase Calculation**: Ensure consistent measurement timing for baseline and peak RAM
   ```python
   # Measure baseline RAM BEFORE model loading/first inference
   baseline_ram_mb = self._get_total_memory_mb()
   
   # Load model or perform first inference (for Android)
   if is_android:
       # First inference loads the model and starts subprocess
       _ = llm(prompt, max_tokens=warmup_tokens, stream=False)
   else:
       llm = self.backend.load_model_safe(model_path, **llama_config)
   
   # Measure peak RAM DURING inference when subprocess is active
   # Track memory during token generation
   peak_ram_mb = max(self._get_total_memory_mb(), baseline_ram_mb)
   
   # Calculate RAM increase (should always be positive)
   ram_increase_mb = peak_ram_mb - baseline_ram_mb
   
   # Validate: RAM increase should be positive for Android
   if is_android and ram_increase_mb < 0:
       logger.warning(f"Negative RAM increase detected: {ram_increase_mb} MB")
   ```

#### 6. **Refactor Measurement Logic**: Extract platform-specific measurement into helper methods
   ```python
   def _measure_load_time(self, llm, model_path, llama_config, prompt, warmup_tokens):
       """Measure load time (platform-specific)."""
       if isinstance(llm, NativeLlamaCpp):
           return self._measure_load_time_android(llm, prompt, warmup_tokens)
       else:
           return self._measure_load_time_standard(model_path, llama_config)
   
   def _measure_peak_memory(self):
       """Measure peak memory including subprocesses."""
       return self._get_total_memory_mb()
   ```

### Implementation Strategy

**Phase 1: Fix Load Time and RAM Measurement**
1. Add `_get_total_memory_mb()` helper method to track subprocess memory
2. Add platform detection using `isinstance(llm, NativeLlamaCpp)`
3. Modify load time measurement to measure during first inference for Android
4. Update all memory sampling calls to use `_get_total_memory_mb()`
5. Ensure baseline RAM is measured BEFORE model loading/first inference
6. Ensure peak RAM is measured DURING inference when subprocess is active

**Phase 2: Fix Decode TPS Outlier**
1. Implement outlier detection helper method `_is_outlier()`
2. Add outlier detection to `profile_all()` method
3. Log warnings for detected outliers
4. Consider adding outlier filtering to statistical summaries

**Phase 3: Fix RAM Increase Calculation**
1. Verify baseline RAM measurement occurs before model loading
2. Verify peak RAM measurement occurs during inference
3. Add validation to detect negative RAM increase values
4. Log warnings if negative values are detected

**Phase 4: Testing and Validation**
1. Write exploration tests to confirm bugs on unfixed code
2. Apply fixes and verify with fix checking tests
3. Write preservation tests to ensure non-Android platforms unchanged
4. Run full quantization profiling on Android to validate metrics

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bugs on unfixed code, then verify the fixes work correctly and preserve existing behavior for non-Android platforms.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bugs BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Write tests that simulate Android profiling with NativeLlamaCpp and assert that measurements are correct. Run these tests on the UNFIXED code to observe failures and understand the root causes.

**Test Cases**:
1. **Load Time Test**: Profile a model on Android and assert load time > 0.5s (will fail on unfixed code, showing 0.00s)
2. **RAM Measurement Test**: Profile a model on Android and assert peak RAM > 300 MB (will fail on unfixed code, showing ~176 MB)
3. **Decode TPS Consistency Test**: Profile a model multiple times and assert all decode TPS values are within 2x of median (will fail on unfixed code, showing 20x outlier)
4. **Subprocess Memory Test**: Check that subprocess memory is included in RAM measurement (will fail on unfixed code)
5. **RAM Increase Test**: Profile a model on Android and assert RAM increase > 0 MB (will fail on unfixed code, showing negative values like -767.03 MB)

**Expected Counterexamples**:
- Load time = 0.00s instead of 1-5s
- Peak RAM = 176 MB instead of 400-800 MB
- First iteration decode TPS = 45778.25 t/s instead of ~2200 t/s
- Subprocess memory not included in total RAM
- RAM increase = -767.03 MB (Q2_K) or -492.98 MB (Q4_0) instead of positive values

**Possible causes**: 
- Load time measured around path validation instead of actual loading
- RAM measurement only captures Python process, not subprocess
- Timing measurement issue or subprocess startup overhead in first iteration
- Baseline and peak RAM measurements inconsistent (subprocess tracking applied at wrong times)

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds (Android with NativeLlamaCpp), the fixed profiler produces the expected behavior.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  result := profile_quantization_fixed(input)
  ASSERT result.load_time_s > 0.5  # Actual load time measured
  ASSERT result.peak_ram_mb > 300  # Subprocess memory included
  ASSERT result.decode_tps < 10000  # No impossible outliers
  ASSERT result.ram_increase_mb > 0  # RAM increase is positive
END FOR
```

**Test Cases**:
1. **Load Time Fix**: Profile Q2_K and Q4_0 models on Android, assert load times are 1-5s
2. **RAM Fix**: Profile Q2_K and Q4_0 models on Android, assert peak RAM is 400-600 MB and 600-800 MB respectively
3. **Decode TPS Fix**: Profile models multiple times, assert all decode TPS values are within 2x of median
4. **Statistical Summary Fix**: Verify that confidence intervals have non-negative lower bounds
5. **RAM Increase Fix**: Profile Q2_K and Q4_0 models on Android, assert RAM increase is positive (~450 MB for Q2_K, ~1100 MB for Q4_0)

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold (non-Android platforms), the fixed profiler produces the same result as the original profiler.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT profile_quantization_original(input) = profile_quantization_fixed(input)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain
- It catches edge cases that manual unit tests might miss
- It provides strong guarantees that behavior is unchanged for all non-Android inputs

**Test Plan**: Observe behavior on UNFIXED code first for X86Backend and JetsonBackend, then write property-based tests capturing that behavior.

**Test Cases**:
1. **X86 Load Time Preservation**: Verify that load time measurement around `load_model_safe()` continues to work for X86Backend
2. **X86 RAM Preservation**: Verify that RAM measurement using `self.process.memory_info().rss` continues to work for X86Backend
3. **Jetson Load Time Preservation**: Verify that load time measurement continues to work for JetsonBackend
4. **Jetson RAM Preservation**: Verify that RAM measurement continues to work for JetsonBackend
5. **Warmup Preservation**: Verify that warmup inference continues to execute on all platforms
6. **Metrics Preservation**: Verify that TTFT, prefill TPS, and other metrics continue to calculate correctly on all platforms

### Unit Tests

- Test `_get_total_memory_mb()` helper method with mock subprocess
- Test platform detection using `isinstance(llm, NativeLlamaCpp)`
- Test load time measurement for Android vs non-Android platforms
- Test outlier detection with known outlier values
- Test memory sampling during token generation

### Property-Based Tests

- Generate random quantization configurations and verify load time > 0 for Android
- Generate random model sizes and verify RAM measurement includes subprocess memory
- Generate random decode TPS sequences and verify outlier detection works correctly
- Test that all non-Android platforms produce identical results before and after fix

### Integration Tests

- Test full quantization profiling workflow on Android with Q2_K and Q4_0 models
- Test that statistical summaries have non-negative confidence intervals
- Test that comparison matrix shows reasonable values across quantization levels
- Test that HTML reports display correct metrics after fix
