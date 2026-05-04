"""
Baseline results and expected metric ranges for regression detection.

This module defines expected performance ranges for different hardware
configurations to detect performance regressions in CI/CD.
"""

from typing import Dict, Tuple

# Type alias for metric ranges (min, max)
MetricRange = Tuple[float, float]

# Baseline results for TinyLlama-1.1B Q4_0 on different hardware
# These are approximate ranges based on typical performance

BASELINE_X86_CPU = {
    "model": "tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
    "hardware": "x86_64 CPU (4 cores, 8GB RAM)",
    "metrics": {
        # Load time in seconds (min, max)
        "load_time_s": (0.5, 5.0),
        
        # Peak RAM in MB (min, max)
        "peak_ram_mb": (500, 1500),
        
        # Time to first token in milliseconds (min, max)
        "ttft_ms": (50, 500),
        
        # Prefill throughput in tokens/second (min, max)
        "prefill_tps": (50, 500),
        
        # Decode throughput in tokens/second (min, max)
        "decode_tps": (5, 50),
        
        # Total inference time for 100 tokens in seconds (min, max)
        "total_time_100_tokens_s": (2, 30),
    },
    "notes": "CPU-only inference on typical CI/CD runner",
}

BASELINE_JETSON_XAVIER_NX_GPU = {
    "model": "tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
    "hardware": "Jetson Xavier NX (6-core ARM, 8GB RAM, GPU)",
    "metrics": {
        # Load time in seconds (min, max)
        "load_time_s": (1.0, 10.0),
        
        # Peak RAM in MB (min, max)
        "peak_ram_mb": (600, 2000),
        
        # GPU memory in MB (min, max)
        "gpu_memory_mb": (400, 1200),
        
        # Time to first token in milliseconds (min, max)
        "ttft_ms": (20, 200),
        
        # Prefill throughput in tokens/second (min, max)
        "prefill_tps": (100, 1000),
        
        # Decode throughput in tokens/second (min, max)
        "decode_tps": (10, 100),
        
        # GPU utilization percentage (min, max)
        "gpu_utilization_pct": (30, 100),
        
        # Total inference time for 100 tokens in seconds (min, max)
        "total_time_100_tokens_s": (1, 15),
    },
    "notes": "GPU-accelerated inference on Jetson Xavier NX",
}

BASELINE_JETSON_XAVIER_NX_CPU = {
    "model": "tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
    "hardware": "Jetson Xavier NX (6-core ARM, 8GB RAM, CPU-only)",
    "metrics": {
        # Load time in seconds (min, max)
        "load_time_s": (0.5, 8.0),
        
        # Peak RAM in MB (min, max)
        "peak_ram_mb": (500, 1800),
        
        # Time to first token in milliseconds (min, max)
        "ttft_ms": (100, 800),
        
        # Prefill throughput in tokens/second (min, max)
        "prefill_tps": (20, 200),
        
        # Decode throughput in tokens/second (min, max)
        "decode_tps": (3, 30),
        
        # Total inference time for 100 tokens in seconds (min, max)
        "total_time_100_tokens_s": (3, 40),
    },
    "notes": "CPU-only inference on Jetson Xavier NX (fallback mode)",
}

# All baselines
ALL_BASELINES = {
    "x86_cpu": BASELINE_X86_CPU,
    "jetson_gpu": BASELINE_JETSON_XAVIER_NX_GPU,
    "jetson_cpu": BASELINE_JETSON_XAVIER_NX_CPU,
}


def check_metric_in_range(
    metric_name: str,
    actual_value: float,
    expected_range: MetricRange,
    tolerance: float = 0.2
) -> Tuple[bool, str]:
    """
    Check if a metric value is within expected range.
    
    Args:
        metric_name: Name of the metric
        actual_value: Actual measured value
        expected_range: Expected (min, max) range
        tolerance: Tolerance factor for range expansion (default 20%)
        
    Returns:
        Tuple of (is_valid, message)
    """
    min_val, max_val = expected_range
    
    # Expand range by tolerance to account for variability
    expanded_min = min_val * (1 - tolerance)
    expanded_max = max_val * (1 + tolerance)
    
    if expanded_min <= actual_value <= expanded_max:
        return True, f"{metric_name}: {actual_value:.2f} is within expected range [{min_val:.2f}, {max_val:.2f}]"
    else:
        return False, (
            f"{metric_name}: {actual_value:.2f} is OUTSIDE expected range [{min_val:.2f}, {max_val:.2f}] "
            f"(with {tolerance*100}% tolerance: [{expanded_min:.2f}, {expanded_max:.2f}])"
        )


def validate_results(results: Dict[str, float], baseline_key: str) -> Dict[str, Tuple[bool, str]]:
    """
    Validate benchmark results against baseline.
    
    Args:
        results: Dictionary of metric_name -> value
        baseline_key: Key for baseline to compare against (e.g., "x86_cpu")
        
    Returns:
        Dictionary of metric_name -> (is_valid, message)
    """
    if baseline_key not in ALL_BASELINES:
        raise ValueError(f"Unknown baseline key: {baseline_key}")
    
    baseline = ALL_BASELINES[baseline_key]
    validation_results = {}
    
    for metric_name, actual_value in results.items():
        if metric_name in baseline["metrics"]:
            expected_range = baseline["metrics"][metric_name]
            is_valid, message = check_metric_in_range(metric_name, actual_value, expected_range)
            validation_results[metric_name] = (is_valid, message)
        else:
            validation_results[metric_name] = (True, f"{metric_name}: No baseline available (skipped)")
    
    return validation_results


# Regression thresholds
REGRESSION_THRESHOLDS = {
    # Performance regressions (slower is worse)
    "ttft_ms": 1.2,  # 20% slower is a regression
    "decode_tps": 0.8,  # 20% slower (lower throughput) is a regression
    "prefill_tps": 0.8,  # 20% slower (lower throughput) is a regression
    
    # Memory regressions (more is worse)
    "peak_ram_mb": 1.2,  # 20% more memory is a regression
    "gpu_memory_mb": 1.2,  # 20% more GPU memory is a regression
    
    # Load time regressions (slower is worse)
    "load_time_s": 1.2,  # 20% slower is a regression
}


def detect_regression(
    metric_name: str,
    current_value: float,
    baseline_value: float
) -> Tuple[bool, str]:
    """
    Detect if a metric shows performance regression.
    
    Args:
        metric_name: Name of the metric
        current_value: Current measured value
        baseline_value: Baseline value to compare against
        
    Returns:
        Tuple of (is_regression, message)
    """
    if metric_name not in REGRESSION_THRESHOLDS:
        return False, f"{metric_name}: No regression threshold defined"
    
    threshold = REGRESSION_THRESHOLDS[metric_name]
    
    # For throughput metrics (higher is better), check if current is too low
    if "tps" in metric_name:
        if current_value < baseline_value * threshold:
            regression_pct = ((baseline_value - current_value) / baseline_value) * 100
            return True, (
                f"{metric_name}: REGRESSION detected! "
                f"Current {current_value:.2f} is {regression_pct:.1f}% lower than baseline {baseline_value:.2f}"
            )
    # For latency/memory metrics (lower is better), check if current is too high
    else:
        if current_value > baseline_value * threshold:
            regression_pct = ((current_value - baseline_value) / baseline_value) * 100
            return True, (
                f"{metric_name}: REGRESSION detected! "
                f"Current {current_value:.2f} is {regression_pct:.1f}% higher than baseline {baseline_value:.2f}"
            )
    
    return False, f"{metric_name}: No regression detected"
