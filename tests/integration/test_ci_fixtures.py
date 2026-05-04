"""
Integration test for CI/CD fixtures and baseline validation.

This test verifies that:
1. Test prompts are properly defined
2. Baseline results are accessible
3. Validation functions work correctly
4. Regression detection works correctly
"""

import pytest
from tests.fixtures import (
    SHORT_PROMPT,
    MEDIUM_PROMPT,
    LONG_PROMPT,
    BASELINE_X86_CPU,
    validate_results,
    detect_regression,
    check_metric_in_range,
)


def test_prompts_are_defined():
    """Test that all required prompts are defined and non-empty."""
    assert SHORT_PROMPT, "SHORT_PROMPT should be defined"
    assert MEDIUM_PROMPT, "MEDIUM_PROMPT should be defined"
    assert LONG_PROMPT, "LONG_PROMPT should be defined"
    
    # Verify prompts have reasonable lengths
    assert len(SHORT_PROMPT) < 200, "SHORT_PROMPT should be short"
    assert len(MEDIUM_PROMPT) > 200, "MEDIUM_PROMPT should be medium length"
    assert len(LONG_PROMPT) > 1000, "LONG_PROMPT should be long"


def test_baseline_structure():
    """Test that baseline results have correct structure."""
    assert "model" in BASELINE_X86_CPU
    assert "hardware" in BASELINE_X86_CPU
    assert "metrics" in BASELINE_X86_CPU
    
    metrics = BASELINE_X86_CPU["metrics"]
    
    # Verify required metrics are present
    required_metrics = [
        "load_time_s",
        "peak_ram_mb",
        "ttft_ms",
        "prefill_tps",
        "decode_tps",
    ]
    
    for metric in required_metrics:
        assert metric in metrics, f"Metric {metric} should be in baseline"
        assert isinstance(metrics[metric], tuple), f"Metric {metric} should be a tuple"
        assert len(metrics[metric]) == 2, f"Metric {metric} should have (min, max)"
        min_val, max_val = metrics[metric]
        assert min_val < max_val, f"Metric {metric} min should be less than max"


def test_metric_range_validation():
    """Test metric range validation function."""
    # Test value within range
    is_valid, message = check_metric_in_range(
        "ttft_ms",
        actual_value=100.0,
        expected_range=(50.0, 500.0),
        tolerance=0.2
    )
    assert is_valid, "Value within range should be valid"
    assert "within expected range" in message
    
    # Test value outside range
    is_valid, message = check_metric_in_range(
        "ttft_ms",
        actual_value=1000.0,
        expected_range=(50.0, 500.0),
        tolerance=0.2
    )
    assert not is_valid, "Value outside range should be invalid"
    assert "OUTSIDE expected range" in message


def test_results_validation():
    """Test results validation against baseline."""
    # Simulate good results (within expected ranges)
    good_results = {
        "load_time_s": 2.0,
        "peak_ram_mb": 800.0,
        "ttft_ms": 150.0,
        "prefill_tps": 200.0,
        "decode_tps": 25.0,
    }
    
    validation = validate_results(good_results, baseline_key="x86_cpu")
    
    # All metrics should be valid
    for metric, (is_valid, message) in validation.items():
        assert is_valid, f"Metric {metric} should be valid: {message}"


def test_regression_detection_throughput():
    """Test regression detection for throughput metrics (higher is better)."""
    # No regression: current is similar to baseline
    is_regression, message = detect_regression(
        "decode_tps",
        current_value=25.0,
        baseline_value=25.0
    )
    assert not is_regression, "Similar values should not trigger regression"
    
    # Regression: current is significantly lower (20%+ lower)
    is_regression, message = detect_regression(
        "decode_tps",
        current_value=15.0,  # 40% lower than baseline
        baseline_value=25.0
    )
    assert is_regression, "Significantly lower throughput should trigger regression"
    assert "REGRESSION detected" in message


def test_regression_detection_latency():
    """Test regression detection for latency metrics (lower is better)."""
    # No regression: current is similar to baseline
    is_regression, message = detect_regression(
        "ttft_ms",
        current_value=150.0,
        baseline_value=150.0
    )
    assert not is_regression, "Similar values should not trigger regression"
    
    # Regression: current is significantly higher (20%+ higher)
    is_regression, message = detect_regression(
        "ttft_ms",
        current_value=200.0,  # 33% higher than baseline
        baseline_value=150.0
    )
    assert is_regression, "Significantly higher latency should trigger regression"
    assert "REGRESSION detected" in message


def test_regression_detection_memory():
    """Test regression detection for memory metrics (lower is better)."""
    # No regression: current is similar to baseline
    is_regression, message = detect_regression(
        "peak_ram_mb",
        current_value=800.0,
        baseline_value=800.0
    )
    assert not is_regression, "Similar values should not trigger regression"
    
    # Regression: current is significantly higher (20%+ higher)
    is_regression, message = detect_regression(
        "peak_ram_mb",
        current_value=1000.0,  # 25% higher than baseline
        baseline_value=800.0
    )
    assert is_regression, "Significantly higher memory usage should trigger regression"
    assert "REGRESSION detected" in message


def test_validation_with_missing_metrics():
    """Test validation handles missing metrics gracefully."""
    # Results with a metric not in baseline
    results = {
        "ttft_ms": 150.0,
        "custom_metric": 42.0,  # Not in baseline
    }
    
    validation = validate_results(results, baseline_key="x86_cpu")
    
    # ttft_ms should be validated
    assert "ttft_ms" in validation
    assert validation["ttft_ms"][0], "ttft_ms should be valid"
    
    # custom_metric should be skipped
    assert "custom_metric" in validation
    assert validation["custom_metric"][0], "Unknown metrics should be marked as valid (skipped)"
    assert "No baseline available" in validation["custom_metric"][1]


def test_invalid_baseline_key():
    """Test that invalid baseline key raises error."""
    results = {"ttft_ms": 150.0}
    
    with pytest.raises(ValueError, match="Unknown baseline key"):
        validate_results(results, baseline_key="invalid_key")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
