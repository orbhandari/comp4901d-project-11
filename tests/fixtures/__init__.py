"""
Test fixtures for LLM benchmark framework.

This package provides:
- Fixed test prompts for reproducible benchmarking
- Baseline results and expected metric ranges
- Regression detection utilities
"""

from .test_prompts import (
    SHORT_PROMPT,
    MEDIUM_PROMPT,
    LONG_PROMPT,
    CACHE_PREFIX,
    CACHE_TEST_PROMPT_1,
    CACHE_TEST_PROMPT_2,
    BATCH_PROMPTS,
    UNICODE_PROMPT,
    SPECIAL_CHARS_PROMPT,
    ALL_PROMPTS,
)

from .baseline_results import (
    BASELINE_X86_CPU,
    BASELINE_JETSON_XAVIER_NX_GPU,
    BASELINE_JETSON_XAVIER_NX_CPU,
    ALL_BASELINES,
    REGRESSION_THRESHOLDS,
    check_metric_in_range,
    validate_results,
    detect_regression,
)

__all__ = [
    # Prompts
    "SHORT_PROMPT",
    "MEDIUM_PROMPT",
    "LONG_PROMPT",
    "CACHE_PREFIX",
    "CACHE_TEST_PROMPT_1",
    "CACHE_TEST_PROMPT_2",
    "BATCH_PROMPTS",
    "UNICODE_PROMPT",
    "SPECIAL_CHARS_PROMPT",
    "ALL_PROMPTS",
    # Baselines
    "BASELINE_X86_CPU",
    "BASELINE_JETSON_XAVIER_NX_GPU",
    "BASELINE_JETSON_XAVIER_NX_CPU",
    "ALL_BASELINES",
    "REGRESSION_THRESHOLDS",
    # Utilities
    "check_metric_in_range",
    "validate_results",
    "detect_regression",
]
