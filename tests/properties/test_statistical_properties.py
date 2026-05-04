"""
Property-based tests for statistical calculations.

**Validates: Requirements 6.1, 6.2, 6.8**

Tests statistical validator functions using hypothesis to generate random inputs.
"""

import math
from hypothesis import given, strategies as st, assume
import pytest

from llm_benchmark.statistics.validator import StatisticalValidator


class TestStatisticalProperties:
    """Property-based tests for statistical calculations."""
    
    @given(st.lists(st.floats(min_value=0, max_value=10000, allow_nan=False, allow_infinity=False), 
                    min_size=3, max_size=100))
    def test_confidence_interval_contains_mean(self, values):
        """
        Property 1: Confidence interval contains mean
        
        **Validates: Requirements 6.1, 6.2**
        
        The 95% confidence interval should always contain the sample mean.
        This is a fundamental property of confidence intervals.
        """
        # Skip if all values are identical (std dev = 0)
        if len(set(values)) == 1:
            return
        
        validator = StatisticalValidator()
        
        # Create runs with single metric
        runs = [{"metric": v} for v in values]
        summaries = validator.summarize_runs(runs)
        
        assert len(summaries) == 1
        summary = summaries[0]
        
        mean = summary.mean
        ci_low, ci_high = summary.confidence_interval_95
        
        # The mean must be within the confidence interval
        assert ci_low <= mean <= ci_high, \
            f"Mean {mean} not in CI [{ci_low}, {ci_high}]"
    
    @given(st.lists(st.floats(min_value=0, max_value=10000, allow_nan=False, allow_infinity=False), 
                    min_size=10, max_size=100))
    def test_outlier_detection_symmetry(self, values):
        """
        Property 2: Outlier detection symmetry
        
        **Validates: Requirements 6.8**
        
        Outliers should be at distribution extremes using the IQR method.
        Specifically, outliers must be outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
        This property verifies that detected outliers are truly at the extremes
        of the distribution according to the IQR criterion.
        """
        # Skip if not enough unique values
        if len(set(values)) < 4:
            return
        
        validator = StatisticalValidator()
        outliers = validator.detect_outliers(values)
        
        if not outliers:
            # No outliers is valid
            return
        
        # Calculate quartiles and IQR using numpy (same as implementation)
        import numpy as np
        sorted_values = sorted(values)
        q1 = float(np.percentile(sorted_values, 25))
        q3 = float(np.percentile(sorted_values, 75))
        iqr = q3 - q1
        
        # Calculate IQR bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Each outlier must be outside the IQR bounds (at distribution extremes)
        for outlier in outliers:
            # Outlier must be in the original dataset
            assert outlier in values, f"Outlier {outlier} not in original values"
            
            # Outlier must be outside the IQR bounds (at extremes)
            is_below_lower = outlier < lower_bound
            is_above_upper = outlier > upper_bound
            
            assert is_below_lower or is_above_upper, \
                f"Outlier {outlier} not outside IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]. " \
                f"Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}"
    
    @given(st.lists(st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False), 
                    min_size=3, max_size=50))
    def test_confidence_interval_width_decreases_with_sample_size(self, base_values):
        """
        Property: Confidence interval width decreases as sample size increases.
        
        **Validates: Requirements 6.1, 6.2**
        
        For a given dataset, adding more samples (by repeating the pattern)
        should result in a narrower confidence interval.
        """
        # Skip if all values are identical
        if len(set(base_values)) == 1:
            return
        
        validator = StatisticalValidator()
        
        # Test with original sample
        runs_small = [{"metric": v} for v in base_values]
        summary_small = validator.summarize_runs(runs_small)[0]
        ci_width_small = summary_small.confidence_interval_95[1] - summary_small.confidence_interval_95[0]
        
        # Test with doubled sample (repeat the pattern)
        runs_large = [{"metric": v} for v in base_values * 2]
        summary_large = validator.summarize_runs(runs_large)[0]
        ci_width_large = summary_large.confidence_interval_95[1] - summary_large.confidence_interval_95[0]
        
        # Larger sample should have narrower CI (or equal if very small std dev)
        # Allow small tolerance for floating point errors
        assert ci_width_large <= ci_width_small * 1.01, \
            f"CI width should decrease: small={ci_width_small}, large={ci_width_large}"
    
    @given(st.lists(st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False), 
                    min_size=3, max_size=50))
    def test_mean_within_data_range(self, values):
        """
        Property: Mean should be within the range of the data.
        
        **Validates: Requirements 6.1**
        
        The calculated mean must be between the minimum and maximum values.
        """
        validator = StatisticalValidator()
        runs = [{"metric": v} for v in values]
        summary = validator.summarize_runs(runs)[0]
        
        min_val = min(values)
        max_val = max(values)
        mean = summary.mean
        
        # Allow small floating point tolerance
        tolerance = 1e-10
        assert min_val - tolerance <= mean <= max_val + tolerance, \
            f"Mean {mean} not in range [{min_val}, {max_val}]"
    
    @given(st.lists(st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False), 
                    min_size=3, max_size=50))
    def test_std_dev_non_negative(self, values):
        """
        Property: Standard deviation should be non-negative.
        
        **Validates: Requirements 6.1**
        
        Standard deviation is always >= 0 by definition.
        """
        validator = StatisticalValidator()
        runs = [{"metric": v} for v in values]
        summary = validator.summarize_runs(runs)[0]
        
        assert summary.std_dev >= 0, \
            f"Standard deviation {summary.std_dev} is negative"
    
    @given(st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False))
    def test_identical_values_zero_std_dev(self, value):
        """
        Property: Identical values should have zero standard deviation.
        
        **Validates: Requirements 6.1**
        
        When all values are the same, std dev should be 0 (or very close due to floating point).
        """
        validator = StatisticalValidator()
        runs = [{"metric": value} for _ in range(5)]
        summary = validator.summarize_runs(runs)[0]
        
        # Allow small floating point tolerance
        tolerance = 1e-10
        assert summary.std_dev < tolerance, \
            f"Std dev for identical values should be ~0, got {summary.std_dev}"
        
        # CI should collapse to the mean (with small tolerance)
        ci_low, ci_high = summary.confidence_interval_95
        assert abs(ci_low - value) < tolerance and abs(ci_high - value) < tolerance, \
            f"CI for identical values should be ~[{value}, {value}], got [{ci_low}, {ci_high}]"
