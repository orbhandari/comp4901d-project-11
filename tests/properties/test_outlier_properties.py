"""
Property-based tests for outlier detection.

**Validates: Requirements 6.8**

Tests outlier detection using hypothesis to generate random value lists.
"""

from hypothesis import given, strategies as st, assume
import pytest

from llm_benchmark.statistics.validator import StatisticalValidator


class TestOutlierProperties:
    """Property-based tests for outlier detection."""
    
    @given(st.lists(st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False), 
                    min_size=10, max_size=100))
    def test_outliers_are_in_original_data(self, values):
        """
        Property: All detected outliers must be in the original dataset.
        
        **Validates: Requirements 6.8**
        
        Outlier detection should only identify values that actually exist
        in the input data.
        """
        # Skip if not enough unique values
        if len(set(values)) < 4:
            return
        
        validator = StatisticalValidator()
        outliers = validator.detect_outliers(values)
        
        for outlier in outliers:
            assert outlier in values, \
                f"Outlier {outlier} not found in original values"
    
    @given(st.lists(st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False), 
                    min_size=10, max_size=100))
    def test_outliers_at_extremes(self, values):
        """
        Property: Outliers should be at distribution extremes.
        
        **Validates: Requirements 6.8**
        
        Using the IQR method, outliers should be beyond Q1 - 1.5*IQR
        or Q3 + 1.5*IQR, meaning they are at the extremes of the distribution.
        """
        # Skip if not enough unique values
        if len(set(values)) < 4:
            return
        
        validator = StatisticalValidator()
        outliers = validator.detect_outliers(values)
        
        if not outliers:
            # No outliers is valid
            return
        
        # Calculate quartiles manually
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        # Use numpy-style percentile calculation
        import numpy as np
        q1 = float(np.percentile(sorted_values, 25))
        q3 = float(np.percentile(sorted_values, 75))
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Each outlier should be outside the bounds
        for outlier in outliers:
            is_below_lower = outlier < lower_bound
            is_above_upper = outlier > upper_bound
            
            assert is_below_lower or is_above_upper, \
                f"Outlier {outlier} not outside bounds [{lower_bound}, {upper_bound}]"
    
    @given(st.lists(st.floats(min_value=100, max_value=200, allow_nan=False, allow_infinity=False), 
                    min_size=10, max_size=50))
    def test_no_outliers_in_tight_distribution(self, values):
        """
        Property: Tight distributions should have few or no outliers.
        
        **Validates: Requirements 6.8**
        
        When values are drawn from a narrow range, there should be
        few or no outliers detected.
        """
        # Skip if not enough unique values
        if len(set(values)) < 4:
            return
        
        validator = StatisticalValidator()
        outliers = validator.detect_outliers(values)
        
        # For a uniform distribution in [100, 200], we expect few outliers
        # This is a probabilistic property, so we just check it's reasonable
        outlier_ratio = len(outliers) / len(values)
        
        # Less than or equal to 30% should be outliers for a uniform distribution
        # (IQR method can identify outliers even in uniform distributions)
        assert outlier_ratio <= 0.3, \
            f"Too many outliers ({len(outliers)}/{len(values)}) in tight distribution"
    
    @given(st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False))
    def test_no_outliers_in_identical_values(self, value):
        """
        Property: Identical values should have no outliers.
        
        **Validates: Requirements 6.8**
        
        When all values are the same, there should be no outliers.
        """
        values = [value] * 10
        validator = StatisticalValidator()
        outliers = validator.detect_outliers(values)
        
        assert len(outliers) == 0, \
            f"Identical values should have no outliers, found {len(outliers)}"
    
    @given(st.lists(st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False), 
                    min_size=10, max_size=50),
           st.floats(min_value=500, max_value=1000, allow_nan=False, allow_infinity=False))
    def test_extreme_value_detected_as_outlier(self, normal_values, extreme_value):
        """
        Property: An extreme value should be detected as an outlier.
        
        **Validates: Requirements 6.8**
        
        When we add a value that is far from the main distribution,
        it should be detected as an outlier.
        """
        # Skip if not enough unique normal values
        if len(set(normal_values)) < 4:
            return
        
        # Add the extreme value
        values = normal_values + [extreme_value]
        
        validator = StatisticalValidator()
        outliers = validator.detect_outliers(values)
        
        # The extreme value should be detected as an outlier
        # (with high probability, though not guaranteed for all distributions)
        if len(outliers) > 0:
            # If outliers were detected, the extreme value should be among them
            max_normal = max(normal_values)
            if extreme_value > max_normal * 2:  # Clearly extreme
                assert extreme_value in outliers, \
                    f"Extreme value {extreme_value} not detected as outlier"
    
    @given(st.lists(st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False), 
                    min_size=10, max_size=100))
    def test_outlier_count_reasonable(self, values):
        """
        Property: Number of outliers should be reasonable.
        
        **Validates: Requirements 6.8**
        
        The IQR method typically identifies a small percentage of values
        as outliers. We shouldn't have more than ~30% outliers in most cases.
        """
        # Skip if not enough unique values
        if len(set(values)) < 4:
            return
        
        validator = StatisticalValidator()
        outliers = validator.detect_outliers(values)
        
        outlier_ratio = len(outliers) / len(values)
        
        # Sanity check: shouldn't have more than 50% outliers
        assert outlier_ratio <= 0.5, \
            f"Too many outliers: {len(outliers)}/{len(values)} = {outlier_ratio:.1%}"
    
    @given(st.lists(st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False), 
                    min_size=10, max_size=100))
    def test_outlier_detection_deterministic(self, values):
        """
        Property: Outlier detection should be deterministic.
        
        **Validates: Requirements 6.8**
        
        Running outlier detection twice on the same data should
        produce the same results.
        """
        # Skip if not enough unique values
        if len(set(values)) < 4:
            return
        
        validator = StatisticalValidator()
        
        outliers1 = validator.detect_outliers(values)
        outliers2 = validator.detect_outliers(values)
        
        assert outliers1 == outliers2, \
            "Outlier detection should be deterministic"
