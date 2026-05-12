"""
Unit tests for statistical validator.

**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.8**

Tests statistical calculations with known values and distributions.
"""

import pytest
import math
from llm_benchmark.statistics.validator import StatisticalValidator


class TestStatisticalValidator:
    """Unit tests for StatisticalValidator."""
    
    def test_mean_calculation(self):
        """
        Test mean calculation with known values.
        
        **Validates: Requirements 6.1**
        """
        validator = StatisticalValidator()
        
        runs = [
            {"ttft_ms": 100.0},
            {"ttft_ms": 150.0},
            {"ttft_ms": 200.0},
        ]
        
        summaries = validator.summarize_runs(runs)
        
        assert len(summaries) == 1
        assert summaries[0].metric_name == "ttft_ms"
        assert summaries[0].mean == 150.0
    
    def test_std_dev_calculation(self):
        """
        Test standard deviation calculation with known values.
        
        **Validates: Requirements 6.1**
        """
        validator = StatisticalValidator()
        
        # Values: 10, 20, 30 -> mean = 20, std dev = 10
        runs = [
            {"metric": 10.0},
            {"metric": 20.0},
            {"metric": 30.0},
        ]
        
        summaries = validator.summarize_runs(runs)
        
        assert len(summaries) == 1
        # Sample std dev with ddof=1
        expected_std = 10.0
        assert abs(summaries[0].std_dev - expected_std) < 0.01
    
    def test_confidence_interval_calculation(self):
        """
        Test confidence interval calculation with known values.
        
        **Validates: Requirements 6.2**
        """
        validator = StatisticalValidator()
        
        # Use values with known statistics
        runs = [
            {"metric": 100.0},
            {"metric": 110.0},
            {"metric": 90.0},
            {"metric": 105.0},
            {"metric": 95.0},
        ]
        
        summaries = validator.summarize_runs(runs)
        
        assert len(summaries) == 1
        summary = summaries[0]
        
        # Mean should be 100
        assert abs(summary.mean - 100.0) < 0.1
        
        # CI should contain the mean
        ci_low, ci_high = summary.confidence_interval_95
        assert ci_low <= summary.mean <= ci_high
        
        # CI should be symmetric around mean
        margin = ci_high - summary.mean
        assert abs((summary.mean - ci_low) - margin) < 0.01
    
    def test_paired_t_test_significant_difference(self):
        """
        Test paired t-test with known significant difference.
        
        **Validates: Requirements 6.3, 6.4, 6.5**
        """
        validator = StatisticalValidator()
        
        # Configuration A: consistently around 100
        runs_a = [
            {"ttft_ms": 100.0},
            {"ttft_ms": 102.0},
            {"ttft_ms": 98.0},
            {"ttft_ms": 101.0},
            {"ttft_ms": 99.0},
        ]
        
        # Configuration B: consistently around 150 (clearly different)
        runs_b = [
            {"ttft_ms": 150.0},
            {"ttft_ms": 152.0},
            {"ttft_ms": 148.0},
            {"ttft_ms": 151.0},
            {"ttft_ms": 149.0},
        ]
        
        comparisons = validator.compare_configurations(runs_a, runs_b)
        
        assert len(comparisons) == 1
        comparison = comparisons[0]
        
        assert comparison.metric_name == "ttft_ms"
        assert abs(comparison.config_a_mean - 100.0) < 1.0
        assert abs(comparison.config_b_mean - 150.0) < 1.0
        assert abs(comparison.difference - 50.0) < 1.0
        
        # Should be statistically significant
        assert comparison.p_value < 0.05
        assert comparison.is_significant is True
    
    def test_paired_t_test_no_difference(self):
        """
        Test paired t-test with no significant difference.
        
        **Validates: Requirements 6.3, 6.4, 6.5**
        """
        validator = StatisticalValidator()
        
        # Both configurations have overlapping distributions
        # The differences are random, not systematic
        runs_a = [
            {"ttft_ms": 100.0},
            {"ttft_ms": 110.0},
            {"ttft_ms": 95.0},
            {"ttft_ms": 105.0},
            {"ttft_ms": 90.0},
        ]
        
        runs_b = [
            {"ttft_ms": 98.0},
            {"ttft_ms": 108.0},
            {"ttft_ms": 97.0},
            {"ttft_ms": 103.0},
            {"ttft_ms": 94.0},
        ]
        
        comparisons = validator.compare_configurations(runs_a, runs_b)
        
        assert len(comparisons) == 1
        comparison = comparisons[0]
        
        # Means should be close
        assert abs(comparison.config_a_mean - 100.0) < 10.0
        assert abs(comparison.config_b_mean - 100.0) < 10.0
        
        # Should not be statistically significant (p > 0.05)
        # The differences are small and not systematic
        assert comparison.is_significant is False
    
    def test_outlier_detection_with_known_outliers(self):
        """
        Test outlier detection with known outliers.
        
        **Validates: Requirements 6.8**
        """
        validator = StatisticalValidator()
        
        # Normal values around 100, with clear outliers at 10 and 200
        values = [95.0, 98.0, 100.0, 102.0, 105.0, 10.0, 200.0]
        
        outliers = validator.detect_outliers(values)
        
        # Should detect the extreme values as outliers
        assert 10.0 in outliers
        assert 200.0 in outliers
        
        # Normal values should not be outliers
        assert 100.0 not in outliers
    
    def test_outlier_detection_no_outliers(self):
        """
        Test outlier detection with no outliers.
        
        **Validates: Requirements 6.8**
        """
        validator = StatisticalValidator()
        
        # Tightly clustered values
        values = [98.0, 99.0, 100.0, 101.0, 102.0]
        
        outliers = validator.detect_outliers(values)
        
        # Should detect no outliers
        assert len(outliers) == 0
    
    def test_multiple_metrics(self):
        """
        Test statistical analysis with multiple metrics.
        
        **Validates: Requirements 6.1, 6.2**
        """
        validator = StatisticalValidator()
        
        runs = [
            {"ttft_ms": 100.0, "decode_tps": 50.0},
            {"ttft_ms": 110.0, "decode_tps": 55.0},
            {"ttft_ms": 90.0, "decode_tps": 45.0},
        ]
        
        summaries = validator.summarize_runs(runs)
        
        assert len(summaries) == 2
        
        # Find summaries by metric name
        ttft_summary = next(s for s in summaries if s.metric_name == "ttft_ms")
        tps_summary = next(s for s in summaries if s.metric_name == "decode_tps")
        
        assert abs(ttft_summary.mean - 100.0) < 0.1
        assert abs(tps_summary.mean - 50.0) < 0.1
    
    def test_minimum_runs_validation(self):
        """
        Test that minimum 3 runs are required.
        
        **Validates: Requirements 6.7**
        """
        validator = StatisticalValidator()
        
        # Only 2 runs
        runs = [
            {"metric": 100.0},
            {"metric": 110.0},
        ]
        
        with pytest.raises(ValueError, match="Minimum 3 runs required"):
            validator.summarize_runs(runs)
    
    def test_comparison_minimum_runs_validation(self):
        """
        Test that minimum 3 runs are required for comparison.
        
        **Validates: Requirements 6.7**
        """
        validator = StatisticalValidator()
        
        runs_a = [{"metric": 100.0}, {"metric": 110.0}]
        runs_b = [{"metric": 150.0}, {"metric": 160.0}]
        
        with pytest.raises(ValueError, match="Minimum 3 runs required"):
            validator.compare_configurations(runs_a, runs_b)
    
    def test_comparison_equal_length_validation(self):
        """
        Test that both configurations must have same number of runs.
        
        **Validates: Requirements 6.3**
        """
        validator = StatisticalValidator()
        
        runs_a = [{"metric": 100.0}, {"metric": 110.0}, {"metric": 105.0}]
        runs_b = [{"metric": 150.0}, {"metric": 160.0}]
        
        with pytest.raises(ValueError, match="same number of runs"):
            validator.compare_configurations(runs_a, runs_b)
    
    def test_empty_runs(self):
        """
        Test handling of empty runs list.
        
        **Validates: Requirements 6.1**
        """
        validator = StatisticalValidator()
        
        summaries = validator.summarize_runs([])
        assert summaries == []
    
    def test_outlier_detection_insufficient_data(self):
        """
        Test outlier detection with insufficient data.
        
        **Validates: Requirements 6.8**
        """
        validator = StatisticalValidator()
        
        # Less than 4 values
        values = [100.0, 110.0, 120.0]
        
        outliers = validator.detect_outliers(values)
        
        # Should return empty list for insufficient data
        assert outliers == []
    
    def test_confidence_interval_with_zero_std_dev(self):
        """
        Test confidence interval when all values are identical.
        
        **Validates: Requirements 6.2**
        """
        validator = StatisticalValidator()
        
        runs = [
            {"metric": 100.0},
            {"metric": 100.0},
            {"metric": 100.0},
        ]
        
        summaries = validator.summarize_runs(runs)
        
        assert len(summaries) == 1
        summary = summaries[0]
        
        assert summary.mean == 100.0
        assert summary.std_dev == 0.0
        
        # CI should collapse to the mean
        ci_low, ci_high = summary.confidence_interval_95
        assert ci_low == 100.0
        assert ci_high == 100.0
