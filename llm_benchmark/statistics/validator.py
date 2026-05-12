"""
Statistical validator for benchmark results.

Provides statistical analysis including mean, standard deviation, confidence intervals,
significance testing, and outlier detection.
"""

import math
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

from llm_benchmark.models import ComparisonResult, StatisticalSummary


class StatisticalValidator:
    """Performs statistical analysis on benchmark results."""
    
    def summarize_runs(self, runs: List[Dict[str, float]]) -> List[StatisticalSummary]:
        """
        Calculate mean, std dev, confidence intervals for each metric.
        
        Args:
            runs: List of dictionaries containing metric values
            
        Returns:
            List of StatisticalSummary objects, one per metric
            
        Raises:
            ValueError: If fewer than 3 runs provided (but not empty)
        """
        if not runs:
            return []
        
        if len(runs) < 3:
            raise ValueError("Minimum 3 runs required for statistical validity")
        
        # Get all metric names from first run
        metric_names = list(runs[0].keys())
        summaries = []
        
        for metric_name in metric_names:
            # Extract values for this metric
            values = [run[metric_name] for run in runs]
            
            # Calculate statistics
            mean = float(np.mean(values))
            std_dev = float(np.std(values, ddof=1))  # Sample std dev
            
            # Calculate 95% confidence interval
            n = len(values)
            ci_lower, ci_upper = self._calculate_confidence_interval(mean, std_dev, n)
            
            # Detect outliers
            outliers = self.detect_outliers(values)
            
            summaries.append(StatisticalSummary(
                metric_name=metric_name,
                mean=mean,
                std_dev=std_dev,
                confidence_interval_95=(ci_lower, ci_upper),
                outliers=outliers
            ))
        
        return summaries
    
    def _calculate_confidence_interval(
        self, 
        mean: float, 
        std_dev: float, 
        n: int
    ) -> Tuple[float, float]:
        """
        Calculate 95% confidence interval.
        
        Uses formula: mean ± 1.96 * (std / sqrt(n))
        
        Args:
            mean: Sample mean
            std_dev: Sample standard deviation
            n: Sample size
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if n == 0:
            return (mean, mean)
        
        margin = 1.96 * (std_dev / math.sqrt(n))
        return (mean - margin, mean + margin)
    
    def compare_configurations(
        self, 
        runs_a: List[Dict[str, float]], 
        runs_b: List[Dict[str, float]]
    ) -> List[ComparisonResult]:
        """
        Perform paired t-tests comparing two configurations.
        
        Args:
            runs_a: List of metric dictionaries for configuration A
            runs_b: List of metric dictionaries for configuration B
            
        Returns:
            List of ComparisonResult objects, one per metric
            
        Raises:
            ValueError: If runs have different lengths or fewer than 3 runs
        """
        if len(runs_a) != len(runs_b):
            raise ValueError("Both configurations must have same number of runs")
        
        if len(runs_a) < 3:
            raise ValueError("Minimum 3 runs required for statistical validity")
        
        if not runs_a or not runs_b:
            return []
        
        # Get all metric names
        metric_names = list(runs_a[0].keys())
        comparisons = []
        
        for metric_name in metric_names:
            # Extract values for this metric
            values_a = [run[metric_name] for run in runs_a]
            values_b = [run[metric_name] for run in runs_b]
            
            # Calculate means
            mean_a = float(np.mean(values_a))
            mean_b = float(np.mean(values_b))
            difference = mean_b - mean_a
            
            # Perform paired t-test
            t_stat, p_value = stats.ttest_rel(values_a, values_b)
            is_significant = bool(p_value < 0.05)
            
            comparisons.append(ComparisonResult(
                metric_name=metric_name,
                config_a_mean=mean_a,
                config_b_mean=mean_b,
                difference=difference,
                p_value=float(p_value),
                is_significant=is_significant
            ))
        
        return comparisons
    
    def detect_outliers(self, values: List[float]) -> List[float]:
        """
        Detect outliers using IQR method.
        
        Outliers are values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        
        Args:
            values: List of numeric values
            
        Returns:
            List of outlier values
        """
        if len(values) < 4:
            return []
        
        # Calculate quartiles
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        iqr = q3 - q1
        
        # Calculate bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Find outliers
        outliers = [v for v in values if v < lower_bound or v > upper_bound]
        
        return outliers
