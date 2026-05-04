"""
Unit tests for VisualizationGenerator.

Tests chart generation, error bar handling, and file output.
"""

import os
import pytest
import tempfile
import shutil
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from llm_benchmark.visualization.visualization_generator import VisualizationGenerator
from llm_benchmark.models import (
    QuantizationResult,
    AblationResult,
    InferenceMetrics,
    StatisticalSummary,
)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def viz_generator(temp_output_dir):
    """Create VisualizationGenerator instance."""
    return VisualizationGenerator(output_dir=temp_output_dir, dpi=100)


@pytest.fixture
def sample_quantization_results():
    """Create sample quantization results."""
    return [
        QuantizationResult(
            quantization="Q8_0",
            load_time_s=2.5,
            peak_ram_mb=4000.0,
            ram_increase_mb=3500.0,
            ttft_ms=150.0,
            prefill_tps=500.0,
            decode_tps=25.0,
            prompt_tokens=100,
            output_tokens=50,
        ),
        QuantizationResult(
            quantization="Q4_0",
            load_time_s=1.8,
            peak_ram_mb=2500.0,
            ram_increase_mb=2000.0,
            ttft_ms=180.0,
            prefill_tps=450.0,
            decode_tps=22.0,
            prompt_tokens=100,
            output_tokens=50,
        ),
        QuantizationResult(
            quantization="Q2_K",
            load_time_s=1.2,
            peak_ram_mb=1500.0,
            ram_increase_mb=1000.0,
            ttft_ms=220.0,
            prefill_tps=400.0,
            decode_tps=18.0,
            prompt_tokens=100,
            output_tokens=50,
        ),
    ]


@pytest.fixture
def sample_statistical_summaries():
    """Create sample statistical summaries."""
    return [
        StatisticalSummary(
            metric_name="Q8_0_ttft_ms",
            mean=150.0,
            std_dev=5.0,
            confidence_interval_95=(145.0, 155.0),
        ),
        StatisticalSummary(
            metric_name="Q4_0_ttft_ms",
            mean=180.0,
            std_dev=8.0,
            confidence_interval_95=(172.0, 188.0),
        ),
        StatisticalSummary(
            metric_name="Q8_0_decode_tps",
            mean=25.0,
            std_dev=1.0,
            confidence_interval_95=(24.0, 26.0),
        ),
    ]


@pytest.fixture
def sample_inference_metrics():
    """Create sample inference metrics."""
    return InferenceMetrics(
        ttft_ms=150.0,
        prefill_tps=500.0,
        decode_tps=25.0,
        total_time_s=5.0,
        prompt_tokens=100,
        output_tokens=50,
        peak_memory_mb=4000.0,
        per_token_latency_ms=[40.0, 42.0, 38.0, 41.0, 39.0, 40.5, 41.5, 38.5, 42.5, 40.0],
    )


@pytest.fixture
def sample_ablation_results():
    """Create sample ablation results."""
    return [
        AblationResult(
            scenario="control",
            configuration={"cache": "none"},
            metrics={"ttft_ms": 200.0, "decode_tps": 20.0},
        ),
        AblationResult(
            scenario="ram_cache_warm",
            configuration={"cache": "ram", "state": "warm"},
            metrics={"ttft_ms": 120.0, "decode_tps": 25.0},
            improvement_over_baseline=40.0,
        ),
        AblationResult(
            scenario="disk_cache_warm",
            configuration={"cache": "disk", "state": "warm"},
            metrics={"ttft_ms": 150.0, "decode_tps": 22.0},
            improvement_over_baseline=25.0,
        ),
    ]


class TestVisualizationGeneratorInit:
    """Test VisualizationGenerator initialization."""
    
    def test_init_creates_output_directory(self, temp_output_dir):
        """Test that initialization creates visualization directory."""
        viz_gen = VisualizationGenerator(output_dir=temp_output_dir, dpi=300)
        
        assert os.path.exists(viz_gen.viz_dir)
        assert viz_gen.dpi == 300
        assert viz_gen.output_dir == temp_output_dir
    
    def test_init_with_existing_directory(self, temp_output_dir):
        """Test initialization with existing directory."""
        viz_dir = os.path.join(temp_output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        viz_gen = VisualizationGenerator(output_dir=temp_output_dir)
        assert os.path.exists(viz_gen.viz_dir)
    
    def test_default_dpi(self, temp_output_dir):
        """Test default DPI value."""
        viz_gen = VisualizationGenerator(output_dir=temp_output_dir)
        assert viz_gen.dpi == 300


class TestQuantizationComparison:
    """Test quantization comparison chart generation."""
    
    def test_plot_quantization_comparison_creates_file(
        self, viz_generator, sample_quantization_results
    ):
        """Test that quantization comparison creates PNG file."""
        output_path = viz_generator.plot_quantization_comparison(sample_quantization_results)
        
        assert output_path != ""
        assert os.path.exists(output_path)
        assert output_path.endswith(".png")
        assert "quantization_comparison" in output_path
    
    def test_plot_quantization_comparison_with_error_bars(
        self, viz_generator, sample_quantization_results, sample_statistical_summaries
    ):
        """Test quantization comparison with error bars from statistical summaries."""
        output_path = viz_generator.plot_quantization_comparison(
            sample_quantization_results, sample_statistical_summaries
        )
        
        assert output_path != ""
        assert os.path.exists(output_path)
    
    def test_plot_quantization_comparison_empty_results(self, viz_generator):
        """Test handling of empty results list."""
        output_path = viz_generator.plot_quantization_comparison([])
        assert output_path == ""
    
    def test_plot_quantization_comparison_single_result(self, viz_generator):
        """Test with single quantization result."""
        single_result = [
            QuantizationResult(
                quantization="Q4_0",
                load_time_s=1.5,
                peak_ram_mb=2000.0,
                ram_increase_mb=1500.0,
                ttft_ms=150.0,
                prefill_tps=450.0,
                decode_tps=22.0,
                prompt_tokens=100,
                output_tokens=50,
            )
        ]
        
        output_path = viz_generator.plot_quantization_comparison(single_result)
        assert output_path != ""
        assert os.path.exists(output_path)


class TestThroughputOverTime:
    """Test throughput over time plot generation."""
    
    def test_plot_throughput_over_time_creates_file(
        self, viz_generator, sample_inference_metrics
    ):
        """Test that throughput plot creates PNG file."""
        output_path = viz_generator.plot_throughput_over_time(sample_inference_metrics)
        
        assert output_path != ""
        assert os.path.exists(output_path)
        assert output_path.endswith(".png")
        assert "throughput_over_time" in output_path
    
    def test_plot_throughput_over_time_empty_latency(self, viz_generator):
        """Test handling of empty per-token latency data."""
        metrics = InferenceMetrics(
            ttft_ms=150.0,
            prefill_tps=500.0,
            decode_tps=25.0,
            total_time_s=5.0,
            prompt_tokens=100,
            output_tokens=50,
            peak_memory_mb=4000.0,
            per_token_latency_ms=[],
        )
        
        output_path = viz_generator.plot_throughput_over_time(metrics)
        assert output_path == ""
    
    def test_plot_throughput_over_time_few_tokens(self, viz_generator):
        """Test with few tokens (no moving average)."""
        metrics = InferenceMetrics(
            ttft_ms=150.0,
            prefill_tps=500.0,
            decode_tps=25.0,
            total_time_s=1.0,
            prompt_tokens=100,
            output_tokens=3,
            peak_memory_mb=4000.0,
            per_token_latency_ms=[40.0, 42.0, 38.0],
        )
        
        output_path = viz_generator.plot_throughput_over_time(metrics)
        assert output_path != ""
        assert os.path.exists(output_path)


class TestMemoryVsSpeedTradeoff:
    """Test memory vs speed tradeoff plot generation."""
    
    def test_plot_memory_vs_speed_creates_file(
        self, viz_generator, sample_quantization_results
    ):
        """Test that memory vs speed plot creates PNG file."""
        output_path = viz_generator.plot_memory_vs_speed_tradeoff(sample_quantization_results)
        
        assert output_path != ""
        assert os.path.exists(output_path)
        assert output_path.endswith(".png")
        assert "memory_vs_speed_tradeoff" in output_path
    
    def test_plot_memory_vs_speed_with_error_bars(
        self, viz_generator, sample_quantization_results, sample_statistical_summaries
    ):
        """Test memory vs speed plot with error bars."""
        output_path = viz_generator.plot_memory_vs_speed_tradeoff(
            sample_quantization_results, sample_statistical_summaries
        )
        
        assert output_path != ""
        assert os.path.exists(output_path)
    
    def test_plot_memory_vs_speed_empty_results(self, viz_generator):
        """Test handling of empty results list."""
        output_path = viz_generator.plot_memory_vs_speed_tradeoff([])
        assert output_path == ""


class TestHeatmap:
    """Test heatmap generation."""
    
    def test_plot_heatmap_creates_file(self, viz_generator):
        """Test that heatmap creates PNG file."""
        # Create sample DataFrame
        data = {
            'hardware': ['x86', 'x86', 'jetson', 'jetson'],
            'quantization': ['Q8_0', 'Q4_0', 'Q8_0', 'Q4_0'],
            'decode_tps': [25.0, 22.0, 18.0, 16.0],
        }
        df = pd.DataFrame(data)
        
        output_path = viz_generator.plot_heatmap(
            df, x_col='quantization', y_col='hardware', 
            value_col='decode_tps', title='Performance Heatmap'
        )
        
        assert output_path != ""
        assert os.path.exists(output_path)
        assert output_path.endswith(".png")
        assert "performance_heatmap" in output_path
    
    def test_plot_heatmap_empty_dataframe(self, viz_generator):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        output_path = viz_generator.plot_heatmap(
            df, x_col='x', y_col='y', value_col='value'
        )
        assert output_path == ""
    
    def test_plot_heatmap_invalid_columns(self, viz_generator):
        """Test handling of invalid column names."""
        data = {'a': [1, 2], 'b': [3, 4]}
        df = pd.DataFrame(data)
        
        output_path = viz_generator.plot_heatmap(
            df, x_col='nonexistent', y_col='b', value_col='a'
        )
        assert output_path == ""


class TestAblationComparison:
    """Test ablation comparison chart generation."""
    
    def test_plot_ablation_comparison_creates_file(
        self, viz_generator, sample_ablation_results
    ):
        """Test that ablation comparison creates PNG file."""
        output_path = viz_generator.plot_ablation_comparison(sample_ablation_results)
        
        assert output_path != ""
        assert os.path.exists(output_path)
        assert output_path.endswith(".png")
        assert "ablation_comparison" in output_path
    
    def test_plot_ablation_comparison_with_baseline(
        self, viz_generator, sample_ablation_results
    ):
        """Test ablation comparison with specified baseline."""
        output_path = viz_generator.plot_ablation_comparison(
            sample_ablation_results, baseline_scenario="control"
        )
        
        assert output_path != ""
        assert os.path.exists(output_path)
    
    def test_plot_ablation_comparison_empty_results(self, viz_generator):
        """Test handling of empty results list."""
        output_path = viz_generator.plot_ablation_comparison([])
        assert output_path == ""
    
    def test_plot_ablation_comparison_no_baseline_match(self, viz_generator):
        """Test when baseline scenario is not found."""
        results = [
            AblationResult(
                scenario="test1",
                configuration={},
                metrics={"ttft_ms": 200.0},
            ),
            AblationResult(
                scenario="test2",
                configuration={},
                metrics={"ttft_ms": 180.0},
            ),
        ]
        
        output_path = viz_generator.plot_ablation_comparison(
            results, baseline_scenario="nonexistent"
        )
        # Should use first result as baseline
        assert output_path != ""
        assert os.path.exists(output_path)


class TestErrorBarExtraction:
    """Test error bar extraction from statistical summaries."""
    
    def test_extract_error_bars_with_summaries(
        self, viz_generator, sample_statistical_summaries
    ):
        """Test extracting error bars from statistical summaries."""
        quantizations = ["Q8_0", "Q4_0"]
        errors = viz_generator._extract_error_bars(
            sample_statistical_summaries, "ttft_ms", quantizations
        )
        
        assert errors is not None
        assert len(errors) == 2
        assert errors[0] == 5.0  # (155 - 145) / 2
        assert errors[1] == 8.0  # (188 - 172) / 2
    
    def test_extract_error_bars_no_summaries(self, viz_generator):
        """Test error bar extraction with no summaries."""
        errors = viz_generator._extract_error_bars(None, "ttft_ms", ["Q8_0"])
        assert errors is None
    
    def test_extract_error_bars_missing_metric(
        self, viz_generator, sample_statistical_summaries
    ):
        """Test error bar extraction for missing metric."""
        quantizations = ["Q8_0", "Q4_0"]
        errors = viz_generator._extract_error_bars(
            sample_statistical_summaries, "nonexistent_metric", quantizations
        )
        
        # Should return list of zeros or None
        assert errors is None or all(e == 0 for e in errors)


class TestGenerateAllVisualizations:
    """Test generating all visualizations at once."""
    
    def test_generate_all_visualizations(
        self,
        viz_generator,
        sample_quantization_results,
        sample_ablation_results,
        sample_inference_metrics,
        sample_statistical_summaries,
    ):
        """Test generating all visualizations."""
        paths = viz_generator.generate_all_visualizations(
            quantization_results=sample_quantization_results,
            ablation_results=sample_ablation_results,
            statistical_summaries=sample_statistical_summaries,
            sample_metrics=sample_inference_metrics,
        )
        
        assert len(paths) > 0
        for path in paths:
            assert os.path.exists(path)
            assert path.endswith(".png")
    
    def test_generate_all_visualizations_partial_data(
        self, viz_generator, sample_quantization_results
    ):
        """Test generating visualizations with partial data."""
        paths = viz_generator.generate_all_visualizations(
            quantization_results=sample_quantization_results,
            ablation_results=[],
            statistical_summaries=None,
            sample_metrics=None,
        )
        
        # Should still generate some visualizations
        assert len(paths) > 0
    
    def test_generate_all_visualizations_empty_data(self, viz_generator):
        """Test generating visualizations with no data."""
        paths = viz_generator.generate_all_visualizations(
            quantization_results=[],
            ablation_results=[],
            statistical_summaries=None,
            sample_metrics=None,
        )
        
        assert len(paths) == 0


class TestFileOutput:
    """Test file output properties."""
    
    def test_output_file_dpi(self, temp_output_dir, sample_quantization_results):
        """Test that output files respect DPI setting."""
        viz_gen = VisualizationGenerator(output_dir=temp_output_dir, dpi=150)
        output_path = viz_gen.plot_quantization_comparison(sample_quantization_results)
        
        assert os.path.exists(output_path)
        # File should exist and be non-empty
        assert os.path.getsize(output_path) > 0
    
    def test_output_file_naming(self, viz_generator, sample_quantization_results):
        """Test output file naming conventions."""
        output_path = viz_generator.plot_quantization_comparison(sample_quantization_results)
        
        assert "quantization_comparison.png" in output_path
        assert viz_generator.viz_dir in output_path
    
    def test_multiple_plots_no_overwrite(
        self, viz_generator, sample_quantization_results
    ):
        """Test that multiple plots don't overwrite each other."""
        path1 = viz_generator.plot_quantization_comparison(sample_quantization_results)
        path2 = viz_generator.plot_memory_vs_speed_tradeoff(sample_quantization_results)
        
        assert path1 != path2
        assert os.path.exists(path1)
        assert os.path.exists(path2)
