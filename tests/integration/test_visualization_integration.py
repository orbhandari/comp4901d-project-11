"""
Integration tests for VisualizationGenerator.

Tests end-to-end visualization generation with realistic data.
"""

import os
import pytest
import tempfile
import shutil
import pandas as pd

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
def realistic_quantization_results():
    """Create realistic quantization results."""
    return [
        QuantizationResult(
            quantization="Q8_0",
            load_time_s=3.2,
            peak_ram_mb=5120.0,
            ram_increase_mb=4500.0,
            ttft_ms=145.3,
            prefill_tps=523.7,
            decode_tps=26.8,
            prompt_tokens=128,
            output_tokens=64,
            gpu_memory_mb=None,
            gpu_utilization_pct=None,
            used_gpu_acceleration=False,
        ),
        QuantizationResult(
            quantization="Q4_K_M",
            load_time_s=2.1,
            peak_ram_mb=3200.0,
            ram_increase_mb=2600.0,
            ttft_ms=168.5,
            prefill_tps=478.2,
            decode_tps=23.4,
            prompt_tokens=128,
            output_tokens=64,
            gpu_memory_mb=None,
            gpu_utilization_pct=None,
            used_gpu_acceleration=False,
        ),
        QuantizationResult(
            quantization="Q4_0",
            load_time_s=1.9,
            peak_ram_mb=2800.0,
            ram_increase_mb=2200.0,
            ttft_ms=182.7,
            prefill_tps=445.1,
            decode_tps=21.9,
            prompt_tokens=128,
            output_tokens=64,
            gpu_memory_mb=None,
            gpu_utilization_pct=None,
            used_gpu_acceleration=False,
        ),
        QuantizationResult(
            quantization="Q2_K",
            load_time_s=1.3,
            peak_ram_mb=1800.0,
            ram_increase_mb=1200.0,
            ttft_ms=225.4,
            prefill_tps=389.6,
            decode_tps=17.2,
            prompt_tokens=128,
            output_tokens=64,
            gpu_memory_mb=None,
            gpu_utilization_pct=None,
            used_gpu_acceleration=False,
        ),
    ]


@pytest.fixture
def realistic_statistical_summaries():
    """Create realistic statistical summaries."""
    return [
        StatisticalSummary(
            metric_name="Q8_0_ttft_ms",
            mean=145.3,
            std_dev=4.2,
            confidence_interval_95=(141.1, 149.5),
            outliers=[],
        ),
        StatisticalSummary(
            metric_name="Q4_K_M_ttft_ms",
            mean=168.5,
            std_dev=6.8,
            confidence_interval_95=(161.7, 175.3),
            outliers=[],
        ),
        StatisticalSummary(
            metric_name="Q8_0_decode_tps",
            mean=26.8,
            std_dev=0.9,
            confidence_interval_95=(25.9, 27.7),
            outliers=[],
        ),
        StatisticalSummary(
            metric_name="Q8_0_peak_ram_mb",
            mean=5120.0,
            std_dev=45.0,
            confidence_interval_95=(5075.0, 5165.0),
            outliers=[],
        ),
    ]


@pytest.fixture
def realistic_inference_metrics():
    """Create realistic inference metrics with per-token latency."""
    # Simulate 50 tokens with realistic latency variation
    latencies = [
        42.1, 41.8, 43.2, 40.9, 42.5, 41.3, 42.8, 40.7, 43.1, 41.9,
        42.3, 41.5, 42.9, 40.8, 43.4, 41.7, 42.6, 41.2, 43.0, 41.4,
        42.7, 41.1, 43.3, 41.6, 42.4, 41.0, 42.2, 41.8, 43.5, 41.3,
        42.8, 41.5, 43.1, 41.9, 42.5, 41.2, 42.9, 41.6, 43.2, 41.4,
        42.6, 41.3, 43.0, 41.7, 42.3, 41.1, 42.7, 41.5, 43.4, 41.8,
    ]
    
    return InferenceMetrics(
        ttft_ms=145.3,
        prefill_tps=523.7,
        decode_tps=26.8,
        total_time_s=6.8,
        prompt_tokens=128,
        output_tokens=50,
        peak_memory_mb=5120.0,
        per_token_latency_ms=latencies,
        gpu_memory_mb=None,
        gpu_utilization_pct=None,
        used_gpu_acceleration=False,
    )


@pytest.fixture
def realistic_ablation_results():
    """Create realistic ablation study results."""
    return [
        AblationResult(
            scenario="control_no_cache",
            configuration={"cache_type": "none", "cache_state": "n/a"},
            metrics={
                "ttft_ms": 198.5,
                "decode_tps": 22.3,
                "memory_mb": 3200.0,
            },
            improvement_over_baseline=None,
        ),
        AblationResult(
            scenario="ram_cache_cold",
            configuration={"cache_type": "ram", "cache_state": "cold"},
            metrics={
                "ttft_ms": 195.2,
                "decode_tps": 22.5,
                "memory_mb": 3450.0,
            },
            improvement_over_baseline=1.7,
        ),
        AblationResult(
            scenario="ram_cache_warm",
            configuration={"cache_type": "ram", "cache_state": "warm"},
            metrics={
                "ttft_ms": 118.7,
                "decode_tps": 28.9,
                "memory_mb": 3450.0,
            },
            improvement_over_baseline=40.2,
        ),
        AblationResult(
            scenario="disk_cache_cold",
            configuration={"cache_type": "disk", "cache_state": "cold"},
            metrics={
                "ttft_ms": 203.4,
                "decode_tps": 21.8,
                "memory_mb": 3250.0,
            },
            improvement_over_baseline=-2.5,
        ),
        AblationResult(
            scenario="disk_cache_warm",
            configuration={"cache_type": "disk", "cache_state": "warm"},
            metrics={
                "ttft_ms": 152.3,
                "decode_tps": 25.4,
                "memory_mb": 3250.0,
            },
            improvement_over_baseline=23.3,
        ),
    ]


class TestVisualizationIntegration:
    """Integration tests for complete visualization workflow."""
    
    def test_generate_complete_visualization_suite(
        self,
        temp_output_dir,
        realistic_quantization_results,
        realistic_statistical_summaries,
        realistic_inference_metrics,
        realistic_ablation_results,
    ):
        """Test generating complete suite of visualizations."""
        viz_gen = VisualizationGenerator(output_dir=temp_output_dir, dpi=300)
        
        # Generate all visualizations
        paths = viz_gen.generate_all_visualizations(
            quantization_results=realistic_quantization_results,
            ablation_results=realistic_ablation_results,
            statistical_summaries=realistic_statistical_summaries,
            sample_metrics=realistic_inference_metrics,
        )
        
        # Verify all expected visualizations were created
        assert len(paths) == 4  # quantization, memory_vs_speed, throughput, ablation
        
        # Verify all files exist and are non-empty
        for path in paths:
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
            assert path.endswith(".png")
        
        # Verify visualization directory structure
        viz_dir = os.path.join(temp_output_dir, "visualizations")
        assert os.path.exists(viz_dir)
        
        # Check for specific files
        expected_files = [
            "quantization_comparison.png",
            "memory_vs_speed_tradeoff.png",
            "throughput_over_time.png",
            "ablation_comparison.png",
        ]
        
        for filename in expected_files:
            filepath = os.path.join(viz_dir, filename)
            assert os.path.exists(filepath)
    
    def test_quantization_comparison_with_confidence_intervals(
        self,
        temp_output_dir,
        realistic_quantization_results,
        realistic_statistical_summaries,
    ):
        """Test quantization comparison includes confidence intervals."""
        viz_gen = VisualizationGenerator(output_dir=temp_output_dir, dpi=300)
        
        output_path = viz_gen.plot_quantization_comparison(
            realistic_quantization_results,
            realistic_statistical_summaries,
        )
        
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
    
    def test_heatmap_with_realistic_data(self, temp_output_dir):
        """Test heatmap generation with realistic benchmark data."""
        viz_gen = VisualizationGenerator(output_dir=temp_output_dir, dpi=300)
        
        # Create realistic heatmap data
        data = {
            'hardware': ['x86_cpu', 'x86_cpu', 'x86_cpu', 'x86_cpu',
                        'jetson_nx', 'jetson_nx', 'jetson_nx', 'jetson_nx'],
            'quantization': ['Q8_0', 'Q4_K_M', 'Q4_0', 'Q2_K',
                           'Q8_0', 'Q4_K_M', 'Q4_0', 'Q2_K'],
            'decode_tps': [26.8, 23.4, 21.9, 17.2,
                          32.5, 28.7, 26.3, 21.8],
        }
        df = pd.DataFrame(data)
        
        output_path = viz_gen.plot_heatmap(
            df,
            x_col='quantization',
            y_col='hardware',
            value_col='decode_tps',
            title='Hardware vs Quantization Performance',
        )
        
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        assert "hardware_vs_quantization_performance" in output_path
    
    def test_throughput_plot_with_many_tokens(self, temp_output_dir):
        """Test throughput plot with realistic token count."""
        viz_gen = VisualizationGenerator(output_dir=temp_output_dir, dpi=300)
        
        # Generate 100 tokens with realistic latency variation
        import random
        random.seed(42)
        latencies = [40 + random.gauss(0, 2) for _ in range(100)]
        
        metrics = InferenceMetrics(
            ttft_ms=145.3,
            prefill_tps=523.7,
            decode_tps=26.8,
            total_time_s=10.5,
            prompt_tokens=128,
            output_tokens=100,
            peak_memory_mb=5120.0,
            per_token_latency_ms=latencies,
        )
        
        output_path = viz_gen.plot_throughput_over_time(metrics)
        
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
    
    def test_ablation_comparison_multiple_metrics(
        self, temp_output_dir, realistic_ablation_results
    ):
        """Test ablation comparison with multiple metrics."""
        viz_gen = VisualizationGenerator(output_dir=temp_output_dir, dpi=300)
        
        output_path = viz_gen.plot_ablation_comparison(
            realistic_ablation_results,
            baseline_scenario="control",
        )
        
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
    
    def test_high_dpi_output(
        self, temp_output_dir, realistic_quantization_results
    ):
        """Test high DPI output for publication quality."""
        viz_gen = VisualizationGenerator(output_dir=temp_output_dir, dpi=600)
        
        output_path = viz_gen.plot_quantization_comparison(
            realistic_quantization_results
        )
        
        assert os.path.exists(output_path)
        # High DPI files should be larger
        file_size = os.path.getsize(output_path)
        assert file_size > 100000  # At least 100KB for high DPI
    
    def test_visualization_directory_organization(
        self,
        temp_output_dir,
        realistic_quantization_results,
        realistic_ablation_results,
    ):
        """Test that visualizations are properly organized in directory."""
        viz_gen = VisualizationGenerator(output_dir=temp_output_dir, dpi=300)
        
        # Generate multiple visualizations
        viz_gen.plot_quantization_comparison(realistic_quantization_results)
        viz_gen.plot_memory_vs_speed_tradeoff(realistic_quantization_results)
        viz_gen.plot_ablation_comparison(realistic_ablation_results)
        
        # Check directory structure
        viz_dir = os.path.join(temp_output_dir, "visualizations")
        assert os.path.exists(viz_dir)
        
        # Count PNG files
        png_files = [f for f in os.listdir(viz_dir) if f.endswith('.png')]
        assert len(png_files) == 3
        
        # Verify no files in parent directory
        parent_files = [f for f in os.listdir(temp_output_dir) 
                       if f.endswith('.png')]
        assert len(parent_files) == 0


class TestVisualizationEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_quantization_level(self, temp_output_dir):
        """Test visualization with single quantization level."""
        viz_gen = VisualizationGenerator(output_dir=temp_output_dir, dpi=300)
        
        single_result = [
            QuantizationResult(
                quantization="Q4_0",
                load_time_s=2.0,
                peak_ram_mb=3000.0,
                ram_increase_mb=2500.0,
                ttft_ms=170.0,
                prefill_tps=450.0,
                decode_tps=22.0,
                prompt_tokens=128,
                output_tokens=64,
            )
        ]
        
        paths = viz_gen.generate_all_visualizations(
            quantization_results=single_result,
            ablation_results=[],
            statistical_summaries=None,
            sample_metrics=None,
        )
        
        assert len(paths) > 0
        for path in paths:
            assert os.path.exists(path)
    
    def test_zero_latency_handling(self, temp_output_dir):
        """Test handling of zero latency values."""
        viz_gen = VisualizationGenerator(output_dir=temp_output_dir, dpi=300)
        
        # Include some zero latencies (edge case)
        metrics = InferenceMetrics(
            ttft_ms=150.0,
            prefill_tps=500.0,
            decode_tps=25.0,
            total_time_s=5.0,
            prompt_tokens=100,
            output_tokens=10,
            peak_memory_mb=4000.0,
            per_token_latency_ms=[40.0, 0.0, 42.0, 0.0, 38.0, 41.0, 39.0, 0.0, 42.5, 40.0],
        )
        
        output_path = viz_gen.plot_throughput_over_time(metrics)
        
        # Should handle zeros gracefully
        assert output_path != ""
        assert os.path.exists(output_path)
    
    def test_large_dataset_performance(self, temp_output_dir):
        """Test performance with large dataset."""
        viz_gen = VisualizationGenerator(output_dir=temp_output_dir, dpi=300)
        
        # Generate large dataset (1000 tokens)
        import random
        random.seed(42)
        latencies = [40 + random.gauss(0, 3) for _ in range(1000)]
        
        metrics = InferenceMetrics(
            ttft_ms=145.3,
            prefill_tps=523.7,
            decode_tps=26.8,
            total_time_s=45.0,
            prompt_tokens=128,
            output_tokens=1000,
            peak_memory_mb=5120.0,
            per_token_latency_ms=latencies,
        )
        
        output_path = viz_gen.plot_throughput_over_time(metrics)
        
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
