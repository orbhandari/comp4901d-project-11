"""
Integration tests for visualization generation.

**Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.8**

Tests visualization generation with sample data to ensure all chart types
and HTML reports are generated correctly.
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
    BenchmarkRun,
    HardwareInfo,
)


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_quantization_results():
    """Create sample quantization results for testing."""
    return [
        QuantizationResult(
            quantization="Q8_0",
            load_time_s=2.5,
            peak_ram_mb=5000.0,
            ram_increase_mb=4500.0,
            ttft_ms=150.0,
            prefill_tps=45.0,
            decode_tps=25.0,
            prompt_tokens=100,
            output_tokens=50,
            gpu_memory_mb=None,
            gpu_utilization_pct=None,
            used_gpu_acceleration=False
        ),
        QuantizationResult(
            quantization="Q4_0",
            load_time_s=1.8,
            peak_ram_mb=3000.0,
            ram_increase_mb=2500.0,
            ttft_ms=180.0,
            prefill_tps=40.0,
            decode_tps=22.0,
            prompt_tokens=100,
            output_tokens=50,
            gpu_memory_mb=None,
            gpu_utilization_pct=None,
            used_gpu_acceleration=False
        ),
        QuantizationResult(
            quantization="Q4_K_M",
            load_time_s=2.0,
            peak_ram_mb=3200.0,
            ram_increase_mb=2700.0,
            ttft_ms=170.0,
            prefill_tps=42.0,
            decode_tps=23.0,
            prompt_tokens=100,
            output_tokens=50,
            gpu_memory_mb=None,
            gpu_utilization_pct=None,
            used_gpu_acceleration=False
        ),
    ]


@pytest.fixture
def sample_inference_metrics():
    """Create sample inference metrics for testing."""
    return InferenceMetrics(
        ttft_ms=150.0,
        prefill_tps=45.0,
        decode_tps=25.0,
        total_time_s=5.0,
        prompt_tokens=100,
        output_tokens=50,
        peak_memory_mb=5000.0,
        per_token_latency_ms=[40.0, 38.0, 42.0, 39.0, 41.0, 40.0, 38.0, 39.0, 41.0, 40.0],
        gpu_memory_mb=None,
        gpu_utilization_pct=None,
        used_gpu_acceleration=False,
        cpu_temp_c=None,
        gpu_temp_c=None,
        power_watts=None
    )


@pytest.fixture
def sample_ablation_results():
    """Create sample ablation results for testing."""
    return [
        AblationResult(
            scenario="control",
            configuration={"cache": "none"},
            metrics={"ttft_ms": 150.0, "decode_tps": 25.0},
            improvement_over_baseline=0.0
        ),
        AblationResult(
            scenario="ram_cache_warm",
            configuration={"cache": "ram", "state": "warm"},
            metrics={"ttft_ms": 100.0, "decode_tps": 30.0},
            improvement_over_baseline=33.3
        ),
        AblationResult(
            scenario="disk_cache_warm",
            configuration={"cache": "disk", "state": "warm"},
            metrics={"ttft_ms": 120.0, "decode_tps": 28.0},
            improvement_over_baseline=20.0
        ),
    ]


@pytest.fixture
def sample_statistical_summaries():
    """Create sample statistical summaries for testing."""
    return [
        StatisticalSummary(
            metric_name="Q8_0_ttft_ms",
            mean=150.0,
            std_dev=5.0,
            confidence_interval_95=(145.0, 155.0),
            outliers=[]
        ),
        StatisticalSummary(
            metric_name="Q4_0_ttft_ms",
            mean=180.0,
            std_dev=6.0,
            confidence_interval_95=(174.0, 186.0),
            outliers=[]
        ),
    ]


class TestVisualizationGeneration:
    """Integration tests for visualization generation."""
    
    def test_bar_chart_generation(self, temp_output_dir, sample_quantization_results):
        """
        Test bar chart generation with sample data.
        
        **Validates: Requirements 9.1**
        """
        viz_gen = VisualizationGenerator(temp_output_dir, dpi=100)
        
        output_path = viz_gen.plot_quantization_comparison(sample_quantization_results)
        
        assert output_path != ""
        assert os.path.exists(output_path)
        assert output_path.endswith("quantization_comparison.png")
        
        # Check file size (should be non-zero)
        file_size = os.path.getsize(output_path)
        assert file_size > 0
    
    def test_bar_chart_with_error_bars(
        self, 
        temp_output_dir, 
        sample_quantization_results,
        sample_statistical_summaries
    ):
        """
        Test bar chart generation with error bars from statistical summaries.
        
        **Validates: Requirements 9.1, 9.6**
        """
        viz_gen = VisualizationGenerator(temp_output_dir, dpi=100)
        
        output_path = viz_gen.plot_quantization_comparison(
            sample_quantization_results,
            sample_statistical_summaries
        )
        
        assert output_path != ""
        assert os.path.exists(output_path)
    
    def test_line_plot_generation(self, temp_output_dir, sample_inference_metrics):
        """
        Test line plot generation with sample data.
        
        **Validates: Requirements 9.2**
        """
        viz_gen = VisualizationGenerator(temp_output_dir, dpi=100)
        
        output_path = viz_gen.plot_throughput_over_time(sample_inference_metrics)
        
        assert output_path != ""
        assert os.path.exists(output_path)
        assert output_path.endswith("throughput_over_time.png")
        
        # Check file size
        file_size = os.path.getsize(output_path)
        assert file_size > 0
    
    def test_scatter_plot_generation(self, temp_output_dir, sample_quantization_results):
        """
        Test scatter plot generation with sample data.
        
        **Validates: Requirements 9.3**
        """
        viz_gen = VisualizationGenerator(temp_output_dir, dpi=100)
        
        output_path = viz_gen.plot_memory_vs_speed_tradeoff(sample_quantization_results)
        
        assert output_path != ""
        assert os.path.exists(output_path)
        assert output_path.endswith("memory_vs_speed_tradeoff.png")
        
        # Check file size
        file_size = os.path.getsize(output_path)
        assert file_size > 0
    
    def test_heatmap_generation(self, temp_output_dir):
        """
        Test heatmap generation with sample data.
        
        **Validates: Requirements 9.4**
        """
        viz_gen = VisualizationGenerator(temp_output_dir, dpi=100)
        
        # Create sample DataFrame for heatmap
        data = {
            'quantization': ['Q8_0', 'Q8_0', 'Q4_0', 'Q4_0'],
            'batch_size': [1, 2, 1, 2],
            'throughput': [25.0, 45.0, 22.0, 40.0]
        }
        df = pd.DataFrame(data)
        
        output_path = viz_gen.plot_heatmap(
            df,
            x_col='batch_size',
            y_col='quantization',
            value_col='throughput',
            title='Throughput Heatmap'
        )
        
        assert output_path != ""
        assert os.path.exists(output_path)
        assert output_path.endswith("throughput_heatmap.png")
        
        # Check file size
        file_size = os.path.getsize(output_path)
        assert file_size > 0
    
    def test_ablation_comparison_chart(self, temp_output_dir, sample_ablation_results):
        """
        Test ablation comparison chart generation.
        
        **Validates: Requirements 9.5**
        """
        viz_gen = VisualizationGenerator(temp_output_dir, dpi=100)
        
        output_path = viz_gen.plot_ablation_comparison(sample_ablation_results)
        
        assert output_path != ""
        assert os.path.exists(output_path)
        assert output_path.endswith("ablation_comparison.png")
        
        # Check file size
        file_size = os.path.getsize(output_path)
        assert file_size > 0
    
    def test_html_report_generation(
        self,
        temp_output_dir,
        sample_quantization_results,
        sample_ablation_results,
        sample_statistical_summaries
    ):
        """
        Test HTML report generation.
        
        **Validates: Requirements 9.8**
        """
        viz_gen = VisualizationGenerator(temp_output_dir, dpi=100)
        
        # Generate visualizations first
        viz_paths = []
        viz_paths.append(viz_gen.plot_quantization_comparison(sample_quantization_results))
        viz_paths.append(viz_gen.plot_memory_vs_speed_tradeoff(sample_quantization_results))
        viz_paths.append(viz_gen.plot_ablation_comparison(sample_ablation_results))
        
        # Create a sample benchmark run
        hardware_info = HardwareInfo(
            os_type="linux_x86",
            cpu_model="Intel Core i7-9700K",
            cpu_cores=8,
            cpu_features=["avx2", "sse4_2"],
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            has_gpu=False,
            gpu_model=None,
            gpu_memory_gb=None,
            gpu_compute_capability=None,
            has_thermal_sensors=True,
            has_power_sensors=False
        )
        
        benchmark_run = BenchmarkRun(
            run_id="test_run_001",
            timestamp="2024-01-15T14:30:00",
            duration_s=120.0,
            hardware_info=hardware_info,
            software_versions={"python": "3.10.0", "llama-cpp-python": "0.2.0"},
            config={"context_size": 2048, "batch_size": 512},
            model_checksums={"model.gguf": "abc123def456"},
            quantization_results=sample_quantization_results,
            ablation_results=sample_ablation_results,
            batch_results=[],
            statistical_summaries=sample_statistical_summaries,
            comparisons=[],
            visualization_paths=viz_paths,
            html_report_path=""
        )
        
        # Generate HTML report
        html_path = viz_gen.generate_html_report(benchmark_run, viz_paths)
        
        assert html_path != ""
        assert os.path.exists(html_path)
        assert html_path.endswith("benchmark_report.html")
        
        # Check file size
        file_size = os.path.getsize(html_path)
        assert file_size > 0
        
        # Check HTML content
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Verify key sections are present
        assert "LLM Benchmark Report" in html_content
        assert "test_run_001" in html_content
        assert "Hardware Information" in html_content
        assert "Quantization Results" in html_content
        assert "Ablation Study Results" in html_content
        assert "Intel Core i7-9700K" in html_content
        
        # Verify embedded images
        assert "data:image/png;base64," in html_content
    
    def test_empty_results_handling(self, temp_output_dir):
        """
        Test that visualization generator handles empty results gracefully.
        
        **Validates: Requirements 9.1, 9.2, 9.3, 9.4**
        """
        viz_gen = VisualizationGenerator(temp_output_dir, dpi=100)
        
        # Test with empty quantization results
        output_path = viz_gen.plot_quantization_comparison([])
        assert output_path == ""
        
        # Test with empty ablation results
        output_path = viz_gen.plot_ablation_comparison([])
        assert output_path == ""
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        output_path = viz_gen.plot_heatmap(empty_df, 'x', 'y', 'value')
        assert output_path == ""
    
    def test_visualization_directory_creation(self, temp_output_dir):
        """
        Test that visualization directory is created automatically.
        
        **Validates: Requirements 9.7**
        """
        viz_gen = VisualizationGenerator(temp_output_dir, dpi=100)
        
        viz_dir = os.path.join(temp_output_dir, "visualizations")
        assert os.path.exists(viz_dir)
        assert os.path.isdir(viz_dir)
    
    def test_custom_dpi_setting(self, temp_output_dir, sample_quantization_results):
        """
        Test that custom DPI setting is respected.
        
        **Validates: Requirements 9.7**
        """
        # Test with minimum required DPI (300)
        viz_gen = VisualizationGenerator(temp_output_dir, dpi=300)
        
        output_path = viz_gen.plot_quantization_comparison(sample_quantization_results)
        
        assert output_path != ""
        assert os.path.exists(output_path)
        
        # File should be larger with higher DPI
        file_size_300 = os.path.getsize(output_path)
        
        # Compare with lower DPI
        temp_dir2 = tempfile.mkdtemp()
        viz_gen_low = VisualizationGenerator(temp_dir2, dpi=100)
        output_path_low = viz_gen_low.plot_quantization_comparison(sample_quantization_results)
        file_size_100 = os.path.getsize(output_path_low)
        
        # Higher DPI should produce larger file
        assert file_size_300 > file_size_100
        
        # Cleanup
        shutil.rmtree(temp_dir2, ignore_errors=True)
    
    def test_generate_all_visualizations(
        self,
        temp_output_dir,
        sample_quantization_results,
        sample_ablation_results,
        sample_inference_metrics,
        sample_statistical_summaries
    ):
        """
        Test generating all visualizations at once.
        
        **Validates: Requirements 9.1, 9.2, 9.3, 9.5**
        """
        viz_gen = VisualizationGenerator(temp_output_dir, dpi=100)
        
        viz_paths = viz_gen.generate_all_visualizations(
            quantization_results=sample_quantization_results,
            ablation_results=sample_ablation_results,
            statistical_summaries=sample_statistical_summaries,
            sample_metrics=sample_inference_metrics
        )
        
        # Should generate multiple visualizations
        assert len(viz_paths) >= 3
        
        # All paths should exist
        for path in viz_paths:
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
