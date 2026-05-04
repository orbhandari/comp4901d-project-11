"""
Test HTML report generation functionality.
"""

import os
import tempfile
import shutil
from pathlib import Path

from llm_benchmark.visualization.visualization_generator import VisualizationGenerator
from llm_benchmark.models import (
    BenchmarkRun,
    HardwareInfo,
    QuantizationResult,
    AblationResult,
    StatisticalSummary,
    ComparisonResult,
)


def test_html_report_generation():
    """Test that HTML report is generated with all sections."""
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create visualization generator
        viz_gen = VisualizationGenerator(output_dir=temp_dir, dpi=100)
        
        # Create sample hardware info
        hardware_info = HardwareInfo(
            os_type="linux_x86",
            cpu_model="Intel Core i7-9700K",
            cpu_cores=8,
            cpu_features=["avx2", "sse4_2"],
            total_ram_gb=32.0,
            available_ram_gb=24.0,
            has_gpu=True,
            gpu_model="NVIDIA RTX 3080",
            gpu_memory_gb=10.0,
            gpu_compute_capability="8.6",
            has_thermal_sensors=True,
            has_power_sensors=False,
        )
        
        # Create sample quantization results
        quant_results = [
            QuantizationResult(
                quantization="Q8_0",
                load_time_s=2.5,
                peak_ram_mb=4096.0,
                ram_increase_mb=3500.0,
                ttft_ms=150.0,
                prefill_tps=250.0,
                decode_tps=45.0,
                prompt_tokens=100,
                output_tokens=50,
                gpu_memory_mb=2048.0,
                gpu_utilization_pct=85.0,
                used_gpu_acceleration=True,
            ),
            QuantizationResult(
                quantization="Q4_0",
                load_time_s=1.8,
                peak_ram_mb=2048.0,
                ram_increase_mb=1800.0,
                ttft_ms=180.0,
                prefill_tps=220.0,
                decode_tps=42.0,
                prompt_tokens=100,
                output_tokens=50,
                gpu_memory_mb=1024.0,
                gpu_utilization_pct=75.0,
                used_gpu_acceleration=True,
            ),
        ]
        
        # Create sample ablation results
        ablation_results = [
            AblationResult(
                scenario="control",
                configuration={"cache_enabled": False},
                metrics={"ttft_ms": 150.0, "decode_tps": 45.0},
                improvement_over_baseline=0.0,
            ),
            AblationResult(
                scenario="warm_cache",
                configuration={"cache_enabled": True, "cache_type": "ram"},
                metrics={"ttft_ms": 80.0, "decode_tps": 48.0},
                improvement_over_baseline=46.7,
            ),
        ]
        
        # Create sample statistical summaries
        statistical_summaries = [
            StatisticalSummary(
                metric_name="ttft_ms",
                mean=150.0,
                std_dev=5.0,
                confidence_interval_95=(145.0, 155.0),
                outliers=[],
            ),
        ]
        
        # Create sample comparisons
        comparisons = [
            ComparisonResult(
                metric_name="ttft_ms",
                config_a_mean=150.0,
                config_b_mean=80.0,
                difference=-70.0,
                p_value=0.001,
                is_significant=True,
            ),
        ]
        
        # Create benchmark run
        benchmark_run = BenchmarkRun(
            run_id="test_20240101_120000",
            timestamp="2024-01-01 12:00:00",
            duration_s=300.0,
            hardware_info=hardware_info,
            software_versions={
                "python": "3.10.0",
                "llama-cpp-python": "0.2.0",
                "numpy": "1.24.0",
            },
            config={
                "context_size": 2048,
                "batch_size": 512,
                "model_path": "/models/test.gguf",
            },
            model_checksums={
                "test.gguf": "abc123def456",
            },
            quantization_results=quant_results,
            ablation_results=ablation_results,
            batch_results=[],
            statistical_summaries=statistical_summaries,
            comparisons=comparisons,
            visualization_paths=[],
            html_report_path="",
        )
        
        # Generate some visualizations first
        viz_paths = []
        
        # Generate quantization comparison
        path = viz_gen.plot_quantization_comparison(quant_results, statistical_summaries)
        if path:
            viz_paths.append(path)
        
        # Generate memory vs speed tradeoff
        path = viz_gen.plot_memory_vs_speed_tradeoff(quant_results, statistical_summaries)
        if path:
            viz_paths.append(path)
        
        # Generate ablation comparison
        path = viz_gen.plot_ablation_comparison(ablation_results)
        if path:
            viz_paths.append(path)
        
        # Generate HTML report
        html_path = viz_gen.generate_html_report(benchmark_run, viz_paths)
        
        # Verify HTML file was created
        assert os.path.exists(html_path), "HTML report file should exist"
        
        # Read HTML content
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Verify key sections are present
        assert "LLM Benchmark Report" in html_content
        assert "test_20240101_120000" in html_content
        assert "Hardware Information" in html_content
        assert "Intel Core i7-9700K" in html_content
        assert "NVIDIA RTX 3080" in html_content
        assert "Quantization Results" in html_content
        assert "Q8_0" in html_content
        assert "Q4_0" in html_content
        assert "Ablation Study Results" in html_content
        assert "control" in html_content
        assert "warm_cache" in html_content
        assert "Software Versions" in html_content
        assert "python" in html_content
        assert "Configuration" in html_content
        assert "Model Checksums" in html_content
        assert "abc123def456" in html_content
        
        # Verify embedded images (base64)
        assert "data:image/png;base64," in html_content
        
        # Verify interactive elements
        assert "collapsible" in html_content
        assert "tooltip" in html_content
        
        # Verify CSS styling
        assert "<style>" in html_content
        assert "font-family" in html_content
        
        # Verify JavaScript
        assert "<script>" in html_content
        assert "addEventListener" in html_content
        
        print(f"✓ HTML report generated successfully at {html_path}")
        print(f"✓ HTML file size: {len(html_content)} bytes")
        print(f"✓ Number of visualizations embedded: {len(viz_paths)}")


def test_html_report_with_minimal_data():
    """Test HTML report generation with minimal data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        viz_gen = VisualizationGenerator(output_dir=temp_dir, dpi=100)
        
        # Create minimal benchmark run
        hardware_info = HardwareInfo(
            os_type="linux_x86",
            cpu_model="Test CPU",
            cpu_cores=4,
            cpu_features=[],
            total_ram_gb=16.0,
            available_ram_gb=12.0,
            has_gpu=False,
        )
        
        benchmark_run = BenchmarkRun(
            run_id="minimal_test",
            timestamp="2024-01-01 00:00:00",
            duration_s=10.0,
            hardware_info=hardware_info,
            software_versions={"python": "3.10.0"},
            config={},
            model_checksums={},
        )
        
        # Generate HTML report with no visualizations
        html_path = viz_gen.generate_html_report(benchmark_run, [])
        
        # Verify HTML file was created
        assert os.path.exists(html_path)
        
        # Read HTML content
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Verify basic structure
        assert "LLM Benchmark Report" in html_content
        assert "minimal_test" in html_content
        assert "Test CPU" in html_content
        
        print(f"✓ Minimal HTML report generated successfully")


if __name__ == "__main__":
    test_html_report_generation()
    test_html_report_with_minimal_data()
    print("\n✅ All HTML report tests passed!")
