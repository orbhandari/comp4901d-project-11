"""
Integration test for HTML report generation with full workflow.
"""

import os
import tempfile
from pathlib import Path

from llm_benchmark.visualization.visualization_generator import VisualizationGenerator
from llm_benchmark.models import (
    BenchmarkRun,
    HardwareInfo,
    QuantizationResult,
    AblationResult,
    InferenceMetrics,
    StatisticalSummary,
    ComparisonResult,
)


def test_full_html_report_workflow():
    """Test complete workflow: generate visualizations -> create HTML report."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create visualization generator
        viz_gen = VisualizationGenerator(output_dir=temp_dir, dpi=150)
        
        # Create comprehensive hardware info
        hardware_info = HardwareInfo(
            os_type="jetson_xavier_nx",
            cpu_model="ARM Cortex-A57",
            cpu_cores=6,
            cpu_features=["neon", "fp16"],
            total_ram_gb=8.0,
            available_ram_gb=6.5,
            has_gpu=True,
            gpu_model="NVIDIA Xavier GPU",
            gpu_memory_gb=8.0,
            gpu_compute_capability="7.2",
            has_thermal_sensors=True,
            has_power_sensors=True,
        )
        
        # Create quantization results with multiple levels
        quant_results = [
            QuantizationResult(
                quantization="Q8_0",
                load_time_s=3.2,
                peak_ram_mb=5120.0,
                ram_increase_mb=4500.0,
                ttft_ms=120.0,
                prefill_tps=280.0,
                decode_tps=50.0,
                prompt_tokens=150,
                output_tokens=75,
                gpu_memory_mb=3072.0,
                gpu_utilization_pct=90.0,
                used_gpu_acceleration=True,
            ),
            QuantizationResult(
                quantization="Q4_K_M",
                load_time_s=2.1,
                peak_ram_mb=2560.0,
                ram_increase_mb=2200.0,
                ttft_ms=145.0,
                prefill_tps=240.0,
                decode_tps=46.0,
                prompt_tokens=150,
                output_tokens=75,
                gpu_memory_mb=1536.0,
                gpu_utilization_pct=82.0,
                used_gpu_acceleration=True,
            ),
            QuantizationResult(
                quantization="Q4_0",
                load_time_s=1.9,
                peak_ram_mb=2304.0,
                ram_increase_mb=2000.0,
                ttft_ms=160.0,
                prefill_tps=230.0,
                decode_tps=44.0,
                prompt_tokens=150,
                output_tokens=75,
                gpu_memory_mb=1280.0,
                gpu_utilization_pct=78.0,
                used_gpu_acceleration=True,
            ),
            QuantizationResult(
                quantization="Q2_K",
                load_time_s=1.2,
                peak_ram_mb=1536.0,
                ram_increase_mb=1300.0,
                ttft_ms=200.0,
                prefill_tps=190.0,
                decode_tps=38.0,
                prompt_tokens=150,
                output_tokens=75,
                gpu_memory_mb=768.0,
                gpu_utilization_pct=65.0,
                used_gpu_acceleration=True,
            ),
        ]
        
        # Create ablation results
        ablation_results = [
            AblationResult(
                scenario="control_no_cache",
                configuration={"cache_enabled": False, "cache_type": None},
                metrics={"ttft_ms": 120.0, "decode_tps": 50.0, "memory_mb": 5120.0},
                improvement_over_baseline=0.0,
            ),
            AblationResult(
                scenario="cold_cache_ram",
                configuration={"cache_enabled": True, "cache_type": "ram"},
                metrics={"ttft_ms": 115.0, "decode_tps": 51.0, "memory_mb": 5200.0},
                improvement_over_baseline=4.2,
            ),
            AblationResult(
                scenario="warm_cache_ram",
                configuration={"cache_enabled": True, "cache_type": "ram", "cache_populated": True},
                metrics={"ttft_ms": 65.0, "decode_tps": 52.0, "memory_mb": 5200.0},
                improvement_over_baseline=45.8,
            ),
            AblationResult(
                scenario="warm_cache_disk",
                configuration={"cache_enabled": True, "cache_type": "disk", "cache_populated": True},
                metrics={"ttft_ms": 85.0, "decode_tps": 51.5, "memory_mb": 5150.0},
                improvement_over_baseline=29.2,
            ),
        ]
        
        # Create batch processing results
        batch_results = [
            AblationResult(
                scenario="batch_size_1",
                configuration={"batch_size": 1},
                metrics={"throughput_tps": 50.0, "latency_ms": 120.0, "memory_mb": 5120.0},
                improvement_over_baseline=0.0,
            ),
            AblationResult(
                scenario="batch_size_4",
                configuration={"batch_size": 4},
                metrics={"throughput_tps": 180.0, "latency_ms": 135.0, "memory_mb": 5800.0},
                improvement_over_baseline=260.0,
            ),
            AblationResult(
                scenario="batch_size_8",
                configuration={"batch_size": 8},
                metrics={"throughput_tps": 320.0, "latency_ms": 150.0, "memory_mb": 6500.0},
                improvement_over_baseline=540.0,
            ),
        ]
        
        # Create statistical summaries
        statistical_summaries = [
            StatisticalSummary(
                metric_name="Q8_0_ttft_ms",
                mean=120.0,
                std_dev=3.5,
                confidence_interval_95=(116.5, 123.5),
                outliers=[],
            ),
            StatisticalSummary(
                metric_name="Q4_K_M_ttft_ms",
                mean=145.0,
                std_dev=4.2,
                confidence_interval_95=(140.8, 149.2),
                outliers=[],
            ),
            StatisticalSummary(
                metric_name="Q8_0_decode_tps",
                mean=50.0,
                std_dev=1.2,
                confidence_interval_95=(48.8, 51.2),
                outliers=[],
            ),
        ]
        
        # Create comparisons
        comparisons = [
            ComparisonResult(
                metric_name="ttft_ms_control_vs_warm_cache",
                config_a_mean=120.0,
                config_b_mean=65.0,
                difference=-55.0,
                p_value=0.0001,
                is_significant=True,
            ),
            ComparisonResult(
                metric_name="decode_tps_Q8_vs_Q4",
                config_a_mean=50.0,
                config_b_mean=44.0,
                difference=-6.0,
                p_value=0.003,
                is_significant=True,
            ),
        ]
        
        # Create sample inference metrics for throughput plot
        sample_metrics = InferenceMetrics(
            ttft_ms=120.0,
            prefill_tps=280.0,
            decode_tps=50.0,
            total_time_s=5.5,
            prompt_tokens=150,
            output_tokens=75,
            peak_memory_mb=5120.0,
            per_token_latency_ms=[20.0, 19.5, 20.5, 19.8, 20.2, 19.9, 20.1, 20.3, 19.7, 20.0] * 7 + [20.0, 19.5, 20.5, 19.8, 20.2],
            gpu_memory_mb=3072.0,
            gpu_utilization_pct=90.0,
            used_gpu_acceleration=True,
            cpu_temp_c=65.0,
            gpu_temp_c=72.0,
            power_watts=15.5,
            cpu_temp_stats=(60.0, 65.0, 70.0),
            gpu_temp_stats=(68.0, 72.0, 76.0),
            power_stats=(14.0, 15.5, 17.0),
            thermal_throttled=False,
        )
        
        # Create benchmark run
        benchmark_run = BenchmarkRun(
            run_id="integration_test_20240115_143000",
            timestamp="2024-01-15 14:30:00",
            duration_s=1800.0,
            hardware_info=hardware_info,
            software_versions={
                "python": "3.10.12",
                "llama-cpp-python": "0.2.27",
                "numpy": "1.24.3",
                "pandas": "2.0.3",
                "matplotlib": "3.7.2",
                "cuda_driver": "11.8",
            },
            config={
                "context_size": 2048,
                "batch_size": 512,
                "n_gpu_layers": 35,
                "model_repo": "TheBloke/Llama-2-7B-GGUF",
                "test_prompt": "Explain quantum computing in simple terms.",
                "warmup_runs": 2,
                "measurement_runs": 5,
            },
            model_checksums={
                "llama-2-7b.Q8_0.gguf": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
                "llama-2-7b.Q4_K_M.gguf": "b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7",
                "llama-2-7b.Q4_0.gguf": "c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8",
                "llama-2-7b.Q2_K.gguf": "d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9",
            },
        )
        
        # Add results to benchmark run
        benchmark_run.quantization_results = quant_results
        benchmark_run.ablation_results = ablation_results
        benchmark_run.batch_results = batch_results
        benchmark_run.statistical_summaries = statistical_summaries
        benchmark_run.comparisons = comparisons
        
        # Generate all visualizations
        viz_paths = viz_gen.generate_all_visualizations(
            quantization_results=quant_results,
            ablation_results=ablation_results,
            statistical_summaries=statistical_summaries,
            sample_metrics=sample_metrics,
        )
        
        print(f"\n✓ Generated {len(viz_paths)} visualizations:")
        for path in viz_paths:
            print(f"  - {os.path.basename(path)}")
        
        # Generate HTML report
        html_path = viz_gen.generate_html_report(benchmark_run, viz_paths)
        
        # Verify HTML file exists
        assert os.path.exists(html_path), "HTML report should exist"
        
        # Read and verify HTML content
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Verify all major sections
        required_sections = [
            "LLM Benchmark Report",
            "integration_test_20240115_143000",
            "Summary Statistics",
            "Hardware Information",
            "jetson_xavier_nx",
            "NVIDIA Xavier GPU",
            "Software Versions",
            "llama-cpp-python",
            "Configuration",
            "context_size",
            "Model Checksums",
            "Quantization Results",
            "Q8_0",
            "Q4_K_M",
            "Q4_0",
            "Q2_K",
            "Ablation Study Results",
            "control_no_cache",
            "warm_cache_ram",
            "Batch Processing Results",
            "batch_size_1",
            "batch_size_8",
            "Statistical Summaries",
            "Statistical Comparisons",
            "Visualizations",
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in html_content:
                missing_sections.append(section)
        
        assert not missing_sections, f"Missing sections: {missing_sections}"
        
        # Verify embedded images
        assert html_content.count("data:image/png;base64,") >= 3, "Should have at least 3 embedded images"
        
        # Verify interactive features
        assert "collapsible" in html_content, "Should have collapsible sections"
        assert "tooltip" in html_content, "Should have tooltips"
        assert "addEventListener" in html_content, "Should have JavaScript interactivity"
        
        # Verify styling
        assert "gradient" in html_content, "Should have gradient styling"
        assert "box-shadow" in html_content, "Should have shadow effects"
        
        # Print report statistics
        print(f"\n✓ HTML Report Statistics:")
        print(f"  - File size: {len(html_content):,} bytes")
        print(f"  - Embedded images: {html_content.count('data:image/png;base64,')}")
        print(f"  - Tables: {html_content.count('<table>')}")
        print(f"  - Collapsible sections: {html_content.count('collapsible')}")
        print(f"  - Tooltips: {html_content.count('tooltip')}")
        print(f"  - Location: {html_path}")
        
        # Verify file size is reasonable (should be > 50KB with embedded images)
        assert len(html_content) > 50000, "HTML report should be substantial with embedded images"
        
        print(f"\n✅ Full HTML report workflow test passed!")
        print(f"   Report available at: {html_path}")


if __name__ == "__main__":
    test_full_html_report_workflow()
