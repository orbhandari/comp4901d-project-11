"""
Unit tests for result persistence functionality.

Tests JSON serialization, CSV export, Markdown generation, and directory structure creation.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from llm_benchmark.models import (
    AblationResult,
    BenchmarkRun,
    ComparisonResult,
    HardwareInfo,
    QuantizationResult,
    StatisticalSummary,
)
from llm_benchmark.results.persistence import ResultsPersistence


@pytest.fixture
def sample_hardware_info():
    """Create sample hardware information."""
    return HardwareInfo(
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
        has_power_sensors=False,
    )


@pytest.fixture
def sample_quantization_results():
    """Create sample quantization results."""
    return [
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
            gpu_memory_mb=None,
            gpu_utilization_pct=None,
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
            gpu_memory_mb=None,
            gpu_utilization_pct=None,
        ),
    ]


@pytest.fixture
def sample_ablation_results():
    """Create sample ablation results."""
    return [
        AblationResult(
            scenario="KV Cache - RAM Cold",
            configuration={"cache_type": "ram", "cache_state": "cold"},
            metrics={"ttft_ms": 150.0, "decode_tps": 45.0},
            improvement_over_baseline=None,
        ),
        AblationResult(
            scenario="KV Cache - RAM Warm",
            configuration={"cache_type": "ram", "cache_state": "warm"},
            metrics={"ttft_ms": 80.0, "decode_tps": 48.0},
            improvement_over_baseline=46.7,
        ),
    ]


@pytest.fixture
def sample_statistical_summaries():
    """Create sample statistical summaries."""
    return [
        StatisticalSummary(
            metric_name="ttft_ms",
            mean=150.5,
            std_dev=5.2,
            confidence_interval_95=(145.3, 155.7),
            outliers=[],
        ),
        StatisticalSummary(
            metric_name="decode_tps",
            mean=45.2,
            std_dev=1.8,
            confidence_interval_95=(43.4, 47.0),
            outliers=[52.0],
        ),
    ]


@pytest.fixture
def sample_comparisons():
    """Create sample comparison results."""
    return [
        ComparisonResult(
            metric_name="ttft_ms",
            config_a_mean=150.0,
            config_b_mean=80.0,
            difference=-70.0,
            p_value=0.001,
            is_significant=True,
        ),
    ]


@pytest.fixture
def sample_benchmark_run(
    sample_hardware_info,
    sample_quantization_results,
    sample_ablation_results,
    sample_statistical_summaries,
    sample_comparisons,
):
    """Create a complete sample benchmark run."""
    return BenchmarkRun(
        run_id="20240115_143022",
        timestamp="2024-01-15 14:30:22",
        duration_s=125.5,
        hardware_info=sample_hardware_info,
        software_versions={
            "python": "3.10.12",
            "llama-cpp-python": "0.2.20",
            "cuda_driver": "not available",
        },
        config={
            "context_size": 2048,
            "batch_size": 512,
            "max_tokens": 100,
        },
        model_checksums={
            "model_q8_0.gguf": "abc123def456",
            "model_q4_0.gguf": "789ghi012jkl",
        },
        quantization_results=sample_quantization_results,
        ablation_results=sample_ablation_results,
        batch_results=[],
        statistical_summaries=sample_statistical_summaries,
        comparisons=sample_comparisons,
        visualization_paths=[],
        html_report_path="",
    )


class TestResultsPersistence:
    """Test suite for ResultsPersistence class."""

    def test_create_run_directory(self):
        """Test directory structure creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            run_id = "20240115_143022"

            run_dir = persistence.create_run_directory(run_id)

            # Check main directory exists
            assert run_dir.exists()
            assert run_dir.name == f"run_{run_id}"

            # Check subdirectories exist
            assert (run_dir / "visualizations").exists()
            assert (run_dir / "logs").exists()
            assert (run_dir / "checkpoints").exists()

    def test_save_json(self, sample_benchmark_run):
        """Test JSON serialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            json_path = Path(tmpdir) / "results.json"

            persistence.save_json(sample_benchmark_run, json_path)

            # Check file exists
            assert json_path.exists()

            # Load and verify content
            with open(json_path, "r") as f:
                data = json.load(f)

            assert data["run_id"] == "20240115_143022"
            assert data["timestamp"] == "2024-01-15 14:30:22"
            assert data["duration_s"] == 125.5
            assert data["hardware_info"]["os_type"] == "linux_x86"
            assert len(data["quantization_results"]) == 2
            assert len(data["ablation_results"]) == 2

    def test_save_csv(self, sample_benchmark_run):
        """Test CSV export format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            csv_path = Path(tmpdir) / "results.csv"

            persistence.save_csv(sample_benchmark_run, csv_path)

            # Check file exists
            assert csv_path.exists()

            # Read and verify content
            with open(csv_path, "r") as f:
                content = f.read()

            # Check for key sections
            assert "# Benchmark Run Metadata" in content
            assert "20240115_143022" in content
            assert "# Quantization Results" in content
            assert "Q8_0" in content
            assert "Q4_0" in content
            assert "# Ablation Study Results" in content
            assert "# Statistical Summaries" in content
            assert "# Statistical Comparisons" in content

    def test_save_markdown(self, sample_benchmark_run):
        """Test Markdown report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            md_path = Path(tmpdir) / "results.md"

            persistence.save_markdown(sample_benchmark_run, md_path)

            # Check file exists
            assert md_path.exists()

            # Read and verify content
            with open(md_path, "r") as f:
                content = f.read()

            # Check for key sections
            assert "# Benchmark Run Report: 20240115_143022" in content
            assert "## Hardware Information" in content
            assert "Intel Core i7-9700K" in content
            assert "## Software Versions" in content
            assert "## Configuration" in content
            assert "## Model Checksums (SHA256)" in content
            assert "## Quantization Results" in content
            assert "## Ablation Study Results" in content
            assert "## Statistical Summaries" in content
            assert "## Statistical Comparisons" in content

            # Check table formatting
            assert "| Quantization | Load Time (s) |" in content
            assert "| Q8_0 |" in content
            assert "| Q4_0 |" in content

    def test_save_hardware_info(self, sample_hardware_info):
        """Test hardware info JSON export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            hw_path = Path(tmpdir) / "hardware_info.json"

            persistence.save_hardware_info(sample_hardware_info, hw_path)

            # Check file exists
            assert hw_path.exists()

            # Load and verify content
            with open(hw_path, "r") as f:
                data = json.load(f)

            assert data["os_type"] == "linux_x86"
            assert data["cpu_model"] == "Intel Core i7-9700K"
            assert data["cpu_cores"] == 8
            assert data["total_ram_gb"] == 16.0

    def test_save_config(self):
        """Test configuration JSON export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            config_path = Path(tmpdir) / "config.json"

            config = {
                "context_size": 2048,
                "batch_size": 512,
                "max_tokens": 100,
            }

            persistence.save_config(config, config_path)

            # Check file exists
            assert config_path.exists()

            # Load and verify content
            with open(config_path, "r") as f:
                data = json.load(f)

            assert data["context_size"] == 2048
            assert data["batch_size"] == 512
            assert data["max_tokens"] == 100

    def test_save_results_all_formats(self, sample_benchmark_run):
        """Test saving results in all formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            run_dir = persistence.create_run_directory(sample_benchmark_run.run_id)

            saved_files = persistence.save_results(sample_benchmark_run, run_dir)

            # Check all expected files were created
            assert "json" in saved_files
            assert "csv" in saved_files
            assert "markdown" in saved_files
            assert "hardware_info" in saved_files
            assert "config" in saved_files

            # Verify all files exist
            for file_path in saved_files.values():
                assert Path(file_path).exists()

    def test_get_software_versions(self):
        """Test software version collection."""
        versions = ResultsPersistence.get_software_versions()

        # Check required keys exist
        assert "python" in versions
        assert "llama-cpp-python" in versions
        assert "cuda_driver" in versions

        # Python version should be valid
        assert len(versions["python"]) > 0
        assert versions["python"][0].isdigit()

    def test_generate_run_id(self):
        """Test run ID generation."""
        run_id = ResultsPersistence.generate_run_id()

        # Check format: YYYYMMDD_HHMMSS
        assert len(run_id) == 15
        assert run_id[8] == "_"

        # Check it's a valid timestamp format
        datetime.strptime(run_id, "%Y%m%d_%H%M%S")

    def test_empty_results_handling(self, sample_hardware_info):
        """Test handling of benchmark run with no results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)

            # Create minimal benchmark run
            benchmark_run = BenchmarkRun(
                run_id="20240115_143022",
                timestamp="2024-01-15 14:30:22",
                duration_s=10.0,
                hardware_info=sample_hardware_info,
                software_versions={"python": "3.10.12"},
                config={},
                model_checksums={},
                quantization_results=[],
                ablation_results=[],
                batch_results=[],
                statistical_summaries=[],
                comparisons=[],
                visualization_paths=[],
                html_report_path="",
            )

            run_dir = persistence.create_run_directory(benchmark_run.run_id)
            saved_files = persistence.save_results(benchmark_run, run_dir)

            # Should still create all files
            assert len(saved_files) == 5

            # Verify files exist and are valid
            json_path = Path(saved_files["json"])
            assert json_path.exists()

            with open(json_path, "r") as f:
                data = json.load(f)
                assert data["run_id"] == "20240115_143022"
                assert data["quantization_results"] == []

    def test_gpu_hardware_info_serialization(self):
        """Test serialization of hardware info with GPU."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)

            hw_info = HardwareInfo(
                os_type="jetson_xavier_nx",
                cpu_model="ARM Cortex-A78AE",
                cpu_cores=8,
                cpu_features=["neon"],
                total_ram_gb=8.0,
                available_ram_gb=6.0,
                has_gpu=True,
                gpu_model="NVIDIA Xavier GPU",
                gpu_memory_gb=8.0,
                gpu_compute_capability="7.2",
                has_thermal_sensors=True,
                has_power_sensors=True,
            )

            hw_path = Path(tmpdir) / "hardware_info.json"
            persistence.save_hardware_info(hw_info, hw_path)

            # Load and verify GPU fields
            with open(hw_path, "r") as f:
                data = json.load(f)

            assert data["has_gpu"] is True
            assert data["gpu_model"] == "NVIDIA Xavier GPU"
            assert data["gpu_memory_gb"] == 8.0
            assert data["gpu_compute_capability"] == "7.2"

    def test_markdown_table_formatting(self, sample_benchmark_run):
        """Test Markdown table formatting is correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            md_path = Path(tmpdir) / "results.md"

            persistence.save_markdown(sample_benchmark_run, md_path)

            with open(md_path, "r") as f:
                lines = f.readlines()

            # Find quantization results table
            table_start = None
            for i, line in enumerate(lines):
                if "| Quantization | Load Time (s) |" in line:
                    table_start = i
                    break

            assert table_start is not None

            # Check table header separator
            assert "|---" in lines[table_start + 1]

            # Check data rows
            assert "| Q8_0 |" in lines[table_start + 2]
            assert "| Q4_0 |" in lines[table_start + 3]

    def test_csv_numeric_formatting(self, sample_benchmark_run):
        """Test CSV numeric values are properly formatted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            csv_path = Path(tmpdir) / "results.csv"

            persistence.save_csv(sample_benchmark_run, csv_path)

            with open(csv_path, "r") as f:
                content = f.read()

            # Check numeric formatting (2 decimal places)
            assert "2.50" in content  # load_time_s
            assert "150.00" in content  # ttft_ms
            assert "45.00" in content  # decode_tps
