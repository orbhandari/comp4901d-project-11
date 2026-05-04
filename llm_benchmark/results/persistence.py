"""
Result persistence implementation.

Provides serialization and export functionality for benchmark results in multiple formats:
- JSON: Complete machine-readable results
- CSV: Tabular data for spreadsheet import
- Markdown: Human-readable report
"""

import csv
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from llm_benchmark.models import (
    AblationResult,
    BenchmarkRun,
    ComparisonResult,
    HardwareInfo,
    QuantizationResult,
    StatisticalSummary,
)


class ResultsPersistence:
    """Handles persistence of benchmark results to multiple formats."""

    def __init__(self, output_dir: str = "./benchmark_results"):
        """
        Initialize results persistence.

        Args:
            output_dir: Base directory for storing results
        """
        self.output_dir = Path(output_dir)

    def create_run_directory(self, run_id: str) -> Path:
        """
        Create organized directory structure for a benchmark run.

        Creates structure:
        results/
        └── run_TIMESTAMP/
            ├── config.json
            ├── hardware_info.json
            ├── results.json
            ├── results.csv
            ├── results.md
            ├── visualizations/
            ├── logs/
            └── checkpoints/

        Args:
            run_id: Unique identifier for the run (typically timestamp)

        Returns:
            Path to the run directory
        """
        run_dir = self.output_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (run_dir / "visualizations").mkdir(exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)

        return run_dir

    def save_results(self, benchmark_run: BenchmarkRun, run_dir: Path) -> Dict[str, str]:
        """
        Save benchmark results in all formats.

        Args:
            benchmark_run: Complete benchmark run results
            run_dir: Directory to save results in

        Returns:
            Dictionary mapping format names to file paths
        """
        saved_files = {}

        # Save JSON (complete results)
        json_path = run_dir / "results.json"
        self.save_json(benchmark_run, json_path)
        saved_files["json"] = str(json_path)

        # Save CSV (tabular data)
        csv_path = run_dir / "results.csv"
        self.save_csv(benchmark_run, csv_path)
        saved_files["csv"] = str(csv_path)

        # Save Markdown (human-readable report)
        md_path = run_dir / "results.md"
        self.save_markdown(benchmark_run, md_path)
        saved_files["markdown"] = str(md_path)

        # Save hardware info separately for easy reference
        hw_path = run_dir / "hardware_info.json"
        self.save_hardware_info(benchmark_run.hardware_info, hw_path)
        saved_files["hardware_info"] = str(hw_path)

        # Save configuration separately
        config_path = run_dir / "config.json"
        self.save_config(benchmark_run.config, config_path)
        saved_files["config"] = str(config_path)

        return saved_files

    def save_json(self, benchmark_run: BenchmarkRun, path: Path) -> None:
        """
        Save complete benchmark results as JSON.

        Args:
            benchmark_run: Benchmark run to serialize
            path: Output file path
        """
        # Convert dataclass to dictionary
        data = self._benchmark_run_to_dict(benchmark_run)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_csv(self, benchmark_run: BenchmarkRun, path: Path) -> None:
        """
        Save benchmark results as CSV (tabular format).

        Creates a flattened view of quantization results suitable for spreadsheet analysis.

        Args:
            benchmark_run: Benchmark run to export
            path: Output file path
        """
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write metadata header
            writer.writerow(["# Benchmark Run Metadata"])
            writer.writerow(["Run ID", benchmark_run.run_id])
            writer.writerow(["Timestamp", benchmark_run.timestamp])
            writer.writerow(["Duration (s)", f"{benchmark_run.duration_s:.2f}"])
            writer.writerow(["Hardware", benchmark_run.hardware_info.os_type])
            writer.writerow([])

            # Write quantization results
            if benchmark_run.quantization_results:
                writer.writerow(["# Quantization Results"])
                writer.writerow([
                    "Quantization",
                    "Load Time (s)",
                    "Peak RAM (MB)",
                    "RAM Increase (MB)",
                    "TTFT (ms)",
                    "Prefill TPS",
                    "Decode TPS",
                    "Prompt Tokens",
                    "Output Tokens",
                    "GPU Memory (MB)",
                    "GPU Utilization (%)",
                ])

                for result in benchmark_run.quantization_results:
                    writer.writerow([
                        result.quantization,
                        f"{result.load_time_s:.2f}",
                        f"{result.peak_ram_mb:.2f}",
                        f"{result.ram_increase_mb:.2f}",
                        f"{result.ttft_ms:.2f}",
                        f"{result.prefill_tps:.2f}",
                        f"{result.decode_tps:.2f}",
                        result.prompt_tokens,
                        result.output_tokens,
                        f"{result.gpu_memory_mb:.2f}" if result.gpu_memory_mb else "N/A",
                        f"{result.gpu_utilization_pct:.2f}" if result.gpu_utilization_pct else "N/A",
                    ])
                writer.writerow([])

            # Write ablation results
            if benchmark_run.ablation_results:
                writer.writerow(["# Ablation Study Results"])
                writer.writerow(["Scenario", "Configuration", "Metrics", "Improvement"])

                for result in benchmark_run.ablation_results:
                    config_str = json.dumps(result.configuration)
                    metrics_str = json.dumps(result.metrics)
                    improvement = f"{result.improvement_over_baseline:.2f}%" if result.improvement_over_baseline else "N/A"
                    writer.writerow([result.scenario, config_str, metrics_str, improvement])
                writer.writerow([])

            # Write batch results
            if benchmark_run.batch_results:
                writer.writerow(["# Batch Processing Results"])
                writer.writerow(["Scenario", "Configuration", "Metrics", "Improvement"])

                for result in benchmark_run.batch_results:
                    config_str = json.dumps(result.configuration)
                    metrics_str = json.dumps(result.metrics)
                    improvement = f"{result.improvement_over_baseline:.2f}%" if result.improvement_over_baseline else "N/A"
                    writer.writerow([result.scenario, config_str, metrics_str, improvement])
                writer.writerow([])

            # Write statistical summaries
            if benchmark_run.statistical_summaries:
                writer.writerow(["# Statistical Summaries"])
                writer.writerow(["Metric", "Mean", "Std Dev", "CI 95% Low", "CI 95% High", "Outliers"])

                for summary in benchmark_run.statistical_summaries:
                    outliers_str = ", ".join(f"{o:.2f}" for o in summary.outliers) if summary.outliers else "None"
                    writer.writerow([
                        summary.metric_name,
                        f"{summary.mean:.2f}",
                        f"{summary.std_dev:.2f}",
                        f"{summary.confidence_interval_95[0]:.2f}",
                        f"{summary.confidence_interval_95[1]:.2f}",
                        outliers_str,
                    ])
                writer.writerow([])

            # Write comparisons
            if benchmark_run.comparisons:
                writer.writerow(["# Statistical Comparisons"])
                writer.writerow(["Metric", "Config A Mean", "Config B Mean", "Difference", "P-Value", "Significant"])

                for comp in benchmark_run.comparisons:
                    writer.writerow([
                        comp.metric_name,
                        f"{comp.config_a_mean:.2f}",
                        f"{comp.config_b_mean:.2f}",
                        f"{comp.difference:.2f}",
                        f"{comp.p_value:.4f}",
                        "Yes" if comp.is_significant else "No",
                    ])

    def save_markdown(self, benchmark_run: BenchmarkRun, path: Path) -> None:
        """
        Save benchmark results as Markdown (human-readable report).

        Args:
            benchmark_run: Benchmark run to format
            path: Output file path
        """
        lines = []

        # Title and metadata
        lines.append(f"# Benchmark Run Report: {benchmark_run.run_id}")
        lines.append("")
        lines.append(f"**Timestamp:** {benchmark_run.timestamp}")
        lines.append(f"**Duration:** {benchmark_run.duration_s:.2f} seconds")
        lines.append("")

        # Hardware information
        lines.append("## Hardware Information")
        lines.append("")
        hw = benchmark_run.hardware_info
        lines.append(f"- **Platform:** {hw.os_type}")
        lines.append(f"- **CPU:** {hw.cpu_model} ({hw.cpu_cores} cores)")
        lines.append(f"- **CPU Features:** {', '.join(hw.cpu_features)}")
        lines.append(f"- **RAM:** {hw.total_ram_gb:.2f} GB total, {hw.available_ram_gb:.2f} GB available")
        if hw.has_gpu:
            lines.append(f"- **GPU:** {hw.gpu_model}")
            lines.append(f"- **GPU Memory:** {hw.gpu_memory_gb:.2f} GB")
            lines.append(f"- **GPU Compute Capability:** {hw.gpu_compute_capability}")
        else:
            lines.append("- **GPU:** Not available")
        lines.append(f"- **Thermal Sensors:** {'Available' if hw.has_thermal_sensors else 'Not available'}")
        lines.append(f"- **Power Sensors:** {'Available' if hw.has_power_sensors else 'Not available'}")
        lines.append("")

        # Software versions
        lines.append("## Software Versions")
        lines.append("")
        for name, version in benchmark_run.software_versions.items():
            lines.append(f"- **{name}:** {version}")
        lines.append("")

        # Configuration
        lines.append("## Configuration")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(benchmark_run.config, indent=2))
        lines.append("```")
        lines.append("")

        # Model checksums
        if benchmark_run.model_checksums:
            lines.append("## Model Checksums (SHA256)")
            lines.append("")
            for filename, checksum in benchmark_run.model_checksums.items():
                lines.append(f"- **{filename}:** `{checksum}`")
            lines.append("")

        # Quantization results
        if benchmark_run.quantization_results:
            lines.append("## Quantization Results")
            lines.append("")
            lines.append("| Quantization | Load Time (s) | Peak RAM (MB) | TTFT (ms) | Prefill TPS | Decode TPS | Tokens (in/out) |")
            lines.append("|--------------|---------------|---------------|-----------|-------------|------------|-----------------|")

            for result in benchmark_run.quantization_results:
                lines.append(
                    f"| {result.quantization} | "
                    f"{result.load_time_s:.2f} | "
                    f"{result.peak_ram_mb:.2f} | "
                    f"{result.ttft_ms:.2f} | "
                    f"{result.prefill_tps:.2f} | "
                    f"{result.decode_tps:.2f} | "
                    f"{result.prompt_tokens}/{result.output_tokens} |"
                )
            lines.append("")

        # Ablation results
        if benchmark_run.ablation_results:
            lines.append("## Ablation Study Results")
            lines.append("")
            for result in benchmark_run.ablation_results:
                lines.append(f"### {result.scenario}")
                lines.append("")
                lines.append("**Configuration:**")
                lines.append("```json")
                lines.append(json.dumps(result.configuration, indent=2))
                lines.append("```")
                lines.append("")
                lines.append("**Metrics:**")
                for metric, value in result.metrics.items():
                    lines.append(f"- {metric}: {value:.2f}")
                if result.improvement_over_baseline is not None:
                    lines.append(f"- **Improvement over baseline:** {result.improvement_over_baseline:.2f}%")
                lines.append("")

        # Batch results
        if benchmark_run.batch_results:
            lines.append("## Batch Processing Results")
            lines.append("")
            for result in benchmark_run.batch_results:
                lines.append(f"### {result.scenario}")
                lines.append("")
                lines.append("**Configuration:**")
                lines.append("```json")
                lines.append(json.dumps(result.configuration, indent=2))
                lines.append("```")
                lines.append("")
                lines.append("**Metrics:**")
                for metric, value in result.metrics.items():
                    lines.append(f"- {metric}: {value:.2f}")
                if result.improvement_over_baseline is not None:
                    lines.append(f"- **Improvement over baseline:** {result.improvement_over_baseline:.2f}%")
                lines.append("")

        # Statistical summaries
        if benchmark_run.statistical_summaries:
            lines.append("## Statistical Summaries")
            lines.append("")
            lines.append("| Metric | Mean | Std Dev | 95% CI | Outliers |")
            lines.append("|--------|------|---------|--------|----------|")

            for summary in benchmark_run.statistical_summaries:
                ci_str = f"[{summary.confidence_interval_95[0]:.2f}, {summary.confidence_interval_95[1]:.2f}]"
                outliers_str = f"{len(summary.outliers)} detected" if summary.outliers else "None"
                lines.append(
                    f"| {summary.metric_name} | "
                    f"{summary.mean:.2f} | "
                    f"{summary.std_dev:.2f} | "
                    f"{ci_str} | "
                    f"{outliers_str} |"
                )
            lines.append("")

        # Comparisons
        if benchmark_run.comparisons:
            lines.append("## Statistical Comparisons")
            lines.append("")
            lines.append("| Metric | Config A | Config B | Difference | P-Value | Significant |")
            lines.append("|--------|----------|----------|------------|---------|-------------|")

            for comp in benchmark_run.comparisons:
                sig_str = "✓ Yes" if comp.is_significant else "✗ No"
                lines.append(
                    f"| {comp.metric_name} | "
                    f"{comp.config_a_mean:.2f} | "
                    f"{comp.config_b_mean:.2f} | "
                    f"{comp.difference:.2f} | "
                    f"{comp.p_value:.4f} | "
                    f"{sig_str} |"
                )
            lines.append("")

        # Visualizations
        if benchmark_run.visualization_paths:
            lines.append("## Visualizations")
            lines.append("")
            for viz_path in benchmark_run.visualization_paths:
                viz_name = Path(viz_path).name
                lines.append(f"- [{viz_name}]({viz_path})")
            lines.append("")

        # HTML report link
        if benchmark_run.html_report_path:
            lines.append("## Interactive Report")
            lines.append("")
            lines.append(f"[View Interactive HTML Report]({benchmark_run.html_report_path})")
            lines.append("")

        # Write to file
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def save_hardware_info(self, hardware_info: HardwareInfo, path: Path) -> None:
        """
        Save hardware information as JSON.

        Args:
            hardware_info: Hardware information to save
            path: Output file path
        """
        data = asdict(hardware_info)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save_config(self, config: Dict[str, Any], path: Path) -> None:
        """
        Save configuration as JSON.

        Args:
            config: Configuration dictionary
            path: Output file path
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def _benchmark_run_to_dict(self, benchmark_run: BenchmarkRun) -> Dict[str, Any]:
        """
        Convert BenchmarkRun to dictionary for JSON serialization.

        Handles nested dataclasses and special types.

        Args:
            benchmark_run: Benchmark run to convert

        Returns:
            Dictionary representation
        """
        return {
            "run_id": benchmark_run.run_id,
            "timestamp": benchmark_run.timestamp,
            "duration_s": benchmark_run.duration_s,
            "hardware_info": asdict(benchmark_run.hardware_info),
            "software_versions": benchmark_run.software_versions,
            "config": benchmark_run.config,
            "model_checksums": benchmark_run.model_checksums,
            "quantization_results": [asdict(r) for r in benchmark_run.quantization_results],
            "ablation_results": [asdict(r) for r in benchmark_run.ablation_results],
            "batch_results": [asdict(r) for r in benchmark_run.batch_results],
            "statistical_summaries": [asdict(s) for s in benchmark_run.statistical_summaries],
            "comparisons": [asdict(c) for c in benchmark_run.comparisons],
            "visualization_paths": benchmark_run.visualization_paths,
            "html_report_path": benchmark_run.html_report_path,
        }

    @staticmethod
    def get_software_versions() -> Dict[str, str]:
        """
        Collect software version information for reproducibility.

        Returns:
            Dictionary mapping software names to versions
        """
        versions = {
            "python": sys.version.split()[0],
        }

        # Try to get llama-cpp-python version
        try:
            import llama_cpp
            versions["llama-cpp-python"] = getattr(llama_cpp, "__version__", "unknown")
        except ImportError:
            versions["llama-cpp-python"] = "not installed"

        # Try to get CUDA version (if available)
        try:
            import pynvml
            pynvml.nvmlInit()
            versions["cuda_driver"] = pynvml.nvmlSystemGetDriverVersion()
            pynvml.nvmlShutdown()
        except Exception:
            versions["cuda_driver"] = "not available"

        # Get other common dependencies
        for package in ["numpy", "pandas", "matplotlib", "psutil"]:
            try:
                mod = __import__(package)
                versions[package] = getattr(mod, "__version__", "unknown")
            except ImportError:
                versions[package] = "not installed"

        return versions

    @staticmethod
    def generate_run_id() -> str:
        """
        Generate a unique run ID based on timestamp.

        Returns:
            Run ID string in format YYYYMMDD_HHMMSS
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")
