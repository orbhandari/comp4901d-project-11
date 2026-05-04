"""
Visualization generator for benchmark results.

Creates charts and graphs from benchmark data including bar charts, line plots,
scatter plots, heatmaps, and HTML reports.
"""

import os
import logging
import base64
from typing import List, Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from jinja2 import Template

from llm_benchmark.models import (
    QuantizationResult,
    AblationResult,
    InferenceMetrics,
    StatisticalSummary,
    BenchmarkRun,
)

# Use non-interactive backend for server environments
matplotlib.use('Agg')

logger = logging.getLogger(__name__)


class VisualizationGenerator:
    """
    Generates visualizations from benchmark results.
    
    Creates bar charts, line plots, scatter plots, heatmaps, and HTML reports
    with error bars and confidence intervals.
    """
    
    def __init__(self, output_dir: str, dpi: int = 300):
        """
        Initialize the visualization generator.
        
        Args:
            output_dir: Directory to save visualization files
            dpi: Resolution for saved images (default: 300)
        """
        self.output_dir = output_dir
        self.dpi = dpi
        self.viz_dir = os.path.join(output_dir, "visualizations")
        
        # Create visualization directory if it doesn't exist
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Set seaborn style for better-looking plots
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        logger.info(f"VisualizationGenerator initialized with output_dir={output_dir}, dpi={dpi}")
    
    def plot_quantization_comparison(
        self,
        results: List[QuantizationResult],
        statistical_summaries: Optional[List[StatisticalSummary]] = None
    ) -> str:
        """
        Create bar chart comparing metrics across quantization levels.
        
        Args:
            results: List of quantization profiling results (may include multiple iterations)
            statistical_summaries: Optional statistical summaries with confidence intervals
            
        Returns:
            Path to saved PNG file
            
        **Validates: Requirements 9.1**
        """
        if not results:
            logger.warning("No quantization results to plot")
            return ""
        
        # Group results by quantization level and calculate means
        results_by_quant = {}
        for r in results:
            if r.quantization not in results_by_quant:
                results_by_quant[r.quantization] = []
            results_by_quant[r.quantization].append(r)
        
        # Calculate mean values for each quantization level
        quantizations = list(results_by_quant.keys())
        ttft_values = [np.mean([r.ttft_ms for r in results_by_quant[q]]) for q in quantizations]
        prefill_tps_values = [np.mean([r.prefill_tps for r in results_by_quant[q]]) for q in quantizations]
        decode_tps_values = [np.mean([r.decode_tps for r in results_by_quant[q]]) for q in quantizations]
        memory_values = [np.mean([r.peak_ram_mb for r in results_by_quant[q]]) for q in quantizations]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Quantization Level Comparison', fontsize=16, fontweight='bold')
        
        # Extract error bars from statistical summaries if available
        ttft_errors = self._extract_error_bars(statistical_summaries, "ttft_ms", quantizations)
        prefill_errors = self._extract_error_bars(statistical_summaries, "prefill_tps", quantizations)
        decode_errors = self._extract_error_bars(statistical_summaries, "decode_tps", quantizations)
        memory_errors = self._extract_error_bars(statistical_summaries, "peak_ram_mb", quantizations)
        
        # Plot 1: Time to First Token
        axes[0, 0].bar(quantizations, ttft_values, yerr=ttft_errors, capsize=5, alpha=0.8)
        axes[0, 0].set_title('Time to First Token (TTFT)', fontweight='bold')
        axes[0, 0].set_ylabel('TTFT (ms)')
        axes[0, 0].set_xlabel('Quantization Level')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Prefill Throughput
        axes[0, 1].bar(quantizations, prefill_tps_values, yerr=prefill_errors, 
                       capsize=5, alpha=0.8, color='green')
        axes[0, 1].set_title('Prefill Throughput', fontweight='bold')
        axes[0, 1].set_ylabel('Tokens/Second')
        axes[0, 1].set_xlabel('Quantization Level')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Plot 3: Decode Throughput
        axes[1, 0].bar(quantizations, decode_tps_values, yerr=decode_errors,
                       capsize=5, alpha=0.8, color='orange')
        axes[1, 0].set_title('Decode Throughput', fontweight='bold')
        axes[1, 0].set_ylabel('Tokens/Second')
        axes[1, 0].set_xlabel('Quantization Level')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Plot 4: Peak Memory Usage
        axes[1, 1].bar(quantizations, memory_values, yerr=memory_errors,
                       capsize=5, alpha=0.8, color='red')
        axes[1, 1].set_title('Peak Memory Usage', fontweight='bold')
        axes[1, 1].set_ylabel('Memory (MB)')
        axes[1, 1].set_xlabel('Quantization Level')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.viz_dir, "quantization_comparison.png")
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved quantization comparison chart to {output_path}")
        return output_path
    
    def plot_throughput_over_time(self, metrics: InferenceMetrics) -> str:
        """
        Create line plot showing throughput over time during inference.
        
        Args:
            metrics: Inference metrics containing per-token latency data
            
        Returns:
            Path to saved PNG file
            
        **Validates: Requirements 9.2**
        """
        if not metrics.per_token_latency_ms:
            logger.warning("No per-token latency data available")
            return ""
        
        # Calculate throughput (tokens per second) from latency
        latencies = metrics.per_token_latency_ms
        throughputs = [1000.0 / latency if latency > 0 else 0 for latency in latencies]
        token_indices = list(range(1, len(throughputs) + 1))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot throughput over time
        ax.plot(token_indices, throughputs, marker='o', markersize=4, 
                linewidth=2, alpha=0.7, label='Token Generation Rate')
        
        # Add moving average for trend
        if len(throughputs) >= 5:
            window_size = min(5, len(throughputs))
            moving_avg = pd.Series(throughputs).rolling(window=window_size, center=True).mean()
            ax.plot(token_indices, moving_avg, linewidth=3, alpha=0.8, 
                   color='red', label=f'{window_size}-Token Moving Average')
        
        # Add mean line
        mean_throughput = np.mean(throughputs)
        ax.axhline(y=mean_throughput, color='green', linestyle='--', 
                  linewidth=2, alpha=0.7, label=f'Mean: {mean_throughput:.2f} tok/s')
        
        ax.set_title('Token Generation Throughput Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Token Index', fontsize=12)
        ax.set_ylabel('Throughput (tokens/second)', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.viz_dir, "throughput_over_time.png")
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved throughput over time plot to {output_path}")
        return output_path
    
    def plot_memory_vs_speed_tradeoff(
        self,
        results: List[QuantizationResult],
        statistical_summaries: Optional[List[StatisticalSummary]] = None
    ) -> str:
        """
        Create scatter plot showing memory usage versus inference speed tradeoffs.
        
        Args:
            results: List of quantization profiling results (may include multiple iterations)
            statistical_summaries: Optional statistical summaries with confidence intervals
            
        Returns:
            Path to saved PNG file
            
        **Validates: Requirements 9.3**
        """
        if not results:
            logger.warning("No quantization results to plot")
            return ""
        
        # Group results by quantization level and calculate means
        results_by_quant = {}
        for r in results:
            if r.quantization not in results_by_quant:
                results_by_quant[r.quantization] = []
            results_by_quant[r.quantization].append(r)
        
        # Calculate mean values for each quantization level
        quantizations = list(results_by_quant.keys())
        memory_values = [np.mean([r.peak_ram_mb for r in results_by_quant[q]]) for q in quantizations]
        decode_tps_values = [np.mean([r.decode_tps for r in results_by_quant[q]]) for q in quantizations]
        
        # Extract error bars if available
        memory_errors = self._extract_error_bars(statistical_summaries, "peak_ram_mb", quantizations)
        decode_errors = self._extract_error_bars(statistical_summaries, "decode_tps", quantizations)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot with error bars
        colors = sns.color_palette("husl", len(quantizations))
        for i, (quant, mem, speed) in enumerate(zip(quantizations, memory_values, decode_tps_values)):
            xerr = memory_errors[i] if memory_errors else None
            yerr = decode_errors[i] if decode_errors else None
            
            ax.errorbar(mem, speed, xerr=xerr, yerr=yerr, 
                       fmt='o', markersize=12, capsize=5, capthick=2,
                       color=colors[i], label=quant, alpha=0.8)
        
        # Add annotations for each point
        for i, quant in enumerate(quantizations):
            ax.annotate(quant, (memory_values[i], decode_tps_values[i]),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, alpha=0.7)
        
        ax.set_title('Memory vs Speed Tradeoff', fontsize=14, fontweight='bold')
        ax.set_xlabel('Peak Memory Usage (MB)', fontsize=12)
        ax.set_ylabel('Decode Throughput (tokens/second)', fontsize=12)
        ax.legend(loc='best', title='Quantization')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.viz_dir, "memory_vs_speed_tradeoff.png")
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved memory vs speed tradeoff plot to {output_path}")
        return output_path
    
    def plot_heatmap(
        self,
        results: pd.DataFrame,
        x_col: str,
        y_col: str,
        value_col: str,
        title: str = "Performance Heatmap"
    ) -> str:
        """
        Create heatmap showing performance across two dimensions.
        
        Args:
            results: DataFrame containing benchmark results
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            value_col: Column name for heatmap values
            title: Title for the heatmap
            
        Returns:
            Path to saved PNG file
            
        **Validates: Requirements 9.4**
        """
        if results.empty:
            logger.warning("No data to plot heatmap")
            return ""
        
        # Pivot data for heatmap
        try:
            pivot_data = results.pivot(index=y_col, columns=x_col, values=value_col)
        except Exception as e:
            logger.error(f"Failed to pivot data for heatmap: {e}")
            return ""
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd',
                   cbar_kws={'label': value_col}, ax=ax, linewidths=0.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        
        plt.tight_layout()
        
        # Save figure
        filename = title.lower().replace(' ', '_') + '.png'
        output_path = os.path.join(self.viz_dir, filename)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved heatmap to {output_path}")
        return output_path
    
    def plot_ablation_comparison(
        self,
        results: List[AblationResult],
        baseline_scenario: str = "control"
    ) -> str:
        """
        Create before/after comparison charts for ablation studies.
        
        Args:
            results: List of ablation study results
            baseline_scenario: Name of the baseline scenario for comparison
            
        Returns:
            Path to saved PNG file
            
        **Validates: Requirements 9.5**
        """
        if not results:
            logger.warning("No ablation results to plot")
            return ""
        
        # Find baseline result
        baseline = None
        for r in results:
            if baseline_scenario.lower() in r.scenario.lower():
                baseline = r
                break
        
        if not baseline:
            logger.warning(f"Baseline scenario '{baseline_scenario}' not found")
            baseline = results[0]  # Use first result as baseline
        
        # Extract scenarios and metrics
        scenarios = [r.scenario for r in results if r != baseline]
        
        # Get all metric names from baseline
        metric_names = list(baseline.metrics.keys())
        
        if not metric_names:
            logger.warning("No metrics found in ablation results")
            return ""
        
        # Create figure with subplots for each metric
        num_metrics = len(metric_names)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 4 * num_metrics))
        
        # Handle single metric case
        if num_metrics == 1:
            axes = [axes]
        
        fig.suptitle('Ablation Study: Before/After Comparison', 
                    fontsize=16, fontweight='bold')
        
        for idx, metric_name in enumerate(metric_names):
            ax = axes[idx]
            
            # Get baseline value
            baseline_value = baseline.metrics.get(metric_name, 0)
            
            # Get values for other scenarios
            scenario_values = []
            improvements = []
            for r in results:
                if r != baseline:
                    value = r.metrics.get(metric_name, 0)
                    scenario_values.append(value)
                    
                    # Calculate improvement percentage
                    if baseline_value > 0:
                        improvement = ((value - baseline_value) / baseline_value) * 100
                    else:
                        improvement = 0
                    improvements.append(improvement)
            
            # Create grouped bar chart
            x = np.arange(len(scenarios))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, [baseline_value] * len(scenarios), 
                          width, label='Baseline', alpha=0.8, color='gray')
            bars2 = ax.bar(x + width/2, scenario_values, width, 
                          label='Optimized', alpha=0.8, color='green')
            
            # Add improvement percentages as text
            for i, (bar, improvement) in enumerate(zip(bars2, improvements)):
                height = bar.get_height()
                sign = '+' if improvement >= 0 else ''
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{sign}{improvement:.1f}%',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_title(f'{metric_name}', fontweight='bold')
            ax.set_ylabel('Value')
            ax.set_xlabel('Scenario')
            ax.set_xticks(x)
            ax.set_xticklabels(scenarios, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.viz_dir, "ablation_comparison.png")
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved ablation comparison chart to {output_path}")
        return output_path
    
    def _extract_error_bars(
        self,
        statistical_summaries: Optional[List[StatisticalSummary]],
        metric_name: str,
        quantizations: List[str]
    ) -> Optional[List[float]]:
        """
        Extract error bar values from statistical summaries.
        
        Args:
            statistical_summaries: List of statistical summaries
            metric_name: Name of the metric to extract
            quantizations: List of quantization levels
            
        Returns:
            List of error values (half of confidence interval width) or None
            
        **Validates: Requirements 9.6**
        """
        if not statistical_summaries:
            return None
        
        errors = []
        for quant in quantizations:
            # Find matching summary
            summary = None
            for s in statistical_summaries:
                # Match by metric name and quantization field
                if s.metric_name == metric_name and s.quantization == quant:
                    summary = s
                    break
            
            if summary:
                # Calculate error as half of CI width
                ci_low, ci_high = summary.confidence_interval_95
                error = (ci_high - ci_low) / 2.0
                errors.append(error)
            else:
                errors.append(0)
        
        return errors if any(errors) else None
    
    def generate_all_visualizations(
        self,
        quantization_results: List[QuantizationResult],
        ablation_results: List[AblationResult],
        statistical_summaries: Optional[List[StatisticalSummary]] = None,
        sample_metrics: Optional[InferenceMetrics] = None
    ) -> List[str]:
        """
        Generate all standard visualizations for a benchmark run.
        
        Args:
            quantization_results: Quantization profiling results
            ablation_results: Ablation study results
            statistical_summaries: Statistical summaries with confidence intervals
            sample_metrics: Sample inference metrics for throughput plot
            
        Returns:
            List of paths to generated visualization files
        """
        visualization_paths = []
        
        # Generate quantization comparison
        if quantization_results:
            path = self.plot_quantization_comparison(quantization_results, statistical_summaries)
            if path:
                visualization_paths.append(path)
        
        # Generate memory vs speed tradeoff
        if quantization_results:
            path = self.plot_memory_vs_speed_tradeoff(quantization_results, statistical_summaries)
            if path:
                visualization_paths.append(path)
        
        # Generate throughput over time
        if sample_metrics and sample_metrics.per_token_latency_ms:
            path = self.plot_throughput_over_time(sample_metrics)
            if path:
                visualization_paths.append(path)
        
        # Generate ablation comparison
        if ablation_results:
            path = self.plot_ablation_comparison(ablation_results)
            if path:
                visualization_paths.append(path)
        
        logger.info(f"Generated {len(visualization_paths)} visualizations")
        return visualization_paths

    def generate_html_report(
        self,
        benchmark_run: BenchmarkRun,
        visualization_paths: List[str]
    ) -> str:
        """
        Generate interactive HTML report with embedded visualizations.
        
        Args:
            benchmark_run: Complete benchmark run results
            visualization_paths: List of paths to PNG visualization files
            
        Returns:
            Path to saved HTML report file
            
        **Validates: Requirements 9.8**
        """
        logger.info("Generating HTML report...")
        
        # Read and encode visualizations as base64
        encoded_images = {}
        for viz_path in visualization_paths:
            if os.path.exists(viz_path):
                with open(viz_path, 'rb') as f:
                    img_data = f.read()
                    encoded = base64.b64encode(img_data).decode('utf-8')
                    img_name = os.path.basename(viz_path)
                    encoded_images[img_name] = encoded
        
        # Prepare summary statistics
        summary_stats = self._prepare_summary_statistics(benchmark_run)
        
        # Prepare comparison tables
        quant_table = self._prepare_quantization_table(benchmark_run.quantization_results)
        ablation_table = self._prepare_ablation_table(benchmark_run.ablation_results)
        batch_table = self._prepare_ablation_table(benchmark_run.batch_results)
        
        # Create HTML template
        html_template = self._get_html_template()
        
        # Render template
        template = Template(html_template)
        html_content = template.render(
            run_id=benchmark_run.run_id,
            timestamp=benchmark_run.timestamp,
            duration_s=benchmark_run.duration_s,
            hardware_info=benchmark_run.hardware_info,
            software_versions=benchmark_run.software_versions,
            config=benchmark_run.config,
            model_checksums=benchmark_run.model_checksums,
            summary_stats=summary_stats,
            quant_table=quant_table,
            ablation_table=ablation_table,
            batch_table=batch_table,
            statistical_summaries=benchmark_run.statistical_summaries,
            comparisons=benchmark_run.comparisons,
            encoded_images=encoded_images,
            visualization_paths=visualization_paths
        )
        
        # Save HTML report
        output_path = os.path.join(self.output_dir, "benchmark_report.html")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Saved HTML report to {output_path}")
        return output_path
    
    def _prepare_summary_statistics(self, benchmark_run: BenchmarkRun) -> Dict[str, Any]:
        """Prepare summary statistics for the HTML report."""
        summary = {
            'total_quantizations': len(benchmark_run.quantization_results),
            'total_ablations': len(benchmark_run.ablation_results),
            'total_batch_tests': len(benchmark_run.batch_results),
            'has_gpu': benchmark_run.hardware_info.has_gpu,
        }
        
        # Calculate best quantization by decode throughput
        if benchmark_run.quantization_results:
            best_quant = max(benchmark_run.quantization_results, key=lambda x: x.decode_tps)
            summary['best_quantization'] = best_quant.quantization
            summary['best_decode_tps'] = best_quant.decode_tps
            
            # Calculate memory range
            memory_values = [r.peak_ram_mb for r in benchmark_run.quantization_results]
            summary['min_memory_mb'] = min(memory_values)
            summary['max_memory_mb'] = max(memory_values)
            
            # Calculate speed range
            speed_values = [r.decode_tps for r in benchmark_run.quantization_results]
            summary['min_decode_tps'] = min(speed_values)
            summary['max_decode_tps'] = max(speed_values)
        
        return summary
    
    def _prepare_quantization_table(self, results: List[QuantizationResult]) -> Dict[str, Any]:
        """Prepare quantization results as table data, grouped by iteration."""
        # Group results by iteration
        results_by_iteration = {}
        for result in results:
            iteration = result.iteration
            if iteration not in results_by_iteration:
                results_by_iteration[iteration] = []
            
            row = {
                'quantization': result.quantization,
                'load_time_s': f"{result.load_time_s:.2f}",
                'peak_ram_mb': f"{result.peak_ram_mb:.2f}",
                'ram_increase_mb': f"{result.ram_increase_mb:.2f}",
                'ttft_ms': f"{result.ttft_ms:.2f}",
                'prefill_tps': f"{result.prefill_tps:.2f}",
                'decode_tps': f"{result.decode_tps:.2f}",
                'prompt_tokens': result.prompt_tokens,
                'output_tokens': result.output_tokens,
                'gpu_memory_mb': f"{result.gpu_memory_mb:.2f}" if result.gpu_memory_mb else "N/A",
                'gpu_utilization_pct': f"{result.gpu_utilization_pct:.2f}" if result.gpu_utilization_pct else "N/A",
                'used_gpu': "Yes" if result.used_gpu_acceleration else "No",
                'iteration': iteration
            }
            results_by_iteration[iteration].append(row)
        
        # Sort iterations
        sorted_iterations = sorted(results_by_iteration.keys())
        
        return {
            'iterations': sorted_iterations,
            'results_by_iteration': results_by_iteration
        }
    
    def _prepare_ablation_table(self, results: List[AblationResult]) -> List[Dict[str, Any]]:
        """Prepare ablation results as table data."""
        table_data = []
        for result in results:
            row = {
                'scenario': result.scenario,
                'configuration': result.configuration,
                'metrics': result.metrics,
                'improvement': f"{result.improvement_over_baseline:.2f}%" if result.improvement_over_baseline is not None else "N/A"
            }
            table_data.append(row)
        return table_data
    
    def _get_html_template(self) -> str:
        """Return the HTML template for the benchmark report."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Benchmark Report - {{ run_id }}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        
        h2 {
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }
        
        h3 {
            color: #555;
            margin-top: 25px;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        .metadata {
            background: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }
        
        .metadata-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }
        
        .metadata-item {
            padding: 10px;
            background: white;
            border-radius: 4px;
        }
        
        .metadata-label {
            font-weight: bold;
            color: #7f8c8d;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .metadata-value {
            color: #2c3e50;
            font-size: 1.1em;
            margin-top: 5px;
        }
        
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .summary-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .summary-card.green {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }
        
        .summary-card.orange {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        
        .summary-card.blue {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }
        
        .summary-card-value {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .summary-card-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        th {
            background: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }
        
        td {
            padding: 12px;
            border-bottom: 1px solid #ecf0f1;
        }
        
        tr:hover {
            background: #f8f9fa;
        }
        
        .collapsible {
            background: #3498db;
            color: white;
            cursor: pointer;
            padding: 15px;
            width: 100%;
            border: none;
            text-align: left;
            outline: none;
            font-size: 1.1em;
            font-weight: 600;
            border-radius: 5px;
            margin-top: 10px;
            transition: background 0.3s;
        }
        
        .collapsible:hover {
            background: #2980b9;
        }
        
        .collapsible:after {
            content: '\\002B';
            color: white;
            font-weight: bold;
            float: right;
            margin-left: 5px;
        }
        
        .collapsible.active:after {
            content: "\\2212";
        }
        
        .content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
            background: #f8f9fa;
            border-radius: 0 0 5px 5px;
        }
        
        .content-inner {
            padding: 20px;
        }
        
        .visualization {
            margin: 30px 0;
            text-align: center;
        }
        
        .visualization img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .visualization-title {
            font-weight: 600;
            margin-bottom: 15px;
            color: #2c3e50;
            font-size: 1.2em;
        }
        
        .hardware-info {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }
        
        .hardware-info ul {
            list-style: none;
            padding-left: 0;
        }
        
        .hardware-info li {
            padding: 8px 0;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .hardware-info li:last-child {
            border-bottom: none;
        }
        
        .hardware-info strong {
            color: #3498db;
            display: inline-block;
            min-width: 200px;
        }
        
        .config-block {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            line-height: 1.5;
        }
        
        .tooltip {
            position: relative;
            display: inline-block;
            border-bottom: 1px dotted #3498db;
            cursor: help;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 8px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.85em;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }
        
        .badge-success {
            background: #27ae60;
            color: white;
        }
        
        .badge-warning {
            background: #f39c12;
            color: white;
        }
        
        .badge-info {
            background: #3498db;
            color: white;
        }
        
        @media print {
            body {
                background: white;
            }
            .container {
                box-shadow: none;
            }
            .collapsible {
                display: none;
            }
            .content {
                max-height: none !important;
                display: block !important;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 LLM Benchmark Report</h1>
        
        <div class="metadata">
            <div class="metadata-grid">
                <div class="metadata-item">
                    <div class="metadata-label">Run ID</div>
                    <div class="metadata-value">{{ run_id }}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Timestamp</div>
                    <div class="metadata-value">{{ timestamp }}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Duration</div>
                    <div class="metadata-value">{{ "%.2f"|format(duration_s) }} seconds</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Platform</div>
                    <div class="metadata-value">{{ hardware_info.os_type }}</div>
                </div>
            </div>
        </div>
        
        {% if summary_stats %}
        <h2>📊 Summary Statistics</h2>
        <div class="summary-cards">
            {% if summary_stats.total_quantizations > 0 %}
            <div class="summary-card">
                <div class="summary-card-value">{{ summary_stats.total_quantizations }}</div>
                <div class="summary-card-label">Quantization Levels Tested</div>
            </div>
            {% endif %}
            
            {% if summary_stats.best_quantization %}
            <div class="summary-card green">
                <div class="summary-card-value">{{ summary_stats.best_quantization }}</div>
                <div class="summary-card-label">Best Quantization ({{ "%.2f"|format(summary_stats.best_decode_tps) }} tok/s)</div>
            </div>
            {% endif %}
            
            {% if summary_stats.max_memory_mb %}
            <div class="summary-card orange">
                <div class="summary-card-value">{{ "%.0f"|format(summary_stats.min_memory_mb) }}-{{ "%.0f"|format(summary_stats.max_memory_mb) }}</div>
                <div class="summary-card-label">Memory Range (MB)</div>
            </div>
            {% endif %}
            
            {% if summary_stats.total_ablations > 0 %}
            <div class="summary-card blue">
                <div class="summary-card-value">{{ summary_stats.total_ablations }}</div>
                <div class="summary-card-label">Ablation Studies</div>
            </div>
            {% endif %}
        </div>
        {% endif %}
        
        <h2>💻 Hardware Information</h2>
        <div class="hardware-info">
            <ul>
                <li><strong>Platform:</strong> {{ hardware_info.os_type }}</li>
                <li><strong>CPU Model:</strong> {{ hardware_info.cpu_model }}</li>
                <li><strong>CPU Cores:</strong> {{ hardware_info.cpu_cores }}</li>
                <li><strong>CPU Features:</strong> {{ hardware_info.cpu_features|join(', ') }}</li>
                <li><strong>Total RAM:</strong> {{ "%.2f"|format(hardware_info.total_ram_gb) }} GB</li>
                <li><strong>Available RAM:</strong> {{ "%.2f"|format(hardware_info.available_ram_gb) }} GB</li>
                {% if hardware_info.has_gpu %}
                <li><strong>GPU:</strong> <span class="badge badge-success">Available</span> {{ hardware_info.gpu_model }}</li>
                <li><strong>GPU Memory:</strong> {{ "%.2f"|format(hardware_info.gpu_memory_gb) }} GB</li>
                <li><strong>GPU Compute Capability:</strong> {{ hardware_info.gpu_compute_capability }}</li>
                {% else %}
                <li><strong>GPU:</strong> <span class="badge badge-warning">Not Available</span></li>
                {% endif %}
                <li><strong>Thermal Sensors:</strong> {{ "Available" if hardware_info.has_thermal_sensors else "Not Available" }}</li>
                <li><strong>Power Sensors:</strong> {{ "Available" if hardware_info.has_power_sensors else "Not Available" }}</li>
            </ul>
        </div>
        
        <button class="collapsible">📦 Software Versions</button>
        <div class="content">
            <div class="content-inner">
                <table>
                    <thead>
                        <tr>
                            <th>Package</th>
                            <th>Version</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for name, version in software_versions.items() %}
                        <tr>
                            <td>{{ name }}</td>
                            <td>{{ version }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        
        <button class="collapsible">⚙️ Configuration</button>
        <div class="content">
            <div class="content-inner">
                <div class="config-block">{{ config|tojson(indent=2) }}</div>
            </div>
        </div>
        
        {% if model_checksums %}
        <button class="collapsible">🔐 Model Checksums (SHA256)</button>
        <div class="content">
            <div class="content-inner">
                <table>
                    <thead>
                        <tr>
                            <th>Model File</th>
                            <th>SHA256 Checksum</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for filename, checksum in model_checksums.items() %}
                        <tr>
                            <td>{{ filename }}</td>
                            <td><code>{{ checksum }}</code></td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        {% endif %}
        
        {% if quant_table %}
        <h2>📈 Quantization Results</h2>
        
        {% for iteration in quant_table.iterations %}
        <button class="collapsible">Iteration {{ iteration }}</button>
        <div class="content">
            <div class="content-inner">
                <table>
                    <thead>
                        <tr>
                            <th>Quantization</th>
                            <th>
                                <span class="tooltip">Load Time (s)
                                    <span class="tooltiptext">Time to load model into memory</span>
                                </span>
                            </th>
                            <th>
                                <span class="tooltip">Peak RAM (MB)
                                    <span class="tooltiptext">Maximum memory usage</span>
                                </span>
                            </th>
                            <th>
                                <span class="tooltip">TTFT (ms)
                                    <span class="tooltiptext">Time to first token</span>
                                </span>
                            </th>
                            <th>
                                <span class="tooltip">Prefill TPS
                                    <span class="tooltiptext">Prefill tokens per second</span>
                                </span>
                            </th>
                            <th>
                                <span class="tooltip">Decode TPS
                                    <span class="tooltiptext">Decode tokens per second</span>
                                </span>
                            </th>
                            <th>Tokens (in/out)</th>
                            <th>GPU</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for row in quant_table.results_by_iteration[iteration] %}
                        <tr>
                            <td><strong>{{ row.quantization }}</strong></td>
                            <td>{{ row.load_time_s }}</td>
                            <td>{{ row.peak_ram_mb }}</td>
                            <td>{{ row.ttft_ms }}</td>
                            <td>{{ row.prefill_tps }}</td>
                            <td>{{ row.decode_tps }}</td>
                            <td>{{ row.prompt_tokens }}/{{ row.output_tokens }}</td>
                            <td>
                                {% if row.used_gpu == "Yes" %}
                                <span class="badge badge-success">Yes</span>
                                {% else %}
                                <span class="badge badge-warning">No</span>
                                {% endif %}
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        {% endfor %}
        {% endif %}
        
        {% if ablation_table %}
        <h2>🔬 Ablation Study Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Scenario</th>
                    <th>Configuration</th>
                    <th>Metrics</th>
                    <th>Improvement</th>
                </tr>
            </thead>
            <tbody>
                {% for row in ablation_table %}
                <tr>
                    <td><strong>{{ row.scenario }}</strong></td>
                    <td><code>{{ row.configuration|tojson }}</code></td>
                    <td>
                        {% for metric, value in row.metrics.items() %}
                        <div>{{ metric }}: {{ "%.2f"|format(value) }}</div>
                        {% endfor %}
                    </td>
                    <td>{{ row.improvement }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}
        
        {% if batch_table %}
        <h2>⚡ Batch Processing Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Scenario</th>
                    <th>Configuration</th>
                    <th>Metrics</th>
                    <th>Improvement</th>
                </tr>
            </thead>
            <tbody>
                {% for row in batch_table %}
                <tr>
                    <td><strong>{{ row.scenario }}</strong></td>
                    <td><code>{{ row.configuration|tojson }}</code></td>
                    <td>
                        {% for metric, value in row.metrics.items() %}
                        <div>{{ metric }}: {{ "%.2f"|format(value) }}</div>
                        {% endfor %}
                    </td>
                    <td>{{ row.improvement }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}
        
        {% if statistical_summaries %}
        <button class="collapsible">📊 Statistical Summaries</button>
        <div class="content">
            <div class="content-inner">
                <table>
                    <thead>
                        <tr>
                            <th>Quantization</th>
                            <th>Metric</th>
                            <th>Mean</th>
                            <th>Std Dev</th>
                            <th>95% CI</th>
                            <th>Outliers</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for summary in statistical_summaries %}
                        <tr>
                            <td><strong>{{ summary.quantization }}</strong></td>
                            <td>{{ summary.metric_name }}</td>
                            <td>{{ "%.2f"|format(summary.mean) }}</td>
                            <td>{{ "%.2f"|format(summary.std_dev) }}</td>
                            <td>[{{ "%.2f"|format(summary.confidence_interval_95[0]) }}, {{ "%.2f"|format(summary.confidence_interval_95[1]) }}]</td>
                            <td>{{ summary.outliers|length if summary.outliers else 0 }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        {% endif %}
        
        {% if comparisons %}
        <button class="collapsible">🔍 Statistical Comparisons</button>
        <div class="content">
            <div class="content-inner">
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Config A Mean</th>
                            <th>Config B Mean</th>
                            <th>Difference</th>
                            <th>P-Value</th>
                            <th>Significant</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for comp in comparisons %}
                        <tr>
                            <td>{{ comp.metric_name }}</td>
                            <td>{{ "%.2f"|format(comp.config_a_mean) }}</td>
                            <td>{{ "%.2f"|format(comp.config_b_mean) }}</td>
                            <td>{{ "%.2f"|format(comp.difference) }}</td>
                            <td>{{ "%.4f"|format(comp.p_value) }}</td>
                            <td>
                                {% if comp.is_significant %}
                                <span class="badge badge-success">✓ Yes</span>
                                {% else %}
                                <span class="badge badge-warning">✗ No</span>
                                {% endif %}
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        {% endif %}
        
        {% if encoded_images %}
        <h2>📊 Visualizations</h2>
        
        {% if 'quantization_comparison.png' in encoded_images %}
        <div class="visualization">
            <div class="visualization-title">Quantization Level Comparison</div>
            <img src="data:image/png;base64,{{ encoded_images['quantization_comparison.png'] }}" alt="Quantization Comparison">
        </div>
        {% endif %}
        
        {% if 'memory_vs_speed_tradeoff.png' in encoded_images %}
        <div class="visualization">
            <div class="visualization-title">Memory vs Speed Tradeoff</div>
            <img src="data:image/png;base64,{{ encoded_images['memory_vs_speed_tradeoff.png'] }}" alt="Memory vs Speed Tradeoff">
        </div>
        {% endif %}
        
        {% if 'throughput_over_time.png' in encoded_images %}
        <div class="visualization">
            <div class="visualization-title">Throughput Over Time</div>
            <img src="data:image/png;base64,{{ encoded_images['throughput_over_time.png'] }}" alt="Throughput Over Time">
        </div>
        {% endif %}
        
        {% if 'ablation_comparison.png' in encoded_images %}
        <div class="visualization">
            <div class="visualization-title">Ablation Study Comparison</div>
            <img src="data:image/png;base64,{{ encoded_images['ablation_comparison.png'] }}" alt="Ablation Comparison">
        </div>
        {% endif %}
        {% endif %}
        
        <div style="margin-top: 50px; padding-top: 20px; border-top: 2px solid #ecf0f1; text-align: center; color: #7f8c8d;">
            <p>Generated by LLM Benchmark Framework | {{ timestamp }}</p>
        </div>
    </div>
    
    <script>
        // Collapsible sections
        var coll = document.getElementsByClassName("collapsible");
        for (var i = 0; i < coll.length; i++) {
            coll[i].addEventListener("click", function() {
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                if (content.style.maxHeight) {
                    content.style.maxHeight = null;
                } else {
                    content.style.maxHeight = content.scrollHeight + "px";
                }
            });
        }
    </script>
</body>
</html>
'''
