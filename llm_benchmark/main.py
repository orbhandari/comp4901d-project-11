"""
Main entry point for the benchmark framework.

Provides command-line interface and orchestrates the complete benchmark workflow.

This script integrates all components:
- Hardware detection and HAL
- Model manager
- Metrics collector
- Quantization profiler
- Test orchestrator
- Ablation engine
- Statistical validator
- Visualization generator
- Error handling
"""

import logging
import os
import sys
from datetime import datetime
import time 
from pathlib import Path
from typing import Dict, List, Optional

from llm_benchmark.config import ConfigParser, BenchmarkConfig
from llm_benchmark.hardware import HardwareDetector, create_backend
from llm_benchmark.logging_config import setup_logging, get_logger
from llm_benchmark.model_manager import ModelManager
from llm_benchmark.models import BenchmarkRun, QuantizationResult, AblationResult
from llm_benchmark.orchestrator import TestOrchestrator
from llm_benchmark.profiler import QuantizationProfiler
from llm_benchmark.results.persistence import ResultsPersistence
from llm_benchmark.visualization import VisualizationGenerator


def validate_dependencies() -> bool:
    """
    Validate that all required dependencies are installed.
    
    On Android, llama-cpp-python is optional since we use native llama.cpp.
    
    Returns:
        True if all dependencies are available, False otherwise
    
    **Validates: Requirements 12.7, 12.8**
    """
    logger = get_logger(__name__)
    missing_packages = []
    
    # Detect if we're on Android
    from llm_benchmark.hardware.detector import HardwareDetector
    hw_info = HardwareDetector.detect()
    is_android = hw_info.os_type == "android"
    
    # Check required packages
    required_packages = {
        'psutil': 'psutil',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'huggingface_hub': 'huggingface-hub'
    }
    
    # llama-cpp-python is only required on non-Android platforms
    if not is_android:
        required_packages['llama_cpp'] = 'llama-cpp-python'
    
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing_packages.append(package_name)
    
    # Check optional packages
    optional_packages = {
        'pynvml': 'pynvml (for GPU monitoring)',
        'yaml': 'pyyaml (for YAML config files)'
    }
    
    # On Android, llama-cpp-python is optional (we use native llama.cpp)
    if is_android:
        optional_packages['llama_cpp'] = 'llama-cpp-python (optional - using native llama.cpp)'
    
    for module_name, package_desc in optional_packages.items():
        try:
            __import__(module_name)
        except (ImportError, RuntimeError) as e:
            # RuntimeError can occur on Android when llama-cpp-python is installed but unsupported
            if isinstance(e, RuntimeError) and "Unsupported platform" in str(e):
                logger.warning(f"Optional package installed but unsupported on this platform: {package_desc}")
            else:
                logger.warning(f"Optional package not installed: {package_desc}")
    
    if missing_packages:
        logger.error("Missing required dependencies:")
        for package in missing_packages:
            logger.error(f"  - {package}")
        logger.info("\nInstall missing packages with:")
        if is_android:
            logger.info("On Android/Termux:")
            logger.info("  pkg install python-numpy python-psutil python-pandas python-matplotlib python-scipy python-pyyaml")
            logger.info("  pip install seaborn huggingface-hub")
        else:
            logger.info(f"  pip install {' '.join(missing_packages)}")
        return False
    
    logger.info("All required dependencies are installed")
    
    # On Android, check for native llama.cpp
    if is_android:
        from pathlib import Path
        llama_cli = Path("~/llama.cpp/build/bin/llama-cli").expanduser()
        if llama_cli.exists():
            logger.info("✅ Native llama.cpp found at ~/llama.cpp/build/bin/llama-cli")
        else:
            logger.warning("⚠️  Native llama.cpp not found at ~/llama.cpp/build/bin/llama-cli")
            logger.warning("   Build it with:")
            logger.warning("     cd ~/llama.cpp")
            logger.warning("     cmake -B build && cmake --build build -j4")
    
    return True


def acquire_models(config: BenchmarkConfig, model_manager: ModelManager) -> Dict[str, str]:
    """
    Acquire all models specified in configuration.
    
    Implements comprehensive error handling for model acquisition:
    - Network failures with exponential backoff retry
    - Authentication failures with clear error messages
    - Disk space exhaustion with informative error
    - Corrupted downloads with checksum verification
    - Skips model and continues with available models on failure
    
    Args:
        config: Benchmark configuration
        model_manager: Model manager instance
    
    Returns:
        Dictionary mapping quantization level to local model path
    
    **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 12.1**
    """
    logger = get_logger(__name__)
    logger.info("=" * 80)
    logger.info("Acquiring Models")
    logger.info("=" * 80)
    
    model_paths = {}
    
    for quant, filename in config.models.items():
        logger.info(f"\nAcquiring model: {quant} ({filename})")
        
        try:
            model_info = model_manager.get_model(
                repo_id=config.repo_id,
                filename=filename
            )
            
            if model_info is None:
                logger.warning(f"Failed to acquire {quant}, skipping this model")
                continue
            
            model_paths[quant] = model_info.local_path
            logger.info(f"✓ {quant}: {model_info.local_path} ({model_info.size_mb:.2f} MB)")
            
        except Exception as e:
            logger.error(f"Failed to acquire {quant}: {e}")
            logger.warning(f"Skipping {quant} and continuing with available models")
            continue
    
    if not model_paths:
        raise RuntimeError("No models could be acquired. Cannot proceed with benchmarking.")
    
    logger.info(f"\nSuccessfully acquired {len(model_paths)}/{len(config.models)} models")
    return model_paths


def run_quantization_profiling(
    config: BenchmarkConfig,
    model_paths: Dict[str, str],
    profiler: QuantizationProfiler
) -> List[QuantizationResult]:
    """
    Run quantization profiling tests for all models with multiple iterations.
    
    Args:
        config: Benchmark configuration
        model_paths: Dictionary mapping quantization to model path
        profiler: Quantization profiler instance
    
    Returns:
        List of QuantizationResult objects (one per model per iteration)
    
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7**
    """
    logger = get_logger(__name__)
    logger.info("=" * 80)
    logger.info("Running Quantization Profiling Tests")
    logger.info("=" * 80)
    logger.info(f"Iterations per model: {config.iterations}")
    
    # Use a standard test prompt
    test_prompt = (
        "Explain the concept of quantization in machine learning models "
        "and its impact on inference performance."
    )
    
    all_results = []
    
    # Run multiple iterations
    for iteration in range(config.iterations):
        logger.info(f"\n{'='*60}")
        logger.info(f"Iteration {iteration + 1}/{config.iterations}")
        logger.info(f"{'='*60}")
        
        results = profiler.profile_all(
            models=model_paths,
            prompt=test_prompt,
            max_tokens=config.max_tokens,
            warmup_tokens=5,
            context_size=config.context_size
        )
        
        # Add iteration number to each result
        for result in results:
            result.iteration = iteration + 1
        
        all_results.extend(results)
        
        # Sleep between iterations if configured
        if iteration < config.iterations - 1 and config.sleep_between_tests_s > 0:
            logger.info(f"\nSleeping {config.sleep_between_tests_s}s before next iteration...")
            time.sleep(config.sleep_between_tests_s)
    
    logger.info(f"\n✓ Quantization profiling complete: {len(all_results)} total results ({config.iterations} iterations × {len(model_paths)} models)")
    return all_results


def run_ablation_studies(
    config: BenchmarkConfig,
    model_paths: Dict[str, str],
    orchestrator: TestOrchestrator
) -> List[AblationResult]:
    """
    Run ablation studies if enabled.
    
    Args:
        config: Benchmark configuration
        model_paths: Dictionary mapping quantization to model path
        orchestrator: Test orchestrator instance
    
    Returns:
        List of AblationResult objects
    
    **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9**
    """
    logger = get_logger(__name__)
    
    if not config.enable_ablation_studies:
        logger.info("Ablation studies disabled, skipping")
        return []
    
    logger.info("=" * 80)
    logger.info("Running Ablation Studies")
    logger.info("=" * 80)
    
    try:
        results = orchestrator.run_ablation_tests(model_paths=model_paths)
        logger.info(f"\n✓ Ablation studies complete: {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Ablation studies failed: {e}", exc_info=True)
        logger.warning("Continuing without ablation results")
        return []


def run_batch_processing_tests(
    config: BenchmarkConfig,
    model_paths: Dict[str, str],
    orchestrator: TestOrchestrator
) -> List[AblationResult]:
    """
    Run batch processing tests if enabled.
    
    Args:
        config: Benchmark configuration
        model_paths: Dictionary mapping quantization to model path
        orchestrator: Test orchestrator instance
    
    Returns:
        List of AblationResult objects
    
    **Validates: Requirements 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8**
    """
    logger = get_logger(__name__)
    
    if not config.enable_batch_testing:
        logger.info("Batch processing tests disabled, skipping")
        return []
    
    logger.info("=" * 80)
    logger.info("Running Batch Processing Tests")
    logger.info("=" * 80)
    
    try:
        results = orchestrator.run_batch_tests()
        logger.info(f"\n✓ Batch processing tests complete: {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Batch processing tests failed: {e}", exc_info=True)
        logger.warning("Continuing without batch results")
        return []


def perform_statistical_validation(
    quantization_results: List[QuantizationResult],
    config: BenchmarkConfig
) -> List:
    """
    Perform statistical validation on results.
    
    Args:
        quantization_results: Quantization profiling results (multiple iterations)
        config: Benchmark configuration
    
    Returns:
        List of StatisticalSummary objects
    
    **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8**
    """
    logger = get_logger(__name__)
    
    if config.iterations < 3:
        logger.warning(
            f"Statistical validation requires at least 3 iterations, "
            f"but only {config.iterations} configured. Skipping statistical validation."
        )
        return []
    
    if not quantization_results:
        logger.warning("No quantization results available for statistical validation")
        return []
    
    logger.info("=" * 80)
    logger.info("Performing Statistical Validation")
    logger.info("=" * 80)
    
    try:
        from llm_benchmark.statistics import StatisticalValidator
        
        validator = StatisticalValidator()
        all_summaries = []
        
        # Group results by quantization level
        results_by_quant = {}
        for result in quantization_results:
            if result.quantization not in results_by_quant:
                results_by_quant[result.quantization] = []
            results_by_quant[result.quantization].append(result)
        
        logger.info(f"Analyzing {len(results_by_quant)} quantization levels with {config.iterations} iterations each")
        
        # Analyze each quantization level
        for quant, results in results_by_quant.items():
            logger.info(f"\nAnalyzing {quant} ({len(results)} iterations)...")
            
            # Convert results to dictionaries of metrics
            runs = []
            for result in results:
                runs.append({
                    'ttft_ms': result.ttft_ms,
                    'prefill_tps': result.prefill_tps,
                    'decode_tps': result.decode_tps,
                    'load_time_s': result.load_time_s,
                    'peak_ram_mb': result.peak_ram_mb,
                    'ram_increase_mb': result.ram_increase_mb
                })
            
            # Calculate statistical summaries
            summaries = validator.summarize_runs(runs)
            
            # Add quantization level to each summary
            for summary in summaries:
                summary.quantization = quant
            
            all_summaries.extend(summaries)
            
            # Log summary statistics
            for summary in summaries:
                logger.info(f"  {summary.metric_name}:")
                logger.info(f"    Mean: {summary.mean:.2f}")
                logger.info(f"    Std Dev: {summary.std_dev:.2f}")
                logger.info(f"    95% CI: [{summary.confidence_interval_95[0]:.2f}, {summary.confidence_interval_95[1]:.2f}]")
                if summary.outliers:
                    logger.info(f"    Outliers: {summary.outliers}")
        
        logger.info(f"\n✓ Statistical validation complete: {len(all_summaries)} summaries generated")
        return all_summaries
        
    except ImportError as e:
        logger.error(f"Failed to import StatisticalValidator: {e}")
        logger.warning("Skipping statistical validation")
        return []
    except Exception as e:
        logger.error(f"Statistical validation failed: {e}", exc_info=True)
        logger.warning("Continuing without statistical validation")
        return []


def generate_visualizations(
    quantization_results: List[QuantizationResult],
    ablation_results: List[AblationResult],
    batch_results: List[AblationResult],
    statistical_summaries: List,
    config: BenchmarkConfig
) -> List[str]:
    """
    Generate visualizations from results.
    
    Args:
        quantization_results: Quantization profiling results
        ablation_results: Ablation study results
        batch_results: Batch processing results
        statistical_summaries: Statistical summaries with confidence intervals
        config: Benchmark configuration
    
    Returns:
        List of paths to generated visualization files
    
    **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7**
    """
    logger = get_logger(__name__)
    logger.info("=" * 80)
    logger.info("Generating Visualizations")
    logger.info("=" * 80)
    
    try:
        viz_gen = VisualizationGenerator(
            output_dir=config.output_dir,
            dpi=config.visualization_dpi
        )
        
        # Get sample metrics for throughput plot
        sample_metrics = None
        if quantization_results:
            # We don't have direct access to InferenceMetrics here
            # This would need to be passed through or stored
            pass
        
        visualization_paths = viz_gen.generate_all_visualizations(
            quantization_results=quantization_results,
            ablation_results=ablation_results,
            statistical_summaries=statistical_summaries,
            sample_metrics=sample_metrics
        )
        
        logger.info(f"\n✓ Generated {len(visualization_paths)} visualizations")
        for path in visualization_paths:
            logger.info(f"  - {path}")
        
        return visualization_paths
        
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}", exc_info=True)
        logger.warning("Continuing without visualizations")
        return []


def generate_reports(
    benchmark_run: BenchmarkRun,
    config: BenchmarkConfig
) -> None:
    """
    Generate reports in all specified formats.
    
    Args:
        benchmark_run: Complete benchmark run results
        config: Benchmark configuration
    
    Raises:
        OSError: If run directory creation fails
        Exception: If all format saves fail
    
    **Validates: Requirements 8.6, 8.7, 9.8**
    """
    logger = get_logger(__name__)
    logger.info("=" * 80)
    logger.info("Generating Reports")
    logger.info("=" * 80)
    
    # Create run directory (let errors propagate)
    persistence = ResultsPersistence(output_dir=config.output_dir)
    logger.debug(f"Output directory (expanded): {persistence.output_dir}")
    logger.debug(f"Output directory exists: {persistence.output_dir.exists()}")
    logger.debug(f"Output directory is writable: {os.access(persistence.output_dir.parent, os.W_OK)}")
    run_dir = persistence.create_run_directory(benchmark_run.run_id)
    logger.info(f"✓ Run directory: {run_dir}")
    
    # Track save results
    save_results = {"succeeded": [], "failed": []}
    
    # Save in all requested formats
    for format_type in config.save_formats:
        if format_type == "json":
            try:
                json_path = run_dir / "results.json"
                persistence.save_json(benchmark_run, json_path)
                logger.info(f"✓ JSON report: {json_path}")
                save_results["succeeded"].append(("json", str(json_path)))
            except Exception as e:
                logger.error(f"✗ JSON report save failed: {e}")
                save_results["failed"].append(("json", str(e)))
        elif format_type == "csv":
            try:
                csv_path = run_dir / "results.csv"
                persistence.save_csv(benchmark_run, csv_path)
                logger.info(f"✓ CSV report: {csv_path}")
                save_results["succeeded"].append(("csv", str(csv_path)))
            except Exception as e:
                logger.error(f"✗ CSV report save failed: {e}")
                save_results["failed"].append(("csv", str(e)))
        elif format_type == "markdown":
            try:
                md_path = run_dir / "results.md"
                persistence.save_markdown(benchmark_run, md_path)
                logger.info(f"✓ Markdown report: {md_path}")
                save_results["succeeded"].append(("markdown", str(md_path)))
            except Exception as e:
                logger.error(f"✗ MARKDOWN report save failed: {e}")
                save_results["failed"].append(("markdown", str(e)))
        elif format_type == "html":
            # Generate HTML report in run directory
            if benchmark_run.visualization_paths:
                try:
                    viz_gen = VisualizationGenerator(
                        output_dir=str(run_dir),
                        dpi=config.visualization_dpi
                    )
                    html_path = viz_gen.generate_html_report(benchmark_run, benchmark_run.visualization_paths)
                    benchmark_run.html_report_path = html_path
                    logger.info(f"✓ HTML report: {html_path}")
                    save_results["succeeded"].append(("html", str(html_path)))
                except Exception as e:
                    logger.error(f"✗ HTML report save failed: {e}")
                    save_results["failed"].append(("html", str(e)))
            else:
                logger.warning("HTML report skipped (no visualizations available)")
    
    # Report summary
    if save_results["failed"]:
        failed_formats = [fmt for fmt, _ in save_results["failed"]]
        logger.warning(
            f"Report generation completed with failures. "
            f"Failed formats: {', '.join(failed_formats)}"
        )
        
        # If ALL formats failed, raise an exception
        if not save_results["succeeded"]:
            raise Exception(
                f"All report formats failed to save. Errors: "
                f"{'; '.join([f'{fmt}: {err}' for fmt, err in save_results['failed']])}"
            )
    else:
        logger.info("✓ All reports generated successfully")


def main():
    """
    Main entry point for the benchmark framework.
    
    Orchestrates the complete benchmark workflow:
    1. Validate dependencies
    2. Parse configuration
    3. Detect hardware and create backend
    4. Initialize model manager, metrics collector, and orchestrator
    5. Acquire models
    6. Run quantization profiling tests
    7. Run ablation studies (if enabled)
    8. Run batch processing tests (if enabled)
    9. Perform statistical validation
    10. Generate visualizations
    11. Generate reports
    12. Handle errors gracefully
    
    **Validates: Requirements 7.1, 7.2, 7.6, 7.7, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 12.7**
    """
    # Parse command-line arguments
    parser = ConfigParser.create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = ConfigParser.load_config(args)
        
        # Set up logging
        setup_logging(config.output_dir)
        logger = get_logger(__name__)
        
        logger.info("=" * 80)
        logger.info("LLM Inference Benchmark Framework")
        logger.info("=" * 80)
        logger.info(f"Run ID: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
        logger.info(f"Output directory: {config.output_dir}")
        
        # Track start time for duration calculation
        start_time = time.time()
        
        # Step 1: Validate dependencies
        logger.info("\n" + "=" * 80)
        logger.info("Step 1: Validating Dependencies")
        logger.info("=" * 80)
        
        if not validate_dependencies():
            logger.error("Dependency validation failed. Cannot proceed.")
            return 1
        
        # Step 2: Detect hardware and create backend
        logger.info("\n" + "=" * 80)
        logger.info("Step 2: Detecting Hardware Platform")
        logger.info("=" * 80)
        
        hw_info = HardwareDetector.detect()
        backend = create_backend(hw_info)
        backend.optimize_for_inference()
        
        logger.info(f"\nPlatform: {hw_info.os_type}")
        logger.info(f"CPU: {hw_info.cpu_model} ({hw_info.cpu_cores} cores)")
        logger.info(f"RAM: {hw_info.total_ram_gb:.2f} GB")
        if hw_info.has_gpu:
            logger.info(f"GPU: {hw_info.gpu_model} ({hw_info.gpu_memory_gb:.2f} GB)")
        
        # Step 3: Initialize components
        logger.info("\n" + "=" * 80)
        logger.info("Step 3: Initializing Components")
        logger.info("=" * 80)
        
        # Initialize model manager
        model_manager = ModelManager(
            cache_dir=config.model_cache_dir,
            hf_token=config.hf_token or os.environ.get("HF_TOKEN")
        )
        logger.info(f"✓ Model manager initialized (cache: {config.model_cache_dir})")
        
        # Initialize metrics collector
        metrics_collector = backend.get_metrics_collector()
        logger.info("✓ Metrics collector initialized")
        
        # Initialize quantization profiler
        profiler = QuantizationProfiler(
            backend=backend,
            metrics_collector=metrics_collector
        )
        logger.info("✓ Quantization profiler initialized")
        
        # Initialize test orchestrator
        orchestrator = TestOrchestrator(
            config=config,
            backend=backend
        )
        logger.info("✓ Test orchestrator initialized")
        
        # Step 4: Acquire models
        logger.info("\n" + "=" * 80)
        logger.info("Step 4: Acquiring Models")
        logger.info("=" * 80)
        
        model_paths = acquire_models(config, model_manager)
        
        # Step 5: Run quantization profiling
        logger.info("\n" + "=" * 80)
        logger.info("Step 5: Running Quantization Profiling")
        logger.info("=" * 80)
        
        quantization_results = run_quantization_profiling(
            config=config,
            model_paths=model_paths,
            profiler=profiler
        )
        
        # Step 6: Run ablation studies (if enabled)
        ablation_results = run_ablation_studies(
            config=config,
            model_paths=model_paths,
            orchestrator=orchestrator
        )
        
        # Step 7: Run batch processing tests (if enabled)
        batch_results = run_batch_processing_tests(
            config=config,
            model_paths=model_paths,
            orchestrator=orchestrator
        )
        
        # Step 8: Perform statistical validation
        logger.info("\n" + "=" * 80)
        logger.info("Step 8: Performing Statistical Validation")
        logger.info("=" * 80)
        
        statistical_summaries = perform_statistical_validation(
            quantization_results=quantization_results,
            config=config
        )
        
        # Step 9: Generate visualizations
        logger.info("\n" + "=" * 80)
        logger.info("Step 9: Generating Visualizations")
        logger.info("=" * 80)
        
        visualization_paths = generate_visualizations(
            quantization_results=quantization_results,
            ablation_results=ablation_results,
            batch_results=batch_results,
            statistical_summaries=statistical_summaries,
            config=config
        )
        
        # Step 10: Create benchmark run object
        logger.info("\n" + "=" * 80)
        logger.info("Step 10: Creating Benchmark Run Object")
        logger.info("=" * 80)
        
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get software versions
        software_versions = {}
        try:
            software_versions['python'] = sys.version.split()[0]
        except:
            pass
        
        try:
            import llama_cpp
            software_versions['llama-cpp-python'] = llama_cpp.__version__
        except:
            pass
        
        # Get model checksums
        model_checksums = {}
        for quant, path in model_paths.items():
            try:
                import hashlib
                sha256_hash = hashlib.sha256()
                with open(path, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        sha256_hash.update(chunk)
                model_checksums[quant] = sha256_hash.hexdigest()
            except:
                model_checksums[quant] = "unknown"
        
        # Calculate benchmark duration
        end_time = time.time()
        duration_s = round(end_time - start_time, 2)
        
        benchmark_run = BenchmarkRun(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            duration_s=duration_s,
            hardware_info=hw_info,
            software_versions=software_versions,
            config=config.__dict__,
            model_checksums=model_checksums,
            quantization_results=quantization_results,
            ablation_results=ablation_results,
            batch_results=batch_results,
            statistical_summaries=statistical_summaries,
            comparisons=[],
            visualization_paths=visualization_paths,
            html_report_path=""
        )
        
        logger.info("✓ Benchmark run object created")
        
        # Step 11: Generate reports
        logger.info("\n" + "=" * 80)
        logger.info("Step 11: Generating Reports")
        logger.info("=" * 80)
        
        generate_reports(benchmark_run, config)
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK COMPLETE")
        logger.info("=" * 80)
        logger.info(f"✓ Quantization results: {len(quantization_results)}")
        logger.info(f"✓ Ablation results: {len(ablation_results)}")
        logger.info(f"✓ Batch results: {len(batch_results)}")
        logger.info(f"✓ Visualizations: {len(visualization_paths)}")
        logger.info(f"✓ Output directory: {config.output_dir}")
        logger.info("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        logger = get_logger(__name__)
        logger.warning("\nBenchmark interrupted by user")
        return 130
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"\nBenchmark failed: {e}", exc_info=True)
        print(f"\nERROR: {e}", file=sys.stderr)
        print("\nFor detailed error information, check the log files in the output directory.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
