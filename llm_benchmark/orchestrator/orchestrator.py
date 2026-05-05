"""
Test orchestrator for automated benchmark execution.

Manages test execution flow, warmup runs, garbage collection,
thermal stabilization, and result checkpointing.
"""

import gc
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_benchmark.config import BenchmarkConfig
from llm_benchmark.hardware.hal import HardwareBackend
from llm_benchmark.metrics.collector import MetricsCollector
from llm_benchmark.models import AblationResult, BenchmarkRun, HardwareInfo
from llm_benchmark.profiler.ablation import AblationEngine

logger = logging.getLogger(__name__)


class TestConfig:
    """
    Configuration for test orchestration.
    
    This is an alias for BenchmarkConfig to match the task requirements.
    All configuration logic is handled by BenchmarkConfig.
    """
    
    def __init__(self, config: BenchmarkConfig):
        """
        Initialize TestConfig from BenchmarkConfig.
        
        Args:
            config: BenchmarkConfig instance
        """
        self._config = config
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to underlying BenchmarkConfig."""
        return getattr(self._config, name)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'TestConfig':
        """
        Load configuration from JSON or YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            TestConfig instance
        """
        from llm_benchmark.config import ConfigParser
        config = ConfigParser.from_file(config_path)
        return cls(config)
    
    @classmethod
    def from_args(cls, args) -> 'TestConfig':
        """
        Create configuration from command-line arguments.
        
        Args:
            args: Parsed command-line arguments
            
        Returns:
            TestConfig instance
        """
        from llm_benchmark.config import ConfigParser
        config = ConfigParser.from_args(args)
        return cls(config)
    
    @property
    def config(self) -> BenchmarkConfig:
        """Get underlying BenchmarkConfig."""
        return self._config


class TestOrchestrator:
    """
    Orchestrates automated test execution with warmup, thermal stabilization,
    and checkpointing.
    
    Responsibilities:
    - Execute test suite with proper warmup and stabilization
    - Enforce garbage collection between test cases
    - Monitor thermal state and wait for stabilization
    - Catch exceptions per test case and continue execution
    - Save intermediate results to checkpoint files
    - Generate final summary report with pass/fail status
    """
    
    def __init__(self, config: BenchmarkConfig, backend: HardwareBackend):
        """
        Initialize test orchestrator.
        
        Args:
            config: Benchmark configuration
            backend: Hardware backend for platform-specific operations
        """
        self.config = config
        self.backend = backend
        self.hw_info = backend.hw_info
        self.metrics_collector = backend.get_metrics_collector()
        
        # Create output directories
        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"
        
        for directory in [self.output_dir, self.checkpoint_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Test execution state
        self.test_results: Dict[str, Any] = {
            'passed': [],
            'failed': [],
            'skipped': []
        }
        
        logger.info("TestOrchestrator initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Warmup runs: {config.warmup_runs}")
        logger.info(f"Sleep between tests: {config.sleep_between_tests_s}s")
        logger.info(f"Thermal threshold: {config.thermal_stabilization_threshold_c}°C")
    
    def run_all_tests(self) -> BenchmarkRun:
        """
        Execute complete test suite.
        
        Returns:
            BenchmarkRun with all results
            
        Raises:
            Exception: If core functionality fails
        """
        logger.info("=" * 80)
        logger.info("Starting benchmark test suite")
        logger.info("=" * 80)
        
        start_time = time.time()
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize benchmark run
        benchmark_run = BenchmarkRun(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            duration_s=0.0,
            hardware_info=self.hw_info,
            software_versions=self._get_software_versions(),
            config=self._config_to_dict(),
            model_checksums={}
        )
        
        # Run quantization profiling (core functionality)
        if self.config.enable_quantization_profiling:
            try:
                logger.info("\n" + "=" * 80)
                logger.info("Running quantization profiling tests")
                logger.info("=" * 80)
                
                quantization_results = self.run_quantization_tests()
                benchmark_run.quantization_results = quantization_results
                
                # Save checkpoint
                self._save_checkpoint(benchmark_run, "quantization_complete")
                
                # Enforce garbage collection before next test phase
                self._enforce_garbage_collection()
                
                self.test_results['passed'].append({
                    'test': 'quantization_profiling',
                    'status': 'passed',
                    'results_count': len(quantization_results)
                })
                
            except Exception as e:
                logger.error(f"Quantization profiling failed: {e}", exc_info=True)
                self.test_results['failed'].append({
                    'test': 'quantization_profiling',
                    'status': 'failed',
                    'error': str(e)
                })
                # Core functionality failed - cannot continue
                raise
        
        # Run ablation studies (optional)
        if self.config.enable_ablation_studies:
            try:
                logger.info("\n" + "=" * 80)
                logger.info("Running ablation studies")
                logger.info("=" * 80)
                
                ablation_results = self.run_ablation_tests()
                benchmark_run.ablation_results = ablation_results
                
                # Save checkpoint
                self._save_checkpoint(benchmark_run, "ablation_complete")
                
                # Enforce garbage collection before next test phase
                self._enforce_garbage_collection()
                
                self.test_results['passed'].append({
                    'test': 'ablation_studies',
                    'status': 'passed',
                    'results_count': len(ablation_results)
                })
                
            except Exception as e:
                logger.error(f"Ablation studies failed: {e}", exc_info=True)
                logger.warning("Continuing without ablation results")
                self.test_results['failed'].append({
                    'test': 'ablation_studies',
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Run batch testing (optional)
        if self.config.enable_batch_testing:
            try:
                logger.info("\n" + "=" * 80)
                logger.info("Running batch processing tests")
                logger.info("=" * 80)
                
                batch_results = self.run_batch_tests()
                benchmark_run.batch_results = batch_results
                
                # Save checkpoint
                self._save_checkpoint(benchmark_run, "batch_complete")
                
                self.test_results['passed'].append({
                    'test': 'batch_testing',
                    'status': 'passed',
                    'results_count': len(batch_results)
                })
                
            except Exception as e:
                logger.error(f"Batch testing failed: {e}", exc_info=True)
                logger.warning("Continuing without batch results")
                self.test_results['failed'].append({
                    'test': 'batch_testing',
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Calculate total duration
        end_time = time.time()
        benchmark_run.duration_s = round(end_time - start_time, 2)
        
        # Generate summary report
        self._generate_summary_report(benchmark_run)
        
        logger.info("\n" + "=" * 80)
        logger.info("Benchmark test suite complete")
        logger.info(f"Duration: {benchmark_run.duration_s}s")
        logger.info(f"Passed: {len(self.test_results['passed'])}")
        logger.info(f"Failed: {len(self.test_results['failed'])}")
        logger.info(f"Skipped: {len(self.test_results['skipped'])}")
        logger.info("=" * 80)
        
        return benchmark_run
    
    def run_quantization_tests(self) -> List[Any]:
        """
        Run quantization profiling tests.
        
        Returns:
            List of QuantizationResult objects
        """
        # Placeholder - will be implemented in future tasks
        logger.info("Quantization profiling not yet implemented")
        return []
    
    def run_ablation_tests(self, model_paths: Dict[str, str] = None) -> List[AblationResult]:
        """
        Run ablation studies.
        
        Args:
            model_paths: Dictionary mapping quantization to model file path
        
        Returns:
            List of AblationResult objects
        """
        logger.info("Starting ablation studies...")
        
        # Create appropriate ablation engine based on platform
        if self.backend.hw_info.os_type == "android":
            from llm_benchmark.profiler.android_ablation import AndroidAblationEngine
            logger.info("Using AndroidAblationEngine for native llama.cpp")
            ablation_engine = AndroidAblationEngine(
                backend=self.backend,
                metrics_collector=self.metrics_collector,
                context_size=self.config.context_size
            )
        else:
            from llm_benchmark.profiler.ablation import AblationEngine
            logger.info("Using standard AblationEngine for llama-cpp-python")
            ablation_engine = AblationEngine(
                backend=self.backend,
                metrics_collector=self.metrics_collector,
                context_size=self.config.context_size
            )
        
        # Get first model for ablation testing
        # Use provided model_paths if available, otherwise try to construct paths
        if model_paths and len(model_paths) > 0:
            # Use the first available model from provided paths
            model_path = list(model_paths.values())[0]
            logger.info(f"Using model for ablation: {model_path}")
        else:
            # Fallback: try to construct paths (old behavior)
            available_models = {}
            for quant, filename in self.config.models.items():
                model_path = os.path.join(self.config.model_cache_dir, filename)
                if os.path.exists(model_path):
                    available_models[quant] = model_path
                    break  # Use first available model
            
            if not available_models:
                logger.warning("No models available for ablation testing")
                return []
            
            model_path = list(available_models.values())[0]
            logger.info(f"Using model for ablation: {model_path}")
        
        # Generate test prompts with substantial shared prefix
        # Adjust prompt length based on context size to avoid exceeding limits
        context_size = getattr(self.config, 'context_size', 2048)
        max_prompt_tokens = int(context_size * 0.8)  # Use 80% of context for prompt
        prompt_prefix = self._generate_long_prompt_prefix(max_tokens=max_prompt_tokens)
        prompt_suffix = " What are the key takeaways from this analysis?"
        
        # Run ablation studies (method differs by platform)
        if self.backend.hw_info.os_type == "android":
            # Android uses prompt cache strategies (--prompt-cache flag)
            results = ablation_engine.test_prompt_cache_strategies(
                model_path=model_path,
                prompt_prefix=prompt_prefix,
                prompt_suffix=prompt_suffix,
                max_tokens=50
            )
        else:
            # Standard platform uses KV cache strategies (llama-cpp-python API)
            results = ablation_engine.test_kv_cache_strategies(
                model_path=model_path,
                prompt_prefix=prompt_prefix,
                prompt_suffix=prompt_suffix,
                max_tokens=50,
                cache_types=self.config.kv_cache_types
            )
        
        logger.info(f"Ablation studies complete. Generated {len(results)} results.")
        
        return results
    
    def _generate_long_prompt_prefix(self, max_tokens: int = 500) -> str:
        """
        Generate a long prompt prefix for cache effectiveness testing.
        
        Args:
            max_tokens: Maximum number of tokens to generate (approximate)
        
        Returns:
            Long prompt prefix string, truncated to fit within max_tokens
        """
        # Generate a substantial prompt about a technical topic
        # This ensures we have enough tokens for effective cache testing
        full_prefix = """
        Artificial Intelligence and Machine Learning have revolutionized numerous industries 
        over the past decade. From healthcare to finance, from transportation to entertainment, 
        AI systems are now integral to modern society. The development of large language models 
        represents a significant milestone in this journey, enabling machines to understand and 
        generate human-like text with unprecedented accuracy.
        
        The architecture of modern language models is based on the transformer design, which was 
        introduced in the seminal paper "Attention is All You Need" by Vaswani et al. in 2017. 
        This architecture relies on self-attention mechanisms that allow the model to weigh the 
        importance of different words in a sequence when processing each word. The key innovation 
        was the ability to process sequences in parallel rather than sequentially, dramatically 
        improving training efficiency.
        
        Large language models are trained on massive datasets containing billions of tokens from 
        diverse sources including books, websites, scientific papers, and code repositories. The 
        training process involves predicting the next token in a sequence, which forces the model 
        to learn patterns, relationships, and structures in language. Through this process, models 
        develop emergent capabilities such as reasoning, translation, summarization, and even 
        basic mathematical problem-solving.
        
        The scale of these models has grown exponentially. Early transformer models had millions 
        of parameters, while modern models can have hundreds of billions of parameters. This 
        scaling has led to improved performance across a wide range of tasks, though it also 
        raises important questions about computational efficiency, environmental impact, and 
        accessibility. Researchers are actively exploring techniques to make these models more 
        efficient, including quantization, pruning, distillation, and sparse architectures.
        
        Quantization is particularly important for deploying large language models in 
        resource-constrained environments. By reducing the precision of model weights from 
        32-bit floating point to 8-bit or even 4-bit integers, we can significantly reduce 
        memory requirements and improve inference speed with minimal impact on model quality. 
        Different quantization schemes offer different tradeoffs between model size, speed, 
        and accuracy. For example, Q8_0 quantization maintains high accuracy while reducing 
        model size by approximately 75%, while Q4_0 quantization achieves even greater 
        compression at the cost of some accuracy degradation.
        
        The deployment of language models also involves important considerations around 
        caching and optimization. Key-Value (KV) caching is a crucial technique that stores 
        intermediate computations from the attention mechanism, allowing subsequent tokens 
        to be generated more efficiently. When processing a prompt, the model computes key 
        and value vectors for each token. By caching these vectors, we can avoid recomputing 
        them when generating output tokens, significantly reducing the time to first token 
        (TTFT) and improving overall throughput.
        
        There are different strategies for implementing KV caching. RAM-based caching stores 
        the key-value pairs in system memory, providing fast access but consuming significant 
        memory resources. Disk-based caching offloads the cache to storage, reducing memory 
        pressure but introducing I/O latency. The choice between these strategies depends on 
        the specific deployment scenario, available hardware resources, and performance 
        requirements. In practice, hybrid approaches that combine both strategies can offer 
        the best balance of performance and resource utilization.
        """
        
        # Approximate token count (1 token ≈ 4 characters)
        full_text = full_prefix.strip()
        estimated_tokens = len(full_text) // 4
        
        if estimated_tokens <= max_tokens:
            return full_text
        
        # Truncate to fit within max_tokens
        target_chars = max_tokens * 4
        truncated = full_text[:target_chars]
        
        # Find the last complete sentence to avoid cutting mid-sentence
        last_period = truncated.rfind('.')
        if last_period > target_chars * 0.8:  # Only truncate if we don't lose too much
            truncated = truncated[:last_period + 1]
        
        return truncated
    
    def run_batch_tests(self) -> List[Any]:
        """
        Run batch processing tests.
        
        Returns:
            List of AblationResult objects
        """
        # Placeholder - will be implemented in future tasks
        logger.info("Batch testing not yet implemented")
        return []
    
    def _warmup(self, llm) -> None:
        """
        Perform warmup runs to stabilize system state.
        
        Args:
            llm: Llama model instance
        """
        if self.config.warmup_runs <= 0:
            return
        
        logger.info(f"Performing {self.config.warmup_runs} warmup runs...")
        
        warmup_prompt = "Hello, world!"
        warmup_tokens = 5
        
        for i in range(self.config.warmup_runs):
            try:
                logger.debug(f"Warmup run {i + 1}/{self.config.warmup_runs}")
                _ = llm(warmup_prompt, max_tokens=warmup_tokens, stream=False)
            except Exception as e:
                logger.warning(f"Warmup run {i + 1} failed: {e}")
                # Continue with remaining warmup runs
        
        logger.info("Warmup complete")
    
    def _thermal_stabilization_delay(self) -> None:
        """
        Wait for thermal stabilization between tests.
        
        Implements comprehensive thermal throttling handling:
        - Checks thermal state before running tests
        - Waits for temperature to drop below threshold if throttled
        - Detects throttling during test execution
        - Flags results as thermally throttled if detected
        - Increases sleep delays between tests if needed
        
        Checks temperature thresholds and waits if system is too hot.
        """
        if not self.hw_info.has_thermal_sensors:
            # No thermal sensors - just use fixed sleep delay
            if self.config.sleep_between_tests_s > 0:
                logger.debug(f"Sleeping {self.config.sleep_between_tests_s}s between tests")
                time.sleep(self.config.sleep_between_tests_s)
            return
        
        # Check thermal state
        is_throttled, current_temp = self._check_thermal_state()
        
        # Initialize cooldown counter
        cooldown_wait_count = 0
        
        if is_throttled:
            logger.warning(
                f"System temperature high ({current_temp:.1f}°C), "
                f"waiting for cooldown below {self.config.thermal_stabilization_threshold_c}°C..."
            )
            
            # Wait for temperature to drop
            max_cooldown_wait = 60  # Maximum 10 minutes (60 * 10s)
            
            while is_throttled and cooldown_wait_count < max_cooldown_wait:
                time.sleep(10)  # Check every 10 seconds
                is_throttled, current_temp = self._check_thermal_state()
                cooldown_wait_count += 1
                
                if is_throttled:
                    logger.debug(f"Current temperature: {current_temp:.1f}°C (waiting {cooldown_wait_count * 10}s)")
            
            if is_throttled:
                logger.warning(
                    f"Temperature still high ({current_temp:.1f}°C) after {cooldown_wait_count * 10}s wait"
                )
                logger.info("Suggestions:")
                logger.info("  - Improve system cooling (check fans, airflow)")
                logger.info("  - Reduce workload intensity")
                logger.info("  - Increase sleep_between_tests_s")
                logger.info("  - Lower thermal_stabilization_threshold_c (not recommended)")
                logger.info("Continuing anyway, but results may be affected by thermal throttling")
            else:
                logger.info(f"Temperature stabilized at {current_temp:.1f}°C")
        
        # Apply standard sleep delay (possibly increased if thermal issues detected)
        sleep_delay = self.config.sleep_between_tests_s
        
        # If we had to wait for cooldown, increase sleep delay for next test
        if cooldown_wait_count > 0:
            # Increase delay by 50% if thermal issues detected
            sleep_delay = int(sleep_delay * 1.5)
            logger.info(f"Increasing sleep delay to {sleep_delay}s due to thermal concerns")
        
        if sleep_delay > 0:
            logger.debug(f"Sleeping {sleep_delay}s between tests")
            time.sleep(sleep_delay)
    
    def _check_thermal_state(self) -> tuple[bool, float]:
        """
        Check if system is thermally throttled.
        
        Returns:
            Tuple of (is_throttled, max_temperature)
        """
        if not self.hw_info.has_thermal_sensors:
            return False, 0.0
        
        temps = []
        
        # Get CPU temperature
        cpu_temp = self.metrics_collector._get_cpu_temperature()
        if cpu_temp is not None:
            temps.append(cpu_temp)
        
        # Get GPU temperature if available
        if self.hw_info.has_gpu:
            gpu_temp = self.metrics_collector._get_gpu_temperature()
            if gpu_temp is not None:
                temps.append(gpu_temp)
        
        if not temps:
            return False, 0.0
        
        max_temp = max(temps)
        is_throttled = max_temp > self.config.thermal_stabilization_threshold_c
        
        return is_throttled, max_temp
    
    def run_test_with_thermal_protection(self, test_fn, test_name: str = "test"):
        """
        Run a test function with thermal protection and throttling detection.
        
        Checks thermal state before running test, waits for stabilization if needed,
        and flags results if thermal throttling is detected during execution.
        
        Args:
            test_fn: Test function to execute
            test_name: Name of test for logging
        
        Returns:
            Test result with thermal_throttled flag if applicable
        """
        # Check thermal state before test
        is_throttled, temp = self._check_thermal_state()
        
        if is_throttled:
            logger.warning(
                f"System temperature high ({temp:.1f}°C) before {test_name}, "
                "waiting for cooldown..."
            )
            self._thermal_stabilization_delay()
        
        # Run the test
        result = test_fn()
        
        # Check if throttling occurred during test
        if result is not None and hasattr(result, 'thermal_throttled'):
            if result.thermal_throttled:
                logger.warning(f"Thermal throttling detected during {test_name}")
                logger.info("Results may be affected by thermal constraints")
        
        return result
    
    def _enforce_garbage_collection(self) -> None:
        """
        Enforce garbage collection between test cases.
        
        Logs memory usage before and after collection.
        """
        import psutil
        
        process = psutil.Process()
        
        # Memory before GC
        mem_before_mb = process.memory_info().rss / (1024 * 1024)
        
        logger.debug(f"Memory before GC: {mem_before_mb:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        # Memory after GC
        mem_after_mb = process.memory_info().rss / (1024 * 1024)
        freed_mb = mem_before_mb - mem_after_mb
        
        logger.debug(f"Memory after GC: {mem_after_mb:.2f} MB (freed {freed_mb:.2f} MB)")
    
    def _save_checkpoint(self, benchmark_run: BenchmarkRun, checkpoint_name: str) -> None:
        """
        Save intermediate results to checkpoint file.
        
        Args:
            benchmark_run: Current benchmark run state
            checkpoint_name: Name for this checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.json"
        
        try:
            # Convert benchmark run to dictionary
            checkpoint_data = {
                'run_id': benchmark_run.run_id,
                'timestamp': benchmark_run.timestamp,
                'checkpoint_name': checkpoint_name,
                'checkpoint_time': datetime.now().isoformat(),
                'hardware_info': self._hardware_info_to_dict(benchmark_run.hardware_info),
                'config': benchmark_run.config,
                'quantization_results': [
                    self._quantization_result_to_dict(r) 
                    for r in benchmark_run.quantization_results
                ],
                'ablation_results': [
                    self._ablation_result_to_dict(r) 
                    for r in benchmark_run.ablation_results
                ],
                'batch_results': [
                    self._ablation_result_to_dict(r) 
                    for r in benchmark_run.batch_results
                ],
                'test_status': self.test_results
            }
            
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            # Don't raise - checkpoint failure shouldn't stop execution
    
    def _generate_summary_report(self, benchmark_run: BenchmarkRun) -> None:
        """
        Generate final summary report with pass/fail status.
        
        Args:
            benchmark_run: Complete benchmark run results
        """
        summary_path = self.output_dir / "summary.txt"
        
        try:
            with open(summary_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("BENCHMARK SUMMARY REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Run ID: {benchmark_run.run_id}\n")
                f.write(f"Timestamp: {benchmark_run.timestamp}\n")
                f.write(f"Duration: {benchmark_run.duration_s}s\n\n")
                
                f.write("Hardware Information:\n")
                f.write(f"  OS Type: {benchmark_run.hardware_info.os_type}\n")
                f.write(f"  CPU: {benchmark_run.hardware_info.cpu_model}\n")
                f.write(f"  CPU Cores: {benchmark_run.hardware_info.cpu_cores}\n")
                f.write(f"  RAM: {benchmark_run.hardware_info.total_ram_gb:.2f} GB\n")
                f.write(f"  GPU: {benchmark_run.hardware_info.gpu_model or 'None'}\n\n")
                
                f.write("Test Results:\n")
                f.write(f"  Passed: {len(self.test_results['passed'])}\n")
                f.write(f"  Failed: {len(self.test_results['failed'])}\n")
                f.write(f"  Skipped: {len(self.test_results['skipped'])}\n\n")
                
                if self.test_results['passed']:
                    f.write("Passed Tests:\n")
                    for test in self.test_results['passed']:
                        f.write(f"  ✓ {test['test']}")
                        if 'results_count' in test:
                            f.write(f" ({test['results_count']} results)")
                        f.write("\n")
                    f.write("\n")
                
                if self.test_results['failed']:
                    f.write("Failed Tests:\n")
                    for test in self.test_results['failed']:
                        f.write(f"  ✗ {test['test']}: {test.get('error', 'Unknown error')}\n")
                    f.write("\n")
                
                if self.test_results['skipped']:
                    f.write("Skipped Tests:\n")
                    for test in self.test_results['skipped']:
                        f.write(f"  - {test['test']}: {test.get('reason', 'Unknown reason')}\n")
                    f.write("\n")
                
                f.write("=" * 80 + "\n")
            
            logger.info(f"Summary report saved: {summary_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
    
    def _get_software_versions(self) -> Dict[str, str]:
        """
        Get software version information.
        
        Returns:
            Dictionary of software versions
        """
        import sys
        
        versions = {
            'python': sys.version.split()[0]
        }
        
        # Try to get package versions
        try:
            import llama_cpp
            versions['llama-cpp-python'] = llama_cpp.__version__
        except:
            versions['llama-cpp-python'] = 'unknown'
        
        try:
            import psutil
            versions['psutil'] = psutil.__version__
        except:
            versions['psutil'] = 'unknown'
        
        # Try to get CUDA version if available
        if self.hw_info.has_gpu:
            try:
                # Try nvidia-ml-py3 first (modern replacement for pynvml)
                try:
                    import nvidia_ml_py3 as nvml
                except ImportError:
                    # Fallback to pynvml for backward compatibility
                    import pynvml as nvml
                
                nvml.nvmlInit()
                cuda_version = nvml.nvmlSystemGetCudaDriverVersion()
                versions['cuda'] = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
                nvml.nvmlShutdown()
            except:
                versions['cuda'] = 'unknown'
        
        return versions
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return {
            'repo_id': self.config.repo_id,
            'models': self.config.models,
            'model_cache_dir': self.config.model_cache_dir,
            'context_size': self.config.context_size,
            'batch_size': self.config.batch_size,
            'max_tokens': self.config.max_tokens,
            'iterations': self.config.iterations,
            'warmup_runs': self.config.warmup_runs,
            'enable_quantization_profiling': self.config.enable_quantization_profiling,
            'enable_ablation_studies': self.config.enable_ablation_studies,
            'enable_batch_testing': self.config.enable_batch_testing,
            'enable_thermal_monitoring': self.config.enable_thermal_monitoring,
            'sleep_between_tests_s': self.config.sleep_between_tests_s,
            'thermal_stabilization_threshold_c': self.config.thermal_stabilization_threshold_c,
            'inference_timeout_s': self.config.inference_timeout_s,
            'output_dir': self.config.output_dir,
            'save_formats': self.config.save_formats,
            'visualization_dpi': self.config.visualization_dpi
        }
    
    def _hardware_info_to_dict(self, hw_info: HardwareInfo) -> Dict[str, Any]:
        """Convert HardwareInfo to dictionary."""
        return {
            'os_type': hw_info.os_type,
            'cpu_model': hw_info.cpu_model,
            'cpu_cores': hw_info.cpu_cores,
            'cpu_features': hw_info.cpu_features,
            'total_ram_gb': hw_info.total_ram_gb,
            'available_ram_gb': hw_info.available_ram_gb,
            'has_gpu': hw_info.has_gpu,
            'gpu_model': hw_info.gpu_model,
            'gpu_memory_gb': hw_info.gpu_memory_gb,
            'gpu_compute_capability': hw_info.gpu_compute_capability,
            'has_thermal_sensors': hw_info.has_thermal_sensors,
            'has_power_sensors': hw_info.has_power_sensors
        }
    
    def _quantization_result_to_dict(self, result) -> Dict[str, Any]:
        """Convert QuantizationResult to dictionary."""
        return {
            'quantization': result.quantization,
            'load_time_s': result.load_time_s,
            'peak_ram_mb': result.peak_ram_mb,
            'ram_increase_mb': result.ram_increase_mb,
            'ttft_ms': result.ttft_ms,
            'prefill_tps': result.prefill_tps,
            'decode_tps': result.decode_tps,
            'prompt_tokens': result.prompt_tokens,
            'output_tokens': result.output_tokens,
            'gpu_memory_mb': result.gpu_memory_mb,
            'gpu_utilization_pct': result.gpu_utilization_pct
        }
    
    def _ablation_result_to_dict(self, result) -> Dict[str, Any]:
        """Convert AblationResult to dictionary."""
        return {
            'scenario': result.scenario,
            'configuration': result.configuration,
            'metrics': result.metrics,
            'improvement_over_baseline': result.improvement_over_baseline
        }
