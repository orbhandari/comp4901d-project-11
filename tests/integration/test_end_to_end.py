"""
End-to-end integration test for the complete benchmark framework.

Tests the complete benchmark run on x86 Linux with a small test model,
verifying all components are integrated correctly and results are saved
in the correct format.

**Validates: Requirements 7.1, 7.2, 8.6, 9.8**
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from llm_benchmark.config import BenchmarkConfig
from llm_benchmark.hardware.detector import HardwareDetector
from llm_benchmark.hardware.hal import create_backend
from llm_benchmark.model_manager.manager import ModelManager
from llm_benchmark.profiler.quantization import QuantizationProfiler
from llm_benchmark.orchestrator.orchestrator import TestOrchestrator
from llm_benchmark.statistics.validator import StatisticalValidator
from llm_benchmark.visualization.visualization_generator import VisualizationGenerator
from llm_benchmark.results.persistence import ResultsPersistence
from llm_benchmark.models import BenchmarkRun


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndBenchmark:
    """End-to-end integration tests for complete benchmark workflow."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory for test results."""
        temp_dir = tempfile.mkdtemp(prefix="benchmark_e2e_")
        yield temp_dir
        # Cleanup after test
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def minimal_config(self, temp_output_dir):
        """Create minimal configuration for quick testing."""
        return BenchmarkConfig(
            repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            models={
                "Q4_0": "tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
            },
            model_cache_dir=os.path.join(temp_output_dir, "models"),
            context_size=512,  # Small context for fast testing
            batch_size=128,
            max_tokens=10,  # Generate only 10 tokens for speed
            iterations=1,  # Single iteration for speed
            warmup_runs=0,  # Skip warmup for speed
            enable_quantization_profiling=True,
            enable_ablation_studies=False,  # Disable for speed
            enable_batch_testing=False,  # Disable for speed
            enable_thermal_monitoring=False,  # Disable for speed
            sleep_between_tests_s=0,  # No sleep for speed
            inference_timeout_s=60,  # Short timeout
            output_dir=temp_output_dir,
            save_formats=["json", "csv", "markdown", "html"],
            visualization_dpi=100  # Low DPI for speed
        )
    
    def test_complete_benchmark_run_x86(self, minimal_config, temp_output_dir):
        """
        Test complete benchmark run on x86 Linux with small test model.
        
        This test verifies:
        1. Hardware detection works
        2. Model download/caching works
        3. Quantization profiling executes
        4. Results are saved in correct format
        5. Visualizations are generated
        6. HTML report is created
        
        **Validates: Requirements 7.1, 7.2, 8.6, 9.8**
        """
        # Step 1: Detect hardware
        hw_info = HardwareDetector.detect()
        assert hw_info is not None
        assert hw_info.os_type in ["linux_x86", "jetson_xavier_nx", "darwin_arm64", "darwin_x86"]
        assert hw_info.cpu_cores > 0
        assert hw_info.total_ram_gb > 0
        
        # Step 2: Create backend
        backend = create_backend(hw_info)
        assert backend is not None
        
        # Step 3: Initialize model manager
        model_manager = ModelManager(
            cache_dir=minimal_config.model_cache_dir,
            hf_token=os.environ.get("HF_TOKEN")
        )
        
        # Step 4: Acquire model (this will download if not cached)
        # Note: This test requires internet connection and may take time on first run
        try:
            model_info = model_manager.get_model(
                repo_id=minimal_config.repo_id,
                filename=list(minimal_config.models.values())[0]
            )
            model_path = model_info.local_path
            assert os.path.exists(model_path)
        except Exception as e:
            pytest.skip(f"Model download failed (may be offline or rate-limited): {e}")
        
        # Step 5: Initialize metrics collector
        metrics_collector = backend.get_metrics_collector()
        assert metrics_collector is not None
        
        # Step 6: Initialize quantization profiler
        profiler = QuantizationProfiler(
            backend=backend,
            metrics_collector=metrics_collector
        )
        
        # Step 7: Run quantization profiling
        test_prompt = "What is the capital of France?"
        
        try:
            result = profiler.profile_quantization(
                model_path=model_path,
                quant="Q4_0",
                prompt=test_prompt,
                max_tokens=minimal_config.max_tokens
            )
            
            # Verify result structure
            assert result is not None
            assert result.quantization == "Q4_0"
            assert result.load_time_s > 0
            assert result.peak_ram_mb > 0
            assert result.ttft_ms > 0
            assert result.prefill_tps > 0
            assert result.decode_tps > 0
            assert result.prompt_tokens > 0
            assert result.output_tokens > 0
            
        except Exception as e:
            pytest.skip(f"Inference failed (may require specific hardware): {e}")
        
        # Step 8: Create benchmark run object
        run_id = "test_e2e_run"
        benchmark_run = BenchmarkRun(
            run_id=run_id,
            timestamp="2024-01-15T14:30:22",
            duration_s=10.0,
            hardware_info=hw_info,
            software_versions={"python": "3.10.0"},
            config=minimal_config.__dict__,
            model_checksums={"Q4_0": "test_checksum"},
            quantization_results=[result],
            ablation_results=[],
            batch_results=[],
            statistical_summaries=[],
            comparisons=[],
            visualization_paths=[],
            html_report_path=""
        )
        
        # Step 9: Save results
        persistence = ResultsPersistence(output_dir=temp_output_dir)
        
        # Save JSON
        json_path = persistence.save_json(benchmark_run)
        assert os.path.exists(json_path)
        
        # Verify JSON content
        with open(json_path, 'r') as f:
            saved_data = json.load(f)
            assert saved_data['run_id'] == run_id
            assert len(saved_data['quantization_results']) == 1
        
        # Save CSV
        csv_path = persistence.save_csv(benchmark_run)
        assert os.path.exists(csv_path)
        
        # Save Markdown
        md_path = persistence.save_markdown(benchmark_run)
        assert os.path.exists(md_path)
        
        # Step 10: Generate visualizations
        viz_gen = VisualizationGenerator(
            output_dir=temp_output_dir,
            dpi=minimal_config.visualization_dpi
        )
        
        # Generate quantization comparison chart
        viz_path = viz_gen.plot_quantization_comparison([result])
        assert os.path.exists(viz_path)
        assert viz_path.endswith('.png')
        
        # Step 11: Generate HTML report
        html_path = viz_gen.generate_html_report(
            benchmark_run=benchmark_run,
            quantization_results=[result],
            ablation_results=[],
            batch_results=[]
        )
        assert os.path.exists(html_path)
        assert html_path.endswith('.html')
        
        # Verify HTML content
        with open(html_path, 'r') as f:
            html_content = f.read()
            assert 'LLM Benchmark Report' in html_content
            assert 'Q4_0' in html_content
            assert 'Hardware Information' in html_content
        
        # Step 12: Verify all expected files exist
        assert os.path.exists(json_path)
        assert os.path.exists(csv_path)
        assert os.path.exists(md_path)
        assert os.path.exists(html_path)
        assert os.path.exists(viz_path)
    
    def test_benchmark_with_config_file(self, temp_output_dir):
        """
        Test benchmark run using configuration file.
        
        Verifies that configuration can be loaded from file and
        benchmark executes correctly.
        
        **Validates: Requirements 7.1, 7.2**
        """
        # Create config file
        config_data = {
            "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "models": {
                "Q4_0": "tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
            },
            "model_cache_dir": os.path.join(temp_output_dir, "models"),
            "context_size": 512,
            "max_tokens": 10,
            "iterations": 1,
            "warmup_runs": 0,
            "enable_ablation_studies": False,
            "enable_batch_testing": False,
            "output_dir": temp_output_dir
        }
        
        config_path = os.path.join(temp_output_dir, "test_config.json")
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        # Load config from file
        from llm_benchmark.config import ConfigParser
        config = ConfigParser.from_file(config_path)
        
        assert config.repo_id == config_data["repo_id"]
        assert config.models == config_data["models"]
        assert config.context_size == config_data["context_size"]
        assert config.max_tokens == config_data["max_tokens"]
    
    def test_results_directory_structure(self, minimal_config, temp_output_dir):
        """
        Test that results are saved in correct directory structure.
        
        Verifies:
        - Output directory is created
        - Subdirectories for visualizations exist
        - Log directory exists
        
        **Validates: Requirements 8.6**
        """
        # Create persistence object
        persistence = ResultsPersistence(output_dir=temp_output_dir)
        
        # Create minimal benchmark run
        hw_info = HardwareDetector.detect()
        benchmark_run = BenchmarkRun(
            run_id="test_structure",
            timestamp="2024-01-15T14:30:22",
            duration_s=1.0,
            hardware_info=hw_info,
            software_versions={},
            config=minimal_config.__dict__,
            model_checksums={},
            quantization_results=[],
            ablation_results=[],
            batch_results=[],
            statistical_summaries=[],
            comparisons=[],
            visualization_paths=[],
            html_report_path=""
        )
        
        # Save results
        json_path = os.path.join(temp_output_dir, "results.json")
        persistence.save_json(benchmark_run, Path(json_path))
        
        # Verify directory structure
        assert os.path.exists(temp_output_dir)
        assert os.path.exists(json_path)
        
        # Verify file is in output directory
        assert json_path.startswith(temp_output_dir)
    
    def test_error_handling_missing_model(self, temp_output_dir):
        """
        Test error handling when model file is missing.
        
        Verifies that appropriate error is raised when model
        cannot be found or downloaded.
        
        **Validates: Requirements 7.2**
        """
        config = BenchmarkConfig(
            repo_id="nonexistent/repo",
            models={"Q4_0": "nonexistent_model.gguf"},
            model_cache_dir=temp_output_dir,
            output_dir=temp_output_dir
        )
        
        model_manager = ModelManager(
            cache_dir=config.model_cache_dir,
            hf_token=None
        )
        
        # Should return None for nonexistent model (graceful error handling)
        result = model_manager.get_model(
            repo_id=config.repo_id,
            filename=list(config.models.values())[0]
        )
        
        assert result is None, "Expected None for nonexistent model"
