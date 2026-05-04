"""
Unit tests for AblationEngine.

Tests the ablation engine's ability to conduct controlled experiments
isolating optimization effects, particularly KV cache strategies.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call

import pytest

from llm_benchmark.models import AblationResult, InferenceMetrics
from llm_benchmark.profiler.ablation import AblationEngine


@pytest.fixture
def mock_backend():
    """Create mock hardware backend."""
    backend = Mock()
    backend.get_llama_config.return_value = {
        "n_ctx": 2048,
        "n_batch": 512,
        "n_threads": 4,
        "n_gpu_layers": 0
    }
    return backend


@pytest.fixture
def mock_metrics_collector():
    """Create mock metrics collector."""
    collector = Mock()
    
    # Create mock inference metrics
    mock_metrics = Mock(spec=InferenceMetrics)
    mock_metrics.ttft_ms = 100.0
    mock_metrics.prefill_tps = 50.0
    mock_metrics.decode_tps = 20.0
    mock_metrics.prompt_tokens = 100
    mock_metrics.output_tokens = 50
    mock_metrics.peak_memory_mb = 1000.0
    mock_metrics.gpu_memory_mb = None
    mock_metrics.gpu_utilization_pct = None
    
    collector.collect_inference_metrics.return_value = mock_metrics
    
    return collector


@pytest.fixture
def ablation_engine(mock_backend, mock_metrics_collector):
    """Create AblationEngine instance with mocked dependencies."""
    return AblationEngine(
        backend=mock_backend,
        metrics_collector=mock_metrics_collector
    )


class TestAblationEngineInit:
    """Test AblationEngine initialization."""
    
    def test_init(self, mock_backend, mock_metrics_collector):
        """Test basic initialization."""
        engine = AblationEngine(mock_backend, mock_metrics_collector)
        
        assert engine.backend == mock_backend
        assert engine.metrics == mock_metrics_collector
        assert engine.temp_dirs == []
    
    def test_init_creates_process_handle(self, mock_backend, mock_metrics_collector):
        """Test that initialization creates process handle."""
        engine = AblationEngine(mock_backend, mock_metrics_collector)
        
        assert engine.process is not None


class TestControlRun:
    """Test control run (no caching) functionality."""
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_run_control(self, mock_llama_class, ablation_engine, mock_metrics_collector):
        """Test control run without caching."""
        # Setup mock
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        # Run control test
        result = ablation_engine._run_control(
            model_path="/path/to/model.gguf",
            prompt="Test prompt",
            max_tokens=50
        )
        
        # Verify result
        assert isinstance(result, AblationResult)
        assert result.scenario == "control_no_cache"
        assert result.configuration["cache_enabled"] is False
        assert result.configuration["cache_type"] is None
        assert result.improvement_over_baseline is None
        
        # Verify metrics
        assert "ttft_ms" in result.metrics
        assert "prefill_tps" in result.metrics
        assert "decode_tps" in result.metrics
        assert "memory_overhead_mb" in result.metrics
        assert "peak_memory_mb" in result.metrics
        
        # Verify Llama was called with cache disabled
        call_kwargs = mock_llama_class.call_args[1]
        assert call_kwargs["cache"] is False
        
        # Verify metrics collector was called
        mock_metrics_collector.collect_inference_metrics.assert_called_once()
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_run_control_uses_backend_config(self, mock_llama_class, ablation_engine, mock_backend):
        """Test that control run uses backend configuration."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        ablation_engine._run_control(
            model_path="/path/to/model.gguf",
            prompt="Test prompt",
            max_tokens=50
        )
        
        # Verify backend config was requested
        mock_backend.get_llama_config.assert_called_once()
        
        # Verify Llama was called with backend config
        call_kwargs = mock_llama_class.call_args[1]
        assert "n_ctx" in call_kwargs
        assert "n_batch" in call_kwargs


class TestColdCacheRun:
    """Test cold cache run functionality."""
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_run_cold_cache_ram(self, mock_llama_class, ablation_engine):
        """Test cold run with RAM cache."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        result = ablation_engine._run_cold_cache(
            model_path="/path/to/model.gguf",
            prompt="Test prompt",
            max_tokens=50,
            cache_type="ram",
            baseline_ttft=100.0
        )
        
        # Verify result
        assert isinstance(result, AblationResult)
        assert result.scenario == "cold_ram_cache"
        assert result.configuration["cache_enabled"] is True
        assert result.configuration["cache_type"] == "ram"
        assert result.configuration["cache_state"] == "empty"
        
        # Verify improvement calculation
        assert result.improvement_over_baseline is not None
        
        # Verify Llama was called with cache enabled
        call_kwargs = mock_llama_class.call_args[1]
        assert call_kwargs["cache"] is True
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_run_cold_cache_disk(self, mock_llama_class, ablation_engine):
        """Test cold run with disk cache."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        result = ablation_engine._run_cold_cache(
            model_path="/path/to/model.gguf",
            prompt="Test prompt",
            max_tokens=50,
            cache_type="disk",
            baseline_ttft=100.0
        )
        
        # Verify result
        assert result.scenario == "cold_disk_cache"
        assert result.configuration["cache_type"] == "disk"
        
        # Verify Llama was called with disk cache config
        call_kwargs = mock_llama_class.call_args[1]
        assert call_kwargs["cache"] is True
        assert call_kwargs["cache_type"] == "disk"
        assert "cache_dir" in call_kwargs
        
        # Verify temp directory was created
        assert len(ablation_engine.temp_dirs) > 0
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_run_cold_cache_improvement_calculation(self, mock_llama_class, ablation_engine):
        """Test improvement calculation in cold cache run."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        baseline_ttft = 100.0
        
        result = ablation_engine._run_cold_cache(
            model_path="/path/to/model.gguf",
            prompt="Test prompt",
            max_tokens=50,
            cache_type="ram",
            baseline_ttft=baseline_ttft
        )
        
        # Verify improvement is calculated correctly
        # improvement = ((baseline - actual) / baseline) * 100
        actual_ttft = result.metrics["ttft_ms"]
        expected_improvement = ((baseline_ttft - actual_ttft) / baseline_ttft) * 100
        
        assert result.improvement_over_baseline == pytest.approx(expected_improvement, rel=0.01)


class TestWarmCacheRun:
    """Test warm cache run functionality."""
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_run_warm_cache_ram(self, mock_llama_class, ablation_engine, mock_metrics_collector):
        """Test warm run with RAM cache."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        result = ablation_engine._run_warm_cache(
            model_path="/path/to/model.gguf",
            prompt_prefix="Long prefix " * 100,
            prompt_suffix=" suffix",
            max_tokens=50,
            cache_type="ram",
            baseline_ttft=100.0
        )
        
        # Verify result
        assert isinstance(result, AblationResult)
        assert result.scenario == "warm_ram_cache"
        assert result.configuration["cache_enabled"] is True
        assert result.configuration["cache_type"] == "ram"
        assert result.configuration["cache_state"] == "populated"
        
        # Verify cache memory overhead is tracked
        assert "cache_memory_overhead_mb" in result.metrics
        assert "total_memory_overhead_mb" in result.metrics
        
        # Verify metrics collector was called twice (warmup + measurement)
        assert mock_metrics_collector.collect_inference_metrics.call_count == 2
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_run_warm_cache_disk(self, mock_llama_class, ablation_engine):
        """Test warm run with disk cache."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        result = ablation_engine._run_warm_cache(
            model_path="/path/to/model.gguf",
            prompt_prefix="Long prefix " * 100,
            prompt_suffix=" suffix",
            max_tokens=50,
            cache_type="disk",
            baseline_ttft=100.0
        )
        
        # Verify result
        assert result.scenario == "warm_disk_cache"
        assert result.configuration["cache_type"] == "disk"
        
        # Verify temp directory was created
        assert len(ablation_engine.temp_dirs) > 0
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_run_warm_cache_two_inferences(self, mock_llama_class, ablation_engine, mock_metrics_collector):
        """Test that warm cache run performs two inferences."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        ablation_engine._run_warm_cache(
            model_path="/path/to/model.gguf",
            prompt_prefix="Long prefix",
            prompt_suffix=" suffix",
            max_tokens=50,
            cache_type="ram",
            baseline_ttft=100.0
        )
        
        # Verify two inferences were performed
        assert mock_metrics_collector.collect_inference_metrics.call_count == 2
        
        # Verify first inference used warmup prompt
        first_call = mock_metrics_collector.collect_inference_metrics.call_args_list[0]
        first_prompt = first_call[1]["prompt"]
        assert "[warmup]" in first_prompt
        
        # Verify second inference used actual prompt
        second_call = mock_metrics_collector.collect_inference_metrics.call_args_list[1]
        second_prompt = second_call[1]["prompt"]
        assert "suffix" in second_prompt


class TestKVCacheStrategies:
    """Test complete KV cache strategy testing."""
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_test_kv_cache_strategies_all_types(self, mock_llama_class, ablation_engine):
        """Test KV cache strategies with all cache types."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        results = ablation_engine.test_kv_cache_strategies(
            model_path="/path/to/model.gguf",
            prompt_prefix="Long prefix " * 200,  # ~600 tokens
            prompt_suffix=" suffix",
            max_tokens=50,
            cache_types=["ram", "disk"]
        )
        
        # Verify we got 5 results (control + 2 cold + 2 warm)
        assert len(results) == 5
        
        # Verify scenarios
        scenarios = [r.scenario for r in results]
        assert "control_no_cache" in scenarios
        assert "cold_ram_cache" in scenarios
        assert "warm_ram_cache" in scenarios
        assert "cold_disk_cache" in scenarios
        assert "warm_disk_cache" in scenarios
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_test_kv_cache_strategies_ram_only(self, mock_llama_class, ablation_engine):
        """Test KV cache strategies with RAM only."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        results = ablation_engine.test_kv_cache_strategies(
            model_path="/path/to/model.gguf",
            prompt_prefix="Long prefix " * 200,
            prompt_suffix=" suffix",
            max_tokens=50,
            cache_types=["ram"]
        )
        
        # Verify we got 3 results (control + cold + warm)
        assert len(results) == 3
        
        scenarios = [r.scenario for r in results]
        assert "control_no_cache" in scenarios
        assert "cold_ram_cache" in scenarios
        assert "warm_ram_cache" in scenarios
        assert "cold_disk_cache" not in scenarios
        assert "warm_disk_cache" not in scenarios
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_test_kv_cache_strategies_validates_prefix_length(self, mock_llama_class, ablation_engine):
        """Test that short prefix generates warning."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        # Use short prefix (< 100 tokens)
        with patch('llm_benchmark.profiler.ablation.logger') as mock_logger:
            ablation_engine.test_kv_cache_strategies(
                model_path="/path/to/model.gguf",
                prompt_prefix="Short",  # Very short prefix
                prompt_suffix=" suffix",
                max_tokens=50,
                cache_types=["ram"]
            )
            
            # Verify warning was logged
            mock_logger.warning.assert_called()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "too short" in warning_msg.lower()
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_test_kv_cache_strategies_cleanup(self, mock_llama_class, ablation_engine):
        """Test that cache directories are cleaned up."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        results = ablation_engine.test_kv_cache_strategies(
            model_path="/path/to/model.gguf",
            prompt_prefix="Long prefix " * 200,
            prompt_suffix=" suffix",
            max_tokens=50,
            cache_types=["disk"]
        )
        
        # Verify temp directories were created during test
        # (they should be cleaned up after, so list should be empty)
        assert len(ablation_engine.temp_dirs) == 0


class TestProcessIsolation:
    """Test process isolation functionality."""
    
    @patch('llm_benchmark.profiler.ablation.gc')
    @patch('llm_benchmark.profiler.ablation.time')
    def test_ensure_process_isolation(self, mock_time, mock_gc, ablation_engine):
        """Test process isolation enforcement."""
        ablation_engine._ensure_process_isolation()
        
        # Verify garbage collection was called
        mock_gc.collect.assert_called_once()
        
        # Verify sleep was called for stabilization
        mock_time.sleep.assert_called_once_with(1)


class TestCacheDirectoryManagement:
    """Test cache directory creation and cleanup."""
    
    def test_create_temp_cache_dir(self, ablation_engine):
        """Test temporary cache directory creation."""
        cache_dir = ablation_engine._create_temp_cache_dir()
        
        # Verify directory was created
        assert cache_dir.exists()
        assert cache_dir.is_dir()
        
        # Verify it's tracked
        assert cache_dir in ablation_engine.temp_dirs
        
        # Cleanup
        cache_dir.rmdir()
    
    def test_cleanup_cache_directories(self, ablation_engine):
        """Test cache directory cleanup."""
        # Create some temp directories
        dir1 = ablation_engine._create_temp_cache_dir()
        dir2 = ablation_engine._create_temp_cache_dir()
        
        # Verify they exist
        assert dir1.exists()
        assert dir2.exists()
        
        # Cleanup
        ablation_engine._cleanup_cache_directories()
        
        # Verify they were removed
        assert not dir1.exists()
        assert not dir2.exists()
        
        # Verify tracking list is cleared
        assert len(ablation_engine.temp_dirs) == 0
    
    def test_cleanup_handles_missing_directories(self, ablation_engine):
        """Test cleanup handles already-deleted directories gracefully."""
        # Create and manually delete a directory
        cache_dir = ablation_engine._create_temp_cache_dir()
        cache_dir.rmdir()
        
        # Cleanup should not raise exception
        ablation_engine._cleanup_cache_directories()
        
        # Verify tracking list is cleared
        assert len(ablation_engine.temp_dirs) == 0


class TestAblationSummary:
    """Test ablation summary logging."""
    
    def test_log_ablation_summary(self, ablation_engine):
        """Test ablation summary logging."""
        results = [
            AblationResult(
                scenario="control_no_cache",
                configuration={"cache_enabled": False},
                metrics={
                    "ttft_ms": 100.0,
                    "prefill_tps": 50.0,
                    "decode_tps": 20.0,
                    "peak_memory_mb": 1000.0
                },
                improvement_over_baseline=None
            ),
            AblationResult(
                scenario="warm_ram_cache",
                configuration={"cache_enabled": True, "cache_type": "ram"},
                metrics={
                    "ttft_ms": 50.0,
                    "prefill_tps": 100.0,
                    "decode_tps": 40.0,
                    "peak_memory_mb": 1200.0
                },
                improvement_over_baseline=50.0
            )
        ]
        
        # Should not raise exception
        with patch('llm_benchmark.profiler.ablation.logger'):
            ablation_engine._log_ablation_summary(results)
    
    def test_log_ablation_summary_empty(self, ablation_engine):
        """Test ablation summary with empty results."""
        # Should not raise exception
        with patch('llm_benchmark.profiler.ablation.logger'):
            ablation_engine._log_ablation_summary([])


class TestPromptCaching:
    """Test prompt caching optimization functionality."""
    
    def test_generate_test_prompts(self, ablation_engine):
        """Test generation of test prompts with varying prefix lengths."""
        prefix_lengths = [100, 500, 1000]
        
        prompts = ablation_engine._generate_test_prompts(prefix_lengths)
        
        # Verify all prefix lengths are present
        assert len(prompts) == 3
        assert 100 in prompts
        assert 500 in prompts
        assert 1000 in prompts
        
        # Verify each prompt has prefix and suffix
        for length in prefix_lengths:
            assert "prefix" in prompts[length]
            assert "suffix" in prompts[length]
            
            # Verify approximate length (4 chars per token)
            prefix = prompts[length]["prefix"]
            expected_chars = length * 4
            # Allow 10% tolerance
            assert abs(len(prefix) - expected_chars) < expected_chars * 0.1
            
            # Verify suffix is unique
            suffix = prompts[length]["suffix"]
            assert str(length) in suffix
    
    def test_generate_test_prompts_single_length(self, ablation_engine):
        """Test prompt generation with single prefix length."""
        prompts = ablation_engine._generate_test_prompts([500])
        
        assert len(prompts) == 1
        assert 500 in prompts
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_run_prompt_caching_test_ram(self, mock_llama_class, ablation_engine, mock_metrics_collector):
        """Test prompt caching with RAM cache."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        # Setup mock metrics for three inferences
        first_metrics = Mock(spec=InferenceMetrics)
        first_metrics.ttft_ms = 100.0
        first_metrics.prompt_tokens = 100
        first_metrics.output_tokens = 10
        
        second_metrics = Mock(spec=InferenceMetrics)
        second_metrics.ttft_ms = 100.0
        second_metrics.prefill_tps = 50.0
        second_metrics.decode_tps = 20.0
        second_metrics.prompt_tokens = 110
        second_metrics.output_tokens = 50
        
        third_metrics = Mock(spec=InferenceMetrics)
        third_metrics.ttft_ms = 50.0  # Faster due to caching
        third_metrics.prefill_tps = 100.0
        third_metrics.decode_tps = 40.0
        third_metrics.prompt_tokens = 110
        third_metrics.output_tokens = 50
        
        mock_metrics_collector.collect_inference_metrics.side_effect = [
            first_metrics,
            second_metrics,
            third_metrics
        ]
        
        result = ablation_engine._run_prompt_caching_test(
            model_path="/path/to/model.gguf",
            prefix_prompt="Long prefix " * 100,
            suffix_prompt=" suffix",
            prefix_length=100,
            max_tokens=50,
            cache_type="ram"
        )
        
        # Verify result structure
        assert isinstance(result, AblationResult)
        assert result.scenario == "prompt_cache_ram_100tok"
        assert result.configuration["cache_type"] == "ram"
        assert result.configuration["prefix_length_tokens"] == 100
        
        # Verify metrics
        assert "cache_hit_rate_pct" in result.metrics
        assert "latency_reduction_ms" in result.metrics
        assert "latency_reduction_pct" in result.metrics
        assert "cache_memory_overhead_mb" in result.metrics
        assert "cache_memory_overhead_pct" in result.metrics
        assert "baseline_ttft_ms" in result.metrics
        assert "cached_ttft_ms" in result.metrics
        
        # Verify latency reduction calculation
        assert result.metrics["baseline_ttft_ms"] == 100.0
        assert result.metrics["cached_ttft_ms"] == 50.0
        assert result.metrics["latency_reduction_ms"] == 50.0
        assert result.metrics["latency_reduction_pct"] == 50.0
        
        # Verify improvement
        assert result.improvement_over_baseline == 50.0
        
        # Verify three inferences were performed
        assert mock_metrics_collector.collect_inference_metrics.call_count == 3
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_run_prompt_caching_test_disk(self, mock_llama_class, ablation_engine, mock_metrics_collector):
        """Test prompt caching with disk cache."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        # Setup mock metrics
        first_metrics = Mock(spec=InferenceMetrics)
        first_metrics.ttft_ms = 100.0
        first_metrics.prompt_tokens = 100
        first_metrics.output_tokens = 10
        
        second_metrics = Mock(spec=InferenceMetrics)
        second_metrics.ttft_ms = 100.0
        second_metrics.prefill_tps = 50.0
        second_metrics.decode_tps = 20.0
        second_metrics.prompt_tokens = 110
        second_metrics.output_tokens = 50
        
        third_metrics = Mock(spec=InferenceMetrics)
        third_metrics.ttft_ms = 60.0
        third_metrics.prefill_tps = 80.0
        third_metrics.decode_tps = 35.0
        third_metrics.prompt_tokens = 110
        third_metrics.output_tokens = 50
        
        mock_metrics_collector.collect_inference_metrics.side_effect = [
            first_metrics,
            second_metrics,
            third_metrics
        ]
        
        result = ablation_engine._run_prompt_caching_test(
            model_path="/path/to/model.gguf",
            prefix_prompt="Long prefix " * 100,
            suffix_prompt=" suffix",
            prefix_length=500,
            max_tokens=50,
            cache_type="disk"
        )
        
        # Verify result
        assert result.scenario == "prompt_cache_disk_500tok"
        assert result.configuration["cache_type"] == "disk"
        
        # Verify disk-specific metrics
        assert "disk_io_time_ms" in result.metrics
        assert "cache_file_size_mb" in result.metrics
        
        # Verify temp directory was created
        assert len(ablation_engine.temp_dirs) > 0
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_test_prompt_caching_all_combinations(self, mock_llama_class, ablation_engine, mock_metrics_collector):
        """Test prompt caching with all prefix lengths and cache types."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        # Setup mock metrics
        mock_metrics = Mock(spec=InferenceMetrics)
        mock_metrics.ttft_ms = 100.0
        mock_metrics.prefill_tps = 50.0
        mock_metrics.decode_tps = 20.0
        mock_metrics.prompt_tokens = 100
        mock_metrics.output_tokens = 50
        
        mock_metrics_collector.collect_inference_metrics.return_value = mock_metrics
        
        results = ablation_engine.test_prompt_caching(
            model_path="/path/to/model.gguf",
            prefix_lengths=[100, 500, 1000],
            max_tokens=50,
            cache_types=["ram", "disk"]
        )
        
        # Verify we got results for all combinations
        # 3 prefix lengths × 2 cache types = 6 results
        assert len(results) == 6
        
        # Verify scenarios
        scenarios = [r.scenario for r in results]
        assert "prompt_cache_ram_100tok" in scenarios
        assert "prompt_cache_ram_500tok" in scenarios
        assert "prompt_cache_ram_1000tok" in scenarios
        assert "prompt_cache_disk_100tok" in scenarios
        assert "prompt_cache_disk_500tok" in scenarios
        assert "prompt_cache_disk_1000tok" in scenarios
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_test_prompt_caching_ram_only(self, mock_llama_class, ablation_engine, mock_metrics_collector):
        """Test prompt caching with RAM cache only."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        mock_metrics = Mock(spec=InferenceMetrics)
        mock_metrics.ttft_ms = 100.0
        mock_metrics.prefill_tps = 50.0
        mock_metrics.decode_tps = 20.0
        mock_metrics.prompt_tokens = 100
        mock_metrics.output_tokens = 50
        
        mock_metrics_collector.collect_inference_metrics.return_value = mock_metrics
        
        results = ablation_engine.test_prompt_caching(
            model_path="/path/to/model.gguf",
            prefix_lengths=[100, 500],
            max_tokens=50,
            cache_types=["ram"]
        )
        
        # Verify we got results for RAM only
        assert len(results) == 2
        
        scenarios = [r.scenario for r in results]
        assert "prompt_cache_ram_100tok" in scenarios
        assert "prompt_cache_ram_500tok" in scenarios
        assert all("disk" not in s for s in scenarios)
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_test_prompt_caching_default_parameters(self, mock_llama_class, ablation_engine, mock_metrics_collector):
        """Test prompt caching with default parameters."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        mock_metrics = Mock(spec=InferenceMetrics)
        mock_metrics.ttft_ms = 100.0
        mock_metrics.prefill_tps = 50.0
        mock_metrics.decode_tps = 20.0
        mock_metrics.prompt_tokens = 100
        mock_metrics.output_tokens = 50
        
        mock_metrics_collector.collect_inference_metrics.return_value = mock_metrics
        
        results = ablation_engine.test_prompt_caching(
            model_path="/path/to/model.gguf"
        )
        
        # Default: [100, 500, 1000] × ["ram", "disk"] = 6 results
        assert len(results) == 6
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_test_prompt_caching_cleanup(self, mock_llama_class, ablation_engine, mock_metrics_collector):
        """Test that cache directories are cleaned up after prompt caching tests."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        mock_metrics = Mock(spec=InferenceMetrics)
        mock_metrics.ttft_ms = 100.0
        mock_metrics.prefill_tps = 50.0
        mock_metrics.decode_tps = 20.0
        mock_metrics.prompt_tokens = 100
        mock_metrics.output_tokens = 50
        
        mock_metrics_collector.collect_inference_metrics.return_value = mock_metrics
        
        results = ablation_engine.test_prompt_caching(
            model_path="/path/to/model.gguf",
            prefix_lengths=[100],
            cache_types=["disk"]
        )
        
        # Verify temp directories were cleaned up
        assert len(ablation_engine.temp_dirs) == 0
    
    def test_measure_cache_directory_size(self, ablation_engine):
        """Test cache directory size measurement."""
        # Create temporary directory with test files
        cache_dir = ablation_engine._create_temp_cache_dir()
        
        # Create test files
        test_file1 = cache_dir / "cache1.bin"
        test_file1.write_bytes(b"x" * 1024 * 1024)  # 1 MB
        
        test_file2 = cache_dir / "cache2.bin"
        test_file2.write_bytes(b"y" * 512 * 1024)  # 0.5 MB
        
        # Measure size
        size_mb = ablation_engine._measure_cache_directory_size(cache_dir)
        
        # Verify size (should be ~1.5 MB)
        assert 1.4 <= size_mb <= 1.6
        
        # Cleanup
        ablation_engine._cleanup_cache_directories()
    
    def test_measure_cache_directory_size_empty(self, ablation_engine):
        """Test cache directory size measurement for empty directory."""
        cache_dir = ablation_engine._create_temp_cache_dir()
        
        size_mb = ablation_engine._measure_cache_directory_size(cache_dir)
        
        assert size_mb == 0.0
        
        # Cleanup
        ablation_engine._cleanup_cache_directories()
    
    def test_log_prompt_caching_summary(self, ablation_engine):
        """Test prompt caching summary logging."""
        results = [
            AblationResult(
                scenario="prompt_cache_ram_100tok",
                configuration={"cache_type": "ram", "prefix_length_tokens": 100},
                metrics={
                    "prefix_length_tokens": 100,
                    "cache_hit_rate_pct": 90.0,
                    "latency_reduction_ms": 50.0,
                    "cache_memory_overhead_pct": 5.0,
                    "ttft_ms": 50.0
                },
                improvement_over_baseline=50.0
            ),
            AblationResult(
                scenario="prompt_cache_disk_500tok",
                configuration={"cache_type": "disk", "prefix_length_tokens": 500},
                metrics={
                    "prefix_length_tokens": 500,
                    "cache_hit_rate_pct": 95.0,
                    "latency_reduction_ms": 80.0,
                    "cache_memory_overhead_pct": 3.0,
                    "ttft_ms": 20.0
                },
                improvement_over_baseline=80.0
            )
        ]
        
        # Should not raise exception
        with patch('llm_benchmark.profiler.ablation.logger'):
            ablation_engine._log_prompt_caching_summary(results)
    
    def test_log_prompt_caching_summary_empty(self, ablation_engine):
        """Test prompt caching summary with empty results."""
        # Should not raise exception
        with patch('llm_benchmark.profiler.ablation.logger'):
            ablation_engine._log_prompt_caching_summary([])
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_test_prompt_caching_across_quantizations(self, mock_llama_class, ablation_engine, mock_metrics_collector):
        """Test prompt caching across multiple quantization levels."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        mock_metrics = Mock(spec=InferenceMetrics)
        mock_metrics.ttft_ms = 100.0
        mock_metrics.prefill_tps = 50.0
        mock_metrics.decode_tps = 20.0
        mock_metrics.prompt_tokens = 100
        mock_metrics.output_tokens = 50
        
        mock_metrics_collector.collect_inference_metrics.return_value = mock_metrics
        
        model_paths = {
            "Q8_0": "/path/to/q8.gguf",
            "Q4_0": "/path/to/q4.gguf"
        }
        
        results = ablation_engine.test_prompt_caching_across_quantizations(
            model_paths=model_paths,
            prefix_lengths=[100, 500],
            max_tokens=50,
            cache_types=["ram"]
        )
        
        # Verify we got results for all combinations
        # 2 quantizations × 2 prefix lengths × 1 cache type = 4 results
        assert len(results) == 4
        
        # Verify quantization info is in results
        for result in results:
            assert "quantization" in result.configuration
            assert result.configuration["quantization"] in ["Q8_0", "Q4_0"]
            assert result.configuration["quantization"] in result.scenario
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_test_concurrent_prompt_caching(self, mock_llama_class, ablation_engine, mock_metrics_collector):
        """Test concurrent prompt caching with multiple prompts sharing prefix."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        # Setup mock metrics for warmup + 3 prompts
        mock_metrics = Mock(spec=InferenceMetrics)
        mock_metrics.ttft_ms = 100.0
        mock_metrics.prefill_tps = 50.0
        mock_metrics.decode_tps = 20.0
        mock_metrics.prompt_tokens = 100
        mock_metrics.output_tokens = 50
        
        mock_metrics_collector.collect_inference_metrics.return_value = mock_metrics
        
        shared_prefix = "This is a shared prefix " * 50  # ~200 tokens
        unique_suffixes = [
            " with unique ending 1",
            " with unique ending 2",
            " with unique ending 3"
        ]
        
        results = ablation_engine.test_concurrent_prompt_caching(
            model_path="/path/to/model.gguf",
            shared_prefix=shared_prefix,
            unique_suffixes=unique_suffixes,
            max_tokens=50,
            cache_type="ram"
        )
        
        # Verify we got results for all prompts
        assert len(results) == 3
        
        # Verify each result has correct structure
        for idx, result in enumerate(results, 1):
            assert result.scenario == f"concurrent_prompt_{idx}_ram"
            assert result.configuration["prompt_index"] == idx
            assert result.configuration["total_prompts"] == 3
            assert result.configuration["shared_prefix"] is True
            
            # Verify metrics
            assert "cache_hit_rate_pct" in result.metrics
            assert "ttft_ms" in result.metrics
            assert "prompt_index" in result.metrics
            assert result.metrics["prompt_index"] == idx
        
        # Verify metrics collector was called for warmup + 3 prompts
        assert mock_metrics_collector.collect_inference_metrics.call_count == 4
    
    def test_log_quantization_comparison_summary(self, ablation_engine):
        """Test quantization comparison summary logging."""
        results = [
            AblationResult(
                scenario="prompt_cache_ram_100tok_Q8_0",
                configuration={
                    "quantization": "Q8_0",
                    "cache_type": "ram",
                    "prefix_length_tokens": 100
                },
                metrics={
                    "prefix_length_tokens": 100,
                    "cache_hit_rate_pct": 90.0,
                    "latency_reduction_ms": 50.0,
                    "cache_memory_overhead_pct": 5.0
                },
                improvement_over_baseline=50.0
            ),
            AblationResult(
                scenario="prompt_cache_ram_100tok_Q4_0",
                configuration={
                    "quantization": "Q4_0",
                    "cache_type": "ram",
                    "prefix_length_tokens": 100
                },
                metrics={
                    "prefix_length_tokens": 100,
                    "cache_hit_rate_pct": 92.0,
                    "latency_reduction_ms": 55.0,
                    "cache_memory_overhead_pct": 3.0
                },
                improvement_over_baseline=55.0
            )
        ]
        
        # Should not raise exception
        with patch('llm_benchmark.profiler.ablation.logger'):
            ablation_engine._log_quantization_comparison_summary(results)
    
    def test_log_concurrent_caching_summary(self, ablation_engine):
        """Test concurrent caching summary logging."""
        results = [
            AblationResult(
                scenario="concurrent_prompt_1_ram",
                configuration={"prompt_index": 1},
                metrics={
                    "prompt_index": 1,
                    "ttft_ms": 100.0,
                    "prefill_tps": 50.0,
                    "decode_tps": 20.0,
                    "cache_hit_rate_pct": 90.0,
                    "total_time_s": 5.0
                },
                improvement_over_baseline=None
            ),
            AblationResult(
                scenario="concurrent_prompt_2_ram",
                configuration={"prompt_index": 2},
                metrics={
                    "prompt_index": 2,
                    "ttft_ms": 80.0,
                    "prefill_tps": 60.0,
                    "decode_tps": 25.0,
                    "cache_hit_rate_pct": 95.0,
                    "total_time_s": 4.5
                },
                improvement_over_baseline=None
            )
        ]
        
        # Should not raise exception
        with patch('llm_benchmark.profiler.ablation.logger'):
            ablation_engine._log_concurrent_caching_summary(results)


class TestBatchTesting:
    """Test batch processing and throughput testing functionality."""
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_test_batch_sizes_all_sizes(self, mock_llama_class, ablation_engine, mock_metrics_collector):
        """Test batch processing with all batch sizes."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        # Setup mock metrics
        mock_metrics = Mock(spec=InferenceMetrics)
        mock_metrics.ttft_ms = 100.0
        mock_metrics.prefill_tps = 50.0
        mock_metrics.decode_tps = 20.0
        mock_metrics.prompt_tokens = 100
        mock_metrics.output_tokens = 50
        
        mock_metrics_collector.collect_inference_metrics.return_value = mock_metrics
        
        # Create test prompts
        prompts = [f"Test prompt {i}" for i in range(16)]
        
        results = ablation_engine.test_batch_sizes(
            model_path="/path/to/model.gguf",
            prompts=prompts,
            max_tokens=50,
            batch_sizes=[1, 2, 4, 8, 16]
        )
        
        # Verify we got results for all batch sizes
        assert len(results) == 5
        
        # Verify scenarios
        scenarios = [r.scenario for r in results]
        assert "batch_size_1" in scenarios
        assert "batch_size_2" in scenarios
        assert "batch_size_4" in scenarios
        assert "batch_size_8" in scenarios
        assert "batch_size_16" in scenarios
        
        # Verify each result has correct structure
        for result in results:
            batch_size = result.configuration["batch_size"]
            assert result.scenario == f"batch_size_{batch_size}"
            
            # Verify metrics
            assert "aggregate_throughput_tps" in result.metrics
            assert "avg_latency_per_prompt_ms" in result.metrics
            assert "min_latency_ms" in result.metrics
            assert "max_latency_ms" in result.metrics
            assert "std_latency_ms" in result.metrics
            assert "memory_increase_mb" in result.metrics
            assert "batch_duration_s" in result.metrics
            assert "total_tokens" in result.metrics
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_test_batch_sizes_default_parameters(self, mock_llama_class, ablation_engine, mock_metrics_collector):
        """Test batch processing with default parameters."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        mock_metrics = Mock(spec=InferenceMetrics)
        mock_metrics.ttft_ms = 100.0
        mock_metrics.prefill_tps = 50.0
        mock_metrics.decode_tps = 20.0
        mock_metrics.prompt_tokens = 100
        mock_metrics.output_tokens = 50
        
        mock_metrics_collector.collect_inference_metrics.return_value = mock_metrics
        
        prompts = [f"Test prompt {i}" for i in range(16)]
        
        results = ablation_engine.test_batch_sizes(
            model_path="/path/to/model.gguf",
            prompts=prompts
        )
        
        # Default batch sizes: [1, 2, 4, 8, 16]
        assert len(results) == 5
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_test_batch_sizes_custom_sizes(self, mock_llama_class, ablation_engine, mock_metrics_collector):
        """Test batch processing with custom batch sizes."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        mock_metrics = Mock(spec=InferenceMetrics)
        mock_metrics.ttft_ms = 100.0
        mock_metrics.prefill_tps = 50.0
        mock_metrics.decode_tps = 20.0
        mock_metrics.prompt_tokens = 100
        mock_metrics.output_tokens = 50
        
        mock_metrics_collector.collect_inference_metrics.return_value = mock_metrics
        
        prompts = [f"Test prompt {i}" for i in range(8)]
        
        results = ablation_engine.test_batch_sizes(
            model_path="/path/to/model.gguf",
            prompts=prompts,
            batch_sizes=[1, 4, 8]
        )
        
        assert len(results) == 3
        scenarios = [r.scenario for r in results]
        assert "batch_size_1" in scenarios
        assert "batch_size_4" in scenarios
        assert "batch_size_8" in scenarios
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_test_batch_sizes_insufficient_prompts(self, mock_llama_class, ablation_engine):
        """Test that insufficient prompts raises ValueError."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        # Only 5 prompts but batch size 16 requested
        prompts = [f"Test prompt {i}" for i in range(5)]
        
        with pytest.raises(ValueError, match="Need at least 16 prompts"):
            ablation_engine.test_batch_sizes(
                model_path="/path/to/model.gguf",
                prompts=prompts,
                batch_sizes=[1, 2, 4, 8, 16]
            )
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_run_batch_test_single_prompt(self, mock_llama_class, ablation_engine, mock_metrics_collector):
        """Test batch test with single prompt (batch_size=1)."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        mock_metrics = Mock(spec=InferenceMetrics)
        mock_metrics.ttft_ms = 100.0
        mock_metrics.prefill_tps = 50.0
        mock_metrics.decode_tps = 20.0
        mock_metrics.prompt_tokens = 100
        mock_metrics.output_tokens = 50
        
        mock_metrics_collector.collect_inference_metrics.return_value = mock_metrics
        
        result = ablation_engine._run_batch_test(
            llm=mock_llm,
            prompts=["Test prompt"],
            max_tokens=50,
            batch_size=1,
            baseline_memory_mb=1000.0
        )
        
        # Verify result structure
        assert result.scenario == "batch_size_1"
        assert result.configuration["batch_size"] == 1
        assert result.configuration["num_prompts"] == 1
        
        # Verify metrics
        assert result.metrics["batch_size"] == 1
        assert result.metrics["total_prompt_tokens"] == 100
        assert result.metrics["total_output_tokens"] == 50
        assert result.metrics["total_tokens"] == 150
        assert result.metrics["aggregate_throughput_tps"] > 0
        assert result.metrics["avg_latency_per_prompt_ms"] > 0
        
        # Verify metrics collector was called once
        mock_metrics_collector.collect_inference_metrics.assert_called_once()
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_run_batch_test_multiple_prompts(self, mock_llama_class, ablation_engine, mock_metrics_collector):
        """Test batch test with multiple prompts."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        # Create varying metrics for each prompt
        metrics_list = []
        for i in range(4):
            mock_metrics = Mock(spec=InferenceMetrics)
            mock_metrics.ttft_ms = 100.0 + i * 10  # Varying TTFT
            mock_metrics.prefill_tps = 50.0
            mock_metrics.decode_tps = 20.0
            mock_metrics.prompt_tokens = 100
            mock_metrics.output_tokens = 50
            metrics_list.append(mock_metrics)
        
        mock_metrics_collector.collect_inference_metrics.side_effect = metrics_list
        
        prompts = [f"Test prompt {i}" for i in range(4)]
        
        result = ablation_engine._run_batch_test(
            llm=mock_llm,
            prompts=prompts,
            max_tokens=50,
            batch_size=4,
            baseline_memory_mb=1000.0
        )
        
        # Verify result
        assert result.scenario == "batch_size_4"
        assert result.configuration["batch_size"] == 4
        assert result.configuration["num_prompts"] == 4
        
        # Verify aggregate metrics
        assert result.metrics["total_prompt_tokens"] == 400  # 100 * 4
        assert result.metrics["total_output_tokens"] == 200  # 50 * 4
        assert result.metrics["total_tokens"] == 600
        
        # Verify latency statistics
        assert result.metrics["min_latency_ms"] > 0
        assert result.metrics["max_latency_ms"] >= result.metrics["min_latency_ms"]
        assert result.metrics["avg_latency_per_prompt_ms"] > 0
        assert result.metrics["std_latency_ms"] >= 0
        
        # Verify TTFT average
        expected_avg_ttft = (100.0 + 110.0 + 120.0 + 130.0) / 4
        assert result.metrics["avg_ttft_ms"] == pytest.approx(expected_avg_ttft, rel=0.01)
        
        # Verify metrics collector was called 4 times
        assert mock_metrics_collector.collect_inference_metrics.call_count == 4
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_run_batch_test_throughput_calculation(self, mock_llama_class, ablation_engine, mock_metrics_collector):
        """Test that throughput is calculated correctly."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        mock_metrics = Mock(spec=InferenceMetrics)
        mock_metrics.ttft_ms = 100.0
        mock_metrics.prefill_tps = 50.0
        mock_metrics.decode_tps = 20.0
        mock_metrics.prompt_tokens = 100
        mock_metrics.output_tokens = 50
        
        mock_metrics_collector.collect_inference_metrics.return_value = mock_metrics
        
        with patch('llm_benchmark.profiler.ablation.time') as mock_time:
            # Mock time to control duration
            # Need 4 time calls: batch_start, prompt_start, prompt_end, batch_end
            mock_time.time.side_effect = [0.0, 0.0, 1.0, 1.0]  # 1 second duration
            
            result = ablation_engine._run_batch_test(
                llm=mock_llm,
                prompts=["Test prompt"],
                max_tokens=50,
                batch_size=1,
                baseline_memory_mb=1000.0
            )
            
            # Total tokens: 100 + 50 = 150
            # Duration: 1 second
            # Expected throughput: 150 tokens/s
            assert result.metrics["aggregate_throughput_tps"] == 150.0
            assert result.metrics["batch_duration_s"] == 1.0
    
    def test_identify_optimal_batch_size_highest_throughput(self, ablation_engine):
        """Test optimal batch size identification based on throughput."""
        results = [
            AblationResult(
                scenario="batch_size_1",
                configuration={"batch_size": 1},
                metrics={
                    "aggregate_throughput_tps": 50.0,
                    "avg_latency_per_prompt_ms": 100.0
                },
                improvement_over_baseline=None
            ),
            AblationResult(
                scenario="batch_size_2",
                configuration={"batch_size": 2},
                metrics={
                    "aggregate_throughput_tps": 80.0,
                    "avg_latency_per_prompt_ms": 120.0
                },
                improvement_over_baseline=None
            ),
            AblationResult(
                scenario="batch_size_4",
                configuration={"batch_size": 4},
                metrics={
                    "aggregate_throughput_tps": 120.0,  # Highest throughput
                    "avg_latency_per_prompt_ms": 150.0
                },
                improvement_over_baseline=None
            ),
            AblationResult(
                scenario="batch_size_8",
                configuration={"batch_size": 8},
                metrics={
                    "aggregate_throughput_tps": 100.0,  # Lower throughput
                    "avg_latency_per_prompt_ms": 180.0
                },
                improvement_over_baseline=None
            )
        ]
        
        optimal = ablation_engine._identify_optimal_batch_size(results)
        
        # Batch size 4 has highest throughput with acceptable latency
        assert optimal == 4
    
    def test_identify_optimal_batch_size_latency_constraint(self, ablation_engine):
        """Test optimal batch size respects latency constraint."""
        results = [
            AblationResult(
                scenario="batch_size_1",
                configuration={"batch_size": 1},
                metrics={
                    "aggregate_throughput_tps": 50.0,
                    "avg_latency_per_prompt_ms": 100.0
                },
                improvement_over_baseline=None
            ),
            AblationResult(
                scenario="batch_size_2",
                configuration={"batch_size": 2},
                metrics={
                    "aggregate_throughput_tps": 80.0,
                    "avg_latency_per_prompt_ms": 150.0
                },
                improvement_over_baseline=None
            ),
            AblationResult(
                scenario="batch_size_4",
                configuration={"batch_size": 4},
                metrics={
                    "aggregate_throughput_tps": 120.0,  # Highest throughput
                    "avg_latency_per_prompt_ms": 250.0  # But latency > 2x baseline
                },
                improvement_over_baseline=None
            )
        ]
        
        optimal = ablation_engine._identify_optimal_batch_size(results)
        
        # Batch size 4 has highest throughput but latency is too high (> 2x baseline)
        # Should choose batch size 2
        assert optimal == 2
    
    def test_identify_optimal_batch_size_empty_results(self, ablation_engine):
        """Test optimal batch size with empty results."""
        optimal = ablation_engine._identify_optimal_batch_size([])
        assert optimal == 1
    
    def test_identify_optimal_batch_size_single_result(self, ablation_engine):
        """Test optimal batch size with single result."""
        results = [
            AblationResult(
                scenario="batch_size_1",
                configuration={"batch_size": 1},
                metrics={
                    "aggregate_throughput_tps": 50.0,
                    "avg_latency_per_prompt_ms": 100.0
                },
                improvement_over_baseline=None
            )
        ]
        
        optimal = ablation_engine._identify_optimal_batch_size(results)
        assert optimal == 1
    
    def test_log_batch_testing_summary(self, ablation_engine):
        """Test batch testing summary logging."""
        results = [
            AblationResult(
                scenario="batch_size_1",
                configuration={"batch_size": 1},
                metrics={
                    "aggregate_throughput_tps": 50.0,
                    "avg_latency_per_prompt_ms": 100.0,
                    "min_latency_ms": 95.0,
                    "max_latency_ms": 105.0,
                    "memory_increase_mb": 100.0,
                    "batch_duration_s": 3.0
                },
                improvement_over_baseline=None
            ),
            AblationResult(
                scenario="batch_size_4",
                configuration={"batch_size": 4},
                metrics={
                    "aggregate_throughput_tps": 120.0,
                    "avg_latency_per_prompt_ms": 150.0,
                    "min_latency_ms": 140.0,
                    "max_latency_ms": 160.0,
                    "memory_increase_mb": 200.0,
                    "batch_duration_s": 5.0
                },
                improvement_over_baseline=None
            )
        ]
        
        # Should not raise exception
        with patch('llm_benchmark.profiler.ablation.logger'):
            ablation_engine._log_batch_testing_summary(results)
    
    def test_log_batch_testing_summary_empty(self, ablation_engine):
        """Test batch testing summary with empty results."""
        # Should not raise exception
        with patch('llm_benchmark.profiler.ablation.logger'):
            ablation_engine._log_batch_testing_summary([])
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_test_batch_sizes_memory_tracking(self, mock_llama_class, ablation_engine, mock_metrics_collector):
        """Test that memory is tracked correctly across batch sizes."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        mock_metrics = Mock(spec=InferenceMetrics)
        mock_metrics.ttft_ms = 100.0
        mock_metrics.prefill_tps = 50.0
        mock_metrics.decode_tps = 20.0
        mock_metrics.prompt_tokens = 100
        mock_metrics.output_tokens = 50
        
        mock_metrics_collector.collect_inference_metrics.return_value = mock_metrics
        
        prompts = [f"Test prompt {i}" for i in range(8)]
        
        results = ablation_engine.test_batch_sizes(
            model_path="/path/to/model.gguf",
            prompts=prompts,
            batch_sizes=[1, 2, 4, 8]
        )
        
        # Verify memory is tracked for each batch size
        for result in results:
            assert "memory_increase_mb" in result.metrics
            assert "peak_memory_mb" in result.metrics
            assert result.metrics["memory_increase_mb"] >= 0
            assert result.metrics["peak_memory_mb"] > 0
    
    @patch('llm_benchmark.profiler.ablation.Llama')
    def test_test_batch_sizes_sorted_execution(self, mock_llama_class, ablation_engine, mock_metrics_collector):
        """Test that batch sizes are executed in sorted order."""
        mock_llm = Mock()
        mock_llama_class.return_value = mock_llm
        
        mock_metrics = Mock(spec=InferenceMetrics)
        mock_metrics.ttft_ms = 100.0
        mock_metrics.prefill_tps = 50.0
        mock_metrics.decode_tps = 20.0
        mock_metrics.prompt_tokens = 100
        mock_metrics.output_tokens = 50
        
        mock_metrics_collector.collect_inference_metrics.return_value = mock_metrics
        
        prompts = [f"Test prompt {i}" for i in range(8)]
        
        # Provide batch sizes in unsorted order
        results = ablation_engine.test_batch_sizes(
            model_path="/path/to/model.gguf",
            prompts=prompts,
            batch_sizes=[8, 2, 4, 1]
        )
        
        # Verify results are in sorted order
        batch_sizes = [r.configuration["batch_size"] for r in results]
        assert batch_sizes == [1, 2, 4, 8]
