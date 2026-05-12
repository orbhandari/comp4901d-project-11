"""
Unit tests for QuantizationProfiler.

Tests the quantization profiling functionality including:
- Baseline memory measurement
- Model loading timing
- Warmup inference
- Streaming inference for TTFT capture
- Identical prompts across quantization levels
- Garbage collection between tests
- Comparison matrix generation
"""

import gc
import time
from unittest.mock import Mock, MagicMock, patch, call

import pytest

from llm_benchmark.profiler.quantization import QuantizationProfiler
from llm_benchmark.models import QuantizationResult, InferenceMetrics


class TestQuantizationProfiler:
    """Test suite for QuantizationProfiler class."""
    
    @pytest.fixture
    def mock_backend(self):
        """Create mock hardware backend."""
        backend = Mock()
        backend.get_llama_config.return_value = {
            "n_gpu_layers": 0,
            "use_mlock": True,
            "n_threads": 4
        }
        return backend
    
    @pytest.fixture
    def mock_metrics_collector(self):
        """Create mock metrics collector."""
        collector = Mock()
        
        # Mock inference metrics
        metrics = InferenceMetrics(
            ttft_ms=150.5,
            prefill_tps=250.0,
            decode_tps=45.5,
            total_time_s=2.5,
            prompt_tokens=100,
            output_tokens=50,
            peak_memory_mb=2048.0,
            per_token_latency_ms=[20.0] * 50,
            gpu_memory_mb=None,
            gpu_utilization_pct=None,
            cpu_temp_c=None,
            gpu_temp_c=None,
            power_watts=None
        )
        
        collector.collect_inference_metrics.return_value = metrics
        
        return collector
    
    @pytest.fixture
    def profiler(self, mock_backend, mock_metrics_collector):
        """Create QuantizationProfiler instance with mocked dependencies."""
        return QuantizationProfiler(mock_backend, mock_metrics_collector)
    
    @patch('llm_benchmark.profiler.quantization.Llama')
    @patch('llm_benchmark.profiler.quantization.psutil.Process')
    def test_profile_quantization_measures_baseline_memory(
        self, mock_process_class, mock_llama, profiler, mock_backend, mock_metrics_collector
    ):
        """Test that baseline memory is measured before model load."""
        # Setup mock process
        mock_process = Mock()
        mock_process_class.return_value = mock_process
        
        # Mock memory measurements
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 1024  # 1GB in bytes
        mock_process.memory_info.return_value = mock_memory_info
        
        # Mock Llama model
        mock_model = Mock()
        mock_llama.return_value = mock_model
        mock_model.return_value = {"choices": [{"text": "test"}]}
        
        # Create new profiler with mocked process
        profiler = QuantizationProfiler(mock_backend, mock_metrics_collector)
        
        # Profile quantization
        result = profiler.profile_quantization(
            model_path="/path/to/model.gguf",
            quant="Q4_0",
            prompt="Test prompt",
            max_tokens=10
        )
        
        # Verify baseline memory was measured
        assert mock_process.memory_info.called
        assert result.ram_increase_mb >= 0
    
    @patch('llm_benchmark.profiler.quantization.Llama')
    @patch('llm_benchmark.profiler.quantization.time.perf_counter')
    @patch('llm_benchmark.profiler.quantization.psutil.Process')
    def test_profile_quantization_times_model_loading(
        self, mock_process_class, mock_perf_counter, mock_llama, 
        profiler, mock_backend, mock_metrics_collector
    ):
        """Test that model loading time is measured accurately."""
        # Setup mock process
        mock_process = Mock()
        mock_process_class.return_value = mock_process
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 1024
        mock_process.memory_info.return_value = mock_memory_info
        
        # Mock timing
        mock_perf_counter.side_effect = [
            0.0,    # load_start
            2.5,    # load_end
        ]
        
        # Mock Llama model
        mock_model = Mock()
        mock_llama.return_value = mock_model
        mock_model.return_value = {"choices": [{"text": "test"}]}
        
        # Create new profiler with mocked process
        profiler = QuantizationProfiler(mock_backend, mock_metrics_collector)
        
        # Profile quantization
        result = profiler.profile_quantization(
            model_path="/path/to/model.gguf",
            quant="Q4_0",
            prompt="Test prompt",
            max_tokens=10
        )
        
        # Verify load time was measured
        assert result.load_time_s == 2.5
    
    @patch('llm_benchmark.profiler.quantization.Llama')
    @patch('llm_benchmark.profiler.quantization.psutil.Process')
    def test_profile_quantization_performs_warmup(
        self, mock_process_class, mock_llama, profiler, mock_backend, mock_metrics_collector
    ):
        """Test that warmup inference is performed before measurement."""
        # Setup mock process
        mock_process = Mock()
        mock_process_class.return_value = mock_process
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 1024
        mock_process.memory_info.return_value = mock_memory_info
        
        # Mock Llama model
        mock_model = Mock()
        mock_llama.return_value = mock_model
        mock_model.return_value = {"choices": [{"text": "test"}]}
        
        # Create new profiler with mocked process
        profiler = QuantizationProfiler(mock_backend, mock_metrics_collector)
        
        # Profile quantization with custom warmup tokens
        result = profiler.profile_quantization(
            model_path="/path/to/model.gguf",
            quant="Q4_0",
            prompt="Test prompt",
            max_tokens=10,
            warmup_tokens=5
        )
        
        # Verify warmup was called with correct parameters
        warmup_call = call("Test prompt", max_tokens=5, stream=False)
        assert warmup_call in mock_model.call_args_list
    
    @patch('llm_benchmark.profiler.quantization.Llama')
    @patch('llm_benchmark.profiler.quantization.psutil.Process')
    def test_profile_quantization_uses_streaming_inference(
        self, mock_process_class, mock_llama, profiler, mock_backend, mock_metrics_collector
    ):
        """Test that streaming inference is used to capture TTFT accurately."""
        # Setup mock process
        mock_process = Mock()
        mock_process_class.return_value = mock_process
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 1024
        mock_process.memory_info.return_value = mock_memory_info
        
        # Mock Llama model
        mock_model = Mock()
        mock_llama.return_value = mock_model
        mock_model.return_value = {"choices": [{"text": "test"}]}
        
        # Create new profiler with mocked process
        profiler = QuantizationProfiler(mock_backend, mock_metrics_collector)
        
        # Profile quantization
        result = profiler.profile_quantization(
            model_path="/path/to/model.gguf",
            quant="Q4_0",
            prompt="Test prompt",
            max_tokens=10
        )
        
        # Verify metrics collector was called (which uses streaming)
        mock_metrics_collector.collect_inference_metrics.assert_called_once()
        call_args = mock_metrics_collector.collect_inference_metrics.call_args
        assert call_args[1]['llm'] == mock_model
        assert call_args[1]['prompt'] == "Test prompt"
        assert call_args[1]['max_tokens'] == 10
    
    @patch('llm_benchmark.profiler.quantization.Llama')
    @patch('llm_benchmark.profiler.quantization.psutil.Process')
    def test_profile_quantization_returns_correct_result(
        self, mock_process_class, mock_llama, profiler, mock_backend, mock_metrics_collector
    ):
        """Test that QuantizationResult contains all expected fields."""
        # Setup mock process
        mock_process = Mock()
        mock_process_class.return_value = mock_process
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 1024
        mock_process.memory_info.return_value = mock_memory_info
        
        # Mock Llama model
        mock_model = Mock()
        mock_llama.return_value = mock_model
        mock_model.return_value = {"choices": [{"text": "test"}]}
        
        # Create new profiler with mocked process
        profiler = QuantizationProfiler(mock_backend, mock_metrics_collector)
        
        # Profile quantization
        result = profiler.profile_quantization(
            model_path="/path/to/model.gguf",
            quant="Q4_0",
            prompt="Test prompt",
            max_tokens=10
        )
        
        # Verify result structure
        assert isinstance(result, QuantizationResult)
        assert result.quantization == "Q4_0"
        assert result.load_time_s >= 0
        assert result.peak_ram_mb >= 0
        assert result.ram_increase_mb >= 0
        assert result.ttft_ms == 150.5
        assert result.prefill_tps == 250.0
        assert result.decode_tps == 45.5
        assert result.prompt_tokens == 100
        assert result.output_tokens == 50
    
    @patch('llm_benchmark.profiler.quantization.Llama')
    @patch('llm_benchmark.profiler.quantization.psutil.Process')
    @patch('llm_benchmark.profiler.quantization.gc.collect')
    def test_profile_all_enforces_garbage_collection(
        self, mock_gc_collect, mock_process_class, mock_llama, 
        profiler, mock_backend, mock_metrics_collector
    ):
        """Test that garbage collection is enforced between quantization tests."""
        # Setup mock process
        mock_process = Mock()
        mock_process_class.return_value = mock_process
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 1024
        mock_process.memory_info.return_value = mock_memory_info
        
        # Mock Llama model
        mock_model = Mock()
        mock_llama.return_value = mock_model
        mock_model.return_value = {"choices": [{"text": "test"}]}
        
        # Create new profiler with mocked process
        profiler = QuantizationProfiler(mock_backend, mock_metrics_collector)
        
        # Profile multiple quantization levels
        models = {
            "Q4_0": "/path/to/model-q4_0.gguf",
            "Q8_0": "/path/to/model-q8_0.gguf"
        }
        
        results = profiler.profile_all(
            models=models,
            prompt="Test prompt",
            max_tokens=10
        )
        
        # Verify garbage collection was called between tests
        # Should be called once per quantization level
        assert mock_gc_collect.call_count == 2
    
    @patch('llm_benchmark.profiler.quantization.Llama')
    @patch('llm_benchmark.profiler.quantization.psutil.Process')
    def test_profile_all_uses_identical_prompt(
        self, mock_process_class, mock_llama, profiler, mock_backend, mock_metrics_collector
    ):
        """Test that identical prompt is used across all quantization levels."""
        # Setup mock process
        mock_process = Mock()
        mock_process_class.return_value = mock_process
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 1024
        mock_process.memory_info.return_value = mock_memory_info
        
        # Mock Llama model
        mock_model = Mock()
        mock_llama.return_value = mock_model
        mock_model.return_value = {"choices": [{"text": "test"}]}
        
        # Create new profiler with mocked process
        profiler = QuantizationProfiler(mock_backend, mock_metrics_collector)
        
        # Profile multiple quantization levels
        models = {
            "Q4_0": "/path/to/model-q4_0.gguf",
            "Q8_0": "/path/to/model-q8_0.gguf"
        }
        
        test_prompt = "This is a test prompt"
        
        results = profiler.profile_all(
            models=models,
            prompt=test_prompt,
            max_tokens=10
        )
        
        # Verify metrics collector was called with same prompt for all quantizations
        calls = mock_metrics_collector.collect_inference_metrics.call_args_list
        assert len(calls) == 2
        
        for call_args in calls:
            assert call_args[1]['prompt'] == test_prompt
    
    @patch('llm_benchmark.profiler.quantization.Llama')
    @patch('llm_benchmark.profiler.quantization.psutil.Process')
    def test_profile_all_returns_results_for_all_quantizations(
        self, mock_process_class, mock_llama, profiler, mock_backend, mock_metrics_collector
    ):
        """Test that results are returned for all quantization levels."""
        # Setup mock process
        mock_process = Mock()
        mock_process_class.return_value = mock_process
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 1024
        mock_process.memory_info.return_value = mock_memory_info
        
        # Mock Llama model
        mock_model = Mock()
        mock_llama.return_value = mock_model
        mock_model.return_value = {"choices": [{"text": "test"}]}
        
        # Create new profiler with mocked process
        profiler = QuantizationProfiler(mock_backend, mock_metrics_collector)
        
        # Profile multiple quantization levels
        models = {
            "Q4_0": "/path/to/model-q4_0.gguf",
            "Q8_0": "/path/to/model-q8_0.gguf",
            "Q2_K": "/path/to/model-q2_k.gguf"
        }
        
        results = profiler.profile_all(
            models=models,
            prompt="Test prompt",
            max_tokens=10
        )
        
        # Verify results for all quantization levels
        assert len(results) == 3
        
        quantizations = [r.quantization for r in results]
        assert "Q4_0" in quantizations
        assert "Q8_0" in quantizations
        assert "Q2_K" in quantizations
    
    @patch('llm_benchmark.profiler.quantization.Llama')
    @patch('llm_benchmark.profiler.quantization.psutil.Process')
    def test_profile_quantization_handles_model_load_failure(
        self, mock_process_class, mock_llama, profiler, mock_backend, mock_metrics_collector
    ):
        """Test that model load failures are properly raised."""
        # Setup mock process
        mock_process = Mock()
        mock_process_class.return_value = mock_process
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 1024
        mock_process.memory_info.return_value = mock_memory_info
        
        # Mock Llama to raise exception
        mock_llama.side_effect = RuntimeError("Failed to load model")
        
        # Create new profiler with mocked process
        profiler = QuantizationProfiler(mock_backend, mock_metrics_collector)
        
        # Verify exception is raised
        with pytest.raises(RuntimeError, match="Failed to load model"):
            profiler.profile_quantization(
                model_path="/path/to/model.gguf",
                quant="Q4_0",
                prompt="Test prompt",
                max_tokens=10
            )
    
    @patch('llm_benchmark.profiler.quantization.Llama')
    @patch('llm_benchmark.profiler.quantization.psutil.Process')
    def test_profile_quantization_handles_inference_failure(
        self, mock_process_class, mock_llama, profiler, mock_backend, mock_metrics_collector
    ):
        """Test that inference failures are properly raised."""
        # Setup mock process
        mock_process = Mock()
        mock_process_class.return_value = mock_process
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 1024
        mock_process.memory_info.return_value = mock_memory_info
        
        # Mock Llama model
        mock_model = Mock()
        mock_llama.return_value = mock_model
        mock_model.return_value = {"choices": [{"text": "test"}]}
        
        # Mock metrics collector to raise exception
        mock_metrics_collector.collect_inference_metrics.side_effect = RuntimeError("Inference failed")
        
        # Create new profiler with mocked process
        profiler = QuantizationProfiler(mock_backend, mock_metrics_collector)
        
        # Verify exception is raised
        with pytest.raises(RuntimeError, match="Inference failed"):
            profiler.profile_quantization(
                model_path="/path/to/model.gguf",
                quant="Q4_0",
                prompt="Test prompt",
                max_tokens=10
            )
    
    @patch('llm_benchmark.profiler.quantization.Llama')
    @patch('llm_benchmark.profiler.quantization.psutil.Process')
    def test_profile_quantization_continues_after_warmup_failure(
        self, mock_process_class, mock_llama, profiler, mock_backend, mock_metrics_collector
    ):
        """Test that profiling continues even if warmup fails."""
        # Setup mock process
        mock_process = Mock()
        mock_process_class.return_value = mock_process
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 1024
        mock_process.memory_info.return_value = mock_memory_info
        
        # Mock Llama model - warmup fails, but measurement succeeds
        mock_model = Mock()
        mock_llama.return_value = mock_model
        mock_model.side_effect = [
            RuntimeError("Warmup failed"),  # First call (warmup)
            # Second call is metrics.collect_inference_metrics, which doesn't call model directly
        ]
        
        # Create new profiler with mocked process
        profiler = QuantizationProfiler(mock_backend, mock_metrics_collector)
        
        # Should not raise exception - warmup failure is logged but not fatal
        result = profiler.profile_quantization(
            model_path="/path/to/model.gguf",
            quant="Q4_0",
            prompt="Test prompt",
            max_tokens=10
        )
        
        # Verify result was still created
        assert isinstance(result, QuantizationResult)
        assert result.quantization == "Q4_0"
