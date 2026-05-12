"""
Unit tests for TestOrchestrator and TestConfig.

Tests orchestration logic including warmup, garbage collection,
thermal stabilization, checkpointing, and error handling.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call

import pytest

from llm_benchmark.config import BenchmarkConfig
from llm_benchmark.hardware.hal import HardwareBackend
from llm_benchmark.models import HardwareInfo, BenchmarkRun, QuantizationResult
from llm_benchmark.orchestrator import TestConfig, TestOrchestrator


@pytest.fixture
def mock_hw_info():
    """Create mock hardware info."""
    return HardwareInfo(
        os_type="linux_x86",
        cpu_model="Intel Core i7",
        cpu_cores=8,
        cpu_features=["avx2"],
        total_ram_gb=16.0,
        available_ram_gb=8.0,
        has_gpu=False,
        has_thermal_sensors=False,
        has_power_sensors=False
    )


@pytest.fixture
def mock_backend(mock_hw_info):
    """Create mock hardware backend."""
    backend = Mock(spec=HardwareBackend)
    backend.hw_info = mock_hw_info
    backend.get_metrics_collector.return_value = Mock()
    return backend


@pytest.fixture
def test_config():
    """Create test configuration."""
    return BenchmarkConfig(
        repo_id="test/model",
        models={"Q4_0": "model.gguf"},
        model_cache_dir="./models",
        context_size=2048,
        batch_size=512,
        max_tokens=100,
        iterations=1,
        warmup_runs=2,
        enable_quantization_profiling=True,
        enable_ablation_studies=False,
        enable_batch_testing=False,
        sleep_between_tests_s=1,
        thermal_stabilization_threshold_c=70.0,
        output_dir=tempfile.mkdtemp()
    )


class TestTestConfig:
    """Tests for TestConfig class."""
    
    def test_from_file_json(self, tmp_path):
        """Test loading configuration from JSON file."""
        config_file = tmp_path / "config.json"
        config_data = {
            "repo_id": "test/model",
            "models": {"Q4_0": "model.gguf"},
            "max_tokens": 50
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        config = TestConfig.from_file(str(config_file))
        
        assert config.repo_id == "test/model"
        assert config.models == {"Q4_0": "model.gguf"}
        assert config.max_tokens == 50
    
    def test_from_file_yaml(self, tmp_path):
        """Test loading configuration from YAML file."""
        pytest.importorskip("yaml")
        
        config_file = tmp_path / "config.yaml"
        config_data = """
repo_id: "test/model"
models:
  Q4_0: "model.gguf"
max_tokens: 50
"""
        
        with open(config_file, 'w') as f:
            f.write(config_data)
        
        config = TestConfig.from_file(str(config_file))
        
        assert config.repo_id == "test/model"
        assert config.models == {"Q4_0": "model.gguf"}
        assert config.max_tokens == 50
    
    def test_from_file_not_found(self):
        """Test error when configuration file not found."""
        with pytest.raises(FileNotFoundError):
            TestConfig.from_file("/nonexistent/config.json")
    
    def test_attribute_delegation(self, test_config):
        """Test that TestConfig delegates attributes to BenchmarkConfig."""
        config = TestConfig(test_config)
        
        assert config.repo_id == test_config.repo_id
        assert config.models == test_config.models
        assert config.max_tokens == test_config.max_tokens
        assert config.warmup_runs == test_config.warmup_runs
    
    def test_config_property(self, test_config):
        """Test accessing underlying BenchmarkConfig."""
        config = TestConfig(test_config)
        
        assert config.config is test_config
        assert isinstance(config.config, BenchmarkConfig)


class TestTestOrchestrator:
    """Tests for TestOrchestrator class."""
    
    def test_initialization(self, test_config, mock_backend):
        """Test orchestrator initialization."""
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        assert orchestrator.config == test_config
        assert orchestrator.backend == mock_backend
        assert orchestrator.hw_info == mock_backend.hw_info
        
        # Check directories were created
        assert orchestrator.output_dir.exists()
        assert orchestrator.checkpoint_dir.exists()
        assert orchestrator.logs_dir.exists()
    
    def test_warmup_runs(self, test_config, mock_backend):
        """Test warmup runs are executed."""
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        mock_llm = Mock()
        mock_llm.return_value = "response"
        
        orchestrator._warmup(mock_llm)
        
        # Should call llm twice (warmup_runs=2)
        assert mock_llm.call_count == 2
        
        # Check warmup parameters
        for call_args in mock_llm.call_args_list:
            args, kwargs = call_args
            assert kwargs['max_tokens'] == 5
            assert kwargs['stream'] is False
    
    def test_warmup_with_zero_runs(self, test_config, mock_backend):
        """Test warmup with zero runs does nothing."""
        test_config.warmup_runs = 0
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        mock_llm = Mock()
        
        orchestrator._warmup(mock_llm)
        
        # Should not call llm
        assert mock_llm.call_count == 0
    
    def test_warmup_continues_on_error(self, test_config, mock_backend):
        """Test warmup continues if a run fails."""
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        mock_llm = Mock()
        mock_llm.side_effect = [Exception("Error"), "success"]
        
        # Should not raise exception
        orchestrator._warmup(mock_llm)
        
        # Should attempt both warmup runs
        assert mock_llm.call_count == 2
    
    def test_garbage_collection(self, test_config, mock_backend):
        """Test garbage collection is enforced."""
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        with patch('gc.collect') as mock_gc:
            orchestrator._enforce_garbage_collection()
            
            mock_gc.assert_called_once()
    
    def test_thermal_stabilization_no_sensors(self, test_config, mock_backend):
        """Test thermal stabilization with no sensors uses fixed delay."""
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        start_time = time.time()
        orchestrator._thermal_stabilization_delay()
        elapsed = time.time() - start_time
        
        # Should sleep for configured duration (1s in test config)
        assert elapsed >= test_config.sleep_between_tests_s
        assert elapsed < test_config.sleep_between_tests_s + 0.5
    
    def test_thermal_stabilization_with_sensors_normal_temp(self, test_config, mock_backend):
        """Test thermal stabilization with normal temperature."""
        mock_backend.hw_info.has_thermal_sensors = True
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        # Mock temperature check to return normal temp
        orchestrator._check_thermal_state = Mock(return_value=(False, 50.0))
        
        start_time = time.time()
        orchestrator._thermal_stabilization_delay()
        elapsed = time.time() - start_time
        
        # Should just sleep for configured duration
        assert elapsed >= test_config.sleep_between_tests_s
        assert elapsed < test_config.sleep_between_tests_s + 0.5
    
    def test_thermal_stabilization_with_sensors_high_temp(self, test_config, mock_backend):
        """Test thermal stabilization waits for cooldown."""
        mock_backend.hw_info.has_thermal_sensors = True
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        # Mock temperature check: high temp, then normal
        orchestrator._check_thermal_state = Mock(
            side_effect=[(True, 80.0), (False, 65.0)]
        )
        
        with patch('time.sleep') as mock_sleep:
            orchestrator._thermal_stabilization_delay()
            
            # Should sleep for cooldown (10s) + configured delay (1s)
            sleep_calls = [call_args[0][0] for call_args in mock_sleep.call_args_list]
            assert 10 in sleep_calls  # Cooldown sleep
            assert test_config.sleep_between_tests_s in sleep_calls  # Standard sleep
    
    def test_check_thermal_state_no_sensors(self, test_config, mock_backend):
        """Test thermal state check with no sensors."""
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        is_throttled, temp = orchestrator._check_thermal_state()
        
        assert is_throttled is False
        assert temp == 0.0
    
    def test_check_thermal_state_below_threshold(self, test_config, mock_backend):
        """Test thermal state check below threshold."""
        mock_backend.hw_info.has_thermal_sensors = True
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        # Mock temperature reading
        orchestrator.metrics_collector._get_cpu_temperature = Mock(return_value=60.0)
        
        is_throttled, temp = orchestrator._check_thermal_state()
        
        assert is_throttled is False
        assert temp == 60.0
    
    def test_check_thermal_state_above_threshold(self, test_config, mock_backend):
        """Test thermal state check above threshold."""
        mock_backend.hw_info.has_thermal_sensors = True
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        # Mock temperature reading
        orchestrator.metrics_collector._get_cpu_temperature = Mock(return_value=75.0)
        
        is_throttled, temp = orchestrator._check_thermal_state()
        
        assert is_throttled is True
        assert temp == 75.0
    
    def test_check_thermal_state_with_gpu(self, test_config, mock_backend):
        """Test thermal state check with GPU temperature."""
        mock_backend.hw_info.has_thermal_sensors = True
        mock_backend.hw_info.has_gpu = True
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        # Mock temperature readings
        orchestrator.metrics_collector._get_cpu_temperature = Mock(return_value=60.0)
        orchestrator.metrics_collector._get_gpu_temperature = Mock(return_value=80.0)
        
        is_throttled, temp = orchestrator._check_thermal_state()
        
        # Should use max temperature (GPU)
        assert is_throttled is True
        assert temp == 80.0
    
    def test_save_checkpoint(self, test_config, mock_backend):
        """Test checkpoint saving."""
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        benchmark_run = BenchmarkRun(
            run_id="test_run",
            timestamp="2024-01-01T00:00:00",
            duration_s=100.0,
            hardware_info=mock_backend.hw_info,
            software_versions={},
            config={},
            model_checksums={}
        )
        
        orchestrator._save_checkpoint(benchmark_run, "test_checkpoint")
        
        # Check checkpoint file was created
        checkpoint_path = orchestrator.checkpoint_dir / "test_checkpoint.json"
        assert checkpoint_path.exists()
        
        # Verify checkpoint content
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        assert checkpoint_data['run_id'] == "test_run"
        assert checkpoint_data['checkpoint_name'] == "test_checkpoint"
        assert 'checkpoint_time' in checkpoint_data
    
    def test_save_checkpoint_error_handling(self, test_config, mock_backend):
        """Test checkpoint saving handles errors gracefully."""
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        # Create invalid benchmark run (will cause serialization error)
        benchmark_run = Mock()
        benchmark_run.run_id = "test"
        benchmark_run.timestamp = "2024-01-01"
        benchmark_run.hardware_info = Mock()
        benchmark_run.config = Mock()  # Mock object can't be serialized
        benchmark_run.quantization_results = []
        benchmark_run.ablation_results = []
        benchmark_run.batch_results = []
        
        # Should not raise exception
        orchestrator._save_checkpoint(benchmark_run, "test_checkpoint")
    
    def test_generate_summary_report(self, test_config, mock_backend):
        """Test summary report generation."""
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        # Add some test results
        orchestrator.test_results['passed'].append({
            'test': 'quantization_profiling',
            'status': 'passed',
            'results_count': 2
        })
        orchestrator.test_results['failed'].append({
            'test': 'ablation_studies',
            'status': 'failed',
            'error': 'Test error'
        })
        
        benchmark_run = BenchmarkRun(
            run_id="test_run",
            timestamp="2024-01-01T00:00:00",
            duration_s=100.0,
            hardware_info=mock_backend.hw_info,
            software_versions={},
            config={},
            model_checksums={}
        )
        
        orchestrator._generate_summary_report(benchmark_run)
        
        # Check summary file was created
        summary_path = orchestrator.output_dir / "summary.txt"
        assert summary_path.exists()
        
        # Verify summary content
        with open(summary_path, 'r') as f:
            content = f.read()
        
        assert "BENCHMARK SUMMARY REPORT" in content
        assert "test_run" in content
        assert "Passed: 1" in content
        assert "Failed: 1" in content
        assert "quantization_profiling" in content
        assert "ablation_studies" in content
    
    def test_run_all_tests_quantization_only(self, test_config, mock_backend):
        """Test running only quantization tests."""
        test_config.enable_ablation_studies = False
        test_config.enable_batch_testing = False
        
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        # Mock quantization tests with a small delay to ensure measurable duration
        def mock_quant_tests():
            import time
            time.sleep(0.01)  # Small delay to ensure duration > 0
            return []
        
        orchestrator.run_quantization_tests = Mock(side_effect=mock_quant_tests)
        
        benchmark_run = orchestrator.run_all_tests()
        
        # Should call quantization tests
        orchestrator.run_quantization_tests.assert_called_once()
        
        # Should have results
        assert benchmark_run.run_id is not None
        assert benchmark_run.timestamp is not None
        assert benchmark_run.duration_s > 0
        
        # Should have passed test
        assert len(orchestrator.test_results['passed']) == 1
        assert orchestrator.test_results['passed'][0]['test'] == 'quantization_profiling'
    
    def test_run_all_tests_with_ablation(self, test_config, mock_backend):
        """Test running with ablation studies enabled."""
        test_config.enable_ablation_studies = True
        test_config.enable_batch_testing = False
        
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        # Mock tests
        orchestrator.run_quantization_tests = Mock(return_value=[])
        orchestrator.run_ablation_tests = Mock(return_value=[])
        
        benchmark_run = orchestrator.run_all_tests()
        
        # Should call both tests
        orchestrator.run_quantization_tests.assert_called_once()
        orchestrator.run_ablation_tests.assert_called_once()
        
        # Should have two passed tests
        assert len(orchestrator.test_results['passed']) == 2
    
    def test_run_all_tests_ablation_failure(self, test_config, mock_backend):
        """Test that ablation failure doesn't stop execution."""
        test_config.enable_ablation_studies = True
        test_config.enable_batch_testing = False
        
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        # Mock tests - ablation fails
        orchestrator.run_quantization_tests = Mock(return_value=[])
        orchestrator.run_ablation_tests = Mock(side_effect=Exception("Ablation error"))
        
        # Should not raise exception
        benchmark_run = orchestrator.run_all_tests()
        
        # Should have one passed, one failed
        assert len(orchestrator.test_results['passed']) == 1
        assert len(orchestrator.test_results['failed']) == 1
        assert orchestrator.test_results['failed'][0]['test'] == 'ablation_studies'
    
    def test_run_all_tests_quantization_failure(self, test_config, mock_backend):
        """Test that quantization failure stops execution."""
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        # Mock quantization to fail
        orchestrator.run_quantization_tests = Mock(
            side_effect=Exception("Quantization error")
        )
        
        # Should raise exception
        with pytest.raises(Exception, match="Quantization error"):
            orchestrator.run_all_tests()
        
        # Should have one failed test
        assert len(orchestrator.test_results['failed']) == 1
        assert orchestrator.test_results['failed'][0]['test'] == 'quantization_profiling'
    
    def test_get_software_versions(self, test_config, mock_backend):
        """Test software version collection."""
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        versions = orchestrator._get_software_versions()
        
        assert 'python' in versions
        assert versions['python'] is not None
        assert 'llama-cpp-python' in versions
        assert 'psutil' in versions
    
    def test_config_to_dict(self, test_config, mock_backend):
        """Test configuration conversion to dictionary."""
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        config_dict = orchestrator._config_to_dict()
        
        assert config_dict['repo_id'] == test_config.repo_id
        assert config_dict['models'] == test_config.models
        assert config_dict['max_tokens'] == test_config.max_tokens
        assert config_dict['warmup_runs'] == test_config.warmup_runs
        assert config_dict['sleep_between_tests_s'] == test_config.sleep_between_tests_s
    
    def test_hardware_info_to_dict(self, test_config, mock_backend, mock_hw_info):
        """Test hardware info conversion to dictionary."""
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        hw_dict = orchestrator._hardware_info_to_dict(mock_hw_info)
        
        assert hw_dict['os_type'] == mock_hw_info.os_type
        assert hw_dict['cpu_model'] == mock_hw_info.cpu_model
        assert hw_dict['cpu_cores'] == mock_hw_info.cpu_cores
        assert hw_dict['total_ram_gb'] == mock_hw_info.total_ram_gb
        assert hw_dict['has_gpu'] == mock_hw_info.has_gpu


class TestOrchestratorAdvancedFeatures:
    """
    Additional tests for test orchestrator advanced features.
    
    **Validates: Requirements 7.3, 7.4, 7.5, 7.6, 7.8**
    """
    
    def test_warmup_run_execution_with_different_counts(self, test_config, mock_backend):
        """
        Test warmup run execution with different warmup counts.
        
        **Validates: Requirement 7.3**
        """
        # Test with 3 warmup runs
        test_config.warmup_runs = 3
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        mock_llm = Mock()
        mock_llm.return_value = "warmup response"
        
        orchestrator._warmup(mock_llm)
        
        # Should call llm exactly 3 times
        assert mock_llm.call_count == 3
        
        # All calls should use warmup parameters
        for call_args in mock_llm.call_args_list:
            args, kwargs = call_args
            assert kwargs['max_tokens'] == 5
            assert kwargs['stream'] is False
    
    def test_garbage_collection_enforcement_between_tests(self, test_config, mock_backend):
        """
        Test that garbage collection is enforced between test cases.
        
        **Validates: Requirement 7.4**
        """
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        # Mock the test methods
        orchestrator.run_quantization_tests = Mock(return_value=[])
        orchestrator.run_ablation_tests = Mock(return_value=[])
        
        with patch('gc.collect') as mock_gc:
            # Enable ablation to have multiple test phases
            test_config.enable_ablation_studies = True
            
            orchestrator.run_all_tests()
            
            # GC should be called between test phases
            # At least once after quantization, once after ablation
            assert mock_gc.call_count >= 2
    
    def test_thermal_stabilization_delay_logic_with_monitoring(self, test_config, mock_backend):
        """
        Test thermal stabilization delay logic with temperature monitoring.
        
        **Validates: Requirement 7.5**
        """
        mock_backend.hw_info.has_thermal_sensors = True
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        # Mock temperature readings: starts high, then cools down
        temp_readings = [85.0, 80.0, 75.0, 70.0, 65.0]
        temp_iter = iter(temp_readings)
        
        def mock_check_thermal():
            temp = next(temp_iter, 60.0)
            is_throttled = temp > test_config.thermal_stabilization_threshold_c
            return is_throttled, temp
        
        orchestrator._check_thermal_state = mock_check_thermal
        
        with patch('time.sleep') as mock_sleep:
            orchestrator._thermal_stabilization_delay()
            
            # Should have waited for temperature to drop
            # Multiple sleep calls for cooldown
            assert mock_sleep.call_count >= 2
    
    def test_checkpoint_saving_and_recovery(self, test_config, mock_backend):
        """
        Test checkpoint saving and recovery functionality.
        
        **Validates: Requirement 7.6**
        """
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        # Create a benchmark run with some results
        benchmark_run = BenchmarkRun(
            run_id="test_checkpoint_run",
            timestamp="2024-01-15T10:30:00",
            duration_s=150.0,
            hardware_info=mock_backend.hw_info,
            software_versions={"python": "3.10"},
            config={"test": "config"},
            model_checksums={"model.gguf": "abc123"}
        )
        
        # Add some quantization results
        benchmark_run.quantization_results = [
            QuantizationResult(
                quantization="Q4_0",
                load_time_s=5.0,
                peak_ram_mb=2000.0,
                ram_increase_mb=1500.0,
                ttft_ms=100.0,
                prefill_tps=50.0,
                decode_tps=25.0,
                prompt_tokens=10,
                output_tokens=20
            )
        ]
        
        # Save checkpoint
        checkpoint_name = "after_quantization"
        orchestrator._save_checkpoint(benchmark_run, checkpoint_name)
        
        # Verify checkpoint file exists
        checkpoint_path = orchestrator.checkpoint_dir / f"{checkpoint_name}.json"
        assert checkpoint_path.exists()
        
        # Load and verify checkpoint content
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        assert checkpoint_data['run_id'] == "test_checkpoint_run"
        assert checkpoint_data['checkpoint_name'] == checkpoint_name
        assert 'checkpoint_time' in checkpoint_data
        # Data is stored directly, not wrapped in 'data' key
        assert 'quantization_results' in checkpoint_data
        
        # Verify quantization results were saved
        assert len(checkpoint_data['quantization_results']) == 1
    
    def test_error_handling_continue_on_failure(self, test_config, mock_backend):
        """
        Test error handling that continues on failure for optional tests.
        
        **Validates: Requirement 7.8**
        """
        test_config.enable_ablation_studies = True
        test_config.enable_batch_testing = True
        
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        # Mock tests - quantization succeeds, ablation fails, batch succeeds
        orchestrator.run_quantization_tests = Mock(return_value=[
            QuantizationResult(
                quantization="Q4_0",
                load_time_s=5.0,
                peak_ram_mb=2000.0,
                ram_increase_mb=1500.0,
                ttft_ms=100.0,
                prefill_tps=50.0,
                decode_tps=25.0,
                prompt_tokens=10,
                output_tokens=20
            )
        ])
        orchestrator.run_ablation_tests = Mock(side_effect=Exception("Ablation failed"))
        orchestrator.run_batch_tests = Mock(return_value=[])
        
        # Should not raise exception
        benchmark_run = orchestrator.run_all_tests()
        
        # Verify all tests were attempted
        orchestrator.run_quantization_tests.assert_called_once()
        orchestrator.run_ablation_tests.assert_called_once()
        orchestrator.run_batch_tests.assert_called_once()
        
        # Verify results tracking
        assert len(orchestrator.test_results['passed']) == 2  # quantization and batch
        assert len(orchestrator.test_results['failed']) == 1  # ablation
        
        # Verify failed test details
        failed_test = orchestrator.test_results['failed'][0]
        assert failed_test['test'] == 'ablation_studies'
        assert 'Ablation failed' in failed_test['error']
    
    def test_checkpoint_saving_after_each_test_phase(self, test_config, mock_backend):
        """
        Test that checkpoints are saved after each test phase.
        
        **Validates: Requirement 7.8**
        """
        test_config.enable_ablation_studies = True
        test_config.enable_batch_testing = True
        
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        # Mock all test methods
        orchestrator.run_quantization_tests = Mock(return_value=[])
        orchestrator.run_ablation_tests = Mock(return_value=[])
        orchestrator.run_batch_tests = Mock(return_value=[])
        
        # Track checkpoint saves
        checkpoint_saves = []
        original_save = orchestrator._save_checkpoint
        
        def track_checkpoint(benchmark_run, name):
            checkpoint_saves.append(name)
            return original_save(benchmark_run, name)
        
        orchestrator._save_checkpoint = track_checkpoint
        
        # Run all tests
        orchestrator.run_all_tests()
        
        # Verify checkpoints were saved after each phase
        assert 'quantization_complete' in checkpoint_saves
        assert 'ablation_complete' in checkpoint_saves
        assert 'batch_complete' in checkpoint_saves
    
    def test_thermal_stabilization_with_configurable_threshold(self, test_config, mock_backend):
        """
        Test thermal stabilization with configurable threshold.
        
        **Validates: Requirement 7.5**
        """
        # Set custom threshold
        test_config.thermal_stabilization_threshold_c = 75.0
        mock_backend.hw_info.has_thermal_sensors = True
        
        orchestrator = TestOrchestrator(test_config, mock_backend)
        
        # Test temperature just below threshold
        orchestrator._check_thermal_state = Mock(return_value=(False, 74.0))
        
        with patch('time.sleep') as mock_sleep:
            orchestrator._thermal_stabilization_delay()
            
            # Should only sleep for standard delay, not cooldown
            sleep_calls = [call_args[0][0] for call_args in mock_sleep.call_args_list]
            assert test_config.sleep_between_tests_s in sleep_calls
            assert 10 not in sleep_calls  # No cooldown sleep
        
        # Test temperature above threshold
        orchestrator._check_thermal_state = Mock(
            side_effect=[(True, 80.0), (False, 70.0)]
        )
        
        with patch('time.sleep') as mock_sleep:
            orchestrator._thermal_stabilization_delay()
            
            # Should sleep for cooldown
            sleep_calls = [call_args[0][0] for call_args in mock_sleep.call_args_list]
            assert 10 in sleep_calls  # Cooldown sleep
