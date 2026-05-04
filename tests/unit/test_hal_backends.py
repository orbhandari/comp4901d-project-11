"""
Unit tests for Hardware Abstraction Layer (HAL) backends.

Tests X86Backend and JetsonBackend configuration generation, GPU layer
calculation, and fallback behavior.

Requirements: 4.1, 4.2, 4.5
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from llm_benchmark.hardware.hal import X86Backend, JetsonBackend, create_backend
from llm_benchmark.models import HardwareInfo


class TestX86Backend:
    """Test X86Backend configuration and behavior."""
    
    def test_get_llama_config_cpu_only(self):
        """Test X86Backend generates CPU-only configuration."""
        hw_info = HardwareInfo(
            os_type="linux_x86",
            cpu_model="Intel Core i7-9700K",
            cpu_cores=8,
            cpu_features=["avx2", "avx512"],
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            has_gpu=False,
            gpu_model=None,
            gpu_memory_gb=None,
            gpu_compute_capability=None,
            has_thermal_sensors=True,
            has_power_sensors=False
        )
        
        backend = X86Backend(hw_info)
        config = backend.get_llama_config()
        
        # Verify CPU-only configuration
        assert config["n_gpu_layers"] == 0, "X86 should use CPU-only (0 GPU layers)"
        assert config["use_mlock"] is True, "X86 should use mlock for performance"
        assert config["n_threads"] == 8, "Should use all CPU cores"
    
    def test_get_llama_config_with_different_core_counts(self):
        """Test X86Backend adapts thread count to CPU cores."""
        test_cases = [
            (4, 4),   # 4 cores -> 4 threads
            (8, 8),   # 8 cores -> 8 threads
            (16, 16), # 16 cores -> 16 threads
            (32, 32), # 32 cores -> 32 threads
        ]
        
        for cpu_cores, expected_threads in test_cases:
            hw_info = HardwareInfo(
                os_type="linux_x86",
                cpu_model="Test CPU",
                cpu_cores=cpu_cores,
                cpu_features=[],
                total_ram_gb=16.0,
                available_ram_gb=8.0,
                has_gpu=False,
                gpu_model=None,
                gpu_memory_gb=None,
                gpu_compute_capability=None,
                has_thermal_sensors=False,
                has_power_sensors=False
            )
            
            backend = X86Backend(hw_info)
            config = backend.get_llama_config()
            
            assert config["n_threads"] == expected_threads, \
                f"Expected {expected_threads} threads for {cpu_cores} cores"
    
    def test_get_metrics_collector(self):
        """Test X86Backend creates metrics collector."""
        hw_info = HardwareInfo(
            os_type="linux_x86",
            cpu_model="Test CPU",
            cpu_cores=8,
            cpu_features=[],
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            has_gpu=False,
            gpu_model=None,
            gpu_memory_gb=None,
            gpu_compute_capability=None,
            has_thermal_sensors=False,
            has_power_sensors=False
        )
        
        backend = X86Backend(hw_info)
        collector = backend.get_metrics_collector()
        
        # Verify collector is created and configured for x86
        assert collector is not None
        assert collector.hw_info == hw_info


class TestJetsonBackend:
    """Test JetsonBackend configuration and GPU layer calculation."""
    
    def test_calculate_gpu_layers_with_8gb_memory(self):
        """Test GPU layer calculation with 8GB GPU memory."""
        hw_info = HardwareInfo(
            os_type="jetson_xavier_nx",
            cpu_model="ARM Cortex-A57",
            cpu_cores=6,
            cpu_features=[],
            total_ram_gb=8.0,
            available_ram_gb=6.0,
            has_gpu=True,
            gpu_model="NVIDIA Tegra X1",
            gpu_memory_gb=8.0,
            gpu_compute_capability="5.3",
            has_thermal_sensors=True,
            has_power_sensors=True
        )
        
        backend = JetsonBackend(hw_info)
        gpu_layers = backend._calculate_gpu_layers()
        
        # 8GB * 0.8 * 1024 / 100 = ~65 layers
        assert 60 <= gpu_layers <= 70, \
            f"Expected ~65 layers for 8GB GPU, got {gpu_layers}"
    
    def test_calculate_gpu_layers_with_4gb_memory(self):
        """Test GPU layer calculation with 4GB GPU memory."""
        hw_info = HardwareInfo(
            os_type="jetson_xavier_nx",
            cpu_model="ARM Cortex-A57",
            cpu_cores=6,
            cpu_features=[],
            total_ram_gb=8.0,
            available_ram_gb=6.0,
            has_gpu=True,
            gpu_model="NVIDIA Tegra X1",
            gpu_memory_gb=4.0,
            gpu_compute_capability="5.3",
            has_thermal_sensors=True,
            has_power_sensors=True
        )
        
        backend = JetsonBackend(hw_info)
        gpu_layers = backend._calculate_gpu_layers()
        
        # 4GB * 0.8 * 1024 / 100 = ~32 layers
        assert 30 <= gpu_layers <= 35, \
            f"Expected ~32 layers for 4GB GPU, got {gpu_layers}"
    
    def test_calculate_gpu_layers_with_16gb_memory(self):
        """Test GPU layer calculation with 16GB GPU memory."""
        hw_info = HardwareInfo(
            os_type="jetson_xavier_nx",
            cpu_model="ARM Cortex-A57",
            cpu_cores=6,
            cpu_features=[],
            total_ram_gb=16.0,
            available_ram_gb=12.0,
            has_gpu=True,
            gpu_model="NVIDIA Tegra X1",
            gpu_memory_gb=16.0,
            gpu_compute_capability="5.3",
            has_thermal_sensors=True,
            has_power_sensors=True
        )
        
        backend = JetsonBackend(hw_info)
        gpu_layers = backend._calculate_gpu_layers()
        
        # 16GB * 0.8 * 1024 / 100 = ~131 layers
        assert 125 <= gpu_layers <= 135, \
            f"Expected ~131 layers for 16GB GPU, got {gpu_layers}"
    
    def test_fallback_when_gpu_unavailable(self):
        """Test fallback to CPU-only when GPU is unavailable."""
        hw_info = HardwareInfo(
            os_type="jetson_xavier_nx",
            cpu_model="ARM Cortex-A57",
            cpu_cores=6,
            cpu_features=[],
            total_ram_gb=8.0,
            available_ram_gb=6.0,
            has_gpu=False,  # GPU not available
            gpu_model=None,
            gpu_memory_gb=None,
            gpu_compute_capability=None,
            has_thermal_sensors=True,
            has_power_sensors=True
        )
        
        backend = JetsonBackend(hw_info)
        gpu_layers = backend._calculate_gpu_layers()
        
        # Should fall back to 0 layers (CPU-only)
        assert gpu_layers == 0, \
            "Should fall back to CPU-only when GPU unavailable"
    
    def test_fallback_when_gpu_memory_none(self):
        """Test fallback when GPU memory is None."""
        hw_info = HardwareInfo(
            os_type="jetson_xavier_nx",
            cpu_model="ARM Cortex-A57",
            cpu_cores=6,
            cpu_features=[],
            total_ram_gb=8.0,
            available_ram_gb=6.0,
            has_gpu=True,
            gpu_model="NVIDIA Tegra X1",
            gpu_memory_gb=None,  # Memory info unavailable
            gpu_compute_capability="5.3",
            has_thermal_sensors=True,
            has_power_sensors=True
        )
        
        backend = JetsonBackend(hw_info)
        gpu_layers = backend._calculate_gpu_layers()
        
        # Should fall back to 0 layers when memory info unavailable
        assert gpu_layers == 0, \
            "Should fall back to CPU-only when GPU memory unavailable"
    
    def test_get_llama_config_with_gpu(self):
        """Test JetsonBackend generates GPU-accelerated configuration."""
        hw_info = HardwareInfo(
            os_type="jetson_xavier_nx",
            cpu_model="ARM Cortex-A57",
            cpu_cores=6,
            cpu_features=[],
            total_ram_gb=8.0,
            available_ram_gb=6.0,
            has_gpu=True,
            gpu_model="NVIDIA Tegra X1",
            gpu_memory_gb=8.0,
            gpu_compute_capability="5.3",
            has_thermal_sensors=True,
            has_power_sensors=True
        )
        
        backend = JetsonBackend(hw_info)
        config = backend.get_llama_config()
        
        # Verify GPU-accelerated configuration
        assert config["n_gpu_layers"] > 0, "Should use GPU layers"
        assert 60 <= config["n_gpu_layers"] <= 70, \
            f"Expected ~65 GPU layers, got {config['n_gpu_layers']}"
        assert config["use_mlock"] is False, \
            "Jetson should not use mlock (limited RAM)"
        assert config["n_threads"] == 4, \
            "Jetson should use 4 threads (leave cores for system)"
    
    def test_get_llama_config_without_gpu(self):
        """Test JetsonBackend configuration when GPU unavailable."""
        hw_info = HardwareInfo(
            os_type="jetson_xavier_nx",
            cpu_model="ARM Cortex-A57",
            cpu_cores=6,
            cpu_features=[],
            total_ram_gb=8.0,
            available_ram_gb=6.0,
            has_gpu=False,
            gpu_model=None,
            gpu_memory_gb=None,
            gpu_compute_capability=None,
            has_thermal_sensors=True,
            has_power_sensors=True
        )
        
        backend = JetsonBackend(hw_info)
        config = backend.get_llama_config()
        
        # Should fall back to CPU-only
        assert config["n_gpu_layers"] == 0, \
            "Should use CPU-only when GPU unavailable"
        assert config["use_mlock"] is False
        assert config["n_threads"] == 4
    
    def test_get_metrics_collector(self):
        """Test JetsonBackend creates metrics collector."""
        hw_info = HardwareInfo(
            os_type="jetson_xavier_nx",
            cpu_model="ARM Cortex-A57",
            cpu_cores=6,
            cpu_features=[],
            total_ram_gb=8.0,
            available_ram_gb=6.0,
            has_gpu=True,
            gpu_model="NVIDIA Tegra X1",
            gpu_memory_gb=8.0,
            gpu_compute_capability="5.3",
            has_thermal_sensors=True,
            has_power_sensors=True
        )
        
        backend = JetsonBackend(hw_info)
        collector = backend.get_metrics_collector()
        
        # Verify collector is created and configured for Jetson
        assert collector is not None
        assert collector.hw_info == hw_info


class TestCreateBackend:
    """Test backend factory function."""
    
    def test_create_x86_backend(self):
        """Test create_backend returns X86Backend for x86 platform."""
        hw_info = HardwareInfo(
            os_type="linux_x86",
            cpu_model="Intel Core i7",
            cpu_cores=8,
            cpu_features=[],
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            has_gpu=False,
            gpu_model=None,
            gpu_memory_gb=None,
            gpu_compute_capability=None,
            has_thermal_sensors=False,
            has_power_sensors=False
        )
        
        backend = create_backend(hw_info)
        
        assert isinstance(backend, X86Backend), \
            "Should create X86Backend for linux_x86"
        assert backend.hw_info == hw_info
    
    def test_create_jetson_backend(self):
        """Test create_backend returns JetsonBackend for Jetson platform."""
        hw_info = HardwareInfo(
            os_type="jetson_xavier_nx",
            cpu_model="ARM Cortex-A57",
            cpu_cores=6,
            cpu_features=[],
            total_ram_gb=8.0,
            available_ram_gb=6.0,
            has_gpu=True,
            gpu_model="NVIDIA Tegra X1",
            gpu_memory_gb=8.0,
            gpu_compute_capability="5.3",
            has_thermal_sensors=True,
            has_power_sensors=True
        )
        
        backend = create_backend(hw_info)
        
        assert isinstance(backend, JetsonBackend), \
            "Should create JetsonBackend for jetson_xavier_nx"
        assert backend.hw_info == hw_info
    
    def test_create_backend_defaults_to_x86(self):
        """Test create_backend defaults to X86Backend for unknown platforms."""
        hw_info = HardwareInfo(
            os_type="unknown_platform",
            cpu_model="Unknown CPU",
            cpu_cores=4,
            cpu_features=[],
            total_ram_gb=8.0,
            available_ram_gb=4.0,
            has_gpu=False,
            gpu_model=None,
            gpu_memory_gb=None,
            gpu_compute_capability=None,
            has_thermal_sensors=False,
            has_power_sensors=False
        )
        
        backend = create_backend(hw_info)
        
        assert isinstance(backend, X86Backend), \
            "Should default to X86Backend for unknown platforms"
