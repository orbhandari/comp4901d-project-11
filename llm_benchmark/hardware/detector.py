"""
Hardware detection module.

Detects platform type, CPU/GPU capabilities, memory, and sensors.
"""

import logging
import os
import platform
from pathlib import Path
from typing import List, Optional

import psutil

from llm_benchmark.models import HardwareInfo

logger = logging.getLogger(__name__)


class HardwareDetector:
    """Detect hardware platform and capabilities."""
    
    @staticmethod
    def detect() -> HardwareInfo:
        """
        Detect hardware platform and capabilities.
        
        Returns:
            HardwareInfo with detected platform information
        """
        logger.info("Detecting hardware platform...")
        
        # Detect OS type
        os_type = HardwareDetector._detect_os_type()
        
        # Detect CPU information
        cpu_model = HardwareDetector._detect_cpu_model()
        cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count()
        cpu_features = HardwareDetector._detect_cpu_features()
        
        # Detect memory
        mem = psutil.virtual_memory()
        total_ram_gb = mem.total / (1024 ** 3)
        available_ram_gb = mem.available / (1024 ** 3)
        
        # Detect GPU
        has_gpu, gpu_model, gpu_memory_gb, gpu_compute_capability = \
            HardwareDetector._detect_gpu()
        
        # Detect sensors
        has_thermal_sensors = HardwareDetector._detect_thermal_sensors()
        has_power_sensors = HardwareDetector._detect_power_sensors()
        
        hw_info = HardwareInfo(
            os_type=os_type,
            cpu_model=cpu_model,
            cpu_cores=cpu_cores,
            cpu_features=cpu_features,
            total_ram_gb=round(total_ram_gb, 2),
            available_ram_gb=round(available_ram_gb, 2),
            has_gpu=has_gpu,
            gpu_model=gpu_model,
            gpu_memory_gb=gpu_memory_gb,
            gpu_compute_capability=gpu_compute_capability,
            has_thermal_sensors=has_thermal_sensors,
            has_power_sensors=has_power_sensors
        )
        
        logger.info(f"Detected platform: {os_type}")
        logger.info(f"CPU: {cpu_model} ({cpu_cores} cores)")
        logger.info(f"RAM: {total_ram_gb:.2f} GB total, {available_ram_gb:.2f} GB available")
        if has_gpu:
            logger.info(f"GPU: {gpu_model} ({gpu_memory_gb:.2f} GB)")
        else:
            logger.info("GPU: Not detected")
        
        return hw_info
    
    @staticmethod
    def _detect_os_type() -> str:
        """Detect OS type (linux_x86 or jetson_xavier_nx)."""
        system = platform.system()
        machine = platform.machine()
        
        if system != "Linux":
            logger.warning(f"Unsupported OS: {system}. Assuming linux_x86.")
            return "linux_x86"
        
        # Check for Jetson markers
        jetson_markers = [
            "/etc/nv_tegra_release",
            "/sys/firmware/devicetree/base/model"
        ]
        
        for marker in jetson_markers:
            if Path(marker).exists():
                try:
                    with open(marker, 'r') as f:
                        content = f.read().lower()
                        if 'jetson' in content or 'xavier' in content:
                            return "jetson_xavier_nx"
                except Exception as e:
                    logger.debug(f"Error reading {marker}: {e}")
        
        # Default to x86
        return "linux_x86"
    
    @staticmethod
    def _detect_cpu_model() -> str:
        """Detect CPU model."""
        try:
            # Try to read from /proc/cpuinfo
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('model name'):
                        return line.split(':', 1)[1].strip()
        except Exception as e:
            logger.debug(f"Error reading /proc/cpuinfo: {e}")
        
        # Fallback to platform.processor()
        processor = platform.processor()
        if processor:
            return processor
        
        return "Unknown CPU"
    
    @staticmethod
    def _detect_cpu_features() -> List[str]:
        """Detect CPU features (SIMD instructions)."""
        features = []
        
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('flags'):
                        flags = line.split(':', 1)[1].strip().split()
                        
                        # Check for common SIMD features
                        simd_features = ['avx', 'avx2', 'avx512f', 'sse', 'sse2', 'sse3', 'ssse3', 'sse4_1', 'sse4_2']
                        for feature in simd_features:
                            if feature in flags:
                                features.append(feature)
                        break
        except Exception as e:
            logger.debug(f"Error detecting CPU features: {e}")
        
        return features
    
    @staticmethod
    def _detect_gpu() -> tuple:
        """
        Detect GPU availability and capabilities.
        
        Returns:
            Tuple of (has_gpu, gpu_model, gpu_memory_gb, gpu_compute_capability)
        """
        try:
            import pynvml
            
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_model = pynvml.nvmlDeviceGetName(handle)
                
                # Decode if bytes
                if isinstance(gpu_model, bytes):
                    gpu_model = gpu_model.decode('utf-8')
                
                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_gb = mem_info.total / (1024 ** 3)
                
                # Get compute capability
                try:
                    major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
                    minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[1]
                    gpu_compute_capability = f"{major}.{minor}"
                except:
                    gpu_compute_capability = "Unknown"
                
                pynvml.nvmlShutdown()
                
                return True, gpu_model, round(gpu_memory_gb, 2), gpu_compute_capability
            
            pynvml.nvmlShutdown()
        except Exception as e:
            logger.debug(f"GPU detection failed: {e}")
        
        return False, None, None, None
    
    @staticmethod
    def _detect_thermal_sensors() -> bool:
        """Detect if thermal sensors are available."""
        thermal_dir = Path("/sys/class/thermal")
        
        if not thermal_dir.exists():
            return False
        
        # Check if any thermal zones exist
        thermal_zones = list(thermal_dir.glob("thermal_zone*"))
        return len(thermal_zones) > 0
    
    @staticmethod
    def _detect_power_sensors() -> bool:
        """Detect if power sensors are available."""
        hwmon_dir = Path("/sys/class/hwmon")
        
        if not hwmon_dir.exists():
            return False
        
        # Check if any hwmon devices exist
        hwmon_devices = list(hwmon_dir.glob("hwmon*"))
        
        # Also check for Jetson-specific power rails
        jetson_power_paths = [
            "/sys/bus/i2c/drivers/ina3221x",
            "/sys/devices/3160000.i2c/i2c-0/0-0040/iio:device0"
        ]
        
        for path in jetson_power_paths:
            if Path(path).exists():
                return True
        
        return len(hwmon_devices) > 0
