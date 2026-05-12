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
        """Detect OS type (linux_x86, jetson_xavier_nx, or android)."""
        system = platform.system()
        machine = platform.machine()
        
        # Check for Android first
        if HardwareDetector._is_android():
            return "android"
        
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
    def _is_android() -> bool:
        """Check if running on Android."""
        # Check for Android-specific markers
        android_markers = [
            "/system/build.prop",
            "/system/bin/app_process",
            "/system/bin/dalvikvm"
        ]
        
        for marker in android_markers:
            if Path(marker).exists():
                return True
        
        # Check environment variables
        if os.environ.get('ANDROID_ROOT') or os.environ.get('ANDROID_DATA'):
            return True
        
        # Check if running in Termux
        if os.environ.get('PREFIX', '').endswith('/com.termux/files/usr'):
            return True
        
        return False
    
    @staticmethod
    def _detect_cpu_model() -> str:
        """Detect CPU model."""
        try:
            # Try to read from /proc/cpuinfo
            with open('/proc/cpuinfo', 'r') as f:
                content = f.read()
                
                # For x86/x64
                for line in content.split('\n'):
                    if line.startswith('model name'):
                        return line.split(':', 1)[1].strip()
                
                # For ARM (Android, Jetson)
                for line in content.split('\n'):
                    if line.startswith('Hardware'):
                        return line.split(':', 1)[1].strip()
                
                # Try Processor field
                for line in content.split('\n'):
                    if line.startswith('Processor'):
                        return line.split(':', 1)[1].strip()
        except Exception as e:
            logger.debug(f"Error reading /proc/cpuinfo: {e}")
        
        # Try Android-specific properties
        if HardwareDetector._is_android():
            try:
                # Try to read from build.prop
                with open('/system/build.prop', 'r') as f:
                    for line in f:
                        if line.startswith('ro.product.board') or line.startswith('ro.board.platform'):
                            return line.split('=', 1)[1].strip()
            except Exception as e:
                logger.debug(f"Error reading build.prop: {e}")
        
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
        # First, try to detect if this is a Jetson device
        is_jetson = False
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip().lower()
                if 'jetson' in model or 'xavier' in model or 'nano' in model or 'orin' in model:
                    is_jetson = True
                    logger.debug(f"Detected Jetson device: {model}")
        except:
            pass
        
        # Try different NVML libraries with Jetson-specific handling
        nvml_modules = []
        
        if is_jetson:
            # On Jetson, pynvml might be more reliable due to older CUDA versions
            nvml_modules = [
                ("pynvml", "pynvml (Jetson-optimized)"),
                ("nvidia_ml_py3", "nvidia-ml-py3")
            ]
        else:
            # On other systems, prefer nvidia-ml-py3
            nvml_modules = [
                ("nvidia_ml_py3", "nvidia-ml-py3"),
                ("pynvml", "pynvml (legacy)")
            ]
        
        for module_name, description in nvml_modules:
            try:
                if module_name == "nvidia_ml_py3":
                    import nvidia_ml_py3 as nvml
                else:
                    import pynvml as nvml
                
                logger.debug(f"Trying GPU detection with {description}")
                
                nvml.nvmlInit()
                device_count = nvml.nvmlDeviceGetCount()
                
                if device_count > 0:
                    handle = nvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_model = nvml.nvmlDeviceGetName(handle)
                    
                    # Decode if bytes
                    if isinstance(gpu_model, bytes):
                        gpu_model = gpu_model.decode('utf-8')
                    
                    # Get memory info
                    mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_gb = mem_info.total / (1024 ** 3)
                    
                    # Get compute capability
                    try:
                        major = nvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
                        minor = nvml.nvmlDeviceGetCudaComputeCapability(handle)[1]
                        gpu_compute_capability = f"{major}.{minor}"
                    except:
                        gpu_compute_capability = "Unknown"
                    
                    nvml.nvmlShutdown()
                    
                    logger.info(f"GPU detected using {description}: {gpu_model}")
                    return True, gpu_model, round(gpu_memory_gb, 2), gpu_compute_capability
                
                nvml.nvmlShutdown()
                logger.debug(f"No GPU devices found with {description}")
                
            except ImportError:
                logger.debug(f"{module_name} not available")
                continue
            except Exception as e:
                logger.debug(f"GPU detection failed with {description}: {e}")
                continue
        
        # If NVML libraries fail, try nvidia-smi as fallback for Jetson
        if is_jetson:
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                                     capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0 and result.stdout.strip():
                    lines = result.stdout.strip().split('\n')
                    if lines:
                        parts = lines[0].split(', ')
                        if len(parts) >= 2:
                            gpu_model = parts[0].strip()
                            try:
                                gpu_memory_mb = float(parts[1].strip())
                                gpu_memory_gb = gpu_memory_mb / 1024
                                logger.info(f"GPU detected via nvidia-smi fallback: {gpu_model}")
                                return True, gpu_model, round(gpu_memory_gb, 2), "Unknown"
                            except ValueError:
                                pass
            except Exception as e:
                logger.debug(f"nvidia-smi fallback failed: {e}")
        
        logger.debug("No GPU detected")
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
