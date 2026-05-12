#!/usr/bin/env python3
"""
Test script for Jetson GPU detection and monitoring.

This script helps diagnose GPU detection issues on NVIDIA Jetson devices.
"""

import sys
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_nvidia_smi():
    """Test if nvidia-smi command works."""
    print("=" * 60)
    print("Testing nvidia-smi command...")
    print("=" * 60)
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ nvidia-smi works:")
            print(result.stdout)
            return True
        else:
            print("❌ nvidia-smi failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi command not found")
        return False
    except Exception as e:
        print(f"❌ nvidia-smi error: {e}")
        return False

def test_nvidia_ml_py3():
    """Test nvidia-ml-py3 library."""
    print("\n" + "=" * 60)
    print("Testing nvidia-ml-py3...")
    print("=" * 60)
    
    try:
        import nvidia_ml_py3 as nvml
        print("✅ nvidia-ml-py3 imported successfully")
        
        try:
            nvml.nvmlInit()
            print("✅ NVML initialized")
            
            device_count = nvml.nvmlDeviceGetCount()
            print(f"✅ Device count: {device_count}")
            
            if device_count > 0:
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                name = nvml.nvmlDeviceGetName(handle)
                print(f"✅ GPU 0 name: {name}")
                
                # Test memory info
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                total_gb = mem_info.total / (1024**3)
                print(f"✅ GPU memory: {total_gb:.2f} GB")
                
                # Test temperature
                try:
                    temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                    print(f"✅ GPU temperature: {temp}°C")
                except Exception as e:
                    print(f"⚠️ Temperature read failed: {e}")
                
                # Test power
                try:
                    power_mw = nvml.nvmlDeviceGetPowerUsage(handle)
                    power_w = power_mw / 1000.0
                    print(f"✅ GPU power: {power_w:.1f}W")
                except Exception as e:
                    print(f"⚠️ Power read failed: {e}")
                
                # Test utilization
                try:
                    util = nvml.nvmlDeviceGetUtilizationRates(handle)
                    print(f"✅ GPU utilization: {util.gpu}%")
                except Exception as e:
                    print(f"⚠️ Utilization read failed: {e}")
            
            nvml.nvmlShutdown()
            print("✅ NVML shutdown successful")
            return True
            
        except Exception as e:
            print(f"❌ NVML operation failed: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ nvidia-ml-py3 not available: {e}")
        return False
    except Exception as e:
        print(f"❌ nvidia-ml-py3 error: {e}")
        return False

def test_pynvml():
    """Test legacy pynvml library."""
    print("\n" + "=" * 60)
    print("Testing pynvml (legacy)...")
    print("=" * 60)
    
    try:
        import pynvml
        print("✅ pynvml imported successfully")
        
        try:
            pynvml.nvmlInit()
            print("✅ NVML initialized")
            
            device_count = pynvml.nvmlDeviceGetCount()
            print(f"✅ Device count: {device_count}")
            
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                print(f"✅ GPU 0 name: {name}")
                
                # Test memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_gb = mem_info.total / (1024**3)
                print(f"✅ GPU memory: {total_gb:.2f} GB")
                
                # Test temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    print(f"✅ GPU temperature: {temp}°C")
                except Exception as e:
                    print(f"⚠️ Temperature read failed: {e}")
                
                # Test power
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    power_w = power_mw / 1000.0
                    print(f"✅ GPU power: {power_w:.1f}W")
                except Exception as e:
                    print(f"⚠️ Power read failed: {e}")
            
            pynvml.nvmlShutdown()
            print("✅ NVML shutdown successful")
            return True
            
        except Exception as e:
            print(f"❌ NVML operation failed: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ pynvml not available: {e}")
        return False
    except Exception as e:
        print(f"❌ pynvml error: {e}")
        return False

def test_benchmark_detection():
    """Test the benchmark framework's GPU detection."""
    print("\n" + "=" * 60)
    print("Testing benchmark framework GPU detection...")
    print("=" * 60)
    
    try:
        from llm_benchmark.hardware.detector import HardwareDetector
        
        hw_info = HardwareDetector.detect()
        
        print(f"Platform: {hw_info.os_type}")
        print(f"CPU: {hw_info.cpu_model} ({hw_info.cpu_cores} cores)")
        print(f"RAM: {hw_info.total_ram_gb:.2f} GB")
        print(f"Has GPU: {hw_info.has_gpu}")
        
        if hw_info.has_gpu:
            print(f"✅ GPU detected: {hw_info.gpu_model}")
            print(f"✅ GPU Memory: {hw_info.gpu_memory_gb:.2f} GB")
            if hasattr(hw_info, 'gpu_compute_capability'):
                print(f"✅ Compute Capability: {hw_info.gpu_compute_capability}")
            return True
        else:
            print("❌ No GPU detected by benchmark framework")
            return False
            
    except Exception as e:
        print(f"❌ Benchmark detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_jetson_specific():
    """Check Jetson-specific information."""
    print("\n" + "=" * 60)
    print("Checking Jetson-specific information...")
    print("=" * 60)
    
    # Check if this is a Jetson device
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
            print(f"Device model: {model}")
            
            if 'jetson' in model.lower() or 'xavier' in model.lower() or 'nano' in model.lower():
                print("✅ This appears to be a Jetson device")
                return True
            else:
                print("⚠️ This may not be a Jetson device")
                return False
    except Exception as e:
        print(f"⚠️ Could not read device model: {e}")
    
    # Check Jetpack version
    try:
        import subprocess
        result = subprocess.run(['dpkg', '-l', 'nvidia-jetpack'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ JetPack installed:")
            for line in result.stdout.split('\n'):
                if 'nvidia-jetpack' in line:
                    print(f"  {line}")
        else:
            print("⚠️ JetPack not found via dpkg")
    except Exception as e:
        print(f"⚠️ Could not check JetPack: {e}")
    
    return False

def main():
    """Run all GPU detection tests."""
    print("JETSON GPU DETECTION DIAGNOSTIC")
    print("=" * 60)
    print("This script tests GPU detection on NVIDIA Jetson devices.")
    print("Run this on your Jetson device to diagnose GPU detection issues.")
    print()
    
    # Check if this is a Jetson
    is_jetson = check_jetson_specific()
    
    # Test nvidia-smi
    nvidia_smi_works = test_nvidia_smi()
    
    # Test libraries
    nvidia_ml_py3_works = test_nvidia_ml_py3()
    pynvml_works = test_pynvml()
    
    # Test benchmark framework
    benchmark_works = test_benchmark_detection()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"Jetson device: {'✅' if is_jetson else '⚠️'}")
    print(f"nvidia-smi: {'✅' if nvidia_smi_works else '❌'}")
    print(f"nvidia-ml-py3: {'✅' if nvidia_ml_py3_works else '❌'}")
    print(f"pynvml: {'✅' if pynvml_works else '❌'}")
    print(f"Benchmark detection: {'✅' if benchmark_works else '❌'}")
    
    print("\nRECOMMENDATIONS:")
    
    if not nvidia_smi_works:
        print("❌ NVIDIA drivers not working. Install JetPack or NVIDIA drivers.")
    elif not nvidia_ml_py3_works and not pynvml_works:
        print("❌ No NVML library available. Install one of:")
        print("   pip install nvidia-ml-py3")
        print("   pip install pynvml")
    elif nvidia_ml_py3_works and not benchmark_works:
        print("⚠️ nvidia-ml-py3 works but benchmark detection fails.")
        print("   This may be a bug in the benchmark framework.")
    elif pynvml_works and not nvidia_ml_py3_works:
        print("⚠️ Only pynvml works. Consider installing nvidia-ml-py3:")
        print("   pip install nvidia-ml-py3")
    elif benchmark_works:
        print("✅ Everything working correctly!")
    else:
        print("❌ Unknown issue. Check the detailed output above.")

if __name__ == "__main__":
    main()