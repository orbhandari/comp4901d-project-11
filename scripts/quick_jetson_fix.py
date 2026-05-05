#!/usr/bin/env python3
"""
Quick Jetson GPU Fix Script

This script attempts to quickly fix the most common Jetson GPU detection issues.
"""

import subprocess
import sys
import os

def run_cmd(cmd):
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("QUICK JETSON GPU FIX")
    print("="*50)
    
    # Check if we can run nvidia-smi
    print("1. Checking nvidia-smi...")
    success, stdout, stderr = run_cmd("nvidia-smi")
    
    if success:
        print("✅ nvidia-smi works!")
        print("GPU drivers are installed correctly.")
    else:
        print("❌ nvidia-smi failed")
        print("You need to install NVIDIA drivers first.")
        print("\nTo fix this, run:")
        print("sudo scripts/install_jetson_drivers.sh")
        return
    
    # Check Python packages
    print("\n2. Checking Python packages...")
    
    # Try pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        print(f"✅ pynvml works! Found {count} GPU(s)")
        pynvml.nvmlShutdown()
        
        # Test the benchmark framework
        print("\n3. Testing benchmark framework...")
        try:
            from llm_benchmark.hardware.detector import HardwareDetector
            hw_info = HardwareDetector.detect()
            
            if hw_info.has_gpu:
                print("✅ Benchmark framework detects GPU!")
                print(f"GPU: {hw_info.gpu_model}")
                print(f"Memory: {hw_info.gpu_memory_gb:.2f} GB")
                print("\n🎉 Everything is working correctly!")
            else:
                print("❌ Benchmark framework doesn't detect GPU")
                print("This might be a bug in the framework.")
        except Exception as e:
            print(f"❌ Benchmark framework error: {e}")
        
        return
        
    except ImportError:
        print("❌ pynvml not installed")
    except Exception as e:
        print(f"❌ pynvml error: {e}")
    
    # Try nvidia-ml-py3
    try:
        import nvidia_ml_py3 as nvml
        nvml.nvmlInit()
        count = nvml.nvmlDeviceGetCount()
        print(f"✅ nvidia-ml-py3 works! Found {count} GPU(s)")
        nvml.nvmlShutdown()
        return
    except ImportError:
        print("❌ nvidia-ml-py3 not installed")
    except Exception as e:
        print(f"❌ nvidia-ml-py3 error: {e}")
    
    # If we get here, Python packages are the issue
    print("\n❌ No working Python NVML packages found")
    print("\nTo fix this, install pynvml:")
    print("pip install pynvml")
    print("\nOr try nvidia-ml-py3:")
    print("pip install nvidia-ml-py3")
    
    # Try to install pynvml automatically
    print("\nAttempting to install pynvml automatically...")
    success, stdout, stderr = run_cmd("pip install pynvml")
    
    if success:
        print("✅ pynvml installed successfully!")
        print("Now test again:")
        print("python scripts/test_jetson_gpu.py")
    else:
        print("❌ Failed to install pynvml")
        print("You may need to install it manually:")
        print("pip install pynvml")

if __name__ == "__main__":
    main()