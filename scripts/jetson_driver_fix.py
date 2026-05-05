#!/usr/bin/env python3
"""
Jetson Driver Diagnostic and Fix Script

This script helps diagnose and fix NVIDIA driver issues on Jetson devices.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and return success status and output."""
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    if description:
        print(f"Purpose: {description}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        
        print(f"Exit code: {result.returncode}")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
        return False, "", "Timeout"
    except Exception as e:
        print(f"❌ Command failed: {e}")
        return False, "", str(e)

def check_jetson_model():
    """Check if this is a Jetson device and what model."""
    print("\n" + "="*60)
    print("CHECKING JETSON MODEL")
    print("="*60)
    
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
            print(f"✅ Device model: {model}")
            
            if any(keyword in model.lower() for keyword in ['jetson', 'xavier', 'nano', 'orin']):
                print("✅ This is a Jetson device")
                return True, model
            else:
                print("⚠️ This may not be a Jetson device")
                return False, model
    except Exception as e:
        print(f"❌ Could not read device model: {e}")
        return False, "Unknown"

def check_basic_system():
    """Check basic system information."""
    print("\n" + "="*60)
    print("BASIC SYSTEM CHECK")
    print("="*60)
    
    # Check architecture
    success, stdout, stderr = run_command("uname -a", "Check system architecture")
    
    # Check Ubuntu version
    success, stdout, stderr = run_command("lsb_release -a", "Check Ubuntu version")
    
    # Check if we're running as root
    if os.geteuid() == 0:
        print("✅ Running as root")
    else:
        print("⚠️ Not running as root (some commands may fail)")

def check_nvidia_packages():
    """Check what NVIDIA packages are installed."""
    print("\n" + "="*60)
    print("NVIDIA PACKAGE CHECK")
    print("="*60)
    
    # Check for JetPack
    success, stdout, stderr = run_command("dpkg -l | grep -i nvidia", "List NVIDIA packages")
    
    if not success or not stdout.strip():
        print("❌ No NVIDIA packages found")
        return False
    
    # Check specifically for JetPack
    success, stdout, stderr = run_command("dpkg -l | grep jetpack", "Check JetPack installation")
    
    # Check for CUDA
    success, stdout, stderr = run_command("dpkg -l | grep cuda", "Check CUDA packages")
    
    return True

def check_nvidia_files():
    """Check for NVIDIA driver files."""
    print("\n" + "="*60)
    print("NVIDIA FILES CHECK")
    print("="*60)
    
    # Check for nvidia-smi
    nvidia_smi_paths = [
        "/usr/bin/nvidia-smi",
        "/usr/local/cuda/bin/nvidia-smi",
        "/opt/nvidia/l4t-usb-device-mode/nvidia-smi"
    ]
    
    nvidia_smi_found = False
    for path in nvidia_smi_paths:
        if Path(path).exists():
            print(f"✅ Found nvidia-smi at: {path}")
            nvidia_smi_found = True
            break
    
    if not nvidia_smi_found:
        print("❌ nvidia-smi not found in standard locations")
        # Try to find it
        success, stdout, stderr = run_command("find /usr -name nvidia-smi 2>/dev/null", "Search for nvidia-smi")
        if stdout.strip():
            print(f"Found nvidia-smi at: {stdout.strip()}")
        else:
            print("❌ nvidia-smi not found anywhere")
    
    # Check for NVML library
    nvml_paths = [
        "/usr/lib/aarch64-linux-gnu/libnvidia-ml.so",
        "/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1",
        "/usr/local/cuda/lib64/libnvidia-ml.so",
        "/usr/local/cuda/targets/aarch64-linux/lib/libnvidia-ml.so"
    ]
    
    nvml_found = False
    for path in nvml_paths:
        if Path(path).exists():
            print(f"✅ Found NVML library at: {path}")
            nvml_found = True
            break
    
    if not nvml_found:
        print("❌ NVML library not found")
        # Try to find it
        success, stdout, stderr = run_command("find /usr -name '*nvidia-ml*' 2>/dev/null", "Search for NVML library")
        if stdout.strip():
            print(f"Found NVML files:\n{stdout}")
        else:
            print("❌ No NVML library found")
    
    return nvidia_smi_found and nvml_found

def check_python_environment():
    """Check Python environment and installed packages."""
    print("\n" + "="*60)
    print("PYTHON ENVIRONMENT CHECK")
    print("="*60)
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check pip packages
    success, stdout, stderr = run_command("pip list | grep -i nvidia", "Check NVIDIA Python packages")
    
    # Try to import the packages
    print("\nTesting Python imports:")
    
    try:
        import pynvml
        print("✅ pynvml imported successfully")
    except ImportError as e:
        print(f"❌ pynvml import failed: {e}")
    except Exception as e:
        print(f"❌ pynvml error: {e}")
    
    try:
        import nvidia_ml_py3
        print("✅ nvidia-ml-py3 imported successfully")
    except ImportError as e:
        print(f"❌ nvidia-ml-py3 import failed: {e}")
    except Exception as e:
        print(f"❌ nvidia-ml-py3 error: {e}")

def suggest_fixes():
    """Suggest fixes based on the diagnosis."""
    print("\n" + "="*60)
    print("SUGGESTED FIXES")
    print("="*60)
    
    print("Based on the diagnosis above, try these fixes in order:")
    print()
    
    print("1. INSTALL/REINSTALL JETPACK:")
    print("   sudo apt update")
    print("   sudo apt install nvidia-jetpack")
    print("   sudo reboot")
    print()
    
    print("2. IF JETPACK FAILS, TRY MANUAL DRIVER INSTALL:")
    print("   # Check what's available")
    print("   apt search nvidia-driver")
    print("   # Install appropriate driver")
    print("   sudo apt install nvidia-driver-470  # or latest version")
    print("   sudo reboot")
    print()
    
    print("3. INSTALL PYTHON PACKAGES:")
    print("   pip install pynvml")
    print("   # OR")
    print("   pip install nvidia-ml-py3")
    print()
    
    print("4. CHECK ENVIRONMENT VARIABLES:")
    print("   export PATH=/usr/local/cuda/bin:$PATH")
    print("   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
    print("   # Add to ~/.bashrc to make permanent")
    print()
    
    print("5. IF STILL FAILING, CHECK JETSON DEVELOPER FORUM:")
    print("   https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/")
    print()
    
    print("6. ALTERNATIVE: USE JETPACK SDK MANAGER")
    print("   Download from: https://developer.nvidia.com/nvidia-sdk-manager")
    print("   Flash a fresh JetPack image")

def main():
    """Run complete Jetson driver diagnosis."""
    print("JETSON DRIVER DIAGNOSTIC TOOL")
    print("="*60)
    print("This tool will diagnose NVIDIA driver issues on Jetson devices.")
    print("Run with sudo for complete diagnosis.")
    print()
    
    # Check if this is a Jetson
    is_jetson, model = check_jetson_model()
    
    if not is_jetson:
        print("⚠️ This doesn't appear to be a Jetson device.")
        print("This diagnostic is specifically for NVIDIA Jetson devices.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Run all checks
    check_basic_system()
    packages_ok = check_nvidia_packages()
    files_ok = check_nvidia_files()
    check_python_environment()
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    
    print(f"Jetson device: {'✅' if is_jetson else '❌'}")
    print(f"NVIDIA packages: {'✅' if packages_ok else '❌'}")
    print(f"NVIDIA files: {'✅' if files_ok else '❌'}")
    
    if packages_ok and files_ok:
        print("\n✅ NVIDIA drivers appear to be installed correctly.")
        print("The issue may be with Python package installation or environment.")
        print("Try: pip install pynvml")
    else:
        print("\n❌ NVIDIA drivers are not properly installed.")
        print("You need to install JetPack or NVIDIA drivers.")
    
    suggest_fixes()

if __name__ == "__main__":
    main()