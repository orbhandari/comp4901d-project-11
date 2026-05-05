#!/bin/bash

# Jetson Driver Installation Script
# This script installs NVIDIA drivers and Python packages for Jetson devices

set -e  # Exit on any error

echo "============================================================"
echo "JETSON DRIVER INSTALLATION SCRIPT"
echo "============================================================"
echo "This script will install NVIDIA drivers and Python packages"
echo "for GPU detection on Jetson devices."
echo ""

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "✅ Running as root"
else
   echo "❌ This script must be run as root"
   echo "Please run: sudo $0"
   exit 1
fi

# Check if this is a Jetson device
echo "Checking if this is a Jetson device..."
if [ -f /proc/device-tree/model ]; then
    MODEL=$(cat /proc/device-tree/model)
    echo "Device model: $MODEL"
    
    if [[ $MODEL == *"Jetson"* ]] || [[ $MODEL == *"Xavier"* ]] || [[ $MODEL == *"Nano"* ]] || [[ $MODEL == *"Orin"* ]]; then
        echo "✅ This is a Jetson device"
    else
        echo "⚠️ This may not be a Jetson device"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo "⚠️ Cannot determine device model"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "============================================================"
echo "STEP 1: UPDATE SYSTEM PACKAGES"
echo "============================================================"

apt update
apt upgrade -y

echo ""
echo "============================================================"
echo "STEP 2: INSTALL JETPACK (NVIDIA DRIVERS)"
echo "============================================================"

# Check if JetPack is already installed
if dpkg -l | grep -q nvidia-jetpack; then
    echo "✅ JetPack already installed"
    dpkg -l | grep nvidia-jetpack
else
    echo "Installing JetPack..."
    apt install -y nvidia-jetpack
fi

echo ""
echo "============================================================"
echo "STEP 3: INSTALL ADDITIONAL NVIDIA PACKAGES"
echo "============================================================"

# Install additional packages that might be needed
apt install -y \
    nvidia-cuda-toolkit \
    nvidia-cuda-dev \
    nvidia-utils-470 \
    libnvidia-ml1 \
    || echo "⚠️ Some packages may not be available on this system"

echo ""
echo "============================================================"
echo "STEP 4: INSTALL PYTHON PACKAGES"
echo "============================================================"

# Install Python packages for the current user (not as root)
echo "Installing Python packages for user..."

# Get the original user (not root)
ORIGINAL_USER=${SUDO_USER:-$USER}
ORIGINAL_HOME=$(eval echo ~$ORIGINAL_USER)

if [ "$ORIGINAL_USER" != "root" ]; then
    echo "Installing for user: $ORIGINAL_USER"
    
    # Install pynvml (most reliable on Jetson)
    sudo -u $ORIGINAL_USER pip install pynvml
    
    # Try to install nvidia-ml-py3 as well
    sudo -u $ORIGINAL_USER pip install nvidia-ml-py3 || echo "⚠️ nvidia-ml-py3 installation failed (this is OK)"
    
else
    echo "Installing for root user..."
    pip install pynvml
    pip install nvidia-ml-py3 || echo "⚠️ nvidia-ml-py3 installation failed (this is OK)"
fi

echo ""
echo "============================================================"
echo "STEP 5: SET ENVIRONMENT VARIABLES"
echo "============================================================"

# Add CUDA to PATH and LD_LIBRARY_PATH
BASHRC_FILE="$ORIGINAL_HOME/.bashrc"

if [ "$ORIGINAL_USER" != "root" ]; then
    echo "Adding CUDA environment variables to $BASHRC_FILE"
    
    # Check if already added
    if ! grep -q "CUDA PATH" "$BASHRC_FILE" 2>/dev/null; then
        sudo -u $ORIGINAL_USER tee -a "$BASHRC_FILE" > /dev/null << 'EOF'

# CUDA PATH (added by Jetson driver installation script)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF
        echo "✅ Environment variables added to $BASHRC_FILE"
    else
        echo "✅ Environment variables already present"
    fi
fi

echo ""
echo "============================================================"
echo "STEP 6: VERIFICATION"
echo "============================================================"

echo "Checking nvidia-smi..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi found"
    nvidia-smi --version || echo "⚠️ nvidia-smi version check failed"
else
    echo "❌ nvidia-smi not found"
fi

echo ""
echo "Checking NVML library..."
NVML_PATHS=(
    "/usr/lib/aarch64-linux-gnu/libnvidia-ml.so"
    "/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1"
    "/usr/local/cuda/lib64/libnvidia-ml.so"
)

NVML_FOUND=false
for path in "${NVML_PATHS[@]}"; do
    if [ -f "$path" ]; then
        echo "✅ NVML library found: $path"
        NVML_FOUND=true
        break
    fi
done

if [ "$NVML_FOUND" = false ]; then
    echo "❌ NVML library not found"
fi

echo ""
echo "============================================================"
echo "INSTALLATION COMPLETE"
echo "============================================================"

echo "Next steps:"
echo "1. Reboot your system:"
echo "   sudo reboot"
echo ""
echo "2. After reboot, test GPU detection:"
echo "   python scripts/test_jetson_gpu.py"
echo ""
echo "3. If still having issues, run the diagnostic:"
echo "   python scripts/jetson_driver_fix.py"
echo ""
echo "4. Test the benchmark framework:"
echo "   python -c \"from llm_benchmark.hardware.detector import HardwareDetector; print(HardwareDetector.detect())\""

echo ""
echo "⚠️  IMPORTANT: You must reboot for the drivers to take effect!"
echo "Run: sudo reboot"