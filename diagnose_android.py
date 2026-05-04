#!/usr/bin/env python3
"""
Diagnostic script for Android setup issues.

Run this on your Android device to diagnose import problems.
"""

import os
import sys
from pathlib import Path

print("=" * 70)
print("ANDROID DIAGNOSTIC SCRIPT")
print("=" * 70)

# 1. Check Python version
print(f"\n1. Python version: {sys.version}")
print(f"   Python executable: {sys.executable}")

# 2. Check current directory
print(f"\n2. Current directory: {os.getcwd()}")

# 3. Check if we're in the right directory
repo_root = Path.cwd()
if not (repo_root / "llm_benchmark").exists():
    print("   ❌ ERROR: Not in repository root!")
    print("   Please cd to comp4901d-project-11 directory")
    sys.exit(1)
else:
    print("   ✅ In repository root")

# 4. Check directory structure
print("\n3. Checking directory structure...")
required_dirs = [
    "llm_benchmark",
    "llm_benchmark/hardware",
    "llm_benchmark/inference",
    "llm_benchmark/metrics",
    "llm_benchmark/profiler",
    "llm_benchmark/orchestrator",
    "llm_benchmark/results",
    "llm_benchmark/visualization",
]

for dir_path in required_dirs:
    full_path = repo_root / dir_path
    if full_path.exists():
        print(f"   ✅ {dir_path}")
    else:
        print(f"   ❌ {dir_path} - MISSING!")

# 5. Check __init__.py files
print("\n4. Checking __init__.py files...")
required_init_files = [
    "llm_benchmark/__init__.py",
    "llm_benchmark/hardware/__init__.py",
    "llm_benchmark/inference/__init__.py",
    "llm_benchmark/metrics/__init__.py",
    "llm_benchmark/profiler/__init__.py",
    "llm_benchmark/orchestrator/__init__.py",
    "llm_benchmark/results/__init__.py",
    "llm_benchmark/visualization/__init__.py",
]

missing_init = []
for init_file in required_init_files:
    full_path = repo_root / init_file
    if full_path.exists():
        print(f"   ✅ {init_file}")
    else:
        print(f"   ❌ {init_file} - MISSING!")
        missing_init.append(init_file)

# 6. Check Python path
print("\n5. Python path:")
for path in sys.path[:5]:  # Show first 5 entries
    print(f"   - {path}")

# 7. Try imports
print("\n6. Testing imports...")
try:
    import llm_benchmark
    print("   ✅ llm_benchmark")
except ImportError as e:
    print(f"   ❌ llm_benchmark: {e}")

try:
    from llm_benchmark.hardware.detector import HardwareDetector
    print("   ✅ llm_benchmark.hardware.detector")
except ImportError as e:
    print(f"   ❌ llm_benchmark.hardware.detector: {e}")

try:
    from llm_benchmark.inference.native_llama import NativeLlamaCpp
    print("   ✅ llm_benchmark.inference.native_llama")
except ImportError as e:
    print(f"   ❌ llm_benchmark.inference.native_llama: {e}")

try:
    from llm_benchmark.results.persistence import ResultsPersistence
    print("   ✅ llm_benchmark.results.persistence")
except ImportError as e:
    print(f"   ❌ llm_benchmark.results.persistence: {e}")

try:
    from llm_benchmark.visualization.visualization_generator import VisualizationGenerator
    print("   ✅ llm_benchmark.visualization.visualization_generator")
except ImportError as e:
    print(f"   ❌ llm_benchmark.visualization.visualization_generator: {e}")

# 8. Summary
print("\n" + "=" * 70)
if missing_init:
    print("❌ ISSUES FOUND!")
    print("\nMissing __init__.py files:")
    for f in missing_init:
        print(f"  - {f}")
    print("\nSOLUTION:")
    print("  1. Re-clone the repository:")
    print("     cd ~")
    print("     rm -rf comp4901d-project-11")
    print("     git clone <repo-url> comp4901d-project-11")
    print("\n  2. Or create missing __init__.py files:")
    for f in missing_init:
        print(f"     touch {f}")
else:
    print("✅ ALL CHECKS PASSED!")
    print("\nIf you still get import errors, try:")
    print("  1. Restart Termux")
    print("  2. Run: export PYTHONPATH=$PWD:$PYTHONPATH")
    print("  3. Run the benchmark again")

print("=" * 70)
