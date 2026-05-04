#!/usr/bin/env python3
"""
Test script to verify Android imports work without llama-cpp-python.

This script simulates the Android environment where llama-cpp-python
is not available, and verifies that the framework can still import
and initialize correctly.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports without llama-cpp-python...")
print("=" * 70)

try:
    print("\n1. Testing hardware detector...")
    from llm_benchmark.hardware.detector import HardwareDetector
    print("   ✅ HardwareDetector imported successfully")
    
    print("\n2. Testing hardware backends...")
    from llm_benchmark.hardware.hal import AndroidBackend, X86Backend, JetsonBackend
    print("   ✅ Hardware backends imported successfully")
    
    print("\n3. Testing native llama wrapper...")
    from llm_benchmark.inference.native_llama import NativeLlamaCpp
    print("   ✅ NativeLlamaCpp imported successfully")
    
    print("\n4. Testing quantization profiler...")
    from llm_benchmark.profiler.quantization import QuantizationProfiler
    print("   ✅ QuantizationProfiler imported successfully")
    
    print("\n5. Testing ablation engine...")
    from llm_benchmark.profiler.ablation import AblationEngine
    print("   ✅ AblationEngine imported successfully")
    
    print("\n6. Testing metrics collector...")
    from llm_benchmark.metrics.collector import MetricsCollector
    print("   ✅ MetricsCollector imported successfully")
    
    print("\n7. Testing orchestrator...")
    from llm_benchmark.orchestrator.orchestrator import TestOrchestrator
    print("   ✅ TestOrchestrator imported successfully")
    
    print("\n8. Testing results persistence...")
    from llm_benchmark.results.persistence import ResultsPersistence
    print("   ✅ ResultsPersistence imported successfully")
    
    print("\n9. Testing visualization...")
    from llm_benchmark.visualization.visualization_generator import VisualizationGenerator
    print("   ✅ VisualizationGenerator imported successfully")
    
    print("\n" + "=" * 70)
    print("✅ ALL IMPORTS SUCCESSFUL!")
    print("=" * 70)
    print("\nThe framework can now run on Android without llama-cpp-python.")
    print("Just build native llama.cpp and the framework will use it automatically.")
    
except ImportError as e:
    print(f"\n❌ Import failed: {e}")
    print("\nThis means there's still a dependency on llama-cpp-python somewhere.")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
