#!/usr/bin/env python3
"""
Example: Basic Quantization Profiling

This example demonstrates how to run a basic quantization profiling benchmark
comparing different quantization levels (Q8_0, Q4_0, Q4_K_M, Q2_K).

Usage:
    python examples/basic_quantization_profiling.py

Requirements:
    - Internet connection for model download (first run only)
    - At least 8GB RAM
    - Models will be cached in ./models directory

**Validates: Requirement 11.6**
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_benchmark.config import BenchmarkConfig
from llm_benchmark.hardware.detector import HardwareDetector
from llm_benchmark.hardware.hal import create_backend
from llm_benchmark.model_manager.manager import ModelManager
from llm_benchmark.profiler.quantization import QuantizationProfiler


def main():
    """Run basic quantization profiling benchmark."""
    
    print("=" * 80)
    print("Basic Quantization Profiling Example")
    print("=" * 80)
    print()
    
    # Step 1: Create configuration
    print("Step 1: Creating configuration...")
    config = BenchmarkConfig(
        repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        models={
            "Q8_0": "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
            "Q4_0": "tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
            "Q2_K": "tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
        },
        model_cache_dir="./models",
        context_size=2048,
        max_tokens=50,
        iterations=3,
        warmup_runs=1,
        output_dir="./results/quantization_profiling"
    )
    print(f"✓ Configuration created")
    print(f"  - Repository: {config.repo_id}")
    print(f"  - Quantizations: {', '.join(config.models.keys())}")
    print(f"  - Iterations: {config.iterations}")
    print()
    
    # Step 2: Detect hardware
    print("Step 2: Detecting hardware...")
    hw_info = HardwareDetector.detect()
    print(f"✓ Hardware detected")
    print(f"  - Platform: {hw_info.os_type}")
    print(f"  - CPU: {hw_info.cpu_model} ({hw_info.cpu_cores} cores)")
    print(f"  - RAM: {hw_info.total_ram_gb:.2f} GB")
    if hw_info.has_gpu:
        print(f"  - GPU: {hw_info.gpu_model} ({hw_info.gpu_memory_gb:.2f} GB)")
    print()
    
    # Step 3: Create backend
    print("Step 3: Creating hardware backend...")
    backend = create_backend(hw_info)
    backend.optimize_for_inference()
    print(f"✓ Backend created and optimized")
    print()
    
    # Step 4: Initialize model manager
    print("Step 4: Initializing model manager...")
    model_manager = ModelManager(
        cache_dir=config.model_cache_dir,
        hf_token=None  # Use HF_TOKEN environment variable if needed
    )
    print(f"✓ Model manager initialized")
    print(f"  - Cache directory: {config.model_cache_dir}")
    print()
    
    # Step 5: Download/acquire models
    print("Step 5: Acquiring models...")
    model_paths = {}
    for quant, filename in config.models.items():
        print(f"  - Acquiring {quant}...")
        try:
            model_info = model_manager.get_model(
                repo_id=config.repo_id,
                filename=filename
            )
            model_paths[quant] = model_info.local_path
            print(f"    ✓ {quant}: {model_info.size_mb:.2f} MB")
        except Exception as e:
            print(f"    ✗ Failed to acquire {quant}: {e}")
            continue
    print()
    
    if not model_paths:
        print("ERROR: No models could be acquired. Exiting.")
        return 1
    
    # Step 6: Initialize profiler
    print("Step 6: Initializing quantization profiler...")
    metrics_collector = backend.get_metrics_collector()
    profiler = QuantizationProfiler(
        backend=backend,
        metrics_collector=metrics_collector
    )
    print(f"✓ Profiler initialized")
    print()
    
    # Step 7: Run profiling
    print("Step 7: Running quantization profiling...")
    print()
    
    test_prompt = "Explain the concept of machine learning in simple terms."
    results = []
    
    for quant, model_path in model_paths.items():
        print(f"Profiling {quant}...")
        print(f"  - Model: {model_path}")
        print(f"  - Prompt: {test_prompt[:50]}...")
        print(f"  - Max tokens: {config.max_tokens}")
        print()
        
        try:
            result = profiler.profile_quantization(
                model_path=model_path,
                quant=quant,
                prompt=test_prompt,
                max_tokens=config.max_tokens
            )
            results.append(result)
            
            print(f"  Results:")
            print(f"    - Load time: {result.load_time_s:.2f}s")
            print(f"    - Peak RAM: {result.peak_ram_mb:.2f} MB")
            print(f"    - TTFT: {result.ttft_ms:.2f} ms")
            print(f"    - Prefill throughput: {result.prefill_tps:.2f} tokens/s")
            print(f"    - Decode throughput: {result.decode_tps:.2f} tokens/s")
            print(f"    - Prompt tokens: {result.prompt_tokens}")
            print(f"    - Output tokens: {result.output_tokens}")
            print()
            
        except Exception as e:
            print(f"  ✗ Profiling failed: {e}")
            print()
            continue
    
    # Step 8: Display comparison
    if len(results) > 1:
        print("=" * 80)
        print("Quantization Comparison")
        print("=" * 80)
        print()
        
        print(f"{'Quantization':<15} {'Load (s)':<12} {'RAM (MB)':<12} {'TTFT (ms)':<12} {'Decode (t/s)':<15}")
        print("-" * 80)
        
        for result in results:
            print(f"{result.quantization:<15} "
                  f"{result.load_time_s:<12.2f} "
                  f"{result.peak_ram_mb:<12.2f} "
                  f"{result.ttft_ms:<12.2f} "
                  f"{result.decode_tps:<15.2f}")
        print()
        
        # Find best quantization for different metrics
        best_speed = max(results, key=lambda r: r.decode_tps)
        best_memory = min(results, key=lambda r: r.peak_ram_mb)
        best_ttft = min(results, key=lambda r: r.ttft_ms)
        
        print("Recommendations:")
        print(f"  - Fastest decode: {best_speed.quantization} ({best_speed.decode_tps:.2f} tokens/s)")
        print(f"  - Lowest memory: {best_memory.quantization} ({best_memory.peak_ram_mb:.2f} MB)")
        print(f"  - Lowest TTFT: {best_ttft.quantization} ({best_ttft.ttft_ms:.2f} ms)")
        print()
    
    print("=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
