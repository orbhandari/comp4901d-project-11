#!/usr/bin/env python3
"""
Example: Batch Processing Tests

This example demonstrates how to run batch processing tests to find the
optimal batch size for throughput-oriented workloads.

Usage:
    python examples/batch_processing.py

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
from llm_benchmark.profiler.ablation import AblationEngine


def main():
    """Run batch processing tests."""
    
    print("=" * 80)
    print("Batch Processing Tests Example")
    print("=" * 80)
    print()
    
    # Step 1: Create configuration
    print("Step 1: Creating configuration...")
    config = BenchmarkConfig(
        repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        models={
            "Q4_0": "tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
        },
        model_cache_dir="./models",
        context_size=2048,
        max_tokens=50,
        iterations=2,
        warmup_runs=1,
        enable_batch_testing=True,
        batch_sizes=[1, 2, 4, 8],
        output_dir="./results/batch_processing"
    )
    print(f"✓ Configuration created")
    print(f"  - Batch sizes to test: {config.batch_sizes}")
    print()
    
    # Step 2: Detect hardware
    print("Step 2: Detecting hardware...")
    hw_info = HardwareDetector.detect()
    print(f"✓ Hardware detected")
    print(f"  - Platform: {hw_info.os_type}")
    print(f"  - CPU cores: {hw_info.cpu_cores}")
    print(f"  - RAM: {hw_info.total_ram_gb:.2f} GB")
    if hw_info.has_gpu:
        print(f"  - GPU: {hw_info.gpu_model}")
    print()
    
    # Step 3: Create backend
    print("Step 3: Creating hardware backend...")
    backend = create_backend(hw_info)
    backend.optimize_for_inference()
    print(f"✓ Backend created")
    print()
    
    # Step 4: Acquire model
    print("Step 4: Acquiring model...")
    model_manager = ModelManager(
        cache_dir=config.model_cache_dir,
        hf_token=None
    )
    
    try:
        model_info = model_manager.get_model(
            repo_id=config.repo_id,
            filename=list(config.models.values())[0]
        )
        model_path = model_info.local_path
        print(f"✓ Model acquired: {model_info.size_mb:.2f} MB")
        print()
    except Exception as e:
        print(f"✗ Failed to acquire model: {e}")
        return 1
    
    # Step 5: Initialize ablation engine
    print("Step 5: Initializing ablation engine...")
    ablation_engine = AblationEngine(backend=backend)
    print(f"✓ Ablation engine initialized")
    print()
    
    # Step 6: Run batch processing tests
    print("Step 6: Running batch processing tests...")
    print()
    print("Testing different batch sizes to find optimal throughput...")
    print()
    
    try:
        batch_results = ablation_engine.test_batch_sizes(
            model_path=model_path,
            batch_sizes=config.batch_sizes
        )
        
        print("Batch Processing Results:")
        print(f"{'Batch Size':<15} {'Throughput (t/s)':<20} {'Latency (ms)':<20} {'Memory (MB)':<15}")
        print("-" * 70)
        
        best_throughput = None
        best_batch_size = None
        
        for result in batch_results:
            batch_size = result.configuration.get('batch_size', 1)
            throughput = result.metrics.get('aggregate_throughput_tps', 0)
            latency = result.metrics.get('avg_latency_ms', 0)
            memory = result.metrics.get('peak_memory_mb', 0)
            
            print(f"{batch_size:<15} {throughput:<20.2f} {latency:<20.2f} {memory:<15.2f}")
            
            if best_throughput is None or throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = batch_size
        
        print()
        print("=" * 80)
        print("Analysis")
        print("=" * 80)
        print()
        
        if best_batch_size:
            print(f"Optimal batch size: {best_batch_size}")
            print(f"  - Achieves {best_throughput:.2f} tokens/s throughput")
            print()
        
        print("Key Observations:")
        print("  - Larger batch sizes increase aggregate throughput")
        print("  - Per-prompt latency increases with batch size")
        print("  - Memory usage scales with batch size")
        print("  - Optimal batch size balances throughput and latency")
        print()
        
        # Calculate efficiency metrics
        if len(batch_results) > 1:
            single_throughput = batch_results[0].metrics.get('aggregate_throughput_tps', 1)
            
            print("Throughput Scaling:")
            for result in batch_results:
                batch_size = result.configuration.get('batch_size', 1)
                throughput = result.metrics.get('aggregate_throughput_tps', 0)
                scaling = (throughput / single_throughput) if single_throughput > 0 else 0
                efficiency = (scaling / batch_size * 100) if batch_size > 0 else 0
                
                print(f"  - Batch {batch_size}: {scaling:.2f}x speedup ({efficiency:.1f}% efficiency)")
            print()
        
    except Exception as e:
        print(f"✗ Batch processing tests failed: {e}")
        print()
        return 1
    
    print("=" * 80)
    print("Batch Processing Tests Complete!")
    print("=" * 80)
    print()
    print("Recommendations:")
    print(f"  - Use batch size {best_batch_size} for maximum throughput")
    print("  - Use batch size 1 for minimum latency")
    print("  - Monitor memory usage to avoid OOM errors")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
