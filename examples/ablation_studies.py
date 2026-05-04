#!/usr/bin/env python3
"""
Example: Ablation Studies

This example demonstrates how to run ablation studies to measure the impact
of different optimization strategies (KV cache, prompt caching).

Usage:
    python examples/ablation_studies.py

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
    """Run ablation studies benchmark."""
    
    print("=" * 80)
    print("Ablation Studies Example")
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
        enable_ablation_studies=True,
        kv_cache_types=["ram", "disk"],
        prompt_cache_prefix_lengths=[100, 500],
        output_dir="./results/ablation_studies"
    )
    print(f"✓ Configuration created")
    print(f"  - KV cache types: {', '.join(config.kv_cache_types)}")
    print(f"  - Prompt cache prefix lengths: {config.prompt_cache_prefix_lengths}")
    print()
    
    # Step 2: Detect hardware
    print("Step 2: Detecting hardware...")
    hw_info = HardwareDetector.detect()
    print(f"✓ Hardware detected: {hw_info.os_type}")
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
    
    # Step 6: Run KV cache ablation studies
    print("Step 6: Running KV cache ablation studies...")
    print()
    print("Testing KV cache strategies (RAM vs Disk, Cold vs Warm)...")
    print()
    
    try:
        kv_results = ablation_engine.test_kv_cache_strategies(model_path=model_path)
        
        print("KV Cache Results:")
        print(f"{'Scenario':<30} {'TTFT (ms)':<15} {'Improvement':<15}")
        print("-" * 60)
        
        baseline_ttft = None
        for result in kv_results:
            ttft = result.metrics.get('ttft_ms', 0)
            
            if result.scenario == 'control':
                baseline_ttft = ttft
                improvement = "baseline"
            elif baseline_ttft:
                improvement_pct = ((baseline_ttft - ttft) / baseline_ttft) * 100
                improvement = f"{improvement_pct:+.1f}%"
            else:
                improvement = "N/A"
            
            print(f"{result.scenario:<30} {ttft:<15.2f} {improvement:<15}")
        
        print()
        
    except Exception as e:
        print(f"✗ KV cache ablation failed: {e}")
        print()
    
    # Step 7: Run prompt caching ablation studies
    print("Step 7: Running prompt caching ablation studies...")
    print()
    print("Testing prompt caching with different prefix lengths...")
    print()
    
    try:
        prompt_results = ablation_engine.test_prompt_caching(
            model_path=model_path,
            prefix_lengths=config.prompt_cache_prefix_lengths
        )
        
        print("Prompt Caching Results:")
        print(f"{'Prefix Length':<20} {'Cache Hit Rate':<20} {'Latency Reduction':<20}")
        print("-" * 60)
        
        for result in prompt_results:
            prefix_len = result.configuration.get('prefix_length', 0)
            hit_rate = result.metrics.get('cache_hit_rate', 0)
            latency_reduction = result.metrics.get('latency_reduction_ms', 0)
            
            print(f"{prefix_len:<20} {hit_rate:<20.1f}% {latency_reduction:<20.2f} ms")
        
        print()
        
    except Exception as e:
        print(f"✗ Prompt caching ablation failed: {e}")
        print()
    
    print("=" * 80)
    print("Ablation Studies Complete!")
    print("=" * 80)
    print()
    print("Key Findings:")
    print("  - KV cache strategies can significantly reduce TTFT for repeated prompts")
    print("  - Prompt caching is most effective with longer shared prefixes")
    print("  - RAM-based caching is faster but uses more memory")
    print("  - Disk-based caching is slower but more memory-efficient")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
