#!/usr/bin/env python3
"""
Debug Android Ablation Results

This script helps diagnose issues with Android ablation study results.
"""

import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_llama_server_cache_modes():
    """Test llama-server with different cache modes to verify behavior."""
    print("="*60)
    print("TESTING LLAMA-SERVER CACHE MODES")
    print("="*60)
    
    try:
        from llm_benchmark.inference.native_llama_server import NativeLlamaServer, ABLATION_CACHE_CONFIG
        
        # Test model path (adjust as needed)
        model_path = "~/storage/shared/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
        model_path = Path(model_path).expanduser()
        
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            print("Please adjust the model path in the script")
            return
        
        test_prompt = "What is artificial intelligence? Explain in simple terms."
        
        for scenario, config in ABLATION_CACHE_CONFIG.items():
            print(f"\n--- Testing {scenario} ---")
            print(f"Description: {config['description']}")
            print(f"Cache mode: {config['cache_mode'].value}")
            print(f"Enable prompt cache: {config['enable_prompt_cache']}")
            
            try:
                # Create llama-server instance
                llm = NativeLlamaServer(
                    model_path=str(model_path),
                    n_ctx=1024,
                    n_threads=4,
                    n_batch=256,
                    cache_mode=config['cache_mode'].value,
                    host="127.0.0.1",
                    port=8080 + hash(scenario) % 1000  # Different port for each test
                )
                
                print(f"✅ Server started (PID: {llm.last_subprocess_pid})")
                
                # Test inference
                start_time = time.time()
                
                response_chunks = []
                for chunk in llm(
                    prompt=test_prompt,
                    max_tokens=50,
                    enable_prompt_cache=config['enable_prompt_cache']
                ):
                    response_chunks.append(chunk)
                    if len(response_chunks) == 1:  # First token
                        ttft = (time.time() - start_time) * 1000
                        print(f"TTFT: {ttft:.2f}ms")
                
                total_time = (time.time() - start_time) * 1000
                print(f"Total time: {total_time:.2f}ms")
                print(f"Response chunks: {len(response_chunks)}")
                print(f"Peak memory: {llm.subprocess_peak_memory_kb / 1024:.2f}MB")
                
                # Close server
                llm.close()
                print("✅ Server closed")
                
            except Exception as e:
                print(f"❌ Test failed: {e}")
                import traceback
                traceback.print_exc()
            
            print("-" * 40)
            time.sleep(2)  # Brief pause between tests
    
    except ImportError as e:
        print(f"❌ Cannot import NativeLlamaServer: {e}")
    except Exception as e:
        print(f"❌ Test setup failed: {e}")

def check_memory_monitoring():
    """Check if memory monitoring is working correctly."""
    print("\n" + "="*60)
    print("TESTING MEMORY MONITORING")
    print("="*60)
    
    try:
        import psutil
        import os
        
        # Get current process memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)
        print(f"Current process memory: {initial_memory:.2f}MB")
        
        # Test subprocess memory monitoring
        print("\nTesting subprocess memory monitoring...")
        
        # Start a simple subprocess
        import subprocess
        proc = subprocess.Popen(['sleep', '5'])
        
        try:
            # Monitor its memory
            sub_process = psutil.Process(proc.pid)
            sub_memory = sub_process.memory_info().rss / (1024 * 1024)
            print(f"Subprocess memory: {sub_memory:.2f}MB")
            print(f"Subprocess PID: {proc.pid}")
            
        except Exception as e:
            print(f"❌ Subprocess monitoring failed: {e}")
        finally:
            proc.terminate()
            proc.wait()
        
        print("✅ Memory monitoring appears to work")
        
    except Exception as e:
        print(f"❌ Memory monitoring test failed: {e}")

def analyze_results_file(results_path):
    """Analyze the results file for anomalies."""
    print("\n" + "="*60)
    print("ANALYZING RESULTS FILE")
    print("="*60)
    
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        print("Quantization results:")
        if 'quantization_results' in results:
            for result in results['quantization_results']:
                print(f"  {result['quantization']}: {result['peak_ram_mb']:.2f}MB peak RAM")
        
        print("\nAblation results:")
        if 'ablation_results' in results:
            for result in results['ablation_results']:
                scenario = result['scenario']
                ttft = result['metrics']['ttft_ms']
                memory = result['metrics']['peak_memory_mb']
                cache_mode = result['configuration'].get('cache_mode', 'unknown')
                
                print(f"  {scenario}: TTFT={ttft:.2f}ms, Memory={memory:.2f}MB, Cache={cache_mode}")
        
        # Check for anomalies
        print("\nANOMALY DETECTION:")
        
        if 'ablation_results' in results:
            ablation_memories = [r['metrics']['peak_memory_mb'] for r in results['ablation_results']]
            quant_memories = [r['peak_ram_mb'] for r in results.get('quantization_results', [])]
            
            if quant_memories and ablation_memories:
                avg_quant_memory = sum(quant_memories) / len(quant_memories)
                avg_ablation_memory = sum(ablation_memories) / len(ablation_memories)
                
                ratio = avg_ablation_memory / avg_quant_memory
                
                print(f"Average quantization memory: {avg_quant_memory:.2f}MB")
                print(f"Average ablation memory: {avg_ablation_memory:.2f}MB")
                print(f"Ratio: {ratio:.2f}")
                
                if ratio < 0.5:
                    print("🚨 ANOMALY: Ablation memory much lower than quantization memory")
                    print("   This suggests memory monitoring issues during ablation")
                
                if ratio > 2.0:
                    print("🚨 ANOMALY: Ablation memory much higher than quantization memory")
            
            # Check TTFT patterns
            ttfts = [(r['scenario'], r['metrics']['ttft_ms']) for r in results['ablation_results']]
            ttfts.sort(key=lambda x: x[1])  # Sort by TTFT
            
            print(f"\nTTFT ranking (fastest to slowest):")
            for scenario, ttft in ttfts:
                print(f"  {scenario}: {ttft:.2f}ms")
            
            # Expected order should be: warm_cache < cold_cache < control
            if len(ttfts) >= 3:
                fastest = ttfts[0][0]
                slowest = ttfts[-1][0]
                
                if fastest == 'cold_cache' and slowest == 'control':
                    print("🚨 ANOMALY: Cold cache is fastest, control is slowest")
                    print("   This suggests cache settings might be inverted")
    
    except Exception as e:
        print(f"❌ Results analysis failed: {e}")

def main():
    """Run all diagnostic tests."""
    print("ANDROID ABLATION DIAGNOSTIC TOOL")
    print("="*60)
    print("This tool helps diagnose issues with Android ablation results.")
    print()
    
    # Test memory monitoring
    check_memory_monitoring()
    
    # Test llama-server cache modes
    test_llama_server_cache_modes()
    
    # Analyze results file if provided
    results_files = list(Path("~/storage/shared/benchmark_results").expanduser().glob("*/results.json"))
    if results_files:
        latest_results = max(results_files, key=lambda p: p.stat().st_mtime)
        print(f"\nAnalyzing latest results: {latest_results}")
        analyze_results_file(latest_results)
    else:
        print("\nNo results files found to analyze")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    print("1. Check if llama-server is actually using the cache flags:")
    print("   - Look at llama-server startup logs")
    print("   - Verify --cache-ram and --no-cache-prompt flags")
    
    print("\n2. Verify memory monitoring:")
    print("   - Check if subprocess PID tracking works")
    print("   - Ensure memory sampling thread is running")
    
    print("\n3. Test with smaller context size:")
    print("   - Try context_size: 512 instead of 2048")
    print("   - This might reveal memory measurement issues")
    
    print("\n4. Check llama-server logs:")
    print("   - Look for cache-related messages")
    print("   - Verify which cache mode is actually active")

if __name__ == "__main__":
    main()