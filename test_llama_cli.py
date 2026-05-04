#!/usr/bin/env python3
"""
Test script to diagnose llama-cli behavior on Android.

This script tests the native llama-cli directly to understand its output format.
"""

import subprocess
import sys
import time
from pathlib import Path

def test_llama_cli():
    """Test llama-cli with a simple prompt."""
    
    # Configuration
    llama_cli = Path("~/llama.cpp/build/bin/llama-cli").expanduser()
    model_path = Path("~/storage/shared/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf").expanduser()
    
    # Check if files exist
    if not llama_cli.exists():
        print(f"ERROR: llama-cli not found at {llama_cli}")
        return 1
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return 1
    
    print(f"✓ llama-cli found: {llama_cli}")
    print(f"✓ Model found: {model_path}")
    print()
    
    # Test 1: Simple generation
    print("=" * 60)
    print("TEST 1: Simple generation with --simple-io")
    print("=" * 60)
    
    cmd = [
        str(llama_cli),
        "-m", str(model_path),
        "-n", "20",  # Generate 20 tokens
        "-p", "Hello, my name is",
        "--simple-io",
        "--no-display-prompt",
        "--log-disable",
        "-e"  # Exit after generation (non-interactive)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    print("Output:")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        elapsed = time.time() - start_time
        
        print(result.stdout)
        print("-" * 60)
        print(f"Return code: {result.returncode}")
        print(f"Elapsed time: {elapsed:.2f}s")
        
        if result.stderr:
            print(f"Stderr: {result.stderr}")
        
    except subprocess.TimeoutExpired:
        print("ERROR: Command timed out after 60 seconds")
        return 1
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    
    print()
    
    # Test 2: Without --simple-io
    print("=" * 60)
    print("TEST 2: Generation without --simple-io")
    print("=" * 60)
    
    cmd = [
        str(llama_cli),
        "-m", str(model_path),
        "-n", "20",
        "-p", "Hello, my name is",
        "--no-display-prompt",
        "--log-disable",
        "-e"  # Exit after generation (non-interactive)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    print("Output:")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        elapsed = time.time() - start_time
        
        print(result.stdout)
        print("-" * 60)
        print(f"Return code: {result.returncode}")
        print(f"Elapsed time: {elapsed:.2f}s")
        
        if result.stderr:
            print(f"Stderr: {result.stderr}")
        
    except subprocess.TimeoutExpired:
        print("ERROR: Command timed out after 60 seconds")
        return 1
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    
    print()
    
    # Test 3: Streaming with Popen
    print("=" * 60)
    print("TEST 3: Streaming with Popen (character-by-character)")
    print("=" * 60)
    
    cmd = [
        str(llama_cli),
        "-m", str(model_path),
        "-n", "20",
        "-p", "Hello, my name is",
        "--no-display-prompt",
        "--log-disable",
        "-e"  # Exit after generation (non-interactive)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    print("Output (streaming):")
    print("-" * 60)
    
    start_time = time.time()
    char_count = 0
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Read character by character with timeout
        timeout = 60
        last_char_time = time.time()
        
        while True:
            # Check timeout
            if time.time() - last_char_time > timeout:
                print(f"\nERROR: Timeout - no output for {timeout} seconds")
                process.kill()
                break
            
            # Check if process finished
            if process.poll() is not None:
                # Read remaining output
                remaining = process.stdout.read()
                if remaining:
                    print(remaining, end='', flush=True)
                    char_count += len(remaining)
                break
            
            # Try to read one character
            try:
                char = process.stdout.read(1)
                if char:
                    print(char, end='', flush=True)
                    char_count += 1
                    last_char_time = time.time()
                else:
                    time.sleep(0.01)
            except:
                time.sleep(0.01)
        
        elapsed = time.time() - start_time
        
        print()
        print("-" * 60)
        print(f"Characters read: {char_count}")
        print(f"Return code: {process.returncode}")
        print(f"Elapsed time: {elapsed:.2f}s")
        
        stderr = process.stderr.read()
        if stderr:
            print(f"Stderr: {stderr}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    
    print()
    print("=" * 60)
    print("All tests complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(test_llama_cli())
