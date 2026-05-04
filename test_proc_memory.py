#!/usr/bin/env python3
"""
Test script to verify 'ps' command memory reading works on Android.
"""

import os
import subprocess
import time

def read_memory_from_ps(pid):
    """Read memory usage using ps command."""
    try:
        result = subprocess.run(
            ['ps', '-o', 'rss=', '-p', str(pid)],
            capture_output=True,
            text=True,
            timeout=1
        )
        
        if result.returncode == 0 and result.stdout.strip():
            # RSS is in KB
            rss_kb = int(result.stdout.strip())
            rss_mb = rss_kb / 1024
            return rss_mb
        else:
            print(f"ps command returned no data for PID {pid}")
            
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError, FileNotFoundError) as e:
        print(f"Failed to read memory using ps command: {e}")
    
    return 0

# Test 1: Read current process memory
current_pid = os.getpid()
current_memory = read_memory_from_ps(current_pid)
print(f"Current process (PID {current_pid}): {current_memory:.2f} MB")

# Test 2: Spawn a subprocess and read its memory
print("\nSpawning a subprocess...")
process = subprocess.Popen(
    ["sleep", "5"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

subprocess_pid = process.pid
print(f"Subprocess PID: {subprocess_pid}")

# Wait a moment for the process to start
time.sleep(0.5)

# Read subprocess memory
subprocess_memory = read_memory_from_ps(subprocess_pid)
print(f"Subprocess memory: {subprocess_memory:.2f} MB")

# Clean up
process.terminate()
process.wait()

if subprocess_memory > 0:
    print("\n✅ SUCCESS: Can read subprocess memory using 'ps' command")
    print("This means the Android metrics fix will work!")
else:
    print("\n❌ FAILED: Cannot read subprocess memory using 'ps'")
    print("This is unexpected - 'ps' should work on all Unix-like systems")

# Test 3: Show what ps output looks like
print("\n--- Sample ps output for current process ---")
result = subprocess.run(
    ['ps', '-o', 'pid,rss,vsz,comm', '-p', str(current_pid)],
    capture_output=True,
    text=True
)
print(result.stdout)
