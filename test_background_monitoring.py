#!/usr/bin/env python3
"""
Test that background memory monitoring works correctly.
"""

import subprocess
import threading
import time

def test_background_monitoring():
    """Test background memory monitoring of a subprocess."""
    
    # Start a subprocess that allocates some memory
    process = subprocess.Popen(
        ["python", "-c", "import time; x = ' ' * (10 * 1024 * 1024); time.sleep(2)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    pid = process.pid
    peak_memory_kb = 0
    stop_monitoring = threading.Event()
    
    def monitor_memory():
        """Monitor subprocess memory in background."""
        nonlocal peak_memory_kb
        while not stop_monitoring.is_set():
            try:
                result = subprocess.run(
                    ['ps', '-o', 'rss=', '-p', str(pid)],
                    capture_output=True,
                    text=True,
                    timeout=0.1
                )
                if result.returncode == 0 and result.stdout.strip():
                    rss_kb = int(result.stdout.strip())
                    peak_memory_kb = max(peak_memory_kb, rss_kb)
                    print(f"Sampled: {rss_kb / 1024:.2f} MB, Peak: {peak_memory_kb / 1024:.2f} MB")
            except:
                pass
            time.sleep(0.05)  # Sample every 50ms
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
    monitor_thread.start()
    
    # Wait for subprocess to complete
    process.wait()
    
    # Stop monitoring
    stop_monitoring.set()
    monitor_thread.join(timeout=1)
    
    print(f"\n✅ Peak memory captured: {peak_memory_kb / 1024:.2f} MB")
    
    if peak_memory_kb > 0:
        print("SUCCESS: Background monitoring works!")
        return True
    else:
        print("FAILED: No memory was captured")
        return False

if __name__ == "__main__":
    test_background_monitoring()
