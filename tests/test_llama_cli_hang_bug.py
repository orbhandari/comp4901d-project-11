"""
Bug Condition Exploration Test for llama-cli Hang Issue

This test MUST FAIL on unfixed code - failure confirms the bug exists.
DO NOT attempt to fix the test or the code when it fails.

The test encodes the expected behavior and will validate the fix when it passes.
"""

import os
import signal
import subprocess
import time
import pytest
from pathlib import Path


class TestLlamaCliHangBug:
    """
    Property 1: Bug Condition - llama-cli Hang Detection
    
    CRITICAL: This test MUST FAIL on unfixed code.
    GOAL: Surface counterexamples that demonstrate the bug exists.
    """
    
    @pytest.fixture
    def llama_cli_path(self):
        """Get path to llama-cli binary."""
        path = Path("~/llama.cpp/build/bin/llama-cli").expanduser()
        if not path.exists():
            pytest.skip("llama-cli binary not found")
        return path
    
    @pytest.fixture
    def model_path(self):
        """Get path to test model."""
        path = Path("~/storage/shared/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf").expanduser()
        if not path.exists():
            pytest.skip("Test model not found")
        return path
    
    def test_llama_cli_exits_cleanly_within_timeout(self, llama_cli_path, model_path):
        """
        Test that llama-cli generates tokens and exits cleanly without hanging.
        
        EXPECTED ON UNFIXED CODE: FAIL (process hangs, timeout expires)
        EXPECTED ON FIXED CODE: PASS (process exits cleanly)
        
        Bug Condition: llama-cli enters conversation mode and hangs indefinitely
        Expected Behavior: Process generates tokens and exits within timeout
        """
        # Short timeout to make test fail quickly on unfixed code
        timeout_seconds = 10
        
        cmd = [
            str(llama_cli_path),
            "-m", str(model_path),
            "-n", "10",  # Generate only 10 tokens
            "-p", "Hello",
        ]
        
        start_time = time.time()
        
        try:
            # Run with timeout
            result = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds
            )
            
            elapsed = time.time() - start_time
            
            # Verify process exited cleanly
            assert result.returncode == 0, \
                f"Process exited with non-zero code: {result.returncode}"
            
            # Verify it completed within reasonable time
            assert elapsed < timeout_seconds, \
                f"Process took too long: {elapsed:.2f}s (timeout: {timeout_seconds}s)"
            
            # Verify output doesn't contain conversation mode markers
            assert ">" not in result.stdout, \
                f"Output contains conversation mode marker '>': {result.stdout[:200]}"
            
            # Verify some output was generated
            assert len(result.stdout.strip()) > 0, \
                "No output generated"
            
        except subprocess.TimeoutExpired as e:
            # This is the EXPECTED failure on unfixed code
            pytest.fail(
                f"BUG CONFIRMED: Process hung and timed out after {timeout_seconds}s. "
                f"This confirms llama-cli enters conversation mode. "
                f"stdout: {e.stdout[:200] if e.stdout else 'None'}"
            )
    
    def test_llama_cli_does_not_print_conversation_markers(self, llama_cli_path, model_path):
        """
        Test that llama-cli doesn't print '>' conversation mode markers.
        
        EXPECTED ON UNFIXED CODE: FAIL (stdout contains '>')
        EXPECTED ON FIXED CODE: PASS (no '>' in output)
        
        Bug Condition: llama-cli prints infinite '>' characters
        Expected Behavior: Output contains only generated text, no prompts
        """
        timeout_seconds = 10
        
        cmd = [
            str(llama_cli_path),
            "-m", str(model_path),
            "-n", "10",
            "-p", "Hello",
        ]
        
        try:
            result = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds
            )
            
            # Check for conversation mode markers
            if ">" in result.stdout:
                pytest.fail(
                    f"BUG CONFIRMED: Output contains conversation mode marker '>'. "
                    f"stdout: {result.stdout[:200]}"
                )
            
        except subprocess.TimeoutExpired as e:
            # Also a bug - process hung
            pytest.fail(
                f"BUG CONFIRMED: Process hung (timeout). "
                f"stdout: {e.stdout[:200] if e.stdout else 'None'}"
            )
    
    def test_llama_cli_responds_to_sigterm(self, llama_cli_path, model_path):
        """
        Test that llama-cli responds to SIGTERM signal.
        
        EXPECTED ON UNFIXED CODE: FAIL (process ignores SIGTERM)
        EXPECTED ON FIXED CODE: PASS (process terminates on SIGTERM)
        
        Bug Condition: Process ignores termination signals in conversation mode
        Expected Behavior: Process terminates when SIGTERM is sent
        """
        cmd = [
            str(llama_cli_path),
            "-m", str(model_path),
            "-n", "10",
            "-p", "Hello",
        ]
        
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give it a moment to start
        time.sleep(2)
        
        # Send SIGTERM
        process.send_signal(signal.SIGTERM)
        
        # Wait for it to terminate (with timeout)
        try:
            process.wait(timeout=5)
            # Good - process terminated
        except subprocess.TimeoutExpired:
            # Process didn't terminate - this is the bug
            process.kill()  # Force kill for cleanup
            process.wait()
            pytest.fail(
                "BUG CONFIRMED: Process ignored SIGTERM signal. "
                "This indicates conversation mode is blocking signal handling."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
