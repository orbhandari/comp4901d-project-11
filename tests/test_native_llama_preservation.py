"""
Preservation Property Tests for Native llama.cpp Wrapper

These tests capture the baseline behavior that MUST be preserved after the fix.
Tests should PASS on unfixed code for successful executions.

Property 2: Preservation - Streaming and TTFT Measurement
"""

import time
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from llm_benchmark.inference.native_llama import NativeLlamaCpp


class TestNativeLlamaPreservation:
    """
    Property 2: Preservation - Streaming and TTFT Measurement
    
    IMPORTANT: These tests capture baseline behavior to preserve.
    They should PASS on unfixed code for successful executions.
    """
    
    @pytest.fixture
    def model_path(self):
        """Get path to test model."""
        path = Path("~/storage/shared/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf").expanduser()
        if not path.exists():
            pytest.skip("Test model not found")
        return str(path)
    
    def test_streaming_output_format_preserved(self, model_path):
        """
        Test that character-by-character streaming format is preserved.
        
        Preservation Requirement 3.1: Generated text format must remain unchanged
        Preservation Requirement 3.6: Streaming must work for TTFT measurement
        """
        # This test verifies the output format structure
        # It should pass on both unfixed and fixed code
        
        # Mock successful execution to test format
        with patch('subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.communicate.return_value = ("Hello world", "")
            mock_process.returncode = 0
            mock_popen.return_value = mock_process
            
            llm = NativeLlamaCpp(model_path, n_ctx=512, n_threads=4)
            
            # Generate text
            chunks = list(llm("Test prompt", max_tokens=10))
            
            # Verify streaming format
            assert len(chunks) > 0, "Should yield at least one chunk"
            
            # Verify each chunk has correct structure
            for chunk in chunks[:-1]:  # All except last
                assert 'choices' in chunk
                assert len(chunk['choices']) > 0
                assert 'text' in chunk['choices'][0]
                assert 'finish_reason' in chunk['choices'][0]
                assert chunk['choices'][0]['finish_reason'] is None
            
            # Verify last chunk has finish_reason
            last_chunk = chunks[-1]
            assert last_chunk['choices'][0]['finish_reason'] == 'stop'
    
    def test_configuration_parameters_preserved(self, model_path):
        """
        Test that configuration parameters (n_ctx, n_threads, n_batch) are preserved.
        
        Preservation Requirement 3.2: Configuration parameters must remain unchanged
        """
        n_ctx = 512
        n_threads = 6
        n_batch = 256
        
        llm = NativeLlamaCpp(
            model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch
        )
        
        # Verify configuration is stored
        assert llm.n_ctx == n_ctx
        assert llm.n_threads == n_threads
        assert llm.n_batch == n_batch
        assert llm.model_path == Path(model_path)
    
    def test_tokenization_approximation_preserved(self, model_path):
        """
        Test that tokenization approximation (1 token ≈ 4 characters) is preserved.
        
        Preservation Requirement 3.4: Tokenization approximation must remain unchanged
        """
        llm = NativeLlamaCpp(model_path)
        
        # Test various text lengths
        test_cases = [
            (b"Hello", 1),  # 5 chars -> 1 token (min)
            (b"Hello world", 2),  # 11 chars -> 2 tokens
            (b"This is a longer test string", 7),  # 29 chars -> 7 tokens
        ]
        
        for text, expected_tokens in test_cases:
            tokens = llm.tokenize(text)
            assert len(tokens) == expected_tokens, \
                f"Expected {expected_tokens} tokens for {len(text)} chars, got {len(tokens)}"
    
    def test_binary_selection_logic_preserved(self, model_path):
        """
        Test that binary selection logic works correctly.
        
        Preservation Requirement: Binary selection should try alternatives
        """
        llm = NativeLlamaCpp(model_path)
        
        # Verify binary was selected
        assert llm.llama_cli_path is not None
        assert llm.llama_cli_path.exists()
        assert llm.binary_type in ["llama-cli", "main", "llama-simple"]
    
    def test_error_handling_preserved(self, model_path):
        """
        Test that error handling behavior is preserved.
        
        Preservation Requirement 3.5: Error logging and exceptions must remain unchanged
        """
        # Test with non-existent model
        with pytest.raises(FileNotFoundError) as exc_info:
            NativeLlamaCpp("/nonexistent/model.gguf")
        
        assert "Model not found" in str(exc_info.value)
    
    def test_create_completion_interface_preserved(self, model_path):
        """
        Test that create_completion interface is preserved.
        
        Preservation Requirement: Alternative interface must continue to work
        """
        with patch('subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.communicate.return_value = ("Test output", "")
            mock_process.returncode = 0
            mock_popen.return_value = mock_process
            
            llm = NativeLlamaCpp(model_path)
            
            # Test create_completion interface
            chunks = list(llm.create_completion("Test", max_tokens=10))
            
            # Verify it returns same format as __call__
            assert len(chunks) > 0
            assert 'choices' in chunks[0]


class TestNativeLlamaPreservationPropertyBased:
    """
    Property-based tests for stronger preservation guarantees.
    
    These generate many test cases automatically to catch edge cases.
    """
    
    @pytest.fixture
    def model_path(self):
        """Get path to test model."""
        path = Path("~/storage/shared/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf").expanduser()
        if not path.exists():
            pytest.skip("Test model not found")
        return str(path)
    
    @pytest.mark.parametrize("n_ctx", [256, 512, 1024, 2048])
    @pytest.mark.parametrize("n_threads", [2, 4, 6, 8])
    def test_configuration_combinations_preserved(self, model_path, n_ctx, n_threads):
        """
        Test that various configuration combinations work correctly.
        
        Property: For all valid configurations, initialization succeeds
        """
        llm = NativeLlamaCpp(
            model_path,
            n_ctx=n_ctx,
            n_threads=n_threads
        )
        
        assert llm.n_ctx == n_ctx
        assert llm.n_threads == n_threads
    
    @pytest.mark.parametrize("text_length", [4, 8, 16, 32, 64, 128])
    def test_tokenization_scales_correctly(self, model_path, text_length):
        """
        Test that tokenization approximation scales correctly with text length.
        
        Property: For all text lengths, token count ≈ length / 4
        """
        llm = NativeLlamaCpp(model_path)
        
        text = b"a" * text_length
        tokens = llm.tokenize(text)
        expected = max(1, text_length // 4)
        
        assert len(tokens) == expected, \
            f"Expected {expected} tokens for {text_length} chars, got {len(tokens)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
