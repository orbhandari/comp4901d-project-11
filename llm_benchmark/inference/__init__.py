"""
Inference module.

Provides inference backends for different platforms.
"""

from llm_benchmark.inference.native_llama import NativeLlamaCpp

__all__ = ['NativeLlamaCpp']
