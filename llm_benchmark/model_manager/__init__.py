"""
Model management module.

Handles model acquisition from Hugging Face Hub, local caching,
integrity verification, and GGUF format validation.
"""

from llm_benchmark.model_manager.manager import ModelAcquisitionError, ModelManager

__all__ = ["ModelManager", "ModelAcquisitionError"]
