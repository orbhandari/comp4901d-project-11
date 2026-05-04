"""
Hardware detection and abstraction layer.

Provides platform detection, capability discovery, and hardware-specific
optimizations for x86 Linux and Jetson Xavier NX.
"""

from llm_benchmark.hardware.detector import HardwareDetector
from llm_benchmark.hardware.hal import HardwareBackend, X86Backend, JetsonBackend, create_backend

__all__ = [
    'HardwareDetector',
    'HardwareBackend',
    'X86Backend',
    'JetsonBackend',
    'create_backend',
]
