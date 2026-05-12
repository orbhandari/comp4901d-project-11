"""
Results persistence and reporting module.

Handles serialization of benchmark results to multiple formats (JSON, CSV, Markdown)
and manages organized directory structure for result storage.
"""

from llm_benchmark.results.persistence import ResultsPersistence

__all__ = ["ResultsPersistence"]
