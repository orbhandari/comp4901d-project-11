"""
Logging infrastructure for the benchmark framework.

Provides file and console handlers with appropriate formatting and log levels.
"""

import logging
import os
from pathlib import Path
from typing import Optional


def setup_logging(output_dir: str, log_level: int = logging.INFO) -> None:
    """
    Configure logging with file and console handlers.
    
    Args:
        output_dir: Directory for log files
        log_level: Minimum log level (default: INFO)
    """
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Main log: INFO and above
    main_handler = logging.FileHandler(log_dir / "benchmark.log")
    main_handler.setLevel(logging.INFO)
    main_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # Error log: ERROR and above
    error_handler = logging.FileHandler(log_dir / "errors.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
        'File: %(pathname)s:%(lineno)d\n'
        'Function: %(funcName)s\n',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # Console: INFO and above (show progress in terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '[%(levelname)s] %(message)s'
    ))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Add handlers
    root_logger.addHandler(main_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)
    
    # Log initial message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log directory: {log_dir}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
