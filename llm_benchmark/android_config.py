"""
Android Configuration Data Model for llama-server Ablation Support.

This module defines the AndroidConfig data model that provides configuration
schema for llama-server options, validation for configuration parameters,
and appropriate default values for Android ablation studies.

The AndroidConfig enables automatic backend selection between llama-cli and
llama-server based on ablation configuration and binary availability.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class CacheMode(Enum):
    """Cache mode configuration for llama-server."""
    
    NONE = "none"          # --cache-ram 0 --no-cache-prompt (true no-cache baseline)
    RAM_ONLY = "ram_only"  # --no-cache-prompt (KV cache only)
    DISK_ONLY = "disk_only" # --cache-ram 0 (prompt cache only)
    BOTH = "both"          # No cache restrictions (both KV and prompt cache)


@dataclass
class AndroidConfig:
    """
    Configuration schema for Android llama-server ablation support.
    
    This configuration enables automatic backend selection and provides
    all necessary options for llama-server integration with proper cache
    control for accurate ablation studies.
    
    Attributes:
        enable_ablation_studies: Enable ablation studies mode
        use_llama_server_for_ablation: Explicit backend selection override
        llama_server_host: Host address for llama-server HTTP API
        llama_server_port: Port number for llama-server HTTP API
        llama_server_path: Custom path to llama-server binary
        llama_server_timeout: Request timeout for HTTP API calls in seconds
        cache_mode: Default cache mode for llama-server initialization
        enable_prompt_cache_by_default: Default prompt cache setting for requests
    """
    
    # Core ablation configuration
    enable_ablation_studies: bool = False
    use_llama_server_for_ablation: Optional[bool] = None
    
    # llama-server HTTP API configuration
    llama_server_host: str = "127.0.0.1"
    llama_server_port: int = 8080
    llama_server_path: Optional[str] = None
    llama_server_timeout: int = 600
    
    # Cache control configuration
    cache_mode: CacheMode = CacheMode.BOTH
    enable_prompt_cache_by_default: bool = False
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate_network_config()
        self._validate_timeout_config()
        self._validate_cache_config()
        self._validate_ablation_config()
    
    def _validate_network_config(self) -> None:
        """Validate network-related configuration parameters."""
        # Validate host address
        if not self.llama_server_host:
            raise ValueError("llama_server_host cannot be empty")
        
        if not isinstance(self.llama_server_host, str):
            raise ValueError("llama_server_host must be a string")
        
        # Validate port number
        if not isinstance(self.llama_server_port, int):
            raise ValueError("llama_server_port must be an integer")
        
        if not (1 <= self.llama_server_port <= 65535):
            raise ValueError(
                f"llama_server_port must be between 1 and 65535, got {self.llama_server_port}"
            )
        
        # Warn about common port conflicts
        if self.llama_server_port in [80, 443, 22, 21, 25]:
            logger.warning(
                f"Port {self.llama_server_port} is commonly used by system services. "
                "Consider using a different port (e.g., 8080, 8081) to avoid conflicts."
            )
    
    def _validate_timeout_config(self) -> None:
        """Validate timeout configuration parameters."""
        if not isinstance(self.llama_server_timeout, int):
            raise ValueError("llama_server_timeout must be an integer")
        
        if self.llama_server_timeout <= 0:
            raise ValueError(
                f"llama_server_timeout must be positive, got {self.llama_server_timeout}"
            )
        
        # Warn about very short timeouts
        if self.llama_server_timeout < 60:
            logger.warning(
                f"llama_server_timeout ({self.llama_server_timeout}s) is very short. "
                "Consider using at least 60 seconds for reliable inference."
            )
        
        # Warn about very long timeouts
        if self.llama_server_timeout > 3600:
            logger.warning(
                f"llama_server_timeout ({self.llama_server_timeout}s) is very long. "
                "Consider using a shorter timeout to detect hanging processes."
            )
    
    def _validate_cache_config(self) -> None:
        """Validate cache-related configuration parameters."""
        # Validate cache_mode is a valid enum value
        if not isinstance(self.cache_mode, CacheMode):
            if isinstance(self.cache_mode, str):
                # Try to convert string to enum
                try:
                    self.cache_mode = CacheMode(self.cache_mode)
                except ValueError:
                    valid_modes = [mode.value for mode in CacheMode]
                    raise ValueError(
                        f"Invalid cache_mode '{self.cache_mode}'. "
                        f"Valid options: {valid_modes}"
                    )
            else:
                raise ValueError(
                    f"cache_mode must be a CacheMode enum or string, "
                    f"got {type(self.cache_mode)}"
                )
        
        # Validate enable_prompt_cache_by_default
        if not isinstance(self.enable_prompt_cache_by_default, bool):
            raise ValueError("enable_prompt_cache_by_default must be a boolean")
    
    def _validate_ablation_config(self) -> None:
        """Validate ablation study configuration parameters."""
        # Validate enable_ablation_studies
        if not isinstance(self.enable_ablation_studies, bool):
            raise ValueError("enable_ablation_studies must be a boolean")
        
        # Validate use_llama_server_for_ablation
        if self.use_llama_server_for_ablation is not None:
            if not isinstance(self.use_llama_server_for_ablation, bool):
                raise ValueError("use_llama_server_for_ablation must be a boolean or None")
        
        # Validate llama_server_path if provided
        if self.llama_server_path is not None:
            if not isinstance(self.llama_server_path, str):
                raise ValueError("llama_server_path must be a string or None")
            
            if not self.llama_server_path.strip():
                raise ValueError("llama_server_path cannot be empty string")
    
    def get_server_url(self) -> str:
        """
        Get the complete server URL for HTTP API requests.
        
        Returns:
            Complete URL string for llama-server HTTP API
        """
        return f"http://{self.llama_server_host}:{self.llama_server_port}"
    
    def get_health_url(self) -> str:
        """
        Get the health check endpoint URL.
        
        Returns:
            Complete URL string for health check endpoint
        """
        return f"{self.get_server_url()}/health"
    
    def get_completion_url(self) -> str:
        """
        Get the completion endpoint URL.
        
        Returns:
            Complete URL string for completion endpoint
        """
        return f"{self.get_server_url()}/completion"
    
    def should_use_llama_server(self, platform: str) -> bool:
        """
        Determine whether to use llama-server based on configuration and platform.
        
        Implements the backend selection logic from the requirements:
        1. Explicit configuration overrides automatic selection
        2. Ablation mode prefers llama-server for Android platform
        3. Fallback to llama-cli if llama-server not available
        
        Args:
            platform: Platform identifier (e.g., "android", "linux", "macos")
        
        Returns:
            True if llama-server should be used, False for llama-cli
        """
        # Explicit configuration takes precedence
        if self.use_llama_server_for_ablation is not None:
            logger.debug(
                f"Using explicit configuration: use_llama_server_for_ablation="
                f"{self.use_llama_server_for_ablation}"
            )
            return self.use_llama_server_for_ablation
        
        # Automatic selection based on ablation studies and platform
        if self.enable_ablation_studies and platform == "android":
            logger.debug(
                "Ablation studies enabled on Android platform, preferring llama-server"
            )
            return True
        
        # Default to llama-cli for other cases
        logger.debug(
            f"Using llama-cli (ablation_studies={self.enable_ablation_studies}, "
            f"platform={platform})"
        )
        return False
    
    def get_cache_flags(self) -> list[str]:
        """
        Get command-line cache control flags for llama-server based on cache_mode.
        
        Returns:
            List of command-line flags for cache control
        """
        flags = []
        
        if self.cache_mode == CacheMode.NONE:
            flags.extend(["--cache-ram", "0", "--no-cache-prompt"])
        elif self.cache_mode == CacheMode.RAM_ONLY:
            flags.append("--no-cache-prompt")
        elif self.cache_mode == CacheMode.DISK_ONLY:
            flags.extend(["--cache-ram", "0"])
        # CacheMode.BOTH uses no cache restriction flags
        
        return flags
    
    def get_request_cache_setting(self, enable_prompt_cache: Optional[bool] = None) -> bool:
        """
        Get the cache_prompt setting for HTTP API requests.
        
        Args:
            enable_prompt_cache: Override for specific request, None uses default
        
        Returns:
            Boolean value for cache_prompt parameter in API requests
        """
        if enable_prompt_cache is not None:
            return enable_prompt_cache
        
        return self.enable_prompt_cache_by_default
    
    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary for serialization.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "enable_ablation_studies": self.enable_ablation_studies,
            "use_llama_server_for_ablation": self.use_llama_server_for_ablation,
            "llama_server_host": self.llama_server_host,
            "llama_server_port": self.llama_server_port,
            "llama_server_path": self.llama_server_path,
            "llama_server_timeout": self.llama_server_timeout,
            "cache_mode": self.cache_mode.value,
            "enable_prompt_cache_by_default": self.enable_prompt_cache_by_default,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "AndroidConfig":
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
        
        Returns:
            AndroidConfig instance
        """
        # Handle cache_mode conversion from string
        if "cache_mode" in config_dict and isinstance(config_dict["cache_mode"], str):
            config_dict = config_dict.copy()
            config_dict["cache_mode"] = CacheMode(config_dict["cache_mode"])
        
        return cls(**config_dict)


# Ablation scenario cache configuration mapping
ABLATION_CACHE_CONFIG = {
    "control": {
        "cache_mode": CacheMode.NONE,
        "enable_prompt_cache": False,
        "description": "True no-cache baseline with both RAM and disk caching disabled"
    },
    "cold_cache": {
        "cache_mode": CacheMode.BOTH,
        "enable_prompt_cache": True,
        "description": "Cache enabled but empty (first request creates cache)"
    },
    "warm_cache": {
        "cache_mode": CacheMode.BOTH,
        "enable_prompt_cache": True,
        "description": "Cache enabled and populated (subsequent requests reuse cache)"
    },
    "ram_only": {
        "cache_mode": CacheMode.RAM_ONLY,
        "enable_prompt_cache": False,
        "description": "KV cache only (RAM), no disk-based prompt cache"
    },
    "disk_only": {
        "cache_mode": CacheMode.DISK_ONLY,
        "enable_prompt_cache": True,
        "description": "Prompt cache only (disk), no RAM-based KV cache"
    }
}


def create_default_android_config() -> AndroidConfig:
    """
    Create AndroidConfig with sensible defaults for Android ablation studies.
    
    Returns:
        AndroidConfig instance with default values optimized for Android
    """
    return AndroidConfig(
        enable_ablation_studies=False,
        use_llama_server_for_ablation=None,  # Auto-detect based on ablation + platform
        llama_server_host="127.0.0.1",
        llama_server_port=8080,
        llama_server_path=None,  # Auto-detect at ~/llama.cpp/build/bin/llama-server
        llama_server_timeout=600,  # 10 minutes for mobile inference
        cache_mode=CacheMode.BOTH,
        enable_prompt_cache_by_default=False,  # Conservative default
    )


def validate_android_config(config: AndroidConfig) -> list[str]:
    """
    Validate AndroidConfig and return list of warnings/recommendations.
    
    Args:
        config: AndroidConfig instance to validate
    
    Returns:
        List of warning/recommendation messages
    """
    warnings = []
    
    # Check for potential performance issues
    if config.enable_ablation_studies and config.llama_server_timeout < 300:
        warnings.append(
            f"llama_server_timeout ({config.llama_server_timeout}s) may be too short "
            "for mobile inference during ablation studies. Consider 300+ seconds."
        )
    
    # Check for cache configuration recommendations
    if config.enable_ablation_studies and config.cache_mode != CacheMode.BOTH:
        warnings.append(
            f"cache_mode is set to {config.cache_mode.value} but ablation studies "
            "typically need dynamic cache control. Consider using CacheMode.BOTH "
            "and controlling cache per-request."
        )
    
    # Check for network configuration
    if config.llama_server_host != "127.0.0.1" and config.llama_server_host != "localhost":
        warnings.append(
            f"llama_server_host is set to {config.llama_server_host}. "
            "For security, consider using localhost (127.0.0.1) unless "
            "remote access is specifically needed."
        )
    
    return warnings