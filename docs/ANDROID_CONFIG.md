# AndroidConfig Data Model

The `AndroidConfig` data model provides configuration schema for Android llama-server ablation support, enabling accurate cache ablation studies on Android devices.

## Overview

The AndroidConfig enables automatic backend selection between llama-cli and llama-server based on ablation configuration and binary availability. When `enable_ablation_studies: true` is configured for Android, the system automatically uses llama-server if available, falling back to llama-cli with appropriate warnings.

## Key Features

- **True cache ablation**: Complete control over RAM and disk caching for control runs
- **Per-request cache control**: Fine-grained control over prompt caching without server restarts  
- **Backward compatibility**: Existing llama-cli workflows remain unchanged
- **Automatic selection**: Intelligent backend selection based on configuration and availability

## Configuration Schema

```python
@dataclass
class AndroidConfig:
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
```

## Cache Modes

The `CacheMode` enum defines cache control strategies:

- `NONE`: `--cache-ram 0 --no-cache-prompt` (true no-cache baseline)
- `RAM_ONLY`: `--no-cache-prompt` (KV cache only)
- `DISK_ONLY`: `--cache-ram 0` (prompt cache only)
- `BOTH`: No cache restrictions (both KV and prompt cache)

## Backend Selection Logic

The backend selection follows this priority order:

1. **Explicit Configuration**: If `use_llama_server_for_ablation` is set, use that value
2. **Ablation Mode**: If `enable_ablation_studies` is true and platform is Android, prefer llama-server
3. **Availability Check**: Verify llama-server binary exists at expected path
4. **Fallback**: Use llama-cli if llama-server is not available

## Usage Examples

### Basic Configuration

```python
from llm_benchmark.android_config import AndroidConfig, CacheMode

# Create default configuration
config = AndroidConfig()

# Enable ablation studies
config = AndroidConfig(enable_ablation_studies=True)

# Explicit llama-server selection
config = AndroidConfig(use_llama_server_for_ablation=True)
```

### Ablation Scenario Configuration

```python
from llm_benchmark.android_config import ABLATION_CACHE_CONFIG

# Configure for control scenario (no cache)
control_config = ABLATION_CACHE_CONFIG["control"]
config = AndroidConfig(
    cache_mode=control_config["cache_mode"],  # CacheMode.NONE
    enable_prompt_cache_by_default=control_config["enable_prompt_cache"]  # False
)

# Configure for warm cache scenario
warm_config = ABLATION_CACHE_CONFIG["warm_cache"]
config = AndroidConfig(
    cache_mode=warm_config["cache_mode"],  # CacheMode.BOTH
    enable_prompt_cache_by_default=warm_config["enable_prompt_cache"]  # True
)
```

### JSON Configuration

```json
{
  "android_config": {
    "enable_ablation_studies": true,
    "use_llama_server_for_ablation": null,
    "llama_server_host": "127.0.0.1",
    "llama_server_port": 8080,
    "llama_server_path": null,
    "llama_server_timeout": 900,
    "cache_mode": "both",
    "enable_prompt_cache_by_default": false
  }
}
```

## Validation and Warnings

The AndroidConfig performs comprehensive validation:

- **Network Configuration**: Validates host and port parameters
- **Timeout Configuration**: Ensures positive timeout values with warnings for extreme values
- **Cache Configuration**: Validates cache mode and prompt cache settings
- **Ablation Configuration**: Validates ablation study parameters

## Utility Methods

### URL Generation

```python
config = AndroidConfig(llama_server_host="192.168.1.100", llama_server_port=9090)

config.get_server_url()      # "http://192.168.1.100:9090"
config.get_health_url()      # "http://192.168.1.100:9090/health"
config.get_completion_url()  # "http://192.168.1.100:9090/completion"
```

### Backend Selection

```python
config = AndroidConfig(enable_ablation_studies=True)

config.should_use_llama_server("android")  # True
config.should_use_llama_server("linux")    # False
```

### Cache Control

```python
config = AndroidConfig(cache_mode=CacheMode.NONE)

config.get_cache_flags()  # ["--cache-ram", "0", "--no-cache-prompt"]
config.get_request_cache_setting(True)   # True (override)
config.get_request_cache_setting()       # False (default)
```

### Serialization

```python
# Convert to dictionary
config_dict = config.to_dict()

# Create from dictionary
config = AndroidConfig.from_dict(config_dict)
```

## Ablation Scenario Mapping

The `ABLATION_CACHE_CONFIG` provides predefined configurations for ablation scenarios:

```python
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
```

## Requirements Validation

The AndroidConfig satisfies the following requirements:

- **Requirement 8.1**: Support for `use_llama_server_for_ablation` configuration option
- **Requirement 8.2**: Explicit backend selection override capability
- **Requirement 8.3**: Default to automatic selection based on ablation settings
- **Requirement 8.4**: Default to true on Android when ablation is enabled
- **Requirement 8.5**: Support for `llama_server_port` configuration with default 8080
- **Requirement 8.6**: Support for `llama_server_host` configuration with default "127.0.0.1"

## See Also

- [Android Setup Guide](ANDROID_SETUP.md)
- [Ablation Studies Documentation](ABLATION_CACHE_PREFIX.md)
- [Configuration Examples](../configs/android_llama_server_example.json)