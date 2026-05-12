"""
Configuration management for the benchmark framework.

Handles configuration parsing from JSON/YAML files and command-line arguments,
with validation and default values.
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    
    # Model Configuration
    repo_id: str
    models: Dict[str, str]  # quantization -> filename
    model_cache_dir: str = "./models"
    
    # Test Parameters
    context_size: int = 2048
    batch_size: int = 512
    max_tokens: int = 100
    iterations: int = 3
    warmup_runs: int = 2
    
    # Test Selection
    enable_quantization_profiling: bool = True
    enable_ablation_studies: bool = True
    enable_batch_testing: bool = True
    enable_thermal_monitoring: bool = True
    
    # Ablation Configuration
    kv_cache_types: List[str] = field(default_factory=lambda: ["ram", "disk"])
    prompt_cache_prefix_lengths: List[int] = field(default_factory=lambda: [100, 500, 1000])
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    
    # Orchestration
    sleep_between_tests_s: int = 5
    thermal_stabilization_threshold_c: float = 70.0
    inference_timeout_s: int = 300
    
    # Output
    output_dir: str = "./benchmark_results"
    save_formats: List[str] = field(default_factory=lambda: ["json", "csv", "markdown", "html"])
    visualization_dpi: int = 300
    
    # Authentication
    hf_token: Optional[str] = None
    
    # Android Configuration
    android_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.context_size <= 0:
            raise ValueError("context_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative")
        if self.sleep_between_tests_s < 0:
            raise ValueError("sleep_between_tests_s must be non-negative")
        if self.inference_timeout_s <= 0:
            raise ValueError("inference_timeout_s must be positive")
        if not self.models:
            raise ValueError("models dictionary cannot be empty")
        if not self.repo_id:
            raise ValueError("repo_id cannot be empty")


class ConfigParser:
    """Parse configuration from files and command-line arguments."""
    
    @staticmethod
    def from_file(config_path: str) -> BenchmarkConfig:
        """Load configuration from JSON or YAML file."""
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                if not HAS_YAML:
                    raise ImportError("PyYAML is required for YAML configuration files. Install with: pip install pyyaml")
                config_dict = yaml.safe_load(f)
            elif path.suffix == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {path.suffix}")
        
        # Filter out comment fields (keys starting with underscore)
        config_dict = {k: v for k, v in config_dict.items() if not k.startswith('_')}
        
        return BenchmarkConfig(**config_dict)
    
    @staticmethod
    def from_args(args: argparse.Namespace) -> BenchmarkConfig:
        """Create configuration from command-line arguments."""
        config_dict = {}
        
        # Required arguments
        if hasattr(args, 'repo_id') and args.repo_id:
            config_dict['repo_id'] = args.repo_id
        
        if hasattr(args, 'models') and args.models:
            # Parse models from command line format: "Q4_0:model.gguf,Q8_0:model2.gguf"
            models = {}
            for model_spec in args.models.split(','):
                quant, filename = model_spec.split(':')
                models[quant] = filename
            config_dict['models'] = models
        
        # Optional arguments
        optional_args = [
            'model_cache_dir', 'context_size', 'batch_size', 'max_tokens',
            'iterations', 'warmup_runs',
            'sleep_between_tests_s', 'thermal_stabilization_threshold_c',
            'inference_timeout_s', 'output_dir', 'visualization_dpi', 'hf_token'
        ]
        
        for arg_name in optional_args:
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                config_dict[arg_name] = getattr(args, arg_name)
        
        # Handle boolean flags (disable flags invert the default True values)
        if hasattr(args, 'disable_quantization_profiling') and args.disable_quantization_profiling:
            config_dict['enable_quantization_profiling'] = False
        if hasattr(args, 'disable_ablation_studies') and args.disable_ablation_studies:
            config_dict['enable_ablation_studies'] = False
        if hasattr(args, 'disable_batch_testing') and args.disable_batch_testing:
            config_dict['enable_batch_testing'] = False
        if hasattr(args, 'disable_thermal_monitoring') and args.disable_thermal_monitoring:
            config_dict['enable_thermal_monitoring'] = False
        
        return BenchmarkConfig(**config_dict)
    
    @staticmethod
    def create_argument_parser() -> argparse.ArgumentParser:
        """Create argument parser for command-line interface."""
        epilog = """
Examples:
  # Run with configuration file
  python -m llm_benchmark --config config.json
  
  # Run with command-line arguments
  python -m llm_benchmark --repo-id "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF" \\
    --models "Q4_0:model-Q4_0.gguf,Q8_0:model-Q8_0.gguf" \\
    --iterations 5 --output-dir ./results
  
  # Override config file with command-line arguments
  python -m llm_benchmark --config config.json --iterations 10 --disable-ablation-studies
  
  # Quick test with minimal configuration
  python -m llm_benchmark --config config.json --iterations 1 --disable-batch-testing

For more information, see the documentation at:
https://github.com/yourusername/llm-benchmark
"""
        
        parser = argparse.ArgumentParser(
            description="Comprehensive LLM Inference Benchmarking Framework",
            epilog=epilog,
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Version
        parser.add_argument(
            '--version',
            action='version',
            version='%(prog)s 1.0.0'
        )
        
        # Configuration file
        parser.add_argument(
            '--config', '-c',
            type=str,
            metavar='PATH',
            help='Path to configuration file (JSON or YAML)'
        )
        
        # Model configuration group
        model_group = parser.add_argument_group('Model Configuration')
        model_group.add_argument(
            '--repo-id',
            type=str,
            metavar='REPO',
            help='Hugging Face repository ID (e.g., "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")'
        )
        
        model_group.add_argument(
            '--models',
            type=str,
            metavar='MODELS',
            help='Models to test in format "QUANT:FILENAME,QUANT:FILENAME" (e.g., "Q4_0:model.gguf,Q8_0:model2.gguf")'
        )
        
        model_group.add_argument(
            '--model-cache-dir',
            type=str,
            metavar='DIR',
            help='Directory for caching downloaded models (default: ./models)'
        )
        
        # Test parameters group
        test_group = parser.add_argument_group('Test Parameters')
        test_group.add_argument(
            '--context-size',
            type=int,
            metavar='SIZE',
            help='Context size for model (default: 2048)'
        )
        
        test_group.add_argument(
            '--batch-size',
            type=int,
            metavar='SIZE',
            help='Batch size for inference (default: 512)'
        )
        
        test_group.add_argument(
            '--max-tokens',
            type=int,
            metavar='N',
            help='Maximum tokens to generate (default: 100)'
        )
        
        test_group.add_argument(
            '--iterations',
            type=int,
            metavar='N',
            help='Number of benchmark iterations (default: 3)'
        )
        
        test_group.add_argument(
            '--warmup-runs',
            type=int,
            metavar='N',
            help='Number of warmup runs before measurement (default: 2)'
        )
        
        # Test selection group (use action flags for boolean arguments)
        selection_group = parser.add_argument_group('Test Selection')
        selection_group.add_argument(
            '--disable-quantization-profiling',
            action='store_true',
            dest='disable_quantization_profiling',
            help='Disable quantization profiling tests (enabled by default)'
        )
        
        selection_group.add_argument(
            '--disable-ablation-studies',
            action='store_true',
            dest='disable_ablation_studies',
            help='Disable ablation studies (enabled by default)'
        )
        
        selection_group.add_argument(
            '--disable-batch-testing',
            action='store_true',
            dest='disable_batch_testing',
            help='Disable batch processing tests (enabled by default)'
        )
        
        selection_group.add_argument(
            '--disable-thermal-monitoring',
            action='store_true',
            dest='disable_thermal_monitoring',
            help='Disable thermal monitoring (enabled by default)'
        )
        
        # Orchestration group
        orchestration_group = parser.add_argument_group('Orchestration')
        orchestration_group.add_argument(
            '--sleep-between-tests',
            type=int,
            metavar='SECONDS',
            dest='sleep_between_tests_s',
            help='Sleep duration between tests in seconds (default: 5)'
        )
        
        orchestration_group.add_argument(
            '--thermal-threshold',
            type=float,
            metavar='CELSIUS',
            dest='thermal_stabilization_threshold_c',
            help='Thermal stabilization threshold in Celsius (default: 70.0)'
        )
        
        orchestration_group.add_argument(
            '--inference-timeout',
            type=int,
            metavar='SECONDS',
            dest='inference_timeout_s',
            help='Inference timeout in seconds (default: 300)'
        )
        
        # Output group
        output_group = parser.add_argument_group('Output')
        output_group.add_argument(
            '--output-dir',
            type=str,
            metavar='DIR',
            help='Output directory for results (default: ./benchmark_results)'
        )
        
        output_group.add_argument(
            '--visualization-dpi',
            type=int,
            metavar='DPI',
            help='DPI for visualization images (default: 300)'
        )
        
        # Authentication group
        auth_group = parser.add_argument_group('Authentication')
        auth_group.add_argument(
            '--hf-token',
            type=str,
            metavar='TOKEN',
            help='Hugging Face API token (can also use HF_TOKEN environment variable)'
        )
        
        return parser
    
    @staticmethod
    def load_config(args: Optional[argparse.Namespace] = None) -> BenchmarkConfig:
        """
        Load configuration from file and/or command-line arguments.
        
        Priority: command-line arguments > configuration file > defaults
        """
        file_config = None
        
        # Load from file if specified
        if args and hasattr(args, 'config') and args.config:
            file_config = ConfigParser.from_file(args.config)
        
        # If we have a file config, start with it and apply overrides
        if file_config:
            config_dict = file_config.__dict__.copy()
            
            # Apply command-line overrides
            if args:
                # Model configuration overrides
                if hasattr(args, 'repo_id') and args.repo_id:
                    config_dict['repo_id'] = args.repo_id
                
                if hasattr(args, 'models') and args.models:
                    models = {}
                    for model_spec in args.models.split(','):
                        quant, filename = model_spec.split(':')
                        models[quant] = filename
                    config_dict['models'] = models
                
                # Optional parameter overrides
                optional_args = [
                    'model_cache_dir', 'context_size', 'batch_size', 'max_tokens',
                    'iterations', 'warmup_runs',
                    'sleep_between_tests_s', 'thermal_stabilization_threshold_c',
                    'inference_timeout_s', 'output_dir', 'visualization_dpi', 'hf_token'
                ]
                
                for arg_name in optional_args:
                    if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                        config_dict[arg_name] = getattr(args, arg_name)
                
                # Boolean flag overrides
                if hasattr(args, 'disable_quantization_profiling') and args.disable_quantization_profiling:
                    config_dict['enable_quantization_profiling'] = False
                if hasattr(args, 'disable_ablation_studies') and args.disable_ablation_studies:
                    config_dict['enable_ablation_studies'] = False
                if hasattr(args, 'disable_batch_testing') and args.disable_batch_testing:
                    config_dict['enable_batch_testing'] = False
                if hasattr(args, 'disable_thermal_monitoring') and args.disable_thermal_monitoring:
                    config_dict['enable_thermal_monitoring'] = False
            
            return BenchmarkConfig(**config_dict)
        
        # No file config, try to create from command-line arguments only
        if args and hasattr(args, 'repo_id') and args.repo_id:
            return ConfigParser.from_args(args)
        
        # No valid configuration source
        raise ValueError(
            "Configuration must be provided via --config file or "
            "--repo-id and --models arguments"
        )
