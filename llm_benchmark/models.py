"""
Data models for the benchmark framework.

Defines dataclasses for hardware information, metrics, results, and other
data structures used throughout the framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class HardwareInfo:
    """Hardware platform information."""
    
    os_type: str  # "linux_x86" | "jetson_xavier_nx"
    cpu_model: str
    cpu_cores: int
    cpu_features: List[str]  # ["avx2", "avx512", ...]
    total_ram_gb: float
    available_ram_gb: float
    has_gpu: bool
    gpu_model: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    gpu_compute_capability: Optional[str] = None
    has_thermal_sensors: bool = False
    has_power_sensors: bool = False


@dataclass
class ModelInfo:
    """Model file information."""
    
    quantization: str
    filename: str
    local_path: str
    sha256: str
    size_mb: float


@dataclass
class InferenceMetrics:
    """Metrics collected during inference."""
    
    ttft_ms: float
    prefill_tps: float
    decode_tps: float
    total_time_s: float
    prompt_tokens: int
    output_tokens: int
    peak_memory_mb: float
    per_token_latency_ms: List[float] = field(default_factory=list)
    gpu_memory_mb: Optional[float] = None
    gpu_utilization_pct: Optional[float] = None
    used_gpu_acceleration: bool = False
    cpu_temp_c: Optional[float] = None
    gpu_temp_c: Optional[float] = None
    power_watts: Optional[float] = None
    # Aggregated thermal/power stats (min, avg, max)
    cpu_temp_stats: Optional[tuple] = None  # (min, avg, max) in Celsius
    gpu_temp_stats: Optional[tuple] = None  # (min, avg, max) in Celsius
    power_stats: Optional[tuple] = None  # (min, avg, max) in watts
    thermal_throttled: bool = False


@dataclass
class QuantizationResult:
    """Results from quantization profiling."""
    
    quantization: str
    load_time_s: float
    peak_ram_mb: float
    ram_increase_mb: float
    ttft_ms: float
    prefill_tps: float
    decode_tps: float
    prompt_tokens: int
    output_tokens: int
    gpu_memory_mb: Optional[float] = None
    gpu_utilization_pct: Optional[float] = None
    used_gpu_acceleration: bool = False
    iteration: int = 1  # Iteration number for statistical analysis


@dataclass
class AblationResult:
    """Results from ablation studies."""
    
    scenario: str
    configuration: Dict[str, any]
    metrics: Dict[str, float]
    improvement_over_baseline: Optional[float] = None


@dataclass
class StatisticalSummary:
    """Statistical summary of multiple runs."""
    
    metric_name: str
    mean: float
    std_dev: float
    confidence_interval_95: Tuple[float, float]
    outliers: List[float] = field(default_factory=list)
    quantization: Optional[str] = None  # Quantization level this summary applies to


@dataclass
class ComparisonResult:
    """Statistical comparison between two configurations."""
    
    metric_name: str
    config_a_mean: float
    config_b_mean: float
    difference: float
    p_value: float
    is_significant: bool  # p < 0.05


@dataclass
class BenchmarkRun:
    """Complete benchmark run results."""
    
    # Metadata
    run_id: str
    timestamp: str
    duration_s: float
    
    # Environment
    hardware_info: HardwareInfo
    software_versions: Dict[str, str]
    config: Dict[str, any]
    model_checksums: Dict[str, str]
    
    # Results
    quantization_results: List[QuantizationResult] = field(default_factory=list)
    ablation_results: List[AblationResult] = field(default_factory=list)
    batch_results: List[AblationResult] = field(default_factory=list)
    
    # Statistical Analysis
    statistical_summaries: List[StatisticalSummary] = field(default_factory=list)
    comparisons: List[ComparisonResult] = field(default_factory=list)
    
    # Visualizations
    visualization_paths: List[str] = field(default_factory=list)
    html_report_path: str = ""
