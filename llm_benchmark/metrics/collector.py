"""
Metrics collection implementation.

Provides real-time measurement of inference metrics including timing,
memory usage, GPU utilization, thermal, and power.
"""

import logging
import signal
import time
from contextlib import contextmanager
from typing import List, Optional

import psutil

from llm_benchmark.models import HardwareInfo, InferenceMetrics
from llm_benchmark.metrics.monitors import ThermalMonitor, PowerMonitor

logger = logging.getLogger(__name__)


class InferenceTimeoutError(Exception):
    """Exception raised when inference exceeds timeout limit."""
    pass


class MetricsCollector:
    """
    Collects comprehensive metrics during inference.
    
    Implements high-resolution timing, memory tracking, GPU metrics,
    and thermal/power monitoring.
    """
    
    def __init__(self, hw_info: HardwareInfo):
        """
        Initialize metrics collector.
        
        Args:
            hw_info: Hardware information from detector
        """
        self.hw_info = hw_info
        self.process = psutil.Process()
        
        # Initialize GPU monitoring if available
        self.nvml_initialized = False
        self.gpu_handle = None
        
        if self.hw_info.has_gpu:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml_initialized = True
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                logger.info("GPU monitoring initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU monitoring: {e}")
        
        # Initialize thermal and power monitors
        self.thermal_monitor = None
        self.power_monitor = None
        
        if self.hw_info.has_thermal_sensors:
            try:
                self.thermal_monitor = ThermalMonitor()
                logger.info("Thermal monitoring initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize thermal monitoring: {e}")
        
        if self.hw_info.has_power_sensors:
            try:
                self.power_monitor = PowerMonitor()
                logger.info("Power monitoring initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize power monitoring: {e}")
    
    def __del__(self):
        """Cleanup NVML on destruction."""
        if self.nvml_initialized:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except:
                pass
    
    @contextmanager
    def _timeout_context(self, seconds: int):
        """
        Context manager for timeout protection using signal.alarm.
        
        Args:
            seconds: Timeout in seconds
        
        Raises:
            InferenceTimeoutError: If operation exceeds timeout
        """
        def timeout_handler(signum, frame):
            raise InferenceTimeoutError(f"Inference timed out after {seconds}s")
        
        # Save original handler
        original_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        
        try:
            yield
        finally:
            # Restore original handler and cancel alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)
    
    def start_monitoring(self, throttle_threshold_c: float = 85.0) -> None:
        """
        Start background monitoring threads for thermal/power.
        
        Args:
            throttle_threshold_c: Temperature threshold for throttling detection
        """
        if self.thermal_monitor:
            self.thermal_monitor.start_monitoring(throttle_threshold_c)
        
        if self.power_monitor:
            self.power_monitor.start_monitoring()
    
    def stop_monitoring(self) -> dict:
        """
        Stop background monitoring and aggregate results.
        
        Returns:
            Dictionary with aggregated thermal and power statistics
        """
        results = {
            'cpu_temp_stats': None,
            'gpu_temp_stats': None,
            'thermal_throttled': False,
            'power_stats': None
        }
        
        if self.thermal_monitor:
            cpu_stats, gpu_stats, throttled = self.thermal_monitor.stop_monitoring()
            results['cpu_temp_stats'] = cpu_stats
            results['gpu_temp_stats'] = gpu_stats
            results['thermal_throttled'] = throttled
        
        if self.power_monitor:
            power_stats = self.power_monitor.stop_monitoring()
            results['power_stats'] = power_stats
        
        return results
    
    def collect_inference_metrics(
        self,
        llm,
        prompt: str,
        max_tokens: int,
        enable_background_monitoring: bool = True,
        timeout_s: int = 300,
        enable_prompt_cache: bool = False
    ) -> Optional[InferenceMetrics]:
        """
        Collect comprehensive metrics during inference with timeout protection.
        
        Implements timeout protection using signal.alarm (default 300s).
        Terminates inference after timeout and returns None.
        
        Args:
            llm: Llama model instance
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            enable_background_monitoring: Enable background thermal/power monitoring
            timeout_s: Timeout in seconds (default: 300)
            enable_prompt_cache: Enable prompt caching for this request (NativeLlamaServer only)
        
        Returns:
            InferenceMetrics with all collected measurements, or None if timeout occurs
        
        Raises:
            ValueError: If prompt is empty (programming error)
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        
        try:
            # Use timeout context manager
            with self._timeout_context(timeout_s):
                return self._collect_inference_metrics_impl(
                    llm, prompt, max_tokens, enable_background_monitoring, enable_prompt_cache
                )
        except InferenceTimeoutError as e:
            logger.error(f"Inference timed out after {timeout_s}s")
            logger.info("Suggestions:")
            logger.info(f"  - Reduce max_tokens (current: {max_tokens})")
            logger.info("  - Reduce context_size")
            logger.info("  - Use smaller quantization")
            logger.info("  - Increase timeout_s if hardware is slow")
            
            # Stop monitoring if it was started
            if enable_background_monitoring:
                try:
                    self.stop_monitoring()
                except:
                    pass
            
            return None
    
    def _collect_inference_metrics_impl(
        self,
        llm,
        prompt: str,
        max_tokens: int,
        enable_background_monitoring: bool = True,
        enable_prompt_cache: bool = False
    ) -> InferenceMetrics:
        """
        Collect comprehensive metrics during inference.
        
        Args:
            llm: Llama model instance
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            enable_background_monitoring: Enable background thermal/power monitoring
            enable_prompt_cache: Enable prompt caching for this request (NativeLlamaServer only)
        
        Returns:
            InferenceMetrics with all collected measurements
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        
        # Start background monitoring
        if enable_background_monitoring:
            self.start_monitoring()
        
        # Measure baseline memory
        baseline_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        
        # Start timing
        start_time = time.perf_counter()
        
        # Track per-token latency
        per_token_latency_ms: List[float] = []
        first_token_time: Optional[float] = None
        last_token_time = start_time
        
        # Count tokens
        prompt_tokens = 0
        output_tokens = 0
        
        # Tokenize prompt to count tokens
        try:
            prompt_tokens = len(llm.tokenize(prompt.encode('utf-8')))
        except Exception as e:
            logger.warning(f"Failed to tokenize prompt: {e}")
            # Fallback: estimate based on characters (rough approximation)
            prompt_tokens = len(prompt) // 4
        
        # Perform streaming inference to capture TTFT
        try:
            # Use streaming to capture first token time
            # Check if llm supports enable_prompt_cache parameter (NativeLlamaServer)
            if hasattr(llm, '__call__') and 'enable_prompt_cache' in llm.__call__.__code__.co_varnames:
                stream = llm(
                    prompt,
                    max_tokens=max_tokens,
                    stream=True,
                    echo=False,
                    enable_prompt_cache=enable_prompt_cache
                )
            else:
                # Fallback for NativeLlamaCpp and other backends
                stream = llm(
                    prompt,
                    max_tokens=max_tokens,
                    stream=True,
                    echo=False
                )
            
            for chunk in stream:
                current_time = time.perf_counter()
                
                # Capture TTFT (time to first token)
                if first_token_time is None:
                    first_token_time = current_time
                
                # Track per-token latency
                token_latency_ms = (current_time - last_token_time) * 1000
                per_token_latency_ms.append(round(token_latency_ms, 2))
                last_token_time = current_time
                
                # Count output tokens
                output_tokens += 1
        
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            # Stop monitoring before re-raising
            if enable_background_monitoring:
                self.stop_monitoring()
            raise
        
        # End timing
        end_time = time.perf_counter()
        total_time_s = end_time - start_time
        
        # Stop background monitoring and get aggregated results
        monitoring_results = {}
        if enable_background_monitoring:
            monitoring_results = self.stop_monitoring()
        
        # Calculate TTFT in milliseconds
        if first_token_time is None:
            # No tokens generated
            ttft_ms = 0.0
            ttft_s = 0.0
        else:
            ttft_s = first_token_time - start_time
            ttft_ms = ttft_s * 1000
        
        # Calculate throughput metrics
        # Prefill throughput: prompt_tokens / ttft_s
        if ttft_s > 0 and prompt_tokens > 0:
            prefill_tps = prompt_tokens / ttft_s
        else:
            prefill_tps = 0.0
        
        # Decode throughput: (output_tokens - 1) / decode_duration
        # Note: First output token is part of prefill, so we subtract 1
        decode_duration = total_time_s - ttft_s
        if decode_duration > 0 and output_tokens > 1:
            decode_tps = (output_tokens - 1) / decode_duration
        else:
            decode_tps = 0.0
        
        # Measure peak memory usage
        peak_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        
        # Collect GPU metrics if available
        gpu_memory_mb = None
        gpu_utilization_pct = None
        used_gpu_acceleration = False
        
        if self.nvml_initialized and self.gpu_handle:
            try:
                gpu_memory_mb = self._get_gpu_memory_mb()
                gpu_utilization_pct = self._get_gpu_utilization_pct()
                # GPU acceleration is considered "used" if we successfully collected GPU metrics
                used_gpu_acceleration = True
            except Exception as e:
                logger.warning(f"Failed to collect GPU metrics: {e}")
        
        # Collect instantaneous thermal metrics if available (for backward compatibility)
        cpu_temp_c = None
        gpu_temp_c = None
        
        if self.hw_info.has_thermal_sensors:
            try:
                cpu_temp_c = self._get_cpu_temperature()
                if self.hw_info.has_gpu:
                    gpu_temp_c = self._get_gpu_temperature()
            except Exception as e:
                logger.warning(f"Failed to collect thermal metrics: {e}")
        
        # Collect instantaneous power metrics if available (for backward compatibility)
        power_watts = None
        
        if self.hw_info.has_power_sensors:
            try:
                power_watts = self._get_power_consumption()
            except Exception as e:
                logger.warning(f"Failed to collect power metrics: {e}")
        
        # Create metrics object
        metrics = InferenceMetrics(
            ttft_ms=round(ttft_ms, 2),
            prefill_tps=round(prefill_tps, 2),
            decode_tps=round(decode_tps, 2),
            total_time_s=round(total_time_s, 2),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            peak_memory_mb=round(peak_memory_mb, 2),
            per_token_latency_ms=per_token_latency_ms,
            gpu_memory_mb=round(gpu_memory_mb, 2) if gpu_memory_mb else None,
            gpu_utilization_pct=round(gpu_utilization_pct, 2) if gpu_utilization_pct else None,
            used_gpu_acceleration=used_gpu_acceleration,
            cpu_temp_c=round(cpu_temp_c, 2) if cpu_temp_c else None,
            gpu_temp_c=round(gpu_temp_c, 2) if gpu_temp_c else None,
            power_watts=round(power_watts, 2) if power_watts else None,
            # Aggregated thermal/power stats from background monitoring
            cpu_temp_stats=monitoring_results.get('cpu_temp_stats'),
            gpu_temp_stats=monitoring_results.get('gpu_temp_stats'),
            power_stats=monitoring_results.get('power_stats'),
            thermal_throttled=monitoring_results.get('thermal_throttled', False)
        )
        
        logger.info(f"Metrics collected: TTFT={metrics.ttft_ms}ms, "
                   f"Prefill={metrics.prefill_tps} t/s, "
                   f"Decode={metrics.decode_tps} t/s")
        
        if metrics.thermal_throttled:
            logger.warning("Thermal throttling detected during inference")
        
        return metrics
    
    def _get_gpu_memory_mb(self) -> float:
        """
        Get current GPU memory usage in MB.
        
        Returns:
            GPU memory usage in megabytes
        """
        import pynvml
        
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        return mem_info.used / (1024 * 1024)
    
    def _get_gpu_utilization_pct(self) -> float:
        """
        Get current GPU utilization percentage.
        
        Returns:
            GPU utilization as percentage (0-100)
        """
        import pynvml
        
        utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        return float(utilization.gpu)
    
    def _get_cpu_temperature(self) -> Optional[float]:
        """
        Get CPU temperature in Celsius.
        
        Returns:
            CPU temperature or None if unavailable
        """
        try:
            temps = psutil.sensors_temperatures()
            
            # Try common sensor names
            for sensor_name in ['coretemp', 'cpu_thermal', 'k10temp']:
                if sensor_name in temps:
                    # Return the first temperature reading
                    if temps[sensor_name]:
                        return temps[sensor_name][0].current
            
            # Fallback: return first available temperature
            for sensor_list in temps.values():
                if sensor_list:
                    return sensor_list[0].current
        
        except Exception as e:
            logger.debug(f"Failed to read CPU temperature: {e}")
        
        return None
    
    def _get_gpu_temperature(self) -> Optional[float]:
        """
        Get GPU temperature in Celsius.
        
        Returns:
            GPU temperature or None if unavailable
        """
        if not self.nvml_initialized or not self.gpu_handle:
            return None
        
        try:
            import pynvml
            
            temp = pynvml.nvmlDeviceGetTemperature(
                self.gpu_handle,
                pynvml.NVML_TEMPERATURE_GPU
            )
            return float(temp)
        
        except Exception as e:
            logger.debug(f"Failed to read GPU temperature: {e}")
            return None
    
    def _get_power_consumption(self) -> Optional[float]:
        """
        Get power consumption in watts.
        
        Returns:
            Power consumption or None if unavailable
        """
        # Try GPU power first (more reliable on Jetson)
        if self.nvml_initialized and self.gpu_handle:
            try:
                import pynvml
                
                # Power in milliwatts
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                return power_mw / 1000.0
            
            except Exception as e:
                logger.debug(f"Failed to read GPU power: {e}")
        
        # Fallback: try system power sensors
        # This is platform-specific and may not be available
        # On Jetson, power rails are typically in /sys/bus/i2c/drivers/ina3221x
        # For now, return None if GPU power is unavailable
        
        return None
