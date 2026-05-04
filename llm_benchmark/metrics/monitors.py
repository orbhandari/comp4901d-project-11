"""
Thermal and power monitoring implementation.

Provides background monitoring of thermal and power metrics during inference.
"""

import logging
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class ThermalMonitor:
    """
    Monitor thermal sensors in background thread.
    
    Reads from /sys/class/thermal and aggregates temperature readings
    over the monitoring period.
    """
    
    def __init__(self):
        """Initialize thermal monitor."""
        self.cpu_temps: List[float] = []
        self.gpu_temps: List[float] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.thermal_throttled = False
        self.throttle_threshold_c = 85.0  # Default throttle threshold
        
        # Discover thermal zones
        self.cpu_thermal_zones = self._discover_cpu_thermal_zones()
        self.gpu_thermal_zones = self._discover_gpu_thermal_zones()
        
        logger.debug(f"Discovered {len(self.cpu_thermal_zones)} CPU thermal zones")
        logger.debug(f"Discovered {len(self.gpu_thermal_zones)} GPU thermal zones")
    
    def _discover_cpu_thermal_zones(self) -> List[Path]:
        """
        Discover CPU thermal zones.
        
        Returns:
            List of paths to CPU thermal zone temperature files
        """
        thermal_dir = Path("/sys/class/thermal")
        if not thermal_dir.exists():
            return []
        
        cpu_zones = []
        
        # Iterate through thermal zones
        for zone_dir in sorted(thermal_dir.glob("thermal_zone*")):
            type_file = zone_dir / "type"
            temp_file = zone_dir / "temp"
            
            if not type_file.exists() or not temp_file.exists():
                continue
            
            try:
                zone_type = type_file.read_text().strip().lower()
                
                # Common CPU thermal zone types
                cpu_types = [
                    'x86_pkg_temp',
                    'coretemp',
                    'cpu_thermal',
                    'cpu-thermal',
                    'k10temp',
                    'zenpower'
                ]
                
                if any(cpu_type in zone_type for cpu_type in cpu_types):
                    cpu_zones.append(temp_file)
                    logger.debug(f"Found CPU thermal zone: {zone_type} at {temp_file}")
            
            except Exception as e:
                logger.debug(f"Error reading thermal zone {zone_dir}: {e}")
        
        return cpu_zones
    
    def _discover_gpu_thermal_zones(self) -> List[Path]:
        """
        Discover GPU thermal zones.
        
        Returns:
            List of paths to GPU thermal zone temperature files
        """
        thermal_dir = Path("/sys/class/thermal")
        if not thermal_dir.exists():
            return []
        
        gpu_zones = []
        
        # Iterate through thermal zones
        for zone_dir in sorted(thermal_dir.glob("thermal_zone*")):
            type_file = zone_dir / "type"
            temp_file = zone_dir / "temp"
            
            if not type_file.exists() or not temp_file.exists():
                continue
            
            try:
                zone_type = type_file.read_text().strip().lower()
                
                # Common GPU thermal zone types
                gpu_types = [
                    'gpu',
                    'tegra',
                    'nvidia',
                    'amdgpu'
                ]
                
                if any(gpu_type in zone_type for gpu_type in gpu_types):
                    gpu_zones.append(temp_file)
                    logger.debug(f"Found GPU thermal zone: {zone_type} at {temp_file}")
            
            except Exception as e:
                logger.debug(f"Error reading thermal zone {zone_dir}: {e}")
        
        return gpu_zones
    
    def _read_temperature(self, temp_file: Path) -> Optional[float]:
        """
        Read temperature from thermal zone file.
        
        Args:
            temp_file: Path to temperature file
        
        Returns:
            Temperature in Celsius or None if read fails
        """
        try:
            temp_millidegrees = int(temp_file.read_text().strip())
            temp_celsius = temp_millidegrees / 1000.0
            return temp_celsius
        except Exception as e:
            logger.debug(f"Error reading temperature from {temp_file}: {e}")
            return None
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop running at 1 Hz."""
        while self.monitoring:
            # Read CPU temperatures
            cpu_temps = []
            for zone in self.cpu_thermal_zones:
                temp = self._read_temperature(zone)
                if temp is not None:
                    cpu_temps.append(temp)
            
            if cpu_temps:
                max_cpu_temp = max(cpu_temps)
                self.cpu_temps.append(max_cpu_temp)
                
                # Check for thermal throttling
                if max_cpu_temp > self.throttle_threshold_c:
                    self.thermal_throttled = True
                    logger.warning(f"Thermal throttling detected: CPU {max_cpu_temp}°C")
            
            # Read GPU temperatures
            gpu_temps = []
            for zone in self.gpu_thermal_zones:
                temp = self._read_temperature(zone)
                if temp is not None:
                    gpu_temps.append(temp)
            
            if gpu_temps:
                max_gpu_temp = max(gpu_temps)
                self.gpu_temps.append(max_gpu_temp)
                
                # Check for thermal throttling
                if max_gpu_temp > self.throttle_threshold_c:
                    self.thermal_throttled = True
                    logger.warning(f"Thermal throttling detected: GPU {max_gpu_temp}°C")
            
            # Sleep for 1 second (1 Hz sampling)
            time.sleep(1.0)
    
    def start_monitoring(self, throttle_threshold_c: float = 85.0) -> None:
        """
        Start background thermal monitoring.
        
        Args:
            throttle_threshold_c: Temperature threshold for throttling detection
        """
        if self.monitoring:
            logger.warning("Thermal monitoring already running")
            return
        
        self.throttle_threshold_c = throttle_threshold_c
        self.cpu_temps = []
        self.gpu_temps = []
        self.thermal_throttled = False
        self.monitoring = True
        
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="ThermalMonitor"
        )
        self.monitor_thread.start()
        logger.debug("Thermal monitoring started")
    
    def stop_monitoring(self) -> Tuple[Optional[float], Optional[float], bool]:
        """
        Stop background thermal monitoring and return aggregated results.
        
        Returns:
            Tuple of (cpu_temp_stats, gpu_temp_stats, thermal_throttled)
            where temp_stats is (min, avg, max) or None if no data
        """
        if not self.monitoring:
            logger.warning("Thermal monitoring not running")
            return None, None, False
        
        self.monitoring = False
        
        # Wait for monitor thread to finish
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        # Aggregate CPU temperatures
        cpu_stats = None
        if self.cpu_temps:
            cpu_stats = (
                round(min(self.cpu_temps), 2),
                round(sum(self.cpu_temps) / len(self.cpu_temps), 2),
                round(max(self.cpu_temps), 2)
            )
            logger.debug(f"CPU temp stats: min={cpu_stats[0]}°C, "
                        f"avg={cpu_stats[1]}°C, max={cpu_stats[2]}°C")
        
        # Aggregate GPU temperatures
        gpu_stats = None
        if self.gpu_temps:
            gpu_stats = (
                round(min(self.gpu_temps), 2),
                round(sum(self.gpu_temps) / len(self.gpu_temps), 2),
                round(max(self.gpu_temps), 2)
            )
            logger.debug(f"GPU temp stats: min={gpu_stats[0]}°C, "
                        f"avg={gpu_stats[1]}°C, max={gpu_stats[2]}°C")
        
        logger.debug("Thermal monitoring stopped")
        
        return cpu_stats, gpu_stats, self.thermal_throttled
    
    def get_current_temperatures(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Get current CPU and GPU temperatures.
        
        Returns:
            Tuple of (cpu_temp, gpu_temp) in Celsius or None if unavailable
        """
        cpu_temp = None
        if self.cpu_thermal_zones:
            temps = []
            for zone in self.cpu_thermal_zones:
                temp = self._read_temperature(zone)
                if temp is not None:
                    temps.append(temp)
            if temps:
                cpu_temp = max(temps)
        
        gpu_temp = None
        if self.gpu_thermal_zones:
            temps = []
            for zone in self.gpu_thermal_zones:
                temp = self._read_temperature(zone)
                if temp is not None:
                    temps.append(temp)
            if temps:
                gpu_temp = max(temps)
        
        return cpu_temp, gpu_temp


class PowerMonitor:
    """
    Monitor power consumption in background thread.
    
    Reads from /sys/class/hwmon and Jetson-specific power rails,
    aggregating power measurements over the monitoring period.
    """
    
    def __init__(self):
        """Initialize power monitor."""
        self.power_readings: List[float] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Discover power sensors
        self.power_sensors = self._discover_power_sensors()
        
        logger.debug(f"Discovered {len(self.power_sensors)} power sensors")
    
    def _discover_power_sensors(self) -> List[Path]:
        """
        Discover power sensors.
        
        Returns:
            List of paths to power sensor files
        """
        sensors = []
        
        # Check hwmon devices
        hwmon_dir = Path("/sys/class/hwmon")
        if hwmon_dir.exists():
            for hwmon_device in sorted(hwmon_dir.glob("hwmon*")):
                # Look for power input files
                for power_file in hwmon_device.glob("power*_input"):
                    sensors.append(power_file)
                    logger.debug(f"Found hwmon power sensor: {power_file}")
        
        # Check Jetson-specific power rails (INA3221)
        jetson_power_paths = [
            "/sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power0_input",
            "/sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power1_input",
            "/sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power2_input",
        ]
        
        for pattern in jetson_power_paths:
            for power_file in Path("/").glob(pattern.lstrip("/")):
                if power_file.exists():
                    sensors.append(power_file)
                    logger.debug(f"Found Jetson power sensor: {power_file}")
        
        # Alternative Jetson power paths
        jetson_alt_paths = [
            "/sys/devices/3160000.i2c/i2c-0/0-0040/iio:device0/in_power0_input",
            "/sys/devices/3160000.i2c/i2c-0/0-0040/iio:device0/in_power1_input",
            "/sys/devices/3160000.i2c/i2c-0/0-0041/iio:device1/in_power0_input",
        ]
        
        for path_str in jetson_alt_paths:
            power_file = Path(path_str)
            if power_file.exists() and power_file not in sensors:
                sensors.append(power_file)
                logger.debug(f"Found Jetson power sensor (alt): {power_file}")
        
        return sensors
    
    def _read_power(self, power_file: Path) -> Optional[float]:
        """
        Read power from sensor file.
        
        Args:
            power_file: Path to power sensor file
        
        Returns:
            Power in watts or None if read fails
        """
        try:
            power_value = int(power_file.read_text().strip())
            
            # Power files can be in milliwatts or microwatts
            # Heuristic: if value > 100000, assume microwatts
            if power_value > 100000:
                power_watts = power_value / 1_000_000.0
            else:
                power_watts = power_value / 1000.0
            
            return power_watts
        except Exception as e:
            logger.debug(f"Error reading power from {power_file}: {e}")
            return None
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop running at 1 Hz."""
        while self.monitoring:
            # Read all power sensors
            power_readings = []
            for sensor in self.power_sensors:
                power = self._read_power(sensor)
                if power is not None and power > 0:
                    power_readings.append(power)
            
            # Aggregate total power (sum of all rails)
            if power_readings:
                total_power = sum(power_readings)
                self.power_readings.append(total_power)
            
            # Sleep for 1 second (1 Hz sampling)
            time.sleep(1.0)
    
    def start_monitoring(self) -> None:
        """Start background power monitoring."""
        if self.monitoring:
            logger.warning("Power monitoring already running")
            return
        
        self.power_readings = []
        self.monitoring = True
        
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="PowerMonitor"
        )
        self.monitor_thread.start()
        logger.debug("Power monitoring started")
    
    def stop_monitoring(self) -> Optional[Tuple[float, float, float]]:
        """
        Stop background power monitoring and return aggregated results.
        
        Returns:
            Tuple of (min, avg, max) power in watts or None if no data
        """
        if not self.monitoring:
            logger.warning("Power monitoring not running")
            return None
        
        self.monitoring = False
        
        # Wait for monitor thread to finish
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        # Aggregate power readings
        if not self.power_readings:
            logger.debug("No power readings collected")
            return None
        
        power_stats = (
            round(min(self.power_readings), 2),
            round(sum(self.power_readings) / len(self.power_readings), 2),
            round(max(self.power_readings), 2)
        )
        
        logger.debug(f"Power stats: min={power_stats[0]}W, "
                    f"avg={power_stats[1]}W, max={power_stats[2]}W")
        
        logger.debug("Power monitoring stopped")
        
        return power_stats
    
    def get_current_power(self) -> Optional[float]:
        """
        Get current power consumption.
        
        Returns:
            Current power in watts or None if unavailable
        """
        if not self.power_sensors:
            return None
        
        power_readings = []
        for sensor in self.power_sensors:
            power = self._read_power(sensor)
            if power is not None and power > 0:
                power_readings.append(power)
        
        if not power_readings:
            return None
        
        return sum(power_readings)
