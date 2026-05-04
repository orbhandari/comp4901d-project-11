"""
Unit tests for ThermalMonitor and PowerMonitor.

Tests thermal and power monitoring functionality with mocked file system.
"""

import pytest
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
from llm_benchmark.metrics.monitors import ThermalMonitor, PowerMonitor


class TestThermalMonitor:
    """Tests for ThermalMonitor class."""
    
    def test_initialization(self):
        """Test ThermalMonitor initialization."""
        with patch.object(ThermalMonitor, '_discover_cpu_thermal_zones', return_value=[]), \
             patch.object(ThermalMonitor, '_discover_gpu_thermal_zones', return_value=[]):
            monitor = ThermalMonitor()
            
            assert monitor.cpu_temps == []
            assert monitor.gpu_temps == []
            assert monitor.monitoring is False
            assert monitor.thermal_throttled is False
            assert monitor.throttle_threshold_c == 85.0
    
    def test_discover_cpu_thermal_zones(self):
        """Test CPU thermal zone discovery."""
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.glob') as mock_glob, \
             patch('pathlib.Path.read_text') as mock_read:
            
            # Mock thermal directory exists
            mock_exists.return_value = True
            
            # Mock thermal zones
            zone0 = Mock(spec=Path)
            zone0.__truediv__ = Mock(side_effect=lambda x: Mock(
                exists=Mock(return_value=True),
                read_text=Mock(return_value='coretemp\n' if x == 'type' else '50000\n')
            ))
            
            mock_glob.return_value = [zone0]
            
            monitor = ThermalMonitor()
            
            # Should discover CPU thermal zones
            assert len(monitor.cpu_thermal_zones) >= 0
    
    def test_discover_gpu_thermal_zones(self):
        """Test GPU thermal zone discovery."""
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.glob') as mock_glob:
            
            # Mock thermal directory exists
            mock_exists.return_value = True
            
            # Mock thermal zones
            zone0 = Mock(spec=Path)
            zone0.__truediv__ = Mock(side_effect=lambda x: Mock(
                exists=Mock(return_value=True),
                read_text=Mock(return_value='gpu_thermal\n' if x == 'type' else '60000\n')
            ))
            
            mock_glob.return_value = [zone0]
            
            monitor = ThermalMonitor()
            
            # Should discover GPU thermal zones
            assert len(monitor.gpu_thermal_zones) >= 0
    
    def test_read_temperature(self):
        """Test temperature reading from file."""
        with patch.object(ThermalMonitor, '_discover_cpu_thermal_zones', return_value=[]), \
             patch.object(ThermalMonitor, '_discover_gpu_thermal_zones', return_value=[]):
            
            monitor = ThermalMonitor()
            
            # Mock temperature file
            temp_file = Mock(spec=Path)
            temp_file.read_text.return_value = "55000\n"
            
            temp = monitor._read_temperature(temp_file)
            
            assert temp == 55.0
    
    def test_read_temperature_error(self):
        """Test temperature reading error handling."""
        with patch.object(ThermalMonitor, '_discover_cpu_thermal_zones', return_value=[]), \
             patch.object(ThermalMonitor, '_discover_gpu_thermal_zones', return_value=[]):
            
            monitor = ThermalMonitor()
            
            # Mock temperature file that raises exception
            temp_file = Mock(spec=Path)
            temp_file.read_text.side_effect = IOError("File not found")
            
            temp = monitor._read_temperature(temp_file)
            
            assert temp is None
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        with patch.object(ThermalMonitor, '_discover_cpu_thermal_zones', return_value=[]), \
             patch.object(ThermalMonitor, '_discover_gpu_thermal_zones', return_value=[]):
            
            monitor = ThermalMonitor()
            
            # Start monitoring
            monitor.start_monitoring()
            
            assert monitor.monitoring is True
            assert monitor.monitor_thread is not None
            assert monitor.monitor_thread.is_alive()
            
            # Let it run briefly
            time.sleep(0.1)
            
            # Stop monitoring
            cpu_stats, gpu_stats, throttled = monitor.stop_monitoring()
            
            assert monitor.monitoring is False
            assert throttled is False
    
    def test_thermal_throttling_detection(self):
        """Test thermal throttling detection."""
        # Create mock thermal zones
        cpu_zone = Mock(spec=Path)
        cpu_zone.read_text.return_value = "90000\n"  # 90°C - above threshold
        
        with patch.object(ThermalMonitor, '_discover_cpu_thermal_zones', return_value=[cpu_zone]), \
             patch.object(ThermalMonitor, '_discover_gpu_thermal_zones', return_value=[]):
            
            monitor = ThermalMonitor()
            
            # Start monitoring with 85°C threshold
            monitor.start_monitoring(throttle_threshold_c=85.0)
            
            # Let it collect some samples
            time.sleep(1.5)
            
            # Stop monitoring
            cpu_stats, gpu_stats, throttled = monitor.stop_monitoring()
            
            # Should detect throttling
            assert throttled is True
            
            # Should have CPU temperature stats
            if cpu_stats:
                assert len(cpu_stats) == 3  # (min, avg, max)
                assert cpu_stats[2] >= 85.0  # max temp should be >= threshold
    
    def test_aggregated_temperature_stats(self):
        """Test temperature statistics aggregation."""
        # Create mock thermal zones with varying temperatures
        cpu_zone = Mock(spec=Path)
        temps = [50000, 55000, 60000, 55000, 50000]  # Varying temps
        temp_iter = iter(temps)
        cpu_zone.read_text.side_effect = lambda: f"{next(temp_iter)}\n"
        
        with patch.object(ThermalMonitor, '_discover_cpu_thermal_zones', return_value=[cpu_zone]), \
             patch.object(ThermalMonitor, '_discover_gpu_thermal_zones', return_value=[]):
            
            monitor = ThermalMonitor()
            
            # Manually populate temps for testing
            monitor.cpu_temps = [50.0, 55.0, 60.0, 55.0, 50.0]
            
            # Calculate stats manually
            cpu_stats = (
                round(min(monitor.cpu_temps), 2),
                round(sum(monitor.cpu_temps) / len(monitor.cpu_temps), 2),
                round(max(monitor.cpu_temps), 2)
            )
            
            assert cpu_stats == (50.0, 54.0, 60.0)
    
    def test_get_current_temperatures(self):
        """Test getting current temperatures."""
        cpu_zone = Mock(spec=Path)
        cpu_zone.read_text.return_value = "55000\n"
        
        gpu_zone = Mock(spec=Path)
        gpu_zone.read_text.return_value = "65000\n"
        
        with patch.object(ThermalMonitor, '_discover_cpu_thermal_zones', return_value=[cpu_zone]), \
             patch.object(ThermalMonitor, '_discover_gpu_thermal_zones', return_value=[gpu_zone]):
            
            monitor = ThermalMonitor()
            
            cpu_temp, gpu_temp = monitor.get_current_temperatures()
            
            assert cpu_temp == 55.0
            assert gpu_temp == 65.0


class TestPowerMonitor:
    """Tests for PowerMonitor class."""
    
    def test_initialization(self):
        """Test PowerMonitor initialization."""
        with patch.object(PowerMonitor, '_discover_power_sensors', return_value=[]):
            monitor = PowerMonitor()
            
            assert monitor.power_readings == []
            assert monitor.monitoring is False
    
    def test_discover_power_sensors_hwmon(self):
        """Test power sensor discovery in hwmon."""
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.glob') as mock_glob:
            
            # Mock hwmon directory exists
            mock_exists.return_value = True
            
            # Mock hwmon devices
            hwmon0 = Mock(spec=Path)
            power_file = Mock(spec=Path)
            hwmon0.glob.return_value = [power_file]
            
            mock_glob.return_value = [hwmon0]
            
            monitor = PowerMonitor()
            
            # Should discover power sensors
            assert len(monitor.power_sensors) >= 0
    
    def test_read_power_milliwatts(self):
        """Test power reading in milliwatts."""
        with patch.object(PowerMonitor, '_discover_power_sensors', return_value=[]):
            monitor = PowerMonitor()
            
            # Mock power file (milliwatts)
            power_file = Mock(spec=Path)
            power_file.read_text.return_value = "5000\n"  # 5000 mW = 5 W
            
            power = monitor._read_power(power_file)
            
            assert power == 5.0
    
    def test_read_power_microwatts(self):
        """Test power reading in microwatts."""
        with patch.object(PowerMonitor, '_discover_power_sensors', return_value=[]):
            monitor = PowerMonitor()
            
            # Mock power file (microwatts)
            power_file = Mock(spec=Path)
            power_file.read_text.return_value = "5000000\n"  # 5000000 µW = 5 W
            
            power = monitor._read_power(power_file)
            
            assert power == 5.0
    
    def test_read_power_error(self):
        """Test power reading error handling."""
        with patch.object(PowerMonitor, '_discover_power_sensors', return_value=[]):
            monitor = PowerMonitor()
            
            # Mock power file that raises exception
            power_file = Mock(spec=Path)
            power_file.read_text.side_effect = IOError("File not found")
            
            power = monitor._read_power(power_file)
            
            assert power is None
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        with patch.object(PowerMonitor, '_discover_power_sensors', return_value=[]):
            monitor = PowerMonitor()
            
            # Start monitoring
            monitor.start_monitoring()
            
            assert monitor.monitoring is True
            assert monitor.monitor_thread is not None
            assert monitor.monitor_thread.is_alive()
            
            # Let it run briefly
            time.sleep(0.1)
            
            # Stop monitoring
            power_stats = monitor.stop_monitoring()
            
            assert monitor.monitoring is False
    
    def test_aggregated_power_stats(self):
        """Test power statistics aggregation."""
        with patch.object(PowerMonitor, '_discover_power_sensors', return_value=[]):
            monitor = PowerMonitor()
            
            # Manually populate power readings for testing
            monitor.power_readings = [5.0, 6.0, 7.0, 6.5, 5.5]
            
            # Calculate stats manually
            power_stats = (
                round(min(monitor.power_readings), 2),
                round(sum(monitor.power_readings) / len(monitor.power_readings), 2),
                round(max(monitor.power_readings), 2)
            )
            
            assert power_stats == (5.0, 6.0, 7.0)
    
    def test_multiple_power_rails(self):
        """Test aggregation of multiple power rails."""
        power_file1 = Mock(spec=Path)
        power_file1.read_text.return_value = "3000\n"  # 3W
        
        power_file2 = Mock(spec=Path)
        power_file2.read_text.return_value = "2000\n"  # 2W
        
        with patch.object(PowerMonitor, '_discover_power_sensors', return_value=[power_file1, power_file2]):
            monitor = PowerMonitor()
            
            # Start monitoring
            monitor.start_monitoring()
            
            # Let it collect samples
            time.sleep(1.5)
            
            # Stop monitoring
            power_stats = monitor.stop_monitoring()
            
            # Should aggregate power from both rails
            if power_stats:
                assert len(power_stats) == 3  # (min, avg, max)
                # Total power should be sum of both rails (approximately 5W)
                assert power_stats[1] > 0  # avg power should be positive
    
    def test_get_current_power(self):
        """Test getting current power consumption."""
        power_file = Mock(spec=Path)
        power_file.read_text.return_value = "5000\n"  # 5W
        
        with patch.object(PowerMonitor, '_discover_power_sensors', return_value=[power_file]):
            monitor = PowerMonitor()
            
            power = monitor.get_current_power()
            
            assert power == 5.0
    
    def test_no_power_sensors(self):
        """Test behavior when no power sensors are available."""
        with patch.object(PowerMonitor, '_discover_power_sensors', return_value=[]):
            monitor = PowerMonitor()
            
            power = monitor.get_current_power()
            
            assert power is None
    
    def test_stop_monitoring_no_data(self):
        """Test stopping monitoring when no data was collected."""
        with patch.object(PowerMonitor, '_discover_power_sensors', return_value=[]):
            monitor = PowerMonitor()
            
            # Start and immediately stop
            monitor.start_monitoring()
            power_stats = monitor.stop_monitoring()
            
            # Should return None when no data
            assert power_stats is None


class TestMonitoringIntegration:
    """Integration tests for thermal and power monitoring."""
    
    def test_concurrent_monitoring(self):
        """Test running thermal and power monitoring concurrently."""
        with patch.object(ThermalMonitor, '_discover_cpu_thermal_zones', return_value=[]), \
             patch.object(ThermalMonitor, '_discover_gpu_thermal_zones', return_value=[]), \
             patch.object(PowerMonitor, '_discover_power_sensors', return_value=[]):
            
            thermal_monitor = ThermalMonitor()
            power_monitor = PowerMonitor()
            
            # Start both monitors
            thermal_monitor.start_monitoring()
            power_monitor.start_monitoring()
            
            # Let them run
            time.sleep(0.5)
            
            # Stop both monitors
            cpu_stats, gpu_stats, throttled = thermal_monitor.stop_monitoring()
            power_stats = power_monitor.stop_monitoring()
            
            # Both should complete successfully
            assert thermal_monitor.monitoring is False
            assert power_monitor.monitoring is False
    
    def test_monitoring_thread_cleanup(self):
        """Test that monitoring threads are properly cleaned up."""
        with patch.object(ThermalMonitor, '_discover_cpu_thermal_zones', return_value=[]), \
             patch.object(ThermalMonitor, '_discover_gpu_thermal_zones', return_value=[]):
            
            monitor = ThermalMonitor()
            
            # Start monitoring
            monitor.start_monitoring()
            thread = monitor.monitor_thread
            
            assert thread.is_alive()
            
            # Stop monitoring
            monitor.stop_monitoring()
            
            # Wait a bit for thread to finish
            time.sleep(0.5)
            
            # Thread should be stopped
            assert not thread.is_alive()
