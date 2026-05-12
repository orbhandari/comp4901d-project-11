"""
Unit tests for HardwareDetector.

Tests hardware detection logic with mocked system calls.
Validates Requirements 1.1-1.7.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

from llm_benchmark.hardware.detector import HardwareDetector
from llm_benchmark.models import HardwareInfo


class TestHardwareDetectorOSType:
    """Test OS type detection (Requirement 1.1)."""
    
    def test_detect_linux_x86(self, mocker):
        """Test detection of Linux x86 platform."""
        mocker.patch('platform.system', return_value='Linux')
        mocker.patch('platform.machine', return_value='x86_64')
        mocker.patch('pathlib.Path.exists', return_value=False)
        
        os_type = HardwareDetector._detect_os_type()
        
        assert os_type == "linux_x86"
    
    def test_detect_jetson_via_nv_tegra_release(self, mocker):
        """Test detection of Jetson via /etc/nv_tegra_release."""
        mocker.patch('platform.system', return_value='Linux')
        mocker.patch('platform.machine', return_value='aarch64')
        
        # Mock Path.exists to return True for /etc/nv_tegra_release
        class MockPath:
            def __init__(self, path):
                self.path = str(path)
            
            def exists(self):
                return self.path == "/etc/nv_tegra_release"
        
        mocker.patch('llm_benchmark.hardware.detector.Path', MockPath)
        
        # Mock the file content - need to include 'jetson' or 'xavier' in lowercase
        mock_file = mock_open(read_data="# R32 (release), REVISION: 7.1, GCID: 29818004, BOARD: t186ref, EABI: aarch64, DATE: Sat Feb 19 17:07:00 UTC 2022\nJetson Xavier NX")
        mocker.patch('builtins.open', mock_file)
        
        os_type = HardwareDetector._detect_os_type()
        
        assert os_type == "jetson_xavier_nx"
    
    def test_detect_jetson_via_device_tree(self, mocker):
        """Test detection of Jetson via device tree."""
        mocker.patch('platform.system', return_value='Linux')
        mocker.patch('platform.machine', return_value='aarch64')
        
        # Mock Path.exists to return True for device tree path
        class MockPath:
            def __init__(self, path):
                self.path = str(path)
            
            def exists(self):
                return self.path == "/sys/firmware/devicetree/base/model"
        
        mocker.patch('llm_benchmark.hardware.detector.Path', MockPath)
        mock_file = mock_open(read_data="NVIDIA Jetson Xavier NX Developer Kit")
        mocker.patch('builtins.open', mock_file)
        
        os_type = HardwareDetector._detect_os_type()
        
        assert os_type == "jetson_xavier_nx"
    
    def test_unsupported_os_defaults_to_linux_x86(self, mocker):
        """Test that unsupported OS defaults to linux_x86."""
        mocker.patch('platform.system', return_value='Windows')
        mocker.patch('pathlib.Path.exists', return_value=False)
        
        os_type = HardwareDetector._detect_os_type()
        
        assert os_type == "linux_x86"


class TestHardwareDetectorCPU:
    """Test CPU detection (Requirements 1.3, 1.5)."""
    
    def test_detect_cpu_model_from_proc_cpuinfo(self, mocker):
        """Test CPU model detection from /proc/cpuinfo."""
        cpuinfo_content = """processor	: 0
vendor_id	: GenuineIntel
cpu family	: 6
model		: 142
model name	: Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz
stepping	: 10
"""
        mock_file = mock_open(read_data=cpuinfo_content)
        mocker.patch('builtins.open', mock_file)
        
        cpu_model = HardwareDetector._detect_cpu_model()
        
        assert cpu_model == "Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz"
    
    def test_detect_cpu_model_fallback_to_platform(self, mocker):
        """Test CPU model detection fallback to platform.processor()."""
        mocker.patch('builtins.open', side_effect=FileNotFoundError)
        mocker.patch('platform.processor', return_value='x86_64')
        
        cpu_model = HardwareDetector._detect_cpu_model()
        
        assert cpu_model == "x86_64"
    
    def test_detect_cpu_model_unknown_fallback(self, mocker):
        """Test CPU model detection with no available information."""
        mocker.patch('builtins.open', side_effect=FileNotFoundError)
        mocker.patch('platform.processor', return_value='')
        
        cpu_model = HardwareDetector._detect_cpu_model()
        
        assert cpu_model == "Unknown CPU"
    
    def test_detect_cpu_features_avx2(self, mocker):
        """Test detection of AVX2 CPU features."""
        cpuinfo_content = """processor	: 0
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb invpcid_single pti ssbd ibrs ibpb stibp tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp md_clear flush_l1d
"""
        mock_file = mock_open(read_data=cpuinfo_content)
        mocker.patch('builtins.open', mock_file)
        
        features = HardwareDetector._detect_cpu_features()
        
        assert 'avx' in features
        assert 'avx2' in features
        assert 'sse' in features
        assert 'sse2' in features
        assert 'sse4_1' in features
        assert 'sse4_2' in features
    
    def test_detect_cpu_features_avx512(self, mocker):
        """Test detection of AVX512 CPU features."""
        cpuinfo_content = """processor	: 0
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cdp_l3 invpcid_single intel_ppin ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm cqm mpx rdt_a avx512f avx512dq rdseed adx smap clflushopt clwb intel_pt avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local split_lock_detect wbnoinvd dtherm ida arat pln pts pku ospke avx512_vnni md_clear flush_l1d arch_capabilities
"""
        mock_file = mock_open(read_data=cpuinfo_content)
        mocker.patch('builtins.open', mock_file)
        
        features = HardwareDetector._detect_cpu_features()
        
        assert 'avx512f' in features
    
    def test_detect_cpu_features_error_handling(self, mocker):
        """Test CPU features detection with file read error."""
        mocker.patch('builtins.open', side_effect=PermissionError)
        
        features = HardwareDetector._detect_cpu_features()
        
        assert features == []


class TestHardwareDetectorMemory:
    """Test memory detection (Requirement 1.4)."""
    
    def test_detect_memory(self, mocker):
        """Test RAM detection using psutil."""
        mock_mem = Mock()
        mock_mem.total = 16 * 1024 ** 3  # 16 GB
        mock_mem.available = 8 * 1024 ** 3  # 8 GB
        mocker.patch('psutil.virtual_memory', return_value=mock_mem)
        mocker.patch('psutil.cpu_count', return_value=4)
        mocker.patch('platform.system', return_value='Linux')
        mocker.patch('pathlib.Path.exists', return_value=False)
        mocker.patch.object(HardwareDetector, '_detect_cpu_model', return_value='Test CPU')
        mocker.patch.object(HardwareDetector, '_detect_cpu_features', return_value=[])
        mocker.patch.object(HardwareDetector, '_detect_gpu', return_value=(False, None, None, None))
        mocker.patch.object(HardwareDetector, '_detect_thermal_sensors', return_value=False)
        mocker.patch.object(HardwareDetector, '_detect_power_sensors', return_value=False)
        
        hw_info = HardwareDetector.detect()
        
        assert hw_info.total_ram_gb == 16.0
        assert hw_info.available_ram_gb == 8.0


class TestHardwareDetectorGPU:
    """Test GPU detection (Requirements 1.2, 1.6)."""
    
    def test_detect_gpu_nvidia(self, mocker):
        """Test NVIDIA GPU detection using pynvml."""
        # Create a mock pynvml module
        mock_pynvml = Mock()
        mock_handle = Mock()
        
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetName.return_value = b'NVIDIA GeForce RTX 3080'
        
        mock_mem_info = Mock()
        mock_mem_info.total = 10 * 1024 ** 3  # 10 GB
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem_info
        mock_pynvml.nvmlDeviceGetCudaComputeCapability.return_value = (8, 6)
        mock_pynvml.nvmlShutdown.return_value = None
        
        # Patch the import inside _detect_gpu
        mocker.patch.dict('sys.modules', {'pynvml': mock_pynvml})
        
        has_gpu, gpu_model, gpu_memory_gb, gpu_compute_capability = HardwareDetector._detect_gpu()
        
        assert has_gpu is True
        assert gpu_model == 'NVIDIA GeForce RTX 3080'
        assert gpu_memory_gb == 10.0
        assert gpu_compute_capability == '8.6'
    
    def test_detect_gpu_nvidia_string_name(self, mocker):
        """Test NVIDIA GPU detection with string name (not bytes)."""
        # Create a mock pynvml module
        mock_pynvml = Mock()
        mock_handle = Mock()
        
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetName.return_value = 'NVIDIA Jetson Xavier NX'
        
        mock_mem_info = Mock()
        mock_mem_info.total = 8 * 1024 ** 3  # 8 GB
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem_info
        mock_pynvml.nvmlDeviceGetCudaComputeCapability.return_value = (7, 2)
        mock_pynvml.nvmlShutdown.return_value = None
        
        # Patch the import inside _detect_gpu
        mocker.patch.dict('sys.modules', {'pynvml': mock_pynvml})
        
        has_gpu, gpu_model, gpu_memory_gb, gpu_compute_capability = HardwareDetector._detect_gpu()
        
        assert has_gpu is True
        assert gpu_model == 'NVIDIA Jetson Xavier NX'
        assert gpu_memory_gb == 8.0
        assert gpu_compute_capability == '7.2'
    
    def test_detect_gpu_no_devices(self, mocker):
        """Test GPU detection when no GPU devices are present."""
        # Create a mock pynvml module
        mock_pynvml = Mock()
        
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 0
        mock_pynvml.nvmlShutdown.return_value = None
        
        # Patch the import inside _detect_gpu
        mocker.patch.dict('sys.modules', {'pynvml': mock_pynvml})
        
        has_gpu, gpu_model, gpu_memory_gb, gpu_compute_capability = HardwareDetector._detect_gpu()
        
        assert has_gpu is False
        assert gpu_model is None
        assert gpu_memory_gb is None
        assert gpu_compute_capability is None
    
    def test_detect_gpu_pynvml_not_available(self, mocker):
        """Test GPU detection when pynvml is not available."""
        # Mock the import to raise ImportError
        import builtins
        real_import = builtins.__import__
        
        def mock_import(name, *args, **kwargs):
            if name == 'pynvml':
                raise ImportError("No module named 'pynvml'")
            return real_import(name, *args, **kwargs)
        
        mocker.patch('builtins.__import__', side_effect=mock_import)
        
        has_gpu, gpu_model, gpu_memory_gb, gpu_compute_capability = HardwareDetector._detect_gpu()
        
        assert has_gpu is False
        assert gpu_model is None
        assert gpu_memory_gb is None
        assert gpu_compute_capability is None
    
    def test_detect_gpu_compute_capability_error(self, mocker):
        """Test GPU detection when compute capability query fails."""
        # Create a mock pynvml module
        mock_pynvml = Mock()
        mock_handle = Mock()
        
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetName.return_value = 'NVIDIA GPU'
        
        mock_mem_info = Mock()
        mock_mem_info.total = 8 * 1024 ** 3
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem_info
        mock_pynvml.nvmlDeviceGetCudaComputeCapability.side_effect = Exception("Query failed")
        mock_pynvml.nvmlShutdown.return_value = None
        
        # Patch the import inside _detect_gpu
        mocker.patch.dict('sys.modules', {'pynvml': mock_pynvml})
        
        has_gpu, gpu_model, gpu_memory_gb, gpu_compute_capability = HardwareDetector._detect_gpu()
        
        assert has_gpu is True
        assert gpu_compute_capability == "Unknown"


class TestHardwareDetectorSensors:
    """Test sensor detection (Requirements 1.1, 1.2)."""
    
    def test_detect_thermal_sensors_present(self, mocker):
        """Test thermal sensor detection when sensors are present."""
        # Create a mock Path class
        class MockPath:
            def __init__(self, path):
                self.path_str = str(path)
            
            def exists(self):
                return self.path_str == "/sys/class/thermal"
            
            def glob(self, pattern):
                if self.path_str == "/sys/class/thermal":
                    return [
                        Path('/sys/class/thermal/thermal_zone0'),
                        Path('/sys/class/thermal/thermal_zone1')
                    ]
                return []
        
        mocker.patch('llm_benchmark.hardware.detector.Path', MockPath)
        
        has_thermal = HardwareDetector._detect_thermal_sensors()
        
        assert has_thermal is True
    
    def test_detect_thermal_sensors_absent(self, mocker):
        """Test thermal sensor detection when directory doesn't exist."""
        # Create a mock Path class
        class MockPath:
            def __init__(self, path):
                self.path_str = str(path)
            
            def exists(self):
                return False
            
            def glob(self, pattern):
                return []
        
        mocker.patch('llm_benchmark.hardware.detector.Path', MockPath)
        
        has_thermal = HardwareDetector._detect_thermal_sensors()
        
        assert has_thermal is False
    
    def test_detect_thermal_sensors_empty_directory(self, mocker):
        """Test thermal sensor detection when directory exists but is empty."""
        # Create a mock Path class
        class MockPath:
            def __init__(self, path):
                self.path_str = str(path)
            
            def exists(self):
                return self.path_str == "/sys/class/thermal"
            
            def glob(self, pattern):
                return []
        
        mocker.patch('llm_benchmark.hardware.detector.Path', MockPath)
        
        has_thermal = HardwareDetector._detect_thermal_sensors()
        
        assert has_thermal is False
    
    def test_detect_power_sensors_hwmon(self, mocker):
        """Test power sensor detection via hwmon."""
        mock_path = mocker.patch('pathlib.Path')
        
        def mock_exists(self):
            path_str = str(self)
            if path_str == "/sys/class/hwmon":
                return True
            return False
        
        mocker.patch.object(Path, 'exists', mock_exists)
        
        mock_hwmon_dir = Mock()
        mock_hwmon_dir.glob.return_value = [
            Path('/sys/class/hwmon/hwmon0'),
            Path('/sys/class/hwmon/hwmon1')
        ]
        mock_path.return_value = mock_hwmon_dir
        
        has_power = HardwareDetector._detect_power_sensors()
        
        assert has_power is True
    
    def test_detect_power_sensors_jetson_rails(self, mocker):
        """Test power sensor detection via Jetson power rails."""
        def mock_exists(self):
            path_str = str(self)
            if path_str == "/sys/class/hwmon":
                return True
            elif path_str == "/sys/bus/i2c/drivers/ina3221x":
                return True
            return False
        
        mocker.patch.object(Path, 'exists', mock_exists)
        
        mock_path = mocker.patch('pathlib.Path')
        mock_hwmon_dir = Mock()
        mock_hwmon_dir.glob.return_value = []
        mock_path.return_value = mock_hwmon_dir
        
        has_power = HardwareDetector._detect_power_sensors()
        
        assert has_power is True
    
    def test_detect_power_sensors_absent(self, mocker):
        """Test power sensor detection when no sensors are present."""
        mocker.patch.object(Path, 'exists', return_value=False)
        
        has_power = HardwareDetector._detect_power_sensors()
        
        assert has_power is False


class TestHardwareDetectorIntegration:
    """Integration tests for complete hardware detection (Requirement 1.7)."""
    
    def test_detect_complete_x86_system(self, mocker):
        """Test complete detection of x86 Linux system."""
        # Mock OS detection
        mocker.patch('platform.system', return_value='Linux')
        mocker.patch('platform.machine', return_value='x86_64')
        mocker.patch('pathlib.Path.exists', return_value=False)
        
        # Mock CPU detection
        cpuinfo_content = """processor	: 0
model name	: Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb invpcid_single pti ssbd ibrs ibpb stibp tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp md_clear flush_l1d
"""
        mock_file = mock_open(read_data=cpuinfo_content)
        mocker.patch('builtins.open', mock_file)
        
        # Mock memory detection
        mock_mem = Mock()
        mock_mem.total = 16 * 1024 ** 3
        mock_mem.available = 12 * 1024 ** 3
        mocker.patch('psutil.virtual_memory', return_value=mock_mem)
        mocker.patch('psutil.cpu_count', return_value=4)
        
        # Mock GPU detection (no GPU)
        mocker.patch.object(HardwareDetector, '_detect_gpu', return_value=(False, None, None, None))
        
        # Mock sensor detection
        mocker.patch.object(HardwareDetector, '_detect_thermal_sensors', return_value=True)
        mocker.patch.object(HardwareDetector, '_detect_power_sensors', return_value=False)
        
        hw_info = HardwareDetector.detect()
        
        # Verify all fields
        assert hw_info.os_type == "linux_x86"
        assert hw_info.cpu_model == "Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz"
        assert hw_info.cpu_cores == 4
        assert 'avx2' in hw_info.cpu_features
        assert hw_info.total_ram_gb == 16.0
        assert hw_info.available_ram_gb == 12.0
        assert hw_info.has_gpu is False
        assert hw_info.gpu_model is None
        assert hw_info.gpu_memory_gb is None
        assert hw_info.gpu_compute_capability is None
        assert hw_info.has_thermal_sensors is True
        assert hw_info.has_power_sensors is False
    
    def test_detect_complete_jetson_system(self, mocker):
        """Test complete detection of Jetson Xavier NX system."""
        # Mock OS detection
        mocker.patch('platform.system', return_value='Linux')
        mocker.patch('platform.machine', return_value='aarch64')
        
        def mock_exists(self):
            return str(self) == "/etc/nv_tegra_release"
        
        mocker.patch.object(Path, 'exists', mock_exists)
        mock_file = mock_open(read_data="# R32 (release), REVISION: 7.1, Jetson Xavier NX")
        mocker.patch('builtins.open', mock_file)
        
        # Mock CPU detection
        mocker.patch.object(HardwareDetector, '_detect_cpu_model', return_value='ARM Cortex-A57')
        mocker.patch.object(HardwareDetector, '_detect_cpu_features', return_value=['neon'])
        
        # Mock memory detection
        mock_mem = Mock()
        mock_mem.total = 8 * 1024 ** 3
        mock_mem.available = 6 * 1024 ** 3
        mocker.patch('psutil.virtual_memory', return_value=mock_mem)
        mocker.patch('psutil.cpu_count', return_value=6)
        
        # Mock GPU detection
        mocker.patch.object(HardwareDetector, '_detect_gpu', 
                          return_value=(True, 'NVIDIA Jetson Xavier NX', 8.0, '7.2'))
        
        # Mock sensor detection
        mocker.patch.object(HardwareDetector, '_detect_thermal_sensors', return_value=True)
        mocker.patch.object(HardwareDetector, '_detect_power_sensors', return_value=True)
        
        hw_info = HardwareDetector.detect()
        
        # Verify all fields
        assert hw_info.os_type == "jetson_xavier_nx"
        assert hw_info.cpu_model == 'ARM Cortex-A57'
        assert hw_info.cpu_cores == 6
        assert hw_info.total_ram_gb == 8.0
        assert hw_info.available_ram_gb == 6.0
        assert hw_info.has_gpu is True
        assert hw_info.gpu_model == 'NVIDIA Jetson Xavier NX'
        assert hw_info.gpu_memory_gb == 8.0
        assert hw_info.gpu_compute_capability == '7.2'
        assert hw_info.has_thermal_sensors is True
        assert hw_info.has_power_sensors is True
