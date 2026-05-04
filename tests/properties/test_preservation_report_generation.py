"""
Preservation Property Tests - Successful Report Generation with Absolute Paths

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

This test verifies that the existing successful report generation behavior is preserved
after the bugfix. It tests non-buggy inputs (absolute paths, relative paths, no filesystem errors)
to ensure no regressions are introduced.

**CRITICAL**: This test MUST PASS on unfixed code - passing confirms baseline behavior.
**DO NOT modify this test if it passes - it documents the behavior to preserve.**

**EXPECTED OUTCOME**: Test PASSES on unfixed code (this is correct - it proves baseline works)

The test encodes the preservation requirements - it will continue to pass after the fix,
confirming no regressions were introduced.
"""

import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock
import pytest
from hypothesis import given, strategies as st, settings, assume

from llm_benchmark.main import generate_reports
from llm_benchmark.models import BenchmarkRun, HardwareInfo
from llm_benchmark.config import BenchmarkConfig
from llm_benchmark.results.persistence import ResultsPersistence


@pytest.fixture
def minimal_benchmark_run():
    """Create a minimal benchmark run for testing."""
    hardware_info = HardwareInfo(
        os_type="linux_x86",
        cpu_model="Test CPU",
        cpu_cores=4,
        cpu_features=["avx2"],
        total_ram_gb=8.0,
        available_ram_gb=6.0,
        has_gpu=False,
        gpu_model=None,
        gpu_memory_gb=None,
        gpu_compute_capability=None,
        has_thermal_sensors=False,
        has_power_sensors=False,
    )
    
    return BenchmarkRun(
        run_id="20240115_143022",
        timestamp="2024-01-15 14:30:22",
        duration_s=10.0,
        hardware_info=hardware_info,
        software_versions={"python": "3.10.12"},
        config={},
        model_checksums={},
        quantization_results=[],
        ablation_results=[],
        batch_results=[],
        statistical_summaries=[],
        comparisons=[],
        visualization_paths=[],
        html_report_path="",
    )


@pytest.fixture
def minimal_config():
    """Create a minimal benchmark config for testing."""
    return Mock(
        output_dir="./test_output",
        save_formats=["json", "csv"],
        visualization_dpi=100
    )


class TestPreservationReportGeneration:
    """
    Property 2: Preservation - Successful Report Generation with Absolute Paths
    
    **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
    
    Tests that existing successful behavior is preserved:
    - Absolute paths without tildes work correctly
    - Relative paths work correctly
    - All format saves work when no errors occur
    - Directory structure is created correctly
    """
    
    def test_absolute_path_report_generation_succeeds(self, minimal_benchmark_run, minimal_config):
        """
        Test that generate_reports() with absolute path creates directory and saves all formats.
        
        **Preservation**: Absolute paths should continue to work exactly as before.
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (no regression).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use absolute path
            absolute_path = os.path.join(tmpdir, "benchmark_results")
            minimal_config.output_dir = absolute_path
            minimal_config.save_formats = ["json", "csv", "markdown"]
            
            # Generate reports
            generate_reports(minimal_benchmark_run, minimal_config)
            
            # Verify run directory was created
            run_dir = Path(absolute_path) / "run_20240115_143022"
            assert run_dir.exists(), f"Run directory not created at {run_dir}"
            assert run_dir.is_dir(), f"Run directory is not a directory: {run_dir}"
            
            # Verify subdirectories were created
            assert (run_dir / "visualizations").exists(), "visualizations/ subdirectory not created"
            assert (run_dir / "logs").exists(), "logs/ subdirectory not created"
            assert (run_dir / "checkpoints").exists(), "checkpoints/ subdirectory not created"
            
            # Verify all requested formats were saved
            assert (run_dir / "results.json").exists(), "JSON report not saved"
            assert (run_dir / "results.csv").exists(), "CSV report not saved"
            assert (run_dir / "results.md").exists(), "Markdown report not saved"
            
            # Verify files are not empty
            assert (run_dir / "results.json").stat().st_size > 0, "JSON report is empty"
            assert (run_dir / "results.csv").stat().st_size > 0, "CSV report is empty"
            assert (run_dir / "results.md").stat().st_size > 0, "Markdown report is empty"
    
    def test_relative_path_report_generation_succeeds(self, minimal_benchmark_run, minimal_config):
        """
        Test that generate_reports() with relative path creates directory and saves all formats.
        
        **Preservation**: Relative paths should continue to work exactly as before.
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (no regression).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Change to temp directory
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                # Use relative path
                relative_path = "./benchmark_results"
                minimal_config.output_dir = relative_path
                minimal_config.save_formats = ["json", "csv"]
                
                # Generate reports
                generate_reports(minimal_benchmark_run, minimal_config)
                
                # Verify run directory was created
                run_dir = Path(relative_path) / "run_20240115_143022"
                assert run_dir.exists(), f"Run directory not created at {run_dir}"
                
                # Verify subdirectories were created
                assert (run_dir / "visualizations").exists(), "visualizations/ subdirectory not created"
                assert (run_dir / "logs").exists(), "logs/ subdirectory not created"
                assert (run_dir / "checkpoints").exists(), "checkpoints/ subdirectory not created"
                
                # Verify requested formats were saved
                assert (run_dir / "results.json").exists(), "JSON report not saved"
                assert (run_dir / "results.csv").exists(), "CSV report not saved"
                
            finally:
                os.chdir(original_cwd)
    
    def test_all_format_saves_work_correctly(self, minimal_benchmark_run, minimal_config):
        """
        Test that all format saves (JSON, CSV, Markdown) work correctly when no errors occur.
        
        **Preservation**: All format saves should continue to work exactly as before.
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (no regression).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config.output_dir = tmpdir
            minimal_config.save_formats = ["json", "csv", "markdown"]
            
            # Generate reports
            generate_reports(minimal_benchmark_run, minimal_config)
            
            # Verify all formats were saved
            run_dir = Path(tmpdir) / "run_20240115_143022"
            
            json_path = run_dir / "results.json"
            csv_path = run_dir / "results.csv"
            markdown_path = run_dir / "results.md"
            
            assert json_path.exists(), "JSON report not saved"
            assert csv_path.exists(), "CSV report not saved"
            assert markdown_path.exists(), "Markdown report not saved"
            
            # Verify files contain expected content
            json_content = json_path.read_text()
            assert "run_id" in json_content, "JSON report missing run_id"
            assert "20240115_143022" in json_content, "JSON report missing run_id value"
            
            csv_content = csv_path.read_text()
            assert "Run ID" in csv_content, "CSV report missing Run ID header"
            assert "20240115_143022" in csv_content, "CSV report missing run_id value"
            
            markdown_content = markdown_path.read_text()
            assert "# Benchmark Run Report" in markdown_content, "Markdown report missing title"
            assert "20240115_143022" in markdown_content, "Markdown report missing run_id"
    
    def test_directory_structure_created_correctly(self, minimal_benchmark_run, minimal_config):
        """
        Test that directory structure (visualizations/, logs/, checkpoints/) is created correctly.
        
        **Preservation**: Directory structure should continue to be created exactly as before.
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (no regression).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config.output_dir = tmpdir
            minimal_config.save_formats = ["json"]
            
            # Generate reports
            generate_reports(minimal_benchmark_run, minimal_config)
            
            # Verify directory structure
            run_dir = Path(tmpdir) / "run_20240115_143022"
            
            assert run_dir.exists(), "Run directory not created"
            assert run_dir.is_dir(), "Run directory is not a directory"
            
            # Verify subdirectories
            visualizations_dir = run_dir / "visualizations"
            logs_dir = run_dir / "logs"
            checkpoints_dir = run_dir / "checkpoints"
            
            assert visualizations_dir.exists(), "visualizations/ subdirectory not created"
            assert visualizations_dir.is_dir(), "visualizations/ is not a directory"
            
            assert logs_dir.exists(), "logs/ subdirectory not created"
            assert logs_dir.is_dir(), "logs/ is not a directory"
            
            assert checkpoints_dir.exists(), "checkpoints/ subdirectory not created"
            assert checkpoints_dir.is_dir(), "checkpoints/ is not a directory"
    
    def test_persistence_init_with_absolute_path_unchanged(self):
        """
        Test that ResultsPersistence.__init__() with absolute path works correctly.
        
        **Preservation**: Absolute paths should continue to work exactly as before.
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (no regression).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            absolute_path = os.path.join(tmpdir, "benchmark_results")
            
            # Initialize ResultsPersistence with absolute path
            persistence = ResultsPersistence(output_dir=absolute_path)
            
            # Verify output_dir is set correctly
            assert persistence.output_dir == Path(absolute_path), (
                f"Absolute path not preserved. "
                f"Expected: {Path(absolute_path)}, Got: {persistence.output_dir}"
            )
            
            # Verify it's an absolute path
            assert persistence.output_dir.is_absolute(), (
                f"Output directory is not absolute: {persistence.output_dir}"
            )
    
    def test_persistence_init_with_relative_path_unchanged(self):
        """
        Test that ResultsPersistence.__init__() with relative path works correctly.
        
        **Preservation**: Relative paths should continue to work exactly as before.
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (no regression).
        """
        relative_path = "./benchmark_results"
        
        # Initialize ResultsPersistence with relative path
        persistence = ResultsPersistence(output_dir=relative_path)
        
        # Verify output_dir is set correctly
        assert persistence.output_dir == Path(relative_path), (
            f"Relative path not preserved. "
            f"Expected: {Path(relative_path)}, Got: {persistence.output_dir}"
        )
    
    @given(st.text(min_size=5, max_size=30, alphabet='abcdefghijklmnopqrstuvwxyz0123456789_-'))
    @settings(max_examples=20, deadline=None)
    def test_property_absolute_paths_without_tilde_work(self, dirname):
        """
        Property-based test: For all absolute paths without tildes, directory creation succeeds.
        
        **Preservation**: All absolute paths without tildes should continue to work.
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (no regression).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create absolute path without tilde
            absolute_path = os.path.join(tmpdir, dirname)
            
            # Ensure path doesn't contain tilde
            assume('~' not in absolute_path)
            
            # Initialize persistence
            persistence = ResultsPersistence(output_dir=absolute_path)
            
            # Create run directory
            run_dir = persistence.create_run_directory("test_run_123")
            
            # Verify directory was created
            assert run_dir.exists(), f"Run directory not created: {run_dir}"
            assert run_dir.is_dir(), f"Run directory is not a directory: {run_dir}"
            
            # Verify subdirectories were created
            assert (run_dir / "visualizations").exists(), "visualizations/ not created"
            assert (run_dir / "logs").exists(), "logs/ not created"
            assert (run_dir / "checkpoints").exists(), "checkpoints/ not created"
            
            # Clean up
            shutil.rmtree(run_dir)
    
    @given(st.text(min_size=5, max_size=30, alphabet='abcdefghijklmnopqrstuvwxyz0123456789_-'))
    @settings(max_examples=20, deadline=None)
    def test_property_relative_paths_work(self, dirname):
        """
        Property-based test: For all relative paths, directory creation succeeds.
        
        **Preservation**: All relative paths should continue to work.
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (no regression).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                # Create relative path
                relative_path = f"./{dirname}"
                
                # Ensure path doesn't contain tilde
                assume('~' not in relative_path)
                
                # Initialize persistence
                persistence = ResultsPersistence(output_dir=relative_path)
                
                # Create run directory
                run_dir = persistence.create_run_directory("test_run_456")
                
                # Verify directory was created
                assert run_dir.exists(), f"Run directory not created: {run_dir}"
                assert run_dir.is_dir(), f"Run directory is not a directory: {run_dir}"
                
                # Verify subdirectories were created
                assert (run_dir / "visualizations").exists(), "visualizations/ not created"
                assert (run_dir / "logs").exists(), "logs/ not created"
                assert (run_dir / "checkpoints").exists(), "checkpoints/ not created"
                
            finally:
                os.chdir(original_cwd)
    
    @given(
        formats=st.lists(
            st.sampled_from(["json", "csv", "markdown"]),
            min_size=1,
            max_size=3,
            unique=True
        )
    )
    @settings(max_examples=10, deadline=None)
    def test_property_all_requested_formats_saved(self, formats):
        """
        Property-based test: For all requested formats, files are saved when no errors occur.
        
        **Preservation**: All format combinations should continue to work.
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (no regression).
        """
        # Create minimal benchmark run inline
        hardware_info = HardwareInfo(
            os_type="linux_x86",
            cpu_model="Test CPU",
            cpu_cores=4,
            cpu_features=["avx2"],
            total_ram_gb=8.0,
            available_ram_gb=6.0,
            has_gpu=False,
            gpu_model=None,
            gpu_memory_gb=None,
            gpu_compute_capability=None,
            has_thermal_sensors=False,
            has_power_sensors=False,
        )
        
        minimal_benchmark_run = BenchmarkRun(
            run_id="20240115_143022",
            timestamp="2024-01-15 14:30:22",
            duration_s=10.0,
            hardware_info=hardware_info,
            software_versions={"python": "3.10.12"},
            config={},
            model_checksums={},
            quantization_results=[],
            ablation_results=[],
            batch_results=[],
            statistical_summaries=[],
            comparisons=[],
            visualization_paths=[],
            html_report_path="",
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Mock(
                output_dir=tmpdir,
                save_formats=formats,
                visualization_dpi=100
            )
            
            # Generate reports
            generate_reports(minimal_benchmark_run, config)
            
            # Verify all requested formats were saved
            run_dir = Path(tmpdir) / "run_20240115_143022"
            
            format_files = {
                "json": "results.json",
                "csv": "results.csv",
                "markdown": "results.md"
            }
            
            for format_type in formats:
                file_path = run_dir / format_files[format_type]
                assert file_path.exists(), f"{format_type.upper()} report not saved: {file_path}"
                assert file_path.stat().st_size > 0, f"{format_type.upper()} report is empty"
    
    @given(st.text(min_size=10, max_size=20, alphabet='abcdefghijklmnopqrstuvwxyz0123456789_'))
    @settings(max_examples=15, deadline=None)
    def test_property_subdirectories_created_for_all_runs(self, run_id):
        """
        Property-based test: For all runs, subdirectories are created correctly.
        
        **Preservation**: Subdirectory creation should continue to work for all run IDs.
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (no regression).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize persistence
            persistence = ResultsPersistence(output_dir=tmpdir)
            
            # Create run directory
            run_dir = persistence.create_run_directory(run_id)
            
            # Verify run directory was created
            assert run_dir.exists(), f"Run directory not created: {run_dir}"
            assert run_dir.name == f"run_{run_id}", f"Run directory has wrong name: {run_dir.name}"
            
            # Verify all subdirectories were created
            subdirs = ["visualizations", "logs", "checkpoints"]
            for subdir in subdirs:
                subdir_path = run_dir / subdir
                assert subdir_path.exists(), f"{subdir}/ subdirectory not created"
                assert subdir_path.is_dir(), f"{subdir}/ is not a directory"
            
            # Clean up
            shutil.rmtree(run_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
