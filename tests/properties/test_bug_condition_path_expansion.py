"""
Bug Condition Exploration Test - Path Expansion and Error Propagation Failures

**Validates: Requirements 1.1, 1.2, 1.3, 1.4**

This test explores the bug condition where generate_reports() fails to:
1. Expand tilde paths (~) to absolute paths before directory creation
2. Propagate errors with clear, actionable messages
3. Track and report individual format save failures

**CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists.
**DO NOT attempt to fix the test or the code when it fails.**

**EXPECTED OUTCOME**: Test FAILS (this is correct - it proves the bug exists)

The test encodes the expected behavior - it will validate the fix when it passes after implementation.
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck

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


class TestBugConditionPathExpansion:
    """
    Property 1: Bug Condition - Path Expansion and Error Propagation Failures
    
    **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
    
    Tests that the bug exists in unfixed code:
    - Tilde paths are not expanded, causing directory creation to fail
    - Errors are caught silently without propagation
    - Individual format failures are not tracked or reported
    """
    
    def test_tilde_path_not_expanded_in_persistence_init(self):
        """
        Test that ResultsPersistence.__init__() does NOT expand tilde paths.
        
        **Bug Condition**: When output_dir contains '~', it should be expanded to absolute path.
        **Current Behavior**: Path('~') creates literal '~' directory instead of expanding.
        
        **Expected on unfixed code**: This test FAILS because tilde is not expanded.
        **Expected after fix**: This test PASSES because tilde is expanded.
        """
        # Create a tilde path
        tilde_path = "~/test_benchmark_results"
        
        # Initialize ResultsPersistence with tilde path
        persistence = ResultsPersistence(output_dir=tilde_path)
        
        # BUG: The output_dir should be expanded, but it's not
        # After fix: persistence.output_dir should equal Path.home() / "test_benchmark_results"
        # Before fix: persistence.output_dir equals Path("~/test_benchmark_results")
        
        expected_expanded = Path.home() / "test_benchmark_results"
        
        # This assertion SHOULD FAIL on unfixed code
        assert persistence.output_dir == expected_expanded, (
            f"Tilde path not expanded! "
            f"Expected: {expected_expanded}, "
            f"Got: {persistence.output_dir}. "
            f"Bug confirmed: Path('~') creates literal '~' directory instead of expanding."
        )
    
    def test_generate_reports_with_tilde_path_succeeds_after_expansion(self, minimal_benchmark_run, minimal_config):
        """
        Test that generate_reports() with tilde path expands correctly and succeeds.
        
        **Bug Condition**: When output_dir contains '~', path should be expanded before directory creation.
        **Current Behavior (unfixed)**: Path not expanded, creates literal '~' directory or fails.
        
        **Expected on unfixed code**: This test FAILS because path is not expanded.
        **Expected after fix**: This test PASSES because path is expanded and directory is created.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a tilde path that should be expanded
            # Note: We can't use ~/nonexistent because mkdir(parents=True) will create it
            # Instead, verify that the path is expanded correctly
            minimal_config.output_dir = "~/test_benchmark_results_expansion_test"
            
            # After fix: generate_reports() should expand the path and create directory successfully
            # Before fix: Would create literal '~' directory or fail silently
            
            try:
                generate_reports(minimal_benchmark_run, minimal_config)
                
                # After fix: Directory should be created under home directory
                expected_dir = Path.home() / "test_benchmark_results_expansion_test" / "run_20240115_143022"
                
                assert expected_dir.exists(), (
                    f"Directory not created at expanded path. Expected: {expected_dir}"
                )
                
                # Verify it's an absolute path under home directory
                assert expected_dir.is_absolute(), "Run directory is not absolute"
                assert str(expected_dir).startswith(str(Path.home())), (
                    f"Run directory not under home. Expected to start with {Path.home()}, got {expected_dir}"
                )
                
                # Clean up
                shutil.rmtree(Path.home() / "test_benchmark_results_expansion_test")
                
            except Exception as e:
                # If exception is raised, it should be informative
                pytest.fail(f"generate_reports() raised unexpected exception: {e}")
    
    def test_create_run_directory_with_tilde_path_creates_literal_tilde_dir(self):
        """
        Test that create_run_directory() with tilde path attempts to create literal '~' directory.
        
        **Bug Condition**: Path('~').mkdir() creates literal '~' directory in current working directory.
        **Current Behavior**: No path expansion before mkdir().
        
        **Expected on unfixed code**: This test FAILS because literal '~' directory is created.
        **Expected after fix**: This test PASSES because path is expanded before mkdir().
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Change to temp directory to avoid polluting workspace
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                # Create persistence with tilde path
                tilde_path = "~/test_benchmark_results"
                persistence = ResultsPersistence(output_dir=tilde_path)
                
                # Try to create run directory
                # BUG: This will attempt to create a literal '~' directory
                try:
                    run_dir = persistence.create_run_directory("20240115_143022")
                    
                    # After fix: run_dir should be under home directory
                    # Before fix: run_dir is under current directory with literal '~'
                    
                    # Check if literal '~' directory was created (bug behavior)
                    literal_tilde_dir = Path(tmpdir) / "~"
                    
                    # This assertion SHOULD FAIL on unfixed code (literal ~ dir exists)
                    assert not literal_tilde_dir.exists(), (
                        f"Bug confirmed: Literal '~' directory created at {literal_tilde_dir}. "
                        f"Path expansion not performed before mkdir()."
                    )
                    
                    # After fix: run_dir should be absolute path under home directory
                    assert run_dir.is_absolute(), (
                        f"Bug: run_dir is not absolute path. Got: {run_dir}"
                    )
                    
                    # After fix: run_dir should start with home directory
                    assert str(run_dir).startswith(str(Path.home())), (
                        f"Bug: run_dir not under home directory. "
                        f"Expected to start with {Path.home()}, got {run_dir}"
                    )
                    
                except Exception as e:
                    # If exception is raised, check if it's informative
                    error_msg = str(e).lower()
                    
                    # After fix: error should mention path expansion or provide clear guidance
                    assert any(keyword in error_msg for keyword in ["expand", "tilde", "~", "absolute"]), (
                        f"Exception raised but not informative about path expansion issue: {e}"
                    )
                    
            finally:
                os.chdir(original_cwd)
    
    def test_permission_denied_error_not_propagated(self, minimal_benchmark_run, minimal_config):
        """
        Test that permission denied errors are caught silently without propagation.
        
        **Bug Condition**: When directory creation fails due to permissions, error is caught.
        **Current Behavior**: Exception caught, logged, but not propagated to caller.
        
        **Expected on unfixed code**: This test FAILS because no exception is raised.
        **Expected after fix**: This test PASSES because exception is propagated with clear message.
        """
        # Use a path that will fail due to permissions (if running as non-root)
        # /root is typically not writable by non-root users
        minimal_config.output_dir = "/root/benchmark_results_test_12345"
        
        # Skip test if running as root (permission test won't work)
        if os.geteuid() == 0:
            pytest.skip("Running as root, permission test not applicable")
        
        # BUG: generate_reports() should raise an exception with clear error message
        # Current behavior: Catches exception, logs warning, returns None
        
        # This should raise an exception (OSError with permission denied)
        # On unfixed code, it will NOT raise an exception
        with pytest.raises(Exception) as exc_info:
            generate_reports(minimal_benchmark_run, minimal_config)
        
        # After fix, the exception message should mention permission denied
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["permission", "denied", "access"]), (
            f"Exception message does not mention permission issue: {exc_info.value}"
        )
    
    def test_individual_format_failures_not_tracked(self, minimal_benchmark_run, minimal_config):
        """
        Test that individual format save failures are not tracked or reported.
        
        **Bug Condition**: When one format fails, no tracking of which formats succeeded/failed.
        **Current Behavior**: Exception caught at top level, no granular failure tracking.
        
        **Expected on unfixed code**: This test FAILS because failures are not tracked.
        **Expected after fix**: This test PASSES because failures are tracked and reported.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config.output_dir = tmpdir
            minimal_config.save_formats = ["json", "csv", "markdown"]
            
            # Mock one of the save methods to fail
            with patch.object(ResultsPersistence, 'save_csv', side_effect=Exception("CSV save failed")):
                # BUG: After fix, this should either:
                # 1. Continue saving other formats and report CSV failure, OR
                # 2. Raise exception listing all failures
                #
                # Current behavior: Entire operation fails, no indication of which format failed
                
                try:
                    generate_reports(minimal_benchmark_run, minimal_config)
                    
                    # After fix: JSON and Markdown should be saved despite CSV failure
                    json_path = Path(tmpdir) / "run_20240115_143022" / "results.json"
                    markdown_path = Path(tmpdir) / "run_20240115_143022" / "results.md"
                    
                    # This assertion SHOULD FAIL on unfixed code (files not created)
                    assert json_path.exists(), (
                        f"Bug confirmed: JSON not saved when CSV fails. "
                        f"Individual format failures not handled gracefully."
                    )
                    
                    assert markdown_path.exists(), (
                        f"Bug confirmed: Markdown not saved when CSV fails. "
                        f"Individual format failures not handled gracefully."
                    )
                    
                except Exception as e:
                    # If exception is raised, it should list which formats failed
                    error_msg = str(e).lower()
                    
                    # After fix: error should mention specific format that failed
                    assert "csv" in error_msg, (
                        f"Exception does not specify which format failed: {e}"
                    )
    
    @given(st.text(min_size=1, max_size=50).filter(lambda x: '~' in x and not x.startswith('~/')))
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
    def test_property_tilde_in_middle_of_path_not_expanded(self, path_with_tilde):
        """
        Property-based test: Tilde in middle of path is not expanded.
        
        **Bug Condition**: Only leading '~' should be expanded, but code doesn't handle this correctly.
        **Current Behavior**: Path() doesn't expand tilde at all without .expanduser().
        
        **Expected on unfixed code**: This test FAILS because tilde is not expanded.
        **Expected after fix**: This test PASSES because .expanduser() is called.
        """
        # Ensure path doesn't start with ~ (only leading ~ should be expanded)
        assume(not path_with_tilde.startswith('~'))
        
        # Create persistence with path containing tilde
        persistence = ResultsPersistence(output_dir=path_with_tilde)
        
        # After fix: output_dir should have tilde expanded (only if at start)
        # Before fix: output_dir is literal path with tilde
        
        # For paths with ~ in middle, expanduser() should not change them
        expected = Path(path_with_tilde).expanduser()
        
        # This assertion SHOULD FAIL on unfixed code if ~ is at start
        assert persistence.output_dir == expected, (
            f"Path expansion not performed. "
            f"Expected: {expected}, Got: {persistence.output_dir}"
        )
    
    @given(st.text(min_size=5, max_size=30, alphabet='abcdefghijklmnopqrstuvwxyz0123456789_-'))
    @settings(max_examples=20, deadline=None)
    def test_property_tilde_prefix_paths_not_expanded(self, dirname):
        """
        Property-based test: Paths starting with '~/' are not expanded to absolute paths.
        
        **Bug Condition**: '~/dirname' should expand to '/home/user/dirname'.
        **Current Behavior**: Path('~/dirname') creates literal '~/dirname' path.
        
        **Expected on unfixed code**: This test FAILS because tilde is not expanded.
        **Expected after fix**: This test PASSES because .expanduser() is called.
        """
        # Create tilde-prefixed path
        tilde_path = f"~/{dirname}"
        
        # Initialize persistence
        persistence = ResultsPersistence(output_dir=tilde_path)
        
        # After fix: should be expanded to absolute path under home directory
        expected_expanded = Path.home() / dirname
        
        # This assertion SHOULD FAIL on unfixed code
        assert persistence.output_dir == expected_expanded, (
            f"Tilde path not expanded! "
            f"Expected: {expected_expanded}, "
            f"Got: {persistence.output_dir}. "
            f"Bug confirmed: Path('~/{dirname}') not expanded to absolute path."
        )
        
        # After fix: output_dir should be absolute
        assert persistence.output_dir.is_absolute(), (
            f"Bug: output_dir is not absolute. Got: {persistence.output_dir}"
        )
        
        # After fix: output_dir should start with home directory
        assert str(persistence.output_dir).startswith(str(Path.home())), (
            f"Bug: output_dir not under home directory. "
            f"Expected to start with {Path.home()}, got {persistence.output_dir}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
