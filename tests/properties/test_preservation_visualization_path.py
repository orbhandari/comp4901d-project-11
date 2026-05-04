"""
Preservation Property Tests - Other Subdirectories Still Created

**Validates: Requirements 3.1, 3.2, 3.3**

This test verifies preservation of existing directory creation behavior that must remain
unchanged after the fix. The fix should ONLY remove the per-run visualizations/ subdirectory
creation, while preserving all other directory creation operations.

**CRITICAL**: These tests MUST PASS on unfixed code - passing confirms baseline behavior.
**EXPECTED OUTCOME**: Tests PASS on unfixed code (confirming behavior to preserve).
**EXPECTED OUTCOME**: Tests PASS after fix (confirming behavior is preserved).

These tests follow the observation-first methodology: observe behavior on unfixed code,
then write tests capturing that behavior (minus the bug).
"""

import tempfile
from pathlib import Path
import pytest
from hypothesis import given, strategies as st, settings, assume

from llm_benchmark.results.persistence import ResultsPersistence


# Strategy for generating valid run_id strings
# Matches typical timestamp format and custom strings used in the codebase
run_id_strategy = st.one_of(
    # Timestamp format: YYYYMMDD_HHMMSS
    st.from_regex(r'^[0-9]{8}_[0-9]{6}$', fullmatch=True),
    # Custom alphanumeric strings with underscores and hyphens
    st.text(min_size=1, max_size=50, alphabet='abcdefghijklmnopqrstuvwxyz0123456789_-'),
    # Short identifiers
    st.text(min_size=1, max_size=10, alphabet='abcdefghijklmnopqrstuvwxyz0123456789'),
)


class TestPreservationVisualizationPath:
    """
    Property 2: Preservation - Other Subdirectories Still Created
    
    **Validates: Requirements 3.1, 3.2, 3.3**
    
    Tests that the fix preserves existing directory creation behavior:
    - Main run_<run_id>/ directory is created
    - logs/ subdirectory is created
    - checkpoints/ subdirectory is created
    - Method returns correct Path object
    - Error handling is preserved
    """
    
    def test_main_run_directory_created(self):
        """
        Test that create_run_directory() creates the main run_<run_id>/ directory.
        
        **Preservation**: Main run directory must always be created.
        **Current Behavior (unfixed)**: Directory is created correctly.
        
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (behavior preserved).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            run_id = "20240115_143022"
            
            # Create run directory
            run_dir = persistence.create_run_directory(run_id)
            
            # Verify main run directory exists
            assert run_dir.exists(), (
                f"Main run directory not created: {run_dir}. "
                f"This is a regression - the main run directory must always be created."
            )
            
            # Verify it's a directory
            assert run_dir.is_dir(), f"Run path exists but is not a directory: {run_dir}"
            
            # Verify directory name format
            assert run_dir.name == f"run_{run_id}", (
                f"Run directory has incorrect name: {run_dir.name}. "
                f"Expected: run_{run_id}"
            )
    
    def test_logs_subdirectory_created(self):
        """
        Test that create_run_directory() creates the logs/ subdirectory.
        
        **Preservation**: logs/ subdirectory must always be created.
        **Current Behavior (unfixed)**: Subdirectory is created correctly.
        
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (behavior preserved).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            run_id = "20240115_143022"
            
            # Create run directory
            run_dir = persistence.create_run_directory(run_id)
            
            # Verify logs/ subdirectory exists
            logs_subdir = run_dir / "logs"
            assert logs_subdir.exists(), (
                f"logs/ subdirectory not created: {logs_subdir}. "
                f"This is a regression - logs/ subdirectory must always be created."
            )
            
            # Verify it's a directory
            assert logs_subdir.is_dir(), f"logs/ exists but is not a directory: {logs_subdir}"
    
    def test_checkpoints_subdirectory_created(self):
        """
        Test that create_run_directory() creates the checkpoints/ subdirectory.
        
        **Preservation**: checkpoints/ subdirectory must always be created.
        **Current Behavior (unfixed)**: Subdirectory is created correctly.
        
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (behavior preserved).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            run_id = "20240115_143022"
            
            # Create run directory
            run_dir = persistence.create_run_directory(run_id)
            
            # Verify checkpoints/ subdirectory exists
            checkpoints_subdir = run_dir / "checkpoints"
            assert checkpoints_subdir.exists(), (
                f"checkpoints/ subdirectory not created: {checkpoints_subdir}. "
                f"This is a regression - checkpoints/ subdirectory must always be created."
            )
            
            # Verify it's a directory
            assert checkpoints_subdir.is_dir(), (
                f"checkpoints/ exists but is not a directory: {checkpoints_subdir}"
            )
    
    def test_method_returns_correct_path_object(self):
        """
        Test that create_run_directory() returns the correct Path object.
        
        **Preservation**: Method must return Path object for the run directory.
        **Current Behavior (unfixed)**: Returns correct Path object.
        
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (behavior preserved).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            run_id = "20240115_143022"
            
            # Create run directory
            run_dir = persistence.create_run_directory(run_id)
            
            # Verify return type is Path
            assert isinstance(run_dir, Path), (
                f"create_run_directory() returned wrong type: {type(run_dir)}. "
                f"Expected: pathlib.Path"
            )
            
            # Verify returned path points to the run directory
            expected_path = Path(tmpdir) / f"run_{run_id}"
            assert run_dir == expected_path, (
                f"create_run_directory() returned incorrect path: {run_dir}. "
                f"Expected: {expected_path}"
            )
    
    def test_error_handling_when_directory_creation_fails(self):
        """
        Test that create_run_directory() raises OSError when directory creation fails.
        
        **Preservation**: Error handling must be preserved.
        **Current Behavior (unfixed)**: Raises OSError with descriptive message.
        
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (behavior preserved).
        """
        # Use an invalid path that will cause directory creation to fail
        # On Unix-like systems, /dev/null is a special file, not a directory
        invalid_output_dir = "/dev/null/invalid_path"
        
        persistence = ResultsPersistence(output_dir=invalid_output_dir)
        run_id = "20240115_143022"
        
        # Verify OSError is raised
        with pytest.raises(OSError) as exc_info:
            persistence.create_run_directory(run_id)
        
        # Verify error message is descriptive
        error_message = str(exc_info.value)
        assert "Failed to create" in error_message or "Permission denied" in error_message or "Not a directory" in error_message, (
            f"OSError raised but with unexpected message: {error_message}. "
            f"Expected descriptive error message about directory creation failure."
        )
    
    def test_all_subdirectories_created_together(self):
        """
        Test that all required subdirectories are created in a single call.
        
        **Preservation**: All subdirectories (logs/, checkpoints/) must be created together.
        **Current Behavior (unfixed)**: All subdirectories are created.
        
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (behavior preserved).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            run_id = "20240115_143022"
            
            # Create run directory
            run_dir = persistence.create_run_directory(run_id)
            
            # Verify all required subdirectories exist
            required_subdirs = ["logs", "checkpoints"]
            
            for subdir_name in required_subdirs:
                subdir_path = run_dir / subdir_name
                assert subdir_path.exists(), (
                    f"Required subdirectory not created: {subdir_name}. "
                    f"Path: {subdir_path}"
                )
                assert subdir_path.is_dir(), (
                    f"Required subdirectory exists but is not a directory: {subdir_name}. "
                    f"Path: {subdir_path}"
                )
    
    def test_multiple_runs_create_separate_directories(self):
        """
        Test that multiple runs create separate run directories with their own subdirectories.
        
        **Preservation**: Each run must have its own isolated directory structure.
        **Current Behavior (unfixed)**: Separate directories are created correctly.
        
        **Expected on unfixed code**: This test PASSES (baseline behavior).
        **Expected after fix**: This test PASSES (behavior preserved).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            
            # Create multiple run directories
            run_ids = ["run_001", "run_002", "run_003"]
            run_dirs = []
            
            for run_id in run_ids:
                run_dir = persistence.create_run_directory(run_id)
                run_dirs.append(run_dir)
            
            # Verify all run directories exist and are separate
            for i, (run_dir, run_id) in enumerate(zip(run_dirs, run_ids)):
                # Verify main directory exists
                assert run_dir.exists(), f"Run directory not created for {run_id}: {run_dir}"
                
                # Verify logs/ subdirectory exists
                logs_subdir = run_dir / "logs"
                assert logs_subdir.exists(), (
                    f"logs/ subdirectory not created for {run_id}: {logs_subdir}"
                )
                
                # Verify checkpoints/ subdirectory exists
                checkpoints_subdir = run_dir / "checkpoints"
                assert checkpoints_subdir.exists(), (
                    f"checkpoints/ subdirectory not created for {run_id}: {checkpoints_subdir}"
                )
                
                # Verify directories are separate (not the same path)
                for j, other_run_dir in enumerate(run_dirs):
                    if i != j:
                        assert run_dir != other_run_dir, (
                            f"Run directories are not separate: {run_dir} == {other_run_dir}"
                        )
    
    @given(run_id_strategy)
    @settings(max_examples=50, deadline=None)
    def test_property_main_run_directory_created_for_any_run_id(self, run_id):
        """
        Property-based test: Main run directory is created for any valid run_id.
        
        **Preservation**: For any valid run_id, the main run directory must be created.
        **Current Behavior (unfixed)**: Directory is created correctly for all inputs.
        
        **Expected on unfixed code**: This test PASSES (baseline behavior verified).
        **Expected after fix**: This test PASSES (behavior preserved for all inputs).
        """
        # Skip empty or whitespace-only run_ids
        assume(run_id.strip() != "")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            
            # Create run directory with generated run_id
            run_dir = persistence.create_run_directory(run_id)
            
            # Verify main run directory exists
            assert run_dir.exists(), (
                f"Main run directory not created for run_id '{run_id}': {run_dir}. "
                f"This is a regression - the main run directory must always be created."
            )
            
            # Verify it's a directory
            assert run_dir.is_dir(), (
                f"Run path exists but is not a directory for run_id '{run_id}': {run_dir}"
            )
            
            # Verify directory name format
            assert run_dir.name == f"run_{run_id}", (
                f"Run directory has incorrect name for run_id '{run_id}': {run_dir.name}. "
                f"Expected: run_{run_id}"
            )
    
    @given(run_id_strategy)
    @settings(max_examples=50, deadline=None)
    def test_property_logs_subdirectory_created_for_any_run_id(self, run_id):
        """
        Property-based test: logs/ subdirectory is created for any valid run_id.
        
        **Preservation**: For any valid run_id, the logs/ subdirectory must be created.
        **Current Behavior (unfixed)**: Subdirectory is created correctly for all inputs.
        
        **Expected on unfixed code**: This test PASSES (baseline behavior verified).
        **Expected after fix**: This test PASSES (behavior preserved for all inputs).
        """
        # Skip empty or whitespace-only run_ids
        assume(run_id.strip() != "")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            
            # Create run directory with generated run_id
            run_dir = persistence.create_run_directory(run_id)
            
            # Verify logs/ subdirectory exists
            logs_subdir = run_dir / "logs"
            assert logs_subdir.exists(), (
                f"logs/ subdirectory not created for run_id '{run_id}': {logs_subdir}. "
                f"This is a regression - logs/ subdirectory must always be created."
            )
            
            # Verify it's a directory
            assert logs_subdir.is_dir(), (
                f"logs/ exists but is not a directory for run_id '{run_id}': {logs_subdir}"
            )
    
    @given(run_id_strategy)
    @settings(max_examples=50, deadline=None)
    def test_property_checkpoints_subdirectory_created_for_any_run_id(self, run_id):
        """
        Property-based test: checkpoints/ subdirectory is created for any valid run_id.
        
        **Preservation**: For any valid run_id, the checkpoints/ subdirectory must be created.
        **Current Behavior (unfixed)**: Subdirectory is created correctly for all inputs.
        
        **Expected on unfixed code**: This test PASSES (baseline behavior verified).
        **Expected after fix**: This test PASSES (behavior preserved for all inputs).
        """
        # Skip empty or whitespace-only run_ids
        assume(run_id.strip() != "")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            
            # Create run directory with generated run_id
            run_dir = persistence.create_run_directory(run_id)
            
            # Verify checkpoints/ subdirectory exists
            checkpoints_subdir = run_dir / "checkpoints"
            assert checkpoints_subdir.exists(), (
                f"checkpoints/ subdirectory not created for run_id '{run_id}': {checkpoints_subdir}. "
                f"This is a regression - checkpoints/ subdirectory must always be created."
            )
            
            # Verify it's a directory
            assert checkpoints_subdir.is_dir(), (
                f"checkpoints/ exists but is not a directory for run_id '{run_id}': {checkpoints_subdir}"
            )
    
    @given(run_id_strategy)
    @settings(max_examples=50, deadline=None)
    def test_property_method_returns_correct_path_for_any_run_id(self, run_id):
        """
        Property-based test: Method returns correct Path object for any valid run_id.
        
        **Preservation**: For any valid run_id, the method must return the correct Path object.
        **Current Behavior (unfixed)**: Returns correct Path for all inputs.
        
        **Expected on unfixed code**: This test PASSES (baseline behavior verified).
        **Expected after fix**: This test PASSES (behavior preserved for all inputs).
        """
        # Skip empty or whitespace-only run_ids
        assume(run_id.strip() != "")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            
            # Create run directory with generated run_id
            run_dir = persistence.create_run_directory(run_id)
            
            # Verify return type is Path
            assert isinstance(run_dir, Path), (
                f"create_run_directory() returned wrong type for run_id '{run_id}': {type(run_dir)}. "
                f"Expected: pathlib.Path"
            )
            
            # Verify returned path points to the run directory
            expected_path = Path(tmpdir) / f"run_{run_id}"
            assert run_dir == expected_path, (
                f"create_run_directory() returned incorrect path for run_id '{run_id}': {run_dir}. "
                f"Expected: {expected_path}"
            )
    
    @given(run_id_strategy)
    @settings(max_examples=50, deadline=None)
    def test_property_all_required_subdirectories_created_for_any_run_id(self, run_id):
        """
        Property-based test: All required subdirectories are created for any valid run_id.
        
        **Preservation**: For any valid run_id, all required subdirectories must be created.
        **Current Behavior (unfixed)**: All subdirectories are created correctly for all inputs.
        
        **Expected on unfixed code**: This test PASSES (baseline behavior verified).
        **Expected after fix**: This test PASSES (behavior preserved for all inputs).
        """
        # Skip empty or whitespace-only run_ids
        assume(run_id.strip() != "")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            
            # Create run directory with generated run_id
            run_dir = persistence.create_run_directory(run_id)
            
            # Verify all required subdirectories exist
            required_subdirs = ["logs", "checkpoints"]
            
            for subdir_name in required_subdirs:
                subdir_path = run_dir / subdir_name
                assert subdir_path.exists(), (
                    f"Required subdirectory not created for run_id '{run_id}': {subdir_name}. "
                    f"Path: {subdir_path}"
                )
                assert subdir_path.is_dir(), (
                    f"Required subdirectory exists but is not a directory for run_id '{run_id}': {subdir_name}. "
                    f"Path: {subdir_path}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
