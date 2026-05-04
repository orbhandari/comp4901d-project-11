"""
Bug Condition Exploration Test - Per-Run Visualizations Subdirectory Created

**Validates: Requirements 1.1, 1.2, 1.3**

This test explores the bug condition where create_run_directory() creates an unnecessary
per-run visualizations/ subdirectory that is never used. The VisualizationGenerator class
correctly saves PNG files to the top-level benchmark_results/visualizations/ directory,
but the per-run subdirectory still exists, creating confusion.

**CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists.
**DO NOT attempt to fix the test or the code when it fails.**

**EXPECTED OUTCOME**: Test FAILS (this is correct - it proves the bug exists)

The test encodes the expected behavior - it will validate the fix when it passes after implementation.
"""

import tempfile
from pathlib import Path
import pytest
from hypothesis import given, strategies as st, settings

from llm_benchmark.results.persistence import ResultsPersistence


class TestBugConditionVisualizationPath:
    """
    Property 1: Bug Condition - Per-Run Visualizations Subdirectory Created
    
    **Validates: Requirements 1.1, 1.2, 1.3**
    
    Tests that the bug exists in unfixed code:
    - create_run_directory() creates an unnecessary run_<timestamp>/visualizations/ subdirectory
    - This subdirectory is never used (VisualizationGenerator saves to top-level visualizations/)
    - Multiple runs create multiple empty per-run visualizations/ subdirectories
    """
    
    def test_per_run_visualizations_subdirectory_created(self):
        """
        Test that create_run_directory() creates a per-run visualizations/ subdirectory.
        
        **Bug Condition**: When create_run_directory() is called, it creates run_<id>/visualizations/.
        **Current Behavior (unfixed)**: Subdirectory is created but never used.
        
        **Expected on unfixed code**: This test PASSES (confirming bug exists).
        **Expected after fix**: This test FAILS (subdirectory no longer created).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            run_id = "test_run_001"
            
            # Create run directory
            run_dir = persistence.create_run_directory(run_id)
            
            # BUG: The per-run visualizations/ subdirectory should NOT be created
            # Current behavior: (run_dir / "visualizations").exists() == True
            # After fix: (run_dir / "visualizations").exists() == False
            
            visualizations_subdir = run_dir / "visualizations"
            
            # This assertion SHOULD PASS on unfixed code (bug exists)
            # This assertion SHOULD FAIL after fix (bug is fixed)
            assert visualizations_subdir.exists(), (
                f"Bug NOT confirmed: Per-run visualizations/ subdirectory was not created. "
                f"Expected {visualizations_subdir} to exist (bug behavior), but it doesn't. "
                f"This means the bug may already be fixed, or the test is running on fixed code."
            )
            
            # Additional verification: subdirectory should be a directory
            assert visualizations_subdir.is_dir(), (
                f"Per-run visualizations/ exists but is not a directory: {visualizations_subdir}"
            )
    
    def test_multiple_runs_create_multiple_empty_visualizations_subdirs(self):
        """
        Test that multiple benchmark runs create multiple empty per-run visualizations/ subdirectories.
        
        **Bug Condition**: Each run creates its own visualizations/ subdirectory.
        **Current Behavior (unfixed)**: Multiple empty subdirectories are created.
        
        **Expected on unfixed code**: This test PASSES (confirming bug exists).
        **Expected after fix**: This test FAILS (subdirectories no longer created).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            
            # Create multiple run directories
            run_ids = ["run_001", "run_002", "run_003"]
            run_dirs = []
            
            for run_id in run_ids:
                run_dir = persistence.create_run_directory(run_id)
                run_dirs.append(run_dir)
            
            # BUG: Each run should NOT have its own visualizations/ subdirectory
            # Current behavior: All runs have visualizations/ subdirectories
            # After fix: No runs have visualizations/ subdirectories
            
            for run_dir, run_id in zip(run_dirs, run_ids):
                visualizations_subdir = run_dir / "visualizations"
                
                # This assertion SHOULD PASS on unfixed code (bug exists)
                # This assertion SHOULD FAIL after fix (bug is fixed)
                assert visualizations_subdir.exists(), (
                    f"Bug NOT confirmed for run {run_id}: "
                    f"Per-run visualizations/ subdirectory was not created. "
                    f"Expected {visualizations_subdir} to exist (bug behavior), but it doesn't."
                )
                
                # Verify subdirectories are empty (never used)
                assert list(visualizations_subdir.iterdir()) == [], (
                    f"Per-run visualizations/ subdirectory is not empty: {visualizations_subdir}. "
                    f"Expected empty directory (bug behavior)."
                )
    
    def test_visualizations_subdirectory_is_empty_after_creation(self):
        """
        Test that the per-run visualizations/ subdirectory is empty after creation.
        
        **Bug Condition**: Subdirectory is created but never used (remains empty).
        **Current Behavior (unfixed)**: Empty subdirectory exists.
        
        **Expected on unfixed code**: This test PASSES (confirming bug exists).
        **Expected after fix**: This test FAILS (subdirectory no longer created).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            run_id = "test_run_empty_check"
            
            # Create run directory
            run_dir = persistence.create_run_directory(run_id)
            
            # BUG: The per-run visualizations/ subdirectory should NOT exist
            visualizations_subdir = run_dir / "visualizations"
            
            # This assertion SHOULD PASS on unfixed code (bug exists)
            # This assertion SHOULD FAIL after fix (bug is fixed)
            assert visualizations_subdir.exists(), (
                f"Bug NOT confirmed: Per-run visualizations/ subdirectory was not created. "
                f"Expected {visualizations_subdir} to exist (bug behavior), but it doesn't."
            )
            
            # Verify it's empty (never used)
            contents = list(visualizations_subdir.iterdir())
            assert contents == [], (
                f"Per-run visualizations/ subdirectory is not empty: {contents}. "
                f"Expected empty directory (bug behavior), indicating it's never used."
            )
    
    def test_other_subdirectories_still_created(self):
        """
        Test that logs/ and checkpoints/ subdirectories are still created (preservation check).
        
        **Preservation**: This verifies that other subdirectories are not affected by the bug.
        **Current Behavior**: logs/ and checkpoints/ are created correctly.
        
        **Expected on unfixed code**: This test PASSES (other subdirectories work correctly).
        **Expected after fix**: This test PASSES (other subdirectories still work correctly).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            run_id = "test_run_preservation"
            
            # Create run directory
            run_dir = persistence.create_run_directory(run_id)
            
            # Verify logs/ subdirectory exists
            logs_subdir = run_dir / "logs"
            assert logs_subdir.exists(), (
                f"logs/ subdirectory not created: {logs_subdir}. "
                f"This is a regression - logs/ should always be created."
            )
            assert logs_subdir.is_dir(), f"logs/ is not a directory: {logs_subdir}"
            
            # Verify checkpoints/ subdirectory exists
            checkpoints_subdir = run_dir / "checkpoints"
            assert checkpoints_subdir.exists(), (
                f"checkpoints/ subdirectory not created: {checkpoints_subdir}. "
                f"This is a regression - checkpoints/ should always be created."
            )
            assert checkpoints_subdir.is_dir(), f"checkpoints/ is not a directory: {checkpoints_subdir}"
    
    @given(st.text(min_size=1, max_size=30, alphabet='abcdefghijklmnopqrstuvwxyz0123456789_-'))
    @settings(max_examples=20, deadline=None)
    def test_property_per_run_visualizations_subdirectory_created_for_any_run_id(self, run_id):
        """
        Property-based test: Per-run visualizations/ subdirectory is created for any run_id.
        
        **Bug Condition**: For any valid run_id, create_run_directory() creates visualizations/ subdirectory.
        **Current Behavior (unfixed)**: Subdirectory is always created.
        
        **Expected on unfixed code**: This test PASSES (confirming bug exists for all inputs).
        **Expected after fix**: This test FAILS (subdirectory no longer created for any input).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            
            # Create run directory with generated run_id
            run_dir = persistence.create_run_directory(run_id)
            
            # BUG: The per-run visualizations/ subdirectory should NOT be created
            visualizations_subdir = run_dir / "visualizations"
            
            # This assertion SHOULD PASS on unfixed code (bug exists)
            # This assertion SHOULD FAIL after fix (bug is fixed)
            assert visualizations_subdir.exists(), (
                f"Bug NOT confirmed for run_id '{run_id}': "
                f"Per-run visualizations/ subdirectory was not created. "
                f"Expected {visualizations_subdir} to exist (bug behavior), but it doesn't."
            )
            
            # Verify it's a directory
            assert visualizations_subdir.is_dir(), (
                f"Per-run visualizations/ exists but is not a directory: {visualizations_subdir}"
            )
    
    @given(st.text(min_size=1, max_size=30, alphabet='abcdefghijklmnopqrstuvwxyz0123456789_-'))
    @settings(max_examples=20, deadline=None)
    def test_property_other_subdirectories_preserved_for_any_run_id(self, run_id):
        """
        Property-based test: logs/ and checkpoints/ subdirectories are created for any run_id.
        
        **Preservation**: For any valid run_id, logs/ and checkpoints/ should be created.
        **Current Behavior**: These subdirectories are created correctly.
        
        **Expected on unfixed code**: This test PASSES (preservation verified).
        **Expected after fix**: This test PASSES (preservation maintained).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ResultsPersistence(output_dir=tmpdir)
            
            # Create run directory with generated run_id
            run_dir = persistence.create_run_directory(run_id)
            
            # Verify logs/ subdirectory exists
            logs_subdir = run_dir / "logs"
            assert logs_subdir.exists(), (
                f"logs/ subdirectory not created for run_id '{run_id}': {logs_subdir}. "
                f"This is a regression - logs/ should always be created."
            )
            assert logs_subdir.is_dir(), f"logs/ is not a directory: {logs_subdir}"
            
            # Verify checkpoints/ subdirectory exists
            checkpoints_subdir = run_dir / "checkpoints"
            assert checkpoints_subdir.exists(), (
                f"checkpoints/ subdirectory not created for run_id '{run_id}': {checkpoints_subdir}. "
                f"This is a regression - checkpoints/ should always be created."
            )
            assert checkpoints_subdir.is_dir(), f"checkpoints/ is not a directory: {checkpoints_subdir}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
