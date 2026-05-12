# Implementation Plan

- [x] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Per-Run Visualizations Subdirectory Created
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the bug exists
  - **Scoped PBT Approach**: For deterministic bugs, scope the property to the concrete failing case(s) to ensure reproducibility
  - Test that `ResultsPersistence.create_run_directory(run_id)` creates a `run_<run_id>/visualizations/` subdirectory
  - The test assertions should verify that the per-run `visualizations/` subdirectory exists (this is the bug)
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (this is correct - it proves the bug exists)
  - Document counterexamples found to understand root cause (e.g., "create_run_directory('test_001') creates run_test_001/visualizations/ which remains empty")
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Other Subdirectories Still Created
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for non-buggy directory creation (main run directory, logs/, checkpoints/)
  - Write property-based tests capturing observed behavior patterns from Preservation Requirements
  - Property-based testing generates many test cases for stronger guarantees
  - Test that `create_run_directory(run_id)` creates the main `run_<run_id>/` directory for various run_id formats
  - Test that `create_run_directory(run_id)` creates the `logs/` subdirectory
  - Test that `create_run_directory(run_id)` creates the `checkpoints/` subdirectory
  - Test that the method returns the correct `Path` object
  - Test error handling when directory creation fails (OSError)
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 3. Fix for unnecessary per-run visualizations subdirectory

  - [x] 3.1 Implement the fix
    - Remove the line that creates the per-run `visualizations/` subdirectory in `llm_benchmark/results/persistence.py`
    - Delete line 76-77: `(run_dir / "visualizations").mkdir(exist_ok=True)`
    - Keep the `logs/` subdirectory creation (line 78)
    - Keep the `checkpoints/` subdirectory creation (line 79)
    - Ensure no other changes are made to the method
    - _Bug_Condition: isBugCondition(input) where input.method == "create_run_directory" AND input.creates_subdirectory("visualizations") AND NOT subdirectory_is_used("visualizations")_
    - _Expected_Behavior: For any call to create_run_directory(run_id), the method SHALL NOT create a run_<run_id>/visualizations/ subdirectory_
    - _Preservation: The method SHALL continue to create the main run_<run_id>/ directory and the logs/ and checkpoints/ subdirectories_
    - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 3.3_

  - [x] 3.2 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - No Per-Run Visualizations Subdirectory Created
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior
    - When this test passes, it confirms the expected behavior is satisfied
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 3.3 Verify preservation tests still pass
    - **Property 2: Preservation** - Other Subdirectories Still Created
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm all tests still pass after fix (no regressions)
    - _Requirements: 3.1, 3.2, 3.3_

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
