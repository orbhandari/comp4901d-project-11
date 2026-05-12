# Visualization Path Fix Bugfix Design

## Overview

This bugfix addresses an issue where the `ResultsPersistence.create_run_directory()` method creates an unnecessary per-run `visualizations/` subdirectory at `benchmark_results/run_<timestamp>/visualizations/`, while the `VisualizationGenerator` class correctly saves PNG files to the top-level `benchmark_results/visualizations/` directory. This creates confusion about where visualization files are stored and results in an empty, unused subdirectory for each benchmark run.

The fix is minimal and targeted: remove the line that creates the per-run `visualizations/` subdirectory in `ResultsPersistence.create_run_directory()`. This ensures that only the top-level `visualizations/` directory exists (created by `VisualizationGenerator`), which is shared across all benchmark runs and contains the actual PNG files.

## Glossary

- **Bug_Condition (C)**: The condition that triggers the bug - when `ResultsPersistence.create_run_directory()` is called, it creates an unnecessary `run_<timestamp>/visualizations/` subdirectory
- **Property (P)**: The desired behavior - `ResultsPersistence.create_run_directory()` should NOT create a `visualizations/` subdirectory
- **Preservation**: Existing directory creation behavior that must remain unchanged - the method must continue to create the main `run_<timestamp>/` directory and the `logs/` and `checkpoints/` subdirectories
- **ResultsPersistence**: The class in `llm_benchmark/results/persistence.py` that handles persistence of benchmark results to multiple formats
- **create_run_directory()**: The method that creates the organized directory structure for a benchmark run
- **VisualizationGenerator**: The class in `llm_benchmark/visualization/visualization_generator.py` that generates visualizations and correctly saves them to `output_dir/visualizations/`
- **run_dir**: The per-run directory at `benchmark_results/run_<timestamp>/`
- **viz_dir**: The top-level visualizations directory at `benchmark_results/visualizations/` (shared across all runs)

## Bug Details

### Bug Condition

The bug manifests when `ResultsPersistence.create_run_directory()` is called during benchmark execution. The method creates a per-run `visualizations/` subdirectory at `run_<timestamp>/visualizations/`, but this directory is never used because `VisualizationGenerator` saves PNG files to the top-level `benchmark_results/visualizations/` directory instead.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type MethodCall (ResultsPersistence.create_run_directory)
  OUTPUT: boolean
  
  RETURN input.method == "create_run_directory"
         AND input.creates_subdirectory("visualizations")
         AND NOT subdirectory_is_used("visualizations")
END FUNCTION
```

### Examples

- **Example 1**: When a benchmark run is executed with run_id "20240115_143022", the system creates `benchmark_results/run_20240115_143022/visualizations/` (empty, unused) while PNG files are saved to `benchmark_results/visualizations/` (actual location)
- **Example 2**: After multiple benchmark runs, the directory structure shows:
  - `benchmark_results/visualizations/` (contains quantization_comparison.png, memory_vs_speed_tradeoff.png, etc.)
  - `benchmark_results/run_20240115_143022/visualizations/` (empty)
  - `benchmark_results/run_20240115_150033/visualizations/` (empty)
  - `benchmark_results/run_20240115_162145/visualizations/` (empty)
- **Example 3**: A user looking for visualization files might check the per-run `visualizations/` subdirectory and find it empty, causing confusion about where the files are actually stored
- **Edge case**: Even when no visualizations are generated (e.g., visualization generation fails), the empty per-run `visualizations/` subdirectory is still created unnecessarily

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- The method must continue to create the main `run_<timestamp>/` directory
- The method must continue to create the `logs/` subdirectory for log files
- The method must continue to create the `checkpoints/` subdirectory for checkpoint files
- The method must continue to raise `OSError` if directory creation fails
- The method must continue to return the `Path` object for the run directory
- `VisualizationGenerator` must continue to create and use `output_dir/visualizations/` for PNG files
- The HTML report must continue to embed images as base64 data URIs (not affected by this change)
- Results files (JSON, CSV, Markdown) must continue to be saved to the per-run directory

**Scope:**
All directory creation operations that do NOT involve the per-run `visualizations/` subdirectory should be completely unaffected by this fix. This includes:
- Creation of the main run directory
- Creation of the `logs/` subdirectory
- Creation of the `checkpoints/` subdirectory
- Creation of the top-level `visualizations/` directory by `VisualizationGenerator`
- All file saving operations (results, reports, PNG files)

## Hypothesized Root Cause

Based on the bug description and code analysis, the root cause is clear:

1. **Unnecessary Directory Creation**: The `create_run_directory()` method in `ResultsPersistence` creates a `visualizations/` subdirectory at line 76-77:
   ```python
   (run_dir / "visualizations").mkdir(exist_ok=True)
   ```
   This line was likely added with the assumption that visualization files would be saved to the per-run directory, but this is not how `VisualizationGenerator` works.

2. **Mismatched Architecture**: The `VisualizationGenerator` class is initialized with `output_dir` (the base `benchmark_results/` directory) and creates its own `visualizations/` subdirectory at the top level:
   ```python
   self.viz_dir = os.path.join(output_dir, "visualizations")
   os.makedirs(self.viz_dir, exist_ok=True)
   ```
   This is the correct behavior because visualizations are meant to be shared across all benchmark runs, not isolated per-run.

3. **No Usage of Per-Run Subdirectory**: The per-run `visualizations/` subdirectory is never referenced or used anywhere in the codebase. All PNG files are saved to `self.viz_dir` (the top-level directory), and the HTML report embeds images as base64 data URIs (not file paths).

4. **Design Intent**: The original design intent was for visualizations to be shared across runs (stored at the top level), but the `create_run_directory()` method was incorrectly creating a per-run subdirectory that conflicts with this design.

## Correctness Properties

Property 1: Bug Condition - No Per-Run Visualizations Subdirectory Created

_For any_ call to `ResultsPersistence.create_run_directory(run_id)`, the fixed method SHALL NOT create a `run_<run_id>/visualizations/` subdirectory, ensuring that only the top-level `benchmark_results/visualizations/` directory exists (created by `VisualizationGenerator`).

**Validates: Requirements 2.1, 2.2, 2.3**

Property 2: Preservation - Other Subdirectories Still Created

_For any_ call to `ResultsPersistence.create_run_directory(run_id)`, the fixed method SHALL continue to create the main `run_<run_id>/` directory and the `logs/` and `checkpoints/` subdirectories, preserving all existing directory creation behavior except for the `visualizations/` subdirectory.

**Validates: Requirements 3.1, 3.2, 3.3**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `llm_benchmark/results/persistence.py`

**Function**: `create_run_directory()`

**Specific Changes**:
1. **Remove Visualizations Subdirectory Creation**: Delete or comment out the line that creates the per-run `visualizations/` subdirectory
   - Current code (lines 76-77):
     ```python
     (run_dir / "visualizations").mkdir(exist_ok=True)
     ```
   - Fixed code: Remove this line entirely

2. **Keep Logs Subdirectory Creation**: Ensure the `logs/` subdirectory creation remains unchanged
   - Line 78:
     ```python
     (run_dir / "logs").mkdir(exist_ok=True)
     ```

3. **Keep Checkpoints Subdirectory Creation**: Ensure the `checkpoints/` subdirectory creation remains unchanged
   - Line 79:
     ```python
     (run_dir / "checkpoints").mkdir(exist_ok=True)
     ```

4. **No Changes to VisualizationGenerator**: The `VisualizationGenerator.__init__()` method already correctly creates the top-level `visualizations/` directory and should remain unchanged

5. **No Changes to Main Workflow**: The `main.py` workflow already correctly initializes `VisualizationGenerator` with `output_dir=config.output_dir` (the base directory), so no changes are needed there

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code (verify the per-run subdirectory is created), then verify the fix works correctly (subdirectory is not created) and preserves existing behavior (other subdirectories are still created).

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm that the per-run `visualizations/` subdirectory is created unnecessarily.

**Test Plan**: Write tests that call `ResultsPersistence.create_run_directory()` and assert that a `visualizations/` subdirectory is created in the run directory. Run these tests on the UNFIXED code to observe the bug and confirm the root cause.

**Test Cases**:
1. **Per-Run Visualizations Directory Created**: Call `create_run_directory("test_run_001")` and verify that `run_test_run_001/visualizations/` exists (will pass on unfixed code, demonstrating the bug)
2. **Directory Structure Inspection**: Create multiple run directories and verify that each has its own `visualizations/` subdirectory (will pass on unfixed code)
3. **Unused Directory Detection**: Verify that the per-run `visualizations/` subdirectory is empty after a full benchmark run (will pass on unfixed code, confirming it's unused)
4. **Top-Level Directory Exists**: Verify that `benchmark_results/visualizations/` exists and contains PNG files after visualization generation (will pass on both unfixed and fixed code)

**Expected Counterexamples**:
- The per-run `visualizations/` subdirectory is created but remains empty
- PNG files are saved to the top-level `visualizations/` directory, not the per-run subdirectory
- Multiple benchmark runs result in multiple empty per-run `visualizations/` subdirectories

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds (calls to `create_run_directory()`), the fixed function produces the expected behavior (no per-run `visualizations/` subdirectory is created).

**Pseudocode:**
```
FOR ALL run_id WHERE isBugCondition(create_run_directory(run_id)) DO
  run_dir := create_run_directory_fixed(run_id)
  ASSERT NOT (run_dir / "visualizations").exists()
  ASSERT (output_dir / "visualizations").exists()  # Top-level directory created by VisualizationGenerator
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs, the fixed function continues to create the main run directory and the `logs/` and `checkpoints/` subdirectories.

**Pseudocode:**
```
FOR ALL run_id DO
  run_dir := create_run_directory_fixed(run_id)
  ASSERT run_dir.exists()
  ASSERT (run_dir / "logs").exists()
  ASSERT (run_dir / "checkpoints").exists()
  ASSERT NOT (run_dir / "visualizations").exists()
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain (different run_id formats)
- It catches edge cases that manual unit tests might miss (special characters, long strings, etc.)
- It provides strong guarantees that behavior is unchanged for all valid run_id inputs

**Test Plan**: Observe behavior on UNFIXED code first for directory creation, then write property-based tests capturing that behavior (minus the `visualizations/` subdirectory).

**Test Cases**:
1. **Main Run Directory Creation**: Verify that the main `run_<run_id>/` directory is created for various run_id formats
2. **Logs Subdirectory Creation**: Verify that `logs/` subdirectory is created in the run directory
3. **Checkpoints Subdirectory Creation**: Verify that `checkpoints/` subdirectory is created in the run directory
4. **Error Handling Preservation**: Verify that `OSError` is still raised when directory creation fails (e.g., invalid path, no write permissions)
5. **Return Value Preservation**: Verify that the method still returns the correct `Path` object for the run directory

### Unit Tests

- Test `create_run_directory()` with various run_id formats (timestamp, custom strings, edge cases)
- Test that the main run directory is created
- Test that `logs/` and `checkpoints/` subdirectories are created
- Test that `visualizations/` subdirectory is NOT created
- Test error handling when directory creation fails
- Test that the method returns the correct `Path` object

### Property-Based Tests

- Generate random run_id strings and verify directory structure is correct (no `visualizations/` subdirectory)
- Generate random run_id formats and verify preservation of `logs/` and `checkpoints/` subdirectories
- Test across many scenarios to ensure no edge cases create the `visualizations/` subdirectory

### Integration Tests

- Run a full benchmark workflow and verify directory structure is correct
- Verify that PNG files are saved to the top-level `visualizations/` directory
- Verify that the HTML report is generated correctly with embedded images
- Verify that multiple benchmark runs share the same top-level `visualizations/` directory
- Verify that per-run directories contain only `logs/`, `checkpoints/`, and result files (no `visualizations/`)
