# Bugfix Requirements Document

## Introduction

This document specifies the requirements for fixing a bug where visualization PNG files are being saved to the wrong directory location. The `ResultsPersistence` class is creating an unnecessary per-run `visualizations/` subdirectory, while the `VisualizationGenerator` class is correctly saving to the top-level `benchmark_results/visualizations/` directory. This creates confusion about where visualization files are actually stored and may cause the HTML report to reference incorrect paths.

The fix ensures that:
1. Visualization PNGs are consistently saved to the top-level `benchmark_results/visualizations/` directory (shared across all runs)
2. The per-run `run_<timestamp>/visualizations/` subdirectory is not created unnecessarily
3. The HTML report correctly references images from the top-level visualizations directory

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN `ResultsPersistence.create_run_directory()` is called THEN the system creates an unnecessary `run_<timestamp>/visualizations/` subdirectory

1.2 WHEN visualization files are generated THEN the system saves them to `benchmark_results/visualizations/` (correct location) but the per-run subdirectory still exists (creating confusion)

1.3 WHEN the directory structure is examined THEN the system shows both `benchmark_results/visualizations/` (with actual PNG files) and `benchmark_results/run_<timestamp>/visualizations/` (empty or unused)

### Expected Behavior (Correct)

2.1 WHEN `ResultsPersistence.create_run_directory()` is called THEN the system SHALL NOT create a `run_<timestamp>/visualizations/` subdirectory

2.2 WHEN visualization files are generated THEN the system SHALL save them only to the top-level `benchmark_results/visualizations/` directory

2.3 WHEN the directory structure is examined THEN the system SHALL show only `benchmark_results/visualizations/` (with actual PNG files) and no per-run visualization subdirectories

2.4 WHEN the HTML report is generated THEN the system SHALL correctly reference images from the top-level `benchmark_results/visualizations/` directory

### Unchanged Behavior (Regression Prevention)

3.1 WHEN `ResultsPersistence.create_run_directory()` is called THEN the system SHALL CONTINUE TO create the main `run_<timestamp>/` directory

3.2 WHEN `ResultsPersistence.create_run_directory()` is called THEN the system SHALL CONTINUE TO create the `run_<timestamp>/logs/` subdirectory

3.3 WHEN `ResultsPersistence.create_run_directory()` is called THEN the system SHALL CONTINUE TO create the `run_<timestamp>/checkpoints/` subdirectory

3.4 WHEN `VisualizationGenerator` saves PNG files THEN the system SHALL CONTINUE TO save them to `self.viz_dir` (which is `output_dir/visualizations/`)

3.5 WHEN the HTML report embeds images THEN the system SHALL CONTINUE TO embed them as base64 data URIs

3.6 WHEN results are saved in JSON, CSV, or Markdown formats THEN the system SHALL CONTINUE TO save them to the per-run directory

3.7 WHEN the `VisualizationGenerator` is initialized with `output_dir` THEN the system SHALL CONTINUE TO create the `output_dir/visualizations/` directory

3.8 WHEN multiple benchmark runs are executed THEN the system SHALL CONTINUE TO share the same top-level `visualizations/` directory across all runs
