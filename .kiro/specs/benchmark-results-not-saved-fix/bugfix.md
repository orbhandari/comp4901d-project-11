# Bugfix Requirements Document

## Introduction

This bugfix addresses a critical issue where benchmark results are not saved to the output directory on Android devices despite successful benchmark execution. The root cause is in the `generate_reports()` function in `llm_benchmark/main.py`, which has overly broad exception handling that silently catches and logs errors without properly propagating them or ensuring results are saved. This leaves users with no output files and no clear indication of what went wrong.

The bug is particularly problematic on Android where path expansion issues (e.g., `~` not being expanded), permission problems, or missing directories can prevent file writes. The current implementation catches these exceptions but only logs a warning, resulting in silent failure.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN `generate_reports()` encounters any exception during report generation (permission error, path expansion failure, missing directory, etc.) THEN the system catches the exception with a broad try-except block and only logs a warning without saving any results

1.2 WHEN the benchmark completes successfully but `persistence.create_run_directory()` fails THEN the system logs "Report generation failed" but does not inform the user of the specific error or attempt alternative save locations

1.3 WHEN individual format saves (JSON, CSV, Markdown, HTML) fail within the try-except block THEN the system continues silently without notifying the user which formats failed to save

1.4 WHEN path expansion issues occur on Android (e.g., `~/storage/shared/benchmark_results` not being expanded to absolute path) THEN the system fails to create directories or save files without clear error messages

### Expected Behavior (Correct)

2.1 WHEN `generate_reports()` encounters an exception during report generation THEN the system SHALL propagate the error with a clear, actionable error message indicating the specific failure (e.g., "Failed to create directory: Permission denied at /path/to/dir")

2.2 WHEN `persistence.create_run_directory()` fails THEN the system SHALL log the specific error, attempt to expand paths properly (e.g., convert `~` to absolute path), and if that fails, SHALL raise an exception to notify the user

2.3 WHEN individual format saves (JSON, CSV, Markdown, HTML) fail THEN the system SHALL log which specific format failed with the error details and SHALL continue attempting to save other formats, then report all failures at the end

2.4 WHEN path expansion issues occur on Android THEN the system SHALL properly expand paths using `Path.expanduser()` before attempting directory creation and file writes, and SHALL provide clear error messages if expansion or creation fails

### Unchanged Behavior (Regression Prevention)

3.1 WHEN report generation succeeds without errors THEN the system SHALL CONTINUE TO create the run directory and save all requested formats as before

3.2 WHEN visualizations are generated successfully THEN the system SHALL CONTINUE TO include them in the HTML report and reference them in other formats

3.3 WHEN multiple formats are requested in `config.save_formats` THEN the system SHALL CONTINUE TO attempt to save all requested formats

3.4 WHEN the benchmark completes successfully on non-Android platforms THEN the system SHALL CONTINUE TO save results to the output directory without regression
