#!/bin/bash
# Fix missing __init__.py files

echo "Creating missing __init__.py files..."

# Ensure all directories have __init__.py
touch llm_benchmark/__init__.py
touch llm_benchmark/hardware/__init__.py
touch llm_benchmark/inference/__init__.py
touch llm_benchmark/metrics/__init__.py
touch llm_benchmark/model_manager/__init__.py
touch llm_benchmark/orchestrator/__init__.py
touch llm_benchmark/profiler/__init__.py
touch llm_benchmark/results/__init__.py
touch llm_benchmark/statistics/__init__.py
touch llm_benchmark/visualization/__init__.py

echo "✅ Done! All __init__.py files created."
echo ""
echo "Now try running the benchmark again:"
echo "  python -m llm_benchmark --config configs/android_example.json"
