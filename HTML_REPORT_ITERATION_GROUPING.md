# HTML Report - Iteration Grouping

## What Changed

The "📈 Quantization Results" section in the HTML report now groups results by iteration, making it easy to see the data from each individual test run.

## Visual Layout

### Before (Single Table)
```
📈 Quantization Results
┌─────────────────────────────────────────────────────────────┐
│ Quantization │ Load Time │ Peak RAM │ TTFT │ ... │ GPU     │
├─────────────────────────────────────────────────────────────┤
│ Q4_0         │ 2.50      │ 1200.00  │ 20.0 │ ... │ No      │
│ Q8_0         │ 3.00      │ 1800.00  │ 28.0 │ ... │ No      │
│ Q2_K         │ 2.20      │ 900.00   │ 18.5 │ ... │ No      │
│ Q4_0         │ 2.40      │ 1205.00  │ 19.5 │ ... │ No      │
│ Q8_0         │ 2.90      │ 1805.00  │ 27.5 │ ... │ No      │
│ Q2_K         │ 2.25      │ 905.00   │ 18.8 │ ... │ No      │
│ ... (all iterations mixed together)                         │
└─────────────────────────────────────────────────────────────┘
```

### After (Grouped by Iteration)
```
📈 Quantization Results

▼ Iteration 1
┌─────────────────────────────────────────────────────────────┐
│ Quantization │ Load Time │ Peak RAM │ TTFT │ ... │ GPU     │
├─────────────────────────────────────────────────────────────┤
│ Q4_0         │ 2.50      │ 1200.00  │ 20.0 │ ... │ No      │
│ Q8_0         │ 3.00      │ 1800.00  │ 28.0 │ ... │ No      │
│ Q2_K         │ 2.20      │ 900.00   │ 18.5 │ ... │ No      │
└─────────────────────────────────────────────────────────────┘

▼ Iteration 2
┌─────────────────────────────────────────────────────────────┐
│ Quantization │ Load Time │ Peak RAM │ TTFT │ ... │ GPU     │
├─────────────────────────────────────────────────────────────┤
│ Q4_0         │ 2.40      │ 1205.00  │ 19.5 │ ... │ No      │
│ Q8_0         │ 2.90      │ 1805.00  │ 27.5 │ ... │ No      │
│ Q2_K         │ 2.25      │ 905.00   │ 18.8 │ ... │ No      │
└─────────────────────────────────────────────────────────────┘

▼ Iteration 3
┌─────────────────────────────────────────────────────────────┐
│ Quantization │ Load Time │ Peak RAM │ TTFT │ ... │ GPU     │
├─────────────────────────────────────────────────────────────┤
│ Q4_0         │ 2.60      │ 1198.00  │ 20.5 │ ... │ No      │
│ Q8_0         │ 3.10      │ 1795.00  │ 28.5 │ ... │ No      │
│ Q2_K         │ 2.18      │ 898.00   │ 18.2 │ ... │ No      │
└─────────────────────────────────────────────────────────────┘

... (more iterations)
```

## Features

### 1. Collapsible Sections
- Each iteration is in a collapsible section
- Click to expand/collapse individual iterations
- Default: All iterations collapsed (clean view)
- Click "Iteration 1" to see those results

### 2. Clear Organization
- Easy to compare results across iterations
- Can see variation between runs
- Helps identify outliers or anomalies

### 3. Same Table Format
- Each iteration table has the same columns:
  - Quantization level
  - Load Time (s)
  - Peak RAM (MB)
  - TTFT (ms)
  - Prefill TPS
  - Decode TPS
  - Tokens (in/out)
  - GPU usage

## Example HTML Structure

```html
<h2>📈 Quantization Results</h2>

<!-- Iteration 1 -->
<button class="collapsible">Iteration 1</button>
<div class="content">
    <div class="content-inner">
        <table>
            <thead>...</thead>
            <tbody>
                <tr>
                    <td><strong>Q4_0</strong></td>
                    <td>2.50</td>
                    <td>1200.00</td>
                    ...
                </tr>
                <tr>
                    <td><strong>Q8_0</strong></td>
                    <td>3.00</td>
                    <td>1800.00</td>
                    ...
                </tr>
            </tbody>
        </table>
    </div>
</div>

<!-- Iteration 2 -->
<button class="collapsible">Iteration 2</button>
<div class="content">
    <div class="content-inner">
        <table>
            ...
        </table>
    </div>
</div>

<!-- More iterations... -->
```

## Benefits

### 1. Better Readability
- No more scrolling through a huge table
- Each iteration is clearly separated
- Easy to focus on one iteration at a time

### 2. Easy Comparison
- Can open multiple iterations side-by-side (if browser window is wide)
- Can quickly spot differences between runs
- Can identify which iteration had anomalies

### 3. Cleaner Initial View
- Report loads with iterations collapsed
- User can expand only what they need
- Reduces visual clutter

### 4. Consistent with Other Sections
- Uses same collapsible pattern as:
  - Statistical Summaries
  - Ablation Study Results
  - Batch Processing Results
- Familiar UI pattern

## Implementation Details

### Backend Changes
**File**: `llm_benchmark/visualization/visualization_generator.py`

**Function**: `_prepare_quantization_table()`
- Changed return type from `List[Dict]` to `Dict[str, Any]`
- Groups results by iteration number
- Returns structure:
  ```python
  {
      'iterations': [1, 2, 3, 4, 5],
      'results_by_iteration': {
          1: [result1, result2, result3],
          2: [result4, result5, result6],
          ...
      }
  }
  ```

### Frontend Changes
**File**: `llm_benchmark/visualization/visualization_generator.py` (HTML template)

**Template Loop**:
```jinja2
{% for iteration in quant_table.iterations %}
    <button class="collapsible">Iteration {{ iteration }}</button>
    <div class="content">
        <table>
            {% for row in quant_table.results_by_iteration[iteration] %}
                <tr>...</tr>
            {% endfor %}
        </table>
    </div>
{% endfor %}
```

## User Experience

### Opening the Report
1. User opens `benchmark_report.html`
2. Sees "📈 Quantization Results" heading
3. Sees collapsed sections: "Iteration 1", "Iteration 2", etc.

### Viewing Results
1. Click "Iteration 1" to expand
2. See table with all quantization levels for that iteration
3. Click "Iteration 2" to compare
4. Click again to collapse

### Analyzing Data
- Can keep multiple iterations open simultaneously
- Can quickly scan through iterations by clicking each one
- Can focus on specific iterations of interest
- Can identify outliers by comparing across iterations

## Summary

✅ **Quantization Results are now grouped by iteration**
✅ **Each iteration is in a collapsible section**
✅ **Cleaner, more organized presentation**
✅ **Easier to compare results across iterations**
✅ **Consistent with the rest of the report UI**

The HTML report will now provide a much better user experience for viewing multi-iteration benchmark results!
