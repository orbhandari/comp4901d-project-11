# Statistical Significance Testing Guide

## Current State

The benchmark framework calculates statistical summaries for each quantization level:
- ✅ Mean
- ✅ Standard Deviation  
- ✅ 95% Confidence Intervals
- ✅ Outlier Detection

**BUT** it does NOT compare quantizations to determine if differences are statistically significant.

## How to Determine Significance (Current Method)

### Method 1: Check Confidence Interval Overlap

Look at the **95% Confidence Intervals** in the Statistical Summaries table:

**Example from your latest run:**

| Quantization | Metric | Mean | 95% CI |
|--------------|--------|------|--------|
| Q8_0 | decode_tps | 62.89 | [62.30, 63.49] |
| Q4_0 | decode_tps | 100.02 | [98.21, 101.83] |
| Q2_K | decode_tps | 91.46 | [90.75, 92.17] |

**Interpretation:**
- **Q4_0 vs Q8_0**: CIs don't overlap → **✓ Significantly different**
- **Q4_0 vs Q2_K**: CIs don't overlap → **✓ Significantly different**
- **Q2_K vs Q8_0**: CIs don't overlap → **✓ Significantly different**

**Rule**: If confidence intervals don't overlap, the difference is statistically significant (p < 0.05).

### Method 2: Calculate Effect Size

Calculate the difference relative to standard error:

```
Effect Size = (Mean_A - Mean_B) / sqrt(SE_A² + SE_B²)
```

Where `SE = Std Dev / sqrt(n)`

**Example:**
```
Q4_0 decode: 100.02 ± 2.07 (n=5)
Q2_K decode: 91.46 ± 0.80 (n=5)

SE_Q4 = 2.07 / sqrt(5) = 0.93
SE_Q2 = 0.80 / sqrt(5) = 0.36

Effect Size = (100.02 - 91.46) / sqrt(0.93² + 0.36²)
            = 8.56 / 0.99
            = 8.65 (VERY LARGE)
```

**Interpretation:**
- Effect Size > 2.0 → **Strong significance**
- Effect Size > 3.0 → **Very strong significance**
- Effect Size > 5.0 → **Extremely strong significance**

## What's Missing: Automated Pairwise Comparisons

The framework has a `compare_configurations()` method in `StatisticalValidator` that performs t-tests, but it's **not being used** in the main workflow!

### What It Would Provide

If we add pairwise comparisons, you would get:

| Comparison | Metric | Mean A | Mean B | Difference | p-value | Significant? |
|------------|--------|--------|--------|------------|---------|--------------|
| Q4_0 vs Q8_0 | decode_tps | 100.02 | 62.89 | +37.13 | < 0.001 | ✓ Yes |
| Q4_0 vs Q2_K | decode_tps | 100.02 | 91.46 | +8.56 | < 0.001 | ✓ Yes |
| Q2_K vs Q8_0 | decode_tps | 91.46 | 62.89 | +28.57 | < 0.001 | ✓ Yes |

## Recommended Solution

### Option 1: Manual Analysis (Current)

**For decode_tps (most important metric):**

From your latest run:
- Q4_0: 100.02 ± 2.07 tokens/sec [98.21, 101.83]
- Q2_K: 91.46 ± 0.80 tokens/sec [90.75, 92.17]
- Q8_0: 62.89 ± 0.68 tokens/sec [62.30, 63.49]

**Conclusions:**
1. ✅ **Q4_0 is significantly faster than Q2_K** (8.56 t/s difference, no CI overlap)
2. ✅ **Q4_0 is significantly faster than Q8_0** (37.13 t/s difference, no CI overlap)
3. ✅ **Q2_K is significantly faster than Q8_0** (28.57 t/s difference, no CI overlap)

**All differences are statistically significant at p < 0.05 level.**

### Option 2: Add Automated Comparisons (Recommended)

I can add a new section to the HTML report that shows:

**Pairwise Comparisons (decode_tps)**

| Comparison | Faster | Difference | p-value | Significance |
|------------|--------|------------|---------|--------------|
| Q4_0 vs Q2_K | Q4_0 | +8.56 t/s (+9.4%) | < 0.001 | ✓✓✓ |
| Q4_0 vs Q8_0 | Q4_0 | +37.13 t/s (+59.0%) | < 0.001 | ✓✓✓ |
| Q2_K vs Q8_0 | Q2_K | +28.57 t/s (+45.4%) | < 0.001 | ✓✓✓ |

**Legend:**
- ✓ = Significant (p < 0.05)
- ✓✓ = Highly significant (p < 0.01)
- ✓✓✓ = Extremely significant (p < 0.001)
- ✗ = Not significant (p ≥ 0.05)

## Quick Answer to Your Question

**"How do I determine if Q4_0 is significantly faster than Q2_K and Q8_0?"**

**Answer**: Look at the **95% Confidence Intervals** in the Statistical Summaries section:

1. **Find the metric** you care about (e.g., `decode_tps`)
2. **Compare the CIs** for each quantization
3. **If CIs don't overlap** → Difference is significant (p < 0.05)

**For your latest run:**
- Q4_0 decode: [98.21, 101.83]
- Q2_K decode: [90.75, 92.17]
- Q8_0 decode: [62.30, 63.49]

**No overlap** → **All differences are statistically significant!** ✓

## Implementation Plan

Would you like me to:

1. ✅ **Add automated pairwise comparisons** to the HTML report?
2. ✅ **Add visual indicators** (✓/✗) for significance?
3. ✅ **Add a "Winner" column** showing which quantization is best for each metric?
4. ✅ **Add percentage differences** to make comparisons easier?

This would make it immediately obvious which quantization is significantly better without manual CI checking!
