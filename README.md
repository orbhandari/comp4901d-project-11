# comp4901d-project-11

## Progress 
- Implemented benchmark script for Linux x86 specifically
- Automatically download from Hugging Face (Meta Llama 3.1B) 
- Automatically runs local inference on single prompt
- Quantization profiling on two levels: Q8 and Q4_K_M
- Ablation study on KV cache reuse, both RAM and disk based, respectively. One scenario only.
- Prints simple benchmark matrix.

## Results and discussions
```
========================================
FINAL BENCHMARK MATRIX
========================================
               Device  Quant  Load_Time_s  Peak_RAM_MB  RAM_Increase_MB  TTFT_ms  Prefill_tps  Decode_tps
x86_Arch_Linux_Laptop   Q8_0         0.35      8609.23          8509.74  1836.46        55.54        9.99
x86_Arch_Linux_Laptop Q4_K_M         0.79      8356.86          8191.29  1547.97        65.89       16.17

========================================
PROMPT CACHING ABLATION (RAM Cache)
========================================
Scenario: Large Prefix (RAM)
TTFT_Cold_ms: 14969.76
TTFT_Warm_ms: 228.09
Improvement_ms: 14741.67
Isolated_Cache_Overhead_MB: 701.07

========================================
PROMPT CACHING ABLATION (Disk Cache)
========================================
Scenario: Large Prefix (DISK)
TTFT_Cold_ms: 14285.52
TTFT_Warm_ms: 470.65
Improvement_ms: 13814.87
Isolated_Cache_Overhead_MB: 0.89```
```

- Why Q4_K_M longer load time than Q8?
- Q4_K_M has lower RAM usage than Q8, as expected (see peak RAM and RAM increase)
- Q4_K_M has lower TTFT than Q8.
- Q4_K_M has higher prefill tps and decode tps than Q8.
- Are the above results normal and expected?

- In ablation study, reduction improvement with disk cache higher than RAM cache?
  - I'm guessing due to page cache mechanism of OS. Potential for discussion here.

## Some interesting points
- Number of tokens tested
- Effects of page cache of OS 
- Quantization and how it affects hardware/memory mapping 
- Calling garbage collection 
- The warmup call `_ = llm("warmup")`
- Remembering to delete cache folder
- The "stream" mode and how we used it to compute prefill/decode phase separately in Python
- Ram cache vs disk cache
- SIMD/AVX, packing/unpacking, dequantization, etc.


## Future TODO
- Test on other hardware, i.e. NVIDIA Jetson and MacBook Metal.
- Further check the "invisible" effect of the page cache.
- Further check the effects of SIMD/AVX instructions and how it differs across hardware, esp. the Jetson.
