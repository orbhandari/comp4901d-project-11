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
x86_Arch_Linux_Laptop   Q8_0         0.44      8593.57          8494.06   780.31        15.38        4.51
x86_Arch_Linux_Laptop Q4_K_M         1.03      8337.31          8173.40   633.89        18.93        7.43

========================================
PROMPT CACHING ABLATION (RAM Cache)
========================================
Scenario: Shared Prefix (RAM)
TTFT_Cold_ms: 1990.45
TTFT_Warm_ms: 1522.43
Improvement_ms: 468.02
Reduction_Percent: 23.51

========================================
PROMPT CACHING ABLATION (Disk Cache)
========================================
Scenario: Shared Prefix (DISK)
TTFT_Cold_ms: 2582.98
TTFT_Warm_ms: 1585.11
Improvement_ms: 997.87
Reduction_Percent: 38.63
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
- Check RAM usage for ablation study as well.
- Test MORE prompts for caching study, to see the effects on RAM using RamCache VS DiskCache.
