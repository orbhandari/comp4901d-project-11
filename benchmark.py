import gc
import os
import platform
import shutil
import time

import pandas as pd
import psutil
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from llama_cpp import Llama, LlamaDiskCache, LlamaRAMCache

# ==========================================
# Configuration
# ==========================================
REPO_ID = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
MODELS = {
    "Q8_0": "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
    "Q4_K_M": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
}
MODEL_DIR = "./models"

DEVICE_NAME = "x86_Arch_Linux_Laptop"

CONTEXT_SIZE = 2048
BATCH_SIZE = 512

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


# ==========================================
# Device Information
# ==========================================
def print_device_info():
    print("\n" + "=" * 30)
    print("DEVICE INFORMATION")
    print("=" * 30)
    mem = psutil.virtual_memory()
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Processor: {platform.processor()}")
    print(f"Total RAM: {mem.total / (1024**3):.2f} GB")
    print(f"Available RAM: {mem.available / (1024**3):.2f} GB")
    print("=" * 30 + "\n")


# ==========================================
# Model Acquisition
# ==========================================
def download_models():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    local_paths = {}
    print("Verifying model availability...")
    for quant, filename in MODELS.items():
        expected_path = os.path.join(MODEL_DIR, filename)

        if os.path.exists(expected_path):
            print(f"  -> {quant} already exists at {expected_path}. Skipping download.")
            local_paths[quant] = expected_path
        else:
            print(f"  -> {quant} not found. Downloading from Hugging Face...")
            path = hf_hub_download(
                repo_id=REPO_ID, filename=filename, local_dir=MODEL_DIR
            )
            local_paths[quant] = path
    return local_paths


# ==========================================
# Quantization Profiling
# ==========================================
def run_quant_benchmark(model_path, quant):
    print(f"\nProfiling {quant}...")

    process = psutil.Process()
    baseline_mem = process.memory_info().rss / (1024 * 1024)

    start_load = time.perf_counter()

    llm = Llama(
        model_path=model_path,
        n_ctx=CONTEXT_SIZE,  # This is currently set to the default value
        n_batch=BATCH_SIZE,  # This is currently set to the default value
        verbose=False,
    )

    load_time_s = time.perf_counter() - start_load

    prompt = "Explain the history of the Linux kernel in detail."
    prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))

    _ = llm("Warmup", max_tokens=5)

    start_eval = time.perf_counter()
    streamer = llm(prompt, max_tokens=100, stream=True)

    ttft_s = 0
    decode_start_time = 0
    tokens_generated = 0

    for chunk in streamer:
        if ttft_s == 0:
            ttft_s = time.perf_counter() - start_eval
            decode_start_time = time.perf_counter()
        tokens_generated += 1

    end_eval = time.perf_counter()

    decode_duration = end_eval - decode_start_time
    if decode_duration <= 0:
        decode_tps = 0
    else:
        decode_tps = (tokens_generated - 1) / decode_duration

    peak_mem = process.memory_info().rss / (1024 * 1024)
    mem_increase = peak_mem - baseline_mem
    prefill_tps = prompt_tokens / ttft_s if ttft_s > 0 else 0

    return {
        "Device": DEVICE_NAME,
        "Quant": quant,
        "Load_Time_s": round(load_time_s, 2),
        "Peak_RAM_MB": round(peak_mem, 2),
        "RAM_Increase_MB": round(mem_increase, 2),
        "TTFT_ms": round(ttft_s * 1000, 2),
        "Prefill_tps": round(prefill_tps, 2),
        "Decode_tps": round(decode_tps, 2),
    }


# Caching Ablation
# ==========================================
def run_caching_ablation(model_path, cache_type="ram"):
    print(f"\nStarting Prompt Caching Ablation Study ({cache_type.upper()} Cache)...")

    llm = Llama(model_path=model_path, n_ctx=CONTEXT_SIZE, verbose=False)

    if cache_type == "disk":
        cache_dir = "llm_cache_test"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)
        cache = LlamaDiskCache(cache_dir=cache_dir)
    elif cache_type == "ram":
        cache = LlamaRAMCache(capacity_bytes=2 * (1 << 30))
    else:
        raise ValueError("cache_type must be either 'ram' or 'disk'")

    llm.set_cache(cache)

    prompt_1 = "I am a university student. Please help me complete X."
    prompt_2 = "I am a university student. Please help me complete Y."

    print(f"  -> Testing Cold Cache ({cache_type.upper()})...")
    start_1 = time.time()
    llm(prompt_1, max_tokens=10)
    ttft_cold = (time.time() - start_1) * 1000

    print(f"  -> Testing Warm Cache ({cache_type.upper()})...")
    start_2 = time.time()
    llm(prompt_2, max_tokens=10)
    ttft_warm = (time.time() - start_2) * 1000

    if cache_type == "disk" and os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    return {
        "Scenario": f"Shared Prefix ({cache_type.upper()})",
        "TTFT_Cold_ms": round(ttft_cold, 2),
        "TTFT_Warm_ms": round(ttft_warm, 2),
        "Improvement_ms": round(ttft_cold - ttft_warm, 2),
        "Reduction_Percent": round(((ttft_cold - ttft_warm) / ttft_cold) * 100, 2),
    }


# ==========================================
# Main Orchestrator
# ==========================================
if __name__ == "__main__":
    print_device_info()
    paths = download_models()

    matrix_data = []
    for q_name, q_path in paths.items():
        matrix_data.append(run_quant_benchmark(q_path, q_name))

        gc.collect()
        time.sleep(2)

    ram_results = run_caching_ablation(paths["Q4_K_M"], cache_type="ram")
    disk_results = run_caching_ablation(paths["Q4_K_M"], cache_type="disk")

    print("\n" + "=" * 40)
    print("FINAL BENCHMARK MATRIX")
    print("=" * 40)
    print(pd.DataFrame(matrix_data).to_string(index=False))

    print("\n" + "=" * 40)
    print("PROMPT CACHING ABLATION (RAM Cache)")
    print("=" * 40)
    for k, v in ram_results.items():
        print(f"{k}: {v}")

    print("\n" + "=" * 40)
    print("PROMPT CACHING ABLATION (Disk Cache)")
    print("=" * 40)
    for k, v in disk_results.items():
        print(f"{k}: {v}")
