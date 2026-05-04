# Android Model Paths Configuration

## Issue

The framework was trying to download models from Hugging Face even though you already downloaded them, because the config was pointing to the wrong directory.

## Root Cause

The config had:
```json
"model_cache_dir": "./models"
```

This is a **relative path** to `./models` (in the current directory), but you put the models in `~/storage/shared/models/`.

## Solution

Updated the config to use the correct path:
```json
"model_cache_dir": "~/storage/shared/models"
```

## Config Files Updated

1. **`configs/android_example.json`** - Updated to use `~/storage/shared/models`
2. **`configs/android_config.json`** - New config with correct paths

## For Android Users

### Option 1: Use the Updated Config (Recommended)

```bash
cd ~/comp4901d-project-11
git pull

# Make sure your models are in the right place
ls -lh ~/storage/shared/models/
# Should show:
#   tinyllama-1.1b-chat-v1.0.Q2_K.gguf
#   tinyllama-1.1b-chat-v1.0.Q4_0.gguf

# Run with the updated config
python -m llm_benchmark --config configs/android_config.json
```

### Option 2: Move Models to Match Config

If you prefer to keep models in the repo directory:

```bash
cd ~/comp4901d-project-11

# Create models directory
mkdir -p models

# Move or copy models
cp ~/storage/shared/models/*.gguf models/

# Run with original config
python -m llm_benchmark --config configs/android_example.json
```

### Option 3: Specify Path on Command Line

You can override the config path:

```bash
python -m llm_benchmark \
  --config configs/android_example.json \
  --model-cache-dir ~/storage/shared/models
```

## Verifying Model Paths

Before running the benchmark, verify the models are where the config expects:

```bash
# Check what the config says
grep model_cache_dir configs/android_config.json

# Check if models exist at that path
ls -lh ~/storage/shared/models/*.gguf
```

## Directory Structure

### Recommended (Shared Storage)
```
~/storage/shared/
├── models/
│   ├── tinyllama-1.1b-chat-v1.0.Q2_K.gguf
│   ├── tinyllama-1.1b-chat-v1.0.Q4_0.gguf
│   └── tinyllama-1.1b-chat-v1.0.Q8_0.gguf
└── benchmark_results/
    └── run_TIMESTAMP/
        ├── results.json
        ├── results.csv
        └── results.md
```

**Advantages:**
- ✅ Accessible from Android file manager
- ✅ Easy to view results on phone
- ✅ Survives Termux reinstalls
- ✅ Can share files easily

### Alternative (Repo Directory)
```
~/comp4901d-project-11/
├── models/
│   ├── tinyllama-1.1b-chat-v1.0.Q2_K.gguf
│   └── tinyllama-1.1b-chat-v1.0.Q4_0.gguf
└── benchmark_results/
    └── run_TIMESTAMP/
```

**Advantages:**
- ✅ Everything in one place
- ✅ Easier to manage with git
- ✅ Portable (can zip entire directory)

## Downloading Models

If you need to download models:

```bash
# Create directory
mkdir -p ~/storage/shared/models
cd ~/storage/shared/models

# Download Q2_K (smallest, ~637MB)
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf

# Download Q4_0 (medium, ~669MB)
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf

# Download Q8_0 (largest, ~1.1GB)
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf

# Verify downloads
ls -lh *.gguf
```

## Checking Disk Space

Before downloading:

```bash
# Check available space
df -h ~/storage/shared

# Check model sizes
# Q2_K: ~637MB
# Q4_0: ~669MB
# Q8_0: ~1.1GB
```

## Summary

The config now points to `~/storage/shared/models` which is:
- ✅ Accessible from file manager
- ✅ Where you already put the models
- ✅ Persistent across Termux sessions
- ✅ Easy to manage

Just run:
```bash
python -m llm_benchmark --config configs/android_config.json
```

And it will find your models!

