# Android Quick Start Guide

Get the LLM Benchmark Framework running on Android in 30 minutes!

## Prerequisites

- Android device with 6GB+ RAM
- Termux installed from F-Droid (NOT Google Play)
- Stable internet connection
- ~10GB free storage

## Quick Setup (Copy-Paste Commands)

### 1. Update Termux

```bash
pkg update && pkg upgrade -y
```

### 2. Install All Dependencies

```bash
# Install build tools
pkg install python python-pip git cmake clang binutils rust

# Install Python packages with C extensions (via pkg, NOT pip)
pkg install python-numpy python-psutil python-pandas python-matplotlib python-scipy python-pyyaml

# Install pure Python packages (via pip)
pip install --upgrade pip
pip install seaborn jinja2 python-dotenv huggingface-hub
```

### 3. Build Native llama.cpp (~30 minutes)

```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4

# Verify build
ls -lh build/bin/llama-cli
```

### 4. Clone Benchmark Repository

```bash
cd ~
git clone <your-repo-url> comp4901d-project-11
cd comp4901d-project-11
```

### 5. Setup Hugging Face Token

```bash
echo "HF_TOKEN=your_token_here" > .env
```

### 6. Download Test Model

```bash
mkdir -p ~/storage/shared/models
cd ~/storage/shared/models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf
```

### 7. Run Benchmark

```bash
cd ~/comp4901d-project-11
python -m llm_benchmark --config configs/android_example.json
```

## Test Native llama.cpp (Optional)

Before running the full benchmark, test llama.cpp directly:

```bash
~/llama.cpp/build/bin/llama-cli \
  -m ~/storage/shared/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf \
  -c 512 \
  -n 20 \
  -p "Hello, how are you?"
```

If this works, the benchmark will work too!

## Troubleshooting

### "llama-cli not found"

```bash
# Rebuild llama.cpp
cd ~/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4
```

### "Insufficient RAM"

- Close background apps
- Use Q2_K quantization
- Reduce context_size to 256

### "Device overheating"

- Remove phone case
- Let device cool down
- Increase sleep_between_tests_s to 30

### Package installation fails

**Rule**: Packages with C extensions MUST use `pkg`, not pip!

```bash
# ✅ Correct
pkg install python-numpy python-psutil

# ❌ Wrong
pip install numpy psutil
```

## What's Next?

- View results in `~/storage/shared/benchmark_results/`
- Open HTML report in browser
- Try different quantizations (Q2_K, Q4_0, Q8_0)
- Adjust configuration for your device

## Full Documentation

See [docs/ANDROID_SETUP.md](docs/ANDROID_SETUP.md) for:
- Detailed explanations
- Performance expectations
- Advanced optimization tips
- Complete troubleshooting guide

## How It Works

The framework automatically:
1. Detects you're on Android
2. Looks for `~/llama.cpp/build/bin/llama-cli`
3. Uses native llama.cpp instead of llama-cpp-python
4. Provides the same interface for compatibility
5. Runs all benchmarks normally

No special configuration needed - just build llama.cpp and go!

