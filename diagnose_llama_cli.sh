#!/bin/bash
# Diagnostic script to understand llama-cli behavior

echo "=========================================="
echo "Diagnosing llama-cli on Android"
echo "=========================================="
echo ""

LLAMA_CLI=~/llama.cpp/build/bin/llama-cli
MODEL=~/storage/shared/models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf

echo "1. Checking if llama-cli exists..."
if [ -f "$LLAMA_CLI" ]; then
    echo "✓ Found: $LLAMA_CLI"
else
    echo "✗ Not found: $LLAMA_CLI"
    exit 1
fi

echo ""
echo "2. Checking llama-cli version..."
$LLAMA_CLI --version 2>&1 | head -5

echo ""
echo "3. Checking available flags..."
echo "Looking for non-interactive flags:"
$LLAMA_CLI --help 2>&1 | grep -i -E "(interactive|conversation|chat|simple|once|single)" | head -10

echo ""
echo "4. Checking for 'main' binary (older llama.cpp)..."
LLAMA_MAIN=~/llama.cpp/build/bin/main
if [ -f "$LLAMA_MAIN" ]; then
    echo "✓ Found older 'main' binary: $LLAMA_MAIN"
    echo "  This might work better than llama-cli"
else
    echo "✗ No 'main' binary found"
fi

echo ""
echo "5. Checking for 'llama-simple' binary..."
LLAMA_SIMPLE=~/llama.cpp/build/bin/llama-simple
if [ -f "$LLAMA_SIMPLE" ]; then
    echo "✓ Found: $LLAMA_SIMPLE"
else
    echo "✗ No 'llama-simple' binary found"
fi

echo ""
echo "6. Listing all llama binaries..."
ls -lh ~/llama.cpp/build/bin/ | grep llama

echo ""
echo "7. Testing llama-cli with timeout (will kill after 5 seconds)..."
echo "Command: timeout 5s $LLAMA_CLI -m $MODEL -n 10 -p 'Hello' --log-disable -ngl 0"
echo "Output:"
echo "---"
timeout 5s $LLAMA_CLI -m $MODEL -n 10 -p "Hello" --log-disable -ngl 0 2>&1 || echo "[Process killed by timeout]"
echo "---"

echo ""
echo "8. Checking if --conversation flag exists..."
$LLAMA_CLI --help 2>&1 | grep -i "conversation"

echo ""
echo "9. Checking if -cnv flag exists..."
$LLAMA_CLI --help 2>&1 | grep -E "^\s+-cnv"

echo ""
echo "=========================================="
echo "Diagnosis complete!"
echo "=========================================="
