#!/bin/bash
# Check which llama.cpp binaries are available

echo "=========================================="
echo "Checking llama.cpp Binaries"
echo "=========================================="
echo ""

LLAMA_DIR=~/llama.cpp/build/bin

echo "Looking in: $LLAMA_DIR"
echo ""

if [ ! -d "$LLAMA_DIR" ]; then
    echo "❌ Directory not found: $LLAMA_DIR"
    echo ""
    echo "You need to build llama.cpp first:"
    echo "  cd ~/llama.cpp"
    echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release"
    echo "  cmake --build build --config Release -j4"
    exit 1
fi

echo "Available binaries:"
echo "---"
ls -lh $LLAMA_DIR | grep -E "(main|llama-cli|llama-simple)" || echo "No llama binaries found"
echo "---"
echo ""

# Check each binary
echo "Binary Status:"
echo ""

if [ -f "$LLAMA_DIR/main" ]; then
    echo "✅ main - FOUND (RECOMMENDED - no conversation mode)"
    MAIN_EXISTS=1
else
    echo "❌ main - NOT FOUND"
    MAIN_EXISTS=0
fi

if [ -f "$LLAMA_DIR/llama-simple" ]; then
    echo "✅ llama-simple - FOUND"
else
    echo "❌ llama-simple - NOT FOUND"
fi

if [ -f "$LLAMA_DIR/llama-cli" ]; then
    echo "✅ llama-cli - FOUND (may have conversation mode issues)"
else
    echo "❌ llama-cli - NOT FOUND"
fi

echo ""
echo "=========================================="
echo "Priority Order (what the code will use):"
echo "=========================================="
echo "1. main (preferred)"
echo "2. llama-simple"
echo "3. llama-cli"
echo ""

if [ $MAIN_EXISTS -eq 1 ]; then
    echo "✅ GOOD: 'main' binary exists and will be used"
    echo "   This binary doesn't have conversation mode issues"
else
    echo "⚠️  WARNING: 'main' binary not found"
    echo "   The code will try llama-simple or llama-cli instead"
    echo "   These may have conversation mode issues"
    echo ""
    echo "   To build 'main' binary:"
    echo "     cd ~/llama.cpp"
    echo "     rm -rf build"
    echo "     cmake -B build -DCMAKE_BUILD_TYPE=Release"
    echo "     cmake --build build --config Release -j4"
fi

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
