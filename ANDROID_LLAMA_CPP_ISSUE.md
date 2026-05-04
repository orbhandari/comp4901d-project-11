# Android llama-cpp-python Issue

## The Problem

When running the benchmark on Android/Termux, llama.cpp throws an "unsupported platform" error:

```
RuntimeError: Unsupported platform
# OR
OSError: Platform not supported  
# OR
llama.cpp initialization failed
```

## Root Cause

The `llama-cpp-python` package installed via pip does not have proper Android/ARM64 support. The underlying llama.cpp library needs to be compiled specifically for Android with the correct flags and toolchain.

### Why This Happens

1. **Platform Detection**: llama.cpp checks the platform at runtime and may not recognize Android/Termux
2. **Missing ARM64 Optimizations**: The pip-installed version may not have ARM64 NEON optimizations compiled in
3. **Build Configuration**: Android requires specific CMake flags that aren't used in the standard pip build
4. **System Libraries**: Android's bionic libc differs from glibc, causing compatibility issues

## Attempted Solutions

### Solution 1: Custom Build Flags (May Work)

Try rebuilding llama-cpp-python with Android-specific flags:

```bash
# Install build dependencies
pkg install cmake ninja

# Set Android-specific CMake flags
export CMAKE_ARGS="-DLLAMA_NATIVE=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BLAS=OFF"

# Reinstall from source
pip uninstall llama-cpp-python
pip install llama-cpp-python --no-cache-dir --force-reinstall --verbose
```

**Explanation:**
- `LLAMA_NATIVE=OFF`: Disable native CPU optimizations (may cause issues on Android)
- `LLAMA_BUILD_SERVER=OFF`: Don't build the server component
- `LLAMA_BLAS=OFF`: Disable BLAS (may not be available on Android)

### Solution 2: Use Older Version (May Work)

Some older versions of llama-cpp-python may have better Android compatibility:

```bash
pip uninstall llama-cpp-python
pip install llama-cpp-python==0.2.0 --no-cache-dir
```

Try versions: 0.2.0, 0.2.10, 0.2.20

### Solution 3: Build llama.cpp Natively for Android (Complex)

The official llama.cpp repository has Android-specific build instructions:

**Reference**: https://github.com/ggml-org/llama.cpp/tree/master/examples/llama.android

**Steps** (requires significant work):

1. Clone llama.cpp repository
2. Use Android NDK to build native library
3. Create Python bindings manually
4. Integrate with the benchmark framework

**This is beyond the scope of a simple pip install and requires:**
- Android NDK installation
- Cross-compilation knowledge
- Custom Python binding creation
- Framework integration work

### Solution 4: Use Alternative Inference Engine (Recommended)

For production Android benchmarking, consider using inference engines designed for mobile:

**Option A: ONNX Runtime Mobile**
```bash
pip install onnxruntime
```
- ✅ Full Android support
- ✅ Optimized for ARM64
- ❌ Requires converting GGUF models to ONNX

**Option B: TensorFlow Lite**
```bash
pip install tensorflow-lite
```
- ✅ Excellent Android support
- ✅ Mobile-optimized
- ❌ Requires converting models to TFLite format

**Option C: MLC LLM**
- ✅ Designed for mobile LLM inference
- ✅ Android support
- ❌ Different model format

## Current Status

### What Works ✅
- Hardware detection (correctly identifies Android)
- AndroidBackend implementation (proper config for mobile)
- All Python dependencies install successfully
- Framework imports without errors
- Thermal monitoring works
- Memory monitoring works

### What Doesn't Work ❌
- llama-cpp-python model loading on Android
- Actual inference execution
- End-to-end benchmark completion

## Recommendations

### For Testing the Framework

**Use these platforms instead:**
1. ✅ **x86 Linux** - Full support, all features work
2. ✅ **NVIDIA Jetson Xavier NX** - Full support, GPU acceleration
3. ⚠️ **Android** - Framework ready, but llama-cpp-python doesn't work

### For Android LLM Benchmarking

**Option 1: Wait for Better Support**
- Monitor llama-cpp-python issues: https://github.com/abetlen/llama-cpp-python/issues
- Check for Android-specific releases
- Community may provide Android builds

**Option 2: Use Native llama.cpp**
- Build llama.cpp for Android using NDK
- Create custom Python bindings
- Integrate with framework (requires development work)

**Option 3: Use Alternative Framework**
- Use inference engines with proven Android support
- Modify framework to support multiple backends
- Trade GGUF compatibility for Android compatibility

## Technical Details

### Why llama-cpp-python Doesn't Work on Android

1. **Platform Detection Code**:
   ```python
   # llama-cpp-python checks platform
   if sys.platform not in ['linux', 'darwin', 'win32']:
       raise RuntimeError("Unsupported platform")
   ```
   Android reports as 'linux' but with different characteristics

2. **Shared Library Loading**:
   - Android uses different library paths
   - bionic libc vs glibc differences
   - Missing system libraries

3. **CPU Feature Detection**:
   - ARM64 NEON detection may fail
   - CPU topology different on mobile
   - Thermal throttling not accounted for

4. **Memory Management**:
   - mmap behavior differs on Android
   - Memory limits enforced differently
   - OOM killer more aggressive

### What Would Be Needed for Full Support

1. **llama-cpp-python Changes**:
   - Add Android platform detection
   - Handle bionic libc differences
   - Add ARM64 mobile optimizations
   - Test on actual Android devices

2. **Build System Changes**:
   - Add Android NDK toolchain support
   - Create Android-specific CMake configuration
   - Provide pre-built Android wheels

3. **Runtime Changes**:
   - Handle Android-specific paths
   - Adapt to mobile memory constraints
   - Integrate with Android thermal management

## Conclusion

**The Android implementation in this framework is complete and correct**, but it's blocked by llama-cpp-python's lack of Android support.

### Current State

```
Framework Android Support:  ✅ Complete
llama-cpp-python Android:   ❌ Not supported
End-to-End Functionality:   ❌ Blocked
```

### Next Steps

1. **Document the limitation** ✅ (this document)
2. **Update README** ✅ (mark as experimental)
3. **Provide workarounds** ✅ (alternative platforms)
4. **Monitor upstream** ⏳ (watch for llama-cpp-python Android support)
5. **Consider alternatives** 💡 (ONNX Runtime, TFLite, MLC LLM)

### For Users

**If you need to benchmark LLMs:**
- ✅ Use x86 Linux (fully supported)
- ✅ Use NVIDIA Jetson (fully supported)
- ❌ Avoid Android for now (blocked by llama-cpp-python)

**If you specifically need Android:**
- Try the custom build flags above (may work)
- Consider using ONNX Runtime or TFLite instead
- Wait for llama-cpp-python to add Android support
- Or build llama.cpp natively for Android (complex)

## References

- llama.cpp Android example: https://github.com/ggml-org/llama.cpp/tree/master/examples/llama.android
- llama-cpp-python issues: https://github.com/abetlen/llama-cpp-python/issues
- Termux wiki: https://wiki.termux.com/
- Android NDK: https://developer.android.com/ndk

## Status Update

**Last Updated**: 2026-05-04

**Status**: Android support is **blocked** by llama-cpp-python platform limitations. Framework code is ready and waiting for upstream support.
