# Android/Termux Package Installation Reference

## Complete Package Installation Guide

### Packages That MUST Use `pkg` (C Extensions)

These packages contain native C/C++ code and MUST be installed via Termux's package manager:

| Package | Install Command | Why pkg Required |
|---------|----------------|------------------|
| numpy | `pkg install python-numpy` | C extensions for numerical computing |
| psutil | `pkg install python-psutil` | C extensions for system monitoring |
| pandas | `pkg install python-pandas` | Depends on numpy, has C extensions |
| matplotlib | `pkg install python-matplotlib` | C extensions for plotting |
| scipy | `pkg install python-scipy` | C extensions for scientific computing |
| pyyaml | `pkg install python-pyyaml` | C extensions for YAML parsing |

**Why pkg?**
- Pre-compiled for Android ARM64 architecture
- Avoids compilation errors
- Faster installation
- Guaranteed compatibility

### Packages That Can Use `pip`

These are pure Python packages or compile successfully from source:

| Package | Install Command | Notes |
|---------|----------------|-------|
| seaborn | `pip install seaborn` | Pure Python, uses matplotlib from pkg |
| jinja2 | `pip install jinja2` | Pure Python template engine |
| python-dotenv | `pip install python-dotenv` | Pure Python environment loader |
| huggingface-hub | `pip install huggingface-hub` | Requires rust (install via pkg) |
| llama-cpp-python | `pip install llama-cpp-python` | Compiles from source (10-15 min) |

### Packages to SKIP on Android

These are not needed or not available:

| Package | Reason to Skip |
|---------|---------------|
| pynvml | No NVIDIA GPU on Android |
| pytest | Only needed for development/testing |
| pytest-cov | Only needed for development/testing |
| pytest-mock | Only needed for development/testing |
| hypothesis | Only needed for development/testing |

## Complete Installation Sequence

### Step 1: Install Build Tools

```bash
pkg update && pkg upgrade
pkg install python python-pip git clang cmake binutils rust
```

### Step 2: Install Python Packages with C Extensions (via pkg)

```bash
pkg install python-numpy
pkg install python-psutil
pkg install python-pandas
pkg install python-matplotlib
pkg install python-scipy
pkg install python-pyyaml
```

**Or install all at once:**
```bash
pkg install python-numpy python-psutil python-pandas python-matplotlib python-scipy python-pyyaml
```

### Step 3: Install Pure Python Packages (via pip)

```bash
pip install --upgrade pip
pip install seaborn jinja2 python-dotenv huggingface-hub
```

### Step 4: Install llama-cpp-python (Compiles from Source)

```bash
pip install llama-cpp-python
```

**Note**: This takes 10-15 minutes as it compiles C++ code.

## Verification

After installation, verify all packages:

```bash
# Check pkg-installed packages
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import psutil; print('psutil:', psutil.__version__)"
python -c "import pandas; print('pandas:', pandas.__version__)"
python -c "import matplotlib; print('matplotlib:', matplotlib.__version__)"
python -c "import scipy; print('scipy:', scipy.__version__)"
python -c "import yaml; print('pyyaml: OK')"

# Check pip-installed packages
python -c "import seaborn; print('seaborn:', seaborn.__version__)"
python -c "import jinja2; print('jinja2:', jinja2.__version__)"
python -c "import dotenv; print('python-dotenv: OK')"
python -c "import huggingface_hub; print('huggingface-hub:', huggingface_hub.__version__)"
python -c "import llama_cpp; print('llama-cpp-python:', llama_cpp.__version__)"
```

## Common Errors and Solutions

### Error: "fatal error: 'Python.h' file not found"

**Cause**: Trying to compile a package with C extensions via pip

**Solution**: Use pkg instead
```bash
# Example: If trying to pip install numpy
pip uninstall numpy
pkg install python-numpy
```

### Error: "error: command 'clang' failed with exit status 1"

**Cause**: Compilation failure when using pip for packages with C extensions

**Solution**: Use pkg instead
```bash
# Example: If trying to pip install pandas
pip uninstall pandas
pkg install python-pandas
```

### Error: "No module named 'numpy'" (after pip install)

**Cause**: pip-installed numpy failed to compile properly

**Solution**: Remove and install via pkg
```bash
pip uninstall numpy
pkg install python-numpy
```

### Error: "ImportError: cannot import name 'psutil'"

**Cause**: pip-installed psutil is incompatible

**Solution**: Remove and install via pkg
```bash
pip uninstall psutil
pkg install python-psutil
```

## How to Check if pkg Version Exists

Before trying pip, check if a pkg version exists:

```bash
# Search for package
pkg search python-<package-name>

# Example:
pkg search python-numpy
pkg search python-requests
```

If a `python-<package>` version exists in pkg, **always use that instead of pip**.

## Rule of Thumb

```
┌─────────────────────────────────────────┐
│  Does the package have C extensions?    │
│  (numpy, pandas, scipy, matplotlib,     │
│   psutil, pyyaml, pillow, etc.)         │
└─────────────────┬───────────────────────┘
                  │
         ┌────────┴────────┐
         │                 │
        YES               NO
         │                 │
         ▼                 ▼
   Use pkg install   Use pip install
   python-<package>  <package>
```

## Package Dependency Tree

```
llm-benchmark
├── numpy (pkg) ← Base dependency
│   ├── pandas (pkg)
│   ├── matplotlib (pkg)
│   │   └── seaborn (pip)
│   └── scipy (pkg)
├── psutil (pkg)
├── pyyaml (pkg)
├── jinja2 (pip)
├── python-dotenv (pip)
├── huggingface-hub (pip, requires rust)
└── llama-cpp-python (pip, compiles from source)
```

## Summary

**Install via pkg (6 packages):**
```bash
pkg install python-numpy python-psutil python-pandas python-matplotlib python-scipy python-pyyaml
```

**Install via pip (5 packages):**
```bash
pip install seaborn jinja2 python-dotenv huggingface-hub llama-cpp-python
```

**Skip (5 packages):**
- pynvml (no NVIDIA GPU)
- pytest, pytest-cov, pytest-mock, hypothesis (testing only)

**Total time:**
- pkg installs: ~2-5 minutes
- pip installs: ~15-20 minutes (mostly llama-cpp-python)
- **Total: ~20-25 minutes**
