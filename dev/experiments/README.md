# Testing Directory

This directory contains development and testing scripts organized by purpose. **All temporary test files should be placed here**, not in the project root.

## Directory Structure

```
testing/
├── debug/                   # Debugging and diagnostic scripts
├── comparisons/             # Performance and accuracy comparison tests
├── standalone/              # Isolated functionality tests
├── mlx/                    # MLX-specific tests and validation
├── temporary/              # Temporary tests and experiments
├── results/                # Test output and result files
├── mlx_official_comparison/ # Official MLX comparison files and results
└── README.md               # This file
```

## Directory Purposes

### 🐛 `debug/` - Debugging and Diagnostic Scripts
Scripts for debugging specific components and diagnosing issues.

**Contents:**
- `debug_base_model.py` - Base model diagnostics
- `debug_generation.py` - Text generation debugging
- `debug_lora_generation.py` - LoRA fine-tuned model generation debugging
- `debug_lora_loading.py` - LoRA weight loading diagnostics
- `debug_model_state.py` - Model state inspection
- `debug_training_data.py` - Training data validation

**Usage:** Run when you need to debug specific functionality or diagnose issues.

### 📊 `comparisons/` - Performance and Accuracy Tests
Comprehensive tests comparing our implementation against official implementations or baselines.

**Contents:**
- `official_mlx_comparison.py` - **IMPORTANT**: Compares against official MLX implementation
- `test_mlx_training_comparison.py` - Training performance comparisons
- `test_model_loading_comparison.py` - Model loading benchmarks
- `test_lora_vs_base_accuracy.py` - LoRA vs base model accuracy comparison

**Usage:** Run to validate that our implementation performs correctly compared to official implementations.

### 🧪 `standalone/` - Isolated Functionality Tests
Tests that run specific functionality in isolation, useful for development and CI.

**Contents:**
- `test_base_model_standalone.py` - Base model functionality test
- `test_tinyllama_chat_standalone.py` - TinyLlama chat functionality
- `test_tinyllama_transformers.py` - Transformers integration test

**Usage:** Run to test specific components without dependencies on the full system.

### 🚀 `mlx/` - MLX-Specific Tests
Tests specifically for MLX functionality, format compatibility, and MLX-native features.

**Contents:**
- `test_direct_mlx_format.py` - **IMPORTANT**: Tests MLX format compatibility
- `test_e2e_mlx_data.py` - End-to-end MLX data pipeline test
- `test_mlx_data_loader.py` - MLX data loading validation

**Usage:** Run to validate MLX-specific functionality and compatibility.

### 🗂️ `temporary/` - Temporary Tests and Experiments
Temporary test files, experiments, and one-off scripts. **Clean this directory regularly.**

**Contents:**
- Various temporary test files from development
- Experimental scripts
- Quick validation tests

**Usage:** Place temporary test files here. Review and clean up regularly.

### 📋 `results/` - Test Output and Result Files
Storage for test outputs, results, and generated data files.

**Contents:**
- `generation_test_results.json` - Text generation test results
- `generation_debug_results.json` - Debugging session results
- Other test output files

**Usage:** Store test results and output files here instead of root directory.

### 🏛️ `mlx_official_comparison/` - Official MLX Comparison
Complete official MLX implementation downloaded for comparison testing.

**Contents:**
- Official MLX LoRA implementation files
- Official training data
- Comparison results and artifacts

**Usage:** Reference implementation for validation testing.

## Guidelines for Test Organization

### ✅ **DO**: Place New Tests in Appropriate Categories
- **Debugging a specific issue?** → `debug/`
- **Comparing implementations?** → `comparisons/`
- **Testing isolated functionality?** → `standalone/`
- **MLX-specific testing?** → `mlx/`
- **Quick temporary test?** → `temporary/`

### ✅ **DO**: Use Descriptive Filenames
- `debug_lora_weights.py` (clear purpose)
- `test_official_vs_our_implementation.py` (comparison)
- `standalone_tokenizer_test.py` (isolated test)

### ❌ **DON'T**: Put Test Files in Project Root
- Keep the root directory clean
- All tests belong in `testing/` subdirectories
- Use the main `tests/` directory for official test suite only

### 🧹 **Regular Cleanup**
- Review `temporary/` directory monthly
- Move useful temporary tests to appropriate permanent categories
- Delete obsolete test files
- Update this README when adding new categories

## Running Tests

### Individual Tests
```bash
# Run a specific debug script
python testing/debug/debug_lora_generation.py

# Run a comparison test
python testing/comparisons/official_mlx_comparison.py

# Run MLX format test
python testing/mlx/test_direct_mlx_format.py
```

### Category-wide Testing
```bash
# Run all comparison tests
for test in testing/comparisons/*.py; do python "$test"; done

# Run all MLX tests
for test in testing/mlx/*.py; do python "$test"; done
```

## Important Tests for Validation

### 🔴 **Critical Tests** (Run Before Major Changes)
1. `comparisons/official_mlx_comparison.py` - Validates against official MLX
2. `mlx/test_direct_mlx_format.py` - Tests MLX format compatibility
3. `comparisons/test_lora_vs_base_accuracy.py` - Validates LoRA training works

### 🟡 **Development Tests** (Run During Development)
1. `debug/debug_lora_generation.py` - Debug LoRA issues
2. `standalone/test_base_model_standalone.py` - Test model loading
3. `mlx/test_e2e_mlx_data.py` - Test data pipeline

## Adding New Tests

When creating a new test file:

1. **Choose the right directory** based on purpose
2. **Use descriptive naming** that explains the test's purpose
3. **Include docstring** explaining what the test validates
4. **Update this README** if creating a new category

Example:
```python
#!/usr/bin/env python3
"""
Test LoRA weight serialization and deserialization.

This test validates that LoRA weights can be saved and loaded correctly,
maintaining the same model performance before and after serialization.
"""
```

Remember: **Keep the project root clean** - all test files belong in `testing/`!