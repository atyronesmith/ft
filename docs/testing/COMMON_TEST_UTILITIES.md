# Common Test Utilities

This document describes the common test utilities that have been extracted to reduce code duplication across the FineTune test suite.

## Overview

The test utilities are located in `tests/utils/` and provide reusable functionality for:

- Model configuration creation
- Mock object generation
- Test data creation
- File and directory helpers
- Environment checks
- Common assertions

## Modules

### tests.utils.test_helpers

Main module containing all the helper classes and utilities.

#### ModelConfigFactory

Creates test model configurations with consistent, test-friendly parameters.

```python
from tests.utils import ModelConfigFactory

# Create different types of configs
small_config = ModelConfigFactory.create_small_config()        # Small config for fast tests
sample_config = ModelConfigFactory.create_sample_config()      # Full-sized config
gpt_config = ModelConfigFactory.create_gpt_config()           # GPT-specific config
```

#### MockFactory

Creates mock objects commonly used in tests.

```python
from tests.utils import MockFactory

# Create mock objects
mock_mlx = MockFactory.create_mock_mlx()
mock_torch = MockFactory.create_mock_torch()
mock_weights = MockFactory.create_mock_pytorch_weights()
```

#### DatasetFactory

Creates test datasets for different scenarios.

```python
from tests.utils import DatasetFactory

# Create different types of datasets
sample_data = DatasetFactory.create_sample_dataset(size=5)     # General instruction-following
qa_data = DatasetFactory.create_qa_dataset(size=10)           # Q&A about capitals
math_data = DatasetFactory.create_math_dataset(size=5)        # Simple math problems
```

#### FileHelper

Helpers for creating temporary files and mock directories.

```python
from tests.utils import FileHelper

# Create temporary files
json_file = FileHelper.create_temp_json_file(data)
jsonl_file = FileHelper.create_temp_jsonl_file(data)

# Create mock model directories
hf_dir = FileHelper.create_mock_model_directory(temp_dir, "llama")
mlx_dir = FileHelper.create_mock_mlx_model_directory(temp_dir, "llama")
```

#### TestEnvironment

Environment checking and test skipping utilities.

```python
from tests.utils import TestEnvironment

# Check availability
if TestEnvironment.mlx_available():
    # MLX-specific code

# Skip decorators
@TestEnvironment.skip_if_no_mlx()
def test_mlx_functionality():
    # This test only runs if MLX is available
    pass

# Verbose printing
TestEnvironment.verbose_print("Debug message", enabled=True)
```

#### AssertionHelpers

Common assertion patterns used across tests.

```python
from tests.utils import AssertionHelpers

# Validate common objects
AssertionHelpers.assert_model_config_valid(config)
AssertionHelpers.assert_dataset_valid(dataset)
AssertionHelpers.assert_loss_convergence(losses, min_reduction=0.05)
```

## Benefits

### Before Refactoring

Each test file duplicated similar functionality:

```python
# In test_mlx_models.py
@pytest.fixture
def small_config(self):
    return ModelConfig(
        model_type="llama",
        vocab_size=100,
        hidden_size=64,
        # ... many lines of repeated config
    )

# In test_mlx_loader.py
@pytest.fixture
def small_config(self):
    return ModelConfig(
        model_type="llama",
        vocab_size=1000,  # Different values!
        hidden_size=128,
        # ... many lines of repeated config
    )
```

### After Refactoring

Single source of truth with consistent parameters:

```python
# In any test file
from tests.utils import ModelConfigFactory

def test_something():
    config = ModelConfigFactory.create_small_config()
    # Always consistent, well-tested config
```

## Migration Guide

### Updating Existing Tests

1. **Replace duplicated fixtures:**
   ```python
   # Before
   @pytest.fixture
   def small_config(self):
       return ModelConfig(...)

   # After
   from tests.utils import ModelConfigFactory

   @pytest.fixture
   def small_config(self):
       return ModelConfigFactory.create_small_config()
   ```

2. **Replace manual mock creation:**
   ```python
   # Before
   mock_mlx = MagicMock()
   mock_mlx.core = MagicMock()
   # ... many lines

   # After
   from tests.utils import MockFactory

   mock_mlx = MockFactory.create_mock_mlx()
   ```

3. **Replace manual data creation:**
   ```python
   # Before
   data = [
       {"instruction": "What is...", "output": "..."},
       # ... many lines
   ]

   # After
   from tests.utils import DatasetFactory

   data = DatasetFactory.create_sample_dataset(size=5)
   ```

4. **Replace environment checks:**
   ```python
   # Before
   def _mlx_available():
       try:
           import mlx
           return True
       except ImportError:
           return False

   # After
   from tests.utils import TestEnvironment

   # Use TestEnvironment.mlx_available() directly
   ```

## Common Patterns

### Test Class Setup

```python
from tests.utils import ModelConfigFactory, DatasetFactory, AssertionHelpers

class TestMyFeature:
    def test_basic_functionality(self):
        # Setup
        config = ModelConfigFactory.create_small_config()
        data = DatasetFactory.create_sample_dataset(size=3)

        # Test logic here

        # Assertions using helpers
        AssertionHelpers.assert_model_config_valid(config)
        AssertionHelpers.assert_dataset_valid(data)
```

### Environment-Dependent Tests

```python
from tests.utils import TestEnvironment

class TestMLXFeatures:
    @TestEnvironment.skip_if_no_mlx()
    def test_mlx_functionality(self):
        # Only runs if MLX available
        TestEnvironment.verbose_print("Testing MLX features")
        # Test logic...
```

### File Operations

```python
from tests.utils import FileHelper, DatasetFactory

def test_file_loading(tmp_path):
    # Create test data and files
    data = DatasetFactory.create_qa_dataset(size=5)
    json_file = FileHelper.create_temp_json_file(data)
    model_dir = FileHelper.create_mock_model_directory(tmp_path)

    try:
        # Test logic using files
        pass
    finally:
        # Cleanup handled by helpers
        os.unlink(json_file)
```

## Adding New Common Functionality

When you identify new patterns to extract:

1. Add the functionality to the appropriate class in `tests/utils/test_helpers.py`
2. Add imports to `tests/utils/__init__.py` if needed
3. Update this documentation
4. Create tests in `tests/test_refactoring_example.py` to validate the new functionality

## Testing the Utilities

Run the example tests to validate the utilities work correctly:

```bash
# Test all utility examples
python -m pytest tests/test_refactoring_example.py -v

# Test specific functionality
python -m pytest tests/test_refactoring_example.py::TestRefactoredExamples::test_model_config_creation -v
```

## Benefits Summary

1. **Consistency**: All tests use the same well-tested configurations and data
2. **Maintainability**: Fix issues in one place, benefit all tests
3. **Readability**: Tests focus on what they're testing, not setup boilerplate
4. **Reliability**: Common utilities are well-tested and handle edge cases
5. **Speed**: Reusable patterns speed up test development
6. **Documentation**: Self-documenting through method names and clear interfaces

The common test utilities make the test suite more maintainable and ensure consistency across all tests. When you fix a bug in the common utilities, it automatically fixes potential issues in all tests that use them.
