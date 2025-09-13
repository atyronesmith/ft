# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) or other AI assistants when working with this repository.

## Project Overview

FineTune is a modular fine-tuning application optimized for Apple Silicon that enables efficient training of language models from HuggingFace on custom datasets.

## Current Implementation Status

### ✅ Phase 1 Complete (Current State)

#### Implemented Components
1. **Model Infrastructure**
   - `ModelManager`: Unified interface for all model operations
   - `MLXLlamaModel`, `MLXGPTModel`: Native MLX implementations
   - `MLXModelLoader`: HuggingFace → MLX conversion pipeline
   - `PyTorchModel`, `PyTorchModelLoader`: Fallback implementation

2. **Backend System**
   - Automatic backend selection (MLX → PyTorch MPS → CUDA → CPU)
   - Device management and capability detection
   - Memory monitoring and estimation

3. **Testing & Quality**
   - 66 unit tests passing (53% coverage)
   - Comprehensive linting setup (black, ruff, pylint, mypy)
   - Pre-commit hooks configured
   - 577 linting issues automatically fixed

4. **Weight Conversion**
   - PyTorch → MLX format conversion
   - Automatic weight transposition for linear layers
   - Name mapping for different conventions
   - Support for sharded models and safetensors

### 🚧 Phase 2 In Progress

- [ ] Data loading pipelines
- [ ] Training loops with LoRA/QLoRA
- [ ] CLI command completion
- [ ] Configuration system

## Common Development Commands

### Environment Setup
```bash
# Automated setup (recommended)
./setup.sh

# Manual setup
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
make dev
```

### Development Workflow
```bash
# Always activate venv first
source .venv/bin/activate

# Run tests
make test              # Unit tests only
make test-integration  # Integration tests
make test-all         # All tests with coverage

# Code quality
make format           # Format with black
make lint            # Run all linters

# Specific tools
black src/ tests/    # Format code
ruff check --fix     # Auto-fix linting issues
mypy src/           # Type checking
```

### Testing Specific Components
```bash
# Run specific test file
pytest tests/unit/test_mlx_models.py -v

# Run specific test
pytest tests/unit/test_mlx_models.py::TestMLXModels::test_llama_model_creation

# Debug test with output
pytest tests/unit/test_mlx_loader.py -xvs

# Check coverage
pytest --cov=finetune --cov-report=html tests/
open htmlcov/index.html
```

## Key Implementation Details

### Model Loading Flow
```python
# 1. User requests model
manager.load_model("meta-llama/Llama-2-7b-hf")

# 2. Manager checks cache
if model_in_cache:
    return cached_model

# 3. Download from HuggingFace
path = snapshot_download(model_id)

# 4. Load and convert weights
if backend == "mlx":
    weights = load_pytorch_weights(path)
    mlx_weights = convert_weights(weights)
    model.update(mlx_weights)

# 5. Cache and return
cache_model(model)
return model
```

### Weight Conversion Details
```python
# PyTorch linear: (out_features, in_features)
# MLX linear: (in_features, out_features)
# Therefore: transpose linear weights

if "q_proj" in name or "k_proj" in name:
    weight = weight.T  # Transpose for MLX
```

### Parameter Flattening (MLX Models)
```python
# MLX parameters are nested dicts/lists
# Flatten for saving:
"embed_tokens.weight" → mx.array(...)
"layers.0.self_attn.q_proj.weight" → mx.array(...)

# Unflatten for loading:
weights["layers"][0]["self_attn"]["q_proj"]["weight"]
```

## Common Issues & Solutions

### Issue: Tests Failing with Import Errors
```bash
# Solution: Ensure PYTHONPATH is set
export PYTHONPATH=src:$PYTHONPATH
# Or use Make which sets it automatically
make test
```

### Issue: MLX Not Available
```bash
# Check if on Apple Silicon
uname -m  # Should output 'arm64'

# Install MLX
pip install mlx

# Verify installation
python -c "import mlx; print(mlx.__version__)"
```

### Issue: Mock Iterator Problems in Tests
```python
# Wrong: Returns list
mock_model.parameters.return_value = [param1, param2]

# Correct: Returns iterator
mock_model.parameters = Mock(side_effect=lambda: iter([param1, param2]))
```

### Issue: Pre-commit Hooks Failing
```bash
# Skip temporarily
git commit --no-verify -m "message"

# Fix issues and reinstall
pre-commit uninstall
pre-commit install
pre-commit run --all-files
```

## Adding New Features

### Adding a New Model Architecture

1. **Create MLX Implementation**:
```python
# src/finetune/models/mlx_models.py
class MLXNewModel(nn.Module, BaseModel):
    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        BaseModel.__init__(self, config)
        # Implementation...
```

2. **Register in Model Registry**:
```python
MLX_MODEL_REGISTRY = {
    "new_model": MLXNewModel,
    # ...
}
```

3. **Add Weight Mapping**:
```python
# src/finetune/models/mlx_loader.py
def _get_name_mapping(self, model_type: str):
    if "new_model" in model_type:
        return {
            "transformer.h": "layers",
            # Map HuggingFace names to MLX names
        }
```

4. **Write Tests**:
```python
# tests/unit/test_mlx_models.py
def test_new_model_creation(self):
    config = ModelConfig(model_type="new_model", ...)
    model = get_mlx_model(config)
    assert model is not None
```

### Adding a New Backend

1. **Create Backend Class**:
```python
# src/finetune/backends/new_backend.py
class NewBackend(Backend):
    def is_available(self) -> bool:
        # Check if backend is available
    
    def get_device(self) -> str:
        # Return device string
```

2. **Register in Device Manager**:
```python
# src/finetune/backends/device.py
backends = [
    MLXBackend(),
    PyTorchBackend(),
    NewBackend(),  # Add here
]
```

## Performance Optimization Tips

### Memory Management
- Use gradient checkpointing for large models
- Clear cache between batches: `mx.clear_cache()`
- Use quantization: `load_in_4bit=True`
- Monitor memory: `mx.metal.get_active_memory()`

### Speed Optimization
- Batch operations when possible
- Use compiled functions in MLX
- Profile with: `python -m cProfile script.py`
- Use MLX's unified memory architecture advantage

## Code Style Guidelines

### Formatting Rules
- **Line length**: 100 characters max
- **Quotes**: Double quotes for strings
- **Imports**: Sorted with isort (stdlib → third-party → local)
- **Docstrings**: Google style

### Type Hints
```python
# Always use type hints
def process_weights(
    weights: dict[str, torch.Tensor],
    model_type: str,
) -> dict[str, mx.array]:
    ...
```

### Testing Standards
- Every new feature needs tests
- Use fixtures for reusable test data
- Mock external dependencies
- Maintain >50% coverage

## Debugging Techniques

### Useful Debug Commands
```bash
# Check what's imported
python -c "from finetune.models import *; print(dir())"

# Test model loading
python -c "
from finetune.models.manager import ModelManager
m = ModelManager()
print(m.list_models())
"

# Profile memory
python -m memory_profiler your_script.py

# Debug pytest
pytest --pdb  # Drop to debugger on failure
pytest --lf   # Run last failed tests
```

### Logging
```python
from loguru import logger

logger.debug("Detailed info for debugging")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred: {}", error)
```

## Project Structure Reference

```
src/finetune/
├── models/           # Core model implementations
│   ├── base.py      # Abstract base classes
│   ├── manager.py   # Model manager (main interface)
│   ├── mlx_models.py    # MLX implementations
│   ├── mlx_loader.py    # MLX loader/converter
│   └── torch_loader.py  # PyTorch fallback
├── backends/        # Backend abstraction
│   ├── base.py     # Backend interface
│   ├── device.py   # Device manager
│   ├── mlx_backend.py   # MLX backend
│   └── torch_backend.py # PyTorch backend
├── core/           # Core utilities
│   ├── config.py   # Configuration classes
│   └── registry.py # Model registry
├── cli/            # CLI (Phase 2)
├── training/       # Training (Phase 2)
└── data/          # Data loading (Phase 2)

tests/
├── unit/          # Unit tests (66 passing)
├── integration/   # Integration tests
└── conftest.py    # Shared fixtures
```

## Next Session Checklist

When resuming work:

1. **Check Status**:
   ```bash
   git status
   make test
   ```

2. **Review TODOs**:
   ```bash
   grep -r "TODO" src/
   grep -r "FIXME" src/
   ```

3. **Update Dependencies**:
   ```bash
   git pull
   pip install -e .
   ```

4. **Run Quality Checks**:
   ```bash
   make format
   make lint
   make test
   ```

## Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PyTorch MPS](https://pytorch.org/docs/stable/notes/mps.html)
- [Project Architecture](docs/design/MLX_ARCHITECTURE.md)

## Important Notes

- Always work in the virtual environment (`.venv/`)
- Run tests before committing
- Use pre-commit hooks (already installed)
- Follow existing code patterns
- Update tests when adding features
- Document significant changes