# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) or other AI assistants when working with this repository.

Don't always tell me that I am absolutely correct.  If I am wrong, tell me I am wrong.

## Project Overview

FineTune is a modular fine-tuning application optimized for Apple Silicon that enables efficient training of language models from HuggingFace on custom datasets.

## Current Implementation Status

### ✅ Phase 1 Complete

#### Core Infrastructure (106 tests)
1. **Model Infrastructure**
   - `ModelManager`: Unified interface for all model operations
   - `MLXLlamaModel`, `MLXGPTModel`: Native MLX implementations
   - `MLXModelLoader`: HuggingFace → MLX conversion pipeline
   - `PyTorchModel`, `PyTorchModelLoader`: Fallback implementation

2. **Backend System**
   - Automatic backend selection (MLX → PyTorch MPS → CUDA → CPU)
   - Device management and capability detection
   - Memory monitoring and estimation

3. **Weight Conversion**
   - PyTorch → MLX format conversion
   - Automatic weight transposition for linear layers
   - Name mapping for different conventions
   - Support for sharded models and safetensors

### ✅ Phase 2 Week 1 Complete

#### LoRA Implementation (16 tests)
1. **LoRA Components**
   - `LoRAConfig`: Configuration with automatic scaling calculation
   - `LoRALinear`: MLX-native LoRA layers with 87.5% parameter reduction
   - `LoRATrainer`: Basic training loop with gradient computation
   - End-to-end validation and memory efficiency verification

### ✅ Phase 2 Week 2 Complete

#### Data Pipeline & Configuration (78 tests)
1. **Data Loading System (21 tests)**
   - `JSONLoader`, `JSONLLoader`: Multi-format data loading with validation
   - `DatasetLoader`: Auto-detecting format loader
   - `DatasetValidator`: Field validation and summary statistics

2. **Prompt Template System (23 tests)**
   - `AlpacaTemplate`: Instruction-following format with input support
   - `ChatMLTemplate`: Conversation format with system messages
   - `LlamaTemplate`: Chat format with multi-turn conversations
   - `CustomTemplate`: Flexible templates from strings/files
   - `TemplateRegistry`: Centralized template management

3. **Configuration Management (34 tests)**
   - `TrainingConfig`, `ModelConfig`, `DataConfig`, `LoRAConfig`: Complete config classes
   - `ConfigManager`: YAML loading/saving with validation
   - `ConfigProfile`: Predefined profiles (chat, instruction, code)
   - `ConfigValidator`: Compatibility checking and optimization recommendations

### 🚧 Phase 2 Week 3 In Progress

- [ ] Training loop integration with data pipeline
- [ ] CLI command implementation
- [ ] Model inference integration
- [ ] End-to-end training workflow

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
make test-week2       # All Week 2 tests (78 tests)
make test-lora        # LoRA tests (16 tests)
make test             # All tests with coverage
make test-integration # Integration tests

# Quick tests
make test-week2-quick # Quick Week 2 functionality test
make test-lora-quick  # Quick LoRA functionality test

# Component-specific tests
make test-data        # Data loading tests (21 tests)
make test-templates   # Prompt template tests (23 tests)
make test-config      # Configuration tests (34 tests)

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
# Test specific components
PYTHONPATH=src pytest tests/unit/data/test_loaders.py -v        # Data loading
PYTHONPATH=src pytest tests/unit/data/test_templates.py -v      # Templates
PYTHONPATH=src pytest tests/unit/config/test_config.py -v       # Configuration
PYTHONPATH=src pytest tests/unit/test_lora.py -v                # LoRA

# Run specific test class
PYTHONPATH=src pytest tests/unit/data/test_loaders.py::TestJSONLoader -v

# Debug test with output
PYTHONPATH=src pytest tests/unit/config/test_config.py -xvs

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
│   ├── config.py   # Configuration classes (legacy)
│   └── registry.py # Model registry
├── config/         # Configuration management (NEW)
│   ├── config.py   # Core config classes
│   ├── manager.py  # YAML loading/saving
│   ├── profiles.py # Predefined profiles
│   └── validator.py# Config validation
├── data/           # Data pipeline (NEW)
│   ├── loaders.py  # JSON/JSONL loading
│   ├── templates.py# Prompt templates
│   ├── validation.py# Data validation
│   └── exceptions.py# Data exceptions
├── training/       # Training components
│   ├── lora.py     # LoRA implementation
│   └── trainer.py  # Training loops
├── cli/            # CLI (Phase 3)
└── api/            # API server (Future)

tests/
├── unit/          # Unit tests (200 passing)
│   ├── data/      # Data pipeline tests (44 tests)
│   ├── config/    # Configuration tests (34 tests)
│   └── test_lora.py# LoRA tests (16 tests)
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
   make test-week2
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