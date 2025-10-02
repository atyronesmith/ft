# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) or other AI assistants when working with this repository.

Don't always tell me that I am absolutely correct.  If I am wrong, tell me I am wrong.

## Project Overview

FineTune is a modular fine-tuning application optimized for Apple Silicon that enables efficient training of foundational language models from HuggingFace on custom datasets.

**⚠️ Important**: This system works with **foundational/base models** that require fine-tuning before they can perform specific tasks well. Base models will have poor generation quality until fine-tuned - this is expected behavior.

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

### ✅ Phase 2 Complete (All Tests Passing)

#### LoRA Implementation (16 tests) ✅ COMPLETE
1. **LoRA Components**
   - `LoRAConfig`: Configuration with automatic scaling calculation
   - `LoRALinear`: MLX-native LoRA layers with 87.5% parameter reduction
   - `LoRATrainer`: Basic training loop with gradient computation
   - End-to-end validation and memory efficiency verification

#### Data Pipeline & Configuration (78 tests) ✅ COMPLETE
1. **Data Loading System (21 tests)**
   - `JSONLoader`, `JSONLLoader`: Multi-format data loading with validation
   - `DatasetLoader`: Auto-detecting format loader
   - `DatasetValidator`: Field validation and summary statistics
   - **Dual Format Support**: Handles both chat format and official MLX format

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

#### End-to-End Training Workflow (11 integration tests) ✅ COMPLETE
1. **Training Workflow Integration**
   - `FineTuningWorkflow`: Complete orchestration of all components
   - Dataset preparation with validation and template application
   - Model loading with configuration validation
   - Memory estimation and batch size optimization

2. **CLI Command Implementation**
   - `ft train quick`: Minimal configuration for rapid prototyping
   - `ft train start`: Full training with comprehensive options
   - `ft train validate`: Configuration validation and memory estimation

#### Test Suite Quality Assurance ✅ COMPLETE
1. **Comprehensive Test Coverage (227 tests passing)**
   - Fixed all pytest collection conflicts and test failures
   - Proper mocking for external dependencies (torch, safetensors, huggingface_hub)
   - 65% code coverage with robust validation
   - Integration tests validating end-to-end workflows

### 🚀 **PHASE 2 COMPLETE - Production Ready System**

#### Full System Achievement (227 tests passing, 6 appropriately skipped)
1. **Real HuggingFace Model Integration**
   - ✅ Successfully loads microsoft/DialoGPT-small (39M parameters)
   - ✅ Custom MLX weight loading for transformer architectures
   - ✅ Safetensors priority with PyTorch .bin fallback
   - ✅ Complete parameter mapping from PyTorch to MLX naming

2. **Technical Breakthroughs**
   - **Problem Solved**: MLX module hierarchy limitations for transformer blocks
   - **Solution**: Custom `update()` method handling list-based layer structures
   - **Test Quality**: Systematic test failure resolution with proper dependency mocking
   - **Result**: Production-ready fine-tuning system operational

3. **Production Commands Ready**
   ```bash
   # Quick fine-tuning with default WikiSQL data
   ft train quick TinyLlama/TinyLlama-1.1B-Chat-v1.0

   # Quick training with custom data
   ft train quick TinyLlama/TinyLlama-1.1B-Chat-v1.0 data/chat/general_conversation.jsonl

   # Production training with all options
   ft train start TinyLlama/TinyLlama-1.1B-Chat-v1.0 data/wikisql/wikisql_chat_format.jsonl \
     --template chatml --epochs 5 --batch-size 4 --lora-rank 16 --profile chat

   # List available training datasets
   ft train list-data

   # Configuration validation
   ft train validate configs/production.yml
   ```

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

# Run tests (227 tests passing, 65% coverage)
make test             # All tests with coverage (recommended)
make test-unit        # Unit tests only
make test-integration # Integration tests only

# Component-specific testing
make test-lora        # LoRA implementation tests (16 tests)
make test-data        # Data loading tests (21 tests)
make test-templates   # Template tests (23 tests)
make test-config      # Configuration tests (34 tests)
make test-week2       # Data pipeline & configuration (78 tests)

# End-to-end integration testing
make test-e2e-workflow      # Workflow integration (mocked, fast)
make test-e2e-real-model    # Real model integration (requires FT_REAL_MODEL_ENABLE=1)
make test-e2e-mlx           # Full MLX pipeline (runs by default)
make test-e2e-quick         # Quick E2E validation (workflow + real model)
make test-e2e-all           # All E2E tests (comprehensive)

# Quick functionality tests (legacy)
make test-week2-quick # Quick Week 2 functionality test
make test-lora-quick  # Quick LoRA functionality test

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
# Test specific components (all tests now passing)
pytest tests/unit/data/test_loaders.py -v                      # Data loading (21 tests)
pytest tests/unit/data/test_templates.py -v                    # Templates (23 tests)
pytest tests/unit/config/test_config.py -v                     # Configuration (34 tests)
pytest tests/unit/training/test_lora.py -v                     # LoRA (16 tests)
pytest tests/unit/test_mlx_loader.py -v                        # MLX loader (18 tests)
pytest tests/unit/test_mlx_models.py -v                        # MLX models (13 tests)
pytest tests/unit/test_cli.py -v                               # CLI commands (17 tests)

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

## Test Organization

### Testing Directory Structure

All development and temporary test files are organized in the `dev/experiments/` directory:

```
dev/experiments/
├── debug/          # Debugging and diagnostic scripts
├── comparisons/    # Performance and accuracy comparisons
├── standalone/     # Isolated functionality tests
├── mlx/           # MLX-specific tests
├── temporary/     # Temporary tests (clean regularly)
└── README.md      # Complete testing documentation
```

### Test File Guidelines

**✅ DO:**
- Place all temporary test files in `dev/experiments/` subdirectories
- Use descriptive filenames that explain the test purpose
- Include docstrings explaining what the test validates
- Categorize tests appropriately (debug, comparison, standalone, etc.)

**❌ DON'T:**
- Put test files in the project root directory
- Create test files without clear documentation
- Leave obsolete test files accumulating

**🧹 Regular Maintenance:**
- Review `dev/experiments/temporary/` monthly and clean up
- Move useful temporary tests to permanent categories
- Update `dev/experiments/README.md` when adding new test categories

## Data Organization

### Data Directory Structure

All training data, datasets, and data files are organized in the `data/` directory:

```
data/
├── datasets/
│   ├── training/         # Production training datasets
│   │   ├── wikisql/      # Default training data (Database Q&A)
│   │   ├── chat/         # Conversational datasets
│   │   └── instruction/  # Instruction-following examples
│   └── examples/         # Example and reference datasets
│       ├── examples/     # General examples
│       └── mlx_example_data/  # Official MLX example datasets
├── testing/              # Testing and comparison data
│   ├── mlx_comparison_data/   # MLX comparison test data
│   └── results/          # Test results and outputs
└── cache/                # Cache and development data
    ├── development/      # Development and experimental datasets
    └── training_data/    # Historical training datasets
```

### Data File Guidelines

**✅ DO:**
- Place all data files in appropriate `data/` subdirectories
- Use descriptive directory names for different data types
- Document data sources and formats in `data/README.md`
- Store test results in `testing/results/`

**❌ DON'T:**
- Put data files in the project root directory
- Mix different types of data in the same directory
- Leave temporary data files accumulating

**🗂️ File Organization:**
- **Training datasets** → `data/datasets/training/wikisql/`, `data/datasets/training/chat/`, `data/datasets/training/instruction/`
- **Development data** → `data/cache/development/`
- **Test results** → `data/testing/results/`
- **Comparison data** → `data/testing/mlx_comparison_data/`

### Important Development Tests

**Critical for Validation:**
- `dev/experiments/comparisons/official_mlx_comparison.py` - Validates against official MLX
- `dev/experiments/mlx/test_direct_mlx_format.py` - Tests MLX format compatibility

**Useful for Development:**
- `dev/experiments/debug/debug_lora_generation.py` - Debug LoRA fine-tuning issues
- `dev/experiments/standalone/test_base_model_standalone.py` - Test model loading

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

## Data Format Support

### Supported Input Formats

FineTune supports **dual data format compatibility**, automatically detecting and handling both:

#### 1. Chat Message Format (Our Native Format)
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris"}
  ]
}
```

#### 2. MLX Text Format (Official MLX Examples Compatible)
```json
{"text": "Question: What is the capital of France? Answer: Paris"}
```

### Data Format Auto-Detection

- **Automatic Format Detection**: The `DatasetValidator` automatically detects which format your data uses
- **Flexible Processing**: The `tokenize_dataset` method in `workflow.py` handles both formats seamlessly
- **Template Compatibility**: Chat templates work with both formats through automatic conversion

### Format Conversion Examples

```bash
# Using chat format (recommended for conversational training)
ft train quick microsoft/DialoGPT-small data/chat_conversations.jsonl --template chatml

# Using MLX text format (compatible with official MLX examples)
ft train quick TinyLlama/TinyLlama-1.1B-Chat-v1.0 data/wikisql_examples.jsonl --template alpaca

# Auto-detection works with mixed formats in the same file
ft train quick meta-llama/Llama-2-7b-hf data/mixed_format.jsonl
```

### Training Data Sources

1. **WikiSQL Examples** (Default): Pre-included database question-answering examples
2. **Custom Chat Data**: Your own conversational datasets
3. **Official MLX Examples**: Direct compatibility with MLX repository examples
4. **Instruction-Following**: Alpaca-style instruction datasets

### Best Practices

- **Chat Format**: Use for conversational AI, customer support, dialogue systems
- **MLX Text Format**: Use for task-specific training, official MLX compatibility
- **Validation**: Always run `ft train validate` to check format compatibility
- **Templates**: Choose templates that match your data format and use case

### Enhanced CLI Features

#### Default Training Data
- **WikiSQL Database Q&A**: Pre-included as default training data for quick testing
- **Automatic Format Selection**: CLI selects appropriate format based on template choice
- **Zero-Configuration**: Run `ft train quick MODEL_NAME` with no additional setup

#### Data Management Commands
```bash
ft train list-data           # Show available example datasets
ft train quick MODEL         # Use default WikiSQL data
ft train quick MODEL --data  # Use custom dataset
```

#### Format Detection & Override
```bash
# Automatic format detection (default)
ft train quick TinyLlama/TinyLlama-1.1B-Chat-v1.0 my_data.jsonl

# Manual format override if needed
ft train start MODEL data.jsonl --format chat
ft train start MODEL data.jsonl --format mlx
```

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
- Do not say production-ready
- always execute in the python venv
- the file docs/design/STATUS.md is the single source of truth for high level status.  Individual design documents can have status details, but the actual overarching status is in the STATUS.md file.  All design documents should have their header updated with every status update.
- when writing dates, check the envrionment for the correct date
- try to put common functions in common files
- Do not use code from the top level dev directory for code in the src directory