# FineTune Setup Guide

## Current Status

**✅ Phase 1 Complete**: Core infrastructure implemented
- Model loading system with HuggingFace integration
- MLX and PyTorch backends operational
- 66 unit tests passing (53% coverage)
- Comprehensive linting and formatting tools configured

## Prerequisites

### System Requirements
- **macOS**: 12.0+ (Monterey or later) for MLX support
- **Python**: 3.11 or higher (3.13 tested)
- **Hardware**: Apple Silicon (M1/M2/M3/M4) recommended for MLX
- **Memory**: 16GB+ RAM recommended
- **Storage**: 10GB+ free space for models and dependencies

### Verify System
```bash
# Check macOS version
sw_vers

# Check Python version
python3 --version  # Should be 3.11+

# Check if on Apple Silicon
uname -m  # Should output 'arm64' for Apple Silicon
```

## Installation Methods

### Method 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/finetune.git
cd finetune

# Run the automated setup script
./setup.sh
```

The script automatically:
- Checks Python version compatibility
- Creates virtual environment in `.venv/`
- Installs all dependencies
- Detects and installs MLX on Apple Silicon
- Installs PyTorch with appropriate backend
- Sets up development tools and pre-commit hooks
- Runs tests to verify installation

### Method 2: Manual Setup with venv

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install package in development mode
pip install -e .

# Install test dependencies
pip install pytest pytest-mock pytest-cov loguru

# Install MLX (Apple Silicon only)
if [[ $(uname -m) == 'arm64' ]]; then
    pip install mlx
fi

# Install PyTorch
pip install torch torchvision torchaudio

# Install HuggingFace ecosystem
pip install transformers accelerate datasets tokenizers safetensors huggingface_hub

# Install development tools
pip install black ruff isort mypy pylint pre-commit

# Install pre-commit hooks
pre-commit install
```

### Method 3: Using Make Commands

```bash
# The Makefile assumes .venv exists
python3 -m venv .venv
source .venv/bin/activate

# Install everything
make dev

# Or step by step
make install      # Core package
make test         # Run tests
make lint         # Run linters
```

## Verify Installation

### 1. Run Tests
```bash
# Using Make
make test

# Or directly
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/ -v
```

Expected output:
- **66+ tests passing**
- **53%+ code coverage**
- No import errors

### 2. Check Imports
```bash
# Test core functionality
python -c "from finetune.models.manager import ModelManager; print('✅ Core imports working')"

# Check backends
python -c "import mlx; print('✅ MLX available')" 2>/dev/null || echo "⚠️  MLX not available"
python -c "import torch; print('✅ PyTorch available')"
python -c "import transformers; print('✅ Transformers available')"
```

### 3. Test Model Loading
```python
from finetune.models.manager import ModelManager
from finetune.backends.device import device_manager

# Check selected backend
backend = device_manager.get_optimal_backend()
print(f"Selected backend: {backend.name}")
# Should be 'mlx' on Apple Silicon, 'pytorch' elsewhere

# Initialize manager
manager = ModelManager()
models = manager.list_models()
print(f"Found {len(models)} cached models")
```

## Configuration

### Environment Variables

Create a `.env` file:
```bash
cp .env.example .env
```

Edit `.env`:
```bash
# HuggingFace token for private models
HUGGINGFACE_TOKEN=your_token_here

# Cache directory (default: ~/.cache/finetune)
FINETUNE_HOME=/path/to/cache

# Logging level
FINETUNE_LOG_LEVEL=INFO
```

### Training Configuration

Edit `train.yml` for training parameters:
```yaml
model:
  name: "meta-llama/Llama-2-7b-hf"
  quantization: "4bit"
  
training:
  method: "lora"
  batch_size: 4
  learning_rate: 2e-4
  num_epochs: 3
  
lora:
  r: 16
  alpha: 32
  target_modules: ["q_proj", "v_proj"]
```

## Development Workflow

### Daily Development

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Pull latest changes
git pull

# 3. Update dependencies if needed
pip install -e .

# 4. Run tests before starting
make test

# 5. Make your changes...

# 6. Format code
make format  # or: black src/ tests/

# 7. Run linters
make lint    # or: ruff check src/ tests/

# 8. Run tests
make test

# 9. Commit (pre-commit hooks will run)
git commit -m "Your message"
```

### Available Make Commands

```bash
make help          # Show all available commands
make dev           # Install all dev dependencies
make test          # Run unit tests only
make test-integration  # Run integration tests
make test-all      # Run all tests with coverage
make format        # Format code with black
make lint          # Run ruff and pylint
make clean         # Clean cache and temp files
```

## Project Structure

```
ft/
├── src/finetune/         # Main package
│   ├── models/          # Model implementations
│   │   ├── base.py     # Abstract base classes
│   │   ├── manager.py  # Model manager
│   │   ├── mlx_models.py    # MLX implementations
│   │   ├── mlx_loader.py    # MLX model loader
│   │   └── torch_loader.py  # PyTorch loader
│   ├── backends/        # Backend abstraction
│   │   ├── device.py   # Device management
│   │   ├── mlx_backend.py   # MLX backend
│   │   └── torch_backend.py # PyTorch backend
│   ├── core/           # Core utilities
│   │   ├── config.py   # Configuration
│   │   └── registry.py # Model registry
│   └── cli/            # CLI (in development)
├── tests/              # Test suite
│   ├── unit/          # Unit tests (66 passing)
│   └── integration/   # Integration tests
├── docs/              # Documentation
│   └── design/       # Architecture docs
└── scripts/          # Utility scripts
```

## Troubleshooting

### Common Issues

#### Virtual Environment Not Activated
```bash
# Check if in venv
which python  # Should show .venv/bin/python

# Activate if needed
source .venv/bin/activate
```

#### Import Errors
```bash
# Ensure PYTHONPATH includes src
export PYTHONPATH=src:$PYTHONPATH

# Or reinstall package
pip install -e .
```

#### MLX Not Available
```bash
# Only works on Apple Silicon
uname -m  # Must output 'arm64'

# Install MLX
pip install mlx

# Verify
python -c "import mlx; print('MLX version:', mlx.__version__)"
```

#### Test Failures
```bash
# Clear cache
pytest --cache-clear

# Run with verbose output
pytest -xvs tests/unit/

# Run specific test
pytest tests/unit/test_mlx_models.py -k test_llama_model_creation
```

#### Pre-commit Hook Issues
```bash
# Skip hooks temporarily
git commit --no-verify -m "message"

# Reinstall hooks
pre-commit uninstall
pre-commit install

# Run manually
pre-commit run --all-files
```

#### Memory Issues
- Reduce batch size in configuration
- Use stronger quantization (4-bit instead of 8-bit)
- Close other applications

## IDE Setup

### VS Code

Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreter": "${workspaceFolder}/.venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "editor.formatOnSave": true,
    "editor.rulers": [100]
}
```

### PyCharm
1. File → Settings → Project → Python Interpreter
2. Select `.venv/bin/python` as interpreter
3. Enable pytest as test runner
4. Set black as formatter

## Next Steps

After successful setup:

1. **Explore the codebase**
   ```bash
   # View model implementations
   ls src/finetune/models/
   
   # Run specific tests to understand functionality
   pytest tests/unit/test_mlx_models.py -v
   ```

2. **Try loading a model**
   ```python
   from finetune.models.manager import ModelManager
   
   manager = ModelManager()
   # Small model for testing
   model = manager.load_model("gpt2")
   ```

3. **Read documentation**
   - Architecture: `docs/design/MLX_ARCHITECTURE.md`
   - Integration: `docs/design/MLX_INTEGRATION.md`

4. **Check test coverage**
   ```bash
   pytest --cov=finetune --cov-report=html tests/
   open htmlcov/index.html
   ```

## Getting Help

- **README**: Main documentation with examples
- **Architecture Docs**: `/docs/design/` folder
- **Test Examples**: `/tests/unit/` for usage patterns
- **Issues**: Report bugs or ask questions on GitHub

## Current Limitations

As of Phase 1 completion:
- ✅ Model loading and conversion working
- ✅ Backend abstraction complete
- ⚠️ Training loops not yet implemented (Phase 2)
- ⚠️ CLI commands partial (Phase 2)
- ⚠️ Web UI not yet available (Phase 3)
- ⚠️ REST API not yet available (Phase 3)