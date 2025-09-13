# Project Structure

**Last Updated**: September 2025  
**Status**: ✅ Core Structure Implemented (Phase 1)

## Directory Layout

```
ft/
├── src/
│   └── finetune/
│       ├── __init__.py
│       ├── __main__.py           # Entry point for `python -m finetune`
│       ├── cli/                  # CLI interface
│       │   ├── __init__.py
│       │   ├── app.py            # Main Typer app
│       │   ├── commands/
│       │   │   ├── __init__.py
│       │   │   ├── train.py      # Training commands
│       │   │   ├── model.py      # Model management
│       │   │   ├── dataset.py    # Dataset operations
│       │   │   ├── evaluate.py   # Evaluation commands
│       │   │   └── serve.py      # Inference server
│       │   └── utils.py          # CLI utilities
│       │
│       ├── core/                 # Core functionality
│       │   ├── __init__.py
│       │   ├── config.py         # Configuration management
│       │   ├── registry.py       # Model/dataset registry
│       │   ├── cache.py          # Caching system
│       │   └── exceptions.py     # Custom exceptions
│       │
│       ├── models/               # Model handling
│       │   ├── __init__.py
│       │   ├── base.py           # Abstract base classes
│       │   ├── loaders.py        # Model loading
│       │   ├── adapters.py       # LoRA/QLoRA adapters
│       │   ├── quantization.py   # Quantization logic
│       │   └── hub.py            # HuggingFace Hub interface
│       │
│       ├── data/                 # Data processing
│       │   ├── __init__.py
│       │   ├── dataset.py        # Dataset base class
│       │   ├── loaders/          # Format-specific loaders
│       │   │   ├── __init__.py
│       │   │   ├── json.py
│       │   │   ├── csv.py
│       │   │   ├── parquet.py
│       │   │   └── text.py
│       │   ├── templates.py      # Prompt templates
│       │   ├── preprocessing.py  # Data preprocessing
│       │   └── validation.py     # Data validation
│       │
│       ├── training/             # Training pipeline
│       │   ├── __init__.py
│       │   ├── trainer.py        # Main trainer class
│       │   ├── mlx_trainer.py    # MLX-specific trainer
│       │   ├── torch_trainer.py  # PyTorch fallback
│       │   ├── callbacks.py      # Training callbacks
│       │   ├── optimizers.py     # Custom optimizers
│       │   └── metrics.py        # Evaluation metrics
│       │
│       ├── inference/            # Inference engine
│       │   ├── __init__.py
│       │   ├── engine.py         # Inference engine
│       │   ├── server.py         # FastAPI server
│       │   └── streaming.py      # Token streaming
│       │
│       ├── api/                  # REST API
│       │   ├── __init__.py
│       │   ├── app.py            # FastAPI app
│       │   ├── routes/
│       │   │   ├── __init__.py
│       │   │   ├── training.py
│       │   │   ├── models.py
│       │   │   ├── datasets.py
│       │   │   └── inference.py
│       │   ├── schemas.py        # Pydantic models
│       │   └── websocket.py      # WebSocket handlers
│       │
│       ├── ui/                   # Web UI
│       │   ├── __init__.py
│       │   ├── app.py            # Streamlit app
│       │   ├── pages/
│       │   │   ├── dashboard.py
│       │   │   ├── training.py
│       │   │   ├── models.py
│       │   │   └── datasets.py
│       │   └── components.py     # Reusable UI components
│       │
│       └── utils/                # Utilities
│           ├── __init__.py
│           ├── logging.py        # Logging configuration
│           ├── memory.py         # Memory management
│           ├── system.py         # System utilities
│           └── progress.py       # Progress tracking
│
├── configs/                      # Configuration files
│   ├── default.yaml              # Default training config
│   ├── profiles/                 # Preset profiles
│   │   ├── chat.yaml
│   │   ├── instruction.yaml
│   │   ├── code.yaml
│   │   └── domain.yaml
│   └── models/                   # Model-specific configs
│       ├── llama.yaml
│       ├── mistral.yaml
│       └── phi.yaml
│
├── tests/                        # Test suite
│   ├── unit/
│   │   ├── test_models.py
│   │   ├── test_data.py
│   │   ├── test_training.py
│   │   └── test_inference.py
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   ├── test_api.py
│   │   └── test_cli.py
│   └── fixtures/
│       ├── models/
│       └── datasets/
│
├── scripts/                      # Utility scripts
│   ├── setup_mlx.py             # MLX setup helper
│   ├── download_models.py       # Batch model downloader
│   ├── benchmark.py             # Performance benchmarking
│   └── convert_model.py         # Model format conversion
│
├── examples/                     # Example usage
│   ├── notebooks/
│   │   ├── quickstart.ipynb
│   │   ├── custom_dataset.ipynb
│   │   └── advanced_lora.ipynb
│   ├── datasets/
│   │   ├── sample_chat.jsonl
│   │   └── sample_instruction.json
│   └── configs/
│       └── example_config.yaml
│
├── docs/                         # Documentation
│   ├── index.md
│   ├── getting_started.md
│   ├── user_guide/
│   │   ├── installation.md
│   │   ├── cli_reference.md
│   │   ├── configuration.md
│   │   └── training.md
│   ├── api_reference/
│   │   └── modules/
│   └── development/
│       ├── contributing.md
│       └── architecture.md
│
├── .github/                      # GitHub configuration
│   ├── workflows/
│   │   ├── test.yml
│   │   ├── lint.yml
│   │   └── release.yml
│   └── ISSUE_TEMPLATE/
│
├── pyproject.toml               # Project metadata & dependencies
├── setup.py                     # Setup script (if needed)
├── requirements.txt             # Pinned dependencies
├── requirements-dev.txt         # Development dependencies
├── Makefile                     # Development commands
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
├── .pre-commit-config.yaml      # Pre-commit hooks
├── README.md                    # Project README
├── LICENSE                      # MIT License
├── CHANGELOG.md                 # Version history
├── CLAUDE.md                    # Claude Code instructions
├── ARCHITECTURE.md              # Architecture documentation
├── TECH_STACK.md               # Technology stack
└── PROJECT_STRUCTURE.md        # This file
```

## Module Responsibilities

### Core Modules

**`finetune.core`**
- Central configuration management
- Global registry for models and datasets
- Caching infrastructure
- Shared exceptions and error handling

**`finetune.models`**
- Model loading and initialization
- Adapter injection (LoRA/QLoRA)
- Quantization operations
- HuggingFace Hub integration

**`finetune.data`**
- Dataset loading and validation
- Format conversion and normalization
- Template application
- Train/validation splitting

**`finetune.training`**
- Training loop orchestration
- Optimizer and scheduler management
- Checkpoint saving/loading
- Metric calculation and logging

**`finetune.inference`**
- Model inference pipeline
- Token generation and streaming
- Batch processing
- Response formatting

### Interface Modules

**`finetune.cli`**
- Command-line interface
- Argument parsing and validation
- Progress display and user interaction
- Configuration file handling

**`finetune.api`**
- RESTful API endpoints
- Request/response schemas
- WebSocket connections
- Authentication and rate limiting

**`finetune.ui`**
- Web dashboard interface
- Real-time training visualization
- Model and dataset management UI
- Interactive configuration builder

### Support Modules

**`finetune.utils`**
- Logging configuration
- Memory monitoring
- System information gathering
- Progress tracking utilities

## Key Files

### Configuration Files

**`pyproject.toml`**
```toml
[build-system]
requires = ["setuptools>=69.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "finetune"
version = "0.1.0"
description = "Generic fine-tuning application for Mac M4"
authors = [{name = "Your Name", email = "your.email@example.com"}]
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

[project.scripts]
ft = "finetune.cli.app:main"

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "ruff>=0.1",
    "black>=23.12",
    "mypy>=1.7",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.black]
line-length = 100
target-version = ["py311", "py312"]

[tool.mypy]
python_version = "3.11"
strict = true
```

**`Makefile`**
```makefile
.PHONY: install dev test lint format clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v --cov=finetune

lint:
	ruff check src/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/

run-api:
	uvicorn finetune.api.app:app --reload --port 8000

run-ui:
	streamlit run src/finetune/ui/app.py

docker-build:
	docker build -t finetune:latest .

docker-run:
	docker run -it --rm -v ~/.cache:/root/.cache finetune:latest
```

## Development Workflow

### Initial Setup
```bash
# Clone repository
git clone <repo-url>
cd ft

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install in development mode
make dev
```

### Development Cycle
1. Create feature branch: `git checkout -b feature/new-feature`
2. Make changes and write tests
3. Run tests: `make test`
4. Format code: `make format`
5. Lint code: `make lint`
6. Commit changes with conventional commits
7. Push and create pull request

### Testing Strategy
- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows
- **Performance tests**: Benchmark training and inference

### Release Process
1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag v0.1.0`
4. Push tag: `git push origin v0.1.0`
5. GitHub Actions builds and publishes

This structure provides clear separation of concerns, making the codebase maintainable and extensible while supporting all planned features.