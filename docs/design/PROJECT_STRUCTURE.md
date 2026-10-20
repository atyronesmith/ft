# Project Structure

> Canonical Header
- Version: 0.1.0
- Status: See STATUS.md
- Owners: Architecture TL; DevEx Owner
- Last Updated: 2025-09-16
- Linked Commit: 682ba289170b (describe: 682ba28)

## Directory Layout
## Document Scope
- Describes the repository layout and intended responsibilities.
- For live status, test counts, and phase completion, see STATUS.md.

## Out of Scope / Planned
- Modules marked PHASE 3/4 are placeholders; content may change.


```
ft/
├── src/
│   └── finetune/
│       ├── __init__.py            ✅ COMPLETE
│       ├── __main__.py           # Entry point for `python -m finetune` 🚧 PHASE 2
│       ├── cli/                  # CLI interface 🚧 PHASE 2
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
│       ├── core/                 # Core functionality ✅ COMPLETE
│       │   ├── __init__.py        ✅
│       │   ├── config.py         # Configuration management ✅
│       │   ├── registry.py       # Model/dataset registry ✅
│       │   ├── cache.py          # Caching system 🚧 PHASE 2
│       │   └── exceptions.py     # Custom exceptions ✅
│       │
│       ├── models/               # Model handling ✅ COMPLETE
│       │   ├── __init__.py        ✅
│       │   ├── base.py           # Abstract base classes ✅
│       │   ├── manager.py        # Model manager (main interface) ✅
│       │   ├── mlx_models.py     # MLX implementations ✅
│       │   ├── mlx_loader.py     # MLX loader/converter ✅
│       │   └── torch_loader.py   # PyTorch fallback ✅
│       │
│       ├── backends/             # Backend abstraction ✅ COMPLETE
│       │   ├── __init__.py        ✅
│       │   ├── base.py           # Backend interface ✅
│       │   ├── device.py         # Device manager ✅
│       │   ├── mlx_backend.py    # MLX backend ✅
│       │   └── torch_backend.py  # PyTorch backend ✅
│       │
│       ├── config/               # Configuration management ✅ COMPLETE
│       │   ├── __init__.py        ✅
│       │   ├── config.py         # Core config classes ✅
│       │   ├── manager.py        # YAML loading/saving ✅
│       │   ├── profiles.py       # Predefined profiles ✅
│       │   └── validator.py      # Config validation ✅
│       │
│       ├── data/                 # Data pipeline ✅ COMPLETE
│       │   ├── __init__.py        ✅
│       │   ├── loaders.py        # JSON/JSONL loading ✅
│       │   ├── templates.py      # Prompt templates ✅
│       │   ├── validation.py     # Data validation ✅
│       │   └── exceptions.py     # Data exceptions ✅
│       │
│       ├── training/             # Training components ✅ PARTIAL
│       │   ├── __init__.py        ✅
│       │   ├── lora.py           # LoRA implementation ✅
│       │   ├── trainer.py        # Main trainer class ✅
│       │   ├── callbacks.py      # Training callbacks 🚧 PHASE 3
│       │   ├── optimizers.py     # Custom optimizers 🚧 PHASE 3
│       │   └── metrics.py        # Evaluation metrics 🚧 PHASE 3
│       │
│       ├── inference/            # Inference engine 🚧 PHASE 3
│       │   ├── __init__.py
│       │   ├── engine.py         # Inference engine
│       │   ├── server.py         # FastAPI server
│       │   └── streaming.py      # Token streaming
│       │
│       ├── api/                  # REST API 🚧 PHASE 4
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
│       ├── ui/                   # Web UI 🚧 PHASE 4
│       │   ├── __init__.py
│       │   ├── app.py            # Streamlit app
│       │   ├── pages/
│       │   │   ├── dashboard.py
│       │   │   ├── training.py
│       │   │   ├── models.py
│       │   │   └── datasets.py
│       │   └── components.py     # Reusable UI components
│       │
│       └── utils/                # Utilities 🚧 PHASE 2
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
├── tests/                        # Test suite ✅ COMPREHENSIVE (200 tests)
│   ├── conftest.py               # Shared test configuration
│   ├── unit/                     # Unit tests ✅ COMPLETE
│   │   ├── test_lora.py          # LoRA implementation tests (16 tests) ✅
│   │   ├── test_models.py        # Model infrastructure (Phase 1)
│   │   ├── test_backends.py      # Backend selection (Phase 1)
│   │   ├── test_core.py          # Core utilities (Phase 1)
│   │   ├── data/                 # Data pipeline tests ✅ COMPLETE
│   │   │   ├── test_loaders.py   # Data loading tests (21 tests) ✅
│   │   │   └── test_templates.py # Template tests (23 tests) ✅
│   │   └── config/               # Configuration tests ✅ COMPLETE
│   │       └── test_config.py    # Configuration tests (34 tests) ✅
│   ├── integration/              # Integration tests (Phase 1)
│   │   ├── test_mlx_models.py    # MLX model loading
│   │   ├── test_torch_models.py  # PyTorch fallback
│   │   ├── test_conversion.py    # Weight conversion
│   │   └── test_training.py      # Training pipeline
│   └── fixtures/                # Test data and utilities
│       ├── models/              # Small test models
│       ├── datasets/            # Sample datasets
│       └── utils.py             # Test helper functions
│
├── scripts/                      # Utility scripts ✅ COMPLETE
│   ├── setup_mlx.py             # MLX setup helper
│   ├── download_models.py       # Batch model downloader
│   ├── benchmark.py             # Performance benchmarking
│   ├── convert_model.py         # Model format conversion
│   └── completion/              # Shell completion scripts
│       ├── ft-make-completion.bash  # Bash completion
│       └── ft-make-completion.zsh   # Zsh completion
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
├── Makefile                     # Development commands ✅ COMPLETE (29 targets)
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

**`Makefile`** ✅ COMPLETE - Organized Development Commands
```makefile
# Environment & Setup (9 commands)
.PHONY: uv-check install dev install-all update lock create-dirs init

# Testing (6 commands)
.PHONY: test test-unit test-base test-integration test-lora test-lora-quick

# Code Quality (4 commands)
.PHONY: lint format check pre-commit

# Running Applications (4 commands)
.PHONY: run-api run-ui run-cli info

# Docker, Documentation, Utilities, Help
.PHONY: docker-build docker-run docs docs-serve clean setup-mlx benchmark
.PHONY: update-deps completion completion-install help

# Key LoRA Testing Commands:
test-lora:           # Run all 16 LoRA tests (comprehensive validation)
test-lora-quick:     # Run quick LoRA functionality check (2 seconds)

# Organized help with categories:
# 📦 Environment & Setup | 🧪 Testing | 🔍 Code Quality | 🚀 Running Apps
# 🐳 Docker | 📚 Documentation | 🛠️ Utilities | ℹ️ Help

# Shell completion support:
completion:          # Generate bash/zsh completion scripts
completion-install:  # Auto-install completion for current shell
```
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
