# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FineTune is a modular fine-tuning application optimized for Apple Silicon (M4) that enables efficient training of language models from HuggingFace on custom datasets.

## Common Development Commands

### Setup and Installation
```bash
make dev          # Install with Poetry, including dev dependencies
make install      # Install production dependencies only
make install-all  # Install with all optional extras
make shell        # Enter Poetry virtual environment shell
```

### Development Workflow
```bash
make test         # Run all tests with coverage
make lint         # Run linting checks (ruff, mypy)
make format       # Format code with black and ruff
make clean        # Clean build artifacts
make update       # Update all dependencies
make lock         # Update poetry.lock file
```

### Running Services
```bash
make run-api      # Start FastAPI server on port 8000
make run-ui       # Start Streamlit UI on port 8501
make run-cli      # Run CLI
make info         # Show system and backend information
```

### Testing
```bash
make test-unit        # Run unit tests only
make test-integration # Run integration tests
poetry run pytest tests/unit/test_models.py  # Run specific test file
```

### Poetry Usage
All commands are automatically run in the Poetry-managed virtual environment. The environment is created in `.venv/` within the project directory. You can also enter the shell directly:
```bash
poetry shell      # Enter virtual environment
exit             # Exit virtual environment
```

## Architecture Overview

### Core Components

1. **MLX-First Training**: Primary backend using Apple's MLX framework for M4 optimization, with PyTorch MPS fallback
2. **Modular Pipeline**: Pluggable loaders for datasets, configurable training methods (LoRA/QLoRA/full)
3. **Multi-Interface**: CLI (Typer), Web UI (Streamlit), REST API (FastAPI)

### Key Design Patterns

- **Registry Pattern**: Central model and dataset registry in `core/registry.py`
- **Strategy Pattern**: Swappable training backends (MLX vs PyTorch) in `training/`
- **Template Method**: Dataset processing pipeline in `data/dataset.py`
- **Observer Pattern**: Training callbacks and metrics in `training/callbacks.py`

### Module Structure

```
src/finetune/
├── cli/        # Command-line interface (Typer-based)
├── core/       # Core abstractions and configuration
├── models/     # Model loading, adapters, quantization
├── data/       # Dataset processing and templates
├── training/   # Training loops and optimization
├── inference/  # Inference engine and serving
├── api/        # REST API endpoints
└── ui/         # Web dashboard
```

### Important Files

- `core/config.py`: Central configuration management using Hydra/OmegaConf
- `models/loaders.py`: HuggingFace model loading with caching
- `training/mlx_trainer.py`: MLX-specific training implementation
- `data/templates.py`: Prompt templates (Alpaca, ChatML, etc.)

## Development Tips

1. **Memory Management**: The project uses unified memory monitoring. Check `utils/memory.py` for utilities.

2. **Testing Models**: Use small models like `gpt2` or `bert-base` for testing to avoid long download times.

3. **Configuration**: Default configs are in `configs/`. Use profile configs for common scenarios.

4. **Logging**: Uses loguru for structured logging. Check logs in `~/.finetune/logs/`.

5. **Database**: SQLite database at `~/.finetune/finetune.db` tracks models, datasets, and training jobs.

## Common Issues and Solutions

1. **MLX Import Errors**: Run `make setup-mlx` to ensure proper MLX installation
2. **Memory Issues**: Reduce batch size or enable gradient checkpointing in config
3. **Model Download Failures**: Set `HUGGINGFACE_TOKEN` environment variable

## Project Status

Currently in active development (v0.1.0-alpha). Core architecture is defined, implementation is ongoing.