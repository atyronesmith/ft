# FineTune Setup Guide

## Prerequisites

- macOS 12.0+ (Monterey or later) with Apple Silicon (M1/M2/M3/M4)
- Python 3.11 or 3.12
- At least 16GB RAM
- 50GB+ free disk space for models

## Quick Start

### 1. Install Poetry (if not already installed)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Add Poetry to your PATH by adding this to your shell profile (`~/.zshrc` or `~/.bash_profile`):
```bash
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Clone the Repository

```bash
git clone https://github.com/yourusername/finetune.git
cd finetune
```

### 3. Install Dependencies

For basic installation:
```bash
make install
```

For development with all tools:
```bash
make dev
```

For full installation with all optional features:
```bash
make install-all
```

### 4. Verify Installation

```bash
make info
```

This will show your system information and available backends.

## Poetry Commands

Poetry automatically manages a virtual environment for the project. The virtual environment is created in `.venv/` within the project directory.

### Basic Commands

```bash
# Enter the virtual environment shell
poetry shell

# Run a command in the virtual environment
poetry run ft --help

# Install dependencies
poetry install

# Add a new dependency
poetry add package-name

# Add a development dependency
poetry add --group dev package-name

# Update all dependencies
poetry update

# Show installed packages
poetry show
```

### Managing Extras

The project has several optional dependency groups:

- `ui` - Streamlit web interface
- `monitoring` - Weights & Biases, advanced plotting
- `config` - Hydra configuration framework
- `db` - Database migrations with Alembic
- `quantization` - Bitsandbytes for model quantization

Install specific extras:
```bash
# Install with UI support
poetry install -E ui

# Install with all extras
poetry install --all-extras
```

## Configuration

### 1. Training Configuration

Edit `train.yml` to configure your training parameters:

```yaml
model:
  name: "meta-llama/Llama-2-7b-hf"
  quantization: "4bit"
  
training:
  method: "lora"
  batch_size: 4
  learning_rate: 2e-4
  num_epochs: 3
```

### 2. Secrets Configuration

Copy the example passwords file and add your credentials:

```bash
cp passwords.yml.example passwords.yml
```

Edit `passwords.yml` with your API keys:
```yaml
huggingface:
  token: "your-token-here"
```

**Important:** `passwords.yml` is gitignored and should never be committed.

## Environment Variables

You can also use environment variables instead of `passwords.yml`:

```bash
export HUGGINGFACE_TOKEN="your-token-here"
export WANDB_API_KEY="your-wandb-key"
```

## Troubleshooting

### Poetry Not Found

If `make` commands fail with "Poetry not found":
1. Ensure Poetry is installed: `curl -sSL https://install.python-poetry.org | python3 -`
2. Add to PATH: `export PATH="$HOME/.local/bin:$PATH"`
3. Restart your terminal

### MLX Installation Issues

MLX is only available on Apple Silicon Macs. If you're on Intel Mac or Linux:
- The system will automatically fall back to PyTorch
- MLX will be skipped during installation

### Memory Issues

If you encounter out-of-memory errors:
1. Reduce batch size in `train.yml`
2. Enable gradient checkpointing
3. Use stronger quantization (4-bit instead of 8-bit)

### Virtual Environment Issues

Poetry creates the virtual environment automatically. If you have issues:

```bash
# Remove existing virtual environment
rm -rf .venv

# Recreate it
poetry install
```

## Development Workflow

### 1. Always Use Virtual Environment

Poetry ensures you're always in the virtual environment when using `poetry run` or `poetry shell`.

### 2. Running Commands

```bash
# Option 1: Use poetry run prefix
poetry run ft train
poetry run pytest

# Option 2: Enter shell first
poetry shell
ft train
pytest
```

### 3. Updating Dependencies

When pulling new changes:
```bash
git pull
poetry install  # This syncs your environment with poetry.lock
```

### 4. Adding New Dependencies

```bash
# Add to main dependencies
poetry add numpy

# Add to dev dependencies
poetry add --group dev pytest-benchmark

# Add optional dependency
poetry add --optional wandb
```

## IDE Setup

### VS Code

1. Install Python extension
2. Select interpreter: `Cmd+Shift+P` → "Python: Select Interpreter" → Choose `.venv/bin/python`

### PyCharm

1. Settings → Project → Python Interpreter
2. Select "Poetry Environment"
3. PyCharm will automatically detect `.venv/`

## Next Steps

1. Initialize a project: `poetry run ft init`
2. Download a model: `poetry run ft models pull gpt2`
3. Prepare your dataset: `poetry run ft dataset prepare data/training.jsonl`
4. Start training: `poetry run ft train`

## Getting Help

- Run `poetry run ft --help` for CLI documentation
- Check `docs/` for detailed guides
- Report issues at: https://github.com/yourusername/finetune/issues