#!/bin/bash
# Setup script for the MLX Fine-tuning Framework

echo "🚀 Setting up MLX Fine-tuning Framework..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher is required (found $python_version)"
    exit 1
fi

echo "✅ Python $python_version found"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -e .

# Install development dependencies
echo "📦 Installing development dependencies..."
pip install pytest pytest-mock pytest-cov loguru

# Install MLX if on Apple Silicon
if [[ $(uname -m) == 'arm64' ]] && [[ $(uname) == 'Darwin' ]]; then
    echo "🍎 Detected Apple Silicon, installing MLX..."
    pip install mlx
fi

# Install PyTorch
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio

# Install Transformers and related packages
echo "🤗 Installing Transformers and related packages..."
pip install transformers accelerate datasets tokenizers safetensors huggingface_hub

# Install linting tools
echo "🧹 Installing linting tools..."
pip install black ruff isort mypy pylint pre-commit

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
.venv/bin/pre-commit install

# Run tests to verify installation
echo "🧪 Running tests to verify installation..."
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/ --tb=no -q

echo ""
echo "✨ Setup complete! Your development environment is ready."
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run tests:"
echo "  make test"
echo ""
echo "To run linting:"
echo "  make lint"
echo ""
echo "Happy coding! 🎉"
