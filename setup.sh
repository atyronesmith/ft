#!/bin/bash
# Setup script for the MLX Fine-tuning Framework (uv-managed).
set -e

echo "🚀 Setting up MLX Fine-tuning Framework..."

if ! command -v uv >/dev/null 2>&1; then
    echo "❌ uv is required: brew install uv  (or: curl -LsSf https://astral.sh/uv/install.sh | sh)"
    exit 1
fi
echo "✅ uv $(uv --version | awk '{print $2}') found"

# uv picks an interpreter satisfying requires-python (>=3.11), creates .venv,
# installs the locked dependency set (dev group included) and the package
# itself in editable mode. mlx is a platform-marked dependency, so it is
# installed on Apple Silicon only, from the same lock.
echo "📦 Syncing environment from uv.lock..."
uv sync --all-groups

echo "🪝 Installing pre-commit hooks..."
uv run pre-commit install

echo "🧪 Running unit tests to verify installation..."
PYTHONPATH=src uv run pytest tests/unit/ --tb=no -q

echo ""
echo "✨ Setup complete."
echo ""
echo "Run tools through uv (no activation needed):   uv run ft --help"
echo "Or activate the environment:                   source .venv/bin/activate"
echo "Tests: make test     Lint: make lint"
