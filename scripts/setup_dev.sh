#!/bin/bash

# Setup script for development environment

echo "Setting up FineTune development environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install basic dependencies
echo "Installing basic dependencies..."
pip install pyyaml pydantic typer rich loguru psutil

# Install MLX if on Apple Silicon
if [[ $(uname -m) == "arm64" ]]; then
    echo "Detected Apple Silicon, installing MLX..."
    pip install mlx
fi

# Install package in development mode
echo "Installing package in development mode..."
pip install -e .

echo "Setup complete! Activate the environment with: source venv/bin/activate"
