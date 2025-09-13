# FineTune

A powerful, modular fine-tuning application optimized for Apple Silicon (M4) that enables efficient training of language models from HuggingFace on custom datasets.

## Features

- 🚀 **Apple Silicon Optimized**: Built-in MLX support for M4 chips with PyTorch MPS fallback
- 🎯 **Multiple Training Methods**: LoRA, QLoRA, and full fine-tuning support
- 📊 **Rich Dataset Support**: JSON, CSV, Parquet, and custom format loaders
- 🎨 **Intuitive Interfaces**: CLI, Web UI, and REST API
- 💾 **Smart Memory Management**: Dynamic batch sizing and gradient checkpointing
- 🔧 **Flexible Configuration**: YAML-based configs with preset profiles
- 📈 **Real-time Monitoring**: TensorBoard integration and live metrics
- 🔄 **Automatic Model Conversion**: Export to GGUF, ONNX, and CoreML formats

## Quick Start

### Installation

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone https://github.com/yourusername/finetune.git
cd finetune

# Install dependencies with Poetry (creates virtual environment automatically)
make dev

# Or for production use
make install
```

### Basic Usage

All commands should be run with Poetry to ensure the virtual environment is used:

```bash
# Initialize a new project
poetry run ft init

# Download a model from HuggingFace
poetry run ft models pull meta-llama/Llama-2-7b-hf

# Prepare your dataset
poetry run ft dataset prepare data/training.jsonl

# Start training (uses train.yml configuration)
poetry run ft train

# Launch the web UI
poetry run ft ui

# Start the API server
poetry run ft serve

# Or enter the Poetry shell first
poetry shell
ft init
ft train
```

## Project Structure

```
ft/
├── src/finetune/     # Main package source code
├── configs/          # Configuration templates
├── tests/            # Test suite
├── scripts/          # Utility scripts
├── examples/         # Example notebooks and datasets
└── docs/             # Documentation
```

## Training Example

Create a configuration file `my_training.yaml`:

```yaml
model:
  name: "meta-llama/Llama-2-7b-hf"
  quantization: "4bit"
  
training:
  method: "lora"
  learning_rate: 2e-4
  batch_size: 4
  num_epochs: 3
  
lora:
  r: 16
  alpha: 32
  target_modules: ["q_proj", "v_proj"]
  
dataset:
  path: "data/my_dataset.jsonl"
  template: "alpaca"
  max_length: 2048
```

Then run:

```bash
ft train --config my_training.yaml
```

## CLI Commands

```bash
ft --help                    # Show all commands
ft models list               # List available models
ft models pull MODEL_NAME    # Download a model
ft dataset validate FILE     # Validate dataset format
ft train                     # Start training
ft evaluate                  # Run evaluation
ft export --format gguf      # Export trained model
ft serve                     # Launch inference server
```

## Web Interface

Launch the web dashboard for a visual interface:

```bash
ft ui
# Open http://localhost:8501
```

Features:
- Model browser and downloader
- Dataset upload and preview
- Visual configuration builder
- Real-time training metrics
- Model evaluation interface

## API Endpoints

Start the REST API server:

```bash
ft serve --port 8000
```

Key endpoints:
- `POST /api/training/start` - Start training job
- `GET /api/training/{job_id}/status` - Check training status
- `POST /api/inference/generate` - Generate text
- `GET /api/models` - List available models

## Development

### Setup Development Environment

```bash
# Poetry manages the virtual environment automatically
# Install development dependencies
make dev

# Run tests
make test

# Format code
make format

# Run linters
make lint

# Enter Poetry shell for development
poetry shell

# Update dependencies
poetry update
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test suite
pytest tests/unit/test_models.py

# Run with coverage
pytest --cov=finetune
```

## System Requirements

- macOS 12.0+ (Monterey or later)
- Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- 16GB+ RAM recommended
- 50GB+ free disk space for models

## Configuration

Configuration files use YAML format and support:
- Model selection and quantization
- Training hyperparameters
- Dataset processing options
- Memory management settings
- Logging and monitoring preferences

See `configs/default.yaml` for all available options.

## Performance Tips

1. **Memory Optimization**: Use QLoRA for models >7B parameters
2. **Batch Size**: Start with 4 and adjust based on memory
3. **Gradient Accumulation**: Increase for larger effective batches
4. **Mixed Precision**: Enable FP16 for faster training
5. **Cache Management**: Set `FINETUNE_HOME` for persistent storage

## Troubleshooting

### Common Issues

**Out of Memory**
- Reduce batch size
- Enable gradient checkpointing
- Use stronger quantization (4-bit)

**Slow Training**
- Ensure MLX is properly installed
- Check unified memory usage
- Reduce sequence length

**Model Loading Errors**
- Verify HuggingFace token is set
- Check internet connection
- Clear model cache

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built with:
- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [Transformers](https://github.com/huggingface/transformers) - HuggingFace's model library
- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning

## Support

- Documentation: [https://finetune.readthedocs.io](https://finetune.readthedocs.io)
- Issues: [GitHub Issues](https://github.com/yourusername/finetune/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/finetune/discussions)