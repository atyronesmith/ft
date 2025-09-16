# Technology Stack & Libraries

**Last Updated**: September 15, 2025
**Status**: ✅ Phase 1 Core Stack Complete, ✅ PHASE 2 COMPLETE - All Training Libraries Implemented

## Core ML Frameworks

### Primary Training Backend ✅ IMPLEMENTED
- **MLX** (0.15+): Apple's ML framework optimized for Apple Silicon
  - ✅ Native M4 acceleration
  - ✅ Unified memory architecture support
  - ✅ Efficient matrix operations
  - ✅ Automatic differentiation
  - ✅ Full model conversion pipeline

### Secondary Training Backend ✅ IMPLEMENTED
- **PyTorch** (2.1+): Fallback for unsupported operations
  - ✅ MPS (Metal Performance Shaders) backend
  - ✅ Seamless fallback mechanism
  - ✅ Extensive model ecosystem
  - ✅ Proven training stability

## Model & NLP Libraries

### Model Management ✅ IMPLEMENTED
- **Transformers** (4.36+): HuggingFace's model library
  - ✅ Model downloading and caching
  - ✅ Tokenizer support
  - ✅ Pre-trained weight loading
  - ✅ Configuration management
  - ✅ Safetensors and sharded model support

### Optimization Techniques ✅ IMPLEMENTED
- **PEFT** (0.7+): Parameter-Efficient Fine-Tuning
  - ✅ LoRA implementation for MLX with 87.5% parameter reduction
  - ✅ Memory-efficient training workflow
  - ✅ Adapter layer integration
  - ✅ Configuration-driven LoRA rank and alpha

- **bitsandbytes** (0.41+): Quantization library
  - 4-bit and 8-bit quantization
  - QLoRA support
  - Memory optimization

### Tokenization
- **tiktoken**: OpenAI's fast tokenizer
- **sentencepiece**: For Llama/Mistral models
- **tokenizers**: HuggingFace's fast tokenizers

## Data Processing ✅ IMPLEMENTED

### Dataset Handling ✅ COMPLETE
- **Custom Data Loaders**: Native JSON/JSONL loading with auto-detection
  - ✅ JSONLoader and JSONLLoader with validation (21 tests)
  - ✅ Memory-efficient streaming for large datasets
  - ✅ Built-in data quality metrics and validation
  - ✅ Field validation with required field checking

### Template System ✅ COMPLETE
- **Prompt Templates**: Multi-format template engine (23 tests)
  - ✅ AlpacaTemplate: Instruction-following format
  - ✅ ChatMLTemplate: Conversation format with system messages
  - ✅ LlamaTemplate: Chat format with multi-turn support
  - ✅ CustomTemplate: Flexible templates from strings/files
  - ✅ TemplateRegistry: Centralized template management

### Data Validation ✅ COMPLETE
- **pydantic** (2.5+): Data validation and settings
  - ✅ DatasetValidator with field validation
  - ✅ Summary statistics and data quality metrics
  - ✅ Configuration validation with memory estimation

## Infrastructure

### Database
- **SQLite3**: Built-in Python support
  - Model registry
  - Training history
  - Job queue management

- **sqlalchemy** (2.0+): ORM for database operations
- **alembic** (1.13+): Database migrations

### Configuration ✅ IMPLEMENTED
- **Custom Configuration System**: YAML-based training configuration (34 tests)
  - ✅ TrainingConfig, ModelConfig, DataConfig, LoRAConfig classes
  - ✅ ConfigManager: YAML loading/saving with validation
  - ✅ ConfigProfile: Predefined profiles (chat, instruction, code)
  - ✅ ConfigValidator: Compatibility checking and memory estimation
- **python-dotenv** (1.0+): Environment variable management

## Web & API

### Backend Framework
- **FastAPI** (0.104+): Modern web framework
  - Automatic API documentation
  - WebSocket support for real-time updates
  - Async support
  - Type hints validation

### API Components
- **uvicorn** (0.24+): ASGI server
- **pydantic** (2.5+): Request/response models
- **python-multipart** (0.0.6+): File uploads
- **websockets** (12.0+): Real-time communication

### Frontend (Dashboard)
- **Streamlit** (1.29+): Rapid dashboard development
  - Real-time metric visualization
  - Interactive configuration
  - File upload interface
  
Alternative:
- **Gradio** (4.8+): ML-focused UI components

## CLI & Terminal

### CLI Framework ✅ IMPLEMENTED
- **Typer** (0.9+): Modern CLI creation
  - ✅ Complete ft train command suite
  - ✅ Type hints support with automatic validation
  - ✅ Automatic help generation and command completion
  - ✅ Rich integration for progress display

### Terminal UI ✅ IMPLEMENTED
- **Rich** (13.7+): Beautiful terminal formatting
  - ✅ Progress bars for training status
  - ✅ Tables and panels for configuration display
  - ✅ Syntax highlighting for code examples
  - ✅ Live displays for CLI progress tracking

- **click** (8.1+): Command parsing (Typer dependency)
- **questionary** (2.0+): Interactive prompts

## Monitoring & Visualization

### Metrics & Logging
- **tensorboard** (2.15+): Training visualization
- **wandb** (0.16+): Optional cloud logging
- **loguru** (0.7+): Structured logging
- **tqdm** (4.66+): Progress bars

### Plotting
- **matplotlib** (3.8+): Basic plotting
- **plotly** (5.18+): Interactive visualizations

## Development Tools

### Testing ✅ IMPLEMENTED
- **pytest** (7.4+): Testing framework
  - ✅ 290+ tests passing (106 Phase 1 + 184 Phase 2)
  - ✅ Comprehensive model conversion testing
  - ✅ Backend fallback validation
  - ✅ Memory management verification
  - ✅ End-to-end integration testing (11 integration tests)
  - ✅ Complete data pipeline validation (78 tests)
  - ✅ LoRA training component testing (16 tests)

### Code Quality ✅ IMPLEMENTED
- **ruff** (0.1+): Fast Python linter
  - ✅ Zero linting issues
  - ✅ Automated formatting
- **black** (23.12+): Code formatter
  - ✅ 100-character line length
  - ✅ Consistent code style
- **mypy** (1.7+): Static type checking
  - ✅ Full type coverage
  - ✅ Strict mode enabled
- **pre-commit** (3.6+): Git hooks
  - ✅ Automated quality checks

### Documentation
- **mkdocs** (1.5+): Documentation site generator
- **mkdocs-material** (9.5+): Material theme
- **pdoc** (14.1+): API documentation

## System Integration

### Process Management
- **psutil** (5.9+): System monitoring
- **py-spy** (0.3+): Performance profiling
- **memory-profiler** (0.61+): Memory usage analysis

### File System
- **watchdog** (3.0+): File system monitoring
- **pathlib**: Built-in path manipulation
- **aiofiles** (23.2+): Async file operations

## Packaging & Distribution

### Package Management
- **setuptools** (69.0+): Package building
- **wheel** (0.42+): Built distribution format
- **twine** (4.0+): PyPI upload tool

### Application Bundling
- **PyInstaller** (6.3+): Standalone executables
- **py2app** (0.28+): macOS app bundles

## Optional/Advanced Libraries

### Model Conversion
- **onnx** (1.15+): Model interchange format
- **coremltools** (7.1+): Core ML conversion
- **llama-cpp-python** (0.2+): GGUF format support

### Performance
- **numba** (0.58+): JIT compilation
- **cupy** (13.0+): GPU arrays (if CUDA available)
- **ray** (2.9+): Distributed computing (future)

### Security
- **cryptography** (41.0+): Encryption for tokens
- **python-jose** (3.3+): JWT tokens
- **passlib** (1.7+): Password hashing

## Installation Strategy

### Base Installation
```bash
pip install mlx transformers peft datasets typer fastapi rich
```

### Full Installation
```bash
pip install -r requirements.txt  # All dependencies
```

### Development Installation
```bash
pip install -e ".[dev]"  # Includes testing and linting tools
```

## Version Management

### Python Version
- **Minimum**: Python 3.11
- **Recommended**: Python 3.12
- **Package Manager**: pip or poetry

### Dependency Resolution
- **poetry** (1.7+): Modern dependency management
- **pip-tools** (7.3+): Alternative for requirements.txt

This comprehensive stack ensures optimal performance on Apple Silicon while maintaining flexibility for various model architectures and training scenarios.