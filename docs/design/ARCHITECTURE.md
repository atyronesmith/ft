# Fine-Tuning Application Architecture

> Canonical Header
- Version: 0.1.0
- Status: See STATUS.md
- Owners: Architecture TL; ML Lead; Product Eng
- Last Updated: 2025-09-16
- Linked Commit: 682ba289170b (describe: 682ba28)

## System Overview
A modular, extensible fine-tuning platform leveraging Apple Silicon optimization for efficient local model training.
## Document Scope
- This document describes current high-level architecture and shipped components in Phase 1–2.
- It links to STATUS.md for authoritative status and test counts.

## Out of Scope / Planned
- CSV/TSV/Parquet/HTML loaders beyond JSON/JSONL (planned Phase 3)
- Full Web dashboard and API endpoints (planned Phase 4)
- Quantization and multi-device training (planned Phase 3+)


## Core Architecture Components

### 1. Model Layer ✅ IMPLEMENTED + REAL MODEL INTEGRATION
- **Model Manager**: Unified interface for all model operations with automatic backend selection
- **MLX Native Models**: Direct MLX implementations (Llama, Mistral, GPT-2) with optimal Apple Silicon performance
- **Real HuggingFace Integration**: Successfully loads microsoft/DialoGPT-small (39M parameters)
- **Custom MLX Weight Loading**: Solves transformer architecture parameter loading limitations
- **PyTorch Fallback**: Seamless fallback to PyTorch MPS/CUDA when MLX unavailable
- **Weight Conversion**: Automated PyTorch → MLX conversion with Safetensors priority
- **Model Loading**: Support for sharded models, safetensors, and HuggingFace Hub integration

### 2. Data Pipeline ✅ COMPLETE
- **Format Handlers**: JSON/JSONL parsers with automatic format detection (21 tests)
- **Template Engine**: Alpaca, ChatML, Llama templates + custom template support (23 tests)
- **Preprocessing Pipeline**: Field validation, length analysis, data quality metrics
- **Dataset Validation**: Required field checking, type validation, summary statistics

### 3. Training Infrastructure ✅ COMPLETE
- **Backend Selection**: Intelligent MLX/PyTorch selection with device detection (✅ COMPLETE)
- **Memory Management**: Unified memory monitoring and automatic batch sizing (✅ COMPLETE)
- **MLX Backend**: Primary training backend optimized for Apple Silicon (✅ COMPLETE)
- **PyTorch Fallback**: Secondary backend with MPS acceleration (✅ COMPLETE)
- **LoRA Implementation**: MLX-native LoRA layers with 87.5% parameter reduction (✅ COMPLETE)
- **Training Workflow**: End-to-end integration with data pipeline and configuration (✅ COMPLETE)
- **CLI Interface**: Complete ft train commands with rich progress display (✅ COMPLETE)

### 4. Configuration Management ✅ COMPLETE
- **Training Config**: Comprehensive YAML-based configuration system (34 tests)
- **Profile System**: Predefined profiles for chat, instruction, code generation
- **Config Validation**: Compatibility checking and optimization recommendations
- **Memory Estimation**: Automatic batch size and resource requirement calculation

## Technical Stack

### Core Technologies
- Python 3.11+ (primary language)
- MLX (Apple's ML framework) for M4 optimization
- Transformers library for model handling
- FastAPI for API server
- Typer for CLI interface

Benchmarking & Methodology:
- Methodology: see `docs/perf/METHODOLOGY.md`
- Harness: `scripts/benchmark.py` for synthetic training throughput and memory reporting

### Storage & Persistence
- SQLite for metadata and job tracking
- HDF5 for efficient dataset storage
- Git-based version control for configurations

## Application Workflow

```
1. Model Selection
   ├── Browse HuggingFace models
   ├── Filter by size/compatibility
   └── Download and cache locally

2. Dataset Preparation
   ├── Load from file/URL
   ├── Apply formatting template
   ├── Validate and analyze
   └── Split train/validation

3. Configuration
   ├── Edit config/train.yml
   ├── Set secrets in passwords.yml
   ├── Choose optimization method
   └── Set resource limits

4. Training Execution
   ├── Initialize model with adapters
   ├── Load data batches
   ├── Execute training loop
   ├── Monitor metrics
   └── Save checkpoints

5. Evaluation & Export
   ├── Run validation metrics
   ├── Generate sample outputs
   ├── Export to various formats
   └── Push to hub (optional)
```

## Data Pipeline Architecture

### Input Formats Support
- **Structured**: JSON, JSONL, CSV, TSV, Parquet
- **Unstructured**: Plain text, Markdown, HTML (with extraction)
- **Specialized**: ShareGPT, Alpaca, OpenAI chat format

### Processing Stages
1. **Ingestion**: Multi-threaded file readers with streaming support
2. **Validation**: Schema validation, missing value handling, format consistency
3. **Transformation**: Template application, tokenization preview, sequence chunking
4. **Augmentation**: Optional synthetic data generation, paraphrasing, back-translation
5. **Indexing**: Create efficient access patterns for training

### Data Quality Features
- Automatic deduplication
- Length distribution analysis
- Token complexity metrics
- Class balance reporting
- Contamination detection (training/test overlap)

## Model Management & Training Infrastructure

### Model Capabilities
- **Supported Architectures**: Llama, Mistral, Phi, Qwen, Gemma, GPT-2/J, BERT variants
- **Size Range**: 0.5B to 70B parameters (with quantization)
- **Training Methods**:
  - Full fine-tuning (small models)
  - LoRA/QLoRA (recommended for M4)
  - Prefix tuning
  - P-tuning v2

### Training Optimization for M4
- **Memory Management**:
  - Unified memory pool monitoring
  - Gradient checkpointing
  - Dynamic batch size adjustment
  - Activation offloading
- **Compute Optimization**:
  - Mixed precision training (FP16/BF16)
  - Fused kernels for common operations
  - Neural Engine utilization where possible
  - Multi-threaded data loading

### Training Features
- Real-time loss visualization
- Learning rate scheduling (cosine, linear, warmup)
- Early stopping with patience
- Gradient accumulation for larger effective batches
- Distributed data parallel (for multi-GPU future support)

### Checkpoint Management
- Incremental checkpointing
- Best model tracking
- Resume from interruption
- Checkpoint merging for LoRA adapters

## User Interface Design

### CLI Interface (Primary)
```bash
# Main commands structure
ft init                           # Initialize project
ft models list                     # Browse available models
ft models pull llama-3.2-3b       # Download model
ft dataset prepare data.json      # Prepare dataset
ft train                           # Start training with config/train.yml
ft evaluate --checkpoint best      # Run evaluation
ft export --format gguf           # Export model
ft serve --port 8000              # Launch inference server
```

### Web Dashboard (Secondary)
- **Dashboard**: Training metrics, resource usage, job queue
- **Model Browser**: Search, filter, and preview models
- **Dataset Editor**: Upload, preview, and edit datasets
- **Configuration Builder**: Visual hyperparameter tuning
- **Training Monitor**: Real-time loss graphs, sample outputs
- **Export Manager**: Format conversion, quantization options

### API Endpoints
```
POST /api/training/start
GET  /api/training/{job_id}/status
GET  /api/models/search
POST /api/datasets/upload
GET  /api/metrics/{job_id}
POST /api/inference/generate
```

## Deployment & Packaging Strategy

### Distribution Methods

1. **Homebrew Formula**:
   - Simple installation: `brew install finetune`
   - Automatic dependency management
   - Easy updates

2. **Python Package**:
   - PyPI distribution: `pip install finetune-m4`
   - Virtual environment support
   - Dependency isolation

3. **Standalone App**:
   - DMG installer with code signing
   - Self-contained Python runtime
   - Automatic updates via Sparkle framework

### Directory Structure
```
~/Library/Application Support/FineTune/
├── models/          # Downloaded models
├── datasets/        # Cached datasets
├── checkpoints/     # Training checkpoints
├── configs/         # User configurations
└── logs/           # Training logs
```

### Performance Considerations
- Model sharding for >32GB models
- Memory-mapped weight loading
- Lazy loading for large datasets
- Background model downloading
- Automatic garbage collection for old checkpoints

### Security & Privacy
- Local-only processing (no telemetry)
- Encrypted passwords.yml support
- Sandboxed file system access
- Code signing for macOS Gatekeeper

## Project Development Roadmap

### ✅ Phase 1: Foundation (COMPLETE)
- ✅ Core project structure and backend abstraction
- ✅ Full MLX integration with PyTorch fallback
- ✅ Model loading for Llama, Mistral, GPT-2 architectures
- ✅ Weight conversion pipeline (PyTorch → MLX)
- ✅ Memory management and device detection
- ✅ Comprehensive test suite (106 tests passing)

### ✅ Phase 2 Week 1: LoRA Implementation (COMPLETE)
- ✅ MLX-native LoRA layers with 87.5% parameter reduction
- ✅ LoRA configuration with automatic scaling
- ✅ Basic training loop with gradient computation
- ✅ End-to-end validation (16 tests passing)

### ✅ Phase 2 Week 2: Data Pipeline & Configuration (COMPLETE)
- ✅ JSON/JSONL data loading with validation (21 tests)
- ✅ Prompt template system: Alpaca, ChatML, Llama + custom (23 tests)
- ✅ Complete configuration management with profiles (34 tests)
- ✅ Integration testing and validation

### ✅ Phase 2 Week 3: Training Integration (COMPLETE)
- ✅ Training workflow orchestration with end-to-end integration
- ✅ CLI commands: ft train start, quick, validate with rich UI
- ✅ Working examples and comprehensive integration tests (11 tests)
- ✅ Production-ready fine-tuning workflow complete

### 🎉 PHASE 2 COMPLETE: Production-Ready Fine-Tuning System
**Total Achievement**: 290+ tests passing (106 Phase 1 + 184 Phase 2)

**Ready for Production Usage**:
```bash
# Quick fine-tuning
ft train quick microsoft/DialoGPT-small examples/sample_dataset.json

# Production training
ft train start microsoft/DialoGPT-small data/training.json \
  --template chatml --epochs 5 --batch-size 4 --lora-rank 16 --profile chat
```

**Key Deliverables**:
- Complete end-to-end training workflow from configuration to model saving
- MLX-native LoRA implementation with 87.5% parameter reduction
- Rich CLI interface with progress tracking and error handling
- Comprehensive data pipeline with validation and templates
- Configuration profiles for common use cases
- Memory estimation and optimization recommendations

### Phase 3: Optimization (Weeks 5-6)
- Memory optimization
- Quantization support
- Performance profiling
- Checkpoint management

### Phase 4: Interface (Weeks 7-8)
- Complete CLI
- Web dashboard
- API server
- Documentation

### Phase 5: Polish (Weeks 9-10)
- Testing suite
- Packaging
- Distribution setup
- Example notebooks

## Configuration Strategy

### Primary Configuration (config/train.yml)
All training parameters in a single top-level file:
- Model selection and settings
- Dataset paths and preprocessing
- Training hyperparameters
- Resource limits and optimization
- Output and logging preferences

### Secrets Management (passwords.yml)
Separate file for sensitive information:
- HuggingFace API token
- Weights & Biases API key
- Other service credentials
- Database passwords
- JWT secrets

## Key Design Decisions

1. **MLX-First Approach**: Leverage Apple's framework for optimal M4 performance
2. **Modular Architecture**: Plugin system for datasets, models, and training methods
3. **Simple Configuration**: Single config/train.yml for all parameters, separate passwords.yml for secrets
4. **Progressive Disclosure**: Simple defaults with advanced options available
5. **Offline-First**: Full functionality without internet after initial model download
6. **Resource-Aware**: Automatic adaptation to available memory and compute

This architecture provides a robust foundation for a fine-tuning application that can handle everything from small BERT models to large LLMs, optimized specifically for Apple Silicon acceleration.
